from argparse import ArgumentParser

import numpy as np

from utils.animation import save_trajectory
from utils.args import parse_args, str2bool
from systems import System
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from flax.training import train_state
import optax
import training.qsetup as qsetup
from training.train import train
from utils.plot import show_or_save_fig, log_scale, plot_energy, plot_u_t
import os
import sys
import orbax.checkpoint as ocp
from model import MLP

parser = ArgumentParser()
parser.add_argument('--save_dir', type=str, default=None, help="Specify a path where the data will be stored.")
parser.add_argument('--config', type=str, help='Path to the config yaml file')

# system configuration
parser.add_argument('--test_system', type=str,
                    choices=['double_well', 'double_well_hard', 'double_well_dual_channel', 'mueller_brown'])
parser.add_argument('--start', type=str, help="Path to pdb file with the start structure A")
parser.add_argument('--target', type=str, help="Path to pdb file with the target structure B")
parser.add_argument('--forcefield', type=str, nargs='+', default=['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
                    help="Forcefield for the system")
parser.add_argument('--cv', type=str, choices=['phi_psi'],
                    help="Collective variable used for the system. Needed to plot the energy surface of non-test systems.")

parser.add_argument('--T', type=float, required=True,
                    help="Transition time in the base unit of the system. For molecular simulations, this is in picoseconds.")
parser.add_argument('--xi', type=float)
parser.add_argument('--xi_pos_noise', type=float, default=1e-4,
                    help="For second order SDEs we have to add a small noise to the positional xi. This is the value of this noise.")
parser.add_argument('--temperature', type=float,
                    help="The temperature of the system in Kelvin. Either specify this or xi.")
parser.add_argument('--gamma', type=float, required=True)

parser.add_argument('--ode', type=str, choices=['first_order', 'second_order'], required=True)
parser.add_argument('--parameterization', type=str, choices=['diagonal', 'full_rank'], required=True)

# parameters of Q
parser.add_argument('--num_gaussians', type=int, default=1, help="Number of gaussians in the mixture model.")
parser.add_argument('--trainable_weights', type=str2bool, nargs='?', const=True, default=False,
                    help="Whether the weights of the mixture model are trainable.")

# model parameters
parser.add_argument('--model', type=str, choices=['mlp', 'spline'], default='mlp',
                    help="The model that will be used. Note that spline might not work with all configurations.")

# Spline arguments
parser.add_argument('--num_points', type=int, default=100, help="Number of points in the spline model.")
parser.add_argument('--spline_mode', type=str, choices=['linear', 'cubic'], default='linear')

# MLP arguments
parser.add_argument('--hidden_layers', nargs='+', type=int, help='The dimensions of the hidden layer of the MLP.',
                    default=[128, 128, 128])
parser.add_argument('--activation', type=str, default='swish', choices=['tanh', 'relu', 'swish'],
                    help="Activation function used after every layer.")
parser.add_argument('--resnet', type=str2bool, nargs='?', const=True, default=False,
                    help="Whether to use skip connections in the model.")
parser.add_argument('--internal_coordinates', type=str2bool, nargs='?', const=True, default=False,
                    help="Whether to use internal coordinates for the system. This only works for alanine.")
parser.add_argument('--base_sigma', type=float, required=True, help="Sigma at time t=0 for A and B.")

# training
parser.add_argument('--epochs', type=int, default=10_000, help="Number of epochs the system is training for.")
parser.add_argument('--BS', type=int, default=512, help="Batch size used for training.")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--force_clip', type=float, default=float('inf'), help="Clipping value for the force")

parser.add_argument('--load', type=str2bool, nargs='?', const=True, default=False,
                    help="Continue training and load the model from the save_dir.")
parser.add_argument('--save_interval', type=int, default=1_000, help="Interval at which the model is saved.")

parser.add_argument('--seed', type=int, default=1, help="The seed that will be used for initialization")

# inference
parser.add_argument('--num_paths', type=int, default=1000, help="The number of paths that will be generated.")
parser.add_argument('--dt', type=float, required=True)

# plotting
parser.add_argument('--no_plots', type=str2bool, nargs='?', const=True, default=False, help="Disable all plots.")
parser.add_argument('--log_plots', type=str2bool, nargs='?', const=True, default=False,
                    help="Save plots in log scale where possible")
parser.add_argument('--extension', type=str, default='pdf', help="Extension of the saved plots.")


def main():
    args = parse_args(parser)
    assert args.test_system or args.start and args.target, "Either specify a test system or provide start and target structures"
    assert not (
            args.test_system and args.start and args.target), "Specify either a test system or provide start and target structures, not both"

    assert args.xi or args.temperature, "Either specify xi or temperature"
    assert not (args.xi and args.temperature), "Specify either xi or temperature, not both"

    print(f'Config: {args}')
    os.makedirs(args.save_dir, exist_ok=True)

    if args.test_system:
        system = System.from_name(args.test_system, args.force_clip)
    else:
        system = System.from_pdb(args.start, args.target, args.forcefield, args.cv, args.force_clip)

    if args.no_plots:
        system.plot = None

    if args.xi:
        xi = args.xi
    else:
        kbT = 1.380649 * 6.02214076 * 1e-3 * args.temperature
        xi = jnp.sqrt(2 * kbT * args.gamma / system.mass)

    # We initialize A and B
    if args.ode == 'first_order':
        A = system.A
        B = system.B
    elif args.ode == 'second_order':
        # We pad the A and B matrices with zeros to account for the velocity
        A = jnp.hstack([system.A, jnp.zeros_like(system.A)], dtype=jnp.float32)
        B = jnp.hstack([system.B, jnp.zeros_like(system.B)], dtype=jnp.float32)

        xi_velocity = jnp.ones_like(system.A) * xi
        xi_pos = jnp.zeros_like(xi_velocity) + args.xi_pos_noise

        xi = jnp.concatenate((xi_pos, xi_velocity), axis=-1, dtype=jnp.float32)
    else:
        raise ValueError(f"Unknown ODE: {args.ode}")

    # You can play around with any model here
    # The chosen setup will append a final layer so that the output is mu, sigma, and weights
    model = None
    if args.model == 'mlp':
        model = MLP(args.hidden_layers, args.activation, args.resnet)

    setup = qsetup.construct(system, model, xi, A, B, args)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    params_q = setup.model_q.init(init_key, jnp.zeros([args.BS, 1], dtype=jnp.float32))

    optimizer_q = optax.adam(learning_rate=args.lr)
    state_q = train_state.TrainState.create(apply_fn=setup.model_q.apply, params=params_q, tx=optimizer_q)
    loss_fn = setup.construct_loss(state_q, args.gamma, args.BS)

    ckpt = {'model': state_q, 'losses': []}
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=args.save_interval,
        max_to_keep=3,
        create=True,
        cleanup_tmp_directories=True,
        save_on_steps=[args.epochs]
    )
    checkpoint_manager = ocp.CheckpointManager(os.path.abspath(args.save_dir), orbax_checkpointer, options)

    if args.load:
        if checkpoint_manager.latest_step() is None:
            print("Warning: No checkpoint found.")
        else:
            print('Loading checkpoint:', checkpoint_manager.latest_step())

            state_restored = checkpoint_manager.restore(checkpoint_manager.latest_step())
            # The model needs to be cast to a trainstate object
            state_restored['model'] = checkpoint_manager.restore(checkpoint_manager.latest_step(), items=ckpt)['model']
            ckpt = state_restored

    key, train_key = jax.random.split(key)
    ckpt = train(ckpt, loss_fn, args.epochs, train_key, checkpoint_manager)
    state_q = ckpt['model']
    print("Total number of potential evaluations", args.BS * args.epochs)

    if jnp.isnan(jnp.array(ckpt['losses'])).any():
        print("Warning: Loss contains NaNs")
    plt.title('Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.plot(ckpt['losses'])
    log_scale(args.log_plots, False, True)
    show_or_save_fig(args.save_dir, 'loss_plot', args.extension)

    t = args.T * jnp.linspace(0, 1, args.BS, dtype=jnp.float32).reshape((-1, 1))
    key, path_key = jax.random.split(key)
    mu_t, _, w_logits = state_q.apply_fn(state_q.params, t)
    w = jax.nn.softmax(w_logits)
    print('Weights of mixtures:', w)
    if system.plot:
        mu_t_no_vel = mu_t[:, :, :system.A.shape[0]]
        num_trajectories = jnp.array((w * 100).round(), dtype=int)

        trajectories = jnp.swapaxes(mu_t_no_vel, 0, 1)
        trajectories = (jnp.vstack([trajectories[i].repeat(n, axis=0) for i, n in enumerate(num_trajectories) if n > 0])
                        .reshape(num_trajectories.sum(), -1, mu_t_no_vel.shape[2]))

        system.plot(title='Weighted mean paths', trajectories=trajectories)
        show_or_save_fig(args.save_dir, 'mean_paths', args.extension)

    if system.plot and system.A.shape[0] == 2:
        print('Animating gif, this might take a few seconds ...')
        plot_u_t(system, setup, state_q, args.T, args.save_dir, 'u_t', frames=100)

    key, init_key = jax.random.split(key)
    x_0 = jnp.ones((args.num_paths, A.shape[0]), dtype=jnp.float32) * A
    eps = jax.random.normal(key, shape=x_0.shape, dtype=jnp.float32)
    x_0 += args.base_sigma * eps

    x_t_det = setup.sample_paths(state_q, x_0, args.dt, args.T, args.BS, None)
    # In case we have a second order integration scheme, we remove the velocity for plotting
    x_t_det_no_vel = x_t_det[:, :, :system.A.shape[0]]

    key, path_key = jax.random.split(key)
    x_t_stoch = setup.sample_paths(state_q, x_0, args.dt, args.T, args.BS, path_key)
    x_t_stoch_no_vel = x_t_stoch[:, :, :system.A.shape[0]]
    np.save(f'{args.save_dir}/stochastic_paths.npy', x_t_stoch_no_vel)

    if system.mdtraj_topology:
        save_trajectory(system.mdtraj_topology, x_t_det_no_vel[0].reshape(1, -1, 3), f'{args.save_dir}/det_0.pdb')
        save_trajectory(system.mdtraj_topology, x_t_det_no_vel[-1].reshape(1, -1, 3), f'{args.save_dir}/det_-1.pdb')

        save_trajectory(system.mdtraj_topology, x_t_stoch_no_vel[0].reshape(1, -1, 3), f'{args.save_dir}/stoch_0.pdb')
        save_trajectory(system.mdtraj_topology, x_t_stoch_no_vel[-1].reshape(1, -1, 3), f'{args.save_dir}/stoch_-1.pdb')

    if system.plot:
        plot_energy(system, [x_t_det_no_vel[0], x_t_det_no_vel[-1]], args.log_plots)
        show_or_save_fig(args.save_dir, 'path_energy_deterministic', args.extension)

        system.plot(title='Deterministic Paths', trajectories=x_t_det_no_vel)
        show_or_save_fig(args.save_dir, 'paths_deterministic', args.extension)

        plot_energy(system, [x_t_stoch_no_vel[0], x_t_stoch_no_vel[-1]], args.log_plots)
        show_or_save_fig(args.save_dir, 'path_energy_stochastic', args.extension)

        system.plot(title='Stochastic Paths', trajectories=x_t_stoch_no_vel)
        show_or_save_fig(args.save_dir, 'paths_stochastic', args.extension)

        system.plot(title='Stochastic Paths', trajectories=x_t_stoch_no_vel)
        trajectories_to_plot = 2
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        idx = jax.random.permutation(jax.random.PRNGKey(args.seed), x_t_stoch_no_vel.shape[0])[:trajectories_to_plot]
        for i, c in zip(idx, colors[1:]):
            plt.plot(x_t_stoch_no_vel[i, :, 0].T, x_t_stoch_no_vel[i, :, 1].T, c=c)
        show_or_save_fig(args.save_dir, 'paths_stochastic_and_individual', args.extension)


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        parser.print_usage(file=sys.stderr)
        print(e, file=sys.stderr)
