from argparse import ArgumentParser
from utils.args import parse_args
from systems import System
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from flax.training import train_state
import optax
import training.qsetup as qsetup
from training.train import train
from utils.plot import show_or_save_fig, log_scale
import os
import sys
import orbax.checkpoint as ocp

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
parser.add_argument('--temperature', type=float,
                    help="The temperature of the system in Kelvin. Either specify this or xi.")
parser.add_argument('--gamma', type=float, required=True)

parser.add_argument('--ode', type=str, choices=['first_order', 'second_order'], required=True)
parser.add_argument('--parameterization', type=str, choices=['diagonal', 'low_rank'], required=True)

# parameters of Q
parser.add_argument('--num_gaussians', type=int, default=1, help="Number of gaussians in the mixture model.")
parser.add_argument('--trainable_weights', type=bool, default=False,
                    help="Whether the weights of the mixture model are trainable.")
parser.add_argument('--base_sigma', type=float, required=True, help="Sigma at time t=0 for A and B.")

# training
parser.add_argument('--epochs', type=int, default=10_000, help="Number of epochs the system is training for.")
parser.add_argument('--BS', type=int, default=512, help="Batch size used for training.")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--force_clip', type=float, default=float('inf'), help="Clipping value for the force")

parser.add_argument('--load', type=bool, default=False, const=True, nargs='?',
                    help="Continue training and load the model from the save_dir.")
parser.add_argument('--save_interval', type=int, default=1_000, help="Interval at which the model is saved.")

parser.add_argument('--seed', type=int, default=1, help="The seed that will be used for initialization")

# inference
parser.add_argument('--num_paths', type=int, default=1000, help="The number of paths that will be generated.")
parser.add_argument('--dt', type=float, required=True)

# plotting
parser.add_argument('--log_plots', type=bool, default=False, const=True, nargs='?',
                    help="Save plots in log scale where possible")


def main():
    print("!!!!Next todos: plot ALDP")
    # TODO: internal coordinates
    # TODO: neural network parameterization

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

    if args.xi:
        xi = args.xi
    else:
        kbT = 1.380649 * 6.02214076 * 1e-3 * args.temperature
        xi = jnp.sqrt(2 * kbT * args.gamma / system.mass)

    # TODO: parameterize neural network?
    # TODO: if we find a nice way, maybe this can also include base_sigma
    # You can play around with any model here
    # The chosen setup will append a final layer so that the output is mu, sigma, and weights
    from model import MLP

    model = MLP([128, 128, 128])
    setup = qsetup.construct(system, model, args.ode, args.parameterization, xi, args)

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
            # The model needs to be casted to a trainstate object
            state_restored['model'] = checkpoint_manager.restore(checkpoint_manager.latest_step(), items=ckpt)['model']
            ckpt = state_restored

    key, train_key = jax.random.split(key)
    ckpt = train(ckpt, loss_fn, args.epochs, train_key, checkpoint_manager)
    state_q = ckpt['model']
    print("Total number of potential evaluations", args.BS * args.epochs)

    if jnp.isnan(jnp.array(ckpt['losses'])).any():
        print("Warning: Loss contains NaNs")
    plt.plot(ckpt['losses'])
    log_scale(args.log_plots, False, True)
    show_or_save_fig(args.save_dir, 'loss_plot.pdf')

    print("!!!TODO: how to plot this nicely?")
    t = args.T * jnp.linspace(0, 1, args.BS, dtype=jnp.float32).reshape((-1, 1))
    key, path_key = jax.random.split(key)
    eps = jax.random.normal(path_key, [args.BS, args.num_gaussians, setup.A.shape[-1]])
    mu_t, sigma_t, w_logits = state_q.apply_fn(state_q.params, t)
    w = jax.nn.softmax(w_logits)[None, :, None]
    samples = (w * (mu_t + sigma_t * eps)).sum(axis=1)

    # plot_energy_surface()
    # plt.scatter(samples[:, 0], samples[:, 1])
    # plt.scatter(A[0, 0], A[0, 1], color='red')
    # plt.scatter(B[0, 0], B[0, 1], color='orange')
    # plt.show()

    key, init_key = jax.random.split(key)
    x_0 = jnp.ones((args.num_paths, setup.A.shape[0]), dtype=jnp.float32) * setup.A
    eps = jax.random.normal(key, shape=x_0.shape)
    x_0 += args.base_sigma * eps

    x_t_det = setup.sample_paths(state_q, x_0, args.dt, args.T, args.BS, None)

    if system.plot:
        # In case we have a second order integration scheme, we remove the velocity for plotting
        system.plot(title='Deterministic Paths', trajectories=x_t_det[:, :, :system.A.shape[0]])
        show_or_save_fig(args.save_dir, 'paths_deterministic.pdf')

    key, path_key = jax.random.split(key)
    x_t_stoch = setup.sample_paths(state_q, x_0, args.dt, args.T, args.BS, path_key)

    if system.plot:
        system.plot(title='Stochastic Paths', trajectories=x_t_stoch[:, :, :system.A.shape[0]])
        show_or_save_fig(args.save_dir, 'paths_stochastic.pdf')


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        parser.print_usage(file=sys.stderr)
        print(e, file=sys.stderr)
