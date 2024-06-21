from argparse import ArgumentParser
from utils.args import parse_args
from systems import System
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from flax.training import train_state
import optax
import model.qsetup as qsetup
from model.train import train
from utils.plot import show_or_save_fig
import os

parser = ArgumentParser()
parser.add_argument('--save_dir', type=str, default=None, help="Specify a path where the data will be stored.")
parser.add_argument('--config', type=str, help='Path to the config yaml file')

# system configuration
parser.add_argument('--test_system', type=str,
                    choices=['double_well', 'double_well_hard', 'double_well_dual_channel', 'mueller_brown'])
parser.add_argument('--start', type=str, help="Path to pdb file with the start structure A")
parser.add_argument('--target', type=str, help="Path to pdb file with the target structure B")
parser.add_argument('--ode', type=str, choices=['first-order', 'second-order'], required=True)
parser.add_argument('--parameterization', type=str, choices=['diagonal', 'low-rank'], required=True)

parser.add_argument('--T', type=float, required=True,
                    help="Transition time in the base unit of the system. For molecular simulations, this is in picoseconds.")
parser.add_argument('--xi', type=float, required=True)
parser.add_argument('--gamma', type=float, required=True)

# parameters of Q
parser.add_argument('--num_gaussians', type=int, default=1, help="Number of gaussians in the mixture model.")
parser.add_argument('--trainable_weights', type=bool, default=False,
                    help="Whether the weights of the mixture model are trainable.")
parser.add_argument('--base_sigma', type=float, required=True, help="Sigma at time t=0 for A and B.")

# training
parser.add_argument('--epochs', type=int, default=10_000, help="Number of epochs the system is training for.")
parser.add_argument('--BS', type=int, default=512, help="Batch size used for training.")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

parser.add_argument('--seed', type=int, default=1, help="The seed that will be used for initialization")

# inference
parser.add_argument('--num_paths', type=int, default=1000, help="The number of paths that will be generated.")
parser.add_argument('--dt', type=float, required=True)

if __name__ == '__main__':
    args = parse_args(parser)
    assert args.test_system or args.start and args.target, "Either specify a test system or provide start and target structures"
    assert not (
            args.test_system and args.start and args.target), "Specify either a test system or provide start and target structures, not both"

    print(f'Config: {args}')
    os.makedirs(args.save_dir, exist_ok=True)

    if args.test_system:
        system = System.from_name(args.test_system)
    else:
        raise NotImplementedError
        # system = System.from_forcefield(args.start, args.target)

    # TODO: parameterize neural network?
    # TODO: if we find a nice way, maybe this can also include base_sigma
    # You can play around with any model here
    # The chosen setup will append a final layer so that the output is mu, sigma, and weights
    from model import MLP

    model = MLP([128, 128, 128])
    setup = qsetup.construct(system, model, args.ode, args.parameterization, args)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params_q = setup.model_q.init(init_key, jnp.zeros([args.BS, 1]))

    optimizer_q = optax.adam(learning_rate=args.lr)
    state_q = train_state.TrainState.create(apply_fn=setup.model_q.apply, params=params_q, tx=optimizer_q)
    loss_fn = setup.construct_loss(state_q, args.xi, args.gamma, args.BS)

    key, train_key = jax.random.split(key)
    state_q, loss_plot = train(state_q, loss_fn, args.epochs, train_key)
    print("Number of potential evaluations", args.BS * args.epochs)

    plt.plot(loss_plot)
    show_or_save_fig(args.save_dir, 'loss_plot.pdf')


    # TODO: how to plot this nicely?
    t = args.T * jnp.linspace(0, 1, args.BS).reshape((-1, 1))
    key, path_key = jax.random.split(key)
    eps = jax.random.normal(path_key, [args.BS, args.num_gaussians, system.A.shape[-1]])
    mu_t, sigma_t, w_logits = state_q.apply_fn(state_q.params, t)
    w = jax.nn.softmax(w_logits)[None, :, None]
    samples = (w * (mu_t + sigma_t * eps)).sum(axis=1)

    # plot_energy_surface()
    # plt.scatter(samples[:, 0], samples[:, 1])
    # plt.scatter(A[0, 0], A[0, 1], color='red')
    # plt.scatter(B[0, 0], B[0, 1], color='orange')
    # plt.show()

    key, init_key = jax.random.split(key)
    x_0 = jnp.ones((args.num_paths, system.A.shape[0])) * system.A
    eps = jax.random.normal(key, shape=x_0.shape)
    x_0 += setup.base_sigma * eps

    x_t_det = setup.sample_paths(state_q, x_0, args.dt, args.T, args.BS, None, None)

    if system.plot:
        system.plot(title='Deterministic Paths', trajectories=x_t_det)
        show_or_save_fig(args.save_dir, 'paths_deterministic.pdf')

    key, path_key = jax.random.split(key)
    x_t_stoch = setup.sample_paths(state_q, x_0, args.dt, args.T, args.BS, args.xi, path_key)

    if system.plot:
        system.plot(title='Stochastic Paths', trajectories=x_t_stoch)
        show_or_save_fig(args.save_dir, 'paths_stochastic.pdf')
