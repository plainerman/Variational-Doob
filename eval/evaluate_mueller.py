import numpy as np
import jax.numpy as jnp
import jax
from eval.path_metrics import plot_path_energy
from systems import System
from tps.paths import decorrelated
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

num_paths = 1000
xi = 5
kbT = xi ** 2 / 2
dt = 1e-4
T = 275e-4
N = int(T / dt)

system = System.from_name('mueller_brown', float('inf'))

minima_points = jnp.array([[-0.55828035, 1.44169],
                           [-0.05004308, 0.46666032],
                           [0.62361133, 0.02804632]])


def load(path):
    loaded = np.load(path, allow_pickle=True)
    return [p.astype(np.float32).reshape(-1, 2) for p in loaded]


@jax.jit
def log_path_likelihood(path):
    rand = path[1:] - path[:-1] + dt * system.dUdx(path[:-1])
    return (-system.U(path[0]) / kbT).sum() + jax.scipy.stats.norm.logpdf(rand, scale=jnp.sqrt(dt) * xi).sum()


def plot_hist(system, paths, trajectories_to_plot, seed=1):
    system.plot(trajectories=paths)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    idx = jax.random.permutation(jax.random.PRNGKey(seed), len(paths))[:trajectories_to_plot]
    for i, c in zip(idx, colors[1:]):
        cur_paths = jnp.array(paths[i])
        plt.plot(cur_paths[:, 0].T, cur_paths[:, 1].T, c=c)


if __name__ == '__main__':
    savedir = './out/evaluation/mueller/'
    os.makedirs(savedir, exist_ok=True)

    all_paths = [
        ('one-way-shooting', './out/baselines/mueller/paths-one-way-shooting.npy', 50),
        ('variable-one-way-shooting', './out/baselines/mueller-variable/paths-one-way-shooting.npy', 50),
        ('two-way-shooting', './out/baselines/mueller/paths-two-way-shooting.npy', 0),
        ('variable-two-way-shooting', './out/baselines/mueller-variable/paths-two-way-shooting.npy', 0),
        ('var-doobs', './out/var_doobs/mueller/paths.npy', 0),
    ]

    global_minimum_energy = min(system.U(minima_points))
    for point in minima_points:
        global_minimum_energy = min(global_minimum_energy, minimize(system.U, point).fun)
    print("Global minimum energy", global_minimum_energy)

    all_paths = [(name, path, warmup) for name, path, warmup in all_paths if os.path.exists(path)]
    print('Running script for the following paths:')
    [print(name, path) for name, path, warmup in all_paths]
    assert len(all_paths) > 0, 'No paths found, please consider running tps_baseline_mueller.py first.'

    all_paths = [(name, load(path)[warmup:],) for name, path, warmup in all_paths]
    [print(name, len(path)) for name, path in all_paths]

    for name, paths in all_paths:
        # for this plot we limit ourselves to 250 paths
        plot_hist(system, paths[:250], 2)
        plt.savefig(f'{savedir}/{name}-histogram.pdf', bbox_inches='tight')
        plt.show()

        plot_hist(system, decorrelated(paths)[:250], 2)
        plt.savefig(f'{savedir}/{name}-decorrelated-histogram.pdf', bbox_inches='tight')
        plt.show()

    for name, paths in all_paths:
        print(name, 'decorrelated trajectories:', jnp.round(100 * len(decorrelated(paths)) / len(paths), 2), '%')

    for name, paths in all_paths:
        max_energy = plot_path_energy(paths, system.U, add=-global_minimum_energy, label=name) + global_minimum_energy
        print(name, 'max energy mean:', jnp.round(jnp.mean(max_energy), 2), 'std:', jnp.round(jnp.std(max_energy), 2))
        print(name, 'min max energy: ', jnp.round(jnp.min(max_energy), 2))

    plt.legend()
    plt.ylabel('Maximum energy')
    plt.savefig(f'{savedir}/mueller-max-energy.pdf', bbox_inches='tight')
    plt.show()

    for name, paths in all_paths:
        plot_path_energy(paths, system.U, add=-global_minimum_energy, reduce=jnp.median, label=name)

    plt.legend()
    plt.ylabel('Median energy')
    plt.savefig(f'{savedir}/mueller-median-energy.pdf', bbox_inches='tight')
    plt.show()

    for name, paths in all_paths:
        likelihood = plot_path_energy(paths, log_path_likelihood, reduce=lambda x: x, label=name)
        print(name, 'mean log-likelihood:', jnp.round(jnp.mean(likelihood), 2), 'std:',
              jnp.round(jnp.std(likelihood), 2))
        print(name, 'maximum log-likelihood:', jnp.round(jnp.max(likelihood), 2))

    plt.legend()
    plt.ylabel('log path likelihood')
    plt.savefig(f'{savedir}/mueller-log-path-likelihood.pdf', bbox_inches='tight')
    plt.show()
