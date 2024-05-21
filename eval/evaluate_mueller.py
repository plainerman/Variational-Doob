import numpy as np
import jax.numpy as jnp
import jax
from eval.path_metrics import plot_path_energy
from tps.paths import decorrelated
from tps_baseline_mueller import U, dUdx_fn
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

num_paths = 1000
xi = 5
kbT = xi ** 2 / 2
dt = 1e-4
T = 275e-4
N = int(T / dt)

minima_points = jnp.array([[-0.55828035, 1.44169],
                           [-0.05004308, 0.46666032],
                           [0.62361133, 0.02804632]])


def load(path):
    loaded = np.load(path, allow_pickle=True)
    return [p.astype(np.float32).reshape(-1, 2) for p in loaded]


@jax.jit
def log_path_likelihood(path):
    rand = path[1:] - path[:-1] + dt * dUdx_fn(path[:-1])
    return (-U(path[0]) / kbT).sum() + jax.scipy.stats.norm.logpdf(rand, scale=jnp.sqrt(dt) * xi).sum()


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

    global_minimum_energy = U(minima_points[0])
    for point in minima_points:
        global_minimum_energy = min(global_minimum_energy, minimize(U, point).fun)
    print("Global minimum energy", global_minimum_energy)

    all_paths = [(name, load(path)[warmup:],) for name, path, warmup in all_paths]
    [print(name, len(path)) for name, path in all_paths]

    for name, paths in all_paths:
        print(name, 'decorrelated trajectories:', jnp.round(100 * len(decorrelated(paths)) / len(paths), 2), '%')

    for name, paths in all_paths:
        max_energy = plot_path_energy(paths, U, add=-global_minimum_energy, label=name) + global_minimum_energy
        print(name, 'max energy mean:', jnp.round(jnp.mean(max_energy), 2), 'std:', jnp.round(jnp.std(max_energy), 2))
        print(name, 'min max energy: ', jnp.round(jnp.min(max_energy), 2))

    plt.legend()
    plt.ylabel('Maximum energy')
    plt.savefig(f'{savedir}/mueller-max-energy.pdf', bbox_inches='tight')
    plt.show()

    for name, paths in all_paths:
        plot_path_energy(paths, U, add=-global_minimum_energy, reduce=jnp.median, label=name)

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
