import numpy as np
import jax.numpy as jnp
import jax
from eval.path_metrics import plot_path_energy
from tps_baseline_mueller import U, dUdx_fn, minima_points
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

def load(path):
    return jnp.array(np.load(path, allow_pickle=True).astype(np.float32)).squeeze()


@jax.jit
def log_prob_path(path):
    rand = path[1:] - path[:-1] + dt * dUdx_fn(path[:-1])
    return U(path[0]) + jax.scipy.stats.norm.logpdf(rand, scale=jnp.sqrt(dt) * xi).sum()


if __name__ == '__main__':
    savedir = './out/evaluation/mueller/'
    os.makedirs(savedir, exist_ok=True)

    all_paths = [
        ('one-way-shooting', './out/baselines/mueller/paths-one-way-shooting.npy'),
        ('two-way-shooting', './out/baselines/mueller/paths-two-way-shooting.npy'),
        ('var-doobs', './out/var_doobs/mueller/paths.npy'),
    ]

    num_paths = 1000
    xi = 5
    dt = 1e-4
    T = 275e-4
    N = int(T / dt)

    global_minimum_energy = U(minima_points[0])
    for point in minima_points:
        global_minimum_energy = min(global_minimum_energy, minimize(U, point).fun)
    print("Global minimum energy", global_minimum_energy)

    all_paths = [(name, load(path)) for name, path in all_paths]
    [print(name, path.shape) for name, path in all_paths]

    for name, paths in all_paths:
        plot_path_energy(paths, U, add=-global_minimum_energy, label=name)

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
        plot_path_energy(paths, log_prob_path, reduce=lambda x: x, label=name)
        print('Median energy of:', name, jnp.median(jnp.array([log_prob_path(path) for path in paths])))

    plt.legend()
    plt.ylabel('log path likelihood')
    plt.savefig(f'{savedir}/mueller-log-path-likelihood.pdf', bbox_inches='tight')
    plt.show()
