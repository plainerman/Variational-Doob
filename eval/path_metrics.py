import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_path_energy(paths, U, reduce=jnp.max, add=0, already_ln=False, **kwargs):
    reduced = jnp.array([reduce(U(path)) for path in tqdm(paths)]) + add

    if already_ln:
        # Convert reduced to log10
        reduced = reduced / jnp.log(10)
        plt.plot(jnp.arange(0, len(reduced), 1), reduced, **kwargs)
    else:
        plt.semilogy(jnp.arange(0, len(reduced), 1), reduced, **kwargs)

    return reduced


def plot_iterative_min_max_energy(paths, U, potential_calls):
    reduced = jnp.array([jnp.max(U(path)) for path in tqdm(paths)])

    iterative_min = [reduced[0]]
    for c in reduced[1:]:
        iterative_min.append(min(c, iterative_min[-1]))

    plt.xlabel('Number of potential calls')
    plt.ylabel('Minimum energy of best path so far')
    plt.semilogx(jnp.cumsum(jnp.array(potential_calls)), iterative_min)

    return iterative_min
