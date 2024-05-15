import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_path_energy(paths, U, reduce=jnp.max, add=0, already_ln=False, **kwargs):
    reduced = jnp.array([reduce(U(path)) for path in paths]) + add

    if already_ln:
        # Convert reduced to log10
        reduced = reduced / jnp.log(10)
        plt.plot(jnp.arange(0, len(reduced), 1), reduced, **kwargs)
    else:
        plt.semilogy(jnp.arange(0, len(reduced), 1), reduced, **kwargs)
