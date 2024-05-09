import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_path_energy(paths, U, reduce=jnp.max, already_ln=False):
    reduced = jnp.array([reduce(U(path)) for path in tqdm(paths, 'Computing path metric')])

    if already_ln:
        # Convert reduced to log10
        reduced = reduced / jnp.log(10)
        plt.plot(jnp.arange(0, len(reduced), 1), reduced)
    else:
        plt.semilogy(jnp.arange(0, len(reduced), 1), reduced)
