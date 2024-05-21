import jax.numpy as jnp
from tqdm import tqdm


def decorrelated(paths):
    prev = paths[0]
    decorrelated = [prev]

    for x in tqdm(paths[1:]):
        # check if the two arrays share a common value
        if not jnp.in1d(prev, x).any():
            decorrelated.append(x)
            prev = x

    return decorrelated
