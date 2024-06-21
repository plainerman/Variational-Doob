from flax import linen as nn
from jax.typing import ArrayLike


class MLP(nn.Module):
    hidden_dims: ArrayLike

    @nn.compact
    def __call__(self, t):
        h = t - 0.5
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.swish(h)

        return h
