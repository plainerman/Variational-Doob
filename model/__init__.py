from abc import ABC, abstractmethod
from flax import linen as nn
from jax.typing import ArrayLike


class WrappedModule(ABC, nn.Module):
    other: nn.Module
    T: float

    def __call__(self, t: ArrayLike):
        t = t / self.T

        h = self.other(t)
        return self._post_process(t, h)

    @abstractmethod
    def _post_process(self, t: ArrayLike, h: ArrayLike):
        raise NotImplementedError


class MLP(nn.Module):
    hidden_dims: ArrayLike

    @nn.compact
    def __call__(self, t):
        h = t - 0.5
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.swish(h)

        return h
