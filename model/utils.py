from abc import ABC, abstractmethod
from flax import linen as nn
from jax.typing import ArrayLike


class WrappedModule(ABC, nn.Module):
    """
    This class takes another nn.Module and prepares it such that it:
    1. Takes a time input t and scales it by T.
    2. Passes the scaled t to the module.
    3. Post-processes it such that it describes a q-function.
    """
    other: nn.Module
    T: float

    def __call__(self, t: ArrayLike):
        t = t / self.T

        h = self.other(t)
        return self._post_process(t, h)

    @abstractmethod
    def _post_process(self, t: ArrayLike, h: ArrayLike):
        raise NotImplementedError
