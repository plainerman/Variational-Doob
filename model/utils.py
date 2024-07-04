from abc import ABC, abstractmethod
from typing import Tuple, Any
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

        h, args = self._pre_process(t)
        h = self.other(h)
        return self._post_process(h, *args)

    def _pre_process(self, t: ArrayLike) -> Tuple[ArrayLike, Tuple[Any, ...]]:
        """This function returns a tuple. The first element will be used as an input to the other module,
        and the second value will be passed to the post process function."""
        return t, (t,)

    @abstractmethod
    def _post_process(self, h: ArrayLike, *args):
        raise NotImplementedError
