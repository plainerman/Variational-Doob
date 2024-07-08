from abc import ABC
from flax import linen as nn
import jax.numpy as jnp
from systems import System
from training.qsetup import QSetup
from jax.typing import ArrayLike


class DriftedSetup(QSetup, ABC):
    """A QSetup that has a drift term. This drift term can be either first or second order."""

    def __init__(self, system: System, model_q: nn.Module, xi: ArrayLike, order: str):
        """Either instantiate with first or second order drift."""
        assert order == 'first' or order == 'second', "Order must be either 'first' or 'second'."
        self.order = order

        super().__init__(system, model_q, xi)

    def _drift(self, _x: ArrayLike, gamma: float) -> ArrayLike:
        if self.order == 'first':
            return -self.system.dUdx(_x / (gamma * self.system.mass))
        else:
            # number of dimensions without velocity
            ndim = self.system.A.shape[0]

            return jnp.hstack([_x[:, ndim:] / self.system.mass, -self.system.dUdx(_x[:, :ndim]) - _x[:, ndim:] * gamma])
