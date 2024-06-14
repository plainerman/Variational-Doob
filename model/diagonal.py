from flax import linen as nn
import jax.numpy as jnp
from typing import Union, Dict, Any, Callable
from flax.training.train_state import TrainState
import jax
from flax.typing import FrozenVariableDict
from jax.typing import ArrayLike
from model import WrappedModule
from model.setup import TrainSetup
from systems import System


class DiagonalWrapper(WrappedModule):
    A: ArrayLike
    B: ArrayLike
    num_mixtures: int
    trainable_weights: bool
    base_sigma: float

    @nn.compact
    def _post_process(self, t: ArrayLike, h: ArrayLike):
        print('WARNING: Gaussian Mixture Model not implemented yet')
        ndim = self.A.shape[0]
        h = nn.Dense(2 * ndim)(h)

        mu = (1 - t) * self.A[None, :] + t * self.B[None, :] + (1 - t) * t * h[:, :ndim]
        sigma = (1 - t) * self.base_sigma + t * self.base_sigma + (1 - t) * t * jnp.exp(h[:, ndim:])
        w_logits = self.param('w_logits', nn.initializers.zeros_init(),
                              (self.num_mixtures,)) if self.trainable_weights else jnp.zeros(self.num_mixtures)

        return mu, sigma, w_logits


class FirstOrderSetup(TrainSetup):

    def __init__(self, system: System, model: nn.module, T: float, num_mixtures: int, trainable_weights: bool,
                 base_sigma: float):
        model_q = DiagonalWrapper(model, T, system.A, system.B, num_mixtures, trainable_weights, base_sigma)
        super().__init__(system, model_q)
        self.system = system
        self.T = T

    def construct_loss(self, state_q: TrainState, xi: float, BS: float) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], ArrayLike], float]:
        print('WARNING: Gaussian Mixture Loss not implemented yet')

        def loss_fn(params_q: Union[FrozenVariableDict, Dict[str, Any]], key: ArrayLike) -> float:
            key = jax.random.split(key)
            t = self.T * jax.random.uniform(key[0], [BS, 1])
            eps = jax.random.normal(key[1], [BS, 2])

            mu_t = lambda _t: state_q.apply_fn(params_q, _t)[0]
            sigma_t = lambda _t: state_q.apply_fn(params_q, _t)[1]

            def dmudt(_t):
                _dmudt = jax.jacrev(lambda _t: mu_t(_t).sum(0))
                return _dmudt(_t).squeeze().T

            def dsigmadt(_t):
                _dsigmadt = jax.jacrev(lambda _t: sigma_t(_t).sum(0))
                return _dsigmadt(_t).squeeze().T

            def v_t(_eps, _t):
                u_t = dmudt(_t) + dsigmadt(_t) * _eps
                _x = mu_t(_t) + sigma_t(_t) * _eps
                out = (u_t + self.system.dUdx(_x)) - 0.5 * (xi ** 2) * _eps / sigma_t(t)
                return out

            loss = 0.5 * ((v_t(eps, t) / xi) ** 2).sum(1, keepdims=True)
            return loss.mean()

        return loss_fn
