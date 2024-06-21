from abc import ABC, abstractmethod
from dataclasses import dataclass

from flax import linen as nn
import jax.numpy as jnp
from typing import Union, Dict, Any, Callable, Tuple
from flax.training.train_state import TrainState
import jax
from flax.typing import FrozenVariableDict
from jax.typing import ArrayLike
from model.utils import WrappedModule
from training.qsetup import QSetup
from systems import System
from training.utils import forward_and_derivatives


class DiagonalWrapper(WrappedModule):
    A: ArrayLike
    B: ArrayLike
    num_mixtures: int
    trainable_weights: bool
    base_sigma: float

    @nn.compact
    def _post_process(self, t: ArrayLike, h: ArrayLike):
        ndim = self.A.shape[0]
        num_mixtures = self.num_mixtures
        h = nn.Dense(2 * ndim * num_mixtures)(h)

        mu = (((1 - t) * self.A)[:, None, :] + (t * self.B)[:, None, :] +
              (((1 - t) * t * h[:, :ndim * num_mixtures])[:, None, :]).reshape(-1, num_mixtures, ndim))

        sigma = (
                ((1 - t) * self.base_sigma)[:, None, :] + (t * self.base_sigma)[:, None, :] +
                ((1 - t) * t)[:, None, :] * jnp.exp(h[:, ndim * num_mixtures:]).reshape(-1, num_mixtures, ndim)
        )

        if self.trainable_weights:
            w_logits = self.param('w_logits', nn.initializers.zeros_init(), (num_mixtures,))
        else:
            w_logits = jnp.zeros(num_mixtures)

        return mu, sigma, w_logits


@dataclass
class DiagonalSetup(QSetup, ABC):
    model_q: DiagonalWrapper
    T: float
    base_sigma: float
    num_mixtures: int

    @abstractmethod
    def _drift(self, _x: ArrayLike, gamma: float) -> ArrayLike:
        raise NotImplementedError

    def construct_loss(self, state_q: TrainState, xi: ArrayLike, gamma: float, BS: int) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], ArrayLike], ArrayLike]:

        def loss_fn(params_q: Union[FrozenVariableDict, Dict[str, Any]], key: ArrayLike) -> ArrayLike:
            ndim = self.model_q.A.shape[-1]
            key = jax.random.split(key)

            t = self.T * jax.random.uniform(key[0], [BS, 1])
            eps = jax.random.normal(key[1], [BS, 1, ndim])

            def v_t(_eps, _t):
                """This function is equal to v_t * xi ** 2."""
                _mu_t, _sigma_t, _w_logits, _dmudt, _dsigmadt = forward_and_derivatives(state_q, _t, params_q)
                _i = jax.random.categorical(key[2], _w_logits, shape=[BS, ])

                _x = _mu_t[jnp.arange(BS), _i, None] + _sigma_t[jnp.arange(BS), _i, None] * eps

                if self.num_mixtures == 1:
                    # This completely ignores the weights and saves some time
                    relative_mixture_weights = 1
                else:
                    log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
                    relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]

                log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)
                u_t = (relative_mixture_weights * (1 / _sigma_t * _dsigmadt * (_x - _mu_t) + _dmudt)).sum(axis=1)

                return u_t - self._drift(_x.reshape(BS, ndim), gamma) + 0.5 * (xi ** 2) * log_q_t

            loss = 0.5 * ((v_t(eps, t) / xi) ** 2).sum(-1, keepdims=True)
            return loss.mean()

        return loss_fn

    def u_t(self, state_q: TrainState, t: ArrayLike, x_t: ArrayLike, xi: ArrayLike, *args, **kwargs) -> ArrayLike:
        _mu_t, _sigma_t, _w_logits, _dmudt, _dsigmadt = forward_and_derivatives(state_q, t)
        _x = x_t[:, None, :]

        log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
        relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]

        _u_t = (relative_mixture_weights * (1 / _sigma_t * _dsigmadt * (_x - _mu_t) + _dmudt)).sum(axis=1)

        if xi == 0:
            return _u_t

        log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)

        return _u_t + 0.5 * (xi ** 2) * log_q_t


class FirstOrderSetup(DiagonalSetup):
    def __init__(self, system: System, model: nn.module, T: float, base_sigma: float, num_mixtures: int,
                 trainable_weights: bool):
        model_q = DiagonalWrapper(model, T, system.A, system.B, num_mixtures, trainable_weights, base_sigma)
        super().__init__(system, model_q, T, base_sigma, num_mixtures)

    def _drift(self, _x: ArrayLike, gamma: float) -> ArrayLike:
        return -self.system.dUdx(_x / (gamma * self.system.mass))


class SecondOrderSetup(DiagonalSetup):
    def __init__(self, system: System, model: nn.module, T: float, base_sigma: float, num_mixtures: int,
                 trainable_weights: bool):
        # We pad the A and B matrices with zeros to account for the velocity
        self._A = jnp.hstack([system.A, jnp.zeros_like(system.A)])
        self._B = jnp.hstack([system.B, jnp.zeros_like(system.B)])

        model_q = DiagonalWrapper(model, T, self._A, self._B, num_mixtures, trainable_weights, base_sigma)
        super().__init__(system, model_q, T, base_sigma, num_mixtures)

    def _drift(self, _x: ArrayLike, gamma: float) -> ArrayLike:
        # number of dimensions without velocity
        ndim = self.system.A.shape[0]

        return jnp.hstack([_x[:, ndim:] / self.system.mass, -self.system.dUdx(_x[:, :ndim]) - _x[:, ndim:] * gamma])

    def _xi_to_second_order(self, xi: ArrayLike) -> ArrayLike:
        if xi.shape == self.model_q.A.shape:
            return xi
        
        xi_velocity = jnp.ones_like(self.system.A) * xi
        xi_pos = jnp.zeros_like(xi_velocity) + 1e-4

        return jnp.concatenate((xi_pos, xi_velocity), axis=-1)

    def construct_loss(self, state_q: TrainState, xi: ArrayLike, gamma: float, BS: int) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], ArrayLike], ArrayLike]:
        return super().construct_loss(state_q, self._xi_to_second_order(xi), gamma, BS)

    def u_t(self, state_q: TrainState, t: ArrayLike, x_t: ArrayLike, xi: ArrayLike, *args, **kwargs) -> ArrayLike:
        return super().u_t(state_q, t, x_t, self._xi_to_second_order(xi), *args, **kwargs)

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B
