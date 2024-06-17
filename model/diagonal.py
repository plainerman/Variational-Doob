from flax import linen as nn
import jax.numpy as jnp
from typing import Union, Dict, Any, Callable
from flax.training.train_state import TrainState
import jax
from flax.typing import FrozenVariableDict
from jax.typing import ArrayLike
from model import WrappedModule
from model.qsetup import QSetup
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
        h = nn.Dense(2 * ndim * self.num_mixtures)(h)

        mu = (((1 - t) * self.A)[:, None, :] + (t * self.B)[:, None, :] +
              (((1 - t) * t * h[:, :ndim * self.num_mixtures])[:, None, :]).reshape(-1, self.num_mixtures, ndim))

        sigma = (((1 - t) * self.base_sigma)[:, None, :] + (t * self.base_sigma)[:, None, :] +
                 ((1 - t) * t)[:, None, :] * jnp.exp(h[:, ndim * self.num_mixtures:]).reshape(-1, self.num_mixtures,
                                                                                              ndim))

        w_logits = self.param('w_logits', nn.initializers.zeros_init(),
                              (self.num_mixtures,)) if self.trainable_weights else jnp.zeros(self.num_mixtures)

        return mu, sigma, w_logits


class FirstOrderSetup(QSetup):
    def __init__(self, system: System, model: nn.module, T: float, num_mixtures: int, trainable_weights: bool,
                 base_sigma: float):
        model_q = DiagonalWrapper(model, T, system.A, system.B, num_mixtures, trainable_weights, base_sigma)
        super().__init__(system, model_q, base_sigma)
        self.system = system
        self.num_mixtures = num_mixtures
        self.T = T

    @staticmethod
    def dmudt(state_q: TrainState, t: ArrayLike,
              params_q: Union[FrozenVariableDict, Dict[str, Any]] = None) -> ArrayLike:
        params = state_q.params if params_q is None else params_q
        _dmudt = jax.jacrev(lambda _t: state_q.apply_fn(params, _t)[0].sum(0).T)
        return _dmudt(t).squeeze(axis=-1).T

    @staticmethod
    def dsigmadt(state_q: TrainState, t: ArrayLike,
                 params_q: Union[FrozenVariableDict, Dict[str, Any]] = None) -> ArrayLike:
        params = state_q.params if params_q is None else params_q
        _dsigmadt = jax.jacrev(lambda _t: state_q.apply_fn(params, _t)[1].sum(0).T)
        return _dsigmadt(t).squeeze(axis=-1).T

    def construct_loss(self, state_q: TrainState, xi: float, BS: int) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], ArrayLike], ArrayLike]:

        def loss_fn(params_q: Union[FrozenVariableDict, Dict[str, Any]], key: ArrayLike) -> ArrayLike:
            gamma = 1.0
            print('WARNING: gamma is set to 1 for now')
            ndim = self.system.A.shape[-1]
            key = jax.random.split(key)

            t = self.T * jax.random.uniform(key[0], [BS, 1])
            eps = jax.random.normal(key[1], [BS, 1, ndim])

            def v_t(_eps, _t):
                """This function is equal to v_t * xi ** 2."""
                _mu_t, _sigma_t, _w_logits = state_q.apply_fn(params_q, _t)
                _i = jax.random.categorical(key[2], _w_logits, shape=[BS, ])

                _dmudt = FirstOrderSetup.dmudt(state_q, _t, params_q)
                _dsigmadt = FirstOrderSetup.dsigmadt(state_q, _t, params_q)

                _x = _mu_t[jnp.arange(BS), _i, None] + _sigma_t[jnp.arange(BS), _i, None] * eps

                if self.num_mixtures == 1:
                    # This completely ignores the weights and saves some time
                    relative_mixture_weights = 1
                else:
                    log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
                    relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]

                log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)

                u_t = (relative_mixture_weights * (1 / _sigma_t * _dsigmadt * (_x - _mu_t) + _dmudt)).sum(axis=1)
                b_t = -self.system.dUdx(_x.reshape(BS, ndim)) / (gamma * self.system.mass)

                return u_t - b_t + 0.5 * (xi ** 2) * log_q_t

            loss = 0.5 * ((v_t(eps, t) / xi) ** 2).sum(-1, keepdims=True)
            return loss.mean()

        return loss_fn

    def u_t(self, state_q: TrainState, t: ArrayLike, x_t: ArrayLike, xi: float, *args, **kwargs) -> ArrayLike:
        _mu_t, _sigma_t, _w_logits = state_q.apply_fn(state_q.params, t)
        _dmudt = FirstOrderSetup.dmudt(state_q, t)
        _dsigmadt = FirstOrderSetup.dsigmadt(state_q, t)
        _x = x_t[:, None, :]

        log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
        relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]

        log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)
        _u_t = (relative_mixture_weights * (1 / _sigma_t * _dsigmadt * (_x - _mu_t) + _dmudt)).sum(axis=1)

        return _u_t + 0.5 * (xi ** 2) * log_q_t

    def u_t_det(self, state_q: TrainState, t: ArrayLike, x_t: ArrayLike, *args, **kwargs) -> ArrayLike:
        _mu_t, _sigma_t, _w_logits = state_q.apply_fn(state_q.params, t)
        _dmudt = FirstOrderSetup.dmudt(state_q, t)
        _dsigmadt = FirstOrderSetup.dsigmadt(state_q, t)
        _x = x_t[:, None, :]

        log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
        relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]

        return (relative_mixture_weights * (1 / _sigma_t * _dsigmadt * (_x - _mu_t) + _dmudt)).sum(axis=1)
