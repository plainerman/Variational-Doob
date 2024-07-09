from dataclasses import dataclass
from jax.typing import ArrayLike
from flax import linen as nn
import jax.numpy as jnp
from typing import Union, Dict, Any, Callable, Tuple
from flax.training.train_state import TrainState
import jax
from flax.typing import FrozenVariableDict
from model.utils import WrappedModule
from systems import System
from training.setups.drift import DriftedSetup
from training.utils import forward_and_derivatives


class LowRankWrapper(WrappedModule):
    A: ArrayLike
    B: ArrayLike
    num_mixtures: int
    trainable_weights: bool
    base_sigma: float

    def _pre_process(self, t: ArrayLike) -> Tuple[ArrayLike, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        ndim = self.A.shape[0]

        h_mu = (1 - t) * self.A + t * self.B
        S_0 = jnp.eye(ndim)
        S_0 = S_0 * jnp.vstack([self.base_sigma * jnp.ones((ndim // 2, 1)), self.base_sigma * jnp.ones((ndim // 2, 1))])
        S_0 = S_0[None, ...]
        h_S = (1 - 2 * t * (1 - t))[..., None] * S_0
        return jnp.hstack([h_mu, h_S.reshape(-1, ndim * ndim), t]), (h_mu, h_S, t)

    @nn.compact
    def _post_process(self, h: ArrayLike, h_mu: ArrayLike, h_S: ArrayLike, t: ArrayLike):
        ndim = self.A.shape[0]
        num_mixtures = self.num_mixtures

        print("WARNING: Mixtures for low rank not yet implemented!")
        assert num_mixtures == 1, "Mixtures for low rank not yet implemented!"

        # TODO: I think we can just multiply num_mixtures here and then do reshaping
        h = nn.Dense(ndim + ndim * (ndim + 1) // 2)(h)
        mu = h_mu + (1 - t) * t * h[:, :ndim]

        @jax.vmap
        def get_tril(v):
            a = jnp.zeros((ndim, ndim))
            a = a.at[jnp.tril_indices(ndim)].set(v)
            return a

        S = get_tril(h[:, ndim:])
        S = jnp.tril(2 * jax.nn.sigmoid(S) - 1.0, k=-1) + jnp.eye(ndim)[None, ...] * jnp.exp(S)
        S = h_S + 2 * ((1 - t) * t)[..., None] * S

        if self.trainable_weights:
            w_logits = self.param('w_logits', nn.initializers.zeros_init(), (num_mixtures,))
        else:
            w_logits = jnp.zeros(num_mixtures)

        print('mu.shape', mu.shape)
        print('S.shape', S.shape)

        return mu, S, w_logits


@dataclass
class LowRankSetup(DriftedSetup):
    model_q: LowRankWrapper
    T: float

    def __init__(self, system: System, model_q: LowRankWrapper, xi: ArrayLike, order: str, T: float):
        super().__init__(system, model_q, xi, order)
        self.T = T

    def construct_loss(self, state_q: TrainState, gamma: float, BS: int) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], ArrayLike], ArrayLike]:
        def loss_fn(params_q: Union[FrozenVariableDict, Dict[str, Any]], key: ArrayLike) -> ArrayLike:
            ndim = self.model_q.A.shape[-1]

            key = jax.random.split(key)
            t = self.T * jax.random.uniform(key[0], [BS, 1])
            eps = jax.random.normal(key[1], [BS, ndim, 1])

            mu_t = lambda _t: state_q.apply_fn(params_q, _t)[0]
            S_t = lambda _t: state_q.apply_fn(params_q, _t)[1]

            def dmudt(_t):
                _dmudt = jax.jacrev(lambda _t: mu_t(_t).sum(0))
                return _dmudt(_t).squeeze().T

            def dSdt(_t):
                _dSdt = jax.jacrev(lambda _t: S_t(_t).sum(0))
                return _dSdt(_t).squeeze().T

            def v_t(_eps, _t):
                S_t_val, dSdt_val = S_t(_t), dSdt(_t)
                _x = mu_t(_t) + jax.lax.batch_matmul(S_t_val, _eps).squeeze()
                dlogdx = -jax.scipy.linalg.solve_triangular(jnp.transpose(S_t_val, (0, 2, 1)), _eps)
                # S_t_val_inv = jnp.transpose(jnp.linalg.inv(S_t_val), (0,2,1))
                # dlogdx = -jax.lax.batch_matmul(S_t_val_inv, _eps)
                dSigmadt = jax.lax.batch_matmul(dSdt_val, jnp.transpose(S_t_val, (0, 2, 1)))
                dSigmadt += jax.lax.batch_matmul(S_t_val, jnp.transpose(dSdt_val, (0, 2, 1)))
                u_t = dmudt(_t) - 0.5 * jax.lax.batch_matmul(dSigmadt, dlogdx).squeeze()
                out = (u_t - self._drift(_x.reshape(BS, ndim), gamma)) + 0.5 * (self.xi ** 2) * dlogdx.squeeze()
                return out

            loss = 0.5 * ((v_t(eps, t) / self.xi) ** 2).sum(1, keepdims=True)
            print(loss.shape, 'loss.shape', flush=True)
            return loss.mean()

            # ndim = self.model_q.A.shape[-1]
            # key = jax.random.split(key)
            #
            # t = self.T * jax.random.uniform(key[0], [BS, 1], dtype=jnp.float32)
            # #TODO: the following needs to be changed for num gaussians. It should be BS, num_mitures, ndim
            # eps = jax.random.normal(key[1], [BS, ndim, 1], dtype=jnp.float32)
            #
            # def v_t(_eps, _t):
            #     """This function is equal to v_t * xi ** 2."""
            #     _mu_t, _sigma_t, _w_logits, _dmudt, _dsigmadt = forward_and_derivatives(state_q, _t, params_q)
            #     _i = jax.random.categorical(key[2], _w_logits, shape=[BS, ])
            #
            #     _x = _mu_t[jnp.arange(BS), _i, None] + _sigma_t[jnp.arange(BS), _i, None] * eps
            #
            #     if _mu_t.shape[1] == 1:
            #         # This completely ignores the weights and saves some time
            #         relative_mixture_weights = 1
            #     else:
            #         log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
            #         relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]
            #
            #     log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)
            #     u_t = (relative_mixture_weights * (1 / _sigma_t * _dsigmadt * (_x - _mu_t) + _dmudt)).sum(axis=1)
            #
            #     return u_t - self._drift(_x.reshape(BS, ndim), gamma) + 0.5 * (self.xi ** 2) * log_q_t
            #
            # loss = 0.5 * ((v_t(eps, t) / self.xi) ** 2).sum(-1, keepdims=True)
            # return loss.mean()

        return loss_fn

    def u_t(self, state_q: TrainState, t: ArrayLike, x_t: ArrayLike, deterministic: bool, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError

        # _mu_t, _sigma_t, _w_logits, _dmudt, _dsigmadt = forward_and_derivatives(state_q, t)
        # _x = x_t[:, None, :]
        #
        # log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
        # relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]
        #
        # _u_t = (relative_mixture_weights * (1 / _sigma_t * _dsigmadt * (_x - _mu_t) + _dmudt)).sum(axis=1)
        #
        # if deterministic:
        #     return _u_t
        #
        # log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)
        #
        # return _u_t + 0.5 * (self.xi ** 2) * log_q_t
