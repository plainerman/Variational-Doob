from dataclasses import dataclass
from jax.typing import ArrayLike
from flax import linen as nn
import jax.numpy as jnp
from typing import Union, Dict, Any, Callable, Optional
from flax.training.train_state import TrainState
import jax
from flax.typing import FrozenVariableDict
from model.utils import WrappedModule
from systems import System
from training.setups.drift import DriftedSetup
from training.utils import forward_and_derivatives
from utils.splines import vectorized_cubic_spline, vectorized_linear_spline


class DiagonalSpline(nn.Module):
    n_points: int
    interpolation: str
    T: float
    transform: Optional[Callable[[Any], Any]]
    A: ArrayLike
    B: ArrayLike
    num_mixtures: int
    trainable_weights: bool
    base_sigma: float

    @nn.compact
    def __call__(self, t):
        ndim = self.A.shape[0]
        BS = t.shape[0]
        t = t / self.T
        t_grid = jnp.linspace(0, 1, self.n_points, dtype=jnp.float32)

        A = (jnp.ones((self.num_mixtures, ndim), dtype=self.A.dtype) * self.A).reshape(-1)
        B = (jnp.ones((self.num_mixtures, ndim), dtype=self.A.dtype) * self.B).reshape(-1)

        base_sigma = self.base_sigma * jnp.ones((self.num_mixtures * ndim), dtype=self.A.dtype)
        base_sigma = jnp.log(base_sigma)

        mu_params = self.param('mu_params', lambda rng: jnp.linspace(A, B, self.n_points)[1:-1])
        sigma_params = self.param('sigma_params', lambda rng: jnp.linspace(base_sigma, base_sigma, self.n_points)[1:-1])

        y_grid = jnp.concatenate([A.reshape(1, -1), mu_params, B.reshape(1, -1)])
        sigma_grid = jnp.concatenate([base_sigma.reshape(1, -1), sigma_params, base_sigma.reshape(1, -1)])

        if self.interpolation == 'cubic':
            mu = vectorized_cubic_spline(t.flatten(), t_grid, y_grid)
            sigma = vectorized_cubic_spline(t.flatten(), t_grid, sigma_grid)
        elif self.interpolation == 'linear':
            mu = vectorized_linear_spline(t.flatten(), t_grid, y_grid)
            sigma = vectorized_linear_spline(t.flatten(), t_grid, sigma_grid)
        else:
            raise ValueError(f"Interpolation method {self.interpolation} not recognized.")

        mu = mu.reshape(BS, self.num_mixtures, ndim)
        sigma = jnp.exp(sigma.reshape(BS, self.num_mixtures, ndim))

        if self.trainable_weights:
            w_logits = self.param('w_logits', nn.initializers.zeros_init(), (self.num_mixtures,), dtype=jnp.float32)
        else:
            w_logits = jnp.zeros(self.num_mixtures, dtype=jnp.float32)

        out = (mu, sigma, w_logits)
        if self.transform:
            out = self.transform(out)

        return out


class DiagonalWrapper(WrappedModule):
    A: ArrayLike
    B: ArrayLike
    num_mixtures: int
    trainable_weights: bool
    base_sigma: float

    @nn.compact
    def _post_process(self, h: ArrayLike, t: ArrayLike):
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
            w_logits = self.param('w_logits', nn.initializers.zeros_init(), (num_mixtures,), dtype=jnp.float32)
        else:
            w_logits = jnp.zeros(num_mixtures, dtype=jnp.float32)

        return mu, sigma, w_logits


class DiagonalSetup(DriftedSetup):
    def construct_loss(self, state_q: TrainState, gamma: float, BS: int) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], ArrayLike], ArrayLike]:

        def loss_fn(params_q: Union[FrozenVariableDict, Dict[str, Any]], key: ArrayLike) -> ArrayLike:
            ndim = self.model_q.A.shape[-1]
            key = jax.random.split(key)

            t = self.T * jax.random.uniform(key[0], [BS, 1], dtype=jnp.float32)
            eps = jax.random.normal(key[1], [BS, 1, ndim], dtype=jnp.float32)

            def v_t(_eps, _t):
                """This function is equal to v_t * xi ** 2."""
                _mu_t, _sigma_t, _w_logits, _dmudt, _dsigmadt = forward_and_derivatives(state_q, _t, params_q)
                _i = jax.random.categorical(key[2], _w_logits, shape=[BS, ])

                _x = _mu_t[jnp.arange(BS), _i, None] + _sigma_t[jnp.arange(BS), _i, None] * eps

                if _mu_t.shape[1] == 1:
                    # This completely ignores the weights and saves some time
                    relative_mixture_weights = 1
                else:
                    log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
                    relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]

                log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)
                u_t = (relative_mixture_weights * (1 / _sigma_t * _dsigmadt * (_x - _mu_t) + _dmudt)).sum(axis=1)

                return u_t - self._drift(_x.reshape(BS, ndim), gamma) + 0.5 * (self.xi ** 2) * log_q_t

            loss = 0.5 * ((v_t(eps, t) / self.xi) ** 2).sum(-1, keepdims=True)
            return loss.mean()

        return loss_fn

    def u_t(self, state_q: TrainState, t: ArrayLike, x_t: ArrayLike, deterministic: bool, *args, **kwargs) -> ArrayLike:
        _mu_t, _sigma_t, _w_logits, _dmudt, _dsigmadt = forward_and_derivatives(state_q, t)
        _x = x_t[:, None, :]

        log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
        if _w_logits.shape[0] == 1:
            # This completely ignores the weights and saves some time
            relative_mixture_weights = 1
        else:
            relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]

        _u_t = (relative_mixture_weights * (1 / _sigma_t * _dsigmadt * (_x - _mu_t) + _dmudt)).sum(axis=1)

        if deterministic:
            return _u_t

        log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)

        return _u_t + 0.5 * (self.xi ** 2) * log_q_t
