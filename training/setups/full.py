from jax.typing import ArrayLike
from flax import linen as nn
import jax.numpy as jnp
from typing import Union, Dict, Any, Callable, Tuple, Optional
from flax.training.train_state import TrainState
import jax
from flax.typing import FrozenVariableDict
from model.utils import WrappedModule
from training.setups.drift import DriftedSetup
from training.utils import forward_and_derivatives
from utils.splines import vectorized_cubic_spline, vectorized_linear_spline


def _matmul_batched(a, b):
    return jax.lax.batch_matmul(a, b)


def _dSigmadt_batched(_S_t, _dSdt):
    dSigmadt = jax.lax.batch_matmul(_dSdt, jnp.transpose(_S_t, (0, 2, 1)))
    return dSigmadt + jax.lax.batch_matmul(_S_t, jnp.transpose(_dSdt, (0, 2, 1)))


_dSigmadt_batched = jax.vmap(_dSigmadt_batched, in_axes=(1, 1), out_axes=1)
_matmul_batched = jax.vmap(_matmul_batched, in_axes=1, out_axes=1)


class FullRankSpline(nn.Module):
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
        print("WARNING: Mixtures for full rank spline not yet implemented!")
        assert self.num_mixtures == 1, "Mixtures for full rank not yet implemented!"

        ndim = self.A.shape[0]
        t = t / self.T
        t_grid = jnp.linspace(0, 1, self.n_points, dtype=jnp.float32)
        S_0 = jnp.log(self.base_sigma) * jnp.eye(ndim, dtype=jnp.float32)
        S_0_vec = S_0[jnp.tril_indices(ndim)]
        mu_params = self.param('mu_params', lambda rng: jnp.linspace(self.A, self.B, self.n_points)[1:-1])
        S_params = self.param('S_params', lambda rng: jnp.linspace(S_0_vec, S_0_vec, self.n_points)[1:-1])
        y_grid = jnp.concatenate([self.A.reshape(1, -1), mu_params, self.B.reshape(1, -1)])
        S_grid = jnp.concatenate([S_0_vec[None, :], S_params, S_0_vec[None, :]])

        @jax.vmap
        def get_tril(v):
            a = jnp.zeros((ndim, ndim), dtype=jnp.float32)
            a = a.at[jnp.tril_indices(ndim)].set(v)
            return a

        if self.interpolation == 'cubic':
            mu = vectorized_cubic_spline(t.flatten(), t_grid, y_grid)
            S = vectorized_cubic_spline(t.flatten(), t_grid, S_grid)
        elif self.interpolation == 'linear':
            mu = vectorized_linear_spline(t.flatten(), t_grid, y_grid)
            S = vectorized_linear_spline(t.flatten(), t_grid, S_grid)
        else:
            raise ValueError(f"Interpolation method {self.interpolation} not recognized.")

        S = get_tril(S)
        S = jnp.tril(2 * jax.nn.sigmoid(S) - 1.0, k=-1) + jnp.eye(ndim, dtype=jnp.float32)[None, ...] * jnp.exp(S)

        if self.trainable_weights:
            w_logits = self.param('w_logits', nn.initializers.zeros_init(), (self.num_mixtures,), dtype=jnp.float32)
        else:
            w_logits = jnp.zeros(self.num_mixtures, dtype=jnp.float32)

        out = (mu, S, w_logits)
        if self.transform:
            out = self.transform(out)

        return out


class FullRankWrapper(WrappedModule):
    A: ArrayLike
    B: ArrayLike
    num_mixtures: int
    trainable_weights: bool
    base_sigma: float

    @nn.compact
    def _post_process(self, h: ArrayLike, t: ArrayLike):
        BS = h.shape[0]
        ndim = self.A.shape[0]
        num_mixtures = self.num_mixtures

        h_mu = (1 - t) * self.A + t * self.B
        S_0 = jnp.eye(ndim, dtype=jnp.float32)
        S_0 = S_0 * jnp.vstack([self.base_sigma * jnp.ones((ndim // 2, 1), dtype=jnp.float32),
                                self.base_sigma * jnp.ones((ndim // 2, 1), dtype=jnp.float32)])
        S_0 = S_0[None, ...]
        h_S = (1 - 2 * t * (1 - t))[..., None] * S_0

        h = nn.Dense(self.num_mixtures * (ndim + ndim * (ndim + 1) // 2))(h)
        mu = (
                h_mu[:, None, :] +
                ((1 - t) * t)[:, None, :] * h[:, :self.num_mixtures * ndim].reshape(BS, self.num_mixtures, ndim)
        )

        @jax.vmap  # once for num_mixtures
        @jax.vmap  # once for batch
        def get_tril(v):
            a = jnp.zeros((ndim, ndim), dtype=jnp.float32)
            a = a.at[jnp.tril_indices(ndim)].set(v)
            return a

        S = h[:, self.num_mixtures * ndim:].reshape(BS, self.num_mixtures, ndim * (ndim + 1) // 2)
        S = get_tril(S)
        S = jnp.tril(2 * jax.nn.sigmoid(S) - 1.0, k=-1) + jnp.eye(ndim, dtype=jnp.float32)[None, ...] * jnp.exp(S)
        S = h_S[:, None, ...] + 2 * ((1 - t) * t)[..., None, None] * S

        if self.trainable_weights:
            w_logits = self.param('w_logits', nn.initializers.zeros_init(), (num_mixtures,), dtype=jnp.float32)
        else:
            w_logits = jnp.zeros(num_mixtures, dtype=jnp.float32)

        return mu, S, w_logits


class FullRankSetup(DriftedSetup):
    def construct_loss(self, state_q: TrainState, gamma: float, BS: int) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], ArrayLike], ArrayLike]:
        def loss_fn(params_q: Union[FrozenVariableDict, Dict[str, Any]], key: ArrayLike) -> ArrayLike:
            ndim = self.model_q.A.shape[-1]

            key = jax.random.split(key)
            t = self.T * jax.random.uniform(key[0], [BS, 1], dtype=jnp.float32)
            eps = jax.random.normal(key[1], [BS, ndim, 1], dtype=jnp.float32)

            def v_t(_eps, _t):
                _mu_t, _S_t_val, _w_logits, _dmudt, _dSdt_val = forward_and_derivatives(state_q, _t, params_q)
                _i = jax.random.categorical(key[2], _w_logits, shape=[BS, ])

                _x = (_mu_t[jnp.arange(BS), _i, None] +
                      jax.lax.batch_matmul(_S_t_val[jnp.arange(BS), _i], _eps).squeeze()[:, None, ...])

                if _mu_t.shape[1] == 1:
                    # This completely ignores the weights and saves some time
                    relative_weights = 1
                else:
                    log_q_i = jax.scipy.stats.multivariate_normal.logpdf(_x, _mu_t, _S_t_val)
                    relative_weights = jax.nn.softmax(_w_logits + log_q_i)[..., None]

                def _dlogdx_batched(_S_t):
                    print('TODO: How to handle this epsilon for mixtures?')
                    return -jax.scipy.linalg.solve_triangular(jnp.transpose(_S_t, (0, 2, 1)), _eps)

                _dlogdx_batched = jax.vmap(_dlogdx_batched, in_axes=1, out_axes=1)

                dSigmadt = _dSigmadt_batched(_S_t_val, _dSdt_val)
                dlogdx = _dlogdx_batched(_S_t_val)
                u_t = (relative_weights * (_dmudt - 0.5 * _matmul_batched(dSigmadt, dlogdx).squeeze(-1))).sum(axis=1)
                dlogdx = (relative_weights * dlogdx.squeeze(-1)).sum(axis=1)

                return (u_t - self._drift(_x.reshape(BS, ndim), gamma)) + 0.5 * (self.xi ** 2) * dlogdx

            loss = 0.5 * ((v_t(eps, t) / self.xi) ** 2).sum(1, keepdims=True)
            print(loss.shape, 'loss.shape', 'loss.dtype', loss.dtype, flush=True)
            return jnp.mean(loss)

        return loss_fn

    def u_t(self, state_q: TrainState, t: ArrayLike, x_t: ArrayLike, deterministic: bool, *args, **kwargs) -> ArrayLike:
        _mu_t, _S_t_val, _w_logits, _dmudt, _dSdt_val = forward_and_derivatives(state_q, t)
        _x = x_t[:, None, :]

        if _mu_t.shape[1] == 1:
            # This completely ignores the weights and saves some time
            relative_weights = 1
        else:
            log_q_i = jax.scipy.stats.multivariate_normal.logpdf(_x, _mu_t, _S_t_val)
            relative_weights = jax.nn.softmax(_w_logits + log_q_i)[..., None]

        def solve_triangular_batched(a, b):
            return jax.scipy.linalg.solve_triangular(a, b)

        def dlogdx_batched(_S_t, _STdlogdx):
            return -jax.scipy.linalg.solve_triangular(jnp.transpose(_S_t, (0, 2, 1)), _STdlogdx)

        solve_triangular_batched = jax.vmap(solve_triangular_batched, in_axes=(1, 1), out_axes=1)
        dlogdx_batched = jax.vmap(dlogdx_batched, in_axes=(1, 1), out_axes=1)

        dSigmadt = _dSigmadt_batched(_S_t_val, _dSdt_val)

        STdlogdx = solve_triangular_batched(_S_t_val, (_x - _mu_t)[..., None])
        dlogdx = dlogdx_batched(_S_t_val, STdlogdx)

        _u_t = (relative_weights * (_dmudt + (-0.5 * _matmul_batched(dSigmadt, dlogdx).squeeze(-1)))).sum(axis=1)

        if deterministic:
            return _u_t

        dlogdx = (relative_weights * dlogdx.squeeze(-1)).sum(axis=1)
        return _u_t + 0.5 * self.xi ** 2 * dlogdx
