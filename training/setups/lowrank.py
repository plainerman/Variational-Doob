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


class LowRankSpline(nn.Module):
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
        print("WARNING: Mixtures for low rank not yet implemented!")
        assert self.num_mixtures == 1, "Mixtures for low rank not yet implemented!"

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


class LowRankWrapper(WrappedModule):
    A: ArrayLike
    B: ArrayLike
    num_mixtures: int
    trainable_weights: bool
    base_sigma: float

    @nn.compact
    def _post_process(self, h: ArrayLike, t: ArrayLike):
        ndim = self.A.shape[0]
        num_mixtures = self.num_mixtures

        print("WARNING: Mixtures for low rank not yet implemented!")
        assert num_mixtures == 1, "Mixtures for low rank not yet implemented!"

        h_mu = (1 - t) * self.A + t * self.B
        S_0 = jnp.eye(ndim, dtype=jnp.float32)
        S_0 = S_0 * jnp.vstack([self.base_sigma * jnp.ones((ndim // 2, 1), dtype=jnp.float32),
                                self.base_sigma * jnp.ones((ndim // 2, 1), dtype=jnp.float32)])
        S_0 = S_0[None, ...]
        h_S = (1 - 2 * t * (1 - t))[..., None] * S_0

        # TODO: I think we can just multiply num_mixtures here and then do reshaping
        h = nn.Dense(ndim + ndim * (ndim + 1) // 2)(h)
        mu = h_mu + (1 - t) * t * h[:, :ndim]

        @jax.vmap
        def get_tril(v):
            a = jnp.zeros((ndim, ndim), dtype=jnp.float32)
            a = a.at[jnp.tril_indices(ndim)].set(v)
            return a

        S = get_tril(h[:, ndim:])
        S = jnp.tril(2 * jax.nn.sigmoid(S) - 1.0, k=-1) + jnp.eye(ndim, dtype=jnp.float32)[None, ...] * jnp.exp(S)
        S = h_S + 2 * ((1 - t) * t)[..., None] * S

        if self.trainable_weights:
            w_logits = self.param('w_logits', nn.initializers.zeros_init(), (num_mixtures,), dtype=jnp.float32)
        else:
            w_logits = jnp.zeros(num_mixtures, dtype=jnp.float32)

        return mu, S, w_logits


class LowRankSetup(DriftedSetup):
    def construct_loss(self, state_q: TrainState, gamma: float, BS: int) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], ArrayLike], ArrayLike]:
        def loss_fn(params_q: Union[FrozenVariableDict, Dict[str, Any]], key: ArrayLike) -> ArrayLike:
            ndim = self.model_q.A.shape[-1]

            key = jax.random.split(key)
            t = self.T * jax.random.uniform(key[0], [BS, 1], dtype=jnp.float32)
            eps = jax.random.normal(key[1], [BS, ndim, 1], dtype=jnp.float32)

            def v_t(_eps, _t):
                _mu_t, _S_t_val, _w_logits, _dmudt, _dSdt_val = forward_and_derivatives(state_q, _t, params_q)

                _x = _mu_t + jax.lax.batch_matmul(_S_t_val, _eps).squeeze()
                dlogdx = -jax.scipy.linalg.solve_triangular(jnp.transpose(_S_t_val, (0, 2, 1)), _eps)
                # S_t_val_inv = jnp.transpose(jnp.linalg.inv(S_t_val), (0,2,1))
                # dlogdx = -jax.lax.batch_matmul(S_t_val_inv, _eps)
                dSigmadt = jax.lax.batch_matmul(_dSdt_val, jnp.transpose(_S_t_val, (0, 2, 1)))
                dSigmadt += jax.lax.batch_matmul(_S_t_val, jnp.transpose(_dSdt_val, (0, 2, 1)))
                u_t = _dmudt - 0.5 * jax.lax.batch_matmul(dSigmadt, dlogdx).squeeze()
                out = (u_t - self._drift(_x.reshape(BS, ndim), gamma)) + 0.5 * (self.xi ** 2) * dlogdx.squeeze()
                return out

            loss = 0.5 * ((v_t(eps, t) / self.xi) ** 2).sum(1, keepdims=True)
            print(loss.shape, 'loss.shape', 'loss.dtype', loss.dtype, flush=True)
            return loss.mean()

        return loss_fn

    def u_t(self, state_q: TrainState, t: ArrayLike, x_t: ArrayLike, deterministic: bool, *args, **kwargs) -> ArrayLike:
        _mu_t, _S_t_val, _w_logits, _dmudt, _dSdt_val = forward_and_derivatives(state_q, t)

        dSigmadt = jax.lax.batch_matmul(_dSdt_val, jnp.transpose(_S_t_val, (0, 2, 1)))
        dSigmadt += jax.lax.batch_matmul(_S_t_val, jnp.transpose(_dSdt_val, (0, 2, 1)))
        STdlogdx = jax.scipy.linalg.solve_triangular(_S_t_val, (x_t - _mu_t)[..., None])
        dlogdx = -jax.scipy.linalg.solve_triangular(jnp.transpose(_S_t_val, (0, 2, 1)), STdlogdx)

        if deterministic:
            return _dmudt + (-0.5 * jax.lax.batch_matmul(dSigmadt, dlogdx)).squeeze()

        return _dmudt + (-0.5 * jax.lax.batch_matmul(dSigmadt, dlogdx) + 0.5 * self.xi ** 2 * dlogdx).squeeze()
