import jax
import jax.numpy as jnp
from flax import linen as nn

interp = jax.vmap(jnp.interp, in_axes=(None, None, 1))

class Interpolant(nn.Module):
  T: float
  A: float
  B: float
  ndim: int
  n_points: int = 100
  @nn.compact
  def __call__(self, t):
    t = t/self.T
    ndim = self.ndim
    t_grid = jnp.linspace(0,1,self.n_points)
    S_0 = jnp.log(1e-2)*jnp.eye(ndim)
    S_0_vec = S_0[jnp.tril_indices(ndim)]
    mu_params = self.param('mu_params', lambda rng: jnp.linspace(A[0], B[0], self.n_points)[1:-1])
    S_params = self.param('S_params', lambda rng: jnp.linspace(S_0_vec, S_0_vec, self.n_points)[1:-1])
    y_grid = jnp.concatenate([self.A, mu_params, self.B])
    S_grid = jnp.concatenate([S_0_vec[None,:], S_params, S_0_vec[None,:]])

    @jax.vmap
    def get_tril(v):
      a = jnp.zeros((ndim,ndim))
      a = a.at[jnp.tril_indices(ndim)].set(v)
      return a

    mu = interp(t.flatten(), t_grid, y_grid).T
    S = interp(t.flatten(), t_grid, S_grid).T
    S = get_tril(S)
    S = jnp.tril(2*jax.nn.sigmoid(S) - 1.0, k=-1) + jnp.eye(ndim)[None,...]*jnp.exp(S)
    return mu, S

class MLPfull(nn.Module):
  T: float
  A: float
  B: float
  ndim: int
  xi_0: float = 1e-2
  @nn.compact
  def __call__(self, t):
    t = t/self.T
    ndim = self.ndim
    h_mu = (1-t)*self.A + t*self.B
    S_0 = self.xi_0*jnp.eye(ndim)
    S_0 = S_0[None,...]
    h_S = (1-2*t*(1-t))[...,None]*S_0
    h = jnp.hstack([h_mu, h_S.reshape(-1,ndim*ndim), t])
    h = nn.Dense(256)(h)
    h = nn.swish(h)
    h = nn.Dense(256)(h)
    h = nn.swish(h)
    h = nn.Dense(256)(h)
    h = nn.swish(h)
    h = nn.Dense(ndim + ndim*(ndim+1)//2)(h)
    mu = h_mu + (1-t)*t*h[:,:ndim]

    @jax.vmap
    def get_tril(v):
      a = jnp.zeros((ndim,ndim))
      a = a.at[jnp.tril_indices(ndim)].set(v)
      return a
    # S = h[:,ndim:].reshape(-1,ndim,ndim)
    S = get_tril(h[:,ndim:])
    S = jnp.tril(2*jax.nn.sigmoid(S) - 1.0, k=-1) + jnp.eye(ndim)[None,...]*jnp.exp(S)
    S = h_S + 2*((1-t)*t)[...,None]*S
    return mu, S
  
class MLPdiag(nn.Module):
  T: float
  A: float
  B: float
  ndim: int
  xi_0: float = 1e-2
  @nn.compact
  def __call__(self, t):
    t = t/self.T
    ndim = self.ndim
    h_mu = (1-t)*self.A + t*self.B
    h = jnp.hstack([h_mu, t])
    h = nn.Dense(256)(h)
    h = nn.swish(h)
    h = nn.Dense(256)(h)
    h = nn.swish(h)
    h = nn.Dense(256)(h)
    h = nn.swish(h)
    h = nn.Dense(2*ndim)(h)
    mu = h_mu + (1-t)*t*h[:,:ndim]
    sigma = (1-t)*self.xi_0 + t*self.xi_0 + (1-t)*t*jnp.exp(h[:,ndim:])
    return mu, sigma
