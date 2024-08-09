import jax
import jax.numpy as jnp

def get_parameterization_fn(params, state):
  gauss_params = lambda _t: state.apply_fn(params, _t)
  def dgauss_paramsdt(_t):
    _gauss_params = lambda _t: jax.tree.map(lambda _a: _a.sum(0), gauss_params(_t))
    return jax.tree.map(lambda a: a.squeeze().T, jax.jacrev(_gauss_params)(_t))
  return gauss_params, dgauss_paramsdt

def v_t_diag(_eps, _t, params, state):
  gauss_params, dgauss_paramsdt = get_parameterization_fn(params, state)
  _eps = _eps.squeeze()
  mu_t_val, s_val = gauss_params(_t)
  dmudt_val, dsdt_val = dgauss_paramsdt(_t)
  _x = mu_t_val + jnp.sqrt(s_val)*_eps
  dlogdx = -_eps/jnp.sqrt(s_val)
  u_t = dmudt_val - 0.5*dlogdx*dsdt_val
  out = (u_t - drift(_x)) + 0.5*(xi**2)*dlogdx.squeeze()
  return out

def v_t_full(_eps, _t, params, state):
  gauss_params, dgauss_paramsdt = get_parameterization_fn(params, state)
  mu_t_val, S_t_val = gauss_params(_t)
  dmudt_val, dSdt_val = dgauss_paramsdt(_t)
  _x = mu_t_val + jax.lax.batch_matmul(S_t_val, _eps).squeeze()
  dlogdx = -jax.scipy.linalg.solve_triangular(jnp.transpose(S_t_val, (0,2,1)), _eps)
  dSigmadt = jax.lax.batch_matmul(dSdt_val, jnp.transpose(S_t_val, (0,2,1)))
  dSigmadt += jax.lax.batch_matmul(S_t_val, jnp.transpose(dSdt_val, (0,2,1)))
  u_t = dmudt_val - 0.5*jax.lax.batch_matmul(dSigmadt, dlogdx).squeeze()
  out = (u_t - drift(_x)) + 0.5*(xi**2)*dlogdx.squeeze()
  return out
