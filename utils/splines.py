import jax
import jax.numpy as jnp


def compute_spline_coefficients(x_knots, y_knots):
    n = len(x_knots) - 1
    h = jnp.diff(x_knots)
    b = (jnp.diff(y_knots, axis=0).T / h).T

    u = jnp.zeros(n + 1, dtype=jnp.float32)
    v = jnp.zeros((n + 1,) + y_knots.shape[1:], dtype=jnp.float32)

    u = u.at[1:n].set(2 * (h[:-1] + h[1:]))
    v = v.at[1:n].set(6 * (b[1:] - b[:-1]))

    u = u.at[0].set(1)
    u = u.at[n].set(1)

    # Forward elimination
    factor = (h[:-1] ** 2) / u[:-2]
    u = u.at[1:n].add(-factor)
    v = v.at[1:n].add(-factor[:, None] * v[:-2])

    # Backward substitution
    factor = (h[1:] / u[1:n])
    m = jnp.zeros_like(v)
    m = m.at[1:n].set((v[1:n] - factor[:, None] * v[2:]) / u[1:n][:, None])

    return m


def evaluate_cubic_spline(x, x_knots, y_knots, m):
    i = jnp.searchsorted(x_knots, x) - 1
    i = jnp.clip(i, 0, len(x_knots) - 2)  # Ensure i is within bounds
    h = x_knots[i + 1] - x_knots[i]
    A = (x_knots[i + 1] - x) / h
    B = (x - x_knots[i]) / h
    C = (1 / 6) * (A ** 3 - A) * h ** 2
    D = (1 / 6) * (B ** 3 - B) * h ** 2
    y = A * y_knots[i] + B * y_knots[i + 1] + C * m[i] + D * m[i + 1]
    return y


def compute_cubic_spline(t, x_knots, y_knots):
    m = compute_spline_coefficients(x_knots, y_knots)
    return evaluate_cubic_spline(t, x_knots, y_knots, m)


vectorized_cubic_spline = jax.vmap(compute_cubic_spline, in_axes=(0, None, None))

_vectorized_linear_spline = jax.vmap(jnp.interp, in_axes=(None, None, 1))


def vectorized_linear_spline(t, x_knots, y_knots):
    return _vectorized_linear_spline(t, x_knots, y_knots).T
