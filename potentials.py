import jax.numpy as jnp


def U_double_well(xs, a=1.0, b=-4.0, c=0, d=1.0, beta=1.0):
    if xs.ndim == 1:
        xs = xs.reshape(1, -1)

    x, y = xs[:, 0], xs[:, 1]
    return beta * (a * (x ** 4) + b * (x ** 2) + c * x + 0.5 * d * (y ** 2))


def U_double_well_hard(xs, beta=1.0):
    if xs.ndim == 1:
        xs = xs.reshape(1, -1)

    A = jnp.array([[-3, 0]])
    B = jnp.array([[3, 0]])
    U1 = -(((xs - A) @ jnp.array([[1, 0.5], [0.5, 1.0]])) * (xs - A)).sum(1)
    U2 = -(((xs - B) @ jnp.array([[1, -0.5], [-0.5, 1.0]])) * (xs - B)).sum(1)
    out = -jnp.log(jnp.exp(U1 - jnp.maximum(U1, U2)) + jnp.exp(U2 - jnp.maximum(U1, U2))) - jnp.maximum(U1, U2)
    return beta * out


def U_double_well_dual_channel(xs, beta=1.0):
    if xs.ndim == 1:
        xs = xs.reshape(1, -1)

    x, y = xs[:, 0], xs[:, 1]
    borders = x ** 6 + y ** 6
    e1 = +2.0 * jnp.exp(-(12.0 * (x - 0.00) ** 2 + 12.0 * (y - 0.00) ** 2))
    e2 = -1.0 * jnp.exp(-(12.0 * (x + 0.50) ** 2 + 12.0 * (y + 0.00) ** 2))
    e3 = -1.0 * jnp.exp(-(12.0 * (x - 0.50) ** 2 + 12.0 * (y + 0.00) ** 2))
    return beta * (borders + e1 + e2 + e3)


def U_mueller_brown(xs, beta=1.0):
    if xs.ndim == 1:
        xs = xs.reshape(1, -1)

    x, y = xs[:, 0], xs[:, 1]
    e1 = -200 * jnp.exp(-(x - 1) ** 2 - 10 * y ** 2)
    e2 = -100 * jnp.exp(-x ** 2 - 10 * (y - 0.5) ** 2)
    e3 = -170 * jnp.exp(-6.5 * (0.5 + x) ** 2 + 11 * (x + 0.5) * (y - 1.5) - 6.5 * (y - 1.5) ** 2)
    e4 = 15.0 * jnp.exp(0.7 * (1 + x) ** 2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1) ** 2)
    return beta * (e1 + e2 + e3 + e4)


double_well = (U_double_well, jnp.array([-jnp.sqrt(2), 0]), jnp.array([jnp.sqrt(2), 0]))
double_well_hard = (U_double_well_hard, jnp.array([-3, 0]), jnp.array([3, 0]))
double_well_dual_channel = (U_double_well_dual_channel, jnp.array([-0.5, 0]), jnp.array([0.5, 0]))
mueller_brown = (U_mueller_brown, jnp.array([-0.55828035, 1.44169]), jnp.array([0.62361133, 0.02804632]))
