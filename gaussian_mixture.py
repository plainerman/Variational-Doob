from functools import partial
import utils.toy_plot_helpers as toy
from flax import linen as nn
from flax.training import train_state
import optax
import jax
import jax.numpy as jnp
from tqdm import trange
import matplotlib.pyplot as plt
import os


@jax.jit
def U(xs, beta=1.0):
    x, y = xs[:, 0], xs[:, 1]
    borders = x ** 6 + y ** 6
    e1 = +2.0 * jnp.exp(-(12.0 * (x - 0.00) ** 2 + 12.0 * (y - 0.00) ** 2))
    e2 = -1.0 * jnp.exp(-(12.0 * (x + 0.50) ** 2 + 12.0 * (y + 0.00) ** 2))
    e3 = -1.0 * jnp.exp(-(12.0 * (x - 0.50) ** 2 + 12.0 * (y + 0.00) ** 2))
    return beta * (borders + e1 + e2 + e3)


dUdx_fn = jax.jit(jax.grad(lambda _x: U(_x).sum()))

plot_energy_surface = partial(toy.plot_energy_surface, U, [], jnp.array((-1, 1)), jnp.array((-1, 1)), levels=20)


def create_mlp_q(A, B, T, num_mixtures):
    class MLPq(nn.Module):
        @nn.compact
        def __call__(self, t):
            """
              in_shape: (batch, t)
              out_shape: (batch, num_mixtures, data)
            """
            t = t / T
            h = nn.Dense(128)(t - 0.5)
            h = nn.swish(h)
            h = nn.Dense(128)(h)
            h = nn.swish(h)
            h = nn.Dense(128)(h)
            h = nn.swish(h)
            h = nn.Dense(2 * num_mixtures + num_mixtures)(h)

            mu = (((1 - t) * A)[:, None, :] + (t * B)[:, None, :] +
                  ((1 - t) * t * h[:, :2 * num_mixtures]).reshape(-1, num_mixtures, A.shape[-1]))
            sigma = (1 - t) * 1e-2 * 2.5 + t * 1e-2 * 2.5 + (1 - t) * t * jnp.exp(h[:, 2 * num_mixtures:])
            return mu, sigma[:, :, None]

    return MLPq(), jnp.zeros(num_mixtures)


def train(q, w_logits, epochs):
    BS = 512
    key = jax.random.PRNGKey(1)
    key, *init_key = jax.random.split(key, 3)
    params_q = q.init(init_key[0], jnp.ones([BS, 1]))

    optimizer_q = optax.adam(learning_rate=1e-4)
    state_q = train_state.TrainState.create(apply_fn=q.apply,
                                            params=params_q,
                                            tx=optimizer_q)

    def loss_fn(params_q, key):
        key = jax.random.split(key)

        t = T * jax.random.uniform(key[0], [BS, 1])
        eps = jax.random.normal(key[1], [BS, 1, A.shape[-1]])
        i = jax.random.categorical(key[2], w_logits, shape=(BS,))

        mu_t = lambda _t: state_q.apply_fn(params_q, _t)[0]
        sigma_t = lambda _t: state_q.apply_fn(params_q, _t)[1]

        def dmudt(_t):
            _dmudt = jax.jacrev(lambda _t: mu_t(_t).sum(0).T)
            return _dmudt(_t).squeeze(axis=-1).T

        def dsigmadt(_t):
            _dsigmadt = jax.jacrev(lambda _t: sigma_t(_t).sum(0).T)
            return _dsigmadt(_t).squeeze(axis=-1).T

        dUdx_fn = jax.grad(lambda _x: U(_x).sum())

        def v_t(_eps, _t, _i, _w_logits):
            """This function is equal to v_t * xi ** 2."""
            _mu_t = mu_t(_t)
            _sigma_t = sigma_t(_t)
            _x = _mu_t[jnp.arange(BS), _i, None] + _sigma_t[jnp.arange(BS), _i, None] * eps

            log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
            relative_mixture_weights = jax.nn.softmax(_w_logits + log_q_i)[:, :, None]

            log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)
            u_t = (relative_mixture_weights * (1 / _sigma_t * dsigmadt(_t) * (_x - _mu_t) + dmudt(_t))).sum(axis=1)
            b_t = -dUdx_fn(_x.reshape(BS, A.shape[-1]))

            return u_t - b_t + 0.5 * (xi ** 2) * log_q_t

        loss = 0.5 * ((v_t(eps, t, i, w_logits) / xi) ** 2).sum(-1, keepdims=True)
        print(loss.shape, 'loss.shape', flush=True)
        return loss.mean()

    @jax.jit
    def train_step(state_q, key):
        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grads = grad_fn(state_q.params, key)
        state_q = state_q.apply_gradients(grads=grads)
        return state_q, loss

    key, loc_key = jax.random.split(key)
    state_q, loss = train_step(state_q, loc_key)

    loss_plot = []
    for _ in trange(epochs):
        key, loc_key = jax.random.split(key)
        state_q, loss = train_step(state_q, loc_key)
        loss_plot.append(loss)

    return state_q, loss_plot


def draw_samples_qt(state_q, w_logits, num_samples, T, A, key):
    num_mixtures = len(w_logits)
    w = jax.nn.softmax(w_logits)[None, :, None]
    t = T * jnp.linspace(0, 1, num_samples).reshape((-1, 1))
    eps = jax.random.normal(key, [num_samples, num_mixtures, A.shape[-1]])
    mu_t, sigma_t = state_q.apply_fn(state_q.params, t)
    return (w * (mu_t + sigma_t * eps)).sum(axis=1)


def sample_stochastically(state_q, w_logits, num_samples, T, A, xi, dt, key):
    mu_t = lambda _t: state_q.apply_fn(state_q.params, _t)[0]
    sigma_t = lambda _t: state_q.apply_fn(state_q.params, _t)[1]

    def dmudt(_t):
        _dmudt = jax.jacrev(lambda _t: mu_t(_t).sum(0).T, argnums=0)
        return _dmudt(_t).squeeze(axis=-1).T

    def dsigmadt(_t):
        _dsigmadt = jax.jacrev(lambda _t: sigma_t(_t).sum(0).T)
        return _dsigmadt(_t).squeeze(axis=-1).T

    @jax.jit
    def u_t(_t, _x):
        _mu_t = mu_t(_t)
        _sigma_t = sigma_t(_t)
        _x = _x[:, None, :]

        log_q_i = jax.scipy.stats.norm.logpdf(_x, _mu_t, _sigma_t).sum(-1)
        relative_mixture_weights = jax.nn.softmax(w_logits + log_q_i)[:, :, None]

        log_q_t = -(relative_mixture_weights / (_sigma_t ** 2) * (_x - _mu_t)).sum(axis=1)
        _u_t = (relative_mixture_weights * (1 / _sigma_t * dsigmadt(_t) * (_x - _mu_t) + dmudt(_t))).sum(axis=1)

        return _u_t + 0.5 * (xi ** 2) * log_q_t

    N = int(T / dt)

    key, loc_key = jax.random.split(key)

    x_t = jnp.ones((num_samples, N + 1, 2)) * A
    eps = jax.random.normal(key, shape=(num_samples, A.shape[-1]))
    x_t = x_t.at[:, 0, :].set(x_t[:, 0, :] + sigma_t(jnp.zeros((num_samples, 1)))[:, 0, :] * eps)

    t = jnp.zeros((num_samples, 1))
    for i in trange(N):
        key, loc_key = jax.random.split(key)
        eps = jax.random.normal(key, shape=(num_samples, A.shape[-1]))

        dx = dt * u_t(t, x_t[:, i, :]) + jnp.sqrt(dt) * xi * eps
        x_t = x_t.at[:, i + 1, :].set(x_t[:, i, :] + dx)
        t += dt

    return x_t


if __name__ == '__main__':
    savedir = './out/var_doobs/mixture'
    os.makedirs(savedir, exist_ok=True)

    A = jnp.array([[-0.5, 0]])
    B = jnp.array([[0.5, 0]])
    dt = 5e-4
    T = 1.0
    xi = 0.1
    epochs = 20_000

    q_single, w_logits_single = create_mlp_q(A, B, T, 1)
    q_mixture, w_logits_mixture = create_mlp_q(A, B, T, 2)

    state_q_single, loss_plot_single = train(q_single, w_logits_single, epochs=epochs)
    state_q_mixture, loss_plot_mixture = train(q_mixture, w_logits_mixture, epochs=epochs)

    plt.plot(loss_plot_single, label='single')
    plt.plot(loss_plot_mixture, label='mixture')
    plt.legend()
    plt.show()

    samples_qt_single = draw_samples_qt(state_q_single, w_logits_single, num_samples=1000, T=T, A=A,
                                        key=jax.random.PRNGKey(0))
    samples_qt_mixture = draw_samples_qt(state_q_mixture, w_logits_mixture, num_samples=1000, T=T, A=A,
                                         key=jax.random.PRNGKey(0))

    plot_energy_surface()
    plt.scatter(samples_qt_single[:, 0], samples_qt_single[:, 1], label='single')
    plt.scatter(samples_qt_mixture[:, 0], samples_qt_mixture[:, 1], label='mixture')
    plt.legend()
    plt.show()

    samples_single = sample_stochastically(state_q_single, w_logits_single, num_samples=1000, T=T, A=A, xi=xi, dt=dt,
                                           key=jax.random.PRNGKey(0))
    samples_mixture = sample_stochastically(state_q_mixture, w_logits_mixture, num_samples=1000, T=T, A=A, xi=xi, dt=dt,
                                            key=jax.random.PRNGKey(0))

    plot_energy_surface(trajectories=samples_single)
    plt.savefig(f'{savedir}/toy-gaussian-single.pdf', bbox_inches='tight')
    plt.show()

    plot_energy_surface(trajectories=samples_mixture)
    plt.savefig(f'{savedir}/toy-gaussian-mixture.pdf', bbox_inches='tight')
    plt.show()

    t = T * jnp.linspace(0, 1, 10 * int(T / dt)).reshape((-1, 1))
    _mu_t_single, _sigma_t_single = state_q_single.apply_fn(state_q_single.params, t)
    _mu_t_mixture, _sigma_t_mixture = state_q_mixture.apply_fn(state_q_mixture.params, t)

    vmin = min(_sigma_t_single.min(), _sigma_t_mixture.min())
    vmax = max(_sigma_t_single.max(), _sigma_t_mixture.max())

    plot_energy_surface()
    plt.scatter(_mu_t_single[:, :, 0], _mu_t_single[:, :, 1], c=_sigma_t_single, vmin=vmin, vmax=vmax, rasterized=True)
    plt.colorbar(label=r'$\sigma$')
    plt.savefig(f'{savedir}/toy-gaussian-single-mu.pdf', bbox_inches='tight')
    plt.show()

    plot_energy_surface()
    plt.scatter(_mu_t_mixture[:, :, 0], _mu_t_mixture[:, :, 1], c=_sigma_t_mixture, vmin=vmin, vmax=vmax, rasterized=True)
    plt.colorbar(label=r'$\sigma$')
    plt.savefig(f'{savedir}/toy-gaussian-mixture-mu.pdf', bbox_inches='tight')
    plt.show()
