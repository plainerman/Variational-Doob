from tps_baseline_mueller import U, A, B, plot_energy_surface
from flax import linen as nn
from flax.training import train_state
import optax
import jax
import jax.numpy as jnp
from tqdm import trange
import matplotlib.pyplot as plt
import os
import numpy as np


class MLPq(nn.Module):
    @nn.compact
    def __call__(self, t):
        t = t / T
        h = nn.Dense(128)(t - 0.5)
        h = nn.swish(h)
        h = nn.Dense(128)(h)
        h = nn.swish(h)
        h = nn.Dense(128)(h)
        h = nn.swish(h)
        h = nn.Dense(4)(h)
        mu = (1 - t) * A + t * B + (1 - t) * t * h[:, :2]
        sigma = (1 - t) * 2.5 * 1e-2 + t * 2.5 * 1e-2 + (1 - t) * t * jnp.exp(h[:, 2:])
        return mu, sigma


if __name__ == '__main__':
    savedir = f"out/var_doobs/mueller"
    os.makedirs(savedir, exist_ok=True)

    num_paths = 1000
    xi = 5
    dt = 1e-4
    T = 275e-4
    N = int(T / dt)
    epochs = 2_500

    q = MLPq()

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
        eps = jax.random.normal(key[1], [BS, 2])

        mu_t = lambda _t: state_q.apply_fn(params_q, _t)[0]
        sigma_t = lambda _t: state_q.apply_fn(params_q, _t)[1]

        def dmudt(_t):
            _dmudt = jax.jacrev(lambda _t: mu_t(_t).sum(0))
            return _dmudt(_t).squeeze().T

        def dsigmadt(_t):
            _dsigmadt = jax.jacrev(lambda _t: sigma_t(_t).sum(0))
            return _dsigmadt(_t).squeeze().T

        dUdx_fn = jax.grad(lambda _x: U(_x).sum())

        def v_t(_eps, _t):
            u_t = dmudt(_t) + dsigmadt(_t) * _eps
            _x = mu_t(_t) + sigma_t(_t) * _eps
            out = (u_t + dUdx_fn(_x)) - 0.5 * (xi ** 2) * _eps / sigma_t(t)
            return out

        loss = 0.5 * ((v_t(eps, t) / xi) ** 2).sum(1, keepdims=True)
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
    for i in trange(epochs):
        key, loc_key = jax.random.split(key)
        state_q, loss = train_step(state_q, loc_key)
        loss_plot.append(loss)

    plt.plot(loss_plot)
    plt.show()

    t = T * jnp.linspace(0, 1, BS).reshape((-1, 1))
    key, path_key = jax.random.split(key)
    eps = jax.random.normal(path_key, [BS, 2])
    mu_t, sigma_t = state_q.apply_fn(state_q.params, t)
    samples = mu_t + sigma_t * eps
    plot_energy_surface()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.scatter(A[0, 0], A[0, 1], color='red')
    plt.scatter(B[0, 0], B[0, 1], color='orange')
    plt.show()

    print("Number of potential evaluations", BS * epochs)

    mu_t = lambda _t: state_q.apply_fn(state_q.params, _t)[0]
    sigma_t = lambda _t: state_q.apply_fn(state_q.params, _t)[1]


    def dmudt(_t):
        _dmudt = jax.jacrev(lambda _t: mu_t(_t).sum(0), argnums=0)
        return _dmudt(_t).squeeze().T


    def dsigmadt(_t):
        _dsigmadt = jax.jacrev(lambda _t: sigma_t(_t).sum(0))
        return _dsigmadt(_t).squeeze().T


    u_t = jax.jit(lambda _t, _x: dmudt(_t) + dsigmadt(_t) / sigma_t(_t) * (_x - mu_t(_t)))

    key, loc_key = jax.random.split(key)
    x_t = jnp.ones((BS, N + 1, 2)) * A
    eps = jax.random.normal(key, shape=(BS, 2))
    x_t = x_t.at[:, 0, :].set(x_t[:, 0, :] + sigma_t(jnp.zeros((BS, 1))) * eps)
    t = jnp.zeros((BS, 1))
    for i in trange(N):
        dx = dt * u_t(t, x_t[:, i, :])
        x_t = x_t.at[:, i + 1, :].set(x_t[:, i, :] + dx)
        t += dt

    x_t_det = x_t.copy()

    u_t = jax.jit(
        lambda _t, _x: dmudt(_t) + (dsigmadt(_t) / sigma_t(_t) - 0.5 * (xi / sigma_t(_t)) ** 2) * (_x - mu_t(_t)))

    BS = num_paths
    key, loc_key = jax.random.split(key)
    x_t = jnp.ones((BS, N + 1, 2)) * A
    eps = jax.random.normal(key, shape=(BS, 2))
    x_t = x_t.at[:, 0, :].set(x_t[:, 0, :] + sigma_t(jnp.zeros((BS, 1))) * eps)
    t = jnp.zeros((BS, 1))
    for i in trange(N):
        key, loc_key = jax.random.split(key)
        eps = jax.random.normal(key, shape=(BS, 2))
        dx = dt * u_t(t, x_t[:, i, :]) + jnp.sqrt(dt) * xi * eps
        x_t = x_t.at[:, i + 1, :].set(x_t[:, i, :] + dx)
        t += dt

    x_t_stoch = x_t.copy()

    np.save(f'{savedir}/paths.npy', np.array([jnp.array(p) for p in x_t_stoch], dtype=object), allow_pickle=True)

    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    plot_energy_surface()
    plt.plot(x_t_det[:10, :, 0].T, x_t_det[:10, :, 1].T)
    plt.scatter(A[0, 0], A[0, 1], color='red')
    plt.scatter(B[0, 0], B[0, 1], color='orange')

    plt.subplot(122)
    plot_energy_surface()
    plt.plot(x_t_stoch[:10, :, 0].T, x_t_stoch[:10, :, 1].T)
    plt.scatter(A[0, 0], A[0, 1], color='red')
    plt.scatter(B[0, 0], B[0, 1], color='orange')
    plt.savefig(f'{savedir}/selected_paths_det_vs_stoch.png', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    plot_energy_surface(trajectories=x_t_det)

    plt.subplot(122)
    plot_energy_surface(trajectories=x_t_stoch)
    plt.savefig(f'{savedir}/paths_det_vs_stoch.png', bbox_inches='tight')
    plt.show()

    plot_energy_surface(trajectories=x_t_stoch)
    plt.savefig(f'{savedir}/mueller-variational-doobs.pdf', bbox_inches='tight')
    plt.show()
