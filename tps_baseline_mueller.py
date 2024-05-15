import json

import jax
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
from tps import first_order as tps1
import numpy as np

minima_points = jnp.array([[-0.55828035, 1.44169], [-0.05004308, 0.46666032], [0.62361133, 0.02804632]])
A, B = minima_points[None, 0], minima_points[None, 2]


@jax.jit
def U(xs, beta=1.0):
    if xs.ndim == 1:
        xs = xs[None, :]
    x, y = xs[:, 0], xs[:, 1]
    e1 = -200 * jnp.exp(-(x - 1) ** 2 - 10 * y ** 2)
    e2 = -100 * jnp.exp(-x ** 2 - 10 * (y - 0.5) ** 2)
    e3 = -170 * jnp.exp(-6.5 * (0.5 + x) ** 2 + 11 * (x + 0.5) * (y - 1.5) - 6.5 * (y - 1.5) ** 2)
    e4 = 15.0 * jnp.exp(0.7 * (1 + x) ** 2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1) ** 2)
    return beta * (e1 + e2 + e3 + e4)


dUdx_fn = jax.jit(jax.grad(lambda _x: U(_x).sum()))


def interpolate(points, steps):
    def interpolate_two_points(start, stop, steps):
        t = jnp.linspace(0, 1, steps + 1).reshape(steps + 1, 1)
        interpolated_tensors = jnp.array(start) * (1 - t) + jnp.array(stop) * t
        return interpolated_tensors

    step_size = steps // (len(points) - 1)
    remaining = steps % (len(points) - 1)

    interpolation = []
    for i in range(len(points) - 1):
        cur_step_size = step_size + (1 if i < remaining else 0)
        current = interpolate_two_points(points[i], points[i + 1], cur_step_size)
        interpolation.extend(current if i == 0 else current[1:])

    return interpolation


def plot_energy_surface(points=[], trajectories=[], bins=150, alpha=0.7):
    xlim, ylim = jnp.array((-1.5, 0.9)), jnp.array((-0.5, 1.7))

    x, y = jnp.linspace(xlim[0], xlim[1], bins), jnp.linspace(ylim[0], ylim[1], bins)
    x, y = jnp.meshgrid(x, y, indexing='ij')
    z = U(jnp.stack([x, y], -1).reshape(-1, 2)).reshape([bins, bins])

    # black and white contour plot
    plt.contour(x, y, z, levels=30, cmap='gray')

    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    if len(trajectories) > 0:
        from openpathsampling.analysis import PathHistogram
        from openpathsampling.numerics import HistogramPlotter2D

        hist = PathHistogram(
            left_bin_edges=(xlim[0], ylim[0]),
            bin_widths=(jnp.diff(xlim)[0] / bins, jnp.diff(ylim)[0] / bins),
            interpolate=True, per_traj=True
        )

        [hist.add_trajectory(t) for t in trajectories]

        plotter = HistogramPlotter2D(hist, xlim=xlim, ylim=ylim)
        df = hist().df_2d(x_range=plotter.xrange_, y_range=plotter.yrange_)
        plt.pcolormesh(
            jnp.linspace(xlim[0], xlim[1], df.shape[0]),
            jnp.linspace(ylim[0], ylim[1], df.shape[1]),
            df.values.T.astype(dtype=float),
            vmin=0, vmax=3, cmap='Blues',
            rasterized=True
        )

        plt.colorbar()

    for p in points:
        plt.scatter(p[0], p[1], marker='*')

    for name, pos in zip(['A', 'B', 'C'], minima_points):
        c = plt.Circle(pos, radius=0.1, edgecolor='gray', alpha=alpha, facecolor='white', ls='--', lw=0.7)
        plt.gca().add_patch(c)
        plt.gca().annotate(name, xy=pos, ha="center", va="center")


if __name__ == '__main__':
    savedir = f"out/baselines/mueller"
    os.makedirs(savedir, exist_ok=True)

    num_paths = 1000
    xi = 5
    dt = 1e-4
    T = 275e-4
    N = int(T / dt)
    initial_trajectory = [t.reshape(1, 2) for t in interpolate(minima_points, N)]

    @jax.jit
    def step(_x, _key):
        """Perform one step of forward euler"""
        return _x - dt * dUdx_fn(_x) + jnp.sqrt(dt) * xi * jax.random.normal(_key, _x.shape)


    system = tps1.FirstOrderSystem(
        jax.jit(lambda s: jnp.linalg.norm(s - A) <= 0.1),
        jax.jit(lambda s: jnp.linalg.norm(s - B) <= 0.1),
        step
    )

    for method, name in [
        (tps1.one_way_shooting, 'one-way-shooting'),
        (tps1.two_way_shooting, 'two-way-shooting'),
    ]:
        if os.path.exists(f'{savedir}/paths-{name}.npy') and os.path.exists(f'{savedir}/stats-{name}.json'):
            print(f"Skipping {name} because the results are already present")

            paths = np.load(f'{savedir}/paths-{name}.npy', allow_pickle=True)
            paths = [jnp.array(p.astype(np.float32)) for p in paths]
            with open(f'{savedir}/stats-{name}.json', 'r') as fp:
                statistics = json.load(fp)
        else:
            paths, statistics = tps1.mcmc_shooting(system, method, initial_trajectory, num_paths,
                                                   jax.random.PRNGKey(1), warmup=0, fixed_length=N)

            paths = [jnp.array(p) for p in paths]

            np.save(f'{savedir}/paths-{name}.npy', np.array(paths, dtype=object), allow_pickle=True)
            with open(f'{savedir}/stats-{name}.json', 'w') as fp:
                json.dump(statistics, fp)

        plot_energy_surface(trajectories=paths)
        plt.savefig(f'{savedir}/mueller-{name}.pdf', bbox_inches='tight')
        plt.show()
