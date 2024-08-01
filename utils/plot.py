from tqdm import tqdm
from training.qsetup import QSetup
from typing import Optional, Tuple, Callable
from jax.typing import ArrayLike
import jax.numpy as jnp
import matplotlib.pyplot as plt
from systems import System
from tps.plot import PeriodicPathHistogram
from matplotlib.animation import FuncAnimation, PillowWriter
import jax

from flax.training.train_state import TrainState


def log_scale(log_plot: bool, x: bool, y: bool):
    if log_plot:
        if x:
            plt.gca().set_xscale('log')
        if y:
            plt.gca().set_yscale('log')


def show_or_save_fig(save_dir: Optional[str], name: str, extension: str):
    if save_dir is not None:
        plt.savefig(f'{save_dir}/{name}.{extension}', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_energy(system: System, trajectories: [ArrayLike], log_plot: bool):
    plt.title('Energy of the trajectory')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    energies = jnp.array([system.U(t) for t in trajectories])

    print(f"Min energy: {jnp.min(energies, axis=-1)}")
    print(f"Max energy: {jnp.max(energies, axis=-1)}")

    if log_plot:
        min_energy = jnp.min(energies)
        if min_energy < 0:
            energies -= min_energy
            plt.ylabel('Energy (shifted)')

    log_scale(log_plot, x=False, y=True)
    for e in energies:
        plt.plot(e)


def toy_plot_energy_surface(U, xlim: ArrayLike, ylim: ArrayLike,
                            bins: int = 150, levels: int = 30, *args, **kwargs):
    x, y = jnp.linspace(xlim[0], xlim[1], bins), jnp.linspace(ylim[0], ylim[1], bins)
    x, y = jnp.meshgrid(x, y, indexing='ij')
    z = U(jnp.stack([x, y], -1).reshape(-1, 2)).reshape([bins, bins])

    # black and white contour plot
    plt.contour(x, y, z, levels=levels, colors='black')

    plot_2d(xlim=xlim, ylim=ylim, bins=bins, *args, **kwargs)


def plot_cv(cv: Callable[[ArrayLike], ArrayLike], points: Optional[ArrayLike] = None,
            trajectories: Optional[ArrayLike] = None, *args, **kwargs):
    if trajectories is None:
        cv_trajectories = None
    else:
        cv_trajectories = cv(trajectories.reshape(-1, trajectories.shape[-1])).reshape(trajectories.shape[0],
                                                                                       trajectories.shape[1], 2)
    plot_2d(points=None if points is None else cv(points), trajectories=cv_trajectories, *args, **kwargs)


def plot_2d(
        states: [Tuple[str, ArrayLike]],
        xlim: ArrayLike, ylim: ArrayLike, bins: int,
        points: Optional[ArrayLike] = None,
        trajectories: Optional[ArrayLike] = None,
        periodic: bool = False,
        title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
        xticks: Optional[ArrayLike] = None, yticks: Optional[ArrayLike] = None,
        xticklabels: Optional[list[str]] = None, yticklabels: Optional[list[str]] = None,
        square: bool = False,
        alpha: float = 0.7, radius: float = 0.1
):
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    if trajectories is not None and len(trajectories) > 0:
        if periodic:
            assert jnp.allclose(xlim[1] - xlim[0], ylim[1] - ylim[0]), "Periodic plot requires square plot"
            _plot_periodic_trajectories(trajectories, bins, scale=jnp.pi)
        else:
            _plot_trajectories(trajectories, bins, xlim, ylim)

    if xticks is None:
        xticks = []
    if yticks is None:
        yticks = []

    plt.xticks(xticks)
    plt.yticks(yticks)

    if xticklabels is not None:
        plt.gca().set_xticklabels(xticklabels)
    if yticklabels is not None:
        plt.gca().set_yticklabels(yticklabels)

    if points is not None:
        for p in points:
            plt.scatter(p[0], p[1], marker='*')

    for name, pos in states:
        pos = pos.reshape(2, )
        c = plt.Circle(pos, radius=radius, edgecolor='gray', alpha=alpha, facecolor='white', ls='--', lw=0.7, zorder=10)
        plt.gca().add_patch(c)
        plt.gca().annotate(name, xy=pos, ha="center", va="center", fontsize=14, zorder=11)

    if square:
        plt.gca().set_aspect('equal', adjustable='box')


def _plot_periodic_trajectories(trajectories: ArrayLike, bins: int, scale: float = jnp.pi):
    path_hist = PeriodicPathHistogram(bins, scale=scale)
    for path in tqdm(trajectories, desc='Adding paths to histogram', total=len(trajectories)):
        path_hist.add_path(jnp.array(path))

    path_hist.plot(cmin=0.01)


def _plot_trajectories(trajectories: ArrayLike, bins: int, xlim: ArrayLike, ylim: ArrayLike):
    from openpathsampling.analysis import PathHistogram
    from openpathsampling.numerics import HistogramPlotter2D
    path_hist = PathHistogram(
        left_bin_edges=(xlim[0], ylim[0]),
        bin_widths=(jnp.diff(xlim)[0] / bins, jnp.diff(ylim)[0] / bins),
        interpolate=True, per_traj=True
    )

    for path in tqdm(trajectories, desc='Adding paths to histogram', total=len(trajectories)):
        path_hist.add_trajectory(path)

    plotter = HistogramPlotter2D(path_hist, xlim=xlim, ylim=ylim)
    df = path_hist().df_2d(x_range=plotter.xrange_, y_range=plotter.yrange_)
    plt.pcolormesh(
        jnp.linspace(xlim[0], xlim[1], df.shape[0]),
        jnp.linspace(ylim[0], ylim[1], df.shape[1]),
        df.values.T.astype(dtype=float),
        vmin=0, vmax=3, cmap='Blues',
        rasterized=True
    )


def plot_u_t(system: System, setup: QSetup, state_q: TrainState, T: float, save_dir: str, name: str, frames: int = 100,
             fps: int = 10):
    t = T * jnp.linspace(0, 1, frames, dtype=jnp.float32).reshape((-1, 1))
    mu_t, sigma_t, _ = state_q.apply_fn(state_q.params, t)

    _u_t_func = jax.jit(lambda _t, _points: setup.u_t(state_q, _t * jnp.ones((len(_points), 1)), _points, True))

    def get_lim():
        system.plot()
        x_lim, y_lim = plt.xlim(), plt.ylim()
        plt.clf()
        return x_lim, y_lim

    x_lim, y_lim = get_lim()
    x, y = jnp.meshgrid(jnp.linspace(x_lim[0], x_lim[1], 10), jnp.linspace(y_lim[0], y_lim[1], 10))
    points = jnp.vstack([x.ravel(), y.ravel()], dtype=jnp.float32).T

    x_all, y_all = [], []
    u_all, v_all = [], []
    for t in jnp.linspace(0, T, frames):
        uv = _u_t_func(t, points)
        u, v = uv[:, 0], uv[:, 1]
        u, v = u.reshape(x.shape), v.reshape(y.shape)
        x_all.append(x)
        y_all.append(y)
        u_all.append(u)
        v_all.append(v)

    def animate(i):
        t = jnp.linspace(0, T, frames)[i]
        plt.clf()
        system.plot(title='t / T = {:.2f}'.format(i / frames))

        for j in range(mu_t.shape[1]):
            color = jnp.zeros(frames)
            color = color.at[i].add(1)
            mu_x, mu_y = mu_t[:, j, 0], mu_t[:, j, 1]
            plt.scatter(mu_x[color == 0], mu_y[color == 0], c='blue')
            plt.scatter(mu_x[color == 1], mu_y[color == 1], c='red')

            uv_testpoint = _u_t_func(t, jnp.array([mu_x[color == 1], mu_y[color == 1]]).T)
            plt.quiver(mu_x[color == 1], mu_y[color == 1], uv_testpoint[:, 0], uv_testpoint[:, 1], color='red')

        return plt.quiver(x_all[i], y_all[i], u_all[i], v_all[i]),

    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=frames)
    ani.save(f"{save_dir}/{name}.gif", dpi=300, writer=PillowWriter(fps=fps))
    plt.clf()
