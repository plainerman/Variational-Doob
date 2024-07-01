from tqdm import tqdm
from typing import Optional, Tuple, Callable
from jax.typing import ArrayLike
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tps.plot import PeriodicPathHistogram


def log_scale(log_plot: bool, x: bool, y: bool):
    if log_plot:
        if x:
            plt.gca().set_xscale('log')
        if y:
            plt.gca().set_yscale('log')


def show_or_save_fig(save_dir: Optional[str], name: str):
    if save_dir is not None:
        plt.savefig(f'{save_dir}/{name}', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


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
    print("!!! TODO: We should also plot the histogram of the energy function")
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
            _plot_periodic_trajectories(trajectories, bins)
        else:
            _plot_trajectories(trajectories, bins, xlim, ylim)

    plt.xticks([])
    plt.yticks([])

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


def _plot_periodic_trajectories(trajectories: ArrayLike, bins: int):
    path_hist = PeriodicPathHistogram(bins)
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

# def periodic_2d_plot(samples=None, bins=100, path=None, paths=None, states=None, alpha=1.0, title=None):
#     if title is not None:
#         plt.title(title)
#
#     path_hist = PeriodicPathHistogram()
#     for i, path in tqdm(enumerate(paths), desc='Adding paths to histogram', total=len(paths)):
#         path_hist.add_path(jnp.array(phis_psis(path)))
#
#     if samples is not None:
#         plt.hist2d(samples[:, 0], samples[:, 1], bins=bins, norm=colors.LogNorm(), rasterized=True)
#     plt.xlim(-jnp.pi, jnp.pi)
#     plt.ylim(-jnp.pi, jnp.pi)
#
#     # set ticks
#     plt.gca().set_xticks([-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi])
#     plt.gca().set_xticklabels([r'$-\pi$', r'$-\frac {\pi} {2}$', '0', r'$\frac {\pi} {2}$', r'$\pi$'])
#
#     plt.gca().set_yticks([-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi])
#     plt.gca().set_yticklabels([r'$-\pi$', r'$-\frac {\pi} {2}$', '0', r'$\frac {\pi} {2}$', r'$\pi$'])
#
#     plt.xlabel(r'$\phi$')
#     plt.ylabel(r'$\psi$')
#
#     plt.gca().set_aspect('equal', adjustable='box')
#
#     def draw_path(_path, **kwargs):
#         dist = jnp.sqrt(np.sum(jnp.diff(_path, axis=0) ** 2, axis=1))
#         mask = jnp.hstack([dist > jnp.pi, jnp.array([False])])
#         masked_path_x, masked_path_y = np.ma.MaskedArray(_path[:, 0], mask), np.ma.MaskedArray(_path[:, 1], mask)
#         plt.plot(masked_path_x, masked_path_y, **kwargs)
#
#     if path is not None:
#         draw_path(path, color='red')
#
#     if paths is not None:
#         for path in paths:
#             draw_path(path, color='blue')
#
#     for state in (states if states is not None else []):
#         c = plt.Circle(state['center'], radius=state['radius'], edgecolor='gray', facecolor='white', ls='--', lw=0.7,
#                        alpha=alpha)
#         plt.gca().add_patch(c)
#         plt.gca().annotate(state['name'], xy=state['center'], ha="center", va="center")
