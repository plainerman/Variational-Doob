import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_energy_surface(U, states, xlim, ylim, points=[], trajectories=[], bins=150, levels=30, alpha=0.7, radius=0.1):
    x, y = jnp.linspace(xlim[0], xlim[1], bins), jnp.linspace(ylim[0], ylim[1], bins)
    x, y = jnp.meshgrid(x, y, indexing='ij')
    z = U(jnp.stack([x, y], -1).reshape(-1, 2)).reshape([bins, bins])

    # black and white contour plot
    plt.contour(x, y, z, levels=levels, colors='black')

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

    plt.xticks([])
    plt.yticks([])

    for p in points:
        plt.scatter(p[0], p[1], marker='*')

    for name, pos in states:
        pos = pos.reshape(2,)
        c = plt.Circle(pos, radius=radius, edgecolor='gray', alpha=alpha, facecolor='white', ls='--', lw=0.7, zorder=10)
        plt.gca().add_patch(c)
        plt.gca().annotate(name, xy=pos, ha="center", va="center", fontsize=14, zorder=11)
