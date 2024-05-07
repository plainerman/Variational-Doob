import numpy as np
from skimage.draw import line
from tqdm import tqdm
import matplotlib.pyplot as plt


class PeriodicPathHistogram:
    def __init__(self, bins=250, interpolate=True, scale=np.pi):
        self.bins = bins
        self.interpolate = interpolate
        self.scale = scale
        self.hist = np.zeros((bins, bins))

    def add_paths(self, paths: list[np.ndarray], factors: list[float] = None):
        for path, factor in tqdm(zip(paths, factors or [1] * len(paths)), total=len(paths)):
            self.add_path(path, factor=factor)

    def add_path(self, path: np.ndarray, factor: float = 1):
        """
        Adds a path to the histogram. The path is a list of 2D points in the range [-scale, scale]
        """
        rr, cc = np.array([], dtype=int), np.array([], dtype=int)

        if self.interpolate:
            for i in range(len(path) - 1):
                rr_cur, cc_cur = self._add_path_segment_periodic(path[i], path[i + 1])
                rr = np.concatenate([rr, rr_cur])
                cc = np.concatenate([cc, cc_cur])
        else:
            for p in path:
                point = ((p + self.scale) / (2 * self.scale) * (self.bins - 1)).astype(int)
                cc = np.concatenate([cc, [point[0]]])
                rr = np.concatenate([rr, [point[1]]])

        # we only add it once for each path, so that overlapping segments are not counted multiple times
        self.hist[rr, cc] += factor

    def _add_path_segment_periodic(self, start: np.ndarray, stop: np.ndarray):
        start = np.array(start)
        stop = np.array(stop)

        if np.linalg.norm(start - stop) < self.scale:
            return self._determine_path_segments(start, stop)

        possible_offsets = [
            np.array([0, 2 * self.scale]),
            np.array([0, -2 * self.scale]),
            np.array([2 * self.scale, 0]),
            np.array([-2 * self.scale, 0]),
            np.array([2 * self.scale, 2 * self.scale]),
            np.array([-2 * self.scale, 2 * self.scale]),
            np.array([2 * self.scale, -2 * self.scale]),
            np.array([-2 * self.scale, -2 * self.scale]),
        ]

        def add_shortest_segment(point, target):
            distances = np.array([np.linalg.norm((target + offset) - point) for offset in possible_offsets])
            best_offset_idx = np.argmin(distances)

            best_target = target + possible_offsets[best_offset_idx]
            return self._determine_path_segments(point, best_target)

        # just try each possible combination and use the shortest path
        rr1, cc1 = add_shortest_segment(start, stop)
        rr2, cc2 = add_shortest_segment(stop, start)
        return np.concatenate([rr1, rr2]), np.concatenate([cc1, cc2])

    def _determine_path_segments(self, start: np.ndarray, stop: np.ndarray):
        """
        Start and stop are 2D points in the range [-scale, scale].
        This function converts the points into the corresponding bins and then uses a line to connect those points
        """
        start = ((start + self.scale) / (2 * self.scale) * (self.bins - 1)).astype(int)
        stop = ((stop + self.scale) / (2 * self.scale) * (self.bins - 1)).astype(int)

        rr, cc = line(start[1], start[0], stop[1], stop[0])
        rr_mask = (rr >= 0) & (rr < self.bins)
        cc_mask = (cc >= 0) & (cc < self.bins)
        mask = rr_mask & cc_mask
        rr, cc = rr[mask], cc[mask]

        return rr, cc

    def plot(self, density=False, cmin=None, cmax=None, **kwargs):
        H = self.hist.copy()
        if density:
            H /= H.sum() * (2 * self.scale / self.bins) ** 2

        if cmin is not None:
            H[H < cmin] = None
        if cmax is not None:
            H[H > cmax] = None

        x = np.linspace(self.scale, -self.scale, self.bins)
        y = np.linspace(self.scale, -self.scale, self.bins)
        xv, yv = np.meshgrid(x, y)

        plt.pcolormesh(xv, yv, np.flip(H), **kwargs)
        ticks = np.arange(-self.scale, self.scale + self.scale * 0.01, self.scale / 4)

        plt.xlim(-self.scale, self.scale)
        plt.ylim(-self.scale, self.scale)
        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$\psi$")
        plt.xticks(ticks)
        plt.yticks(ticks)
