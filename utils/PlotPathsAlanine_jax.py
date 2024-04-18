import matplotlib.pyplot as plt
import numpy as np
import torch


class AlaninePotential():
    def __init__(self):
        super().__init__()
        self.open_file()

    def open_file(self):
        file = "./utils/final_frame.dat"

        with open(file) as f:
            lines = f.readlines()

        dims = [90, 90]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:
            # if line == '  \n':
            #     psi = psi + 1
            #     phi = 0
            #     continue
            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor([x, y])
            self.data[i // 90, i % 90] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z

    def drift(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp[:, :2].double(), loc.double(), p=2)
        index = distances.argsort(dim=1)[:, :3]

        x = index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        dims = torch.stack([x, y], 2)

        min = dims.argmin(dim=1)
        max = dims.argmax(dim=1)

        min_x = min[:, 0]
        min_y = min[:, 1]
        max_x = max[:, 0]
        max_y = max[:, 1]

        min_x_dim = dims[range(dims.shape[0]), min_x, :]
        min_y_dim = dims[range(dims.shape[0]), min_y, :]
        max_x_dim = dims[range(dims.shape[0]), max_x, :]
        max_y_dim = dims[range(dims.shape[0]), max_y, :]

        min_x_val = self.data[min_x_dim[:, 0], min_x_dim[:, 1]]
        min_y_val = self.data[min_y_dim[:, 0], min_y_dim[:, 1]]
        max_x_val = self.data[max_x_dim[:, 0], max_x_dim[:, 1]]
        max_y_val = self.data[max_y_dim[:, 0], max_y_dim[:, 1]]

        grad = -1 * torch.stack([max_y_val - min_y_val, max_x_val - min_x_val], dim=1)

        return grad


def new_dihedral(p):
    """http://stackoverflow.com/q/20305272/1128289"""
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array(
        [v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.arctan2(y, x)


def PlotPathsAlanine(paths, target, filename):
    colors = []

    plt.clf()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]

    potential = AlaninePotential()
    xs = np.arange(-np.pi, np.pi + .1, .1)
    ys = np.arange(-np.pi, np.pi + .1, .1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor([x, y]).view(2, -1).T

    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])

    plt.contourf(xs, ys, z, levels=100, zorder=0)

    paths_ = paths.reshape(paths.shape[0], -1, 22, 3)
    # print (paths_.shape, target.shape)

    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / paths_.shape[0]) for i in range(paths_.shape[0])])

    psis_start = []
    phis_start = []

    for i in range(0, paths_.shape[0]):

        psis_start.append((new_dihedral(paths_[i, 0, angle_1, :])))
        phis_start.append(new_dihedral(paths_[i, 0, angle_2, :]))

        psi = []
        phi = []
        for j in range(0, paths_.shape[1]):
            psi.append((new_dihedral(paths_[i, j, angle_1, :])))
            phi.append((new_dihedral(paths_[i, j, angle_2, :])))
        tmp = ax.plot(phi, psi, marker='o', color='b', markersize=2, alpha=1.)
        # linestyle='None', 
        colors.append(tmp[0].get_color())
    ax.scatter(phis_start, psis_start, edgecolors='black', c='w', zorder=100, s=100, marker='*')

    # target = target.squeeze().detach().cpu().numpy()  # .view(-1, 3)
    psis_target = []
    phis_target = []
    psis_target.append((new_dihedral(target[angle_1, :])))
    phis_target.append(new_dihedral(target[angle_2, :]))
    ax.scatter(phis_target, psis_target, edgecolors='w', c='w', zorder=100, s=10)

    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.savefig(f"{filename}_traj.png")
    plt.show()
    plt.clf()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    plt.contourf(xs, ys, z, levels=100, zorder=0)
    tmp = ax.plot(phi, psi, marker='o', color='b', linestyle='None', markersize=2, alpha=1.)
    plt.savefig(f"{filename}_point.png")
    plt.clf()

    return colors