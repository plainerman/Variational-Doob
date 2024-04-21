import os
import jax
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import trange

# install openmm (from conda)
import openmm.app as app
import openmm.unit as unit
# install dmff (from source)
from dmff import Hamiltonian, NeighborList
# install mdtraj
import mdtraj as md

import tps
# helper function for plotting in 2D
from utils.PlotPathsAlanine_jax import PlotPathsAlanine
from matplotlib import colors


def human_format(num):
    """https://stackoverflow.com/a/45846841/4417954"""
    num = float('{:.3g}'.format(num))
    if num >= 1:
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
    else:
        magnitude = 0
        while abs(num) < 1:
            magnitude += 1
            num *= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'm', 'µ', 'n', 'p', 'f'][magnitude])


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


def phis_psis(position, mdtraj_topology):
    traj = md.Trajectory(position.reshape(-1, mdtraj_topology.n_atoms, 3), mdtraj_topology)
    phi = md.compute_phi(traj)[1].squeeze()
    psi = md.compute_psi(traj)[1].squeeze()
    return jnp.array([phi, psi]).T


def ramachandran(samples, bins=100, path=None, paths=None):
    if samples is not None:
        plt.hist2d(samples[:, 0], samples[:, 1], bins=bins, norm=colors.LogNorm(), rasterized=True)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)

    # set ticks
    plt.gca().set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    plt.gca().set_xticklabels([r'$-\pi$', r'$-\frac {\pi} {2}$', '0', r'$\frac {\pi} {2}$', r'$\pi$'])

    plt.gca().set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    plt.gca().set_yticklabels([r'$-\pi$', r'$-\frac {\pi} {2}$', '0', r'$\frac {\pi} {2}$', r'$\pi$'])

    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\psi$')

    plt.gca().set_aspect('equal', adjustable='box')

    def draw_path(_path, **kwargs):
        dist = jnp.sqrt(np.sum(jnp.diff(_path, axis=0) ** 2, axis=1))
        mask = jnp.hstack([dist > jnp.pi, jnp.array([False])])
        masked_path_x, masked_path_y = np.ma.MaskedArray(_path[:, 0], mask), np.ma.MaskedArray(_path[:, 1], mask)
        plt.plot(masked_path_x, masked_path_y, **kwargs)


    if path is not None:
        draw_path(path, color='red')

    if paths is not None:
        for path in paths:
            draw_path(path, color='blue')


T = 2.0
dt = 1.0 * unit.microsecond
dt = dt.value_in_unit(unit.second)

temp = 298.15
temp = 10000  # TODO: remove this
kbT = 1.380649 * 6.02214076 * 1e-3 * temp

if __name__ == '__main__':
    init_pdb = app.PDBFile("./files/AD_c7eq.pdb")
    target_pdb = app.PDBFile("./files/AD_c7ax.pdb")
    mdtraj_topology = md.Topology.from_openmm(init_pdb.topology)

    savedir = f"baselines/alanine"
    os.makedirs(savedir, exist_ok=True)

    # Construct the mass matrix
    mass = [a.element.mass.value_in_unit(unit.dalton) for a in init_pdb.topology.atoms()]
    new_mass = []
    for mass_ in mass:
        for _ in range(3):
            new_mass.append(mass_)
    mass = jnp.array(new_mass)
    # Obtain sigma, gamma is by default 1
    sigma = jnp.sqrt(2 * kbT / mass)

    # Initial and target shape [BS, 66]
    A = jnp.array(init_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).reshape(1, -1)
    B = jnp.array(target_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).reshape(1, -1)

    # Initialize the potential energy with amber forcefields
    ff = Hamiltonian('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    potentials = ff.createPotential(init_pdb.topology,
                                    nonbondedMethod=app.NoCutoff,
                                    nonbondedCutoff=1.0 * unit.nanometers,
                                    constraints=None,
                                    ewaldErrorTolerance=0.0005)
    # Create a box used when calling
    # Calling U by U(x, box, pairs, ff.paramset.parameters), x is [22, 3] and output the energy, if it is batched, use vmap
    box = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]])
    nbList = NeighborList(box, 4.0, potentials.meta["cov_map"])
    nbList.allocate(init_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    pairs = nbList.pairs


    def U(_x):
        _U = potentials.getPotentialFunc()

        return _U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters)


    #TODO: we can introduce gamma here
    def dUdx_fn(_x):
        return jax.grad(lambda _x: U(_x).sum())(_x) / mass


    dUdx_fn = jax.vmap(dUdx_fn)
    dUdx_fn = jax.jit(dUdx_fn)

    @jax.jit
    def step(_x, _key):
        """Perform one step of forward euler"""
        return _x - dt * dUdx_fn(_x) + jnp.sqrt(dt) * sigma * jax.random.normal(_key, _x.shape)

    key = jax.random.PRNGKey(1)

    trajectory = [A]  # or [B]
    _x = trajectory[-1]
    steps = 20_000
    for i in trange(steps):
        key, iter_key = jax.random.split(key)
        _x = step(_x, iter_key)
        trajectory.append(_x)

    trajectory = jnp.array(trajectory).reshape(-1, 66)
    assert not jnp.isnan(trajectory).any()
    trajectory_phi_psi = phis_psis(trajectory, mdtraj_topology)

    trajs = None
    for i in range(10000, 11000):
        traj = md.load_pdb('./files/AD_c7eq.pdb')
        traj.xyz = trajectory[i].reshape(22, 3)
        if trajs is None:
            trajs = traj
        else:
            trajs = trajs.join(traj)
    trajs.save(f'{savedir}/ALDP_forward_euler.pdb')

    plt.title(f"{human_format(steps)} steps @ {temp} K, dt = {human_format(dt)}s")
    ramachandran(trajectory_phi_psi)
    plt.show()


    # TODO: this is work in progress. Get some baselines with tps
    system = tps.System(
        jax.jit(
            lambda s: jnp.all(jnp.linalg.norm(A.reshape(-1, 22, 3) - s.reshape(-1, 22, 3), axis=2) <= 5e-2, axis=1)),
        jax.jit(
            lambda s: jnp.all(jnp.linalg.norm(B.reshape(-1, 22, 3) - s.reshape(-1, 22, 3), axis=2) <= 5e-2, axis=1)),
        step
    )

    # initial_trajectory = [t.reshape(1, -1) for t in interpolate([A, B], 100)]

    #
    # for i in range(10):
    #     key, iter_key = jax.random.split(key)
    #
    #     # ramachandran(None, path=phis_psis(jnp.vstack(initial_trajectory), mdtraj_topology))
    #     # plt.show()
    #
    #
    #     ok, trajectory = tps.one_way_shooting(system, initial_trajectory, 0, key)
    #     trajectory = jnp.array(trajectory)
    #     trajectory = phis_psis(trajectory, mdtraj_topology)
    #     print('ok?', ok)
    #
    #     ramachandran(None, path=phis_psis(jnp.vstack(initial_trajectory), mdtraj_topology), paths=[trajectory])
    #     plt.show()
    #

    # paths = tps.mcmc_shooting(system, tps.one_way_shooting, initial_trajectory, 10, key, warmup=0)
    # paths = [jnp.array(p) for p in paths]
    #
    # print(paths)
    # ramachandran(None, path=[phis_psis(p, mdtraj_topology) for p in paths][-1])
    # plt.show()
