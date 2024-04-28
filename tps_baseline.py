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

from utils.animation import save_trajectory, to_md_traj
from utils.rmsd import kabsch_align, kabsch_rmsd


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
    traj = to_md_traj(mdtraj_topology, position)
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


dt_as_unit = unit.Quantity(value=1, unit=unit.microsecond)
dt_in_ps = dt_as_unit.value_in_unit(unit.picosecond)
dt = dt_as_unit.value_in_unit(unit.second)

gamma_as_unit = 1.0 / unit.second
# actually gamma is 1/s, but we are working without units and just need the correct scaling
# TODO: try to get rid of this duplicate definition
gamma = 1.0 * unit.second
gamma_in_ps = gamma.value_in_unit(unit.picosecond)
gamma = gamma.value_in_unit(unit.second)

temp = 300
kbT = 1.380649 * 6.02214076 * 1e-3 * temp


@jax.jit
def is_within(_phis_psis, _center, _radius, _period=2 * jnp.pi):
    delta = jnp.abs(_center - _phis_psis)
    delta = jnp.where(delta > _period / 2, delta - _period, delta)

    return jnp.hypot(delta[:, 0], delta[:, 1]) < _radius


deg = 180.0 / jnp.pi

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
    # Obtain xi, gamma is by default 1
    xi = jnp.sqrt(2 * kbT / mass / gamma)

    # Initial and target shape [BS, 66]
    A = jnp.array(init_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    B = jnp.array(target_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    A, B = kabsch_align(A, B)
    A, B = A.reshape(1, -1), B.reshape(1, -1)

    # Initialize the potential energy with amber forcefields
    ff = Hamiltonian('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    potentials = ff.createPotential(init_pdb.topology,
                                    nonbondedMethod=app.NoCutoff,
                                    nonbondedCutoff=1.0 * unit.nanometers,
                                    constraints=None,
                                    ewaldErrorTolerance=0.0005)
    # Create a box used when calling
    box = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]])
    nbList = NeighborList(box, 4.0, potentials.meta["cov_map"])
    nbList.allocate(init_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    pairs = nbList.pairs


    def U(_x):
        """
        Calling U by U(x, box, pairs, ff.paramset.parameters), x is [22, 3] and output the energy, if it is batched, use vmap
        """
        _U = potentials.getPotentialFunc()

        return _U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters)


    @jax.jit
    @jax.vmap
    def dUdx_fn(_x):
        return jax.grad(lambda _x: U(_x).sum())(_x) / mass / gamma


    @jax.jit
    def step(_x, _key):
        """Perform one step of forward euler"""
        return _x - dt * dUdx_fn(_x) + jnp.sqrt(dt) * xi * jax.random.normal(_key, _x.shape)


    key = jax.random.PRNGKey(1)
    key, velocity_key = jax.random.split(key)
    steps = 100_000

    trajectory = [A]
    _x = trajectory[-1]

    for i in trange(steps):
        key, iter_key = jax.random.split(key)
        _x = step(_x, iter_key)

        trajectory.append(_x)

    trajectory = jnp.array(trajectory).reshape(-1, 66)

    # save_trajectory(mdtraj_topology, trajectory[-1000:], 'simulation.pdb')

    # we only need to check whether the last frame contains nan, is it propagates
    assert not jnp.isnan(trajectory[-1]).any()
    trajectory_phi_psi = phis_psis(trajectory, mdtraj_topology)

    plt.title(f"{human_format(steps)} steps @ {temp} K, dt = {human_format(dt)}s")
    ramachandran(trajectory_phi_psi)
    plt.show()

    # Choose a system, either phi psi, or rmsd
    # system = tps.System(
    #     jax.jit(jax.vmap(lambda s: kabsch_rmsd(A.reshape(22, 3), s.reshape(22, 3)) < 0.1)),
    #     jax.jit(jax.vmap(lambda s: kabsch_rmsd(B.reshape(22, 3), s.reshape(22, 3)) < 0.1)),
    #     step
    # )

    system = tps.System(
        lambda s: is_within(phis_psis(s, mdtraj_topology).reshape(-1, 2), phis_psis(A, mdtraj_topology), 20 / deg),
        lambda s: is_within(phis_psis(s, mdtraj_topology).reshape(-1, 2), phis_psis(B, mdtraj_topology), 20 / deg),
        step
    )

    filter1 = system.start_state(trajectory)
    filter2 = system.target_state(trajectory)

    plt.title('start')
    ramachandran(trajectory_phi_psi[filter1])
    plt.show()

    plt.title('target')
    ramachandran(trajectory_phi_psi[filter2])
    plt.show()

    initial_trajectory = [t.reshape(1, -1) for t in interpolate([A, B], 100)]
    save_trajectory(mdtraj_topology, jnp.array(initial_trajectory), f'{savedir}/initial_trajectory.pdb')

    paths = tps.mcmc_shooting(system, tps.two_way_shooting, initial_trajectory, 5, key, warmup=2)
    paths = [jnp.array(p) for p in paths]
    # store paths
    np.save(f'{savedir}/paths.npy', np.array(paths, dtype=object), allow_pickle=True)

    print([len(p) for p in paths])
    plt.hist([len(p) for p in paths], bins=jnp.sqrt(len(paths)).astype(int).item())
    plt.show()

    plt.title(f"{human_format(len(paths))} steps @ {temp} K, dt = {human_format(dt)}s")
    ramachandran(jnp.concatenate([phis_psis(p, mdtraj_topology) for p in paths]),
                 path=phis_psis(jnp.array(initial_trajectory), mdtraj_topology))
    plt.show()

    save_trajectory(mdtraj_topology, paths[-1], f'{savedir}/final_trajectory.pdb')
