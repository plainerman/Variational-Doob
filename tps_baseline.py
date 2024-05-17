import os
from functools import partial
import traceback
import jax
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from tqdm import trange, tqdm
import json

# install openmm (from conda)
import openmm.app as app
import openmm.unit as unit
# install dmff (from source)
from dmff import Hamiltonian, NeighborList
# install mdtraj
import mdtraj as md

from eval.path_metrics import plot_path_energy
from tps import first_order as tps1
from tps import second_order as tps2
from tps.plot import PeriodicPathHistogram
# helper function for plotting in 2D
from matplotlib import colors

from utils.angles import phi_psi_from_mdtraj
from utils.animation import save_trajectory, to_md_traj
from utils.rmsd import kabsch_align, kabsch_rmsd

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--mechanism', type=str, choices=['one-way-shooting', 'two-way-shooting'], required=True)
parser.add_argument('--states', type=str, default='phi-psi', choices=['phi-psi', 'rmsd'])
parser.add_argument('--fixed_length', type=int, default=0)
parser.add_argument('--num_paths', type=int, required=True)
parser.add_argument('--num_steps', type=int, default=10,
                    help='The number of MD steps taken at once. More takes longer to compile but runs faster in the end.')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--override', action='store_true')


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


def ramachandran(samples, bins=100, path=None, paths=None, states=None, alpha=1.0):
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

    for state in (states if states is not None else []):
        c = plt.Circle(state['center'], radius=state['radius'], edgecolor='gray', facecolor='white', ls='--', lw=0.7,
                       alpha=alpha)
        plt.gca().add_patch(c)
        plt.gca().annotate(state['name'], xy=state['center'], ha="center", va="center")


dt_as_unit = unit.Quantity(value=1, unit=unit.femtosecond)
dt_in_ps = dt_as_unit.value_in_unit(unit.picosecond)
dt = dt_as_unit.value_in_unit(unit.second)

gamma_as_unit = 1.0 / unit.picosecond
# actually gamma is 1/s, but we are working without units and just need the correct scaling
# TODO: try to get rid of this duplicate definition
gamma = 1.0 * unit.picosecond
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


def step_n(step, _x, _v, n, _key):
    all_x, all_v = jnp.zeros((n, *_x.shape)), jnp.zeros((n, *_v.shape))
    for i in range(n):
        _key, _iter_key = jax.random.split(_key)
        _x, _v = step(_x, _v, _iter_key)
        all_x = all_x.at[i].set(_x)
        all_v = all_v.at[i].set(_v)

    return all_x, all_v


if __name__ == '__main__':
    args = parser.parse_args()

    init_pdb = app.PDBFile("./files/AD_A.pdb")
    target_pdb = app.PDBFile("./files/AD_B.pdb")
    mdtraj_topology = md.Topology.from_openmm(init_pdb.topology)
    phis_psis = phi_psi_from_mdtraj(mdtraj_topology)

    savedir = f"out/baselines/alanine-{args.mechanism}"
    if args.fixed_length > 0:
        savedir += f'-{args.fixed_length}steps'
    if args.states == 'rmsd':
        savedir += '-rmsd'

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


    @jax.jit
    def U(_x):
        """
        Calling U by U(x, box, pairs, ff.paramset.parameters), x is [22, 3] and output the energy, if it is batched, use vmap
        """
        _U = potentials.getPotentialFunc()

        return _U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters)


    @jax.jit
    @jax.vmap
    def dUdx_fn(_x):
        return jax.grad(lambda _x: U(_x).sum())(_x) / mass / gamma_in_ps


    @jax.jit
    def step(_x, _key):
        """Perform one step of forward euler"""
        return _x - dt * dUdx_fn(_x) + jnp.sqrt(dt) * xi * jax.random.normal(_key, _x.shape)


    @jax.jit
    def step_langevin_forward(_x, _v, _key):
        """Perform one step of forward langevin as implemented in openmm"""
        alpha = jnp.exp(-gamma_in_ps * dt_in_ps)
        f_scale = (1 - alpha) / gamma_in_ps
        new_v_det = alpha * _v + f_scale * -dUdx_fn(_x)
        new_v = new_v_det + jnp.sqrt(kbT * (1 - alpha ** 2) / mass) * jax.random.normal(_key, _x.shape)

        return _x + dt_in_ps * new_v, new_v


    @jax.jit
    def step_langevin_log_prob(_x, _v, _new_x, _new_v):
        alpha = jnp.exp(-gamma_in_ps * dt_in_ps)
        f_scale = (1 - alpha) / gamma_in_ps
        new_v_det = alpha * _v + f_scale * -dUdx_fn(_x)
        new_v_rand = new_v_det - _new_v

        return jax.scipy.stats.norm.logpdf(new_v_rand, 0, jnp.sqrt(kbT * (1 - alpha ** 2) / mass)).sum()


    def langevin_log_path_likelihood(path_and_velocities):
        path, velocities = path_and_velocities

        log_prob = (-U(path[0]) / kbT).sum()
        log_prob += jax.scipy.stats.norm.logpdf(velocities[0], 0, jnp.sqrt(kbT / mass)).sum()

        for i in range(1, len(path)):
            log_prob += step_langevin_log_prob(path[i - 1], velocities[i - 1], path[i], velocities[i])

        return log_prob


    @jax.jit
    def step_langevin_backward(_x, _v, _key):
        """Perform one step of backward langevin"""
        alpha = jnp.exp(-gamma_in_ps * dt_in_ps)
        f_scale = (1 - alpha) / gamma_in_ps
        prev_x = _x - dt_in_ps * _v
        prev_v = 1 / alpha * (_v + f_scale * dUdx_fn(prev_x) - jnp.sqrt(
            kbT * (1 - alpha ** 2) / mass) * jax.random.normal(_key, _x.shape))

        return prev_x, prev_v


    # Choose a system, either phi psi, or rmsd
    # system = tps1.System(
    #     jax.jit(jax.vmap(lambda s: kabsch_rmsd(A.reshape(22, 3), s.reshape(22, 3)) < 0.1)),
    #     jax.jit(jax.vmap(lambda s: kabsch_rmsd(B.reshape(22, 3), s.reshape(22, 3)) < 0.1)),
    #     step
    # )

    radius = 20 / deg

    # system = tps1.FirstOrderSystem(
    #     lambda s: is_within(phis_psis(s).reshape(-1, 2), phis_psis(A), radius),
    #     lambda s: is_within(phis_psis(s).reshape(-1, 2), phis_psis(B), radius),
    #     step
    # )

    if args.states == 'rmsd':
        state_A = jax.jit(jax.vmap(lambda s: kabsch_rmsd(A.reshape(22, 3), s.reshape(22, 3)) <= 7.5e-2))
        state_B = jax.jit(jax.vmap(lambda s: kabsch_rmsd(B.reshape(22, 3), s.reshape(22, 3)) <= 7.5e-2))
    elif args.states == 'phi-psi':
        state_A = jax.jit(lambda s: is_within(phis_psis(s).reshape(-1, 2), phis_psis(A), radius))
        state_B = jax.jit(lambda s: is_within(phis_psis(s).reshape(-1, 2), phis_psis(B), radius))
    else:
        raise ValueError(f"Unknown states {args.states}")

    system = tps2.SecondOrderSystem(
        state_A, state_B,
        jax.jit(lambda _x, _v, _key: step_n(step_langevin_forward, _x, _v, args.num_steps, _key)),
        jax.jit(lambda _x, _v, _key: step_n(step_langevin_backward, _x, _v, args.num_steps, _key)),
        jax.jit(lambda key: jnp.sqrt(kbT / mass) * jax.random.normal(key, (1, 66)))
    )

    initial_trajectory = md.load('./files/AD_A_B_500K_initial_trajectory.pdb').xyz.reshape(-1, 1, 66)
    initial_trajectory = [p for p in initial_trajectory]
    save_trajectory(mdtraj_topology, jnp.array(initial_trajectory), f'{savedir}/initial_trajectory.pdb')

    if args.resume:
        paths = [[x for x in p.astype(np.float32)] for p in np.load(f'{savedir}/paths.npy', allow_pickle=True)]
        velocities = [[v for v in p.astype(np.float32)] for p in
                      np.load(f'{savedir}/velocities.npy', allow_pickle=True)]
        with open(f'{savedir}/stats.json', 'r') as fp:
            statistics = json.load(fp)

        stored = {
            'trajectories': [initial_trajectory] + paths,
            'velocities': velocities,
            'statistics': statistics
        }
    else:
        if os.path.exists(f'{savedir}/paths.npy') and not args.override:
            print(f"The target directory is not empy."
                  f"Please use --override to overwrite the existing data or --resume to continue.")
            exit(1)

        stored = None

    if args.mechanism == 'one-way-shooting':
        mechanism = tps2.one_way_shooting
    elif args.mechanism == 'two-way-shooting':
        mechanism = tps2.two_way_shooting
    else:
        raise ValueError(f"Unknown mechanism {args.mechanism}")

    try:
        paths, velocities, statistics = tps2.mcmc_shooting(system, mechanism, initial_trajectory,
                                                           args.num_paths, dt_in_ps, jax.random.PRNGKey(1), warmup=0,
                                                           fixed_length=args.fixed_length,
                                                           stored=stored)
        # paths = tps2.unguided_md(system, B, 1, key)
        paths = [jnp.array(p) for p in paths]
        velocities = [jnp.array(p) for p in velocities]
        # store paths
        np.save(f'{savedir}/paths.npy', np.array(paths, dtype=object), allow_pickle=True)
        np.save(f'{savedir}/velocities.npy', np.array(velocities, dtype=object), allow_pickle=True)
        # save statistics, which is a dictionary
        with open(f'{savedir}/stats.json', 'w') as fp:
            json.dump(statistics, fp)
    except Exception as e:
        print(traceback.format_exc())
        breakpoint()

    print(statistics)

    if args.fixed_length == 0:
        print([len(p) for p in paths])
        plt.hist([len(p) for p in paths], bins=jnp.sqrt(len(paths)).astype(int).item())
        plt.savefig(f'{savedir}/lengths.png', bbox_inches='tight')
        plt.show()

    path_hist = PeriodicPathHistogram()
    for i, path in tqdm(enumerate(paths), desc='Adding paths to histogram', total=len(paths)):
        path_hist.add_path(np.array(phis_psis(path)))

    plt.title(f"{human_format(len(paths))} paths @ {temp} K, dt = {human_format(dt)}s")
    path_hist.plot(cmin=0.01)
    ramachandran(None, states=[
        {'name': 'A', 'center': phis_psis(A).squeeze(), 'radius': radius},
        {'name': 'B', 'center': phis_psis(B).squeeze(), 'radius': radius},
    ], alpha=0.7)
    plt.savefig(f'{savedir}/paths.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    plot_path_energy(paths, jax.vmap(U))
    plt.ylabel('Maximum energy')
    plt.savefig(f'{savedir}/max_energy.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    plot_path_energy(paths, jax.vmap(U), reduce=jnp.median)
    plt.ylabel('Median energy')
    plt.savefig(f'{savedir}/median_energy.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    plot_path_energy(list(zip(paths, velocities)), langevin_log_path_likelihood, reduce=lambda x: x, already_ln=True)
    plt.ylabel('Path Likelihood')
    plt.savefig(f'{savedir}/path_density.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    for i, path in tqdm(enumerate(paths), desc='Saving trajectories', total=len(paths)):
        save_trajectory(mdtraj_topology, jnp.array([kabsch_align(p.reshape(-1, 3), B.reshape(-1, 3))[0] for p in path]),
                        f'{savedir}/trajectory_{i}.pdb')
