import os
import traceback
import jax
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from tqdm import tqdm
import json

import openmm.unit as unit
import mdtraj as md

from eval.path_metrics import plot_path_energy, plot_iterative_min_max_energy
from systems import System
from tps import first_order as tps1
from tps import second_order as tps2
from tps.plot import PeriodicPathHistogram
# helper function for plotting in 2D
from matplotlib import colors

from utils.angles import phi_psi_from_mdtraj
from utils.animation import save_trajectory, to_md_traj
from utils.plot import show_or_save_fig, human_format
from utils.rmsd import kabsch_align, kabsch_rmsd

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--mechanism', type=str, choices=['one-way-shooting', 'two-way-shooting'], required=True)
parser.add_argument('--states', type=str, default='phi-psi', choices=['phi-psi', 'rmsd', 'exact'])
parser.add_argument('--fixed_length', type=int, default=0)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--num_paths', type=int, required=True)
parser.add_argument('--num_steps', type=int, default=10,
                    help='The number of MD steps taken at once. More takes longer to compile but runs faster in the end.')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--override', action='store_true')
parser.add_argument('--ensure_connected', action='store_true',
                    help='Ensure that the initial path connects A with B by prepending A and appending B.')


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

    savedir = f"out/baselines/alanine-{args.mechanism}"
    if args.fixed_length > 0:
        savedir += f'-{args.fixed_length}steps'
    if args.states == 'rmsd':
        savedir += '-rmsd'
    elif args.states == 'exact':
        savedir += '-exact'

    os.makedirs(savedir, exist_ok=True)

    system = System.from_pdb("./files/AD_A.pdb", "./files/AD_B.pdb",
                             ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'], 'phi_psi', float('inf'))
    xi = jnp.sqrt(2 * kbT / system.mass / gamma)
    phis_psis = phi_psi_from_mdtraj(system.mdtraj_topology)


    def U_padded(x):
        x = x.reshape(-1, 66)
        orig_length = x.shape[0]
        padded_length = orig_length // 100 * 100 + 100
        x_empty = jnp.zeros((padded_length, 66))
        x = x_empty.at[:x.shape[0], :].set(x.reshape(-1, 66))
        return system.U(x)[:orig_length]


    @jax.jit
    def step(_x, _key):
        """Perform one step of forward euler"""
        return _x - dt * system.dUdx(_x) / system.mass / gamma_in_ps + jnp.sqrt(dt) * xi * jax.random.normal(_key,
                                                                                                             _x.shape)


    @jax.jit
    def step_langevin_forward(_x, _v, _key):
        """Perform one step of forward langevin as implemented in openmm"""
        alpha = jnp.exp(-gamma_in_ps * dt_in_ps)
        f_scale = (1 - alpha) / gamma_in_ps
        new_v_det = alpha * _v + f_scale * -system.dUdx(_x) / system.mass / gamma_in_ps
        new_v = new_v_det + jnp.sqrt(kbT * (1 - alpha ** 2) / system.mass) * jax.random.normal(_key, _x.shape)

        return _x + dt_in_ps * new_v, new_v


    @jax.jit
    def step_langevin_backward(_x, _v, _key):
        """Perform one step of backward langevin"""
        alpha = jnp.exp(-gamma_in_ps * dt_in_ps)
        f_scale = (1 - alpha) / gamma_in_ps
        prev_x = _x - dt_in_ps * _v
        prev_v = 1 / alpha * (_v + f_scale * system.dUdx(prev_x) / system.mass / gamma_in_ps - jnp.sqrt(
            kbT * (1 - alpha ** 2) / system.mass) * jax.random.normal(_key, _x.shape))

        return prev_x, prev_v


    @jax.jit
    def step_langevin_log_prob(_x, _v, _new_x, _new_v):
        alpha = jnp.exp(-gamma_in_ps * dt_in_ps)
        f_scale = (1 - alpha) / gamma_in_ps
        new_v_det = alpha * _v + f_scale * -system.dUdx(_x) / system.mass / gamma_in_ps
        new_v_rand = new_v_det - _new_v

        return jax.scipy.stats.norm.logpdf(new_v_rand, 0, jnp.sqrt(kbT * (1 - alpha ** 2) / system.mass)).sum()


    def langevin_log_path_likelihood(path_and_velocities):
        path, velocities = path_and_velocities
        assert len(path) == len(velocities), \
            f'path and velocities must have the same length, but got {len(path)} and {len(velocities)}'

        log_prob = (-system.U(path[0]) / kbT).sum()
        log_prob += jax.scipy.stats.norm.logpdf(velocities[0], 0, jnp.sqrt(kbT / system.mass)).sum()

        for i in range(1, len(path)):
            log_prob += step_langevin_log_prob(path[i - 1], velocities[i - 1], path[i], velocities[i])

        return log_prob


    # Choose a system, either phi psi, or rmsd
    # tps_config = tps1.System(
    #     jax.jit(jax.vmap(lambda s: kabsch_rmsd(A.reshape(22, 3), s.reshape(22, 3)) < 0.1)),
    #     jax.jit(jax.vmap(lambda s: kabsch_rmsd(B.reshape(22, 3), s.reshape(22, 3)) < 0.1)),
    #     step
    # )

    radius = 20 / deg

    # tps_config = tps1.FirstOrderSystem(
    #     lambda s: is_within(phis_psis(s).reshape(-1, 2), phis_psis(A), radius),
    #     lambda s: is_within(phis_psis(s).reshape(-1, 2), phis_psis(B), radius),
    #     step
    # )

    if args.states == 'rmsd':
        state_A = jax.jit(jax.vmap(lambda s: kabsch_rmsd(system.A.reshape(22, 3), s.reshape(22, 3)) <= 7.5e-2))
        state_B = jax.jit(jax.vmap(lambda s: kabsch_rmsd(system.B.reshape(22, 3), s.reshape(22, 3)) <= 7.5e-2))
    elif args.states == 'phi-psi':
        state_A = jax.jit(
            lambda s: is_within(phis_psis(s.reshape(-1, 22, 3)).reshape(-1, 2), phis_psis(system.A.reshape(-1, 22, 3)),
                                radius))
        state_B = jax.jit(
            lambda s: is_within(phis_psis(s.reshape(-1, 22, 3)).reshape(-1, 2), phis_psis(system.B.reshape(-1, 22, 3)),
                                radius))
    elif args.states == 'exact':
        from scipy.stats import chi2
        percentile = 0.99
        noise_scale = 1e-4
        threshold = jnp.sqrt(chi2.ppf(percentile, system.A.shape[0]) * noise_scale)
        print(threshold)
        def kabsch_l2(A, B):
            a, b = kabsch_align(A, B)

            return jnp.linalg.norm(a - b)

        state_A = jax.jit(jax.vmap(lambda s: kabsch_l2(system.A.reshape(22, 3), s.reshape(22, 3)) <= threshold))
        state_B = jax.jit(jax.vmap(lambda s: kabsch_l2(system.B.reshape(22, 3), s.reshape(22, 3)) <= threshold))
    else:
        raise ValueError(f"Unknown states {args.states}")

    tps_config = tps2.SecondOrderSystem(
        state_A, state_B,
        jax.jit(lambda _x, _v, _key: step_n(step_langevin_forward, _x, _v, args.num_steps, _key)),
        jax.jit(lambda _x, _v, _key: step_n(step_langevin_backward, _x, _v, args.num_steps, _key)),
        jax.jit(lambda key: jnp.sqrt(kbT / system.mass) * jax.random.normal(key, (1, 66)))
    )

    initial_trajectory = md.load('./files/AD_A_B_500K_initial_trajectory.pdb').xyz.reshape(-1, 1, 66)
    initial_trajectory = [p for p in initial_trajectory]

    if args.ensure_connected:
        initial_trajectory = [system.A] + [p for p in initial_trajectory] + [system.B]

    save_trajectory(system.mdtraj_topology, jnp.array(initial_trajectory), f'{savedir}/initial_trajectory.pdb')

    if args.resume:
        print('Loading stored data.')
        paths = [[x for x in p.astype(np.float32)] for p in tqdm(np.load(f'{savedir}/paths.npy', allow_pickle=True))]
        velocities = [[v for v in p.astype(np.float32)] for p in
                      tqdm(np.load(f'{savedir}/velocities.npy', allow_pickle=True))]
        with open(f'{savedir}/stats.json', 'r') as fp:
            statistics = json.load(fp)

        stored = {
            'trajectories': [initial_trajectory] + paths,
            'velocities': velocities,
            'statistics': statistics
        }

        print('Loaded', len(paths), 'paths.')
    else:
        if os.path.exists(f'{savedir}/paths.npy') and not args.override:
            print(f"The target directory is not empty.\n"
                  f"Please use --override to overwrite the existing data or --resume to continue.")
            exit(1)

        stored = None

    assert ((tps_config.start_state(system.A.reshape(1, -1)) and tps_config.target_state(system.B.reshape(1, -1)))
            or (tps_config.start_state(system.B.reshape(1, -1)) and tps_config.target_state(system.A.reshape(1, -1)))), \
        'A and B are not in the correct states. Please check your settings.'

    if args.mechanism == 'one-way-shooting':
        assert (tps_config.start_state(initial_trajectory[0])
                or tps_config.target_state(initial_trajectory[0])
                or tps_config.start_state(initial_trajectory[-1])
                or tps_config.target_state(initial_trajectory[-1])
                ), 'One-Way shooting requires the initial trajectory to start or end in one of the states.'
        mechanism = tps2.one_way_shooting
    elif args.mechanism == 'two-way-shooting':
        mechanism = tps2.two_way_shooting
    else:
        raise ValueError(f"Unknown mechanism {args.mechanism}")

    try:
        paths, velocities, statistics = tps2.mcmc_shooting(tps_config, mechanism, initial_trajectory,
                                                           args.num_paths, dt_in_ps, jax.random.PRNGKey(1),
                                                           warmup=args.warmup,
                                                           fixed_length=args.fixed_length,
                                                           stored=stored,
                                                           max_force_evaluations=10**10)  # 10billion
        # paths = tps2.unguided_md(tps_config, B, 1, key)
        print('Converting paths to jax.numpy arrays.')
        paths = [jnp.array(p) for p in tqdm(paths)]
        velocities = [jnp.array(p) for p in tqdm(velocities)]

        if not args.resume:
            # If we are resuming, everything is already stored
            print('Storing paths ...')
            np.save(f'{savedir}/paths.npy', np.array(paths, dtype=object), allow_pickle=True)
            print('Storing velocities ...')
            np.save(f'{savedir}/velocities.npy', np.array(velocities, dtype=object), allow_pickle=True)
            # save statistics, which is a dictionary
            with open(f'{savedir}/stats.json', 'w') as fp:
                json.dump(statistics, fp)
    except Exception as e:
        print(traceback.format_exc())
        breakpoint()

    if len(paths) == 0:
        print("No paths found.")
        exit(1)

    print(statistics)
    print('Number of force evaluations', sum(statistics['num_force_evaluations']))

    if args.fixed_length == 0:
        print([len(p) for p in paths])
        plt.hist([len(p) for p in paths], bins=jnp.sqrt(len(paths)).astype(int).item())
        show_or_save_fig(savedir, 'lengths', 'png')

    max_energy = [jnp.max(U_padded(path)) for path in tqdm(paths)]
    max_energy = np.array(max_energy)
    np.save(f'{savedir}/max_energy.npy', max_energy)

    plt.title(f"{human_format(len(paths))} paths @ {temp} K, dt = {human_format(dt)}s")
    system.plot(trajectories=paths, alpha=0.7)
    show_or_save_fig(savedir, 'paths', 'png')

    print("Plotting path-summary metrics.")

    plot_iterative_min_max_energy(paths, U_padded, statistics['num_force_evaluations'])
    show_or_save_fig(savedir, 'iterative_min_max', 'pdf')

    plot_path_energy(paths, jax.vmap(U_padded))
    plt.ylabel('Maximum energy')
    show_or_save_fig(savedir, 'max_energy', 'pdf')

    plot_path_energy(paths, jax.vmap(U_padded), reduce=jnp.median)
    show_or_save_fig(savedir, 'median_energy', 'pdf')

    plot_path_energy(list(zip(paths, velocities)), langevin_log_path_likelihood, reduce=lambda x: x, already_ln=True)
    plt.ylabel('Path Likelihood')
    show_or_save_fig(savedir, 'path_density', 'pdf')

    for i, path in tqdm(enumerate(paths), desc='Saving trajectories', total=len(paths)):
        save_trajectory(system.mdtraj_topology,
                        jnp.array([kabsch_align(p.reshape(-1, 3), system.B.reshape(-1, 3))[0] for p in path]),
                        f'{savedir}/trajectory_{i}.pdb')
