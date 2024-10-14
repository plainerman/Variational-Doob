import json
from functools import partial

import numpy as np
import jax.numpy as jnp
import jax
from eval.path_metrics import plot_path_energy
import matplotlib.pyplot as plt
import os
import openmm.app as app
import openmm.unit as unit
from dmff import Hamiltonian, NeighborList
from tqdm import tqdm

from tps.paths import decorrelated

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

init_pdb = app.PDBFile('./files/AD_A.pdb')
# Construct the mass matrix
mass = [a.element.mass.value_in_unit(unit.dalton) for a in init_pdb.topology.atoms()]
new_mass = []
for mass_ in mass:
    for _ in range(3):
        new_mass.append(mass_)
mass = jnp.array(new_mass)
# Obtain xi, gamma is by default 1
xi = jnp.sqrt(2 * kbT / mass / gamma)

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
@jax.vmap
def U_native(_x):
    """
    Calling U by U(x, box, pairs, ff.paramset.parameters), x is [22, 3] and output the energy, if it is batched, use vmap
    """
    _U = potentials.getPotentialFunc()

    return _U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters).sum()


def U_padded(x):
    x = x.reshape(-1, 66)
    orig_length = x.shape[0]
    padded_length = orig_length // 100 * 100 + 100
    x_empty = jnp.zeros((padded_length, 66))
    x = x_empty.at[:x.shape[0], :].set(x.reshape(-1, 66))
    return U_native(x)[:orig_length]


@jax.jit
@jax.vmap
def dUdx_fn(_x):
    def U(_x):
        """
        Calling U by U(x, box, pairs, ff.paramset.parameters), x is [22, 3] and output the energy, if it is batched, use vmap
        """
        _U = potentials.getPotentialFunc()

        return _U(_x.reshape(22, 3), box, pairs, ff.paramset.parameters)

    return jax.grad(lambda _x: U(_x).sum())(_x) / mass / gamma_in_ps


@jax.jit
def step_langevin_log_prob(_x, _v, _new_x, _new_v):
    alpha = jnp.exp(-gamma_in_ps * dt_in_ps)
    f_scale = (1 - alpha) / gamma_in_ps
    new_v_det = alpha * _v + f_scale * -dUdx_fn(_x.reshape(1, -1))
    new_v_rand = new_v_det - _new_v

    return jax.scipy.stats.norm.logpdf(new_v_rand, 0, jnp.sqrt(kbT * (1 - alpha ** 2) / mass)).sum()


def langevin_log_path_likelihood(path, velocities):
    assert len(path) == len(
        velocities), f'path and velocities must have the same length, but got {len(path)} and {len(velocities)}'
    log_prob = (-U_native(path[0].reshape(1, -1)) / kbT).sum()
    log_prob += jax.scipy.stats.norm.logpdf(velocities[0], 0, jnp.sqrt(kbT / mass)).sum()

    for i in range(1, len(path)):
        log_prob += step_langevin_log_prob(path[i - 1], velocities[i - 1], path[i], velocities[i])

    # log_prob += step_langevin_log_prob(path[:-1], velocities[:-1], path[1:], velocities[1:]).sum()

    return log_prob


def load(path):
    loaded = np.load(path, allow_pickle=True)
    return [p.astype(np.float32).reshape(-1, 66) for p in loaded]


if __name__ == '__main__':
    savedir = './out/evaluation/alanine/'
    os.makedirs(savedir, exist_ok=True)

    all_paths = [
        ('one-way-shooting-var-length-cv', './out/baselines/alanine-one-way-shooting', 50),
        ('one-way-shooting-var-length-rmsd', './out/baselines/alanine-one-way-shooting-rmsd', 50),
        ('one-way-shooting-fixed-length-cv', './out/baselines/alanine-one-way-shooting-1000steps', 50),
        ('one-way-shooting-fixed-length-rmsd', './out/baselines/alanine-one-way-shooting-1000steps-rmsd', 50),
        ('two-way-shooting-var-length-cv', './out/baselines/alanine-two-way-shooting', 0),
        ('two-way-shooting-var-length-rmsd', './out/baselines/alanine-two-way-shooting-rmsd', 0),
        ('two-way-shooting-fixed-length-cv', './out/baselines/alanine-two-way-shooting-1000steps', 0),
    ]

    all_paths = [(name, path, warmup) for name, path, warmup in all_paths if os.path.exists(path)]
    print('Running script for the following paths:')
    [print(name, path) for name, path, warmup in all_paths]
    assert len(all_paths) > 0, 'No paths found, please consider running tps_baseline_mueller.py first.'

    # print relevant statistics:
    for name, file_path, _warmup in all_paths:
        with open(f'{file_path}/stats.json', 'r') as fp:
            statistics = json.load(fp)
            print(name, statistics)

    all_paths = [(name, load(f'{path}/paths.npy')[warmup:], load(f'{path}/velocities.npy')[warmup:]) for
                 name, path, warmup in tqdm(all_paths, desc='loading paths')]
    [print(name, len(path), len(velocities)) for name, path, velocities in all_paths]

    for name, paths, _velocities in all_paths:
        print(name, 'decorrelated trajectories:', jnp.round(100 * len(decorrelated(paths)) / len(paths), 2), '%')

    for name, paths, _velocities in all_paths:
        max_energy = jnp.array([jnp.max(U_padded(path)) for path in tqdm(paths)])
        print(name, 'max energy mean:', jnp.round(jnp.mean(max_energy), 2), 'std:', jnp.round(jnp.std(max_energy), 2))
        print(name, 'min max energy:', jnp.round(jnp.min(max_energy), 2))

    for name, paths, velocities in all_paths:
        log_likelihood = jnp.array(
            [langevin_log_path_likelihood(path, current_velocities) for path, current_velocities in
             tqdm(zip(paths, velocities), total=len(paths))])

        print(name, 'max log likelihood:', jnp.round(jnp.max(log_likelihood), 2))
        print(name, 'mean log likelihood:', jnp.round(jnp.mean(log_likelihood), 2), 'std:',
              jnp.round(jnp.std(log_likelihood), 2))
