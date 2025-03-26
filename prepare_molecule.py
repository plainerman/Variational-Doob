import openmm.app as app
import openmm.unit as unit
import jax.numpy as jnp
import jax
# install dmff (from source)
from dmff import Hamiltonian, NeighborList
# install mdtraj
import mdtraj as md
import numpy as np
from tqdm import trange

from utils.animation import save_trajectory

if __name__ == '__main__':
    temp = 300
    kbT = 1.380649 * 6.02214076 * 1e-3 * temp
    gamma_in_ps = 1.0
    dt_in_ps = 1e-3
    steps = 1_000_000

    files = [('./files/chignolin_folded.pdb', './files/chignolin_folded_relaxed.pdb'),
             ('./files/chignolin_unfolded.pdb', './files/chignolin_unfolded_relaxed.pdb')]

    # !!! IMPORTANT !!!
    # The files still have to be kabsch aligned
    # use A, B = kabsch_align(A, B) to align two structures

    def minimize(pdb, out, steps):
        pdb = app.PDBFile(pdb)
        x = jnp.array(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).reshape(1, -1)

        mass = [a.element.mass.value_in_unit(unit.dalton) for a in pdb.topology.atoms()]
        new_mass = []
        for mass_ in mass:
            for _ in range(3):
                new_mass.append(mass_)
        mass = jnp.array(new_mass, dtype=jnp.float64)

        # Initialize the potential energy with amber forcefields
        ff = Hamiltonian('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
        potentials = ff.createPotential(pdb.topology,
                                        nonbondedMethod=app.NoCutoff,
                                        nonbondedCutoff=1.0 * unit.nanometers,
                                        constraints=None,
                                        ewaldErrorTolerance=0.0005)
        # Create a box used when calling
        box = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]])
        nbList = NeighborList(box, 4.0, potentials.meta["cov_map"])
        nbList.allocate(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        pairs = nbList.pairs

        @jax.jit
        @jax.vmap
        def U(_x):
            """
            Calling U by U(x, box, pairs, ff.paramset.parameters), x is [22, 3] and output the energy, if it is batched, use vmap
            """
            _U = potentials.getPotentialFunc()

            return _U(_x.reshape(-1, 3), box, pairs, ff.paramset.parameters)

        @jax.jit
        def dUdx_fn(_x):
            return jax.grad(lambda _x: U(_x).sum())(_x) / mass / gamma_in_ps

        @jax.jit
        def step_langevin_forward(_x, _v, _key):
            """Perform one step of forward langevin as implemented in openmm"""
            alpha = jnp.exp(-gamma_in_ps * dt_in_ps)
            f_scale = (1 - alpha) / gamma_in_ps
            new_v_det = alpha * _v + f_scale * -dUdx_fn(_x)
            new_v = new_v_det + jnp.sqrt(kbT * (1 - alpha ** 2) / mass) * jax.random.normal(_key, _x.shape)

            return _x + dt_in_ps * new_v, new_v

        print("Initial energy", U(x))
        key = jax.random.PRNGKey(0)
        v = jax.random.normal(jax.random.PRNGKey(1), x.shape) * jnp.sqrt(kbT / mass)
        min_energy, min_x = U(x), x
        for _ in trange(steps):
            key, iter_key = jax.random.split(key)
            x, v = step_langevin_forward(x, v, iter_key)
            energy = U(x)

            if energy < min_energy:
                min_energy = energy
                min_x = x

        print('Final energy', min_energy)
        mdtraj_topology = md.Topology.from_openmm(pdb.topology)
        save_trajectory(mdtraj_topology, min_x.reshape(1, -1, 3), out)


    for pdb, out in files:
        minimize(pdb, out, steps)
