from functools import partial
from utils.plot import toy_plot_energy_surface
import potentials
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Callable, Optional
import openmm.app as app
import openmm.unit as unit
from typing import Self
from utils.pdb import assert_same_molecule
from utils.rmsd import kabsch_align
from dmff import Hamiltonian, NeighborList  # This sets jax to use 64-bit precision
import mdtraj as md


class System:
    def __init__(self, U: Callable[[ArrayLike], ArrayLike], A: ArrayLike, B: ArrayLike, mass: ArrayLike, plot,
                 force_clip: float):
        assert A.shape == B.shape == mass.shape

        self.U = U

        dUdx = jax.grad(lambda _x: U(_x).sum())
        self.dUdx = jax.jit(jax.jit(lambda _x: jnp.clip(dUdx(_x), -force_clip, force_clip)))

        self.A, self.B = A, B
        self.mass = mass

        self.plot = plot

    @classmethod
    def from_name(cls, name: str, force_clip: float) -> Self:
        if name == 'double_well':
            U, A, B = potentials.double_well
        elif name == 'double_well_hard':
            U, A, B = potentials.double_well_hard
        elif name == 'double_well_dual_channel':
            U, A, B = potentials.double_well_dual_channel
            xlim = jnp.array((-1.0, 1.0))
            ylim = jnp.array((-1.0, 1.0))
        elif name == 'mueller_brown':
            U, A, B = potentials.mueller_brown
            xlim = jnp.array((-1.5, 0.9))
            ylim = jnp.array((-0.5, 1.7))
        else:
            raise ValueError(f"Unknown system: {name}")

        plot = partial(toy_plot_energy_surface,
                       U=U, states=list(zip(['A', 'B'], [A, B])), xlim=xlim, ylim=ylim, alpha=1.0
                       )
        mass = jnp.array([1.0, 1.0])
        return cls(U, A, B, mass, plot, force_clip)

    @classmethod
    def from_pdb(cls, A: str, B: str, forcefield: [str], cv: Optional[str], force_clip: float) -> Self:
        A_pdb, B_pdb = app.PDBFile(A), app.PDBFile(B)
        assert_same_molecule(A_pdb, B_pdb)

        mass = [a.element.mass.value_in_unit(unit.dalton) for a in A_pdb.topology.atoms()]
        mass = jnp.broadcast_to(jnp.array(mass, dtype=jnp.float32).reshape(-1, 1), (len(mass), 3)).reshape(-1)

        A = jnp.array(A_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=jnp.float32)
        B = jnp.array(B_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=jnp.float32)
        A, B = kabsch_align(A, B)
        A, B = A.reshape(-1), B.reshape(-1)

        # Initialize the potential energy with amber forcefields
        ff = Hamiltonian(*forcefield)
        potentials = ff.createPotential(A_pdb.topology,
                                        nonbondedMethod=app.NoCutoff,
                                        nonbondedCutoff=1.0 * unit.nanometers,
                                        constraints=None,
                                        ewaldErrorTolerance=0.0005)

        # Create a box used when calling
        box = jnp.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]])
        nbList = NeighborList(box, 4.0, potentials.meta["cov_map"])
        nbList.allocate(A_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))

        _U = potentials.getPotentialFunc()

        @jax.jit
        @jax.vmap
        def U(_x):
            return _U(_x.reshape(22, 3), box, nbList.pairs, ff.paramset.parameters).sum()

        if cv is None:
            plot = None
        elif cv == 'phi_psi':
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown cv: {cv}")

        return cls(U, A, B, mass, plot, force_clip)
