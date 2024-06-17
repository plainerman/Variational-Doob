from functools import partial
import utils.toy_plot_helpers as toy
import potentials
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Callable
import openmm.app as app
import openmm.unit as unit
from typing import Self


class System:
    def __init__(self, U: Callable[[ArrayLike], ArrayLike], A: ArrayLike, B: ArrayLike, mass: ArrayLike, plot):
        assert A.shape == B.shape == mass.shape

        self.U = U
        self.dUdx = jax.jit(jax.grad(lambda _x: U(_x).sum()))

        self.A, self.B = A, B
        self.mass = mass

        self.plot = plot

    @classmethod
    def from_name(cls, name: str) -> Self:
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

        plot = partial(toy.plot_energy_surface, U=U, states=list(zip(['A', 'B'], [A, B])), xlim=xlim, ylim=ylim, alpha=1.0)
        mass = jnp.array([1.0, 1.0])
        return cls(U, A, B, mass, plot)

    @classmethod
    def from_pdb(cls, A: str, B: str, CV: Callable[[ArrayLike], ArrayLike] = None) -> Self:
        # TODO: how to handle alanine with plotting? CV? plot function?
        A = app.PDBFile(A)
        B = app.PDBFile(B)

        # TODO: I don't think that this will work
        assert A.topology == B.topology, "Topologies of A and B must match"

        # kabsch align A and B

        mass = [a.element.mass.value_in_unit(unit.dalton) for a in A.topology.atoms()]
        mass = jnp.broadcast_to(jnp.array(mass).reshape(-1, 1), (len(mass), 3)).reshape(-1)

        # TODO: remove if previous line works
        # assert [a.element.mass.value_in_unit(unit.dalton) for a in A.topology.atoms()] ==  [a.element.mass.value_in_unit(unit.dalton) for a in B.topology.atoms()]

        raise NotImplementedError
