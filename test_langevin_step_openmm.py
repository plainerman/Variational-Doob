import openmm.app as app
import mdtraj as md
import openmm.unit as unit
import jax.numpy as jnp
import jax
from dmff import Hamiltonian, NeighborList
from scipy.constants import physical_constants
from tqdm import trange

if __name__ == '__main__':
    init_pdb = app.PDBFile("./files/AD_c7eq.pdb")
    temp_as_unit = 300 * unit.kelvin
    temp = temp_as_unit.value_in_unit(unit.kelvin)

    # The unit system is weird when you have 1 / ps, so we specify it in ps and then convert back to seconds
    gamma_in_ps = 1
    gamma_as_unit = gamma_in_ps / unit.picosecond
    gamma = gamma_in_ps * unit.picosecond
    gamma = gamma.value_in_unit(unit.second)

    dt_as_unit = 1.0 * unit.femtosecond
    dt_in_ps = dt_as_unit.value_in_unit(unit.picosecond)
    dt = dt_as_unit.value_in_unit(unit.second)

    kbT = 1.380649 * 6.02214076 * 1e-3 * temp

    mdtraj_topology = md.Topology.from_openmm(init_pdb.topology)

    # Construct the mass matrix
    mass = [a.element.mass.value_in_unit(unit.dalton) for a in init_pdb.topology.atoms()]
    new_mass = []
    for mass_ in mass:
        for _ in range(3):
            new_mass.append(mass_)
    mass = jnp.array(new_mass)
    # Obtain xi
    xi = jnp.sqrt(2 * kbT / mass / gamma)

    # Initialize the potential energy with amber forcefields
    ff = Hamiltonian('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    potentials = ff.createPotential(init_pdb.topology,
                                    nonbondedMethod=app.NoCutoff,
                                    nonbondedCutoff=1.0 * unit.nanometers,
                                    constraints=None,
                                    ewaldErrorTolerance=0.0005)
    # Create a box used when calling
    box = jnp.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]])
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


    def dUdx_fn_unscaled(_x):
        return jax.grad(lambda _x: U(_x).sum())(_x)


    dUdx_fn_unscaled = jax.vmap(dUdx_fn_unscaled)
    dUdx_fn_unscaled = jax.jit(dUdx_fn_unscaled)


    @jax.jit
    def dUdx_fn(_x):
        return dUdx_fn_unscaled(_x) / mass / gamma


    @jax.jit
    def step_langevin(_x, _v, _key):
        alpha = jnp.exp(-gamma_in_ps * dt_in_ps)
        f_scale = (1 - alpha) / gamma_in_ps
        new_v_det = alpha * _v + f_scale * -dUdx_fn_unscaled(_x) / mass
        new_v = new_v_det + jnp.sqrt(kbT * (1 - alpha ** 2) / mass) * jax.random.normal(_key, _x.shape)

        return _x + dt_in_ps * new_v, new_v


    def step_langevin_units(_x, _v, _key):
        _x = unit.Quantity(_x.reshape(22, 3), unit.nanometer)
        _v = unit.Quantity(_v.reshape(22, 3), unit.nanometer / unit.picosecond)

        alpha = jnp.exp(-gamma_as_unit * dt_as_unit)
        f_scale = (1 - alpha) / gamma_as_unit

        grad = unit.Quantity(value=dUdx_fn_unscaled(_x.value_in_unit(unit.nanometer).reshape(1, 66)).reshape(22, 3),
                             unit=unit.kilojoule_per_mole / unit.nanometer)
        mass_as_unit = unit.Quantity(mass.reshape(22, 3), unit.dalton)
        new_v_det = alpha * _v + f_scale * -grad / mass_as_unit

        _rand = jax.random.normal(_key, _x.shape)

        # we convert the unadjusted noise variance to SI units
        unadjusted_noise_variance = unit.BOLTZMANN_CONSTANT_kB * temp_as_unit * (1 - alpha ** 2) / mass_as_unit
        # to do this, we need to convert daltons to kg
        noise_scale_SI_units = 1 / physical_constants['unified atomic mass unit'][
            0] * unadjusted_noise_variance.value_in_unit(unit.joule / unit.dalton)

        # in the end, we need to convert the noise back to nanometers
        # since we are working in SI units, our noise is in meters
        noise = unit.Quantity(jnp.sqrt(noise_scale_SI_units) * _rand, unit.meter / unit.second)

        new_v = new_v_det + noise
        return ((_x + dt_as_unit * new_v).value_in_unit(unit.nanometer).reshape(1, 66),
                new_v.value_in_unit(unit.nanometer / unit.picosecond).reshape(1, 66))


    key = jax.random.PRNGKey(1)
    key, velocity_key = jax.random.split(key)
    steps = 100_000

    _x = jnp.array(init_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).reshape(1, -1)

    # sample the initial velocities from the boltzmann distribution
    # again, we compare the velocities in the same way as we did with the positions
    _v_v1 = jax.random.normal(velocity_key, _x.shape) * jnp.sqrt(kbT / mass)

    velocity_variance = unit.Quantity(1 / mass, unit=1 / unit.dalton) * unit.BOLTZMANN_CONSTANT_kB * unit.Quantity(temp, unit=unit.kelvin)
    # Although velocity+variance is of the unit J / Da = m^2 / s^2, openmm cannot handle this directly and we need to convert it
    velocity_variance_in_si = 1 / physical_constants['unified atomic mass unit'][
        0] * velocity_variance.value_in_unit(unit.joule / unit.dalton)
    # velocity_variance_in_si = unit.Quantity(velocity_variance_in_si, unit.meter / unit.second)

    _v_v2 = jnp.sqrt(velocity_variance_in_si) * jax.random.normal(velocity_key, _x.shape)
    _v_v2 = unit.Quantity(_v_v2, unit.meter / unit.second).value_in_unit(unit.nanometer / unit.picosecond)

    assert jnp.allclose(_v_v1, _v_v2), "Initial velocities are not the same!"

    _v = _v_v1

    for i in trange(steps):
        key, iter_key = jax.random.split(key)
        _x_v1, _v_v1 = step_langevin(_x, _v, iter_key)
        _x_v2, _v_v2 = step_langevin_units(_x, _v, iter_key)

        assert jnp.allclose(_x_v1, _x_v2), "Positions are not the same!"
        assert jnp.allclose(_v_v1, _v_v2), "Velocities are not the same!"

        _x = _x_v1
        _v = _v_v1

    print('All tests passed!')
