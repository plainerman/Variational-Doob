import jax.numpy as jnp
import jax
from mdtraj.geometry import indices_phi, indices_psi


@jax.jit
def dihedral(p):
    """http://stackoverflow.com/q/20305272/1128289"""
    b = p[:-1] - p[1:]
    b = b.at[0].set(-b[0])
    v = jnp.array(
        [v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= jnp.sqrt(jnp.einsum('...i,...i', v, v)).reshape(-1, 1)
    b1 = b[1] / jnp.linalg.norm(b[1])
    x = jnp.dot(v[0], v[1])
    m = jnp.cross(v[0], b1)
    y = jnp.dot(m, v[1])
    return jnp.arctan2(y, x)


def phi_psi_from_mdtraj(mdtraj_topology):
    angles_phi = indices_phi(mdtraj_topology)[0]
    angles_psi = indices_psi(mdtraj_topology)[0]

    assert len(angles_phi) == len(angles_psi) == 4

    @jax.jit
    @jax.vmap
    def phi_psi(p):
        p = p.reshape(mdtraj_topology.n_atoms, 3)
        phi = dihedral(p[angles_phi, :])
        psi = dihedral(p[angles_psi, :])

        return jnp.array([phi, psi])

    return phi_psi


if __name__ == '__main__':
    import openmm.app as app
    import openmm.unit as unit
    import mdtraj as md
    from utils.animation import to_md_traj

    init_pdb = app.PDBFile("../files/AD_A.pdb")
    target_pdb = app.PDBFile("../files/AD_B.pdb")
    mdtraj_topology = md.Topology.from_openmm(init_pdb.topology)

    A = jnp.array(init_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    B = jnp.array(target_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))

    phi_psi = phi_psi_from_mdtraj(mdtraj_topology)
    print(phi_psi(A.reshape(1, 22, 3)))
    print(phi_psi(B.reshape(1, 22, 3)))

    traj = to_md_traj(mdtraj_topology, A)
    phi = md.compute_phi(traj)[1].squeeze()
    psi = md.compute_psi(traj)[1].squeeze()
    print(phi, psi)

    traj = to_md_traj(mdtraj_topology, B)
    phi = md.compute_phi(traj)[1].squeeze()
    psi = md.compute_psi(traj)[1].squeeze()
    print(phi, psi)
