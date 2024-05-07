import mdtraj as md


def to_md_traj(mdtraj_topology, trajectory):
    return md.Trajectory(trajectory.reshape(-1, mdtraj_topology.n_atoms, 3), mdtraj_topology)


def save_trajectory(mdtraj_topology, trajectory, out: str,):
    return to_md_traj(mdtraj_topology, trajectory).save(out)
