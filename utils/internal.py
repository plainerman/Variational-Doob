import jax
import jax.numpy as jnp
from jax import grad, jit

# File was ported from:
# https://github.com/VincentStimper/boltzmann-generators/blob/2b177fc155f533933489b8fce8d6483ebad250d3/boltzgen/internal.py


def calc_bonds(ind1, ind2, coords):
    """Calculate bond lengths

    Parameters
    ----------
    ind1 : jnp.ndarray
        A n_bond x 3 array of indices for the coordinates of particle 1
    ind2 : jnp.ndarray
        A n_bond x 3 array of indices for the coordinates of particle 2
    coords : jnp.ndarray
        A n_batch x n_coord array of flattened input coordinates
    """
    p1 = coords[:, ind1]
    p2 = coords[:, ind2]
    return jnp.linalg.norm(p2 - p1, axis=2)


def calc_angles(ind1, ind2, ind3, coords):
    b = coords[:, ind1]
    c = coords[:, ind2]
    d = coords[:, ind3]
    bc = b - c
    bc /= jnp.linalg.norm(bc, axis=2, keepdims=True)
    cd = d - c
    cd /= jnp.linalg.norm(cd, axis=2, keepdims=True)
    cos_angle = jnp.sum(bc * cd, axis=2)
    angle = jnp.arccos(cos_angle)
    return angle


def calc_dihedrals(ind1, ind2, ind3, ind4, coords):
    a = coords[:, ind1]
    b = coords[:, ind2]
    c = coords[:, ind3]
    d = coords[:, ind4]

    b0 = a - b
    b1 = c - b
    b1 /= jnp.linalg.norm(b1, axis=2, keepdims=True)
    b2 = d - c

    v = b0 - jnp.sum(b0 * b1, axis=2, keepdims=True) * b1
    w = b2 - jnp.sum(b2 * b1, axis=2, keepdims=True) * b1
    x = jnp.sum(v * w, axis=2)
    b1xv = jnp.cross(b1, v, axis=2)
    y = jnp.sum(b1xv * w, axis=2)
    angle = jnp.arctan2(y, x)
    return -angle


def reconstruct_cart(cart, ref_atoms, bonds, angles, dihs):
    # Get the positions of the 4 reconstructing atoms
    p1 = cart[:, ref_atoms[:, 0], :]
    p2 = cart[:, ref_atoms[:, 1], :]
    p3 = cart[:, ref_atoms[:, 2], :]

    bonds = jnp.expand_dims(bonds, axis=2)
    angles = jnp.expand_dims(angles, axis=2)
    dihs = jnp.expand_dims(dihs, axis=2)

    # Reconstruct the position of p4
    v1 = p1 - p2
    v2 = p1 - p3

    n = jnp.cross(v1, v2, axis=2)
    n = n / jnp.linalg.norm(n, axis=2, keepdims=True)
    nn = jnp.cross(v1, n, axis=2)
    nn = nn / jnp.linalg.norm(nn, axis=2, keepdims=True)

    n = n * jnp.sin(dihs)
    nn = nn * jnp.cos(dihs)

    v3 = n + nn
    v3 = v3 / jnp.linalg.norm(v3, axis=2, keepdims=True)
    v3 = v3 * bonds * jnp.sin(angles)

    v1 = v1 / jnp.linalg.norm(v1, axis=2, keepdims=True)
    v1 = v1 * bonds * jnp.cos(angles)

    # Store the final position in x
    new_cart = p1 + v3 - v1

    return new_cart


class InternalCoordinateTransform:
    def __init__(self, dims, z_indices=None, cart_indices=None, data=None,
                 ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2}):
        self.dims = dims
        # Setup indexing.
        self._setup_indices(z_indices, cart_indices)
        self._validate_data(data)
        # Setup the mean and standard deviations for each internal coordinate.
        transformed = self._fwd(data)
        # Normalize
        self.default_std = default_std
        self.ind_circ_dih = ind_circ_dih
        self._setup_mean_bonds(transformed)
        transformed = transformed.at[:, self.bond_indices].set(transformed[:, self.bond_indices] - self.mean_bonds)
        self._setup_std_bonds(transformed)
        transformed = transformed.at[:, self.bond_indices].set(transformed[:, self.bond_indices] / self.std_bonds)
        self._setup_mean_angles(transformed)
        transformed = transformed.at[:, self.angle_indices].set(transformed[:, self.angle_indices] - self.mean_angles)
        self._setup_std_angles(transformed)
        transformed = transformed.at[:, self.angle_indices].set(transformed[:, self.angle_indices] / self.std_angles)
        self._setup_mean_dih(transformed)
        transformed = transformed.at[:, self.dih_indices].set(transformed[:, self.dih_indices] - self.mean_dih)
        transformed = self._fix_dih(transformed)
        self._setup_std_dih(transformed)
        transformed = transformed.at[:, self.dih_indices].set(transformed[:, self.dih_indices] / self.std_dih)
        if shift_dih:
            val = jnp.linspace(-jnp.pi, jnp.pi,
                               shift_dih_params['hist_bins'])
            for i in self.ind_circ_dih:
                dih = transformed[:, self.dih_indices[i]]
                dih = dih * self.std_dih[i] + self.mean_dih[i]
                dih = (dih + jnp.pi) % (2 * jnp.pi) - jnp.pi
                hist = jnp.histogram(dih, bins=shift_dih_params['hist_bins'],
                                     range=(-jnp.pi, jnp.pi))[0]
                self.mean_dih = self.mean_dih.at[i].set(val[jnp.argmin(hist)] + jnp.pi)
                dih = (dih - self.mean_dih[i]) / self.std_dih[i]
                dih = (dih + jnp.pi) % (2 * jnp.pi) - jnp.pi
                transformed = transformed.at[:, self.dih_indices[i]].set(dih)

    def to_internal(self, x):
        trans = self._fwd(x)
        trans = trans.at[:, self.bond_indices].set(trans[:, self.bond_indices] - self.mean_bonds)
        trans = trans.at[:, self.bond_indices].set(trans[:, self.bond_indices] / self.std_bonds)
        trans = trans.at[:, self.angle_indices].set(trans[:, self.angle_indices] - self.mean_angles)
        trans = trans.at[:, self.angle_indices].set(trans[:, self.angle_indices] / self.std_angles)
        trans = trans.at[:, self.dih_indices].set(trans[:, self.dih_indices] - self.mean_dih)
        trans = self._fix_dih(trans)
        trans = trans.at[:, self.dih_indices].set(trans[:, self.dih_indices] / self.std_dih)
        return trans

    def _fwd(self, x):
        # we can do everything in parallel...
        inds1 = self.inds_for_atom[self.rev_z_indices[:, 1]]
        inds2 = self.inds_for_atom[self.rev_z_indices[:, 2]]
        inds3 = self.inds_for_atom[self.rev_z_indices[:, 3]]
        inds4 = self.inds_for_atom[self.rev_z_indices[:, 0]]

        # Calculate the bonds, angles, and torsions for a batch.
        bonds = calc_bonds(inds1, inds4, coords=x)
        angles = calc_angles(inds2, inds1, inds4, coords=x)
        dihedrals = calc_dihedrals(inds3, inds2, inds1, inds4, coords=x)

        # Replace the cartesian coordinates with internal coordinates.
        x = x.at[:, inds4[:, 0]].set(bonds)
        x = x.at[:, inds4[:, 1]].set(angles)
        x = x.at[:, inds4[:, 2]].set(dihedrals)
        return x

    def to_cartesian(self, x):
        # Gather all of the atoms represented as Cartesian coordinates.
        n_batch = x.shape[0]
        cart = x[:, self.init_cart_indices].reshape(n_batch, -1, 3)

        # Loop over all of the blocks, where all of the atoms in each block
        # can be built in parallel because they only depend on atoms that
        # are already Cartesian. `atoms_to_build` lists the `n` atoms
        # that can be built as a batch, where the indexing refers to the
        # original atom order. `ref_atoms` has size n x 3, where the indexing
        # refers to the position in `cart`, rather than the original order.
        for block in self.rev_blocks:
            atoms_to_build = block[:, 0]
            ref_atoms = block[:, 1:]

            # Get all of the bonds by retrieving the appropriate columns and
            # un-normalizing.
            bonds = (
                    x[:, 3 * atoms_to_build]
                    * self.std_bonds[self.atom_to_stats[atoms_to_build]]
                    + self.mean_bonds[self.atom_to_stats[atoms_to_build]]
            )

            # Get all of the angles by retrieving the appropriate columns and
            # un-normalizing.
            angles = (
                    x[:, 3 * atoms_to_build + 1]
                    * self.std_angles[self.atom_to_stats[atoms_to_build]]
                    + self.mean_angles[self.atom_to_stats[atoms_to_build]]
            )
            # Get all of the dihedrals by retrieving the appropriate columns and
            # un-normalizing.
            dihs = (
                    x[:, 3 * atoms_to_build + 2]
                    * self.std_dih[self.atom_to_stats[atoms_to_build]]
                    + self.mean_dih[self.atom_to_stats[atoms_to_build]]
            )

            # Fix the dihedrals to lie in [-pi, pi].
            dihs = jnp.where(dihs < jnp.pi, dihs + 2 * jnp.pi, dihs)
            dihs = jnp.where(dihs > jnp.pi, dihs - 2 * jnp.pi, dihs)

            # Compute the Cartesian coordinates for the newly placed atoms.
            new_cart = reconstruct_cart(cart, ref_atoms, bonds, angles, dihs)

            # Concatenate the Cartesian coordinates for the newly placed
            # atoms onto the full set of Cartesian coordinates.
            cart = jnp.concatenate([cart, new_cart], axis=1)
        # Permute cart back into the original order and flatten.
        cart = cart[:, self.rev_perm_inv]
        cart = cart.reshape(n_batch, -1)
        return cart

    def _setup_mean_bonds(self, x):
        self.mean_bonds = jnp.mean(x[:, self.bond_indices], axis=0)

    def _setup_std_bonds(self, x):
        if x.shape[0] > 1:
            self.std_bonds = jnp.std(x[:, self.bond_indices], axis=0)
        else:
            self.std_bonds = jnp.ones_like(self.mean_bonds) * self.default_std['bond']

    def _setup_mean_angles(self, x):
        self.mean_angles = jnp.mean(x[:, self.angle_indices], axis=0)

    def _setup_std_angles(self, x):
        if x.shape[0] > 1:
            self.std_angles = jnp.std(x[:, self.angle_indices], axis=0)
        else:
            self.std_angles = jnp.ones_like(self.mean_angles) * self.default_std['angle']

    def _setup_mean_dih(self, x):
        sin = jnp.mean(jnp.sin(x[:, self.dih_indices]), axis=0)
        cos = jnp.mean(jnp.cos(x[:, self.dih_indices]), axis=0)
        self.mean_dih = jnp.arctan2(sin, cos)

    def _fix_dih(self, x):
        dih = x[:, self.dih_indices]
        dih = (dih + jnp.pi) % (2 * jnp.pi) - jnp.pi
        x = x.at[:, self.dih_indices].set(dih)
        return x

    def _setup_std_dih(self, x):
        if x.shape[0] > 1:
            self.std_dih = jnp.std(x.at[:, self.dih_indices], axis=0)
        else:
            self.std_dih = jnp.ones_like(self.mean_dih) * self.default_std['dih']
            if len(self.ind_circ_dih) > 0:
                self.std_dih = self.std_dih.at[jnp.array(self.ind_circ_dih)].set(1.)

    def _validate_data(self, data):
        if data is None:
            raise ValueError(
                "InternalCoordinateTransform must be supplied with training_data."
            )

        if len(data.shape) != 2:
            raise ValueError("training_data must be n_samples x n_dim array")

        n_dim = data.shape[1]

        if n_dim != self.dims:
            raise ValueError(
                f"training_data must have {self.dims} dimensions, not {n_dim}."
            )

    def _setup_indices(self, z_indices, cart_indices):
        n_atoms = self.dims // 3
        ind_for_atom = jnp.zeros((n_atoms, 3), dtype=jnp.int32)
        for i in range(n_atoms):
            ind_for_atom = ind_for_atom.at[i].set([3 * i, 3 * i + 1, 3 * i + 2])
        self.inds_for_atom = ind_for_atom

        sorted_z_indices = topological_sort(z_indices)
        sorted_z_indices = [
            [item[0], item[1][0], item[1][1], item[1][2]] for item in sorted_z_indices
        ]
        rev_z_indices = list(reversed(sorted_z_indices))

        mod = [item[0] for item in sorted_z_indices]
        modified_indices = []
        for index in mod:
            modified_indices.extend(self.inds_for_atom[index])
        bond_indices = list(modified_indices[0::3])
        angle_indices = list(modified_indices[1::3])
        dih_indices = list(modified_indices[2::3])

        self.modified_indices = jnp.array(modified_indices)
        self.bond_indices = jnp.array(bond_indices)
        self.angle_indices = jnp.array(angle_indices)
        self.dih_indices = jnp.array(dih_indices)
        self.sorted_z_indices = jnp.array(sorted_z_indices)
        self.rev_z_indices = jnp.array(rev_z_indices)

        #
        # Setup indexing for reverse pass.
        #
        # First, create an array that maps from an atom index into mean_bonds, std_bonds, etc.
        atom_to_stats = jnp.zeros(n_atoms, dtype=jnp.int32)
        for i, j in enumerate(mod):
            atom_to_stats = atom_to_stats.at[j].set(i)
        self.atom_to_stats = atom_to_stats

        # Next create permutation vector that is used in the reverse pass. This maps
        # from the original atom indexing to the order that the Cartesian coordinates
        # will be built in. This will be filled in as we go.
        rev_perm = jnp.zeros(n_atoms, dtype=jnp.int32)
        self.rev_perm = rev_perm
        # Next create the inverse of rev_perm. This will be filled in as we go.
        rev_perm_inv = jnp.zeros(n_atoms, dtype=jnp.int32)
        self.rev_perm_inv = rev_perm_inv

        # Create the list of columns that form our initial Cartesian coordinates.
        init_cart_indices = self.inds_for_atom[jnp.array(cart_indices)].reshape(-1)
        self.init_cart_indices = init_cart_indices

        # Update our permutation vectors for the initial Cartesian atoms.
        for i, j in enumerate(cart_indices):
            self.rev_perm = self.rev_perm.at[i].set(j)
            self.rev_perm_inv = self.rev_perm_inv.at[j].set(i)

        # Break Z into blocks, where all of the atoms within a block
        # can be built in parallel, because they only depend on
        # atoms that are already Cartesian.
        all_cart = set(cart_indices)
        current_cart_ind = i + 1
        blocks = []
        while sorted_z_indices:
            next_z_indices = []
            next_cart = set()
            block = []
            for atom1, atom2, atom3, atom4 in sorted_z_indices:
                if (atom2 in all_cart) and (atom3 in all_cart) and (atom4 in all_cart):
                    # We can build this atom from existing Cartesian atoms,
                    # so we add it to the list of Cartesian atoms available for the next block.
                    next_cart.add(atom1)

                    # Add this atom to our permutation matrices.
                    self.rev_perm = self.rev_perm.at[current_cart_ind].set(atom1)
                    self.rev_perm_inv = self.rev_perm_inv.at[atom1].set(current_cart_ind)
                    current_cart_ind += 1

                    # Next, we convert the indices for atoms2-4 from their normal values
                    # to the appropriate indices to index into the Cartesian array.
                    atom2_mod = self.rev_perm_inv[atom2]
                    atom3_mod = self.rev_perm_inv[atom3]
                    atom4_mod = self.rev_perm_inv[atom4]

                    # Finally, we append this information to the current block.
                    block.append([atom1, atom2_mod, atom3_mod, atom4_mod])
                else:
                    # We can't build this atom from existing Cartesian atoms,
                    # so put it on the list for next time.
                    next_z_indices.append([atom1, atom2, atom3, atom4])
            sorted_z_indices = next_z_indices
            all_cart = all_cart.union(next_cart)
            block = jnp.array(block)
            blocks.append(block)
        self.rev_blocks = blocks


def topological_sort(graph_unsorted):
    graph_sorted = []
    graph_unsorted = dict(graph_unsorted)

    while graph_unsorted:
        acyclic = False
        for node, edges in list(graph_unsorted.items()):
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))

        if not acyclic:
            raise RuntimeError("A cyclic dependency occured.")

    return graph_sorted
