from openmm import app


def assert_same_molecule(A: app.PDBFile, B: app.PDBFile):
    """Check whether the two PDB files are equal (up to atom positions)."""
    assert A.topology.getNumChains() == B.topology.getNumChains(), "Number of chains do not match"

    # Compare chains, residues, and atoms
    for chainA, chainB in zip(A.topology.chains(), B.topology.chains()):
        assert len(list(chainA.residues())) == len(list(chainB.residues())), "Number of residues do not match"

        for residueA, residueB in zip(chainA.residues(), chainB.residues()):
            assert len(list(residueA.atoms())) == len(list(residueB.atoms())), "Number of atoms do not match"

            assert [a.element for a in residueA.atoms()] == [a.element for a in
                                                             residueB.atoms()], "Elements do not match"

    assert A.topology.getNumBonds() == B.topology.getNumBonds(), "Number of bonds do not match"
    for bondA, bondB in zip(A.topology.bonds(), B.topology.bonds()):
        assert bondA[0].element == bondB[0].element, "Elements of bond atoms do not match"
        assert bondA[1].element == bondB[1].element, "Elements of bond atoms do not match"
