from rdkit import Chem
from syntheseus.interface.molecule import Molecule

from retrochimera.chem.rewrite import Rewrite
from retrochimera.encoders.featurizers import (
    DGLLifeMoleculeFeaturizer,
    DGLLifeRewriteFeaturizer,
    RawGraph,
    RawRewriteGraph,
)


def test_molecule_featurizer() -> None:
    featurizer = DGLLifeMoleculeFeaturizer.get_default()
    molecule = Molecule("COCCOc1ccc(C(C)=O)c(O)c1")

    raw_graph = featurizer(molecule)
    assert isinstance(raw_graph, RawGraph)
    assert not isinstance(raw_graph, RawRewriteGraph)

    rdkit_mol = Chem.MolFromSmiles(molecule.smiles)

    assert raw_graph.atom_features.shape == (rdkit_mol.GetNumAtoms(), featurizer.num_atom_features)
    assert raw_graph.bond_features.shape == (len(raw_graph.bonds), featurizer.num_bond_features)
    assert raw_graph.bonds.shape == (2 * rdkit_mol.GetNumBonds(), 2)


def test_rewrite_featurizer() -> None:
    featurizer = DGLLifeRewriteFeaturizer.get_default(
        atom_smarts_vocab=["C", "O", "[C&H1&+0&D2]", "[O&H0&+0&D1]", "[C&H1&+0&D3]", "[O&H0&+0&D2]"]
    )
    rewrite = Rewrite.from_rdkit(
        "[C:1]-[C&H1&+0&D2:3]=[O&H0&+0&D1:2]>>[C:1]-[C&H1&+0&D3:3]1-[O&H0&+0&D2:2]-C-C-O-1"
    )

    raw_graph = featurizer(rewrite)
    assert isinstance(raw_graph, RawRewriteGraph)

    rdkit_lhs = Chem.MolFromSmarts(rewrite.lhs)
    rdkit_rhs = Chem.MolFromSmarts(rewrite.rhs)
    rdkit_total_num_atoms = rdkit_lhs.GetNumAtoms() + rdkit_rhs.GetNumAtoms()
    rdkit_total_num_bonds = rdkit_lhs.GetNumBonds() + rdkit_rhs.GetNumBonds()

    assert raw_graph.atom_features.shape == (rdkit_total_num_atoms, featurizer.num_atom_features)
    assert raw_graph.bond_features.shape == (len(raw_graph.bonds), featurizer.num_bond_features)
    assert raw_graph.bonds.shape == (2 * (rdkit_total_num_bonds + len(rewrite.mapping)), 2)
    assert raw_graph.node_in_lhs.shape == (rdkit_total_num_atoms,)
