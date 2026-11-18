import pytest
from rdkit import Chem
from syntheseus.interface.molecule import Molecule

from retrochimera.chem.rewrite import Rewrite, _molecule_bag_from_smiles


def test_cached_components_do_not_share_mutable_graphs_or_metadata() -> None:
    smiles = "C(O)[C@H](F)Cl"
    expected = Molecule(smiles)
    first = _molecule_bag_from_smiles(smiles)
    assert first is not None
    [original] = first
    original.metadata["probe"] = True
    original.rdkit_mol.GetAtomWithIdx(0).SetIsotope(13)
    for _ in range(2):
        fresh = _molecule_bag_from_smiles(smiles)
        assert fresh is not None
        [other] = fresh
        assert other == expected
        assert other.rdkit_mol.ToBinary() == expected.rdkit_mol.ToBinary()
        assert "probe" not in other.metadata


@pytest.mark.parametrize("return_localization", [True, False])
def test_rewrite_removes_deleted_atoms_only_when_localizing(return_localization: bool) -> None:
    rewrite = Rewrite.from_rdkit("[C:1]-[Cl:2]>>[C:1]")
    [result] = rewrite.apply(Molecule("CCCl"), return_localization=return_localization)
    assert [mol.smiles for mol in result.mols] == (["CC"] if return_localization else ["CC", "Cl"])
    assert result.metadata == ({"localizations": [[1, 2]]} if return_localization else {})


def test_rewrite_drops_stale_non_tetrahedral_stereo() -> None:
    rewrite = Rewrite.from_rdkit("[Cl:1]>>[Cl:1]")
    [result] = rewrite.apply(Molecule("F[Pt@SP1](Cl)(Br)I"))
    [product] = result.mols
    assert all(
        not atom.HasProp("_chiralPermutation") or atom.GetIntProp("_chiralPermutation") == 0
        for atom in product.rdkit_mol.GetAtoms()
    )


@pytest.mark.parametrize(
    "template, num_mapped",
    [
        ("[c:10]-[N&H2&+0&D1:2]>>[c:10]-[N&H0&+&D3:2](=O)-[O&-:5]", 2),
        ("[c:0]-[N&H2&+0&D1:2]>>[c:0]-[N&H1&+0&D2:2]-C(=O)-O-C(-C)(-C)-C", 2),
        ("[C:1]-[C&H2&+0&D2:33]-[O&H1&+0&D1:20]>>[C:1]-[C&H0&+0&D3:33](=[O&H0&+0&D1:20])-O-C-C", 3),
        ("[C:1]-[C&H1&+0&D2:3]=[O&H0&+0&D1:2]>>[C:1]-[C&H1&+0&D3:3]1-[O&H0&+0&D2:2]-C-C-O-1", 3),
    ],
)
def test_rewrite(template, num_mapped) -> None:
    rewrite = Rewrite.from_rdkit(template)

    def get_atoms_list(smarts: str):
        return list(Chem.MolFromSmarts(smarts).GetAtoms())

    atoms_lhs = get_atoms_list(rewrite.lhs)
    atoms_rhs = get_atoms_list(rewrite.rhs)

    assert len(rewrite.mapping) == num_mapped

    # Sanity check: atom mapping preserves the atom type.
    for idx_lhs, idx_rhs in rewrite.mapping:
        assert atoms_lhs[idx_lhs].GetAtomicNum() == atoms_rhs[idx_rhs].GetAtomicNum()

    # Matches all of the tested templates exactly once.
    input_mol = Molecule("Nc1ccc(CCO)c(CC=O)c1")
    results = rewrite.apply(input_mol)

    assert len(results) == 1
    [result] = results

    assert len(result.metadata["localizations"]) == 1

    [localization] = result.metadata["localizations"]
    assert len(localization) == num_mapped

    # Sanity check: localization is consistent with atom type.
    atoms_input = list(input_mol.rdkit_mol.GetAtoms())
    for idx_lhs, idx_input in enumerate(localization):
        assert atoms_lhs[idx_lhs].GetAtomicNum() == atoms_input[idx_input].GetAtomicNum()

    # Matches the templates exactly twice, but with the same result.
    input_mol = Molecule("Nc1ccc(CCO)c(CC=O)c1.Nc1ccc(CCO)c(CC=O)c1")
    results = rewrite.apply(input_mol)

    assert len(results) == 1

    for result in results:
        assert len(result.metadata["localizations"]) == 2

    # Matches the templates exactly twice, with different result.
    input_mol = Molecule("Nc1ccc(CCO)c(CC=O)c1.Nc1c(Cl)cc(CCO)c(CC=O)c1")
    results = rewrite.apply(input_mol)

    assert len(results) == 2

    for result in results:
        assert len(result.metadata["localizations"]) == 1

    # Run on a molecule that shouldn't match any of the tested tempaltes.
    assert not rewrite.apply(Molecule("NC=O"))
