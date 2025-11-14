import pytest
from rdkit import Chem
from syntheseus.interface.molecule import Molecule

from retrochimera.chem.rewrite import Rewrite


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
