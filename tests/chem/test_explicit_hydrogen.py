from itertools import combinations

from rdkit import Chem

from retrochimera.chem.explicit_hydrogen import hydrogen_inputs


def test_hydrogen_inputs_do_not_truncate_distinct_attachment_sites() -> None:
    mol = Chem.MolFromSmiles("C" * 24)
    query = Chem.MolFromSmarts("[C:1]-[#1:2].[C:3]-[#1:4]")
    sites = {
        tuple(
            sorted(
                atom.GetNeighbors()[0].GetIdx()
                for atom in graph.GetAtoms()
                if atom.GetAtomicNum() == 1
            )
        )
        for graph, _ in hydrogen_inputs(mol, query)
    }

    expected_pairs = set(combinations(range(mol.GetNumAtoms()), 2))
    assert not (expected_pairs - sites), (
        f"Missing attachment pairs: {sorted(expected_pairs - sites)}"
    )
