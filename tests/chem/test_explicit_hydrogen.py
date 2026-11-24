from itertools import combinations

import pytest
from rdkit import Chem

from retrochimera.chem.explicit_hydrogen import hydrogen_inputs


def test_hydrogen_inputs_preserve_distinct_attachment_sites_below_match_cap() -> None:
    mol = Chem.MolFromSmiles("C" * 8)
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


def test_hydrogen_inputs_bound_raw_matches_before_deduplicating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_matches = Chem.Mol.GetSubstructMatches
    counts = []

    def tracked_matches(mol, *args, **kwargs):
        matches = get_matches(mol, *args, **kwargs)
        counts.append(len(matches))
        return matches

    monkeypatch.setattr(Chem.Mol, "GetSubstructMatches", tracked_matches)
    mol = Chem.MolFromSmiles("C" * 20)
    query = Chem.MolFromSmarts("[C]-[#1].[C].[C].[C]")
    assert list(hydrogen_inputs(mol, query))
    assert counts == [1000]
