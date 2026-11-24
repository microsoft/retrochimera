from typing import Optional

import pytest
from rdkit import Chem
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.chem.utils import molecule_bag_from_smiles

from retrochimera.chem.rewrite import Rewrite


@pytest.mark.parametrize("return_localization", [True, False])
@pytest.mark.parametrize(
    "product,expected",
    [
        ("CCO", "CC=O"),
        ("[2H][C@H](F)CCO", "[2H][C@H](F)CC=O"),
        ("OCCO", "O=CCO"),
        ("COC", None),
    ],
)
def test_rewrite_implicit_hydrogens(
    product: str, expected: Optional[str], return_localization: bool
) -> None:
    target = Molecule(product)
    original = Chem.MolToSmiles(target.rdkit_mol)
    rule = Rewrite.from_rdkit("([C:1](-[#1&^0:2])-[O:3])>>([C:1]=[O:3].[#1:2]-[#1])")
    results = rule.apply(target, return_localization=return_localization)
    assert Chem.MolToSmiles(target.rdkit_mol) == original
    assert [r.mols for r in results] == (
        [molecule_bag_from_smiles(expected + ".[H][H]")] if expected else []
    )
    for result in results:
        assert bool(result.metadata) == return_localization
        if return_localization and product == "OCCO":
            assert {loc[0] for loc in result.metadata["localizations"]} == {1, 2}
        for localization in result.metadata.get("localizations", []):
            assert None in localization
            assert [a.GetAtomicNum() for a in rule.rdkit_lhs_mol.GetAtoms()] == [
                target.rdkit_mol.GetAtomWithIdx(i).GetAtomicNum() if i is not None else 1
                for i in localization
            ]


def test_rewrite_implicit_hydrogens_with_degree_constraint() -> None:
    rule = Rewrite.from_rdkit("([C:1](-[#1&^0:2])-[O&D1:3])>>([C:1]=[O:3].[#1:2]-[#1])")
    results = rule.apply(Molecule("CCO"))
    assert [result.mols for result in results] == [molecule_bag_from_smiles("CC=O.[H][H]")]


@pytest.mark.parametrize("product", ["F[C@H](Cl)Br", "F[C@@H](Cl)Br", "FC(Cl)Br"])
def test_rewrite_implicit_hydrogens_follow_reaction_matcher_chirality(product: str) -> None:
    rule = Rewrite.from_rdkit(
        "([C@:1](-[F:2])(-[Cl:3])(-[Br:4])-[#1:5])>>([C@:1](-[F:2])(-[Cl:3])(-[Br:4])-[#1:5])"
    )
    target = Molecule(product)
    assert not rule.rdkit_reaction.GetSubstructParams().useChirality
    assert rule.rdkit_reaction.RunReactants((Chem.AddHs(target.rdkit_mol),))

    results = rule.apply(target)
    assert [result.mols for result in results] == [molecule_bag_from_smiles(product)]


@pytest.mark.parametrize("product", ["CC", "CCCl"])
def test_rewrite_implicit_hydrogens_in_atom_or_query(product: str) -> None:
    rule = Rewrite.from_rdkit("([C:1]-[#1,#17:2])>>([C:1]-[*:2])")
    results = rule.apply(Molecule(product))
    assert [result.mols for result in results] == [molecule_bag_from_smiles(product)]


@pytest.mark.parametrize(
    "template,product,expected",
    [
        (
            "([C:1](-[#1&^0&z0:2])-[O:3])>>([C:1]=[O:3].[#1:2]-[#1])",
            "CCO",
            "CC=O.[H][H]",
        ),
        (
            "([N:1](-[#1&^0&z1:2])-[#1&^0&z1:3])>>([N+:1](=O)-[O-].[#1:2]-[#1:3])",
            "CN",
            "C[N+](=O)[O-].[H][H]",
        ),
    ],
)
def test_rewrite_detects_hydrogens_with_neighbor_count_predicates(
    template: str, product: str, expected: str
) -> None:
    [result] = Rewrite.from_rdkit(template).apply(Molecule(product))
    assert result.mols == molecule_bag_from_smiles(expected)


@pytest.mark.parametrize(
    "template,product,mismatch,localizations",
    [
        ("[C;$([C](-[#1])-[#1]):1]>>[C:1]", "CC", "FC(F)(F)C(F)(F)F", [[0], [1]]),
        ("[C;$([C]-[O]-[#1]):1]>>[C:1]", "CCO", "COC", [[1]]),
        ("[C;$([C]-[O;$([O]-[#1])]):1]>>[C:1]", "CCO", "COC", [[1]]),
        (
            "([C;$([C]-[O]-[#1]):1]-[#1:2])>>([C:1]-[#1:2])",
            "CO",
            "COC",
            [[0, None]],
        ),
    ],
)
def test_rewrite_matches_recursive_hydrogen_predicates(
    template: str, product: str, mismatch: str, localizations: list[list[Optional[int]]]
) -> None:
    rule = Rewrite.from_rdkit(template)
    target = Molecule(product)
    [result] = rule.apply(target)
    assert result.mols == molecule_bag_from_smiles(product)
    assert result.metadata["localizations"] == localizations
    assert not rule.apply(Molecule(mismatch))


def test_rewrite_deduplicates_hydrogen_expanded_localizations() -> None:
    rule = Rewrite.from_rdkit("([C:1]-[#1,#17:2])>>([C:1]-[*:2])")
    target = Molecule("CCCl")
    [result] = rule.apply(target)
    localizations = result.metadata["localizations"]
    assert len(localizations) == 3
    assert len({tuple(localization) for localization in localizations}) == 3
    assert localizations.count([1, 2]) == 1


@pytest.mark.parametrize("return_localization", [True, False])
@pytest.mark.parametrize(
    "template,product,expected",
    [
        ("[N+:1]>>[N+:1]", "CC[NH3+].[Br-]", "CC[NH3+].[Br-]"),
        ("([N+:1].[Br-:2])>>([N+:1].[Br-:2])", "CC[NH3+].[Br-]", "CC[NH3+].[Br-]"),
        ("[Ca+2:1]>>[Ca+2:1]", "[Ca+2].[Cl-].[Cl-]", "[Ca+2].[Cl-].[Cl-]"),
        ("[N&H3+:1]>>[N&H2&+0:1]", "CC[NH3+].[Br-]", "CCN.[Br-]"),
        ("[N+:1]>>[N+:1]", "CC[NH3+].CC(=O)[O-]", "CC[NH3+].CC(=O)[O-]"),
        ("[N+:1]>>[N+:1]", "CC[NH3+].[Br-].[Br-]", "CC[NH3+].[Br-].[Br-]"),
        (
            "[N+:1]>>[N+:1]",
            "CC[NH3+].CC[NH3+].[Br-].[Br-]",
            "CC[NH3+].CC[NH3+].[Br-].[Br-]",
        ),
        ("[N+:1]>>[N+:1]", "C[NH3+].[Na+].[Cl-].[Cl-]", "C[NH3+].[Na+].[Cl-].[Cl-]"),
        ("[Na+:1]>>[Na+:1]", "[Na+].[C-]#N", "[Na+].[C-]#N"),
        ("[O-:1]>>[O-:1]", "CC(=O)[O-].C[N+](C)(C)C", "CC(=O)[O-].C[N+](C)(C)C"),
        ("[N+:1]>>[N+:1]", "CC[NH3+].[Br-].CCO", "CC[NH3+].[Br-]"),
    ],
)
def test_rewrite_carries_all_unmatched_counterions(
    template: str, product: str, expected: str, return_localization: bool
) -> None:
    [result] = Rewrite.from_rdkit(template).apply(
        Molecule(product), return_localization=return_localization
    )
    assert result.mols == molecule_bag_from_smiles(expected)
    assert bool(result.metadata) == return_localization


def test_rewrite_restores_charge_balancing_spectator_cation() -> None:
    product = "CC(=O)[O-].[Na+]"
    [result] = Rewrite.from_rdkit("[O-:1]>>[O-:1]").apply(Molecule(product))
    assert result.mols == molecule_bag_from_smiles(product)


@pytest.mark.parametrize(
    "template,product,expected",
    [
        (
            "[N&H3+:1]>>[N&H2&+0:1]",
            "[NH3+]CC[NH3+].[Cl-].[Cl-]",
            "NCC[NH3+].[Cl-].[Cl-]",
        ),
        (
            "[N&H2+:1]>>[N&H1&+0:1]",
            "[NH3+]CC[NH2+]CC[NH3+].[Cl-].O=S(=O)([O-])[O-]",
            "[NH3+]CCNCC[NH3+].[Cl-].O=S(=O)([O-])[O-]",
        ),
    ],
)
def test_rewrite_keeps_all_counterions_when_template_changes_charge(
    template: str, product: str, expected: str
) -> None:
    [result] = Rewrite.from_rdkit(template).apply(Molecule(product))
    assert result.mols == molecule_bag_from_smiles(expected)


@pytest.mark.parametrize(
    "product,expected",
    [
        ("C[NH3+].[Cl-]", "C[NH3+]"),
        ("C[NH3+].[Cl-].[Cl-]", "C[NH3+].[Cl-]"),
    ],
)
def test_rewrite_does_not_restore_counterion_deleted_by_template(
    product: str, expected: str
) -> None:
    rule = Rewrite.from_rdkit("([N+:1].[Cl-:2])>>[N+:1]")
    [result] = rule.apply(Molecule(product), return_localization=True)
    assert result.mols == molecule_bag_from_smiles(expected)


@pytest.mark.parametrize("return_localization", [True, False])
def test_rewrite_does_not_copy_fragment_matched_through_synthetic_hydrogen(
    return_localization: bool,
) -> None:
    product = "C[NH3+].[Cl-]"
    [result] = Rewrite.from_rdkit("[H:1]>>[H:1]").apply(
        Molecule(product), return_localization=return_localization
    )
    assert result.mols == molecule_bag_from_smiles(product)
    if return_localization:
        assert all(localization == [None] for localization in result.metadata["localizations"])


def test_rewrite_can_include_unrestored_outcomes_for_training() -> None:
    target = Molecule("C[NH3+].[Cl-]")
    rule = Rewrite.from_rdkit("[N+:1]>>[N+:1]")
    results = rule.apply(target, include_unrestored=True)
    assert {result.mols for result in results} == {
        molecule_bag_from_smiles("C[NH3+]"),
        molecule_bag_from_smiles("C[NH3+].[Cl-]"),
    }
    assert all(result.metadata["localizations"] == [[1]] for result in results)


@pytest.mark.parametrize(
    "template,product",
    [
        ("[Ca+2:1]>>[Ca+2:1]", "[Ca+2].O=C([O-])[O-]"),
        ("[N+:1]>>[N+:1]", "C[NH3+].O=C(O)[O-]"),
    ],
)
def test_rewrite_restores_inorganic_carbon_containing_counterions(
    template: str, product: str
) -> None:
    [result] = Rewrite.from_rdkit(template).apply(Molecule(product))
    assert result.mols == molecule_bag_from_smiles(product)


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
