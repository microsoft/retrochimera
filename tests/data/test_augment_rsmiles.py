import copy
import json
from collections import Counter
from pathlib import Path

import pytest
from rdkit import Chem
from syntheseus.reaction_prediction.data.dataset import DataFold

from retrochimera.cli.augment_rsmiles import AugmentRSMILESConfig, multi_process, run_from_config

# fmt: off
REACTIONS = [
    (
        "[CH2:1]=[CH:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1.[H:9][H:10]",
        "[CH2:1]([H:9])[CH:2]([H:10])[c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1",
        "[CH3:1][CH2:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1",
    ),
    (
        "[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1.[Na+:10].[B-:11]([H:12])([H:13])([H:14])[H:15]",
        "[CH3:1][C:2]([H:12])([O:3][H:16])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1",
        "[CH3:1][CH:2]([OH:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1",
    ),
]
# fmt: on


def components(smiles: str) -> Counter[str]:
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Counter(Chem.MolToSmiles(frag) for frag in Chem.GetMolFrags(mol, asMols=True))


def sample(reactants: str, product: str, **kwargs) -> dict:
    return {
        "reaction": {"mapped_reaction_smiles": f"{reactants}>>{product}"},
        "augmentation": 3,
        "use_random_smiles": False,
        "seed": 0,
        **kwargs,
    }


@pytest.mark.parametrize("reactants,explicit_product,implicit_product", REACTIONS)
@pytest.mark.parametrize("use_random_smiles", [False, True])
@pytest.mark.parametrize("keep_all_reactants", [False, True])
def test_keep_all_reactants(
    reactants: str,
    explicit_product: str,
    implicit_product: str,
    use_random_smiles: bool,
    keep_all_reactants: bool,
) -> None:
    result = multi_process(
        sample(
            reactants,
            implicit_product,
            use_random_smiles=use_random_smiles,
            keep_all_reactants=keep_all_reactants,
        )
    )
    expected_reactants = reactants if keep_all_reactants else reactants.split(".")[0]
    assert result["status"] == 0
    assert len(result["src_smi"]) == len(result["tgt_smi"]) == 3
    for source, target in zip(result["src_smi"], result["tgt_smi"]):
        assert components(source) == components(implicit_product)
        assert components(target) == components(expected_reactants)


@pytest.mark.parametrize("reactants,explicit_product,implicit_product", REACTIONS)
def test_keep_all_reactants_ignores_removed_product_h_roots(
    reactants: str, explicit_product: str, implicit_product: str
) -> None:
    data = sample(reactants, explicit_product, keep_all_reactants=True)
    original = copy.deepcopy(data)
    explicit = multi_process(data)
    implicit = multi_process(sample(reactants, implicit_product, keep_all_reactants=True))

    assert explicit["status"] == implicit["status"] == 0
    assert explicit["src_smi"] == implicit["src_smi"]
    assert explicit["tgt_smi"] == implicit["tgt_smi"]
    assert data == original


@pytest.mark.parametrize("use_random_smiles", [False, True])
def test_keep_all_reactants_preserves_duplicates(use_random_smiles: bool) -> None:
    # fmt: off
    reactants = "[CH3:1][CH2:2][OH:3].CCO.CCO.[Na+]"
    product = "[CH3:1][CH2:2][OH:3]"
    # fmt: on
    result = multi_process(
        sample(
            reactants,
            product,
            use_random_smiles=use_random_smiles,
            keep_all_reactants=True,
        )
    )
    assert result["status"] == 0
    assert len(result["tgt_smi"]) == 3
    for target in result["tgt_smi"]:
        assert components(target) == components(reactants)


@pytest.mark.parametrize("use_random_smiles", [False, True])
def test_keep_all_reactants_sorts_previously_dropped_components(use_random_smiles: bool) -> None:
    # fmt: off
    reactants = "[Na+].[BH4-].OCC.[CH3:1][CH2:2][OH:3].[H][H].CCO"
    product = "[CH3:1][CH2:2][OH:3]"
    # fmt: on
    result = multi_process(
        sample(
            reactants,
            product,
            use_random_smiles=use_random_smiles,
            keep_all_reactants=True,
        )
    )
    assert result["status"] == 0
    assert len(result["tgt_smi"]) == 3
    for target in result["tgt_smi"]:
        ordered_components = [
            Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in target.split(".")
        ]
        assert ordered_components == ["CCO", "CCO", "CCO", "[BH4-]", "[H][H]", "[Na+]"]
    assert result["tgt_smi"][0] == "CCO.CCO.CCO.[BH4-].[H][H].[Na+]"


def test_keep_all_reactants_preserves_existing_alignment() -> None:
    # fmt: off
    reactants = "Br[CH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[OH:8][CH3:9]"
    product = "[CH2:1]([c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1)[O:8][CH3:9]"
    # fmt: on
    legacy = multi_process(sample(reactants, product))
    enabled = multi_process(sample(reactants, product, keep_all_reactants=True))
    assert legacy["status"] == 0
    assert enabled == legacy


def test_keep_all_reactants_config_reaches_saved_samples(tmp_path: Path) -> None:
    reactants, product, _ = REACTIONS[0]
    data_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    reaction = {"mapped_reaction_smiles": f"{reactants}>>{product}"}
    (data_dir / "train.jsonl").write_text(json.dumps(reaction) + "\n")

    run_from_config(
        AugmentRSMILESConfig(
            data_dir=str(data_dir),
            augmented_data_dir=str(output_dir),
            folds=[DataFold.TRAIN],
            augmentation=3,
            num_processes=1,
            keep_all_reactants=True,
        )
    )

    saved = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
    assert len(saved) == 3
    for result in saved:
        assert result["mapped_reaction_smiles"] == reaction["mapped_reaction_smiles"]
        assert components(".".join(mol["smiles"] for mol in result["reactants"])) == components(
            reactants
        )
