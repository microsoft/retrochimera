from typing import Optional

import pytest
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample

from retrochimera.chem.rules import RuleBase
from retrochimera.data.dataset import DataFold, DiskReactionDataset, ReactionDataset
from retrochimera.data.template_reaction_sample import (
    TemplateApplicationResult,
    TemplateReactionSample,
)


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("include_templates", [False, True])
def test_save_and_load(tmp_path: str, mapped: bool, include_templates: bool) -> None:
    sample_cls: type[ReactionSample]
    if include_templates:
        sample_cls = TemplateReactionSample
        sample_kwargs = {"template_application_results": [TemplateApplicationResult(template_id=0)]}
    else:
        sample_cls = ReactionSample
        sample_kwargs = {}

    samples = [
        sample_cls.from_reaction_smiles_strict(reaction_smiles, mapped=mapped, **sample_kwargs)
        for reaction_smiles in [
            "O[c:1]1[cH:2][c:3](=[O:4])[nH:5][cH:6][cH:7]1>>[cH:1]1[cH:2][c:3](=[O:4])[nH:5][cH:6][cH:7]1",
            "CC(C)(C)OC(=O)[N:1]1[CH2:2][CH2:3][C@H:4]([F:5])[CH2:6]1>>[NH:1]1[CH2:2][CH2:3][C@H:4]([F:5])[CH2:6]1",
        ]
    ]

    for fold in DataFold:
        DiskReactionDataset.save_samples_to_file(data_dir=tmp_path, fold=fold, samples=samples)  # type: ignore

    if include_templates:
        rulebase = RuleBase()
        rulebase.add_rule("[BrH;D0;+0:1]>>C-C(=O)-[Br;H0;D1;+0:1]")
        rulebase.add_rule("[C:2]-[CH3;D1;+0:1]>>O-[CH2;D2;+0:1]-[C:2]")
        rulebase.save_to_file(dir=tmp_path)

    # Now try to load the data we just saved.
    dataset = DiskReactionDataset(tmp_path, sample_cls=sample_cls)

    for fold in DataFold:
        assert list(dataset[fold]) == samples

    if include_templates:
        assert dataset.rulebase.rules == rulebase.rules
    else:
        with pytest.raises(FileNotFoundError):
            dataset.rulebase


@pytest.mark.parametrize("min_rule_support", [None, 3, 5])
@pytest.mark.parametrize("max_num_rules", [None, 2, 5])
def test_limit_rulebase(
    reaction_dataset_uneven: ReactionDataset,
    tmp_path: str,
    min_rule_support: Optional[int],
    max_num_rules: Optional[int],
) -> None:
    DiskReactionDataset.save_samples_to_file(
        data_dir=tmp_path, fold=DataFold.TRAIN, samples=reaction_dataset_uneven[DataFold.TRAIN]
    )
    reaction_dataset_uneven.rulebase.save_to_file(dir=tmp_path)

    reaction_dataset_limited = DiskReactionDataset(
        data_dir=tmp_path,
        sample_cls=TemplateReactionSample,
        rulebase_min_rule_support=min_rule_support,
        rulebase_max_num_rules=max_num_rules,
    )

    expected_num_rules = 4
    if min_rule_support is not None and min_rule_support > 4:
        expected_num_rules = min(expected_num_rules, 8 - min_rule_support)
    if max_num_rules is not None:
        expected_num_rules = min(expected_num_rules, max_num_rules)

    assert len(reaction_dataset_limited.rulebase) == expected_num_rules

    samples_limited = list(reaction_dataset_limited[DataFold.TRAIN])
    assert len(samples_limited) == reaction_dataset_uneven.get_num_samples(DataFold.TRAIN)
