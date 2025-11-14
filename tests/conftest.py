import os

import pytest

from retrochimera.chem.rules import RuleBase
from retrochimera.data.dataset import DataFold, InMemoryReactionDataset, ReactionDataset
from retrochimera.data.template_reaction_sample import (
    TemplateApplicationResult,
    TemplateReactionSample,
)


def pytest_configure(config):
    # Set env var to disable wandb during tests (using `wandb.init` would not work for subprocesses)
    os.environ["WANDB_MODE"] = "disabled"


@pytest.fixture
def tiny_rulebase() -> RuleBase:
    rulebase = RuleBase()
    for smarts in [
        "[C:1][C:2][C:3][C:4]>>([C:1][C:2].[C:3][C:4])",
        "[C:1][C:2][C:3][C:4]>>([C:1][C:2][C:3].[C:4])",
    ]:
        rulebase.add_rule(smarts=smarts)

    return rulebase


@pytest.fixture
def rulebase() -> RuleBase:
    rulebase = RuleBase()
    for smarts in [
        "([#6A&h0X3v4+0:20]-[#7A&h1X3v3+0:22])>>([#6A&h0X3v4+0:20]-[#8A&h1X2v2+0:38].[#7A&h2X3v3+0:22])",
        "([#8A&h1X2v2+0:1])>>([#8A&h0X2v2+0:1]-[#6A&h3X4v4+0:16])",
        "([#7A&h2X3v3+0:21])>>([#7A&h0X3v4+:21](=[#8A&h0X&v2+0:25])-[#8A&h0X&v-:26])",
        "([#8A&h0X2v2+0:10]-[#6A&h2X4v4+0:11])>>([#6A&h2X4v4+0:11]-[BrA&h0X&v+0:19].[#8A&h1X2v2+0:10])",
    ]:
        # Value for `n_support` is based on `raw_reaction_data` below.
        rulebase.add_rule(smarts=smarts, n_support=7)

    return rulebase


@pytest.fixture
def rulebase_uneven() -> RuleBase:
    rulebase = RuleBase()
    for idx, smarts in enumerate(
        [
            "([#6A&h0X3v4+0:20]-[#7A&h1X3v3+0:22])>>([#6A&h0X3v4+0:20]-[#8A&h1X2v2+0:38].[#7A&h2X3v3+0:22])",
            "([#8A&h1X2v2+0:1])>>([#8A&h0X2v2+0:1]-[#6A&h3X4v4+0:16])",
            "([#7A&h2X3v3+0:21])>>([#7A&h0X3v4+:21](=[#8A&h0X&v2+0:25])-[#8A&h0X&v-:26])",
            "([#8A&h0X2v2+0:10]-[#6A&h2X4v4+0:11])>>([#6A&h2X4v4+0:11]-[BrA&h0X&v+0:19].[#8A&h1X2v2+0:10])",
        ]
    ):
        # Value for `n_support` is based on `raw_reaction_data_uneven` below.
        rulebase.add_rule(smarts=smarts, n_support=7 - idx)

    return rulebase


@pytest.fixture
def raw_reaction_data() -> list[tuple[str, int]]:
    return [
        ("CCN.O=C(O)c1cccs1>>CCNC(=O)c1cccs1", 0),
        ("CCCCCCN.NCCC(=O)O>>CCCCCCNC(=O)CCN", 0),
        ("CN.O=C(O)c1ccccc1I>>CNC(=O)c1ccccc1I", 0),
        ("Nc1ccccc1.O=C(O)CCl>>O=C(CCl)Nc1ccccc1", 0),
        ("CN.O=C(O)c1cc[nH]c1>>CNC(=O)c1cc[nH]c1", 0),
        ("N#CCC(=O)O.Nc1ccccc1>>N#CCC(=O)Nc1ccccc1", 0),
        ("CC(C)(C)N.O=C(O)CCBr>>CC(C)(C)NC(=O)CCBr", 0),
        ("COCCCCC(=O)OC>>COCCCCC(=O)O", 1),
        ("C=C(C)C(=O)OC>>C=C(C)C(=O)O", 1),
        ("COc1cccc(Cl)c1N>>Nc1c(O)cccc1Cl", 1),
        ("COc1ccc(S)cc1Cl>>Oc1ccc(S)cc1Cl", 1),
        ("COc1cnc(Cl)nc1C>>Cc1nc(Cl)ncc1O", 1),
        ("COc1ccc2ccsc2c1>>Oc1ccc2ccsc2c1", 1),
        ("COC(=O)Cc1ccccc1>>O=C(O)Cc1ccccc1", 1),
        ("O=[N+]([O-])c1ccn[nH]1>>Nc1ccn[nH]1", 2),
        ("COc1ncccc1[N+](=O)[O-]>>COc1ncccc1N", 2),
        ("CNc1ccccc1[N+](=O)[O-]>>CNc1ccccc1N", 2),
        ("O=Cc1ccccc1[N+](=O)[O-]>>Nc1ccccc1C=O", 2),
        ("CCCn1ccc([N+](=O)[O-])n1>>CCCn1ccc(N)n1", 2),
        ("Nc1ccc(F)cc1[N+](=O)[O-]>>Nc1ccc(F)cc1N", 2),
        ("O=[N+]([O-])c1ccn(CCO)n1>>Nc1ccn(CCO)n1", 2),
        ("CCCCBr.OCCCO>>CCCCOCCCO", 3),
        ("C#CCBr.COCCOCCO>>C#CCOCCOCCOC", 3),
        ("C#CCBr.OCC1CCCC1>>C#CCOCC1CCCC1", 3),
        ("CCCCBr.OCc1ccsc1>>CCCCOCc1ccsc1", 3),
        ("CCCCCCO.O=C(O)CBr>>CCCCCCOCC(=O)O", 3),
        ("O=C(O)CBr.OC1CCC1>>O=C(O)COC1CCC1", 3),
        ("BrCCCCBr.OC1CCCC1>>BrCCCCOC1CCCC1", 3),
    ]


@pytest.fixture
def raw_reaction_data_uneven(raw_reaction_data: list[tuple[str, int]]) -> list[tuple[str, int]]:
    rule_to_num_samples_left = [7, 6, 5, 4]

    data = []
    for reaction_smiles, target in raw_reaction_data:
        if rule_to_num_samples_left[target] > 0:
            rule_to_num_samples_left[target] -= 1
            data.append((reaction_smiles, target))

    return data


def build_reaction_datset(raw_data: list[tuple[str, int]], rulebase: RuleBase) -> ReactionDataset:
    data = []
    for reaction_smiles, target in raw_data:
        data.append(
            TemplateReactionSample.from_reaction_smiles_strict(
                reaction_smiles=reaction_smiles,
                mapped=False,
                template_application_results=[TemplateApplicationResult(template_id=target)],
            )
        )

    return InMemoryReactionDataset(samples={fold: data for fold in DataFold}, rulebase=rulebase)


@pytest.fixture
def reaction_dataset(
    raw_reaction_data: list[tuple[str, int]], rulebase: RuleBase
) -> ReactionDataset:
    return build_reaction_datset(raw_reaction_data, rulebase)


@pytest.fixture
def reaction_dataset_uneven(
    raw_reaction_data_uneven: list[tuple[str, int]], rulebase_uneven: RuleBase
) -> ReactionDataset:
    return build_reaction_datset(raw_reaction_data_uneven, rulebase_uneven)
