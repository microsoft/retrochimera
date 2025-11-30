import tempfile
from typing import Sequence

import pytest
from rdkit import Chem
from syntheseus import Bag, Molecule, SingleProductReaction
from syntheseus.reaction_prediction.chem.utils import molecule_bag_from_smiles

from retrochimera.chem.fixing import reverse_rxn
from retrochimera.chem.rules import RuleBase, RuleBasedRetrosynthesizer, RulePrediction


def check_result_against_target(result: Sequence[SingleProductReaction], target: str) -> None:
    predictions = {prediction.reactants for prediction in result}
    assert molecule_bag_from_smiles(target) in predictions


def check_results_against_targets(
    results: list[Sequence[SingleProductReaction]], targets: list[str]
) -> None:
    assert len(results) == len(targets)

    for result, target in zip(results, targets):
        check_result_against_target(result, target)


def test_check_results() -> None:
    # Verify that `check_result(s)_against_target(s)` functions trigger the asserts correctly.
    input = Molecule("CC")
    result_1 = [
        SingleProductReaction(product=input, reactants=Bag([Molecule(smiles)]))
        for smiles in ["c1ccccc1", "c1ccccc1N"]
    ]
    result_2 = [
        SingleProductReaction(product=input, reactants=Bag([Molecule(smiles)]))
        for smiles in ["c1ccccc1N", "c1ccccc1O"]
    ]
    target = "c1ccccc1O"

    # `result_1` does not have the correct prediction:
    with pytest.raises(AssertionError):
        check_result_against_target(result=result_1, target=target)

    # Number of results and targets do not match:
    with pytest.raises(AssertionError):
        check_results_against_targets(results=[result_2, result_2], targets=[target])

    # `result_2` has the correct prediction:
    check_result_against_target(result=result_2, target=target)

    # ...but if we check both results, then things fail:
    with pytest.raises(AssertionError):
        check_results_against_targets(results=[result_2, result_1], targets=[target, target])


def make_retrosynthesizer_from_rulebase(rulebase: RuleBase) -> RuleBasedRetrosynthesizer:
    with tempfile.TemporaryDirectory() as temp_dir:
        rulebase.save_to_file(dir=temp_dir)
        model = RuleBasedRetrosynthesizer(rulebase_dir=temp_dir)

    return model


def test_synth() -> None:
    sma = """([#6A&h0X3v4+0:20][#8A&h1X2v2+0:38].[#7A&h2X3v3+0:22])>>([#6A&h0X3v4+0:20][#7A&h1X3v3+0:22])
            ([#8A&h0X2v2+0:1][#6A&h3X4v4+0:16])>>([#8A&h1X2v2+0:1])
            ([#6A&h0X3v4+0:5][#8A&h1X2v2+0:39].[#7A&h1X3v3+0:7])>>([#6A&h0X3v4+0:5][#7A&h0X3v3+0:7])
            ([#7A&h0X3v4+:21](=[#8A&h0X&v2+0:25])[#8A&h0X&v-:26])>>([#7A&h2X3v3+0:21])
            ([#7A&h2X3v3+0:2].[#6ah0X3v4+0:3][ClA&h0X&v+0:10])>>([#7A&h1X3v3+0:2][#6ah0X3v4+0:3])
            ([#7A&h2X3v3+0:3].[#6A&h0X3v4+0:2][ClA&h0X&v+0:13])>>([#6A&h0X3v4+0:2][#7A&h1X3v3+0:3])
            ([#8A&h0X2v2+0:15][#6A&h2X4v4+0:17][#6A&h3X4v4+0:16])>>([#8A&h1X2v2+0:15])
            ([#6ah0X3v4+0:19][BrA&h0X&v+0:34].[#6ah0X3v4+0:20][#5A&h0X3v3+0:36]([#8A&h1X2v2+0:35])[#8A&h1X2v2+0:37])>>([#6ah0X3v4+0:19]-[#6ah0X3v4+0:20])
            ([#7A&h0X3v3+0:6][#6A&h0X3v4+0:14]([#8A&h0X2v2+0:13][#6A&h0X4v4+0:10]([#6A&h3X4v4+0:9])([#6A&h3X4v4+0:11])[#6A&h3X4v4+0:12])=[#8A&h0X&v2+0:15])>>([#7A&h1X3v3+0:6])
            ([#6A&h2X4v4+0:11][BrA&h0X&v+0:19].[#8A&h1X2v2+0:10])>>([#8A&h0X2v2+0:10][#6A&h2X4v4+0:11])
            """.split()

    rule_base = RuleBase()

    for i in sma:
        rule_base.add_rule(reverse_rxn(i))

    model = make_retrosynthesizer_from_rulebase(rule_base)

    input_smiles = [
        "O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1",
        "CC(C)n1nc(-c2ccc(O)cc2)c2cccc(Cl)c21",
        "CN(C)C(=O)[C@@H]1CCCN1C(=O)OCc1ccccc1",
        "Cc1ccc(C(C)(C)C)cc1N",
        "CC(C)(C)OC(=O)N1CCN(c2nc(-c3ccnc(NC4CCOCC4)c3)cc3cnccc23)CC1",
        "Cc1ccccc1C(=O)Nc1ccc(C(=O)N2Cc3ccccc3Sc3ncccc32)cc1",
        "O=C(O)CCc1c(/C=C2\\C(=O)Nc3ccccc32)[nH]c2c1C(=O)CCC2",
        "Cn1cnc(-c2cc(C#N)ccn2)c1-c1ccc(-n2cccn2)cc1",
        "C[C@H]1CNCCN1C1CCc2ccc(C(F)(F)F)cc21",
        "CCOP(=O)(COCc1nc2c(N)ncnc2n1CCc1ccccc1)OCC",
    ]
    input_mols = [Molecule(smiles) for smiles in input_smiles]
    results = model.predict(input_mols)

    assert len(results) == len(input_mols)

    # Check that the `input` is set correctly.
    for result, input_mol in zip(results, input_mols):
        for prediction in result:
            assert prediction.product == input_mol


def test_rediscovery() -> None:
    data = [
        ("CCN.O=C(O)c1cccs1>>CCNC(=O)c1cccs1", 645),
        ("CCCCCCN.NCCC(=O)O>>CCCCCCNC(=O)CCN", 645),
        ("CN.O=C(O)c1ccccc1I>>CNC(=O)c1ccccc1I", 645),
        ("Nc1ccccc1.O=C(O)CCl>>O=C(CCl)Nc1ccccc1", 645),
        ("CN.O=C(O)c1cc[nH]c1>>CNC(=O)c1cc[nH]c1", 645),
        ("N#CCC(=O)O.Nc1ccccc1>>N#CCC(=O)Nc1ccccc1", 645),
        ("CC(C)(C)N.O=C(O)CCBr>>CC(C)(C)NC(=O)CCBr", 645),
        ("COCCCCC(=O)OC>>COCCCCC(=O)O", 984),
        ("C=C(C)C(=O)OC>>C=C(C)C(=O)O", 984),
        ("COc1cccc(Cl)c1N>>Nc1c(O)cccc1Cl", 984),
        ("COc1ccc(S)cc1Cl>>Oc1ccc(S)cc1Cl", 984),
        ("COc1cnc(Cl)nc1C>>Cc1nc(Cl)ncc1O", 984),
        ("COc1ccc2ccsc2c1>>Oc1ccc2ccsc2c1", 984),
        ("COC(=O)Cc1ccccc1>>O=C(O)Cc1ccccc1", 984),
        ("O=[N+]([O-])c1ccn[nH]1>>Nc1ccn[nH]1", 319),
        ("COc1ncccc1[N+](=O)[O-]>>COc1ncccc1N", 319),
        ("CNc1ccccc1[N+](=O)[O-]>>CNc1ccccc1N", 319),
        ("O=Cc1ccccc1[N+](=O)[O-]>>Nc1ccccc1C=O", 319),
        ("CCCn1ccc([N+](=O)[O-])n1>>CCCn1ccc(N)n1", 319),
        ("Nc1ccc(F)cc1[N+](=O)[O-]>>Nc1ccc(F)cc1N", 319),
        ("O=[N+]([O-])c1ccn(CCO)n1>>Nc1ccn(CCO)n1", 319),
        ("CCCCBr.OCCCO>>CCCCOCCCO", 630),
        ("C#CCBr.COCCOCCO>>C#CCOCCOCCOC", 630),
        ("C#CCBr.OCC1CCCC1>>C#CCOCC1CCCC1", 630),
        ("CCCCBr.OCc1ccsc1>>CCCCOCc1ccsc1", 630),
        ("CCCCCCO.O=C(O)CBr>>CCCCCCOCC(=O)O", 630),
        ("O=C(O)CBr.OC1CCC1>>O=C(O)COC1CCC1", 630),
        ("BrCCCCBr.OC1CCCC1>>BrCCCCOC1CCCC1", 630),
    ]

    rules = {
        645: "([#6A&h0X3v4+0:20]-[#7A&h1X3v3+0:22])>>([#6A&h0X3v4+0:20]-[#8A&h1X2v2+0:38].[#7A&h2X3v3+0:22])",
        984: "([#8A&h1X2v2+0:1])>>([#8A&h0X2v2+0:1]-[#6A&h3X4v4+0:16])",
        319: "([#7A&h2X3v3+0:21])>>([#7A&h0X3v4+:21](=[#8A&h0X&v2+0:25])-[#8A&h0X&v-:26])",
        630: "([#8A&h0X2v2+0:10]-[#6A&h2X4v4+0:11])>>([#6A&h2X4v4+0:11]-[BrA&h0X&v+0:19].[#8A&h1X2v2+0:10])",
    }

    rule_base = RuleBase()

    for k, v in rules.items():
        rule_base.add_rule(v, rule_id=k)

    model = make_retrosynthesizer_from_rulebase(rule_base)

    input_smiles = [i[0].split(">>")[1] for i in data]
    targets = [Chem.CanonSmiles(i[0].split(">>")[0]) for i in data]

    results = model.predict([Molecule(smiles) for smiles in input_smiles])
    check_results_against_targets(results, targets)


def test_RuleBasedRetrosynthesizer() -> None:
    smiles = "COCCCCC(=O)O"
    target = "COCCCCC(=O)OC"

    rules = {
        645: "([#6A&h0X3v4+0:20]-[#7A&h1X3v3+0:22])>>([#6A&h0X3v4+0:20]-[#8A&h1X2v2+0:38].[#7A&h2X3v3+0:22])",
        984: "([#8A&h1X2v2+0:1])>>([#8A&h0X2v2+0:1]-[#6A&h3X4v4+0:16])",
        319: "([#7A&h2X3v3+0:21])>>([#7A&h0X3v4+:21](=[#8A&h0X&v2+0:25])-[#8A&h0X&v-:26])",
        630: "([#8A&h0X2v2+0:10]-[#6A&h2X4v4+0:11])>>([#6A&h2X4v4+0:11]-[BrA&h0X&v+0:19].[#8A&h1X2v2+0:10])",
    }

    rule_base = RuleBase()

    for k, v in rules.items():
        rule_base.add_rule(v, rule_id=k)

    model = make_retrosynthesizer_from_rulebase(rule_base)

    [result] = model.apply_top_rules(
        [Molecule(smiles, identifier=1)], batch_rules_to_apply=[[RulePrediction(id=984, prob=1.0)]]
    )

    rules_to_apply = model._predict_ranked_rules([Molecule(smiles, identifier=1)] * 7, top_k=4)
    assert len(rules_to_apply) == 7

    check_result_against_target(result, target)


def test_rdchiral_templates() -> None:
    # Here we check RDCHIRAL templates, taken from https://www.science.org/doi/10.1126/sciadv.abe4166

    data = [
        ("c1ccccc1[N+](=O)[O-]>>Nc1ccccc1"),
        ("CS(Cl)(=O)=O.OC1CCCC1>>CS(=O)(=O)OC1CCCC1"),
        ("OC(=O)C1=CC=CC=C1.C1CCNCC1>>O=C(N1CCCCC1)C1=CC=CC=C1"),
        ("CN1CCCC1C(=O)OC(C)(C)C>>CN1CCCC1C(O)=O"),
        ("ClC1=CC=CC=N1.FC1CCNC1>>FC1CCN(C1)C1=CC=CC=N1"),
    ]

    rules = {
        1: "([C:3]-[C:2](=[O;D1;H0:4])-[OH;D1;+0:1])>>(C-[O;H0;D2;+0:1]-[C:2](-[C:3])=[O;D1;H0:4])",
        2: "([NH2;D1;+0:1]-[c:2])>>(O=[N+;H0;D3:1](-[O-])-[c:2])",
        3: "([C:5]-[O;H0;D2;+0:6]-[S;H0;D4;+0:1](-[C;D1;H3:2])(=[O;D1;H0:3])=[O;D1;H0:4])>>(Cl-[S;H0;D4;+0:1](-[C;D1;H3:2])(=[O;D1;H0:3])=[O;D1;H0:4].[C:5]-[OH;D1;+0:6])",
        4: "([C:4]-[N;H0;D3;+0:5](-[C:6])-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3])>>(O-[C;H0;D3;+0:1](=[O;D1;H0:2])-[c:3].[C:4]-[NH;D2;+0:5]-[C:6])",
        5: "([C:3]-[C:2](=[O;D1;H0:4])-[OH;D1;+0:1])>>(C-C(-C)(-C)-[O;H0;D2;+0:1]-[C:2](-[C:3])=[O;D1;H0:4])",
        6: "([#7;a:2]:[c;H0;D3;+0:1](:[c:3])-[N;H0;D3;+0:5](-[C:4])-[C:6])>>(Cl-[c;H0;D3;+0:1](:[#7;a:2]):[c:3].[C:4]-[NH;D2;+0:5]-[C:6])",
    }

    rule_base = RuleBase()

    for k, v in rules.items():
        rule_base.add_rule(v, rule_id=k)

    model = make_retrosynthesizer_from_rulebase(rule_base)

    input_smiles = [i.split(">>")[1] for i in data]
    targets = [Chem.CanonSmiles(i.split(">>")[0]) for i in data]

    results = model.predict([Molecule(smiles) for smiles in input_smiles])
    check_results_against_targets(results, targets)
