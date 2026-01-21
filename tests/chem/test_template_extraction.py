import tempfile

from rdchiral import template_extractor
from syntheseus.interface.molecule import Molecule

from retrochimera.chem.rules import RuleBase, RuleBasedRetrosynthesizer
from retrochimera.cli.extract_templates import extract, extract_templates, process_reactions
from retrochimera.data.dataset import DataFold


def test_rdchiral1() -> None:
    rxn = {
        "reactants": "O=C(OCc1ccccc1)[NH:1][CH2:2][CH2:3][CH2:4][CH2:5][C@@H:6]([C:7]([O:8][CH3:9])=[O:10])[NH:11][C:12](=[O:13])[NH:14][c:15]1[cH:16][c:17]([O:18][CH3:19])[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[OH:27]",
        "products": "[NH2:1][CH2:2][CH2:3][CH2:4][CH2:5][C@@H:6]([C:7]([O:8][CH3:9])=[O:10])[NH:11][C:12](=[O:13])[NH:14][c:15]1[cH:16][c:17]([O:18][CH3:19])[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[OH:27]",
        "_id": 0,
    }
    out = template_extractor.extract_from_reaction(rxn)
    assert "reaction_smarts" in out
    assert "reaction_smarts_retro" in out


def test_rdchiral_fail() -> None:
    assert extract(rxn_smiles="NON>>SENSE", rxn_id=0, wrap_smarts=True) is None


def test_rdchiral_OK() -> None:
    out = extract(
        rxn_smiles="C[O:5][C:6]([c:7]1[c:8]([NH:14][S:15](=[O:16])(=[O:17])[c:18]2[cH:19][cH:20][cH:21][c:22]3[c:23]2[n:24][s:25][n:26]3)[cH:9][cH:10][c:11]([Cl:13])[cH:12]1)=[O:27]>>[O:5]=[C:6]([c:7]1[c:8]([NH:14][S:15](=[O:16])(=[O:17])[c:18]2[cH:19][cH:20][cH:21][c:22]3[c:23]2[n:24][s:25][n:26]3)[cH:9][cH:10][c:11]([Cl:13])[cH:12]1)[OH:27]",
        rxn_id=99,
        wrap_smarts=True,
    )
    assert out is not None
    assert out.template is not None


def test_get_templates_from_list() -> None:
    rxns = [
        "C[O:5][C:6]([c:7]1[c:8]([NH:14][S:15](=[O:16])(=[O:17])[c:18]2[cH:19][cH:20][cH:21][c:22]3[c:23]2[n:24][s:25][n:26]3)[cH:9][cH:10][c:11]([Cl:13])[cH:12]1)=[O:27]>>[O:5]=[C:6]([c:7]1[c:8]([NH:14][S:15](=[O:16])(=[O:17])[c:18]2[cH:19][cH:20][cH:21][c:22]3[c:23]2[n:24][s:25][n:26]3)[cH:9][cH:10][c:11]([Cl:13])[cH:12]1)[OH:27]",
        "Br[CH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[Br:9][c:10]1[c:11]([OH:16])[cH:12][cH:13][cH:14][cH:15]1>>[CH2:1]([c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1)[O:16][c:11]1[c:10]([Br:9])[cH:15][cH:14][cH:13][cH:12]1",
    ]
    assert len(extract_templates(rxns, wrap_smarts=True)) == 2


def test_process_reactions() -> None:
    rxns = [
        "C[O:5][C:6]([c:7]1[c:8]([NH:14][S:15](=[O:16])(=[O:17])[c:18]2[cH:19][cH:20][cH:21][c:22]3[c:23]2[n:24][s:25][n:26]3)[cH:9][cH:10][c:11]([Cl:13])[cH:12]1)=[O:27]>>[O:5]=[C:6]([c:7]1[c:8]([NH:14][S:15](=[O:16])(=[O:17])[c:18]2[cH:19][cH:20][cH:21][c:22]3[c:23]2[n:24][s:25][n:26]3)[cH:9][cH:10][c:11]([Cl:13])[cH:12]1)[OH:27]",
        "Br[CH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[Br:9][c:10]1[c:11]([OH:16])[cH:12][cH:13][cH:14][cH:15]1>>[CH2:1]([c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1)[O:16][c:11]1[c:10]([Br:9])[cH:15][cH:14][cH:13][cH:12]1",
    ]
    testrxn = [
        "F[c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1.[C:9]1(=[O:15])[CH2:10][CH2:11][CH2:12][CH2:13][CH2:14]1>>[c:3]1([C:9]2([OH:15])[CH2:10][CH2:11][CH2:12][CH2:13][CH2:14]2)[cH:4][cH:5][cH:6][cH:7][cH:8]1"
    ]
    result = process_reactions(
        reactions={DataFold.TRAIN: rxns, DataFold.VALIDATION: testrxn, DataFold.TEST: testrxn}
    ).samples

    def get_template_idx_from_first_sample(fold: DataFold) -> int:
        return list(result[fold])[0].template_application_results[0].template_id

    assert get_template_idx_from_first_sample(DataFold.TRAIN) in {0, 1}
    assert get_template_idx_from_first_sample(DataFold.VALIDATION) == -1
    assert get_template_idx_from_first_sample(DataFold.TEST) == -1


def test_filtering() -> None:
    rxns = [
        "C[O:5][C:6]([c:7]1[c:8]([NH:14][S:15](=[O:16])(=[O:17])[c:18]2[cH:19][cH:20][cH:21][c:22]3[c:23]2[n:24][s:25][n:26]3)[cH:9][cH:10][c:11]([Cl:13])[cH:12]1)=[O:27]>>[O:5]=[C:6]([c:7]1[c:8]([NH:14][S:15](=[O:16])(=[O:17])[c:18]2[cH:19][cH:20][cH:21][c:22]3[c:23]2[n:24][s:25][n:26]3)[cH:9][cH:10][c:11]([Cl:13])[cH:12]1)[OH:27]",
        "Br[CH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[Br:9][c:10]1[c:11]([OH:16])[cH:12][cH:13][cH:14][cH:15]1>>[CH2:1]([c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1)[O:16][c:11]1[c:10]([Br:9])[cH:15][cH:14][cH:13][cH:12]1",
        "Br[CH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[Br:9][c:10]1[c:11]([OH:16])[cH:12][cH:13][cH:14][cH:15]1>>[CH2:1]([c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1)[O:16][c:11]1[c:10]([Br:9])[cH:15][cH:14][cH:13][cH:12]1",
        "Br[CH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[Br:9][c:10]1[c:11]([OH:16])[cH:12][cH:13][cH:14][cH:15]1>>[CH2:1]([c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1)[O:16][c:11]1[c:10]([Br:9])[cH:15][cH:14][cH:13][cH:12]1",
        "F[c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1.[C:9]1(=[O:15])[CH2:10][CH2:11][CH2:12][CH2:13][CH2:14]1>>[c:3]1([C:9]2([OH:15])[CH2:10][CH2:11][CH2:12][CH2:13][CH2:14]2)[cH:4][cH:5][cH:6][cH:7][cH:8]1",
    ]

    out = process_reactions({DataFold.TRAIN: rxns}, min_template_occurrence=2)
    assert len(out.rulebase) == 1

    out = process_reactions({DataFold.TRAIN: rxns}, min_template_occurrence=0)
    assert len(out.rulebase) == 3

    with tempfile.TemporaryDirectory() as temp_dir:
        out.rulebase.save_to_file(dir=temp_dir)

        # Make sure saving and reloading gives us back the same rulebase.
        reloaded_rulebase = RuleBase.load_from_file(dir=temp_dir)

        assert len(reloaded_rulebase) == len(out.rulebase)
        for rule_id in reloaded_rulebase.rules:
            assert reloaded_rulebase[rule_id].smarts == out.rulebase[rule_id].smarts

        synth = RuleBasedRetrosynthesizer()
        synth.start_server(rulebase_dir=temp_dir)

    res = synth.predict([Molecule("BrC1=CC=CC=C1COC1=CC=CC=C1", identifier=0)])
    assert len(res) > 0
