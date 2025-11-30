import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from retrochimera.chem.utils import get_atom_mapping, normalize_atom_mapping


def test_get_atom_mapping() -> None:
    mol = Chem.MolFromSmarts("[C:1]-[C&H0&+0&D3:5](=[O&H0&D1:4])-O.[C:2]-[N&H2&+0&D1:3]")
    atom_mapping = get_atom_mapping(mol)

    assert atom_mapping == {1: 0, 2: 4, 3: 5, 4: 2, 5: 1}


@pytest.mark.parametrize(
    "smarts, smarts_normalized",
    [
        ("[c:10]-[N:2]>>[c:10]-[N:2](=O)-[O&-:5]", "[c:2]-[N:1]>>[c:2]-[N:1](=O)-[O&-]"),
        ("[#6:3]-[#7:28]>>[#8:12]-[#6:3].[#7:28]", "[#6:1]-[#7:2]>>[#8]-[#6:1].[#7:2]"),
    ],
)
def test_normalize_mapping(smarts: str, smarts_normalized: str) -> None:
    template = AllChem.ReactionFromSmarts(smarts)
    normalize_atom_mapping(template)

    assert AllChem.ReactionToSmarts(template) == smarts_normalized
