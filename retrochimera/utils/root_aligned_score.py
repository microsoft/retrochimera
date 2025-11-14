"""Code adapted from RootAligned (https://github.com/otori-bird/retrosynthesis/blob/main/score.py).

Here it is modified to process predictions given as a tuple of (SMILES, probability).
"""

from rdkit import Chem


def canonicalize_smiles_clear_map(prediction, return_max_frag=True):
    """Convert a prediction tuple (SMILES, probability) into a canonicalized SMILES string.

    Args:
        prediction: Input tuple.
        return_max_frag: Whether to only return the SMILES of the maximum connected fragment.

    Returns:
        Canonicalized SMILES string, and (if `return_max_frag` is set) also the SMILES of the
        maximum connected fragment together with the probability.
    """
    smiles, probability = prediction
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        [
            atom.ClearProp("molAtomMapNumber")
            for atom in mol.GetAtoms()
            if atom.HasProp("molAtomMapNumber")
        ]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            if return_max_frag:
                return "", "", probability
            else:
                return ""
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [Chem.MolFromSmiles(smiles, sanitize=True) for smiles in sub_smi]
            sub_mol_size = [
                (sub_smi[i], len(m.GetAtoms())) for i, m in enumerate(sub_mol) if m is not None
            ]
            if len(sub_mol_size) > 0:
                return (
                    smi,
                    canonicalize_smiles_clear_map(
                        (sorted(sub_mol_size, key=lambda x: x[1], reverse=True)[0][0], probability),
                        return_max_frag=False,
                    ),
                    probability,
                )
            else:
                return smi, "", probability
        else:
            return smi
    else:
        if return_max_frag:
            return "", "", probability
        else:
            return ""


def compute_rank(
    prediction: list[list[tuple[str, str, float]]], alpha: float = 1.0, opt=None
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], list[tuple[int, float]]], int, int]:
    """Compute the rank of each prediction.

    Args:
        prediction: List of tuples (SMILES, max_frag_smiles, probability) for each augmentation.
        alpha: Scaling factor for rank calculation.

    Returns:
        A tuple of:
        - rank: Dictionary mapping (SMILES, max_frag_smiles) to scores.
        - position_prob_info: Dictionary mapping (SMILES, max_frag_smiles) to their relative rank
            (and probability) within each input augmentation's predictions.
        - adaptive_augmentation_size: Number of augmentations performed on the input molecule.
        - effective_augmentation_size: Number of augmentations where the model returned at least one
            valid prediction.
    """
    valid_score = [[k for k in range(len(prediction[j]))] for j in range(len(prediction))]
    rank: dict[tuple[str, str], float] = {}
    highest: dict[tuple[str, str], int] = {}
    position_prob_info: dict[tuple[str, str], list[tuple[int, float]]] = {}
    effective_augmentation_size = 0

    for j in range(len(prediction)):
        reaction_max_probability_map = {}
        for k in range(len(prediction[j])):
            if prediction[j][k][0] == "":
                valid_score[j][k] = opt.beam_size + 1
            if prediction[j][k][0:2] not in reaction_max_probability_map:
                reaction_max_probability_map[prediction[j][k][0:2]] = prediction[j][k][2]

        # error detection and deduplication
        de_error = [
            i[0][0:2]
            for i in sorted(list(zip(prediction[j], valid_score[j])), key=lambda x: x[1])
            if i[0][0] != ""
        ]
        cleaned_prediction = list(
            set(de_error)
        )  # cleaned means deduplicated, invalid smiles removed, sorted by probability
        cleaned_prediction.sort(key=de_error.index)
        if len(cleaned_prediction) != 0:
            effective_augmentation_size += 1
        for k, data in enumerate(cleaned_prediction):
            if data in rank:
                rank[data] += 1 / (alpha * k + 1)
                position_prob_info[data].append(
                    (k + 1, reaction_max_probability_map[data])
                )  # relative rank starting from 1, instead of 0
            else:
                rank[data] = 1 / (alpha * k + 1)
                position_prob_info[data] = [(k + 1, reaction_max_probability_map[data])]
            if data in highest:
                highest[data] = min(k, highest[data])
            else:
                highest[data] = k

    for key in rank.keys():
        rank[key] += highest[key] * -1e8

    adaptive_augmentation_size = len(prediction)

    return rank, position_prob_info, adaptive_augmentation_size, effective_augmentation_size
