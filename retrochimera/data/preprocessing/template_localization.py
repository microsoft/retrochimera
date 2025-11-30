import itertools
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np
from more_itertools import batched
from numpy.typing import NDArray
from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.utils.misc import parallelize

from retrochimera.chem.rewrite import RewriteResult
from retrochimera.chem.rule_application_server import RuleApplicationServer
from retrochimera.data.preprocessing.template_classification import get_unique_template_id
from retrochimera.data.template_reaction_sample import TemplateReactionSample
from retrochimera.encoders.base import Encoder
from retrochimera.encoders.featurizers import RawGraph


@dataclass
class ProcessedSample:
    input: RawGraph  # Input molecule featurized into a graph
    target: int  # ID of the ground-truth rewrite
    loc_target: NDArray  # Superposition of all correct matchings between rewrite lhs and input


def _process_sample(
    sample: tuple[Molecule, Bag[Molecule], int, list[RewriteResult]],
    input_preprocess_fn: Callable[[Molecule], RawGraph],
) -> Optional[tuple[RawGraph, int, list[int]]]:
    input, reactants, target, results = sample

    # Filter down to localizations that give the correct output.
    results = [result for result in results if result.mols == reactants]

    if not results:
        # TODO(krmaziar): This can happen due to chirality differences, possibly because of
        # issues with `rdchiral` and/or `RunReactions`. We skip these samples for now, but
        # should find a better solution later.
        return None

    assert len(results) == 1

    [result] = results
    localizations = result.metadata["localizations"]

    loc_target = np.zeros((len(localizations[0]), input.rdkit_mol.GetNumAtoms()), dtype=np.float32)
    for localization in localizations:
        for idx_rewrite, idx_input in enumerate(localization):
            loc_target[idx_rewrite][idx_input] += 1.0

    loc_target /= loc_target.sum(axis=1, keepdims=True)

    return input_preprocess_fn(input), target, loc_target


def preprocess_samples(
    samples: Iterable[TemplateReactionSample],
    rulebase_dir: Union[str, Path],
    input_encoder: Encoder,
    num_processes: int,
) -> Iterable[ProcessedSample]:
    # Extract the relevant info from each sample.
    samples_processed: Iterable = (
        (
            Molecule(smiles=sample.products_str),
            sample.reactants,
            get_unique_template_id(sample),
        )
        for sample in samples
    )

    server = RuleApplicationServer(
        rulebase_dir=rulebase_dir,
        num_processes=num_processes,
        rule_application_kwargs={"return_localization": True},
    )

    # Get rid of labels which are not in the `rulebase` (e.g. they were too rare).
    samples_processed = (sample for sample in samples_processed if sample[-1] in server.rule_ids)

    # Rerun templates to determine ground-truth localizations.
    samples_processed = itertools.chain.from_iterable(
        zip(
            samples_batch,
            server.apply_rules(
                inputs=[product for product, _, _, in samples_batch],
                rule_ids_to_apply=[[target] for _, _, target in samples_batch],
            ),
        )
        for samples_batch in batched(samples_processed, 4 * num_processes)
    )

    # Strip away superfluous nesting.
    samples_processed = (
        (input, reactants, target, results)
        for (input, reactants, target), [results] in samples_processed
    )

    samples_processed = parallelize(
        partial(_process_sample, input_preprocess_fn=input_encoder.preprocess),
        samples_processed,
        num_processes=num_processes,
    )

    for sample in samples_processed:
        # Skip samples where the rerunning did not produce the expected result.
        if sample is not None:
            yield ProcessedSample(*sample)
