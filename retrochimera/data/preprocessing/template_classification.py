from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Generic, Iterable, Union

from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.utils.misc import parallelize

from retrochimera.chem.rules import RuleBase
from retrochimera.data.template_reaction_sample import TemplateReactionSample
from retrochimera.encoders.base import DataType, Encoder


def get_unique_template_id(sample: TemplateReactionSample) -> int:
    ground_truth_results = [
        result for result in sample.template_application_results if result.ground_truth
    ]

    if len(ground_truth_results) != 1:
        raise ValueError(f"Warning: incorrect number of ground-truth results in {sample}")

    return ground_truth_results[0].template_id


@dataclass
class ProcessedSample(Generic[DataType]):
    input: DataType  # Featurized input molecule
    target: int  # ID of the ground-truth rewrite


def _process_sample(
    sample: tuple[Molecule, int], input_preprocess_fn: Callable[[Molecule], DataType]
) -> ProcessedSample:
    input, target = sample
    return ProcessedSample(input=input_preprocess_fn(input), target=target)


def preprocess_samples(
    samples: Iterable[TemplateReactionSample],
    rulebase_dir: Union[str, Path],
    encoder: Encoder,
    num_processes: int,
) -> Iterable[ProcessedSample]:
    # Extract the relevant info from each sample.
    samples_processed = (
        (Molecule(smiles=sample.products_str), get_unique_template_id(sample)) for sample in samples
    )

    rule_ids = RuleBase.load_rule_ids_from_file(dir=rulebase_dir)

    # Get rid of labels which are not in the `rulebase` (e.g. they were too rare).
    samples_processed = (sample for sample in samples_processed if sample[-1] in rule_ids)

    yield from parallelize(
        partial(_process_sample, input_preprocess_fn=encoder.preprocess),
        samples_processed,
        num_processes=num_processes,
    )
