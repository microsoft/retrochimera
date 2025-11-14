from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Union

from syntheseus.reaction_prediction.utils.misc import parallelize

from retrochimera.data.preprocessing.template_classification import get_unique_template_id
from retrochimera.data.smiles_reaction_sample import SmilesReactionSample
from retrochimera.data.smiles_tokenizer import Tokenizer


@dataclass
class ProcessedSample:
    encoded_products: list[int]  # For pistachio and uspto50k, samples only have one main product
    encoded_reactants: list[int]
    template_id: int  # ID of the ground-truth template


def _process_sample(
    sample: tuple[str, str, int],
    input_preprocess_fn,
    output_preprocess_fn,
) -> ProcessedSample:
    products_smi, reactants_smi, template_id = sample
    encoded_products, _ = input_preprocess_fn([products_smi])
    encoded_reactants, _ = output_preprocess_fn([reactants_smi])
    return ProcessedSample(
        encoded_products=encoded_products[0],
        encoded_reactants=encoded_reactants[0],
        template_id=template_id,
    )


def preprocess_samples(
    samples: Iterable[SmilesReactionSample],
    rulebase_dir: Union[str, Path],
    tokenizer: Tokenizer,
    num_processes: int,
) -> Iterable[ProcessedSample]:
    # Extract (product smiles, reactants smiles, template id) from each sample.
    samples_processed = (
        (sample.raw_products_smiles, sample.raw_reactants_smiles, get_unique_template_id(sample))
        for sample in samples
    )

    # Preprocess the input and output smiles strings.
    input_preprocess_fn = partial(
        tokenizer.encode, pad=False, right_padding=True, add_begin_token=False, add_end_token=False
    )
    output_preprocess_fn = partial(
        tokenizer.encode, pad=False, right_padding=True, add_begin_token=True, add_end_token=True
    )

    yield from parallelize(
        partial(
            _process_sample,
            input_preprocess_fn=input_preprocess_fn,
            output_preprocess_fn=output_preprocess_fn,
        ),
        samples_processed,
        num_processes=num_processes,
    )
