import dataclasses
from collections import Counter
from dataclasses import dataclass
from hashlib import md5
from typing import Any, Iterable, Optional, cast

import joblib
from rdchiral import template_extractor
from syntheseus.reaction_prediction.chem.utils import (
    molecule_bag_from_smiles_strict,
    remove_atom_mapping,
)
from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config
from syntheseus.reaction_prediction.utils.misc import cpu_count, set_random_seed

from retrochimera.chem.rules import RuleBase
from retrochimera.data.dataset import DataFold, DiskReactionDataset
from retrochimera.data.template_reaction_sample import (
    TemplateApplicationResult,
    TemplateReactionSample,
)
from retrochimera.data.utils import load_raw_reactions_files
from retrochimera.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractTemplatesConfig:
    """Config for extracting templates from a dataset."""

    data_dir: str  # Directory containing raw data.
    output_dir: str  # Directory to write output to.
    min_freq: int = 0  # Minimum template frequency in training data for inclusion into the library.
    num_processes: int = cpu_count()  # Number of parallel processes to use.


def _convert(rxn_smiles: str, rxn_id: int) -> dict[str, Any]:
    """Convert into rdchiral format.

    Args:
        rxn_smiles: Mapped reaction SMILES.
        rxn_id: ID of the reaction.

    Returns:
        Dict of the form {"reactants": reactants, "products": prod, "_id": rxn_id}.
    """

    reactants, _, products = rxn_smiles.split(">")
    return {"reactants": reactants, "products": products, "_id": rxn_id}


def extract(rxn_smiles: str, rxn_id: int, wrap_smarts: bool) -> Optional[TemplateReactionSample]:
    # For Pistachio we may have extra info after the reaction SMILES, so get rid of it first.
    rxn_smiles = rxn_smiles.split("\t")[0].split(" ")[0]
    rdch = _convert(rxn_smiles, rxn_id)

    try:
        template = template_extractor.extract_from_reaction(rdch)["reaction_smarts"]

        if wrap_smarts:
            reactants, conditions, products = template.split(">")
            template = f"({reactants})>{conditions}>({products})"

        return TemplateReactionSample(
            reactants=molecule_bag_from_smiles_strict(remove_atom_mapping(rdch["reactants"])),
            products=molecule_bag_from_smiles_strict(remove_atom_mapping(rdch["products"])),
            mapped_reaction_smiles=rxn_smiles,
            template=template,
        )
    except Exception:
        return None


def extract_templates(
    rxn_smiles_list: list[str], wrap_smarts: bool, num_processes: int = cpu_count()
) -> list[TemplateReactionSample]:
    pool = joblib.Parallel(n_jobs=num_processes)
    jobs = (joblib.delayed(extract)(s, i, wrap_smarts) for i, s in enumerate(rxn_smiles_list))

    return [result for result in pool(jobs) if result is not None]


@dataclass
class LabeledMolsAndTemplates:
    samples: dict[DataFold, Iterable[TemplateReactionSample]]
    rulebase: RuleBase


def rulebase_from_templates_list(
    templates: list[str], min_template_occurrence: int = 0
) -> tuple[RuleBase, dict[str, int]]:
    # Count how often templates appear
    counter = Counter(templates)

    rulebase = RuleBase()
    for template, count in counter.most_common(len(counter)):
        if count >= min_template_occurrence:
            rulebase.add_rule(
                smarts=template, rule_hash=md5(template.encode()).hexdigest(), n_support=count
            )

    return rulebase, {rule.smarts: id for id, rule in rulebase.rules.items()}


def process_reactions(
    reactions: dict[DataFold, list[str]],
    min_template_occurrence: int = 0,
    wrap_smarts: bool = True,
    num_processes: int = cpu_count(),
) -> LabeledMolsAndTemplates:
    """Workflow for processing datasets of reactions.
       Performs
        1) template assignment on training set, and builds template library & index
        2) extracts products and assigns template labels to train, test, val datasets.
           Reactions not covered by the template library derived from the training data will receive a label of -1

    Args:
        reactions: Mapping from data fold (train, validation and test) to a list of reaction SMILES.
        min_template_occurrence: How often a template needs to appear in training data to receive a
            label. To reduce the number of false positives, it's often beneficial to increase this
            number.
        wrap_smarts: Whether to transform "reactants>conditions>products" into
            "(reactants)>conditions>(products)".
        num_processes: Number of parallel processes to use.

    Returns:
        Dataclass containing the labelled (mol, template) data, and the template library.
    """
    logger.info("Extracting templates from training set")
    processed_train_samples = extract_templates(
        reactions[DataFold.TRAIN], wrap_smarts=wrap_smarts, num_processes=num_processes
    )

    logger.info("Building the template library")
    rulebase, template_idx = rulebase_from_templates_list(
        templates=[cast(str, sample.template) for sample in processed_train_samples],
        min_template_occurrence=min_template_occurrence,
    )

    logger.info(f"Built template library with size {len(rulebase)}")

    samples = {}
    for fold in reactions:
        logger.info(f"Processing fold {fold}")

        if fold == DataFold.TRAIN:
            processed_samples = processed_train_samples
        else:
            processed_samples = extract_templates(
                reactions[fold], wrap_smarts=wrap_smarts, num_processes=num_processes
            )

        # Samples that use rare templates will be assigned a label of `-1`.
        samples[fold] = label(processed_samples, template_idx=template_idx)
        logger.info(f"{fold}: Original {len(reactions[fold])} samples")
        logger.info(f"{fold}: Processed {len(processed_samples)} samples")

    return LabeledMolsAndTemplates(samples=samples, rulebase=rulebase)


def label(
    samples: Iterable[TemplateReactionSample], template_idx: dict[str, int], rare_idx: int = -1
) -> Iterable[TemplateReactionSample]:
    """Attach template IDs to processed samples.

    Args:
        samples: Samples processed with `extract_templates`.
        template_idx: Template index i.e. mapping from SMARTS to ID.
        rare_idx: Label to assign when a template is not covered by the index.

    Returns:
        Iterable over data samples with template IDs attached.
    """
    for sample in samples:
        yield dataclasses.replace(
            sample,
            template=None,
            template_application_results=[
                TemplateApplicationResult(
                    template_id=template_idx.get(cast(str, sample.template), rare_idx)
                )
            ],
        )


def process_smiles(
    input_dir: str, output_dir: str, min_template_occurrence: int, num_processes: int
) -> None:
    logger.info(
        f"Params: {input_dir} {output_dir}, min template frequency: {min_template_occurrence}"
    )

    # Start by testing the `rdchiral` installation.
    rdchiral_result = template_extractor.extract_from_reaction(_convert("[C:1]>>[C:1]C", 0))
    if "reaction_smarts" not in rdchiral_result:
        raise RuntimeError("The rdchiral installation is not working.")
    elif "reaction_smarts_retro" not in rdchiral_result:
        logger.warning(
            "You seem to be using the original (Python) version of rdchiral. If data processing "
            "takes a long time, consider using the faster C++ version (rdchiral_cpp)."
        )

    output = process_reactions(
        reactions=load_raw_reactions_files(input_dir),
        min_template_occurrence=min_template_occurrence,
        num_processes=num_processes,
    )

    logger.info("Done processing, saving rulebase")
    output.rulebase.save_to_file(dir=output_dir)

    for fold, samples in output.samples.items():
        logger.info(f"Saving samples into fold {fold}")
        DiskReactionDataset.save_samples_to_file(data_dir=output_dir, fold=fold, samples=samples)  # type: ignore


def run_from_config(config: ExtractTemplatesConfig) -> None:
    set_random_seed(0)
    process_smiles(
        config.data_dir,
        config.output_dir,
        min_template_occurrence=config.min_freq,
        num_processes=config.num_processes,
    )


def main(argv: Optional[list[str]]) -> None:
    config: ExtractTemplatesConfig = cli_get_config(argv=argv, config_cls=ExtractTemplatesConfig)
    run_from_config(config)


if __name__ == "__main__":
    main(argv=None)
