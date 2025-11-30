from pathlib import Path

from syntheseus import Molecule
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample

from retrochimera.data.processing import (
    AtomMappingProcessingStep,
    NumAtomsProcessingStep,
    NumReactantsProcessingStep,
    OneMainProductProcessingStep,
    ProcessingStep,
    ProductAmongReactantsProcessingStep,
)


def make_sample(reaction_smiles: str) -> ReactionSample:
    sample = ReactionSample.from_reaction_smiles_strict(reaction_smiles, mapped=True)
    sample.metadata["original_str"] = reaction_smiles  # type: ignore[typeddict-item]

    return sample


def run_step_and_compare(
    step: ProcessingStep, inputs: list[ReactionSample], expected_outputs: list[ReactionSample]
) -> None:
    outputs = list(step.process_samples(inputs))

    for output, expected_output in zip(outputs, expected_outputs):
        assert output.reactants == expected_output.reactants
        assert output.products == expected_output.products
        assert (
            output.metadata.get("repaired_str", output.metadata["original_str"])  # type: ignore[typeddict-item]
            == expected_output.metadata["original_str"]  # type: ignore[typeddict-item]
        )


def test_num_reactants_processing_step(tmpdir: Path) -> None:
    samples = [
        make_sample("[C:1]>>[C:1]"),
        make_sample("[C:1].[C:2]>>[C:1][C:2]"),
        make_sample("[C:1].[C:2].[C:3]>>[C:1][C:2][C:3]"),
    ]

    run_step_and_compare(
        step=NumReactantsProcessingStep(name="1", output_dir=tmpdir, max_reactants_num=2),
        inputs=samples,
        expected_outputs=samples[:2],
    )


def test_one_main_product_processing_step(tmpdir: Path) -> None:
    samples = [
        make_sample("[C:1]>>[C:1]"),
        make_sample("[C:1].[C:2]>>[C:1][C:2]"),
        make_sample("[C:1].[C:2]>>[C:1].[C:2]"),
        make_sample("[C:1].[C:2].[C:3]>>[C:1].[C:2][C:3]"),
        make_sample("[C:1].[C:2].[C:3]>>[C:1][C:2][C:3]"),
        make_sample("[C:1].[C:2].[C:3].[C:4]>>[C:1][C:2][C:3].[C:4]"),
        make_sample("[C:1].[C:2].[C:3].[C:4].[C:5]>>[C:1][C:2][C:3].[C:4][C:5]"),
    ]

    run_step_and_compare(
        step=OneMainProductProcessingStep(
            name="1",
            output_dir=tmpdir,
            min_product_atoms=2,
            common_products={Molecule("[C][C][C]")},
        ),
        inputs=samples,
        expected_outputs=[
            samples[1],
            make_sample("[C:1].[C:2].[C:3]>>[C:2][C:3]"),  # `samples[3]` repaired
            make_sample("[C:1].[C:2].[C:3].[C:4].[C:5]>>[C:4][C:5]"),  # `samples[6]` repaired
        ],
    )


def test_num_atoms_processing_step(tmpdir: Path) -> None:
    samples = [
        make_sample("[C:1].[C:2]>>[C:1][C:2]"),
        make_sample("[C:1].[C:2].[C:3]>>[C:1][C:2][C:3]"),
        make_sample("[C:1].[C:2].CN>>[C:1][C:2]"),
        make_sample("[C:1].[C:2].CN=O>>[C:1][C:2]"),
    ]

    run_step_and_compare(
        step=NumAtomsProcessingStep(
            name="1", output_dir=tmpdir, max_product_atoms=2, max_reactants_to_product_ratio=2
        ),
        inputs=samples,
        expected_outputs=[samples[0], samples[2]],
    )


def test_product_among_reactants_processing_step(tmpdir: Path) -> None:
    samples = [
        make_sample("[C:1].[C:2]>>[C:1][C:2]"),
        make_sample("[C:1][C:2]>>[C:1][C:2]"),
        make_sample("[C:1].[C:2].[C][C]>>[C:1][C:2]"),
    ]

    run_step_and_compare(
        step=ProductAmongReactantsProcessingStep(name="1", output_dir=tmpdir),
        inputs=samples,
        expected_outputs=samples[:1],
    )


def test_atom_mapping_processing_step(tmpdir: Path) -> None:
    samples = [
        make_sample("[C:1].[C:2]>>[C:1][C:2]"),
        make_sample("[C:1].[C:2].[C:3]>>[C:1][C:2]"),
        make_sample("[C:1].[C:2]>>[C:1][C:2][C:3]"),
        make_sample("CC>>CC"),
        make_sample("NC=O.[C:1]>>[C:1]"),
        make_sample("[C:1].[C:1]>>[C:1][C:2]"),
        make_sample("[C:1].[C:2]>>[C:1][C:1]"),
        make_sample("[C:1].[C:2]>>[C:1][C:3]"),
    ]

    run_step_and_compare(
        step=AtomMappingProcessingStep(name="1", output_dir=tmpdir),
        inputs=samples,
        expected_outputs=[
            samples[0],
            make_sample("[C:1].[C:2]>>[C:1][C:2]"),  # `samples[1]` repaired
            make_sample("[C:1].[C:2]>>[C][C:2][C:1]"),  # `samples[2]` repaired
            make_sample("[C:1]>>[C:1]"),  # `samples[4]` repaired
            make_sample("[C:1]>>[C][C:1]"),  # `samples[7]` repaired
        ],
    )
