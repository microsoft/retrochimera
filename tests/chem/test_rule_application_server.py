import random
import tempfile

import pytest
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.utils.misc import set_random_seed

from retrochimera.chem.rule_application_server import RuleApplicationServer
from retrochimera.chem.rules import get_products
from retrochimera.data.dataset import DataFold, ReactionDataset


@pytest.mark.parametrize("num_inputs", [1, 16])
@pytest.mark.parametrize("num_processes", list(range(1, 6)))
def test_apply_rules(reaction_dataset: ReactionDataset, num_inputs: int, num_processes: int):
    # Set a deterministic seed, but make it vary between tests for better coverage.
    set_random_seed(hash((num_inputs, num_processes)) % (2**30))

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the rulebase and let the processes load it. The file will be deleted when we exit the
        # context, but once the server constructor returns the file would only be needed if rule
        # application processes need to be restarted (which shouldn't happen here).
        reaction_dataset.rulebase.save_to_file(dir=temp_dir)
        server = RuleApplicationServer(rulebase_dir=temp_dir, num_processes=num_processes)

    # Take a few random products from the dataset.
    samples: list[ReactionSample] = random.sample(
        list(reaction_dataset[DataFold.TRAIN]), num_inputs
    )
    inputs = [list(sample.products)[0] for sample in samples]

    # Pair the inputs with a random number of randomly chosen rule IDs.
    rule_ids_to_apply = [
        [random.randint(0, 3) for _ in range(random.randint(2, 6))] for _ in inputs
    ]

    # Apply the rules in parallel.
    results_parallel = server.apply_rules(inputs=inputs, rule_ids_to_apply=rule_ids_to_apply)

    assert len(results_parallel) == len(inputs)

    for input, rule_ids, results in zip(inputs, rule_ids_to_apply, results_parallel):
        # Apply the same rules naively and check that the outcome is the same.
        results_naive = []
        for rule_id in rule_ids:
            results_naive.extend(get_products(input, rule=reaction_dataset.rulebase[rule_id].rxn))

        assert sum(results, []) == results_naive
