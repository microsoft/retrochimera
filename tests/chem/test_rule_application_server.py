import gc
import os
import random
import signal
import tempfile
import time
from multiprocessing import Event, Pipe, Process, Value, active_children, get_start_method

import pytest
from syntheseus import Molecule
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.utils.misc import set_random_seed

import retrochimera.chem.rule_application_server as server_module
from retrochimera.chem.rule_application_server import RuleApplicationServer
from retrochimera.chem.rules import RuleBase, get_products
from retrochimera.data.dataset import DataFold, ReactionDataset


@pytest.mark.skipif(get_start_method() != "fork", reason="Injects a worker failure using fork")
@pytest.mark.parametrize("failure", ["exit", "hang"])
def test_worker_failure_preserves_completed_and_remaining_results(monkeypatch, tmp_path, failure):
    original = server_module.get_products
    calls = Value("i", 0)

    def fail_second_application(*args, **kwargs):
        calls.value += 1
        if calls.value == 2:
            if failure == "exit":
                os._exit(7)
            Event().wait()
        return original(*args, **kwargs)

    monkeypatch.setattr(server_module, "get_products", fail_second_application)
    rulebase = RuleBase()
    rulebase.add_rule("[C:1]>>[C:1]O", rule_id=0)
    inputs = [Molecule("CC"), Molecule("CCC")]
    expected = [original(mol, rule=rulebase[0].rxn) for mol in inputs]
    rulebase.save_to_file(dir=tmp_path)
    server = RuleApplicationServer(rulebase_dir=tmp_path, num_processes=1)
    try:
        results = server.apply_rules(
            inputs, [[0, 0], [0]], timeout=None if failure == "exit" else 0.2
        )
        assert results == [[expected[0], []], [expected[1]]]
        assert server.apply_rules(inputs[:1], [[0]]) == [[expected[0]]]
    finally:
        server.close()


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


def test_processes_cleaned_up_on_gc(tiny_rulebase):
    """Verify that worker processes are cleaned up when the server is garbage collected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tiny_rulebase.save_to_file(dir=temp_dir)
        server = RuleApplicationServer(rulebase_dir=temp_dir, num_processes=1)

    workers_before = set(active_children())
    assert len(workers_before) >= 1

    del server
    gc.collect()

    assert not workers_before & set(active_children())


def _ignore_sigterm(ready) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()

    while True:
        time.sleep(1.0)


def test_processes_cleaned_up_when_sigterm_is_ignored():
    """Verify that cleanup escalates when a worker does not respond to SIGTERM."""
    parent_connection, child_connection = Pipe()
    ready = Event()

    worker = Process(target=_ignore_sigterm, args=(ready,))
    worker.start()
    child_connection.close()

    try:
        assert ready.wait(timeout=5.0)
        RuleApplicationServer._finalize_server({0: worker}, {0: parent_connection})

        # Check that the worker was killed and removed from the active children
        assert worker not in active_children()
    finally:
        if worker in active_children():
            # If we got here, the test failed; clean it up to avoid leaving a stray process running
            worker.kill()
            worker.join(timeout=2.0)
            worker.close()

        parent_connection.close()
