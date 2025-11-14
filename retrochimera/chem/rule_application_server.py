import random
import weakref
from collections import deque
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Deque, Optional, Union

from more_itertools import divide
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.utils.misc import cpu_count, suppress_rdkit_outputs

from retrochimera.chem.rewrite import RewriteResult
from retrochimera.chem.rules import RuleBase, get_products
from retrochimera.utils.logging import get_logger

logger = get_logger(__name__)


def _worker_process(
    rulebase_dir: Union[str, Path], rule_ids: list[int], connection: Connection
) -> None:
    rulebase = RuleBase.load_from_file(dir=rulebase_dir, rule_ids=rule_ids)
    rule_application_kwargs = connection.recv()

    # Send a dummy signal when the rulebase has been loaded.
    connection.send(None)

    with suppress_rdkit_outputs():
        while True:
            request = connection.recv()

            if request is None:
                break
            else:
                smiles, rule_ids = request

                # SMILES comes from a `Molecule` object in the main process, so it is already canonical.
                mol = Molecule(smiles, canonicalize=False)

                for rule_id in rule_ids:
                    connection.send(
                        get_products(mol, rule=rulebase[rule_id].rxn, **rule_application_kwargs)
                    )

    connection.close()


class RuleApplicationServer:
    """Server that supports highly parallel application of reaction rules."""

    def __init__(
        self,
        rulebase_dir: Union[str, Path],
        num_processes: int = cpu_count(),
        rule_application_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the server.

        Args:
            rulebase_dir: Directory containing the rulebase file which will be accessed by the
                parallel processes. The constructor will block until all the processes have loaded
                up the rulebase and are ready to process requests. The rulebase file may be read
                again later if one of the processes hangs or fails and has to be restarted.
            num_processes: Number of parallel processes to spawn. The rules will be (roughly)
                equally divided between them.
            rule_application_kwargs: Extra arguments to pass through to `get_products`.
        """
        # First, *randomly* split the rules into `num_processes` groups to assign to each process.
        self.rule_ids = RuleBase.load_rule_ids_from_file(dir=rulebase_dir)

        rule_ids_shuffled = list(self.rule_ids)
        random.shuffle(rule_ids_shuffled)

        self._rule_id_to_process_id: dict[int, int] = {}
        self._rule_ids_per_process: list[list[int]] = [[] for _ in range(num_processes)]

        for process_id, rule_ids in enumerate(divide(num_processes, rule_ids_shuffled)):
            for rule_id in rule_ids:
                self._rule_id_to_process_id[rule_id] = process_id
                self._rule_ids_per_process[process_id].append(rule_id)

        self._rulebase_dir = rulebase_dir
        self._num_processes = num_processes
        self._rule_application_kwargs = rule_application_kwargs or {}

        self._running = True
        self._connections: dict[int, Connection] = {}
        self._processes: dict[int, Process] = {}

        self._start_processes(list(range(num_processes)))

        for process_id in range(num_processes):
            # Block until the process has loaded its chunk of the rulebase.
            self._connections[process_id].recv()

        # Register the `close` method to be called when the server object is garbage collected.
        weakref.finalize(self, self.close)

    def _start_processes(self, process_ids: list[int]) -> None:
        for process_id in process_ids:
            assert process_id not in self._processes
            assert process_id not in self._connections

            parent_conn, child_conn = Pipe()
            worker = Process(
                target=_worker_process,
                args=(self._rulebase_dir, self._rule_ids_per_process[process_id], child_conn),
            )

            worker.start()
            parent_conn.send(self._rule_application_kwargs)

            self._processes[process_id] = worker
            self._connections[process_id] = parent_conn

    def _connection_try_call(
        self, process_id: int, restart_on_failure: bool, fn_name: str, args: tuple
    ) -> Any:
        MAX_RETRIES = 5
        for _ in range(MAX_RETRIES):
            try:
                return getattr(self._connections[process_id], fn_name)(*args)
            except (ConnectionResetError, BrokenPipeError):
                if restart_on_failure:
                    logger.warning(f"Pipe to process {process_id} seems to be corrupted")
                    self._restart_process(process_id)
                else:
                    break

    def _connection_try_send(self, process_id: int, restart_on_failure: bool, data: Any) -> None:
        self._connection_try_call(
            process_id=process_id,
            restart_on_failure=restart_on_failure,
            fn_name="send",
            args=(data,),
        )

    def _connection_try_recv(self, process_id: int, restart_on_failure: bool) -> Any:
        return self._connection_try_call(
            process_id=process_id, restart_on_failure=restart_on_failure, fn_name="recv", args=()
        )

    def _connection_try_close(self, process_id: int) -> None:
        self._connection_try_call(
            process_id=process_id, restart_on_failure=False, fn_name="close", args=()
        )

    def _close_processes(self, process_ids: list[int]) -> None:
        # Filter out processes that are already closed.
        process_ids = [process_id for process_id in process_ids if process_id in self._processes]

        for process_id in process_ids:
            self._connection_try_send(process_id=process_id, restart_on_failure=False, data=None)

        for process_id in process_ids:
            # First wait to see if the process finishes gracefully; if not, SIGTERM it.
            self._processes[process_id].join(timeout=2.0)
            self._processes[process_id].terminate()

        for process_id in process_ids:
            self._processes[process_id].join(timeout=2.0)

        for process_id in process_ids:
            try:
                self._processes[process_id].close()
            except ValueError as e:
                logger.warning(
                    f"Error trying to close a rule application process {process_id}: {e}"
                )

            self._connection_try_close(process_id=process_id)

            del self._processes[process_id]
            del self._connections[process_id]

    def _close_all_processes(self) -> None:
        self._close_processes(list(range(self._num_processes)))

    def _restart_process(self, process_id: int) -> None:
        logger.info(f"Restarting process {process_id}")

        self._close_processes([process_id])
        self._start_processes([process_id])
        self._connections[process_id].recv()  # Wait for the process to load its rulebase.

    def close(self) -> None:
        if not self._running:
            return

        self._close_all_processes()
        self._running = False

    def __getstate__(self) -> dict[str, Any]:
        """Pickling processes is not supported, we have to kill them before pickling the server."""
        self._close_all_processes()
        return self.__dict__

    def apply_rules(
        self,
        inputs: list[Molecule],
        rule_ids_to_apply: list[list[int]],
        timeout: Optional[float] = 60.0,
    ) -> list[list[list[RewriteResult]]]:
        """Apply the given rules to the given inputs in parallel.

        Args:
            inputs: Molecules to apply the rules to.
            rule_ids_to_apply: Rule IDs to apply for each input.
            timeout: Maximum amount of time to wait for applying a single rule. If we cannot obtain
                the result within that time (which could be due to the underlying `rdkit` code
                taking too long to match the rule or any other failures in the child process) then
                no results are returned for that specific rule, as if it did not match.

        Returns:
            A nested list of results in a layout matching `rule_ids_to_apply`.
        """
        # shape: `[batch_size, num_processes, num_rules_to_apply, 2]`
        rules_to_apply_per_process: list[list[Deque[tuple[int, int]]]] = [
            [deque() for _ in range(self._num_processes)] for _ in inputs
        ]

        for input_idx, rule_ids in enumerate(rule_ids_to_apply):
            for idx, rule_id in enumerate(rule_ids):
                rules_to_apply_per_process[input_idx][self._rule_id_to_process_id[rule_id]].append(
                    (idx, rule_id)
                )

        def send_tasks_to_process(process_id: int) -> None:
            for input, rules in zip(inputs, rules_to_apply_per_process):
                if rules[process_id]:
                    self._connection_try_send(
                        process_id=process_id,
                        restart_on_failure=True,
                        data=(input.smiles, [rule_id for _, rule_id in rules[process_id]]),
                    )

        for process_id in range(self._num_processes):
            send_tasks_to_process(process_id)

        # shape: `[batch_size, num_rules_to_apply, num_results]`
        batch_results: list[list[list[RewriteResult]]] = []
        for input_idx, input in enumerate(inputs):
            results: list[list[RewriteResult]] = [[] for _ in rule_ids_to_apply[input_idx]]

            for process_id in range(self._num_processes):
                while rules_to_apply_per_process[input_idx][process_id]:
                    idx, rule_id = rules_to_apply_per_process[input_idx][process_id].popleft()

                    if self._connections[process_id].poll(timeout):
                        results[idx] = self._connections[process_id].recv()
                    else:
                        logger.warning(
                            f"Timed out waiting for process {process_id} for {timeout} seconds "
                            f"(input: {input.smiles}, rule ID: {rule_id})"
                        )

                        self._restart_process(process_id)

                        # Send the remaining tasks to the restarted process (note that completed
                        # tasks were already popped from `rules_to_apply_per_process`).
                        send_tasks_to_process(process_id)

            batch_results.append(results)

        return batch_results
