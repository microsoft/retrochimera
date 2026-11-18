import os
import random
import selectors
import weakref
from collections import deque
from contextlib import ExitStack
from itertools import groupby
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection, wait
from pathlib import Path
from time import monotonic
from typing import Any, Deque, Optional, Union

from more_itertools import divide
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.utils.misc import cpu_count, suppress_rdkit_outputs

from retrochimera.chem.rewrite import RewriteResult, _cached_component
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
            # Bound graph lifetimes to one batch, including on worker shutdown.
            _cached_component.cache_clear()

            if request is None:
                break

            for smiles, rule_ids in request:
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

        # Register cleanup to run when this server object is garbage collected. We cannot pass a
        # bound method (`self.close`) as it contains a reference to `self` which would prevent GC.
        weakref.finalize(
            self, RuleApplicationServer._finalize_server, self._processes, self._connections
        )

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
        else:
            raise RuntimeError(
                f"Could not {fn_name} to process {process_id} after {MAX_RETRIES} tries"
            )

    def _connection_try_send(self, process_id: int, restart_on_failure: bool, data: Any) -> None:
        self._connection_try_call(
            process_id=process_id,
            restart_on_failure=restart_on_failure,
            fn_name="send",
            args=(data,),
        )

    @staticmethod
    def _finalize_server(processes: dict[int, Process], connections: dict[int, Connection]) -> None:
        """Clean up worker processes. Must not reference `self` to allow garbage collection."""
        for process_id in processes:
            try:
                connections[process_id].send(None)
            except (ConnectionResetError, BrokenPipeError, OSError):
                pass

        for process_id in processes:
            # First wait to see if the process finishes gracefully; if not, SIGTERM it.
            processes[process_id].join(timeout=2.0)

            if processes[process_id].is_alive():
                processes[process_id].terminate()

        for process_id in processes:
            processes[process_id].join(timeout=2.0)

            if processes[process_id].is_alive():
                logger.warning(
                    f"Rule application process {process_id} ignored SIGTERM; sending SIGKILL"
                )
                processes[process_id].kill()

        for process_id in processes:
            processes[process_id].join(timeout=5.0)

        for process_id in processes:
            try:
                processes[process_id].close()
            except ValueError as e:
                logger.warning(
                    f"Error trying to close a rule application process {process_id}: {e}"
                )

            try:
                connections[process_id].close()
            except Exception:
                pass

    def _close_processes(self, process_ids: list[int]) -> None:
        process_ids = [process_id for process_id in process_ids if process_id in self._processes]
        if not process_ids:
            return

        # Extract into separate dicts and delegate to the static cleanup method.
        sub_processes = {pid: self._processes.pop(pid) for pid in process_ids}
        sub_connections = {pid: self._connections.pop(pid) for pid in process_ids}
        RuleApplicationServer._finalize_server(sub_processes, sub_connections)

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
            timeout: Maximum amount of time to wait for the next rule result from each worker.
                `None` disables timeouts. If we cannot obtain
                the result within that time (which could be due to the underlying `rdkit` code
                taking too long to match the rule or any other failures in the child process) then
                no results are returned for that specific rule, as if it did not match. The worker
                is restarted and its remaining rules are resubmitted.

        Returns:
            A nested list of results in a layout matching `rule_ids_to_apply`.
        """
        rules_to_apply_per_process: list[Deque[tuple[int, int, int]]] = [
            deque() for _ in range(self._num_processes)
        ]

        for input_idx, rule_ids in enumerate(rule_ids_to_apply):
            for idx, rule_id in enumerate(rule_ids):
                rules_to_apply_per_process[self._rule_id_to_process_id[rule_id]].append(
                    (input_idx, idx, rule_id)
                )

        def send_tasks_to_process(process_id: int) -> None:
            requests = [
                (inputs[input_idx].smiles, [rule_id for _, _, rule_id in rules])
                for input_idx, rules in groupby(
                    rules_to_apply_per_process[process_id], key=lambda entry: entry[0]
                )
            ]
            if requests:
                # Send one message so the worker finishes reading before replying; otherwise both
                # directions of the pipe can fill and deadlock.
                self._connection_try_send(
                    process_id=process_id, restart_on_failure=True, data=requests
                )

        for process_id in range(self._num_processes):
            send_tasks_to_process(process_id)

        # shape: `[batch_size, num_rules_to_apply, num_results]`
        batch_results: list[list[list[RewriteResult]]] = [
            [[] for _ in rule_ids] for rule_ids in rule_ids_to_apply
        ]
        with ExitStack() as stack:
            # Windows pipes/process handles need multiprocessing's wait implementation.
            selector = stack.enter_context(selectors.DefaultSelector()) if os.name != "nt" else None
            handles: dict[Any, tuple[int, Connection]] = {}
            last_progress: dict[int, float] = {}

            def register(process_id: int) -> None:
                if not rules_to_apply_per_process[process_id]:
                    return
                connection = self._connections[process_id]
                for handle in (connection, self._processes[process_id].sentinel):
                    handles[handle] = (process_id, connection)
                    if selector is not None:
                        selector.register(handle, selectors.EVENT_READ, (process_id, connection))
                last_progress[process_id] = monotonic()

            def unregister(process_id: int) -> None:
                for handle in (
                    self._connections[process_id],
                    self._processes[process_id].sentinel,
                ):
                    del handles[handle]
                    if selector is not None:
                        selector.unregister(handle)
                del last_progress[process_id]

            def restart(process_id: int, timed_out: bool) -> None:
                paused = monotonic()
                input_idx, _, rule_id = rules_to_apply_per_process[process_id].popleft()
                if timed_out:
                    logger.warning(
                        f"Timed out waiting for process {process_id} for {timeout} seconds "
                        f"(input: {inputs[input_idx].smiles}, rule ID: {rule_id})"
                    )
                else:
                    logger.warning(
                        f"Process {process_id} failed while applying rule ID {rule_id} "
                        f"(input: {inputs[input_idx].smiles})"
                    )
                unregister(process_id)
                self._restart_process(process_id)
                send_tasks_to_process(process_id)
                register(process_id)
                # Healthy workers cannot be drained while a failed worker reloads its rulebase.
                elapsed = monotonic() - paused
                for peer in last_progress:
                    if peer != process_id:
                        last_progress[peer] += elapsed

            for process_id in range(self._num_processes):
                register(process_id)

            while handles:
                wait_time = (
                    None
                    if timeout is None
                    else max(0.0, min(last_progress.values()) + timeout - monotonic())
                )
                ready: list[tuple[Any, tuple[int, Connection]]]
                if selector is None:
                    ready = [
                        (handle, handles[handle]) for handle in wait(handles, timeout=wait_time)
                    ]
                else:
                    ready = [(key.fileobj, key.data) for key, _ in selector.select(wait_time)]
                handled = set()
                for handle, (process_id, connection) in ready:
                    # Ignore stale events after a restart: file descriptors can be reused,
                    # but the replacement worker has a different connection object.
                    if (
                        not rules_to_apply_per_process[process_id]
                        or self._connections[process_id] is not connection
                    ):
                        continue
                    handled.add(process_id)
                    try:
                        if handle is not connection:
                            # Drain buffered replies before treating a worker exit as a failure.
                            if connection.poll(0):
                                continue
                            raise EOFError("Worker exited")
                        result = connection.recv()
                    except (EOFError, OSError):
                        restart(process_id, timed_out=False)
                        continue
                    input_idx, idx, _ = rules_to_apply_per_process[process_id].popleft()
                    batch_results[input_idx][idx] = result
                    if rules_to_apply_per_process[process_id]:
                        last_progress[process_id] = monotonic()
                    else:
                        unregister(process_id)
                if timeout is not None:
                    for process_id in list(last_progress):
                        if (
                            process_id not in handled
                            and monotonic() - last_progress[process_id] >= timeout
                        ):
                            # A reply may have arrived while another worker's result was decoded.
                            try:
                                if self._connections[process_id].poll(0):
                                    continue
                            except (EOFError, OSError):
                                restart(process_id, timed_out=False)
                                continue
                            restart(process_id, timed_out=True)

        return batch_results
