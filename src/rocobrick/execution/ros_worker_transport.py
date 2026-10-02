"""Host-side transport for one session-owned container ROS worker."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from uuid import uuid4

from rocobrick.execution.ros_session_worker import SCHEMA, read_message, write_atomic
from rocobrick.ros_session_entry import RosSessionRequest, RosSessionResult


class RosWorkerTransport:
    """Atomic mailbox IPC with strict identity, bounded waits and no replay."""

    def __init__(
        self,
        directory: Path,
        token: str,
        handle: object,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Bind one live worker process handle to its tokened mailbox."""
        self.directory = directory
        self.token = token
        self.handle = handle
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()
        self.active: dict[str, tuple[str, RosSessionRequest, Path]] = {}
        self.stopping = False

    def wait_ready(self, robots: tuple[str, ...], timeout_s: float) -> None:
        """Block until the worker attests readiness for exactly these robots."""
        path = self.directory / "ready.json"
        self._wait(path, timeout_s)
        document = read_message(path, self.token)
        if document.get("ready") is not True or document.get("robots") != list(robots):
            raise ValueError("worker readiness identity mismatch")

    def _wait(self, path: Path, timeout_s: float) -> None:
        deadline = self.clock() + timeout_s
        while not path.is_file():
            if self.handle.poll() is not None:
                raise RuntimeError("ROS worker exited before answering")
            if self.clock() >= deadline:
                raise TimeoutError("ROS worker response deadline exceeded")
            self.sleep(0.02)

    def _result(self, path: Path, request: RosSessionRequest) -> RosSessionResult:
        document = read_message(path, self.token)
        result = RosSessionResult.parse(document.get("result"))
        if result.request_id != request.request_id or result.robots != request.robots:
            raise ValueError("worker result identity mismatch")
        expected = {
            (item.robot_id, item.operation.request_id, item.operation.kind.value)
            for item in request.operations
        }
        actual = {
            (record["robot_id"], record["operation_id"], record["kind"])
            for record in result.operations
        }
        if expected != actual or len(result.operations) != len(request.operations):
            raise ValueError("worker result operation set mismatch")
        return result

    def call(
        self, request: RosSessionRequest, result_path: Path, timeout_s: float
    ) -> RosSessionResult:
        """Submit one envelope and return its strictly validated answer.

        Returns:
            The worker's strict session result, persisted to ``result_path``.
        """
        key = uuid4().hex
        mailbox_result = self.directory / "results" / f"{key}.json"
        operation_id = (
            request.operations[0].operation.request_id if request.operations else key
        )
        with self.lock:
            if self.stopping or self.handle.poll() is not None:
                raise RuntimeError("ROS worker unavailable; replay is forbidden")
            if operation_id in self.active:
                raise ValueError("operation is already active")
            if result_path.exists():
                raise ValueError("result path already exists; stale response refused")
            self.active[operation_id] = (key, request, mailbox_result)
            write_atomic(
                self.directory / "requests" / f"{key}.json",
                {
                    "schema": SCHEMA,
                    "token": self.token,
                    "request": request.serialize(),
                    "submitted_wall_s": time.time(),
                },
            )
        try:
            try:
                self._wait(mailbox_result, timeout_s)
            except TimeoutError:
                self.cancel(operation_id, timeout_s=10.0)
                raise
            result = self._result(mailbox_result, request)
            write_atomic(result_path, result.serialize())
            return result
        finally:
            # Unknown terminal states remain available to cleanup cancellation.
            if mailbox_result.is_file():
                with self.lock:
                    self.active.pop(operation_id, None)

    def cancel(
        self, operation_id: str, timeout_s: float = 10.0
    ) -> Mapping[str, object]:
        """Direct one owned operation's cancellation and wait for its result.

        Returns:
            Strict cancel evidence; ``safe`` requires terminal execution proof.
        """
        with self.lock:
            entry = self.active.get(operation_id)
        if entry is None:
            return {
                "attempted": False,
                "terminal": False,
                "safe": False,
                "error": "operation is not active",
            }
        key, request, path = entry
        write_atomic(
            self.directory / "cancels" / f"{key}.json",
            {
                "schema": SCHEMA,
                "token": self.token,
                "request_id": request.request_id,
            },
        )
        try:
            self._wait(path, timeout_s)
            result = self._result(path, request)
            record = result.operations[0]
            inner = record.get("result")
            execution = inner.get("execution") if isinstance(inner, Mapping) else None
            terminal = (
                isinstance(execution, Mapping) and execution.get("terminal") is True
            )
            safe = terminal and execution.get("safe_to_release") is True
            return {
                "attempted": True,
                "terminal": terminal,
                "safe": safe,
                "result_path": str(path),
            }
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            return {
                "attempted": True,
                "terminal": False,
                "safe": False,
                "error": str(error),
            }

    def stop(
        self, robots: tuple[str, ...], timeout_s: float = 25.0
    ) -> dict[str, object]:
        """Orderly stop that passes only on the worker's own attestation.

        Returns:
            Cleanup evidence; ``passed`` requires exit 0 plus the orderly
            attestation with no active requests.
        """
        with self.lock:
            self.stopping = True
        write_atomic(
            self.directory / "stop.json", {"schema": SCHEMA, "token": self.token}
        )
        deadline = self.clock() + timeout_s
        while self.handle.poll() is None and self.clock() < deadline:
            self.sleep(0.02)
        exit_code = self.handle.poll()
        if exit_code is None:
            self.handle.send_signal(signal.SIGKILL)
            return {
                "passed": False,
                "error": "worker shutdown timed out",
                "exit_code": None,
            }
        try:
            self.handle.wait(timeout_s=0.0)
            evidence = read_message(self.directory / "shutdown.json", self.token)
            passed = (
                exit_code == 0
                and evidence.get("orderly") is True
                and evidence.get("robots") == list(robots)
                and evidence.get("active_requests") == []
            )
            return {"passed": passed, "exit_code": exit_code, "evidence": evidence}
        except (OSError, ValueError) as error:
            return {"passed": False, "exit_code": exit_code, "error": str(error)}
