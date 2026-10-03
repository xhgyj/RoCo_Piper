"""Session-owned ROS mailbox worker; operation contracts remain unchanged."""

from __future__ import annotations

import json
import signal
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rocobrick.ros_session_entry import (
    FK_TIMEOUT_S,
    RosSessionRequest,
    RosSessionResult,
    _fallback_session_result,
    _operation_record,
    _refused_operation_result,
    _spin_down_session,
    _spin_up_session,
    perform_session,
)

SCHEMA = "rocobrick-ros-worker/v1"
POLL_S = 0.02
STOP_TIMEOUT_S = 20.0


def write_atomic(path: Path, document: Mapping[str, object]) -> None:
    """Publish a complete document, never a partially written response."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_message(path: Path, token: str) -> dict[str, object]:
    """Reject stale or foreign messages before touching a runtime.

    Returns:
        The validated message document owned by this worker token.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("token") != token:
        raise ValueError("worker message token mismatch")
    if document.get("schema") != SCHEMA:
        raise ValueError("worker message schema mismatch")
    return document


def terminal_safe(result: RosSessionResult) -> bool:
    """A completed thread or ROS shutdown is not physical terminal evidence.

    Returns:
        True only when every dispatched motion record carries both terminal
        and safe-to-release evidence and every other record passed.
    """
    for record in result.operations:
        if record.get("dispatched") is not True:
            continue
        inner = record.get("result")
        if not isinstance(inner, Mapping):
            return False
        execution = inner.get("execution")
        if not isinstance(execution, Mapping):
            return False
        kind = record.get("kind")
        if kind in {"JOINT_TARGET", "LINK6_POSE", "TCP_POSE", "GUARDED_TCP_POSE", "GRIPPER_TARGET"}:
            if execution.get("terminal") is not True:
                return False
            if execution.get("safe_to_release") is not True:
                return False
        elif inner.get("passed") is not True:
            return False
    return True


@dataclass
class ActiveRequest:
    """Ownership survives until the exact result has terminal-safe evidence."""

    request: RosSessionRequest
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: RosSessionResult | None = None
    thread: threading.Thread | None = None
    cancel_thread: threading.Thread | None = None


class SessionWorker:
    """Reuse one spinning ROS session with independent robot execution lanes."""

    def __init__(
        self, directory: Path, token: str, robots: tuple[str, ...], session: object
    ):
        """Own the mailbox of one already spinning multi-robot session."""
        self.directory = directory
        self.token = token
        self.robots = robots
        self.session = session
        self.active: dict[str, ActiveRequest] = {}
        self.owners: dict[str, str] = {}
        self.seen: set[str] = set()
        self.seen_operations: set[str] = set()
        self.admission_lock = threading.Lock()
        self.stopping = threading.Event()
        self.unsafe = False
        self.failed = False
        for name in ("requests", "results", "cancels"):
            (directory / name).mkdir(parents=True, exist_ok=True)

    def _answer(self, key: str, result: RosSessionResult) -> None:
        write_atomic(
            self.directory / "results" / f"{key}.json",
            {
                "schema": SCHEMA,
                "token": self.token,
                "result": result.serialize(),
                "completed_wall_s": time.time(),
            },
        )

    def _refuse(self, key: str, request: RosSessionRequest, error: str) -> None:
        records = tuple(
            _operation_record(
                item,
                _refused_operation_result(item, error, evidence={}),
                dispatched=False,
            )
            for item in request.operations
        )
        self._answer(
            key,
            RosSessionResult(
                request_id=request.request_id,
                robots=self.robots,
                readiness_by_robot={},
                all_ready=False,
                operations=records,
                passed=False,
                error=error,
            ),
        )

    def _execute(self, key: str, entry: ActiveRequest) -> None:
        try:
            entry.result = perform_session(
                self.session,
                entry.request,
                consolidated_readiness=True,
                admission_lock=self.admission_lock,
                cancelled=entry.cancel.is_set,
            )
        except Exception as error:
            entry.result = _fallback_session_result(
                f"worker execution failed: {type(error).__name__}: {error}",
                robots=self.robots,
            )
        finally:
            if entry.result is not None:
                self._answer(key, entry.result)
            entry.done.set()

    def _cancel(self, entry: ActiveRequest) -> None:
        entry.cancel.set()
        if entry.cancel_thread is not None:
            return

        def cancel_goals() -> None:
            # Cancellation can race goal acceptance. Keep intent active until
            # the operation publishes its own terminal evidence.
            while not entry.done.is_set():
                for item in entry.request.operations:
                    self.session.runtime_for(item.robot_id).cancel_active()
                entry.done.wait(0.05)

        entry.cancel_thread = threading.Thread(target=cancel_goals, daemon=True)
        entry.cancel_thread.start()

    def tick(self) -> None:
        """Consume atomic messages and reap results without blocking motion lanes."""
        stop = self.directory / "stop.json"
        if stop.is_file():
            read_message(stop, self.token)
            self.stopping.set()
        for key, entry in list(self.active.items()):
            cancel_path = self.directory / "cancels" / f"{key}.json"
            if cancel_path.is_file():
                message = read_message(cancel_path, self.token)
                if message.get("request_id") != entry.request.request_id:
                    raise ValueError("cancellation request identity mismatch")
                self._cancel(entry)
            if self.stopping.is_set():
                self._cancel(entry)
            if entry.done.is_set():
                safe = entry.result is not None and terminal_safe(entry.result)
                self.unsafe = self.unsafe or not safe
                self.failed = (
                    self.failed or entry.result is None or not entry.result.passed
                )
                if entry.cancel_thread is not None:
                    entry.cancel_thread.join(timeout=0.1)
                    if entry.cancel_thread.is_alive():
                        self.unsafe = True
                        continue
                if safe:
                    for robot_id in {
                        item.robot_id for item in entry.request.operations
                    }:
                        self.owners.pop(robot_id, None)
                self.active.pop(key)
        for path in sorted((self.directory / "requests").glob("*.json")):
            key = path.stem
            if key in self.seen:
                continue
            self.seen.add(key)
            if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError("invalid mailbox request identity")
            message = read_message(path, self.token)
            request = RosSessionRequest.parse(message.get("request"))
            if request.robots != self.robots:
                self._refuse(key, request, "worker robot set mismatch")
                continue
            ids = {item.operation.request_id for item in request.operations}
            requested_robots = {item.robot_id for item in request.operations}
            error = None
            if self.stopping.is_set() or self.failed or self.unsafe:
                error = "worker is stopping or a previous operation failed"
            elif len(request.operations) > 1:
                error = "worker accepts only one operation per phase envelope"
            elif ids & self.seen_operations:
                error = "operation identity was already consumed; replay refused"
            elif requested_robots & self.owners.keys():
                error = "robot already owns an active operation"
            elif not request.operations and self.active:
                error = "global readiness cannot overlap active operations"
            if error is not None:
                self._refuse(key, request, error)
                continue
            self.seen_operations.update(ids)
            entry = ActiveRequest(request)
            self.active[key] = entry
            for robot_id in requested_robots:
                self.owners[robot_id] = key
            entry.thread = threading.Thread(
                target=self._execute, args=(key, entry), daemon=True
            )
            entry.thread.start()

    def run(self) -> bool:
        """Stop admission first, then wait boundedly for terminal-safe results.

        Returns:
            True only when every active request settled with terminal-safe
            evidence before the stop deadline.
        """
        deadline = None
        while True:
            self.tick()
            if self.stopping.is_set():
                if deadline is None:
                    deadline = time.monotonic() + STOP_TIMEOUT_S
                if not self.active:
                    return not self.unsafe
                if time.monotonic() >= deadline:
                    return False
            time.sleep(POLL_S)


def serve(directory: Path, token: str, robots: tuple[str, ...], node_name: str) -> int:
    """Own exactly one ROS lifecycle and publish its cleanup attestation.

    Returns:
        Process exit code; zero only for an attested orderly shutdown.
    """
    session = background = thread = node = None
    worker = None
    orderly = False
    error = None
    previous = signal.getsignal(signal.SIGINT)
    try:
        session, background, thread, node = _spin_up_session(
            node_name, robots, FK_TIMEOUT_S
        )
        session.register()
        worker = SessionWorker(directory, token, robots, session)
        signal.signal(signal.SIGINT, lambda _signum, _frame: worker.stopping.set())
        write_atomic(
            directory / "ready.json",
            {
                "schema": SCHEMA,
                "token": token,
                "robots": list(robots),
                "ready": True,
            },
        )
        orderly = worker.run()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        signal.signal(signal.SIGINT, previous)
        if session is not None:
            try:
                if worker is not None and worker.active:
                    orderly = False
                    for entry in worker.active.values():
                        worker._cancel(entry)
                    for entry in worker.active.values():
                        entry.done.wait(STOP_TIMEOUT_S)
                _spin_down_session(session, background, thread, node)
                orderly = (
                    orderly and not session.shutdown_errors and not thread.is_alive()
                )
            except Exception as exc:
                orderly = False
                error = f"shutdown failed: {type(exc).__name__}: {exc}"
        write_atomic(
            directory / "shutdown.json",
            {
                "schema": SCHEMA,
                "token": token,
                "robots": list(robots),
                "orderly": orderly,
                "error": error,
                "active_requests": sorted(worker.active) if worker is not None else [],
            },
        )
    return 0 if orderly else 1
