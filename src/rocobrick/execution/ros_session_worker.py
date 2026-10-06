"""Session-owned ROS mailbox worker; operation contracts remain unchanged."""

from __future__ import annotations

import json
import math
import signal
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rocobrick.coordinated_session import (
    CoordinatedSessionResult,
    parse_coordinated_request,
)
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
        if kind in {
            "JOINT_TARGET",
            "LINK6_POSE",
            "TCP_POSE",
            "GUARDED_TCP_POSE",
            "GRIPPER_TARGET",
        }:
            if execution.get("terminal") is not True:
                return False
            if execution.get("safe_to_release") is not True:
                return False
        elif kind == "TRANSFER_ATTACHED_PLANNING_SCENE":
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


@dataclass
class ActiveCoordinatedRequest:
    """Own one combined free-space request until terminal-safe evidence."""

    request: object
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: CoordinatedSessionResult | None = None
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
        self.coordinated_active: dict[str, ActiveCoordinatedRequest] = {}
        self.owners: dict[str, str] = {}
        self.seen: set[str] = set()
        self.seen_operations: set[str] = set()
        self.admission_lock = threading.Lock()
        self.stopping = threading.Event()
        self.unsafe = False
        self.failed = False
        for name in ("requests", "results", "cancels", "coordinated_cancels"):
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

    def _answer_document(self, key: str, result: Mapping[str, object]) -> None:
        write_atomic(
            self.directory / "results" / f"{key}.json",
            {
                "schema": SCHEMA,
                "token": self.token,
                "result": dict(result),
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
        """Fan cancellation out to every robot operation in one request."""
        entry.cancel.set()
        if entry.cancel_thread is not None:
            return

        def cancel_goals() -> None:
            while not entry.done.is_set():
                for item in entry.request.operations:
                    self.session.runtime_for(item.robot_id).cancel_active()
                entry.done.wait(0.05)

        entry.cancel_thread = threading.Thread(target=cancel_goals, daemon=True)
        entry.cancel_thread.start()

    def _execute_coordinated(
        self, key: str, entry: ActiveCoordinatedRequest
    ) -> None:
        """Run combined execution off the mailbox tick thread."""
        try:
            execute_fn = getattr(self.session, "execute_coordinated_pre_phase", None)
            if not callable(execute_fn):
                raise RuntimeError("session has no coordinated execution boundary")
            entry.result = execute_fn(entry.request)
        except Exception as error:
            entry.result = CoordinatedSessionResult(
                entry.request.coordination_id,
                entry.request.robots,
                {},
                {},
                {
                    "physical_execution": False,
                    "prepared": False,
                    "started": False,
                    "terminal": False,
                    "safe_to_release": False,
                },
                False,
                str(error),
                entry.request.participants,
            )
        finally:
            if entry.result is not None:
                self._answer_document(
                    key,
                    {
                        "passed": entry.result.passed,
                        "coordination_id": entry.request.coordination_id,
                        "result": entry.result.serialize(),
                        "error": entry.result.error,
                    },
                )
            entry.done.set()

    def _cancel_coordinated(self, entry: ActiveCoordinatedRequest) -> None:
        """Fan cancellation out to every participant runtime."""
        entry.cancel.set()
        if entry.cancel_thread is not None:
            return

        def cancel_goals() -> None:
            participants = tuple(entry.request.participants)
            while not entry.done.is_set():
                for robot_id in participants:
                    runtime = self.session.runtime_for(robot_id)
                    runtime.cancel_active()
                entry.done.wait(0.05)

        entry.cancel_thread = threading.Thread(target=cancel_goals, daemon=True)
        entry.cancel_thread.start()

    def _coordinated(self, key: str, document: object) -> None:
        """Admit one combined plan under the same persistent session ownership."""
        request = parse_coordinated_request(document)
        error = None
        if request.robots != self.robots or not request.plan_only:
            error = "combined mailbox accepts exact session robots and plan-only mode"
        elif (
            self.active
            or self.owners
            or self.failed
            or self.unsafe
            or self.stopping.is_set()
        ):
            error = "combined planning requires an idle safe participant group"
        elif request.coordination_id in self.seen_operations:
            error = "combined request replay refused"
        self.seen_operations.add(request.coordination_id)
        try:
            if error:
                raise ValueError(error)
            result = self.session.unified_plan_only(request)
        except Exception as failure:
            result = CoordinatedSessionResult(
                request.coordination_id,
                request.robots,
                {},
                {},
                {
                    "plan_only": True,
                    "physical_execution": False,
                    "safe_to_release": False,
                },
                False,
                str(failure),
            )
        self._answer(key, result)

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
        for key, entry in list(self.coordinated_active.items()):
            cancel_path = self.directory / "coordinated_cancels" / f"{key}.json"
            if cancel_path.is_file():
                message = read_message(cancel_path, self.token)
                if message.get("coordination_id") != entry.request.coordination_id:
                    raise ValueError("coordinated cancellation identity mismatch")
                self._cancel_coordinated(entry)
            if self.stopping.is_set():
                self._cancel_coordinated(entry)
            if entry.done.is_set():
                lifecycle = entry.result.lifecycle if entry.result is not None else {}
                safe = bool(
                    entry.result is not None
                    and entry.result.participants == entry.request.participants
                    and lifecycle.get("terminal") is True
                    and lifecycle.get("safe_to_release") is True
                )
                self.unsafe = self.unsafe or not safe
                self.failed = (
                    self.failed
                    or entry.result is None
                    or not entry.result.passed
                    or not safe
                )
                if entry.cancel_thread is not None:
                    entry.cancel_thread.join(timeout=0.1)
                    if entry.cancel_thread.is_alive():
                        self.unsafe = True
                        continue
                if safe:
                    for robot_id in entry.request.participants:
                        self.owners.pop(robot_id, None)
                self.coordinated_active.pop(key)
        for path in sorted((self.directory / "requests").glob("*.json")):
            key = path.stem
            if key in self.seen:
                continue
            self.seen.add(key)
            if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError("invalid mailbox request identity")
            message = read_message(path, self.token)
            raw_request = message.get("request")
            if (
                isinstance(raw_request, Mapping)
                and raw_request.get("schema") == "rocobrick-unified-observe/v1"
            ):
                if (
                    self.active
                    or self.owners
                    or raw_request.get("robots") != list(self.robots)
                ):
                    self._answer_document(
                        key,
                        {
                            "passed": False,
                            "error": "shared observation requires idle participants",
                        },
                    )
                else:
                    try:
                        sim_time_fn = getattr(
                            self.session, "latest_shared_sim_time_s", None
                        )
                        if not callable(sim_time_fn):
                            raise RuntimeError(
                                "persistent session exposes no planning-scene "
                                "version or shared clock"
                            )
                        # The planning-scene version read can run a full
                        # GetPlanningScene fingerprint (slow with many world
                        # objects), so it MUST happen BEFORE the scene query:
                        # the snapshot is then captured last and stays inside
                        # the caller's freshness budget at answer time.
                        version = getattr(
                            self.session, "unified_planning_scene_version", None
                        )
                        start_state_fn = getattr(
                            self.session, "unified_start_state", None
                        )
                        start_state = {}
                        if callable(start_state_fn):
                            start_state = start_state_fn()
                            if not isinstance(start_state, Mapping) or not all(
                                isinstance(name, str)
                                and isinstance(value, (int, float))
                                and not isinstance(value, bool)
                                and math.isfinite(float(value))
                                for name, value in start_state.items()
                            ):
                                raise RuntimeError("combined start state is invalid")
                        shared = self.session.observe_all()
                        observed = sim_time_fn()
                        if type(version) is not int or version < 0:
                            raise RuntimeError(
                                "unified planning-scene version is invalid"
                            )
                        if (
                            isinstance(observed, bool)
                            or not isinstance(observed, (int, float))
                            or not math.isfinite(observed)
                            or observed < 0
                        ):
                            raise RuntimeError(
                                "shared simulation-time sample is invalid"
                            )
                        self._answer_document(
                            key,
                            {
                                "passed": True,
                                "request_id": raw_request["request_id"],
                                "scene_snapshot": dict(shared),
                                "planning_scene_version": version,
                                "observed_sim_time_s": float(observed),
                                "start_state": dict(start_state),
                            },
                        )
                    except Exception as error:
                        self._answer_document(
                            key, {"passed": False, "error": str(error)}
                        )
                continue
            if (
                isinstance(raw_request, Mapping)
                and raw_request.get("schema")
                == "rocobrick-coordinated-execute/v1"
            ):
                if (
                    self.active
                    or self.coordinated_active
                    or self.owners
                    or self.failed
                    or self.unsafe
                    or self.stopping.is_set()
                ):
                    self._answer_document(
                        key,
                        {
                            "passed": False,
                            "error": (
                                "coordinated execution requires an idle safe group"
                            ),
                        },
                    )
                    continue
                try:
                    parsed = parse_coordinated_request(raw_request.get("request"))
                    if parsed.robots != self.robots:
                        raise ValueError(
                            "coordinated execution session robots mismatch"
                        )
                    if parsed.coordination_id in self.seen_operations:
                        raise ValueError("coordinated request replay refused")
                    self.seen_operations.add(parsed.coordination_id)
                    entry = ActiveCoordinatedRequest(parsed)
                    self.coordinated_active[key] = entry
                    for robot_id in parsed.participants:
                        self.owners[robot_id] = key
                    entry.thread = threading.Thread(
                        target=self._execute_coordinated,
                        args=(key, entry),
                        daemon=True,
                    )
                    entry.thread.start()
                    entry.thread.join(timeout=0.01)
                except Exception as error:
                    self.failed = True
                    self._answer_document(
                        key, {"passed": False, "error": str(error)}
                    )
                continue
            if (
                isinstance(raw_request, Mapping)
                and raw_request.get("schema")
                == "rocobrick-coordinated-moveit-request/v1"
            ):
                self._coordinated(key, raw_request)
                continue
            request = RosSessionRequest.parse(raw_request)
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
                if not self.active and not self.coordinated_active:
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
                if worker is not None and (worker.active or worker.coordinated_active):
                    orderly = False
                    for entry in worker.active.values():
                        worker._cancel(entry)
                    for entry in worker.coordinated_active.values():
                        worker._cancel_coordinated(entry)
                    for entry in worker.active.values():
                        entry.done.wait(STOP_TIMEOUT_S)
                    for entry in worker.coordinated_active.values():
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
                "unsafe": worker.unsafe if worker is not None else True,
                "retained_robots": sorted(worker.owners) if worker is not None else [],
                "active_requests": sorted(
                    [
                        *worker.active,
                        *worker.coordinated_active,
                    ]
                ) if worker is not None else [],
            },
        )
    return 0 if orderly else 1
