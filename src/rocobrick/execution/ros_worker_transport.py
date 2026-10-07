"""Host-side transport for one session-owned container ROS worker."""

from __future__ import annotations

import math
import signal
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from uuid import uuid4

from rocobrick.adapters.moveit.readonly_combined_plan import READONLY_PLAN_SCHEMA
from rocobrick.coordinated_session import (
    CoordinatedSessionRequest,
    CoordinatedSessionResult,
)
from rocobrick.execution.prepared_motion import (
    PREPARED_MOTION_SCHEMA,
    PreparedMotionRequest,
)
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
        self.active_coordinated: dict[
            str, tuple[str, CoordinatedSessionRequest | PreparedMotionRequest, Path]
        ] = {}
        self.stopping = False
        self.failed = False

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
            if self.stopping or self.failed or self.handle.poll() is not None:
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

    def observe_all(
        self, robots: tuple[str, ...], timeout_s: float
    ) -> Mapping[str, object]:
        """Read exactly one shared host snapshot through the persistent ROS owner.

        Returns:
            Complete scene state with every configured robot.
        """
        snapshot = self.observe_shared(robots, timeout_s)["scene_snapshot"]
        if not isinstance(snapshot, Mapping):
            raise ValueError("shared observation returned no scene snapshot")
        return snapshot

    def observe_monitor(
        self,
        robots: tuple[str, ...],
        timeout_s: float,
    ) -> Mapping[str, object]:
        """Read fresh physics during owned actions; never admit planning/leases.

        Returns:
            A scene sample only, with no synthetic planning-scene version.
        """
        key = uuid4().hex
        path = self.directory / "results" / f"{key}.json"
        with self.lock:
            if self.stopping or self.handle.poll() is not None:
                raise RuntimeError("physical monitor requires a live worker")
            write_atomic(
                self.directory / "requests" / f"{key}.json",
                {
                    "schema": SCHEMA,
                    "token": self.token,
                    "request": {
                        "schema": "rocobrick-scene-monitor/v1",
                        "request_id": key,
                        "robots": list(robots),
                    },
                },
            )
        # Do not hold the transport lock while physical cancellation may need it.
        self._wait(path, timeout_s)
        result = read_message(path, self.token).get("result")
        if (
            not isinstance(result, Mapping)
            or result.get("passed") is not True
            or (
                result.get("request_id") != key
                or result.get("read_only") is not True
                or result.get("physical_admission") is not False
            )
        ):
            raise ValueError("physical monitor identity/read-only proof mismatch")
        snapshot = result.get("scene_snapshot")
        observed = result.get("observed_sim_time_s")
        if (
            not isinstance(snapshot, Mapping)
            or not isinstance(snapshot.get("robots"), Mapping)
            or (set(snapshot["robots"]) != set(robots))
        ):
            raise ValueError("physical monitor participant mismatch")
        sample_time = snapshot.get("sim_time_s")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v < 0
            for v in (observed, sample_time)
        ) or (not 0 <= observed - sample_time <= 0.5):
            raise ValueError("physical monitor scene is stale or from a future clock")
        return dict(snapshot)

    def observe_shared(
        self, robots: tuple[str, ...], timeout_s: float
    ) -> Mapping[str, object]:
        """Read one shared observation bound to its planning-scene version.

        Returns:
            The complete scene snapshot plus the unified planning-scene
            version and the shared simulation-time sample observed with it.
        """
        key = uuid4().hex
        path = self.directory / "results" / f"{key}.json"
        with self.lock:
            if (
                self.active
                or self.failed
                or self.stopping
                or self.handle.poll() is not None
            ):
                raise RuntimeError("shared observation requires an idle worker")
            write_atomic(
                self.directory / "requests" / f"{key}.json",
                {
                    "schema": SCHEMA,
                    "token": self.token,
                    "request": {
                        "schema": "rocobrick-unified-observe/v1",
                        "request_id": key,
                        "robots": list(robots),
                    },
                },
            )
            try:
                self._wait(path, timeout_s)
            except TimeoutError:
                self.failed = True
                raise
            result = read_message(path, self.token).get("result")
            if (
                not isinstance(result, Mapping)
                or result.get("passed") is not True
                or result.get("request_id") != key
            ):
                raise ValueError("shared observation failed or identity mismatched")
            snapshot = result.get("scene_snapshot")
            if not isinstance(snapshot, Mapping) or set(
                snapshot.get("robots", {})
            ) != set(robots):
                raise ValueError("shared observation participant set mismatched")
            version = result.get("planning_scene_version")
            if type(version) is not int or version < 0:
                raise ValueError("shared observation planning-scene version is invalid")
            observed = result.get("observed_sim_time_s")
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(observed)
                or observed < 0
            ):
                raise ValueError("shared observation simulation time is invalid")
            start_state = result.get("start_state", {})
            if not isinstance(start_state, Mapping) or not all(
                isinstance(name, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for name, value in start_state.items()
            ):
                raise ValueError("shared observation combined start state is invalid")
            return {
                "scene_snapshot": snapshot,
                "planning_scene_version": version,
                "observed_sim_time_s": float(observed),
                "start_state": dict(start_state),
            }

    def call_plan_only(
        self,
        request: CoordinatedSessionRequest,
        result_path: Path,
        timeout_s: float,
        *,
        read_only: bool = False,
    ) -> CoordinatedSessionResult:
        """Submit a no-motion combined plan to the existing worker.

        Returns:
            A strictly bound planning result, never operation arrival evidence.
        """
        if not request.plan_only:
            raise ValueError("combined mailbox execution is not implemented")
        key = uuid4().hex
        path = self.directory / "results" / f"{key}.json"
        with self.lock:
            if (
                self.stopping
                or self.failed
                or self.active
                or self.active_coordinated
                or self.handle.poll() is not None
            ):
                raise RuntimeError("combined planning requires an idle live worker")
            if result_path.exists():
                raise ValueError("plan result path already exists")
            write_atomic(
                self.directory / "requests" / f"{key}.json",
                {
                    "schema": SCHEMA,
                    "token": self.token,
                    "request": (
                        {"schema": READONLY_PLAN_SCHEMA, "request": request.serialize()}
                        if read_only
                        else request.serialize()
                    ),
                },
            )
            try:
                self._wait(path, timeout_s)
            except TimeoutError:
                self.failed = True
                raise
            result = CoordinatedSessionResult.parse(
                read_message(path, self.token).get("result")
            )
            if (
                result.coordination_id != request.coordination_id
                or result.robots != request.robots
                or result.participants != request.participants
            ):
                raise ValueError("combined worker result identity mismatch")
            if (
                result.lifecycle.get("physical_execution") is not False
                or result.lifecycle.get("safe_to_release") is not False
            ):
                raise ValueError("plan-only worker result claims physical execution")
            if (
                read_only
                and result.passed
                and result.lifecycle.get("read_only") is not True
            ):
                raise ValueError("read-only planning result is not attested")
            write_atomic(result_path, result.serialize())
            return result

    def execute_coordinated(
        self,
        request: CoordinatedSessionRequest | PreparedMotionRequest,
        result_path: Path,
        timeout_s: float,
    ) -> CoordinatedSessionResult:
        """Submit one combined free-space execution to the worker.

        Returns:
            The worker's strictly validated physical execution result.
        """
        key = uuid4().hex
        path = self.directory / "results" / f"{key}.json"
        with self.lock:
            if (
                self.stopping
                or self.failed
                or self.active
                or self.active_coordinated
                or self.handle.poll() is not None
            ):
                raise RuntimeError("coordinated execution requires an idle live worker")
            if result_path.exists():
                raise ValueError("coordinated execution result path already exists")
            write_atomic(
                self.directory / "requests" / f"{key}.json",
                {
                    "schema": SCHEMA,
                    "token": self.token,
                    "request": {
                        "schema": (
                            PREPARED_MOTION_SCHEMA
                            if isinstance(request, PreparedMotionRequest)
                            else "rocobrick-coordinated-execute/v1"
                        ),
                        "request_id": key,
                        "request": request.serialize(),
                    },
                },
            )
            self.active_coordinated[request.coordination_id] = (key, request, path)
        try:
            try:
                self._wait(path, timeout_s)
            except TimeoutError:
                self.failed = True
                raise
            message = read_message(path, self.token)
            result = message.get("result")
            if (
                not isinstance(result, Mapping)
                or result.get("coordination_id") != request.coordination_id
            ):
                raise ValueError("coordinated execution result identity mismatched")
            parsed = CoordinatedSessionResult.parse(result.get("result"))
            if (
                parsed.robots != request.robots
                or parsed.participants != request.participants
            ):
                raise ValueError("coordinated execution robot identity mismatched")
            if parsed.passed and (
                parsed.lifecycle.get("physical_execution") is not True
                or parsed.lifecycle.get("safe_to_release") is not True
            ):
                raise ValueError(
                    "coordinated execution result lacks physical terminal proof"
                )
            if (
                isinstance(request, PreparedMotionRequest)
                and parsed.passed
                and (
                    parsed.lifecycle.get("prepared_binding_sha256")
                    != request.binding_sha256
                    or parsed.lifecycle.get("replanned") is not False
                    or any(
                        item.get("synchronization_feedback_confirmed") is not True
                        for item in parsed.lifecycle.get("participants", {}).values()
                    )
                    or set(parsed.lifecycle.get("participants", {}))
                    != set(request.participants)
                )
            ):
                raise ValueError(
                    "prepared result lacks exact path/synchronization proof"
                )
            write_atomic(result_path, parsed.serialize())
            return parsed
        except (OSError, RuntimeError, TypeError, ValueError, TimeoutError):
            self.failed = True
            raise
        finally:
            with self.lock:
                self.active_coordinated.pop(request.coordination_id, None)

    def cancel_free_space(
        self, motion_id: str, timeout_s: float = 10.0
    ) -> Mapping[str, object]:
        """Cancel one combined request and await its terminal-safe result.

        Returns:
            Evidence bound to the exact owned combined motion identity.
        """
        with self.lock:
            entry = self.active_coordinated.get(motion_id)
        if entry is None:
            return {
                "motion_id": motion_id,
                "attempted": False,
                "terminal": False,
                "safe": False,
                "error": "free-space request is not active",
            }
        key, request, path = entry
        write_atomic(
            self.directory / "coordinated_cancels" / f"{key}.json",
            {
                "schema": SCHEMA,
                "token": self.token,
                "coordination_id": request.coordination_id,
            },
        )
        try:
            self._wait(path, timeout_s)
            message = read_message(path, self.token)
            parsed = CoordinatedSessionResult.parse(
                message.get("result", {}).get("result")
            )
            lifecycle = parsed.lifecycle
            safe = bool(
                parsed.participants == request.participants
                and lifecycle.get("terminal") is True
                and lifecycle.get("safe_to_release") is True
            )
            return {
                "motion_id": motion_id,
                "attempted": True,
                "terminal": lifecycle.get("terminal") is True,
                "safe": safe,
                "participants": dict(lifecycle.get("participants") or {}),
            }
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            return {
                "motion_id": motion_id,
                "attempted": True,
                "terminal": False,
                "safe": False,
                "error": str(error),
            }

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
            try:
                self._wait(path, timeout_s)
            except TimeoutError:
                self.failed = True
                raise
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
