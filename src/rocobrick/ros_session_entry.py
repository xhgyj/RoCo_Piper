"""Container child entry for the runtime session (R4.2, multi-robot A5).

Runs INSIDE the already-started rootless ROS Humble container through::

    python3 -m rocobrick.ros_session_entry --request /session/request.json \
        --result /session/result.json

It creates exactly one rclpy node with a
:class:`rocobrick.ros_moveit_runtime.RosMoveItRuntime` plus the accepted
:class:`rocobrick.ros_moveit_executor.RosMoveItExecutor`, registers the
transport, waits for readiness through the R1 predicates, executes the one
operation carried by the strict :class:`rocobrick.isaac_backend.RuntimeOperation`
request, writes the strict
:class:`rocobrick.isaac_backend.RuntimeOperationResult` document atomically
(always, also on failure), prints exactly one JSON summary line on stdout,
and shuts the executor/node/rclpy down cleanly.

Multi-robot mode (``--robots piper_0,piper_1``) hosts ONE
:class:`rocobrick.ros_moveit_runtime.RosMoveItRobotSession` on the same
single node: one isolated runtime per plan robot (robot-scoped state topics,
controllers, FK, MoveGroup, arm/gripper actions and scene slicing), the one
ROS domain and the shared ``/clock``. The request is then the strict
:class:`RosSessionRequest` document (``robots`` + robot-targeted
``operations``) and the answer the strict :class:`RosSessionResult`
document: readiness is evaluated for EVERY robot (also for robots with no
operation) and when any robot is not ready no operation is dispatched at
all. Without ``--robots`` the historical single-runtime behavior is
unchanged.

READINESS is a no-motion operation: nothing is ever published or sent.
JOINT_TARGET / LINK6_POSE / TCP_POSE / GRIPPER_TARGET targets are rebuilt
through the formal dataclasses and re-checked with strict admission
immediately before the executor send; APPLY_PLANNING_SCENE and SCENE_QUERY
are no-motion operations executed against the runtime adapter directly (in
multi-robot mode the scene query is sliced to the operation's own robot). No
target derivation, no plan parsing, and no TCP semantics exist here. The
repository ``src`` tree is consumed exactly as mounted read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rocobrick.isaac_backend import (
    OPERATION_RESULT_SCHEMA,
    OperationContractError,
    OperationKind,
    OperationRejected,
    RuntimeOperation,
    RuntimeOperationResult,
    attached_planning_scene_operation,
    planning_scene_update_from_payload,
    scene_query_from_payload,
    target_for_operation,
)
from rocobrick.moveit_contract import (
    GripperTarget,
    JointTarget,
    Link6PoseTarget,
    TcpPoseTarget,
)
from rocobrick.moveit_execution import ExecutionResult, is_success

SUMMARY_SCHEMA = "rocobrick-runtime-operation-summary/v1"
SESSION_SUMMARY_SCHEMA = "rocobrick-ros-session-summary/v1"
SESSION_REQUEST_SCHEMA = "rocobrick-ros-session-request/v1"
SESSION_RESULT_SCHEMA = "rocobrick-ros-session-result/v1"
DEFAULT_NODE_NAME = "rocobrick_session_entry"
FK_TIMEOUT_S = 10.0
THREAD_JOIN_TIMEOUT_S = 2.0
# Canonical schema of the host-bridge scene-state document (kept local so the
# ROS adapter module stays a lazy, in-container-only import).
SCENE_STATE_SCHEMA = "rocobrick-scene-state/v1"

SESSION_REQUEST_KEYS = frozenset(
    {"schema", "request_id", "robots", "timeout_s", "operations"}
)
SESSION_OPERATION_KEYS = frozenset({"robot_id", "operation"})
SESSION_RESULT_KEYS = frozenset(
    {
        "schema",
        "request_id",
        "robots",
        "readiness_by_robot",
        "all_ready",
        "operations",
        "passed",
        "error",
    }
)
SESSION_OPERATION_RECORD_KEYS = frozenset(
    {
        "robot_id",
        "operation_id",
        "kind",
        "dispatched",
        "motion_sent",
        "result",
    }
)


class RosSessionContractError(ValueError):
    """Raised when a robot-session document violates its exact schema."""


def _require_keys(
    document: object, expected: frozenset[str], label: str
) -> Mapping:
    """Require exactly the expected keys on a JSON object."""
    if not isinstance(document, Mapping):
        raise RosSessionContractError(f"{label} must be a JSON object")
    actual = frozenset(str(key) for key in document)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RosSessionContractError(
            f"{label} must carry exactly {sorted(expected)}; "
            f"missing={missing} extra={extra}"
        )
    return document


def _session_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RosSessionContractError(f"{label} must be a non-empty string")
    return value


def _session_robot_id(value: object, label: str) -> str:
    robot_id = _session_text(value, label)
    if "/" in robot_id:
        raise RosSessionContractError(f"{label} must be a ROS-safe robot id")
    return robot_id


@dataclass(frozen=True)
class RobotOperationRequest:
    """One robot-targeted runtime operation inside a session request."""

    robot_id: str
    operation: RuntimeOperation

    def serialize(self) -> dict[str, object]:
        return {"robot_id": self.robot_id, "operation": self.operation.serialize()}

    @classmethod
    def parse(cls, document: object) -> "RobotOperationRequest":
        payload = _require_keys(document, SESSION_OPERATION_KEYS, "session operation")
        robot_id = _session_robot_id(
            payload["robot_id"], "session operation.robot_id"
        )
        try:
            operation = RuntimeOperation.parse(payload["operation"])
        except ValueError as error:
            raise RosSessionContractError(
                f"session operation of {robot_id!r} is invalid: {error}"
            ) from error
        return cls(robot_id=robot_id, operation=operation)


@dataclass(frozen=True)
class RosSessionRequest:
    """Strict robot-session request: the plan robots plus their operations.

    ``operations`` may be empty: the session then answers readiness only,
    which is the zero-motion way to collect ``readiness_by_robot`` for every
    plan robot (including robots with no operation).
    """

    request_id: str
    robots: tuple[str, ...]
    timeout_s: float
    operations: tuple[RobotOperationRequest, ...]

    def serialize(self) -> dict[str, object]:
        return {
            "schema": SESSION_REQUEST_SCHEMA,
            "request_id": self.request_id,
            "robots": list(self.robots),
            "timeout_s": float(self.timeout_s),
            "operations": [item.serialize() for item in self.operations],
        }

    @classmethod
    def parse(cls, document: object) -> "RosSessionRequest":
        payload = _require_keys(document, SESSION_REQUEST_KEYS, "session request")
        if payload["schema"] != SESSION_REQUEST_SCHEMA:
            raise RosSessionContractError(
                f"session request schema {payload['schema']!r} is not "
                f"{SESSION_REQUEST_SCHEMA!r}"
            )
        request_id = _session_text(payload["request_id"], "session request_id")
        raw_robots = payload["robots"]
        if (
            not isinstance(raw_robots, Sequence)
            or isinstance(raw_robots, (str, bytes, bytearray))
        ):
            raise RosSessionContractError("session robots must be an array")
        robots = tuple(
            _session_robot_id(value, "session robots[]") for value in raw_robots
        )
        if not robots:
            raise RosSessionContractError("session robots must not be empty")
        if len(set(robots)) != len(robots):
            raise RosSessionContractError("session robots must be unique")
        timeout_s = payload["timeout_s"]
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise RosSessionContractError(
                f"session timeout_s must be finite and positive, got {timeout_s!r}"
            )
        raw_operations = payload["operations"]
        if (
            not isinstance(raw_operations, Sequence)
            or isinstance(raw_operations, (str, bytes, bytearray))
        ):
            raise RosSessionContractError("session operations must be an array")
        operations: list[RobotOperationRequest] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_operations):
            item = RobotOperationRequest.parse(raw)
            if item.robot_id not in robots:
                raise RosSessionContractError(
                    f"session operation {index} targets robot "
                    f"{item.robot_id!r}, which is not in {list(robots)}"
                )
            if item.operation.request_id in seen:
                raise RosSessionContractError(
                    f"session operation request_id "
                    f"{item.operation.request_id!r} is duplicated"
                )
            seen.add(item.operation.request_id)
            operations.append(item)
        return cls(
            request_id=request_id,
            robots=robots,
            timeout_s=float(timeout_s),
            operations=tuple(operations),
        )


@dataclass(frozen=True)
class RosSessionResult:
    """Strict robot-session answer: per-robot readiness plus per-robot results."""

    request_id: str
    robots: tuple[str, ...]
    readiness_by_robot: Mapping[str, Mapping[str, object]]
    all_ready: bool
    operations: tuple[Mapping[str, object], ...]
    passed: bool
    error: str | None = None

    def serialize(self) -> dict[str, object]:
        return {
            "schema": SESSION_RESULT_SCHEMA,
            "request_id": str(self.request_id),
            "robots": list(self.robots),
            "readiness_by_robot": {
                str(robot_id): dict(report)
                for robot_id, report in self.readiness_by_robot.items()
            },
            "all_ready": bool(self.all_ready),
            "operations": [dict(record) for record in self.operations],
            "passed": bool(self.passed),
            "error": None if self.error is None else str(self.error),
        }

    @classmethod
    def parse(cls, document: object) -> "RosSessionResult":
        payload = _require_keys(document, SESSION_RESULT_KEYS, "session result")
        if payload["schema"] != SESSION_RESULT_SCHEMA:
            raise RosSessionContractError(
                f"session result schema {payload['schema']!r} is not "
                f"{SESSION_RESULT_SCHEMA!r}"
            )
        request_id = _session_text(payload["request_id"], "session result request_id")
        raw_robots = payload["robots"]
        if (
            not isinstance(raw_robots, Sequence)
            or isinstance(raw_robots, (str, bytes, bytearray))
        ):
            raise RosSessionContractError("session result robots must be an array")
        robots = tuple(str(value) for value in raw_robots)
        readiness = payload["readiness_by_robot"]
        if not isinstance(readiness, Mapping):
            raise RosSessionContractError(
                "session result readiness_by_robot must be an object"
            )
        readiness_by_robot: dict[str, Mapping[str, object]] = {}
        for robot_id, report in readiness.items():
            if not isinstance(report, Mapping):
                raise RosSessionContractError(
                    f"session result readiness of {robot_id!r} must be an object"
                )
            readiness_by_robot[str(robot_id)] = dict(report)
        raw_operations = payload["operations"]
        if (
            not isinstance(raw_operations, Sequence)
            or isinstance(raw_operations, (str, bytes, bytearray))
        ):
            raise RosSessionContractError("session result operations must be an array")
        operations: list[Mapping[str, object]] = []
        for record in raw_operations:
            operations.append(
                dict(
                    _require_keys(
                        record,
                        SESSION_OPERATION_RECORD_KEYS,
                        "session operation record",
                    )
                )
            )
        for name in ("all_ready", "passed"):
            if not isinstance(payload[name], bool):
                raise RosSessionContractError(
                    f"session result {name} must be a boolean"
                )
        error = payload["error"]
        if error is not None and not isinstance(error, str):
            raise RosSessionContractError(
                "session result error must be a string or null"
            )
        return cls(
            request_id=request_id,
            robots=robots,
            readiness_by_robot=readiness_by_robot,
            all_ready=bool(payload["all_ready"]),
            operations=tuple(operations),
            passed=bool(payload["passed"]),
            error=error,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the child entry argument parser."""
    parser = argparse.ArgumentParser(
        prog="python3 -m rocobrick.ros_session_entry",
        description=(
            "Execute one bounded runtime operation inside the running "
            "MoveIt container; writes one strict result document and prints "
            "exactly one JSON summary line."
        ),
    )
    parser.add_argument(
        "--request",
        required=False,
        type=Path,
        metavar="PATH",
        help="RuntimeOperation request JSON (mounted session directory)",
    )
    parser.add_argument(
        "--result",
        required=False,
        type=Path,
        metavar="PATH",
        help="RuntimeOperationResult destination JSON (always written)",
    )
    parser.add_argument(
        "--node-name",
        default=DEFAULT_NODE_NAME,
        metavar="NAME",
        help=f"rclpy node name (default: {DEFAULT_NODE_NAME})",
    )
    parser.add_argument(
        "--robots",
        default=None,
        metavar="ID[,ID...]",
        help=(
            "comma-separated plan robot ids; enables the multi-robot session "
            "request/result documents (one isolated runtime per robot on one "
            "node). Omit for the single-runtime operation mode."
        ),
    )
    parser.add_argument("--serve-dir", type=Path)
    parser.add_argument("--worker-token")
    return parser


def parse_robots(value: str | None) -> tuple[str, ...] | None:
    """Parse the ``--robots`` argument into unique robot ids, or None."""
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",")]
    if not parts or any(not item for item in parts):
        raise ValueError("--robots must be a comma-separated list of robot ids")
    robots = tuple(parts)
    if len(set(robots)) != len(robots):
        raise ValueError("--robots must not repeat a robot id")
    for robot_id in robots:
        if "/" in robot_id:
            raise ValueError("--robots entries must be ROS-safe robot ids")
    return robots


def load_request(path: Path) -> RuntimeOperation:
    """Load and strictly parse the operation request document."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return RuntimeOperation.parse(document)


def load_session_request(path: Path) -> RosSessionRequest:
    """Load and strictly parse the robot-session request document."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return RosSessionRequest.parse(document)


def write_result(path: Path, result: object) -> None:
    """Write the result document atomically (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.serialize(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _jsonable(value: object) -> object:
    """Best-effort JSON-safety for details/limits mappings."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def execution_record(outcome: ExecutionResult) -> dict[str, object]:
    """JSON-safe evidence record of one ExecutionResult (no verdict here)."""
    return {
        "status": str(outcome.status.value),
        "moveit_error_code": outcome.moveit_error_code,
        "planned_waypoints": outcome.planned_waypoints,
        "command_messages": outcome.command_messages,
        "final_errors": {
            # An unmeasurable final error (e.g. a mimic-violating gripper
            # sample) stays an explicit null instead of crashing the record.
            name: (None if value is None else float(value))
            for name, value in outcome.final_errors.items()
        },
        "is_success": is_success(outcome),
        "goal_accepted": outcome.goal_accepted,
        "terminal": outcome.terminal,
        "safe_to_release": outcome.safe_to_release,
        "action_status": outcome.action_status,
        "cancel_evidence": _jsonable(dict(outcome.cancel_evidence)),
        "details": _jsonable(dict(outcome.details)),
    }


def _rejected_record(reason: str) -> dict[str, object]:
    """Execution record for a target rejected before any send."""
    return {
        "status": "REJECTED",
        "sent": False,
        "reason": reason,
        "command_messages": 0,
    }


def filter_scene_state(document: Mapping, query: Mapping) -> dict[str, object]:
    """Project a scene-state document onto one normalized scene query.

    Pure helper: the document schema must match exactly, every requested
    brick id must exist in ``document["bricks"]`` and every requested pair
    must exist in ``document["connections"]`` after normalizing both orders
    onto the sorted ``"a|b"`` key. The gripper block and the logical opening
    are copied only when the query asks for them. Any missing requested
    entry raises ``ValueError`` with the offending id/key so the caller can
    report a failed result instead of a partial scene.
    """
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != SCENE_STATE_SCHEMA
    ):
        raise ValueError(
            f"scene-state schema must be exactly {SCENE_STATE_SCHEMA!r}"
        )
    bricks_document = document.get("bricks")
    connections_document = document.get("connections")
    if not isinstance(bricks_document, Mapping):
        raise ValueError("scene-state document carries no bricks mapping")
    if not isinstance(connections_document, Mapping):
        raise ValueError("scene-state document carries no connections mapping")
    bricks: dict[str, object] = {}
    for brick_id in query["brick_physical_instance_ids"]:
        if brick_id not in bricks_document:
            raise ValueError(
                f"scene-state document carries no brick {str(brick_id)!r}"
            )
        bricks[str(brick_id)] = bricks_document[brick_id]
    connections: dict[str, object] = {}
    for pair in query["connection_pairs"]:
        key = "|".join(sorted((str(pair[0]), str(pair[1]))))
        if key not in connections_document:
            raise ValueError(
                f"scene-state document carries no connection {key!r}"
            )
        connections[key] = connections_document[key]
    filtered: dict[str, object] = {
        "sim_time_s": document["sim_time_s"],
        "bricks": bricks,
        "connections": connections,
    }
    # A robot-scoped view names its robot so a per-robot answer is
    # attributable; the canonical single-robot document carries no robot id.
    if "robot_id" in document:
        filtered["robot_id"] = document["robot_id"]
    for key in (
        "session_id", "manifest_sha256", "sequence", "robot_sequence", "tcp_pose"
    ):
        if key in document:
            filtered[key] = document[key]
    if query["include_gripper"]:
        filtered["gripper"] = document.get("gripper")
        filtered["opening_m"] = document.get("opening_m")
    return filtered


def perform_operation(
    operation: RuntimeOperation,
    *,
    runtime: object,
    executor: object,
    fk_timeout_s: float = FK_TIMEOUT_S,
    robot_scoped: bool = False,
    readiness_report: Mapping[str, object] | None = None,
) -> RuntimeOperationResult:
    """Execute one operation against an (already spinning) runtime.

    ``runtime`` is a :class:`rocobrick.ros_moveit_runtime.RosMoveItRuntime`
    and ``executor`` a
    :class:`rocobrick.ros_moveit_executor.RosMoveItExecutor`; both are passed
    in (not constructed) so offline tests can drive this function with
    fakes. READINESS never sends motion; motion operations rebuild the
    formal target and re-run strict admission before delegating exactly once
    to the executor. APPLY_PLANNING_SCENE and SCENE_QUERY pass the same
    readiness gate for uniformity but execute against the runtime adapter
    directly and never send motion.

    With ``robot_scoped=True`` the runtime must be a profiled per-robot
    runtime: readiness is the FULL capability gate of that robot (joints,
    controllers, MoveGroup, arm and gripper actions, FK and the scene
    query) and the SCENE_QUERY operation is sliced to that robot's own
    observation block, so a session can never answer with another robot's
    data.
    """
    evidence: dict[str, object] = {}
    try:
        runtime.register_subscriptions()
        runtime.register_clients()
        if robot_scoped:
            robot_id = str(getattr(runtime, "robot_id", "") or "")
            if not robot_id:
                return RuntimeOperationResult(
                    request_id=operation.request_id,
                    kind=operation.kind,
                    passed=False,
                    execution={"motion_sent": False},
                    evidence=dict(evidence),
                    error=(
                        "robot-scoped operation requires a profiled per-robot "
                        "runtime; this runtime carries no robot id"
                    ),
                )
            # A gripper command is verified from the Isaac finger state after
            # the action.  When the fingertips are physically in contact with
            # the baseplate, the controller broadcaster can temporarily lag
            # that state; do not let that arm-state comparison prevent the
            # gripper action itself.  Arm/TCP operations retain the strict gate.
            require_broadcaster_match = operation.kind != "GRIPPER_TARGET"
            if readiness_report is not None:
                if readiness_report.get("robot_id") != robot_id:
                    raise ValueError("readiness report robot identity mismatch")
                readiness_passed = readiness_report.get("passed") is True
                evidence.update(
                    robot_id=robot_id,
                    waited_ok=readiness_passed,
                    passed=readiness_passed,
                    capability=dict(readiness_report),
                    snapshot=dict(asdict(runtime.readiness_snapshot())),
                )
            else:
                waited_ok = bool(
                    runtime.wait_until_capability_ready(
                        timeout_s=operation.timeout_s,
                        require_broadcaster_match=require_broadcaster_match,
                    )
                )
                capability = runtime.capability_readiness(
                    probe_scene=True,
                    require_broadcaster_match=require_broadcaster_match,
                )
                readiness_passed = bool(waited_ok and capability.passed)
                evidence.update(
                    {
                        "robot_id": robot_id,
                        "waited_ok": waited_ok,
                        "passed": readiness_passed,
                        "capability": capability.serialize(),
                        "snapshot": dict(asdict(runtime.readiness_snapshot())),
                    }
                )
        else:
            waited_ok = bool(runtime.wait_until_ready(timeout_s=operation.timeout_s))
            snapshot = runtime.readiness_snapshot()
            readiness_passed = bool(waited_ok and snapshot.passed)
            evidence.update(
                {
                    "waited_ok": waited_ok,
                    "passed": readiness_passed,
                    "snapshot": dict(asdict(snapshot)),
                }
            )
        if operation.kind is OperationKind.READINESS:
            evidence["motion_sent"] = False
            if not readiness_passed:
                return RuntimeOperationResult(
                    request_id=operation.request_id,
                    kind=operation.kind,
                    passed=False,
                    execution={"motion_sent": False, "goals_sent": 0},
                    evidence=evidence,
                    error=(
                        "readiness phase failed: the runtime snapshot did not "
                        "pass every component"
                    ),
                )
            return RuntimeOperationResult(
                request_id=operation.request_id,
                kind=operation.kind,
                passed=True,
                execution={
                    "motion_sent": False,
                    "goals_sent": 0,
                    "command_messages": 0,
                },
                evidence=evidence,
                error=None,
            )
        if not readiness_passed:
            evidence["motion_sent"] = False
            return RuntimeOperationResult(
                request_id=operation.request_id,
                kind=operation.kind,
                passed=False,
                execution=_rejected_record(
                    "runtime readiness failed; no motion is allowed until the "
                    "snapshot passes every component"
                ),
                evidence=evidence,
                error="readiness phase failed before the goal send",
            )
        if operation.kind is OperationKind.APPLY_PLANNING_SCENE:
            try:
                update = planning_scene_update_from_payload(operation.payload)
            except OperationContractError as error:
                raise OperationRejected(
                    "planning-scene update rejected before apply: "
                    f"{error}"
                ) from error
            outcome = runtime.apply_planning_scene(
                update, timeout_s=operation.timeout_s
            )
            evidence["motion_sent"] = False
            record = {
                "motion_sent": False,
                "goals_sent": 0,
                "passed": bool(outcome.passed),
                "applied_object_ids": [str(value) for value in outcome.applied_object_ids],
                "scene_digest": outcome.scene_digest,
                "details": _jsonable(dict(outcome.details)),
            }
            return RuntimeOperationResult(
                request_id=operation.request_id,
                kind=operation.kind,
                passed=bool(outcome.passed),
                execution=record,
                evidence=evidence,
                error=(
                    None
                    if outcome.passed
                    else "the APPLY_PLANNING_SCENE update did not pass: "
                    f"{record['details'].get('error', 'the applier reported failure')}"
                ),
            )
        if operation.kind is OperationKind.ATTACH_PLANNING_SCENE:
            try:
                attached_operation = attached_planning_scene_operation(
                    operation.payload, operation.timeout_s, operation.request_id
                )
                outcome = runtime.apply_attached_planning_scene(
                    attached_operation.payload, timeout_s=attached_operation.timeout_s
                )
                passed = outcome.get("passed") is True
                record = {
                    "motion_sent": False,
                    "goals_sent": 0,
                    "passed": passed,
                    "attached_object_id": outcome.get("attached_object_id"),
                    "attached_link": outcome.get("attached_link"),
                    "readback_confirmed": outcome.get("readback_confirmed") is True,
                    "details": _jsonable(dict(outcome)),
                }
                return RuntimeOperationResult(
                    request_id=operation.request_id,
                    kind=operation.kind,
                    passed=passed,
                    execution=record,
                    evidence={**evidence, "motion_sent": False},
                    error=None if passed else str(outcome.get("error")),
                )
            except (OperationContractError, AttributeError, ValueError) as error:
                return RuntimeOperationResult(
                    request_id=operation.request_id,
                    kind=operation.kind,
                    passed=False,
                    execution={"motion_sent": False, "goals_sent": 0},
                    evidence={**evidence, "motion_sent": False},
                    error=f"attached planning-scene rejected: {error}",
                )
        if operation.kind is OperationKind.SCENE_QUERY:
            query = scene_query_from_payload(operation.payload)
            try:
                if robot_scoped:
                    document = runtime.query_robot_scene_state(
                        timeout_s=operation.timeout_s
                    )
                else:
                    document = runtime.query_scene_state(
                        timeout_s=operation.timeout_s
                    )
            except RuntimeError as error:
                evidence["motion_sent"] = False
                return RuntimeOperationResult(
                    request_id=operation.request_id,
                    kind=operation.kind,
                    passed=False,
                    execution={"motion_sent": False, "scene_state": None},
                    evidence=evidence,
                    error=f"the scene-state query failed: {error}",
                )
            filtered = filter_scene_state(document, query)
            evidence["motion_sent"] = False
            return RuntimeOperationResult(
                request_id=operation.request_id,
                kind=operation.kind,
                passed=True,
                execution={"motion_sent": False, "scene_state": filtered},
                evidence=evidence,
                error=None,
            )
        target = target_for_operation(operation)
        if isinstance(target, JointTarget):
            outcome = executor.execute_joint(
                target, timeout_s=operation.timeout_s
            )
        elif isinstance(target, Link6PoseTarget):
            outcome = executor.execute_link6_pose(
                target, timeout_s=operation.timeout_s
            )
        elif isinstance(target, TcpPoseTarget):
            outcome = executor.execute_tcp_pose(
                target, timeout_s=operation.timeout_s
            )
        elif isinstance(target, GripperTarget):
            outcome = executor.execute_gripper(
                target, timeout_s=operation.timeout_s
            )
        else:  # pragma: no cover - target_for_operation narrows the type
            raise OperationContractError(
                f"unsupported target type {type(target).__name__}"
            )
        record = execution_record(outcome)
        evidence["motion_sent"] = bool(outcome.command_messages >= 1)
        return RuntimeOperationResult(
            request_id=operation.request_id,
            kind=operation.kind,
            passed=bool(record["is_success"]),
            execution=record,
            evidence=evidence,
            error=(
                None
                if record["is_success"]
                else f"the {str(operation.kind.value)} goal did not pass the "
                "execution gates"
            ),
        )
    except OperationRejected as error:
        return RuntimeOperationResult(
            request_id=operation.request_id,
            kind=operation.kind,
            passed=False,
            execution=_rejected_record(str(error)),
            evidence={**evidence, "motion_sent": False},
            error=str(error),
        )
    except Exception as error:  # noqa: BLE001 - the child must always answer
        return RuntimeOperationResult(
            request_id=operation.request_id,
            kind=operation.kind,
            passed=False,
            execution={"motion_sent": None},
            evidence=dict(evidence),
            error=f"{type(error).__name__}: {error}",
        )


def _operation_record(
    item: RobotOperationRequest,
    outcome: RuntimeOperationResult,
    *,
    dispatched: bool,
) -> dict[str, object]:
    """One robot-targeted session operation record (JSON-safe, strict keys)."""
    return {
        "robot_id": item.robot_id,
        "operation_id": item.operation.request_id,
        "kind": str(item.operation.kind.value),
        "dispatched": bool(dispatched),
        "motion_sent": bool(outcome.execution.get("motion_sent") is True),
        "result": _jsonable(outcome.serialize()),
    }


def _refused_operation_result(
    item: RobotOperationRequest,
    error: str | None,
    *,
    evidence: Mapping[str, object],
) -> RuntimeOperationResult:
    """Zero-motion failed answer for an operation that was never dispatched."""
    return RuntimeOperationResult(
        request_id=item.operation.request_id,
        kind=item.operation.kind,
        passed=False,
        execution={"motion_sent": False, "goals_sent": 0},
        evidence={"robot_id": item.robot_id, **dict(evidence)},
        error=error,
    )


def perform_session(
    session: object,
    request: RosSessionRequest,
    *,
    fk_timeout_s: float = FK_TIMEOUT_S,
    consolidated_readiness: bool = False,
    admission_lock: threading.Lock | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> RosSessionResult:
    """Execute one robot-session request against an (already built) session.

    ``session`` is a
    :class:`rocobrick.ros_moveit_runtime.RosMoveItRobotSession` (or an
    equivalent duck-typed object) and is passed in, not constructed, so
    offline tests can drive this function with injected transports.

    Read-only capability readiness is collected for EVERY configured robot
    first. When any robot is not ready, NO operation is dispatched at all:
    every requested operation is answered with a robot-scoped failed record
    and ``motion_sent=False``. When all robots are ready, each operation runs
    on its own robot's runtime/executor through :func:`perform_operation`
    with ``robot_scoped=True`` in request order, and every record names its
    robot id, so results are never merged across robots.
    """
    readiness: dict[str, Mapping[str, object]] = {}
    records: list[dict[str, object]] = []

    def answer(
        *,
        all_ready: bool,
        passed: bool,
        error: str | None,
    ) -> RosSessionResult:
        completed = {str(record["operation_id"]) for record in records}
        for item in request.operations:
            if item.operation.request_id in completed:
                continue
            records.append(
                _operation_record(
                    item,
                    _refused_operation_result(
                        item, error, evidence={"all_ready": all_ready}
                    ),
                    dispatched=False,
                )
            )
        return RosSessionResult(
            request_id=request.request_id,
            robots=request.robots,
            readiness_by_robot=readiness,
            all_ready=bool(all_ready),
            operations=tuple(records),
            passed=bool(passed),
            error=error,
        )

    try:
        session_robots = tuple(getattr(session, "robot_ids", ()))
        if session_robots != request.robots:
            return answer(
                all_ready=False,
                passed=False,
                error=(
                    f"session robots {list(session_robots)} do not match the "
                    f"request robots {list(request.robots)}"
                ),
            )
        if consolidated_readiness:
            if admission_lock is not None:
                admission_lock.acquire()
            try:
                session.register()
                deadline = time.monotonic() + request.timeout_s
                while True:
                    reports = session.capability_readiness_by_robot(probe_scene=True)
                    waited_ok = bool(reports) and all(
                        report.passed is True for report in reports.values()
                    )
                    if waited_ok or time.monotonic() >= deadline:
                        break
                    if cancelled is not None and cancelled():
                        break
                    time.sleep(0.05)
            finally:
                if admission_lock is not None:
                    admission_lock.release()
        else:
            session.register()
            waited_ok = bool(session.wait_until_ready(timeout_s=request.timeout_s))
            reports = session.capability_readiness_by_robot(probe_scene=True)
        readiness = {
            str(robot_id): dict(report.serialize())
            for robot_id, report in reports.items()
        }
        all_ready = bool(
            waited_ok
            and readiness
            and set(readiness) == set(request.robots)
            and all(bool(report.get("passed")) for report in readiness.values())
        )
        if not all_ready:
            not_ready = sorted(
                robot_id
                for robot_id, report in readiness.items()
                if not bool(report.get("passed"))
            )
            error = (
                "robot readiness failed for "
                f"{not_ready or list(request.robots)}; no operation was "
                "dispatched"
            )
            for item in request.operations:
                records.append(
                    _operation_record(
                        item,
                        _refused_operation_result(
                            item,
                            error,
                            evidence={
                                "robot_ready": readiness.get(item.robot_id),
                                "all_ready": False,
                            },
                        ),
                        dispatched=False,
                    )
                )
            return answer(all_ready=False, passed=False, error=error)
        for item in request.operations:
            if cancelled is not None and cancelled():
                records.append(_operation_record(
                    item,
                    _refused_operation_result(
                        item, "request cancelled before dispatch",
                        evidence={"all_ready": all_ready},
                    ),
                    dispatched=False,
                ))
                break
            outcome = perform_operation(
                item.operation,
                runtime=session.runtime_for(item.robot_id),
                executor=session.executor_for(item.robot_id),
                fk_timeout_s=fk_timeout_s,
                robot_scoped=True,
                readiness_report=(readiness[item.robot_id]
                                  if consolidated_readiness else None),
            )
            records.append(_operation_record(item, outcome, dispatched=True))
            if consolidated_readiness and not outcome.passed:
                break
        failed = [
            str(record["operation_id"])
            for record in records
            if not bool(record["result"].get("passed"))
        ]
        passed = not failed
        return answer(
            all_ready=True,
            passed=passed,
            error=None if passed else f"session operations failed: {failed}",
        )
    except Exception as exc:  # noqa: BLE001 - the child must always answer
        return answer(
            all_ready=False,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _make_fk_hook(
    holder: dict, fk_timeout_s: float, *, tcp: bool = False
) -> Callable[[Mapping[str, float]], object]:
    """Runtime ``fk_verifier`` hook delegating to the live /compute_fk."""

    def fk_hook(state: Mapping[str, float]):
        runtime = holder.get("runtime")
        if runtime is None:
            return None
        verifier = runtime.compute_tcp_fk if tcp else runtime.compute_link6_fk
        return verifier(state, timeout_s=fk_timeout_s)

    return fk_hook


def _spin_up(
    node_name: str, fk_timeout_s: float
) -> tuple[object, object, object, threading.Thread, object]:
    """Create the node, background executor, runtime and executor adapter."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    from rocobrick.ros_moveit_executor import RosMoveItExecutor
    from rocobrick.ros_moveit_runtime import RosMoveItRuntime

    rclpy.init()
    node = rclpy.create_node(node_name)
    background = SingleThreadedExecutor()
    background.add_node(node)
    thread = threading.Thread(target=background.spin, daemon=True)
    thread.start()
    holder: dict = {}
    runtime = RosMoveItRuntime(
        node,
        fk_verifier=_make_fk_hook(holder, fk_timeout_s),
        tcp_fk_verifier=_make_fk_hook(holder, fk_timeout_s, tcp=True),
    )
    holder["runtime"] = runtime
    executor = RosMoveItExecutor(runtime)
    return runtime, executor, background, thread, node


def _spin_down(
    runtime: object, background: object, thread: threading.Thread, node: object
) -> None:
    """Orderly teardown: quiesce the spinning thread before destroying nodes."""
    import rclpy

    runtime.shutdown()
    background.shutdown()
    thread.join(timeout=THREAD_JOIN_TIMEOUT_S)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _spin_up_session(
    node_name: str, robots: tuple[str, ...], fk_timeout_s: float
) -> tuple[object, object, threading.Thread, object]:
    """Create ONE node hosting one isolated runtime per plan robot (A5)."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    from rocobrick.ros_moveit_runtime import RosMoveItRobotSession

    rclpy.init()
    node = rclpy.create_node(node_name)
    background = SingleThreadedExecutor()
    background.add_node(node)
    thread = threading.Thread(target=background.spin, daemon=True)
    thread.start()
    session = RosMoveItRobotSession(node, robots, fk_timeout_s=fk_timeout_s)
    return session, background, thread, node


def _spin_down_session(
    session: object, background: object, thread: threading.Thread, node: object
) -> None:
    """Orderly session teardown: quiesce the spin before destroying the node."""
    import rclpy

    session.shutdown()
    background.shutdown()
    thread.join(timeout=THREAD_JOIN_TIMEOUT_S)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _fallback_result(error: str) -> RuntimeOperationResult:
    """Schema-conform failure answer for an unparseable request."""
    return RuntimeOperationResult(
        request_id="",
        kind=OperationKind.READINESS,
        passed=False,
        execution={"sent": False},
        evidence={},
        error=error,
    )


def _fallback_session_result(
    error: str, *, robots: tuple[str, ...] = ()
) -> RosSessionResult:
    """Schema-conform failure answer for an unparseable session request."""
    return RosSessionResult(
        request_id="",
        robots=robots,
        readiness_by_robot={},
        all_ready=False,
        operations=(),
        passed=False,
        error=error,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the child entry; exactly one stdout JSON summary line is printed.

    Without ``--robots`` the historical single-runtime operation mode runs
    unchanged. With ``--robots`` the multi-robot session mode runs one
    isolated runtime per plan robot on one node and answers the strict
    session result document.
    """
    args = build_parser().parse_args(argv)
    try:
        robots = parse_robots(args.robots)
    except ValueError as error:
        result = _fallback_session_result(
            f"request could not be loaded: {error}", robots=()
        )
        write_result(args.result, result)
        print(
            json.dumps(_session_summary(result, args.result), sort_keys=True),
            flush=True,
        )
        return 2
    if args.serve_dir is not None:
        if not args.worker_token or robots is None or args.request or args.result:
            raise ValueError("serve mode requires token/robots and no one-shot paths")
        from rocobrick.execution.ros_session_worker import serve

        return serve(args.serve_dir, args.worker_token, robots, args.node_name)
    if args.request is None or args.result is None:
        raise ValueError("one-shot mode requires --request and --result")
    if robots is None:
        return _main_operation(args)
    return _main_session(args, robots)


def _main_operation(args: argparse.Namespace) -> int:
    """Single-runtime operation mode (unchanged R4.2 behavior)."""
    try:
        operation = load_request(args.request)
    except (OSError, OperationContractError, ValueError) as error:
        result = _fallback_result(f"request could not be loaded: {error}")
        write_result(args.result, result)
        print(json.dumps(_summary(result, args.result), sort_keys=True), flush=True)
        return 2
    runtime_object = executor_object = background = node = None
    thread: threading.Thread | None = None
    try:
        (
            runtime_object,
            executor_object,
            background,
            thread,
            node,
        ) = _spin_up(args.node_name, min(FK_TIMEOUT_S, operation.timeout_s))
        result = perform_operation(
            operation, runtime=runtime_object, executor=executor_object
        )
    except Exception as error:  # noqa: BLE001 - the child must always answer
        result = RuntimeOperationResult(
            request_id=operation.request_id,
            kind=operation.kind,
            passed=False,
            execution={"sent": False},
            evidence={},
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        if runtime_object is not None:
            try:
                _spin_down(runtime_object, background, thread, node)
            except Exception:  # pragma: no cover - defensive teardown
                pass
    write_result(args.result, result)
    print(json.dumps(_summary(result, args.result), sort_keys=True), flush=True)
    return 0 if result.passed else 1


def _main_session(args: argparse.Namespace, robots: tuple[str, ...]) -> int:
    """Multi-robot session mode: per-robot readiness plus robot-routed ops."""
    try:
        request = load_session_request(args.request)
    except (OSError, OperationContractError, ValueError) as error:
        result = _fallback_session_result(
            f"request could not be loaded: {error}", robots=robots
        )
        write_result(args.result, result)
        print(
            json.dumps(_session_summary(result, args.result), sort_keys=True),
            flush=True,
        )
        return 2
    if request.robots != robots:
        result = _fallback_session_result(
            "session request robots "
            f"{list(request.robots)} do not match --robots {list(robots)}",
            robots=robots,
        )
        write_result(args.result, result)
        print(
            json.dumps(_session_summary(result, args.result), sort_keys=True),
            flush=True,
        )
        return 2
    session = background = node = None
    thread: threading.Thread | None = None
    try:
        session, background, thread, node = _spin_up_session(
            args.node_name, robots, min(FK_TIMEOUT_S, request.timeout_s)
        )
        previous_sigint = signal.getsignal(signal.SIGINT)

        def _cancel_active_runtime(_signum: int, _frame: object) -> None:
            for item in request.operations:
                runtime = session.runtime_for(item.robot_id)
                cancel = getattr(runtime, "cancel_active", None)
                if callable(cancel):
                    cancel()

        signal.signal(signal.SIGINT, _cancel_active_runtime)
        try:
            result = perform_session(session, request)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
    except Exception as error:  # noqa: BLE001 - the child must always answer
        result = RosSessionResult(
            request_id=request.request_id,
            robots=request.robots,
            readiness_by_robot={},
            all_ready=False,
            operations=(),
            passed=False,
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        if session is not None:
            try:
                _spin_down_session(session, background, thread, node)
            except Exception:  # pragma: no cover - defensive teardown
                pass
    write_result(args.result, result)
    print(json.dumps(_session_summary(result, args.result), sort_keys=True), flush=True)
    return 0 if result.passed else 1


def _summary(result: RuntimeOperationResult, result_path: Path) -> dict[str, object]:
    """The single stdout summary line payload."""
    return {
        "schema": SUMMARY_SCHEMA,
        "result_schema": OPERATION_RESULT_SCHEMA,
        "request_id": result.request_id,
        "kind": str(result.kind.value),
        "passed": result.passed,
        "result_path": str(result_path),
        "error": result.error,
    }


def _session_summary(result: RosSessionResult, result_path: Path) -> dict[str, object]:
    """The single stdout summary line payload of the multi-robot session."""
    return {
        "schema": SESSION_SUMMARY_SCHEMA,
        "result_schema": SESSION_RESULT_SCHEMA,
        "request_id": result.request_id,
        "robots": list(result.robots),
        "all_ready": result.all_ready,
        "passed": result.passed,
        "result_path": str(result_path),
        "error": result.error,
    }


if __name__ == "__main__":  # pragma: no cover - container entry point
    raise SystemExit(main())
