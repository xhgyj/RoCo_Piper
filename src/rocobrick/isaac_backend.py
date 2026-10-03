"""Bounded JSON operation contracts for the Isaac/MoveIt runtime session (R4.2).

The parent orchestrator (:mod:`rocobrick.runtime_session`, owned by
``piper-run``) and the container child entry
(:mod:`rocobrick.ros_session_entry`) communicate ONLY through strict JSON
documents: a :class:`RuntimeOperation` request and a
:class:`RuntimeOperationResult` result, both parsed with exact-key
validation so no Python handle, object, or implicit state ever crosses the
process boundary.

This module is pure data plus the accepted admission contract: the builders
run ``rocobrick.admission.strict_action_admission`` BEFORE any serialization,
so an inadmissible target can never be encoded, transported, or executed. The
no-motion scene operations (APPLY_PLANNING_SCENE, SCENE_QUERY) carry strictly
validated documents instead of targets. No STAP/task parsing, no target
derivation, no plan semantics, and no TCP interpretation live here. ROS/Isaac
imports are absent so the module imports cleanly on the host and inside the
container (Python 3.10 compatible through :mod:`rocobrick.compat`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

from rocobrick.admission import strict_action_admission
from rocobrick.compat import StrEnum
from rocobrick.moveit_contract import (
    ARM_JOINT_NAMES,
    GripperTarget,
    JointTarget,
    Link6PoseTarget,
    TcpPoseTarget,
)
from rocobrick.planning_scene import (
    SCHEMA as PLANNING_SCENE_SCHEMA,
)
from rocobrick.planning_scene import (
    PlanningSceneUpdate,
    box_from_aabb,
    box_geometry_digest,
    scene_digest,
)

OPERATION_SCHEMA = "rocobrick-runtime-operation/v1"
OPERATION_RESULT_SCHEMA = "rocobrick-runtime-operation-result/v1"

OPERATION_KEYS = frozenset({
    "schema",
    "kind",
    "payload",
    "timeout_s",
    "request_id",
})
OPERATION_RESULT_KEYS = frozenset({
    "schema",
    "request_id",
    "kind",
    "passed",
    "execution",
    "evidence",
    "error",
})

JOINT_PAYLOAD_KEYS = frozenset({"positions", "goal_tolerance"})
POSE_PAYLOAD_KEYS = frozenset({
    "frame_id",
    "position",
    "orientation_xyzw",
    "position_tolerance_m",
    "orientation_tolerance_rad",
    "tip_link",
    "tcp_status",
})
TCP_PAYLOAD_KEYS = POSE_PAYLOAD_KEYS | {"runtime_model_sha256", "source_sha256"}
GRIPPER_PAYLOAD_KEYS = frozenset({
    "opening_m",
    "duration_s",
    "goal_tolerance",
    "expected_opening_m",
})
SCENE_QUERY_PAYLOAD_KEYS = frozenset({
    "brick_physical_instance_ids",
    "connection_pairs",
    "include_gripper",
})
# Exact top-level keys of the document built from a
# rocobrick.planning_scene.PlanningSceneUpdate (dataclasses.asdict shape).
PLANNING_SCENE_PAYLOAD_KEYS = frozenset({
    "schema",
    "scene_id",
    "boxes",
    "expected_scene_digest",
    "allowed_collision_pairs",
})
PLANNING_SCENE_BOX_KEYS = frozenset({
    "object_id",
    "frame_id",
    "minimum_m",
    "maximum_m",
    "geometry_digest",
})


ATTACHED_SCENE_PAYLOAD_KEYS = frozenset({
    "scene_id", "object_id", "link_name", "frame_id",
    "minimum_m", "maximum_m", "geometry_digest", "touch_links",
})


DETACHED_SCENE_PAYLOAD_KEYS = frozenset({
    "scene_id", "object_id", "link_name", "operation",
})


class OperationKind(StrEnum):
    """Exact operation vocabulary exchanged across the container boundary."""

    READINESS = "READINESS"
    JOINT_TARGET = "JOINT_TARGET"
    LINK6_POSE = "LINK6_POSE"
    TCP_POSE = "TCP_POSE"
    GUARDED_TCP_POSE = "GUARDED_TCP_POSE"
    GRIPPER_TARGET = "GRIPPER_TARGET"
    APPLY_PLANNING_SCENE = "APPLY_PLANNING_SCENE"
    ATTACH_PLANNING_SCENE = "ATTACH_PLANNING_SCENE"
    SCENE_QUERY = "SCENE_QUERY"


class OperationContractError(ValueError):
    """Raised when a runtime operation document violates its exact schema."""


class OperationRejected(OperationContractError):  # noqa: N818 - domain vocabulary
    """Raised when strict admission refuses a target before serialization."""


def _is_finite(value: object) -> bool:
    """Return True when the value is a finite int/float (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _is_sha256_digest(value: object) -> bool:
    """True for a lowercase 64-character SHA-256 hex digest string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_exact_keys(
    document: Mapping[str, object],
    expected: frozenset[str],
    label: str,
) -> None:
    """Require the document to carry exactly the expected keys."""
    if not isinstance(document, Mapping):
        raise OperationContractError(f"{label} must be a JSON object")
    actual = frozenset(str(key) for key in document)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise OperationContractError(
            f"{label} must carry exactly {sorted(expected)}; "
            f"missing={missing} extra={extra}"
        )


@dataclass(frozen=True)
class RuntimeOperation:
    """One bounded runtime operation request (strict JSON document)."""

    kind: OperationKind
    payload: Mapping[str, object]
    timeout_s: float
    request_id: str

    def serialize(self) -> dict[str, object]:
        """Return the exact request document for the child entry."""
        return {
            "schema": OPERATION_SCHEMA,
            "kind": str(self.kind.value),
            "payload": dict(self.payload),
            "timeout_s": float(self.timeout_s),
            "request_id": str(self.request_id),
        }

    @classmethod
    def parse(cls, document: object) -> RuntimeOperation:
        """Parse a request document strictly; any deviation is rejected."""
        _require_exact_keys(document, OPERATION_KEYS, "runtime operation")
        assert isinstance(document, Mapping)  # noqa: S101 - narrowed above
        if document["schema"] != OPERATION_SCHEMA:
            raise OperationContractError(
                f"operation schema {document['schema']!r} is not "
                f"{OPERATION_SCHEMA!r}"
            )
        raw_kind = document["kind"]
        try:
            kind = OperationKind(str(raw_kind))
        except ValueError as error:
            raise OperationContractError(
                f"operation kind {raw_kind!r} is not one of "
                f"{[member.value for member in OperationKind]}"
            ) from error
        payload = document["payload"]
        if not isinstance(payload, Mapping):
            raise OperationContractError("operation payload must be a JSON object")
        timeout_s = document["timeout_s"]
        if not _is_finite(timeout_s) or float(timeout_s) <= 0.0:
            raise OperationContractError(
                f"operation timeout_s must be finite and positive, got {timeout_s!r}"
            )
        request_id = document["request_id"]
        if not isinstance(request_id, str) or not request_id:
            raise OperationContractError(
                "operation request_id must be a non-empty string"
            )
        return cls(
            kind=kind,
            payload=dict(payload),
            timeout_s=float(timeout_s),
            request_id=request_id,
        )


@dataclass(frozen=True)
class RuntimeOperationResult:
    """One bounded runtime operation result (strict JSON document)."""

    request_id: str
    kind: OperationKind
    passed: bool
    execution: Mapping[str, object]
    evidence: Mapping[str, object]
    error: str | None = None

    def serialize(self) -> dict[str, object]:
        """Return the exact result document for the parent orchestrator."""
        return {
            "schema": OPERATION_RESULT_SCHEMA,
            "request_id": str(self.request_id),
            "kind": str(self.kind.value),
            "passed": bool(self.passed),
            "execution": dict(self.execution),
            "evidence": dict(self.evidence),
            "error": None if self.error is None else str(self.error),
        }

    @classmethod
    def parse(cls, document: object) -> RuntimeOperationResult:
        """Parse a result document strictly; any deviation is rejected."""
        _require_exact_keys(document, OPERATION_RESULT_KEYS, "operation result")
        assert isinstance(document, Mapping)  # noqa: S101 - narrowed above
        if document["schema"] != OPERATION_RESULT_SCHEMA:
            raise OperationContractError(
                f"result schema {document['schema']!r} is not "
                f"{OPERATION_RESULT_SCHEMA!r}"
            )
        request_id = document["request_id"]
        if not isinstance(request_id, str) or not request_id:
            raise OperationContractError(
                "result request_id must be a non-empty string"
            )
        try:
            kind = OperationKind(str(document["kind"]))
        except ValueError as error:
            raise OperationContractError(
                f"result kind {document['kind']!r} is not one of "
                f"{[member.value for member in OperationKind]}"
            ) from error
        passed = document["passed"]
        if not isinstance(passed, bool):
            raise OperationContractError("result passed must be a boolean")
        execution = document["execution"]
        evidence = document["evidence"]
        if not isinstance(execution, Mapping) or not isinstance(evidence, Mapping):
            raise OperationContractError(
                "result execution and evidence must be JSON objects"
            )
        error = document["error"]
        if error is not None and not isinstance(error, str):
            raise OperationContractError("result error must be a string or null")
        return cls(
            request_id=request_id,
            kind=kind,
            passed=passed,
            execution=dict(execution),
            evidence=dict(evidence),
            error=error,
        )


def readiness_operation(*, timeout_s: float, request_id: str) -> RuntimeOperation:
    """Build the no-motion readiness operation request."""
    if not _is_finite(timeout_s) or float(timeout_s) <= 0.0:
        raise OperationContractError(
            f"timeout_s must be finite and positive, got {timeout_s!r}"
        )
    if not isinstance(request_id, str) or not request_id:
        raise OperationContractError("request_id must be a non-empty string")
    return RuntimeOperation(
        kind=OperationKind.READINESS,
        payload={},
        timeout_s=float(timeout_s),
        request_id=request_id,
    )


def joint_operation(
    target: JointTarget, timeout_s: float, request_id: str
) -> RuntimeOperation:
    """Admit a joint target, then serialize it as a JOINT_TARGET request.

    Raises:
        OperationContractError: If ``timeout_s`` or ``request_id`` are invalid.
        OperationRejected: If ``strict_action_admission`` refuses the target.
    """
    if not _is_finite(timeout_s) or float(timeout_s) <= 0.0:
        raise OperationContractError(
            f"timeout_s must be finite and positive, got {timeout_s!r}"
        )
    if not isinstance(request_id, str) or not request_id:
        raise OperationContractError("request_id must be a non-empty string")
    decision = strict_action_admission(target)
    if not decision.accepted:
        raise OperationRejected(
            f"joint target rejected before serialization: {decision.reason}"
        )
    payload: dict[str, object] = {
        "positions": {
            name: float(target.positions[name]) for name in ARM_JOINT_NAMES
        },
        "goal_tolerance": float(target.goal_tolerance),
    }
    return RuntimeOperation(
        kind=OperationKind.JOINT_TARGET,
        payload=payload,
        timeout_s=float(timeout_s),
        request_id=request_id,
    )


def link6_pose_operation(
    target: Link6PoseTarget, timeout_s: float, request_id: str
) -> RuntimeOperation:
    """Admit a link6 pose target, then serialize it as a LINK6_POSE request.

    Raises:
        OperationContractError: If ``timeout_s`` or ``request_id`` are invalid.
        OperationRejected: If ``strict_action_admission`` refuses the target.
    """
    if not _is_finite(timeout_s) or float(timeout_s) <= 0.0:
        raise OperationContractError(
            f"timeout_s must be finite and positive, got {timeout_s!r}"
        )
    if not isinstance(request_id, str) or not request_id:
        raise OperationContractError("request_id must be a non-empty string")
    decision = strict_action_admission(target)
    if not decision.accepted:
        raise OperationRejected(
            f"link6 pose target rejected before serialization: {decision.reason}"
        )
    payload: dict[str, object] = {
        "frame_id": str(target.frame_id),
        "position": [float(value) for value in target.position],
        "orientation_xyzw": [float(value) for value in target.orientation_xyzw],
        "position_tolerance_m": float(target.position_tolerance_m),
        "orientation_tolerance_rad": float(target.orientation_tolerance_rad),
        "tip_link": str(target.tip_link),
        "tcp_status": str(target.tcp_status),
    }
    return RuntimeOperation(
        kind=OperationKind.LINK6_POSE,
        payload=payload,
        timeout_s=float(timeout_s),
        request_id=request_id,
    )


def tcp_pose_operation(
    target: TcpPoseTarget, timeout_s: float, request_id: str
) -> RuntimeOperation:
    """Admit a calibrated TCP pose target, then serialize it as TCP_POSE.

    Raises:
        OperationContractError: If ``timeout_s`` or ``request_id`` are invalid.
        OperationRejected: If ``strict_action_admission`` refuses the target.
    """
    if not _is_finite(timeout_s) or float(timeout_s) <= 0.0:
        raise OperationContractError(
            f"timeout_s must be finite and positive, got {timeout_s!r}"
        )
    if not isinstance(request_id, str) or not request_id:
        raise OperationContractError("request_id must be a non-empty string")
    decision = strict_action_admission(target)
    if not decision.accepted:
        raise OperationRejected(
            f"tcp pose target rejected before serialization: {decision.reason}"
        )
    payload: dict[str, object] = {
        "frame_id": str(target.frame_id),
        "position": [float(value) for value in target.position],
        "orientation_xyzw": [float(value) for value in target.orientation_xyzw],
        "position_tolerance_m": float(target.position_tolerance_m),
        "orientation_tolerance_rad": float(target.orientation_tolerance_rad),
        "tip_link": str(target.tip_link),
        "tcp_status": str(target.tcp_status),
        "runtime_model_sha256": str(target.runtime_model_sha256),
        "source_sha256": str(target.source_sha256),
    }
    return RuntimeOperation(
        kind=OperationKind.TCP_POSE,
        payload=payload,
        timeout_s=float(timeout_s),
        request_id=request_id,
    )


def guarded_tcp_operation(
    target: TcpPoseTarget, guard: Mapping[str, object], timeout_s: float, request_id: str
) -> RuntimeOperation:
    """Build a guarded contact target with unchanged TCP admission.

    Returns:
        A strictly validated robot-scoped guarded TCP operation.
    """
    from rocobrick.skills.contact import validate_guard

    admitted = tcp_pose_operation(target, timeout_s, request_id)
    validate_guard(guard)
    return RuntimeOperation(OperationKind.GUARDED_TCP_POSE,
                            {"target": admitted.payload, "guard": dict(guard)},
                            admitted.timeout_s, admitted.request_id)


def gripper_operation(
    target: GripperTarget, timeout_s: float, request_id: str
) -> RuntimeOperation:
    """Admit a gripper target, then serialize it as a GRIPPER_TARGET request.

    Raises:
        OperationContractError: If ``timeout_s`` or ``request_id`` are invalid.
        OperationRejected: If ``strict_action_admission`` refuses the target.
    """
    if not _is_finite(timeout_s) or float(timeout_s) <= 0.0:
        raise OperationContractError(
            f"timeout_s must be finite and positive, got {timeout_s!r}"
        )
    if not isinstance(request_id, str) or not request_id:
        raise OperationContractError("request_id must be a non-empty string")
    decision = strict_action_admission(target)
    if not decision.accepted:
        raise OperationRejected(
            f"gripper target rejected before serialization: {decision.reason}"
        )
    payload: dict[str, object] = {
        "opening_m": float(target.opening_m),
        "duration_s": float(target.duration_s),
        "goal_tolerance": float(target.goal_tolerance),
        "expected_opening_m": [
            float(target.expected_opening_m[0]),
            float(target.expected_opening_m[1]),
        ],
    }
    return RuntimeOperation(
        kind=OperationKind.GRIPPER_TARGET,
        payload=payload,
        timeout_s=float(timeout_s),
        request_id=request_id,
    )


def attached_planning_scene_operation(
    payload: Mapping[str, object], timeout_s: float, request_id: str
) -> RuntimeOperation:
    """Build one strict robot-attached collision-scene operation."""
    if not _is_finite(timeout_s) or float(timeout_s) <= 0.0:
        raise OperationContractError("timeout_s must be finite and positive")
    if not isinstance(request_id, str) or not request_id:
        raise OperationContractError("request_id must be a non-empty string")
    _require_exact_keys(payload, ATTACHED_SCENE_PAYLOAD_KEYS, "attached scene payload")
    for key in ("scene_id", "object_id", "link_name", "frame_id"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise OperationContractError(f"attached scene {key} must be non-empty")
    if not _is_sha256_digest(payload["geometry_digest"]):
        raise OperationContractError("attached scene geometry_digest is invalid")
    minimum = _require_finite_sequence(payload["minimum_m"], 3, "minimum_m")
    maximum = _require_finite_sequence(payload["maximum_m"], 3, "maximum_m")
    if any(low >= high for low, high in zip(minimum, maximum)):
        raise OperationContractError("attached scene maximum_m must exceed minimum_m")
    links = payload["touch_links"]
    if (
        not isinstance(links, (list, tuple))
        or not links
        or any(not isinstance(link, str) or not link for link in links)
        or len(set(links)) != len(links)
    ):
        raise OperationContractError("attached scene touch_links must be unique strings")
    expected = box_geometry_digest(
        object_id=str(payload["object_id"]), frame_id=str(payload["frame_id"]),
        minimum_m=minimum, maximum_m=maximum,
    )
    if expected != payload["geometry_digest"]:
        raise OperationContractError("attached scene geometry_digest does not match")
    return RuntimeOperation(
        kind=OperationKind.ATTACH_PLANNING_SCENE,
        payload=dict(payload), timeout_s=float(timeout_s), request_id=request_id,
    )


def detached_planning_scene_operation(
    payload: Mapping[str, object], timeout_s: float, request_id: str
) -> RuntimeOperation:
    """Build one strict detach of a robot-attached collision object.

    The object leaves the robot state entirely (MoveIt ``REMOVE`` semantics);
    the caller republishes it as a world collision object in a separate
    ``APPLY_PLANNING_SCENE`` update when the placed geometry must stay in
    the planning scene.

    Returns:
        The validated detach runtime operation.
    """
    if not _is_finite(timeout_s) or float(timeout_s) <= 0.0:
        raise OperationContractError("timeout_s must be finite and positive")
    if not isinstance(request_id, str) or not request_id:
        raise OperationContractError("request_id must be a non-empty string")
    _require_exact_keys(payload, DETACHED_SCENE_PAYLOAD_KEYS, "detached scene payload")
    for key in ("scene_id", "object_id", "link_name"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise OperationContractError(f"detached scene {key} must be non-empty")
    if payload["operation"] != "detach":
        raise OperationContractError(
            "detached scene operation must be exactly 'detach'"
        )
    return RuntimeOperation(
        kind=OperationKind.ATTACH_PLANNING_SCENE,
        payload=dict(payload), timeout_s=float(timeout_s), request_id=request_id,
    )


def planning_scene_operation(
    update: Mapping[str, object], timeout_s: float, request_id: str
) -> RuntimeOperation:
    """Validate a planning-scene update document, then serialize it verbatim.

    The payload is the EXACT top-level document of the accepted
    :class:`rocobrick.planning_scene.PlanningSceneUpdate` (its
    ``dataclasses.asdict`` shape); it is re-validated strictly and carried
    unchanged. No motion is admitted here, so strict admission does not run.

    Raises:
        OperationContractError: If ``timeout_s``, ``request_id`` or the
            update document violate their exact schemas.
    """
    if not _is_finite(timeout_s) or float(timeout_s) <= 0.0:
        raise OperationContractError(
            f"timeout_s must be finite and positive, got {timeout_s!r}"
        )
    if not isinstance(request_id, str) or not request_id:
        raise OperationContractError("request_id must be a non-empty string")
    planning_scene_update_from_payload(update)
    return RuntimeOperation(
        kind=OperationKind.APPLY_PLANNING_SCENE,
        payload=dict(update),
        timeout_s=float(timeout_s),
        request_id=request_id,
    )


def scene_query_operation(
    brick_physical_instance_ids: object,
    connection_pairs: object,
    *,
    include_gripper: bool,
    timeout_s: float,
    request_id: str,
) -> RuntimeOperation:
    """Serialize a SCENE_QUERY request (no motion, no admission).

    Raises:
        OperationContractError: If ``timeout_s``, ``request_id`` or the query
            selections violate their exact schemas.
    """
    if not _is_finite(timeout_s) or float(timeout_s) <= 0.0:
        raise OperationContractError(
            f"timeout_s must be finite and positive, got {timeout_s!r}"
        )
    if not isinstance(request_id, str) or not request_id:
        raise OperationContractError("request_id must be a non-empty string")
    if not isinstance(brick_physical_instance_ids, (list, tuple)) or not (
        brick_physical_instance_ids
    ):
        raise OperationContractError(
            "brick_physical_instance_ids must be a non-empty list of strings"
        )
    if not all(
        isinstance(value, str) and value
        for value in brick_physical_instance_ids
    ):
        raise OperationContractError(
            "brick_physical_instance_ids must be a non-empty list of "
            "non-empty strings"
        )
    if not isinstance(connection_pairs, (list, tuple)):
        raise OperationContractError(
            "connection_pairs must be a list of exactly-two-string lists"
        )
    for index, pair in enumerate(connection_pairs):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(value, str) and value for value in pair)
        ):
            raise OperationContractError(
                f"connection_pairs[{index}] must be a list of exactly two "
                "non-empty strings"
            )
    if not isinstance(include_gripper, bool):
        raise OperationContractError("include_gripper must be a boolean")
    payload: dict[str, object] = {
        "brick_physical_instance_ids": [
            str(value) for value in brick_physical_instance_ids
        ],
        "connection_pairs": [
            [str(pair[0]), str(pair[1])] for pair in connection_pairs
        ],
        "include_gripper": bool(include_gripper),
    }
    return RuntimeOperation(
        kind=OperationKind.SCENE_QUERY,
        payload=payload,
        timeout_s=float(timeout_s),
        request_id=request_id,
    )


def _require_finite_sequence(values: object, length: int, label: str) -> tuple:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise OperationContractError(
            f"{label} must be a sequence of exactly {length} numbers"
        )
    if not all(_is_finite(value) for value in values):
        raise OperationContractError(f"{label} must carry only finite numbers")
    return tuple(float(value) for value in values)


def joint_target_from_payload(payload: Mapping[str, object]) -> JointTarget:
    """Rebuild a ``JointTarget`` from a strict JOINT_TARGET payload."""
    _require_exact_keys(payload, JOINT_PAYLOAD_KEYS, "joint payload")
    positions_document = payload["positions"]
    if not isinstance(positions_document, Mapping):
        raise OperationContractError("joint payload positions must be an object")
    names = frozenset(str(key) for key in positions_document)
    expected_names = frozenset(ARM_JOINT_NAMES)
    if names != expected_names:
        raise OperationContractError(
            "joint payload positions must carry exactly the six canonical arm "
            f"joints; missing={sorted(expected_names - names)} "
            f"extra={sorted(names - expected_names)}"
        )
    if not all(_is_finite(value) for value in positions_document.values()):
        raise OperationContractError("joint payload positions must be finite")
    goal_tolerance = payload["goal_tolerance"]
    if not _is_finite(goal_tolerance) or float(goal_tolerance) <= 0.0:
        raise OperationContractError(
            f"joint payload goal_tolerance must be finite and positive, "
            f"got {goal_tolerance!r}"
        )
    return JointTarget(
        positions={
            str(name): float(positions_document[name]) for name in ARM_JOINT_NAMES
        },
        goal_tolerance=float(goal_tolerance),
    )


def link6_pose_target_from_payload(payload: Mapping[str, object]) -> Link6PoseTarget:
    """Rebuild a ``Link6PoseTarget`` from a strict LINK6_POSE payload."""
    _require_exact_keys(payload, POSE_PAYLOAD_KEYS, "pose payload")
    for label in ("frame_id", "tip_link", "tcp_status"):
        if not isinstance(payload[label], str) or not payload[label]:
            raise OperationContractError(f"pose payload {label} must be a string")
    position = _require_finite_sequence(payload["position"], 3, "position")
    orientation = _require_finite_sequence(
        payload["orientation_xyzw"], 4, "orientation_xyzw"
    )
    tolerances = []
    for label in ("position_tolerance_m", "orientation_tolerance_rad"):
        value = payload[label]
        if not _is_finite(value) or float(value) <= 0.0:
            raise OperationContractError(
                f"pose payload {label} must be finite and positive, got {value!r}"
            )
        tolerances.append(float(value))
    return Link6PoseTarget(
        frame_id=str(payload["frame_id"]),
        position=position,  # type: ignore[arg-type]
        orientation_xyzw=orientation,  # type: ignore[arg-type]
        position_tolerance_m=tolerances[0],
        orientation_tolerance_rad=tolerances[1],
        tip_link=str(payload["tip_link"]),
        tcp_status=str(payload["tcp_status"]),
    )


def tcp_pose_target_from_payload(payload: Mapping[str, object]) -> TcpPoseTarget:
    """Rebuild a ``TcpPoseTarget`` from a strict TCP_POSE payload."""
    _require_exact_keys(payload, TCP_PAYLOAD_KEYS, "tcp pose payload")
    for label in ("frame_id", "tip_link", "tcp_status"):
        if not isinstance(payload[label], str) or not payload[label]:
            raise OperationContractError(f"tcp pose payload {label} must be a string")
    for label in ("runtime_model_sha256", "source_sha256"):
        if not _is_sha256_digest(payload[label]):
            raise OperationContractError(
                f"tcp pose payload {label} must be a lowercase SHA-256 digest"
            )
    position = _require_finite_sequence(payload["position"], 3, "position")
    orientation = _require_finite_sequence(
        payload["orientation_xyzw"], 4, "orientation_xyzw"
    )
    tolerances = []
    for label in ("position_tolerance_m", "orientation_tolerance_rad"):
        value = payload[label]
        if not _is_finite(value) or float(value) <= 0.0:
            raise OperationContractError(
                f"tcp pose payload {label} must be finite and positive, "
                f"got {value!r}"
            )
        tolerances.append(float(value))
    return TcpPoseTarget(
        frame_id=str(payload["frame_id"]),
        position=position,  # type: ignore[arg-type]
        orientation_xyzw=orientation,  # type: ignore[arg-type]
        position_tolerance_m=tolerances[0],
        orientation_tolerance_rad=tolerances[1],
        tip_link=str(payload["tip_link"]),
        tcp_status=str(payload["tcp_status"]),
        runtime_model_sha256=str(payload["runtime_model_sha256"]),
        source_sha256=str(payload["source_sha256"]),
    )


def gripper_target_from_payload(payload: Mapping[str, object]) -> GripperTarget:
    """Rebuild a ``GripperTarget`` from a strict GRIPPER_TARGET payload."""
    _require_exact_keys(payload, GRIPPER_PAYLOAD_KEYS, "gripper payload")
    scalars = []
    for label in ("opening_m", "duration_s", "goal_tolerance"):
        value = payload[label]
        if not _is_finite(value):
            raise OperationContractError(
                f"gripper payload {label} must be a finite number, got {value!r}"
            )
        scalars.append(float(value))
    window_document = payload["expected_opening_m"]
    if not isinstance(window_document, (list, tuple)) or len(window_document) != 2:
        raise OperationContractError(
            "gripper payload expected_opening_m must be exactly two numbers"
        )
    if not all(_is_finite(value) for value in window_document):
        raise OperationContractError(
            "gripper payload expected_opening_m must carry only finite numbers"
        )
    low, high = (float(value) for value in window_document)
    if low > high:
        raise OperationContractError(
            "gripper payload expected_opening_m must satisfy lo <= hi, "
            f"got [{low}, {high}]"
        )
    return GripperTarget(
        opening_m=scalars[0],
        duration_s=scalars[1],
        goal_tolerance=scalars[2],
        expected_opening_m=(low, high),
    )


def scene_query_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Normalize a strict SCENE_QUERY payload into its canonical selection."""
    _require_exact_keys(payload, SCENE_QUERY_PAYLOAD_KEYS, "scene query payload")
    ids_document = payload["brick_physical_instance_ids"]
    if not isinstance(ids_document, (list, tuple)) or not ids_document:
        raise OperationContractError(
            "scene query payload brick_physical_instance_ids must be a "
            "non-empty list of strings"
        )
    if not all(isinstance(value, str) and value for value in ids_document):
        raise OperationContractError(
            "scene query payload brick_physical_instance_ids must carry only "
            "non-empty strings"
        )
    pairs_document = payload["connection_pairs"]
    if not isinstance(pairs_document, (list, tuple)):
        raise OperationContractError(
            "scene query payload connection_pairs must be a list of "
            "exactly-two-string lists"
        )
    pairs: list[list[str]] = []
    for index, pair in enumerate(pairs_document):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(value, str) and value for value in pair)
        ):
            raise OperationContractError(
                f"scene query payload connection_pairs[{index}] must be a "
                "list of exactly two non-empty strings"
            )
        pairs.append([str(pair[0]), str(pair[1])])
    include_gripper = payload["include_gripper"]
    if not isinstance(include_gripper, bool):
        raise OperationContractError(
            "scene query payload include_gripper must be a boolean"
        )
    return {
        "brick_physical_instance_ids": [str(value) for value in ids_document],
        "connection_pairs": pairs,
        "include_gripper": bool(include_gripper),
    }


def _planning_scene_box_from_payload(
    box_document: object, index: int
) -> object:
    """Validate one serialized box and rebuild it through ``box_from_aabb``.

    The exact-key, string, finiteness and digest checks reuse the frozen
    planning-scene contract: ``box_from_aabb`` recomputes the canonical box
    digest, so a tampered digest or coordinate pair is rejected here without
    duplicating any geometry logic.
    """
    _require_exact_keys(
        box_document, PLANNING_SCENE_BOX_KEYS, f"boxes[{index}]"
    )
    assert isinstance(box_document, Mapping)  # noqa: S101 - narrowed above
    for label in ("object_id", "frame_id"):
        value = box_document[label]
        if not isinstance(value, str) or not value:
            raise OperationContractError(
                f"planning scene payload boxes[{index}].{label} must be a "
                "non-empty string"
            )
    if not _is_sha256_digest(box_document["geometry_digest"]):
        raise OperationContractError(
            f"planning scene payload boxes[{index}].geometry_digest must be "
            "a lowercase SHA-256 digest"
        )
    minimum = _require_finite_sequence(
        box_document["minimum_m"], 3, f"boxes[{index}].minimum_m"
    )
    maximum = _require_finite_sequence(
        box_document["maximum_m"], 3, f"boxes[{index}].maximum_m"
    )
    try:
        return box_from_aabb(
            object_id=str(box_document["object_id"]),
            frame_id=str(box_document["frame_id"]),
            minimum_m=minimum,  # type: ignore[arg-type]
            maximum_m=maximum,  # type: ignore[arg-type]
        )
    except ValueError as error:
        raise OperationContractError(
            f"planning scene payload boxes[{index}] is not a valid "
            f"axis-aligned box: {error}"
        ) from error


def planning_scene_update_from_payload(
    payload: Mapping[str, object],
) -> PlanningSceneUpdate:
    """Re-validate a serialized planning-scene update and rebuild it.

    The document must be the EXACT top-level shape emitted for a
    :class:`rocobrick.planning_scene.PlanningSceneUpdate`; every box is
    rebuilt through the frozen contract (which re-digests the geometry) and
    the expected scene digest is re-derived with :func:`scene_digest`, so a
    tampered claim can never pass.

    Raises:
        OperationContractError: On any exact-key, type, digest or
            content mismatch.
    """
    _require_exact_keys(
        payload, PLANNING_SCENE_PAYLOAD_KEYS, "planning scene payload"
    )
    assert isinstance(payload, Mapping)  # noqa: S101 - narrowed above
    if payload["schema"] != PLANNING_SCENE_SCHEMA:
        raise OperationContractError(
            f"planning scene payload schema {payload['schema']!r} is not "
            f"{PLANNING_SCENE_SCHEMA!r}"
        )
    scene_id = payload["scene_id"]
    if not isinstance(scene_id, str) or not scene_id:
        raise OperationContractError(
            "planning scene payload scene_id must be a non-empty string"
        )
    expected_digest = payload["expected_scene_digest"]
    if not _is_sha256_digest(expected_digest):
        raise OperationContractError(
            "planning scene payload expected_scene_digest must be a "
            "lowercase SHA-256 digest"
        )
    boxes_document = payload["boxes"]
    if not isinstance(boxes_document, (list, tuple)):
        raise OperationContractError(
            "planning scene payload boxes must be an array"
        )
    boxes = tuple(
        _planning_scene_box_from_payload(box_document, index)
        for index, box_document in enumerate(boxes_document)
    )
    pairs_document = payload["allowed_collision_pairs"]
    if not isinstance(pairs_document, (list, tuple)):
        raise OperationContractError(
            "planning scene payload allowed_collision_pairs must be an array"
        )
    pairs: list[tuple[str, str]] = []
    for index, pair in enumerate(pairs_document):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(value, str) and value for value in pair)
        ):
            raise OperationContractError(
                f"planning scene payload allowed_collision_pairs[{index}] "
                "must be a pair of non-empty strings"
            )
        pairs.append((str(pair[0]), str(pair[1])))
    try:
        update = PlanningSceneUpdate(
            scene_id=scene_id,
            boxes=boxes,  # type: ignore[arg-type]
            expected_scene_digest=str(expected_digest),
            allowed_collision_pairs=tuple(pairs),
        )
    except ValueError as error:
        raise OperationContractError(
            f"planning scene payload is not a valid scene update: {error}"
        ) from error
    if scene_digest(update) != str(expected_digest):
        raise OperationContractError(
            "planning scene payload expected_scene_digest does not match "
            "the scene content"
        )
    return update


def target_for_operation(
    operation: RuntimeOperation,
) -> Union[JointTarget, Link6PoseTarget, TcpPoseTarget, GripperTarget]:
    """Rebuild the admitted motion target carried by a motion operation.

    READINESS, APPLY_PLANNING_SCENE and SCENE_QUERY carry no motion target
    (this raises). Motion payloads are rebuilt through the formal dataclasses
    and re-checked with ``strict_action_admission``; a rejected target raises
    :class:`OperationRejected` so no inadmissible motion is ever sent.

    Raises:
        OperationContractError: If the payload violates its exact schema or
            the kind carries no target.
        OperationRejected: If strict admission refuses the rebuilt target.
    """
    if operation.kind is OperationKind.READINESS:
        raise OperationContractError(
            "a READINESS operation carries no motion target"
        )
    if operation.kind in (
        OperationKind.APPLY_PLANNING_SCENE,
        OperationKind.SCENE_QUERY,
    ):
        raise OperationContractError(
            f"a {str(operation.kind.value)} operation carries no motion target"
        )
    if operation.kind is OperationKind.JOINT_TARGET:
        target: JointTarget | Link6PoseTarget | TcpPoseTarget | GripperTarget = (
            joint_target_from_payload(operation.payload)
        )
    elif operation.kind is OperationKind.LINK6_POSE:
        target = link6_pose_target_from_payload(operation.payload)
    elif operation.kind is OperationKind.TCP_POSE:
        target = tcp_pose_target_from_payload(operation.payload)
    else:
        target = gripper_target_from_payload(operation.payload)
    decision = strict_action_admission(target)
    if not decision.accepted:
        raise OperationRejected(
            f"operation target rejected by strict admission: {decision.reason}"
        )
    return target


__all__ = [
    "GRIPPER_PAYLOAD_KEYS",
    "OPERATION_KEYS",
    "OPERATION_RESULT_KEYS",
    "OPERATION_RESULT_SCHEMA",
    "OPERATION_SCHEMA",
    "PLANNING_SCENE_BOX_KEYS",
    "PLANNING_SCENE_PAYLOAD_KEYS",
    "SCENE_QUERY_PAYLOAD_KEYS",
    "TCP_PAYLOAD_KEYS",
    "ATTACHED_SCENE_PAYLOAD_KEYS",
    "OperationContractError",
    "OperationKind",
    "OperationRejected",
    "RuntimeOperation",
    "RuntimeOperationResult",
    "attached_planning_scene_operation",
    "gripper_operation",
    "gripper_target_from_payload",
    "joint_operation",
    "joint_target_from_payload",
    "link6_pose_operation",
    "link6_pose_target_from_payload",
    "planning_scene_operation",
    "planning_scene_update_from_payload",
    "readiness_operation",
    "scene_query_from_payload",
    "scene_query_operation",
    "target_for_operation",
    "tcp_pose_operation",
    "tcp_pose_target_from_payload",
]
