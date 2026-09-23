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
so an inadmissible target can never be encoded, transported, or executed. No
STAP/task parsing, no target derivation, no plan semantics, and no TCP
interpretation live here. ROS/Isaac imports are absent so the module imports
cleanly on the host and inside the container (Python 3.10 compatible through
:mod:`rocobrick.compat`).
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
    JointTarget,
    Link6PoseTarget,
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


class OperationKind(StrEnum):
    """Exact operation vocabulary exchanged across the container boundary."""

    READINESS = "READINESS"
    JOINT_TARGET = "JOINT_TARGET"
    LINK6_POSE = "LINK6_POSE"


class OperationContractError(ValueError):
    """Raised when a runtime operation document violates its exact schema."""


class OperationRejected(OperationContractError):  # noqa: N818 - domain vocabulary
    """Raised when strict admission refuses a target before serialization."""


def _is_finite(value: object) -> bool:
    """Return True when the value is a finite int/float (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


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


def target_for_operation(
    operation: RuntimeOperation,
) -> Union[JointTarget, Link6PoseTarget]:
    """Rebuild the admitted motion target carried by a motion operation.

    READINESS carries no target (this raises). Motion payloads are rebuilt
    through the formal dataclasses and re-checked with
    ``strict_action_admission``; a rejected target raises
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
    if operation.kind is OperationKind.JOINT_TARGET:
        target: JointTarget | Link6PoseTarget = joint_target_from_payload(
            operation.payload
        )
    else:
        target = link6_pose_target_from_payload(operation.payload)
    decision = strict_action_admission(target)
    if not decision.accepted:
        raise OperationRejected(
            f"operation target rejected by strict admission: {decision.reason}"
        )
    return target


__all__ = [
    "OPERATION_KEYS",
    "OPERATION_RESULT_KEYS",
    "OPERATION_RESULT_SCHEMA",
    "OPERATION_SCHEMA",
    "OperationContractError",
    "OperationKind",
    "OperationRejected",
    "RuntimeOperation",
    "RuntimeOperationResult",
    "joint_operation",
    "joint_target_from_payload",
    "link6_pose_operation",
    "link6_pose_target_from_payload",
    "readiness_operation",
    "target_for_operation",
]
