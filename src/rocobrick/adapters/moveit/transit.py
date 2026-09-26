"""TRANSIT target resolution and ROS session motion adapter."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rocobrick.isaac_backend import tcp_pose_operation
from rocobrick.moveit_contract import TcpPoseTarget
from rocobrick.multi_robot_scheduler import JSONValue
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skill_targets import (
    RobotBaseFrame,
    SkillParams,
    TcpProvenance,
    load_robot_base_frame,
    load_skill_params,
    transit_target,
)


class MotionSession(Protocol):
    """Minimal multi-robot session boundary used by the adapter."""

    def run_motion_operation(
        self, robot_id: str, operation: object
    ) -> Mapping[str, object]:
        """Dispatch one validated runtime motion operation."""


@dataclass(frozen=True)
class MotionExecution:
    """Strict motion outcome normalized for the formal operation executor."""

    passed: bool
    terminal: bool
    safe_to_release: bool
    motion_sent: bool
    evidence: Mapping[str, JSONValue]
    reason: str | None = None


@dataclass(frozen=True)
class TransitContext:
    """Calibration and skill parameters needed for one robot's TRANSIT."""

    base: RobotBaseFrame
    provenance: TcpProvenance
    params: SkillParams


def load_transit_contexts(
    config_path: Path,
    skill_params_path: Path,
    robot_ids: Sequence[str],
    *,
    runtime_model_sha256: str,
    source_sha256: str,
) -> dict[str, TransitContext]:
    """Build immutable TRANSIT contexts from the authorized run inputs.

    Target generation is intentionally downstream of TCP authorization: both
    provenance digests are required and validated before any adapter exists.
    The base frame and numeric skill parameters are read once for the run so
    every target is derived from the same configuration snapshot.

    Returns:
        One calibrated transit context per validated robot.
    """
    if (
        not isinstance(runtime_model_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", runtime_model_sha256) is None
    ):
        raise ValueError("runtime_model_sha256 must be a lowercase SHA-256 digest")
    if (
        not isinstance(source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    params = load_skill_params(Path(skill_params_path))
    contexts: dict[str, TransitContext] = {}
    for robot_id in robot_ids:
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robot_ids must contain non-empty strings")
        contexts[robot_id] = TransitContext(
            base=load_robot_base_frame(Path(config_path), robot_id),
            provenance=TcpProvenance(
                runtime_model_sha256=runtime_model_sha256,
                source_sha256=source_sha256,
            ),
            params=params,
        )
    return contexts


def _aabb(
    scene: Mapping[str, object], physical_instance_id: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bricks = scene.get("bricks")
    if not isinstance(bricks, Mapping):
        raise ValueError("scene has no bricks mapping")
    brick = bricks.get(physical_instance_id)
    if not isinstance(brick, Mapping):
        raise ValueError(f"brick {physical_instance_id!r} is missing from scene")
    minimum = brick.get("aabb_min")
    maximum = brick.get("aabb_max")
    if (
        not isinstance(minimum, (list, tuple))
        or not isinstance(maximum, (list, tuple))
        or len(minimum) != 3
        or len(maximum) != 3
    ):
        raise ValueError(f"brick {physical_instance_id!r} has no valid AABB")
    try:
        low = tuple(float(value) for value in minimum)
        high = tuple(float(value) for value in maximum)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"brick {physical_instance_id!r} AABB is not numeric"
        ) from error
    if any(low[index] > high[index] for index in range(3)):
        raise ValueError(f"brick {physical_instance_id!r} AABB is inverted")
    return low, high  # type: ignore[return-value]


def resolve_part_transit_target(
    scene: Mapping[str, object], physical_instance_id: str, context: TransitContext
) -> TcpPoseTarget:
    """Resolve a live part AABB into a calibrated TCP transit target.

    Returns:
        The robot-scoped TCP pose above the measured part.
    """
    minimum, maximum = _aabb(scene, physical_instance_id)
    return transit_target(
        minimum, maximum, context.base, context.provenance, context.params
    )


def resolve_transit_target(
    scene: Mapping[str, object], operation: RobotOperation, context: TransitContext
) -> TcpPoseTarget:
    """Resolve the currently supported TRANSIT destination.

    Part destinations are derived from the live measured AABB. Connection
    destinations are deliberately rejected until the scene carries metric
    connection geometry; its current boolean connection map cannot define a
    TCP pose.

    Returns:
        The supported destination's calibrated TCP target.
    """
    target = operation.target
    if target is None:
        raise ValueError("TRANSIT requires a target")
    values = target.values
    destination = values.get("to")
    if not isinstance(destination, Mapping):
        raise ValueError("TRANSIT target.to is missing")
    kind = destination.get("kind")
    if kind == "part":
        physical_instance_id = destination.get("physical_instance_id")
        if not isinstance(physical_instance_id, str) or not physical_instance_id:
            raise ValueError("TRANSIT part destination has no physical_instance_id")
        return resolve_part_transit_target(scene, physical_instance_id, context)
    if kind == "connection":
        raise ValueError("CONNECTION_GEOMETRY_UNAVAILABLE")
    raise ValueError(f"unsupported TRANSIT destination kind {kind!r}")


class SessionTcpMotionAdapter:
    """Synchronous adapter from a formal TRANSIT target to ROS session TCP_POSE."""

    def __init__(self, session: MotionSession, robot_id: str) -> None:
        """Bind the motion session to one validated robot identity."""
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robot_id must be a non-empty string")
        self._session = session
        self._robot_id = robot_id

    def execute(
        self, operation: RobotOperation, target: TcpPoseTarget
    ) -> MotionExecution:
        """Execute once and require final live FK proof.

        Returns:
            The terminal motion result with its original runtime evidence.
        """
        runtime_operation = tcp_pose_operation(
            target, operation_timeout(operation), operation.operation_id
        )
        record = self._session.run_motion_operation(self._robot_id, runtime_operation)
        result = record.get("result")
        if not isinstance(result, Mapping):
            return MotionExecution(
                False, False, False, bool(record.get("motion_sent")), {},
                "MALFORMED_RUNTIME_RESULT",
            )
        execution = result.get("execution")
        evidence: dict[str, JSONValue] = {}
        if isinstance(result.get("evidence"), Mapping):
            evidence["runtime"] = dict(result["evidence"])  # type: ignore[assignment]
        if not isinstance(execution, Mapping):
            return MotionExecution(
                False, False, False, bool(record.get("motion_sent")), evidence,
                "MALFORMED_EXECUTION_RESULT",
            )
        evidence["execution"] = dict(execution)  # type: ignore[assignment]
        final_errors = execution.get("final_errors")
        details = execution.get("details")
        pose_ok = False
        if (
            execution.get("goal_accepted") is True
            and execution.get("action_status") == "SUCCEEDED"
            and isinstance(final_errors, Mapping)
            and isinstance(details, Mapping)
            and details.get("settled_at_rest") is True
        ):
            position_error = final_errors.get("position_error_m")
            orientation_error = final_errors.get("orientation_error_rad")
            pose_ok = (
                isinstance(position_error, (int, float))
                and not isinstance(position_error, bool)
                and math.isfinite(float(position_error))
                and float(position_error) <= target.position_tolerance_m
                and isinstance(orientation_error, (int, float))
                and not isinstance(orientation_error, bool)
                and math.isfinite(float(orientation_error))
                and float(orientation_error) <= target.orientation_tolerance_rad
            )
        evidence["final_pose_feedback_confirmed"] = pose_ok
        sent_value = execution.get("command_messages", 0)
        sent = (
            isinstance(sent_value, (int, float))
            and not isinstance(sent_value, bool)
            and sent_value >= 1
        )
        terminal = execution.get("terminal") is True
        safe = execution.get("safe_to_release") is True
        passed = (
            result.get("passed") is True
            and execution.get("is_success") is True
            and pose_ok
        )
        return MotionExecution(
            passed,
            terminal,
            safe,
            sent,
            evidence,
            None
            if passed
            else str(result.get("error") or "TRANSIT_FINAL_FEEDBACK_UNCONFIRMED"),
        )


def operation_timeout(operation: RobotOperation) -> float:
    value = operation.parameters.get("timeout_s", 30.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) <= 0
    ):
        return 30.0
    return float(value)


__all__ = [
    "MotionExecution",
    "MotionSession",
    "SessionTcpMotionAdapter",
    "TransitContext",
    "load_transit_contexts",
    "resolve_part_transit_target",
    "resolve_transit_target",
]
