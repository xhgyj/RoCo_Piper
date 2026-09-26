"""MoveIt execution protocol and result abstractions (transport-free).

Only the protocol/result layer lives here: a ROS-side executor implements
:class:`MoveItExecutor` for joint and link6 pose goals; the pure
:func:`is_success` predicate applies the P1 gates (terminal SUCCEEDED status,
MoveIt error code SUCCESS, at least MIN_WAYPOINTS planned points). No rclpy,
no action/message imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from rocobrick.compat import StrEnum
from rocobrick.moveit_contract import (
    MIN_WAYPOINTS,
    SUCCESS_CODE,
    JointTarget,
    Link6PoseTarget,
)


class ExecutionStatus(StrEnum):
    """Terminal and pre-terminal MoveGroup action outcomes."""

    ACCEPTED = "ACCEPTED"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"
    TIMED_OUT = "TIMED_OUT"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ExecutionResult:
    """One MoveGroup plan-and-execute outcome with its live evidence."""

    status: ExecutionStatus
    moveit_error_code: int | None = None
    planned_waypoints: int = 0
    command_messages: int = 0
    final_errors: Mapping[str, float] = field(default_factory=dict)
    details: Mapping[str, object] = field(default_factory=dict)
    # Additive action lifecycle evidence. Existing callers remain compatible.
    goal_accepted: bool | None = None
    terminal: bool = False
    safe_to_release: bool = False
    action_status: str | None = None
    cancel_evidence: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class MoveItExecutor(Protocol):
    """Transport boundary for MoveGroup goals; ROS nodes implement this."""

    def execute_joint(
        self, target: JointTarget, *, timeout_s: float
    ) -> ExecutionResult: ...

    def execute_link6_pose(
        self, target: Link6PoseTarget, *, timeout_s: float
    ) -> ExecutionResult: ...


def is_success(result: ExecutionResult) -> bool:
    """True only for SUCCEEDED + error code SUCCESS + at least MIN_WAYPOINTS."""
    return (
        result.status == ExecutionStatus.SUCCEEDED
        and result.moveit_error_code == SUCCESS_CODE
        and result.planned_waypoints >= MIN_WAYPOINTS
    )


__all__ = ["ExecutionResult", "ExecutionStatus", "MoveItExecutor", "is_success"]
