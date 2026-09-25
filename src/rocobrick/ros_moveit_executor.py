"""MoveItExecutor implementation over the ROS MoveIt runtime adapter (R2).

Thin transport boundary implementing
:class:`rocobrick.moveit_execution.MoveItExecutor` (and its R5.A calibrated
TCP extension :class:`TcpMoveItExecutor`) on top of
:class:`rocobrick.ros_moveit_runtime.RosMoveItRuntime`. Strict R1 admission
(``strict_action_admission``) gates the executor: inadmissible joint, link6
pose, TCP pose or gripper targets are rejected with the admission
reason/details, zero command messages and no runtime send. Admitted targets
are delegated to the
runtime exactly once and its result is returned unchanged; the runtime
re-checks admission at the transport boundary while the executor
re-implements no validation logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from rocobrick.admission import strict_action_admission
from rocobrick.moveit_contract import (
    GripperTarget,
    JointTarget,
    Link6PoseTarget,
    TcpPoseTarget,
)
from rocobrick.moveit_execution import ExecutionResult, ExecutionStatus, MoveItExecutor


@runtime_checkable
class TcpMoveItExecutor(MoveItExecutor, Protocol):
    """R5.A executor extension for project-calibrated TCP pose goals."""

    def execute_tcp_pose(
        self, target: TcpPoseTarget, *, timeout_s: float
    ) -> ExecutionResult: ...


class RosMoveItExecutor:
    """MoveItExecutor protocol implementation delegating to a RosMoveItRuntime.

    The runtime instance owns the robot isolation (each robot has its own
    runtime); the executor is additionally keyed by ``robot_id`` so every
    rejection it produces is attributable to exactly one robot. The id is
    taken from the runtime's profile when the runtime carries one, so a
    piper_1 executor can never be constructed around a piper_0 runtime
    silently: an explicit mismatch is rejected at construction.
    """

    def __init__(self, runtime: Any, *, robot_id: str | None = None) -> None:
        runtime_robot_id = getattr(runtime, "robot_id", "") or None
        if (
            robot_id is not None
            and runtime_robot_id is not None
            and robot_id != runtime_robot_id
        ):
            raise ValueError(
                f"executor robot_id {robot_id!r} does not match the runtime "
                f"robot_id {runtime_robot_id!r}"
            )
        self._runtime = runtime
        self._robot_id = robot_id if robot_id is not None else runtime_robot_id

    @property
    def robot_id(self) -> str | None:
        """The robot this executor is bound to, or None for a canonical one."""
        return self._robot_id

    def _rejected(
        self, kind: str, reason: str, admission: Mapping[str, object]
    ) -> ExecutionResult:
        """Robot-attributed rejection record before any runtime send."""
        details: dict[str, object] = {
            "kind": kind,
            "reason": reason,
            "admission": dict(admission),
            "sent": False,
        }
        if self._robot_id is not None:
            details["robot_id"] = self._robot_id
        return ExecutionResult(
            status=ExecutionStatus.REJECTED,
            details=details,
            command_messages=0,
        )

    def execute_joint(self, target: JointTarget, *, timeout_s: float) -> ExecutionResult:
        """Admit a joint target; admitted targets delegate to the runtime."""
        decision = strict_action_admission(target)
        if not decision.accepted:
            return self._rejected("joint", decision.reason, decision.details)
        return self._runtime.send_joint(target, timeout_s)

    def execute_link6_pose(
        self, target: Link6PoseTarget, *, timeout_s: float
    ) -> ExecutionResult:
        """Admit a pose target; admitted targets delegate to the runtime."""
        decision = strict_action_admission(target)
        if not decision.accepted:
            return self._rejected("pose", decision.reason, decision.details)
        return self._runtime.send_link6_pose(target, timeout_s)

    def execute_tcp_pose(
        self, target: TcpPoseTarget, *, timeout_s: float
    ) -> ExecutionResult:
        """Admit a calibrated TCP pose target; admitted targets delegate once.

        The behavior mirrors :meth:`execute_link6_pose` exactly; the
        resolved/scoped calibration metadata is enforced by admission, and
        the runtime goal targets the ``arm_tcp`` group without any offset
        arithmetic (the calibration lives only in the model).
        """
        decision = strict_action_admission(target)
        if not decision.accepted:
            return self._rejected("tcp_pose", decision.reason, decision.details)
        return self._runtime.send_tcp_pose(target, timeout_s)

    def execute_gripper(
        self, target: GripperTarget, *, timeout_s: float
    ) -> ExecutionResult:
        """Admit a gripper target; admitted targets delegate once.

        The behavior mirrors :meth:`execute_tcp_pose` exactly; the measured
        opening verdict (including the legitimate ABORTED-on-contact case)
        is decided by the runtime, never here.
        """
        decision = strict_action_admission(target)
        if not decision.accepted:
            return self._rejected("gripper", decision.reason, decision.details)
        return self._runtime.send_gripper(target, timeout_s)
