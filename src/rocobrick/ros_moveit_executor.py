"""MoveItExecutor implementation over the ROS MoveIt runtime adapter (R2).

Thin transport boundary implementing
:class:`rocobrick.moveit_execution.MoveItExecutor` (and its R5.A calibrated
TCP extension :class:`TcpMoveItExecutor`) on top of
:class:`rocobrick.ros_moveit_runtime.RosMoveItRuntime`. Strict R1 admission
(``strict_action_admission``) gates the executor: inadmissible joint, link6
pose or TCP pose targets are rejected with the admission reason/details, zero
command messages and no runtime send. Admitted targets are delegated to the
runtime exactly once and its result is returned unchanged; the runtime
re-checks admission at the transport boundary while the executor
re-implements no validation logic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rocobrick.admission import strict_action_admission
from rocobrick.moveit_contract import JointTarget, Link6PoseTarget, TcpPoseTarget
from rocobrick.moveit_execution import ExecutionResult, ExecutionStatus, MoveItExecutor


@runtime_checkable
class TcpMoveItExecutor(MoveItExecutor, Protocol):
    """R5.A executor extension for project-calibrated TCP pose goals."""

    def execute_tcp_pose(
        self, target: TcpPoseTarget, *, timeout_s: float
    ) -> ExecutionResult: ...


class RosMoveItExecutor:
    """MoveItExecutor protocol implementation delegating to a RosMoveItRuntime."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def execute_joint(self, target: JointTarget, *, timeout_s: float) -> ExecutionResult:
        """Admit a joint target; admitted targets delegate to the runtime."""
        decision = strict_action_admission(target)
        if not decision.accepted:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                details={
                    "kind": "joint",
                    "reason": decision.reason,
                    "admission": dict(decision.details),
                    "sent": False,
                },
                command_messages=0,
            )
        return self._runtime.send_joint(target, timeout_s)

    def execute_link6_pose(
        self, target: Link6PoseTarget, *, timeout_s: float
    ) -> ExecutionResult:
        """Admit a pose target; admitted targets delegate to the runtime."""
        decision = strict_action_admission(target)
        if not decision.accepted:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                details={
                    "kind": "pose",
                    "reason": decision.reason,
                    "admission": dict(decision.details),
                    "sent": False,
                },
                command_messages=0,
            )
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
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                details={
                    "kind": "tcp_pose",
                    "reason": decision.reason,
                    "admission": dict(decision.details),
                    "sent": False,
                },
                command_messages=0,
            )
        return self._runtime.send_tcp_pose(target, timeout_s)
