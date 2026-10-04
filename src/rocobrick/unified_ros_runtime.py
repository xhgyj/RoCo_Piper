"""Unified ROS-side dual-arm runtime facade.

The facade owns the combined MoveIt planning endpoint and the shared scene
admission boundary. Robot-scoped runtimes remain participant views for joint,
controller and observation evidence; they are never used as a coordinated
planning fallback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rocobrick.coordinated_ros_runtime import CoordinatedMoveItPlanRuntime
from rocobrick.coordinated_session import CoordinatedSessionRequest
from rocobrick.unified_dual_arm_runtime import (
    SuccessorPrePhaseEvidence,
    SuccessorPrePhaseRequest,
)
from rocobrick.unified_readiness import UnifiedDualArmReadiness


class UnifiedDualArmRuntime:
    """One combined planner plus robot-scoped controller participant views."""

    def __init__(
        self,
        robot_session: object,
        combined_runtime: CoordinatedMoveItPlanRuntime,
    ) -> None:
        """Bind the combined runtime to one namespaced robot session."""
        self.robot_session = robot_session
        self.combined_runtime = combined_runtime

    @property
    def robot_ids(self) -> tuple[str, ...]:
        """Return the exact participant robot ids."""
        return tuple(getattr(self.robot_session, "robot_ids", ()))

    def register(self) -> None:
        """Register combined and participant transports exactly once."""
        register = getattr(self.robot_session, "register", None)
        if not callable(register):
            raise RuntimeError("robot participant session has no register boundary")
        register()
        self.combined_runtime.register()

    def ready(self) -> bool:
        """Return combined MoveGroup action readiness only."""
        return self.combined_runtime.ready()

    def readiness(self, request: CoordinatedSessionRequest) -> UnifiedDualArmReadiness:
        """Evaluate one shared combined readiness snapshot."""
        self.register()
        readiness = getattr(self.robot_session, "unified_readiness", None)
        if not callable(readiness):
            raise RuntimeError("robot session has no unified readiness boundary")
        result = readiness(request)
        if not isinstance(result, UnifiedDualArmReadiness):
            raise RuntimeError("unified readiness returned an invalid contract")
        return result

    def plan(self, request: CoordinatedSessionRequest) -> Mapping[str, object]:
        """Plan one combined pre-phase without claiming physical arrival."""
        self.register()
        return self.combined_runtime.plan(request, timeout_s=request.timeout_s)

    async def execute_pre_phase(
        self, request: SuccessorPrePhaseRequest
    ) -> SuccessorPrePhaseEvidence:
        """Plan-only successor pre-phase; execution remains fail-closed."""
        if request.terminal_cleanup:
            return _failed_pre_phase(request, "terminal cleanup is not implemented")
        targets = _targets_from_successors(request)
        coordinated = CoordinatedSessionRequest(
            coordination_id=f"pre-phase-{request.predecessor_operation_id}",
            robots=request.participants,
            scene_digest=request.scene_digest,
            planning_group="dual_arm",
            plan_only=True,
            timeout_s=60.0,
            targets=targets,
        )
        readiness = self.readiness(coordinated)
        if not readiness.passed:
            return _failed_pre_phase(request, "unified readiness failed")
        raw = self.plan(coordinated)
        return SuccessorPrePhaseEvidence(
            predecessor_operation_id=request.predecessor_operation_id,
            successor_operation_ids=request.successor_operation_ids,
            participants=request.participants,
            scene_digest=request.scene_digest,
            planned=True,
            executed=False,
            terminal=False,
            safe=False,
            resource_transfer_confirmed=False,
            evidence={
                "unified_readiness": readiness.serialize(),
                "combined_plan": dict(raw),
                "physical_execution": False,
            },
            terminal_cleanup=False,
            reason="combined plan admitted; controller execution is not implemented",
        )

    async def cancel(self, predecessor_operation_id: str) -> Mapping[str, object]:
        """Cancel a pre-phase plan request when a future goal is active."""
        return self.combined_runtime.cancel()


def _targets_from_successors(
    request: SuccessorPrePhaseRequest,
) -> Mapping[str, Mapping[str, object]]:
    """Require geometry-derived participant targets from successor operations."""
    targets: dict[str, Mapping[str, object]] = {}
    for operation in request.successor_operations:
        raw = operation.parameters.get("pre_phase_targets")
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"successor {operation.operation_id} lacks pre_phase_targets"
            )
        for robot_id, target in raw.items():
            if str(robot_id) in targets:
                raise ValueError(f"duplicate pre-phase target for {robot_id!r}")
            if not isinstance(target, Mapping):
                raise ValueError(f"pre-phase target for {robot_id!r} is invalid")
            targets[str(robot_id)] = dict(target)
    if set(targets) != set(request.participants):
        raise ValueError("pre-phase targets do not cover all participants")
    return targets


def _failed_pre_phase(
    request: SuccessorPrePhaseRequest,
    reason: str,
) -> SuccessorPrePhaseEvidence:
    """Return one explicit fail-closed pre-phase result."""
    return SuccessorPrePhaseEvidence(
        predecessor_operation_id=request.predecessor_operation_id,
        successor_operation_ids=request.successor_operation_ids,
        participants=request.participants,
        scene_digest=request.scene_digest,
        planned=False,
        executed=False,
        terminal=False,
        safe=False,
        resource_transfer_confirmed=False,
        evidence={},
        terminal_cleanup=request.terminal_cleanup,
        reason=reason,
    )


__all__ = ["UnifiedDualArmRuntime"]
