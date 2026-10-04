"""Unified ROS-side successor pre-phase runtime facade.

This facade owns the semantic boundary between a predecessor terminal state and
the next skill contact boundary. The current implementation intentionally
supports plan-only admission only; it never reports physical pre-phase success
without controller execution and terminal-safe measured evidence.
"""

from __future__ import annotations

from collections.abc import Mapping

from rocobrick.coordinated_session import CoordinatedSessionRequest
from rocobrick.coordinated_session_entry import perform_coordinated_plan
from rocobrick.unified_dual_arm_runtime import (
    SuccessorPrePhaseEvidence,
    SuccessorPrePhaseRequest,
)


class UnifiedDualArmRuntimeFacade:
    """Single semantic runtime boundary around combined MoveIt planning."""

    def __init__(self, session: object) -> None:
        """Bind one session exposing a coordinated child-session dispatch."""
        self._session = session

    async def execute_pre_phase(
        self, request: SuccessorPrePhaseRequest
    ) -> SuccessorPrePhaseEvidence:
        """Plan a successor pre-phase and fail closed before physical execution."""
        if not request.successor_operations and not request.terminal_cleanup:
            return self._failed(request, "pre-phase has no successor operation")
        if request.terminal_cleanup:
            return self._failed(
                request,
                "terminal cleanup execution is not implemented",
            )
        try:
            targets = self._targets_from_successors(request)
        except ValueError as error:
            return self._failed(request, str(error))
        coordinated = CoordinatedSessionRequest(
            coordination_id=f"pre-phase-{request.predecessor_operation_id}",
            robots=request.participants,
            scene_digest=request.scene_digest,
            planning_group="dual_arm",
            plan_only=True,
            timeout_s=60.0,
            targets=targets,
        )
        dispatch = getattr(self._session, "execute_successor_pre_phase", None)
        if callable(dispatch):
            result = dispatch(coordinated)
        else:
            result = perform_coordinated_plan(self._session, coordinated)
        evidence = {
            "coordinated_plan": result.serialize(),
            "phase": "successor_pre_phase",
            "physical_execution": False,
        }
        return SuccessorPrePhaseEvidence(
            predecessor_operation_id=request.predecessor_operation_id,
            successor_operation_ids=request.successor_operation_ids,
            participants=request.participants,
            scene_digest=request.scene_digest,
            planned=result.passed,
            executed=False,
            terminal=False,
            safe=False,
            resource_transfer_confirmed=False,
            evidence=evidence,
            reason=(
                "combined plan admitted but controller execution is not implemented"
                if result.passed
                else result.error or "combined plan admission failed"
            ),
        )

    async def cancel(self, predecessor_operation_id: str) -> Mapping[str, object]:
        """Cancel a future pre-phase; no physical goal exists in plan-only mode."""
        return {
            "attempted": False,
            "terminal": True,
            "safe": True,
            "reason": "no controller execution was started",
            "predecessor_operation_id": predecessor_operation_id,
        }

    def _targets_from_successors(
        self, request: SuccessorPrePhaseRequest
    ) -> Mapping[str, Mapping[str, object]]:
        """Extract already geometry-derived TCP targets from successors."""
        targets: dict[str, Mapping[str, object]] = {}
        for operation in request.successor_operations:
            raw = operation.parameters.get("pre_phase_targets")
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"successor {operation.operation_id} lacks geometry-derived pre_phase_targets"
                )
            for robot_id, target in raw.items():
                if robot_id in targets:
                    raise ValueError(f"duplicate pre-phase target for {robot_id!r}")
                if not isinstance(target, Mapping):
                    raise ValueError(f"pre-phase target for {robot_id!r} is invalid")
                targets[str(robot_id)] = dict(target)
        if set(targets) != set(request.participants):
            raise ValueError("pre-phase targets do not cover all participants")
        return targets

    @staticmethod
    def _failed(
        request: SuccessorPrePhaseRequest,
        reason: str,
    ) -> SuccessorPrePhaseEvidence:
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


__all__ = ["UnifiedDualArmRuntimeFacade"]
