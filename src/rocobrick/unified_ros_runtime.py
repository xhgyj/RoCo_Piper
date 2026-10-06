"""ROS-side combined planning boundary with fail-closed physical execution."""

from __future__ import annotations

from collections.abc import Mapping

from rocobrick.coordinated_ros_runtime import CoordinatedMoveItPlanRuntime
from rocobrick.coordinated_session import CoordinatedSessionRequest
from rocobrick.unified_dual_arm_runtime import (
    SuccessorPrePhaseEvidence,
    SuccessorPrePhaseRequest,
    failed_pre_phase,
)
from rocobrick.unified_readiness import UnifiedDualArmReadiness


class UnifiedDualArmRuntime:
    """One combined planner plus robot-scoped controller participant views."""

    def __init__(
        self, robot_session: object, combined_runtime: CoordinatedMoveItPlanRuntime
    ) -> None:
        """Bind the combined runtime to its participant session."""
        self.robot_session = robot_session
        self.combined_runtime = combined_runtime

    @property
    def robot_ids(self) -> tuple[str, ...]:
        """Return the participant identities."""
        return tuple(getattr(self.robot_session, "robot_ids", ()))

    def register(self) -> None:
        """Register combined and participant transports."""
        register = getattr(self.robot_session, "register", None)
        if not callable(register):
            raise RuntimeError("robot participant session has no register boundary")
        register()
        self.combined_runtime.register()

    def ready(self) -> bool:
        """Return combined action-server readiness only."""
        return self.combined_runtime.ready()

    def readiness(self, request: CoordinatedSessionRequest) -> UnifiedDualArmReadiness:
        """Evaluate combined readiness.

        Returns:
            The participant session's explicit readiness contract.
        """
        self.register()
        readiness = getattr(self.robot_session, "unified_readiness", None)
        if not callable(readiness):
            raise RuntimeError("robot session has no unified readiness boundary")
        result = readiness(request)
        if not isinstance(result, UnifiedDualArmReadiness):
            raise RuntimeError("unified readiness returned an invalid contract")
        return result

    def plan(self, request: CoordinatedSessionRequest) -> Mapping[str, object]:
        """Plan without claiming arrival.

        Returns:
            Raw combined planning evidence.
        """
        self.register()
        return self.combined_runtime.plan(request, timeout_s=request.timeout_s)

    def derive_pre_phase_targets(self, operation, shared, contexts):
        """Derive calibrated contact-boundary targets without editing the operation.

        Returns:
            Complete participant TCP targets from the same shared snapshot.
        """
        from rocobrick.skills.pre_phase_targets import derive_pre_phase_targets

        return derive_pre_phase_targets(operation, shared, contexts)

    def observe_all(self):
        """Read one shared scene document.

        Returns:
            The complete participant and object observation.
        """
        return self.robot_session.observe_all()

    def plan_only(self, request):
        """Use the serialized scene/readiness/trajectory admission path.

        Returns:
            No-motion combined planning evidence.
        """
        return self.robot_session.unified_plan_only(request)

    async def execute_pre_phase(
        self, request: SuccessorPrePhaseRequest
    ) -> SuccessorPrePhaseEvidence:
        """Reject physical pre-phase until bound target and controller adapters exist.

        Returns:
            Bound failure evidence that retains physical leases.
        """
        return failed_pre_phase(
            request,
            "unified pre-phase target derivation and controller execution "
            "are not implemented",
        )

    async def cancel_pre_phase(self, transition_id: str) -> Mapping[str, object]:
        """Cancel combined planning without claiming controller arrival.

        Returns:
            Planning cancellation evidence bound to the transition identity.
        """
        return {**self.combined_runtime.cancel(), "transition_id": transition_id}


__all__ = ["UnifiedDualArmRuntime"]
