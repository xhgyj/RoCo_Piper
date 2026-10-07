"""Execute a bound, already planned path without MoveGroup or scene writes.

Collision certification remains the caller's planning responsibility. This
boundary adds live admission, actual timestamped controller goals, measured
tracking and terminal evidence; it never infers custody from arm arrival.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from rocobrick.adapters.moveit.coordinated_arm import CoordinatedArmParticipant
from rocobrick.coordinated_execution import CoordinatedExecutionCoordinator
from rocobrick.coordinated_session import CoordinatedSessionResult
from rocobrick.execution.prepared_motion import PreparedMotionRequest


class PreparedExecutionSession(Protocol):
    """Live read-only scene and robot-scoped action access, with no planner."""

    robot_ids: tuple[str, ...]

    def observe_all(self) -> Mapping[str, object]:
        """Read one fresh complete shared physics snapshot."""
        ...

    def unified_planning_scene_version(self) -> int:
        """Read actual world/attachment version, without applying a scene."""
        ...

    def unified_start_state(self) -> Mapping[str, float]:
        """Read all measured joints, including both grippers."""
        ...

    def latest_shared_sim_time_s(self) -> float:
        """Read the shared simulation clock, never a zero fallback."""
        ...

    def runtime_for(self, robot_id: str) -> object:
        """Return this exact robot's owned controller runtime."""
        ...


def perform_prepared_motion(
    session: PreparedExecutionSession,
    request: PreparedMotionRequest,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> CoordinatedSessionResult:
    """Admit and execute the exact supplied path with no hidden replanning.

    Returns:
        Strict physical lifecycle and binding evidence, not custody transfer.
    """
    attempted = False
    try:
        if session.robot_ids != request.robots:
            raise ValueError("prepared motion robots differ from the live session")

        def admit_live() -> None:
            if cancel_requested is not None and cancel_requested():
                raise ValueError("prepared motion canceled before dispatch")
            version = session.unified_planning_scene_version()
            joints = session.unified_start_state()
            snapshot = session.observe_all()
            request.validate_live(
                snapshot, version, joints, session.latest_shared_sim_time_s()
            )

        admit_live()
        participants = {
            robot: CoordinatedArmParticipant(
                session.runtime_for(robot), policy=request.policy
            )
            for robot in request.participants
        }
        coordinator = CoordinatedExecutionCoordinator(participants)
        # Barrier is relative to the current clock, not the old planning tick.
        # Each arm still refuses an expired barrier at dispatch.
        barrier = session.latest_shared_sim_time_s() + 1.0
        attempted = True
        lifecycle = coordinator.execute(
            request.operation,
            start_at_sim_time_s=barrier,
            require_synchronization_feedback=True,
            before_start=admit_live,
        )
        evidence = lifecycle.serialize()
        evidence.update(
            physical_execution=any(
                item.get("motion_sent") is True or item.get("started") is True
                for item in lifecycle.participants.values()
            ),
            start_barrier_sim_time_s=barrier,
            prepared_binding_sha256=request.binding_sha256,
            replanned=False,
        )
        passed = lifecycle.started and lifecycle.terminal and lifecycle.safe_to_release
        return CoordinatedSessionResult(
            request.coordination_id,
            request.robots,
            {"live_start_and_scene_confirmed": True},
            {"prepared_operation": request.serialize()["operation"]},
            evidence,
            passed,
            lifecycle.error,
            request.participants,
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
        return CoordinatedSessionResult(
            request.coordination_id,
            request.robots,
            {},
            {},
            {
                "physical_execution": attempted,
                "prepared": False,
                "started": False,
                "terminal": not attempted,
                "safe_to_release": not attempted,
                "replanned": False,
            },
            False,
            str(error),
            request.participants,
        )
