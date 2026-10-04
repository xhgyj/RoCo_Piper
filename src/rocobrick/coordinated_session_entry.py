"""Plan-only combined MoveIt request execution entry point."""

from __future__ import annotations

from rocobrick.coordinated_session import (
    CoordinatedSessionRequest,
    CoordinatedSessionResult,
)
from rocobrick.dual_arm_coordinator import DualArmPlanOnlyCoordinator


def perform_coordinated_plan(
    session: object,
    request: CoordinatedSessionRequest,
) -> CoordinatedSessionResult:
    """Route one combined request to the dedicated plan-only runtime.

    The session must expose ``coordinated_readiness`` and
    ``coordinated_plan_runtime``. Missing either capability is an explicit
    failure; two independent robot runtimes are never used as a fallback.

    Returns:
        Strict coordinated result evidence.
    """
    if tuple(getattr(session, "robot_ids", ())) != request.robots:
        return _failed(request, "coordinated session robot set does not match request")
    readiness_fn = getattr(session, "coordinated_readiness", None)
    runtime_fn = getattr(session, "coordinated_plan_runtime", None)
    if not callable(readiness_fn) or not callable(runtime_fn):
        return _failed(
            request,
            "coordinated MoveIt runtime is unavailable; no motion was dispatched",
        )
    try:
        readiness_document = readiness_fn(request)
        binding = {
            robot_id: tuple(session.runtime_for(robot_id).arm_joint_names)
            for robot_id in request.robots
        }
        coordinator = DualArmPlanOnlyCoordinator(binding)
        readiness = coordinator.admit_readiness(readiness_document, request)
        runtime = runtime_fn()
        raw = runtime.plan(request, timeout_s=request.timeout_s)
        operation = coordinator.admit_plan(
            request,
            joint_names=raw["joint_names"],
            points=raw["points"],
        )
        trajectory = {
            "joint_names": list(raw["joint_names"]),
            "points": [dict(point) for point in raw["points"]],
            "coordination_id": operation.coordination_id,
            "scene_digest": operation.scene_digest,
        }
        lifecycle = {
            "coordination_id": operation.coordination_id,
            "prepared": False,
            "started": False,
            "terminal": True,
            "safe_to_release": False,
            "physical_execution": False,
            "plan_only": True,
        }
        return CoordinatedSessionResult(
            coordination_id=request.coordination_id,
            robots=request.robots,
            readiness=readiness.__dict__,
            trajectory=trajectory,
            lifecycle=lifecycle,
            passed=True,
        )
    except Exception as exc:  # noqa: BLE001 - child must publish strict failure
        return _failed(request, f"{type(exc).__name__}: {exc}")


def _failed(request: CoordinatedSessionRequest, error: str) -> CoordinatedSessionResult:
    return CoordinatedSessionResult(
        coordination_id=request.coordination_id,
        robots=request.robots,
        readiness={},
        trajectory={},
        lifecycle={
            "coordination_id": request.coordination_id,
            "prepared": False,
            "started": False,
            "terminal": False,
            "safe_to_release": False,
            "physical_execution": False,
            "plan_only": request.plan_only,
        },
        passed=False,
        error=error,
    )


__all__ = ["perform_coordinated_plan"]
