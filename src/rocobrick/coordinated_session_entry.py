"""Plan-only combined MoveIt request execution entry point."""

from __future__ import annotations

from collections.abc import Mapping

from rocobrick.coordinated_session import (
    CoordinatedSessionRequest,
    CoordinatedSessionResult,
)
from rocobrick.dual_arm_coordinator import DualArmPlanOnlyCoordinator


def perform_coordinated_plan(
    session: object,
    request: CoordinatedSessionRequest,
    participants_executor: object = None,
) -> CoordinatedSessionResult:
    """Route one combined request to the coordinated runtime.

    The session must expose ``coordinated_readiness`` and
    ``coordinated_plan_runtime``. Missing either capability is an explicit
    failure; two independent robot runtimes are never used as a fallback.

    With ``participants_executor`` the admitted split trajectories are also
    dispatched through that executor (prepare-all/start-all physical
    lifecycle); without it the result remains strict plan-only evidence.

    Returns:
        Strict coordinated result evidence.
    """
    from dataclasses import replace

    execution_attempted = False
    if request.operation:
        try:
            from rocobrick.skills.pre_phase_targets import derive_wire_targets

            request = replace(
                request,
                targets=derive_wire_targets(
                    request.operation, request.scene_snapshot, request.target_contexts
                ),
            )
        except Exception as error:
            return _failed(request, f"pre-phase target derivation failed: {error}")
    if any(
        isinstance(target.get("tcp_pose"), Mapping)
        for target in request.targets.values()
    ):
        # The combined group plans in joint space, but IK is intentionally
        # delayed until after readiness and scene admission. IK retries can
        # consume enough simulation time to stale the bound snapshot; the
        # admission gates must run against the live request first.
        pass
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
        if not request.plan_only and not callable(participants_executor):
            raise ValueError(
                "physical coordinated execution requires participant executor"
            )
        if not request.scene_snapshot:
            raise ValueError("combined plan requires a bound shared scene snapshot")
        if not request.physical_facts or set(request.physical_facts) != set(
            request.participants
        ):
            raise ValueError(
                "combined planning requires known per-participant physical facts"
            )
        readiness_document = readiness_fn(request)
        # Staleness is judged against the session's CURRENT scene version,
        # BEFORE this request's own world apply: sync legitimately re-applies
        # the admitted snapshot and may bump the version afterwards, so
        # comparing the post-sync version against the caller's pre-sync
        # capture would reject every request after the first.
        version_fn = getattr(session, "unified_planning_scene_version", None)
        if (
            request.scene_version
            and callable(version_fn)
            and version_fn() != request.scene_version
        ):
            raise ValueError("requested planning scene version is stale")
        sync = getattr(session, "sync_unified_scene", None)
        if not callable(sync):
            raise ValueError("unified planning scene synchronization is unavailable")
        scene_proof = sync(request)
        if (
            not isinstance(scene_proof, dict)
            or scene_proof.get("readback_confirmed") is not True
            or scene_proof.get("snapshot_digest") != request.scene_digest
        ):
            raise ValueError("unified scene synchronization returned unbound readback")
        if (
            request.scene_version
            and scene_proof.get("scene_version") < request.scene_version
        ):
            raise ValueError("requested planning scene version is stale")
        if any(
            isinstance(target.get("tcp_pose"), Mapping)
            for target in request.targets.values()
        ):
            compose = getattr(session, "unified_joint_targets", None)
            start_state_fn = getattr(session, "unified_start_state", None)
            if not callable(compose) or not callable(start_state_fn):
                raise ValueError(
                    "per-arm IK/start-state composition is unavailable; "
                    "no motion was dispatched"
                )
            try:
                joint_targets = compose(request)
                start_state = start_state_fn()
            except Exception as error:
                raise ValueError(
                    f"per-arm IK/start-state composition failed: "
                    f"{type(error).__name__}: {error}"
                ) from error
            request = replace(
                request,
                start_state=start_state,
                targets={
                    robot: (
                        {
                            **request.targets[robot],
                            "joint_positions": joint_targets[robot],
                        }
                        if robot in joint_targets
                        else request.targets[robot]
                    )
                    for robot in request.targets
                },
            )
        binding = {
            robot_id: tuple(session.runtime_for(robot_id).arm_joint_names)
            for robot_id in request.participants
        }
        coordinator = DualArmPlanOnlyCoordinator(binding)
        readiness = coordinator.admit_readiness(readiness_document, request)
        runtime = runtime_fn()
        raw = runtime.plan(request, timeout_s=request.timeout_s)
        validate_scene = getattr(session, "validate_unified_scene", None)
        if not callable(validate_scene) or validate_scene(scene_proof) is not True:
            raise ValueError("unified scene changed while planning")
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
            "planning_scene": dict(scene_proof),
        }
        if callable(participants_executor):
            # Crossing this boundary may prepare or start physical actions.
            # Exceptions after it cannot prove that the participants stopped.
            execution_attempted = True
            executed = participants_executor(operation)
            if not isinstance(executed, Mapping):
                raise ValueError("coordinated executor returned invalid evidence")
            lifecycle = dict(executed)
            lifecycle["physical_execution"] = True
            passed = bool(
                lifecycle.get("prepared") is True
                and lifecycle.get("started") is True
                and lifecycle.get("terminal") is True
                and lifecycle.get("safe_to_release") is True
            )
            return CoordinatedSessionResult(
                coordination_id=request.coordination_id,
                robots=request.robots,
                participants=request.participants,
                readiness=readiness.__dict__,
                trajectory=trajectory,
                lifecycle=lifecycle,
                passed=passed,
                error=None if passed else str(lifecycle.get("error")),
            )
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
        return _failed(
            request,
            f"{type(exc).__name__}: {exc}",
            execution_attempted=execution_attempted,
        )


def _failed(
    request: CoordinatedSessionRequest,
    error: str,
    *,
    execution_attempted: bool = False,
) -> CoordinatedSessionResult:
    """Attest no dispatch only before entering the physical executor.

    Returns:
        Failed planning evidence; uncertain execution still fails closed.
    """
    return CoordinatedSessionResult(
        coordination_id=request.coordination_id,
        robots=request.robots,
        participants=request.participants,
        readiness={},
        trajectory={},
        lifecycle={
            "coordination_id": request.coordination_id,
            "prepared": False,
            "started": False,
            "terminal": not execution_attempted,
            "safe_to_release": not execution_attempted and not request.plan_only,
            "physical_execution": execution_attempted,
            "dispatch_attempted": execution_attempted,
            "plan_only": request.plan_only,
        },
        passed=False,
        error=error,
    )


__all__ = ["perform_coordinated_plan"]
