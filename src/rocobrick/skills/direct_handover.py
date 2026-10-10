"""Direct dual-arm HANDOVER execution through one persistent runtime.

This path intentionally skips the staged/prepared/contact-proof stack. It derives
a shared tabletop-relative region from the two bases and measured grasp,
validates separate paired goals, and executes through the persistent session.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

from rocobrick.coordinated_session import FreeSpaceMotionRequest
from rocobrick.multi_robot_scheduler import OperationExecution, OperationStatus
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.handover_planning import calibrated_target
from rocobrick.skills.handover_zone import generate_zone_targets, zone_from_request
from rocobrick.skills.observation import Pose
from rocobrick.skills.targets import _quat_rotate_vector
from rocobrick.skills.placement_geometry import compose, inverse

from rocobrick.adapters.moveit.handover_live import LiveSimulationHandover


def _joint_targets_from_trajectory(
    trajectory: Mapping[str, object],
    participants: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    names = trajectory["joint_names"]
    endpoint = trajectory["points"][-1]["positions"]
    joints = dict(zip(names, endpoint))
    return {
        robot: {
            "joint_positions": {
                f"{robot}_joint{i}": joints[f"{robot}_joint{i}"]
                for i in range(1, 7)
            }
        }
        for robot in participants
    }


def _giver_reposition_targets(
    request: Any,
    blend: float,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    zone = zone_from_request(request)
    observed = request.observation(request.donor.robot_id)
    local_center = tuple(
        (a + b) / 2
        for a, b in zip(
            request.object_geometry.minimum_m,
            request.object_geometry.maximum_m,
        )
    )
    measured_center = compose(
        observed.object_pose,
        Pose(local_center, (0, 0, 0, 1)),
    ).position_xyz
    central_center = tuple(
        zone.midpoint[i] + 0.08 * zone.forward[i]
        if i < 2
        else zone.tabletop_z_m + 0.16
        for i in range(3)
    )
    center = tuple(
        measured_center[i] * (1.0 - blend) + central_center[i] * blend
        for i in range(3)
    )
    object_orientation = observed.object_pose.orientation_xyzw
    object_goal = Pose(
        tuple(
            center[i]
            - _quat_rotate_vector(object_orientation, local_center)[i]
            for i in range(3)
        ),
        object_orientation,
    )
    held = compose(inverse(observed.object_pose), observed.tcp_pose)
    giver_goal = compose(object_goal, held)
    targets = {
        request.donor.robot_id: dict(
            calibrated_target(giver_goal, request.donor, request.profile)
        ),
        request.receiver.robot_id: {
            "maintain_pose": True,
            "joint_hold_tolerance_rad": request.profile.joint_hold_tolerance_rad,
        },
    }
    return targets, {
        "blend": blend,
        "zone_center": list(central_center),
        "object_center": list(center),
        "giver_goal": asdict(giver_goal),
        "receiver_target": "maintain_pose",
    }


class DirectHandoverBackend:
    """One-shot paired MoveIt planning and execution backend."""

    def __init__(self, owner: "DirectSimulationHandover"):
        self.owner = owner
        self._motion_ids: dict[str, str] = {}

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        motion_id = f"{operation.operation_id}:direct-handover"
        evidence: dict[str, object] = {"path": "direct_dual_arm_zone_execution"}
        motion_sent = False
        try:
            plan = getattr(self.owner.session, "execute_successor_pre_phase", None)
            execute = getattr(self.owner.session, "execute_coordinated_pre_phase", None)
            if not callable(plan) or not callable(execute):
                raise RuntimeError("persistent combined planning/execution is unavailable")
            attempts: list[dict[str, object]] = []
            evidence["target_attempts"] = attempts
            participants = None
            reposition_attempts: list[dict[str, object]] = []
            evidence["giver_reposition_attempts"] = reposition_attempts
            reposition_selected = None
            reposition_request = await asyncio.to_thread(
                self.owner.capture, operation
            )
            reposition_participants = (
                reposition_request.donor.robot_id,
                reposition_request.receiver.robot_id,
            )
            # Try the central target first, then shorter interpolated moves.
            # Every attempt keeps the receiver at its measured pose and carries
            # the brick with the giver's measured object-to-TCP transform.
            for blend in (0.75, 0.55, 0.35):
                reposition_targets, reposition_geometry = _giver_reposition_targets(
                    reposition_request, blend
                )
                reposition_motion = FreeSpaceMotionRequest(
                    motion_id=f"{operation.operation_id}:giver-reposition-plan-{blend:.2f}",
                    session_robots=reposition_request.session_robots,
                    participants=reposition_participants,
                    scene_snapshot=reposition_request.scene_snapshot,
                    scene_version=reposition_request.scene_version,
                    targets=reposition_targets,
                    start_state=reposition_request.start_state,
                    timeout_s=8.0,
                    physical_facts={
                        robot: {"role": role}
                        for robot, role in zip(
                            reposition_participants, ("giver", "receiver")
                        )
                    },
                )
                planned = await asyncio.to_thread(
                    plan, reposition_motion.to_coordinated_request(plan_only=True)
                )
                attempt = {
                    "blend": blend,
                    "geometry": reposition_geometry,
                    "targets": reposition_targets,
                    "passed": planned.passed,
                    "error": planned.error,
                }
                reposition_attempts.append(attempt)
                if planned.passed:
                    reposition_selected = (
                        planned,
                        reposition_motion,
                        reposition_targets,
                        reposition_geometry,
                    )
                    break
            if reposition_selected is None:
                errors = [attempt["error"] for attempt in reposition_attempts]
                raise ValueError(
                    f"giver reposition plan failed before handover: {errors}"
                )
            reposition_plan, reposition_motion, reposition_targets, reposition_geometry = (
                reposition_selected
            )
            reposition_motion_id = f"{operation.operation_id}:giver-reposition"
            reposition_joint_targets = _joint_targets_from_trajectory(
                reposition_plan.trajectory, reposition_participants
            )
            reposition_execute_motion = replace(
                reposition_motion,
                motion_id=reposition_motion_id,
                targets=reposition_joint_targets,
                timeout_s=30.0,
            )
            self._motion_ids[operation.operation_id] = reposition_motion_id
            reposition_result = await asyncio.to_thread(
                execute,
                reposition_execute_motion.to_coordinated_request(plan_only=False),
            )
            reposition_lifecycle = (
                dict(reposition_result.lifecycle)
                if isinstance(
                    getattr(reposition_result, "lifecycle", None), Mapping
                )
                else {}
            )
            motion_sent = motion_sent or bool(
                reposition_lifecycle.get("started") is True
                or reposition_lifecycle.get("physical_execution") is True
                or reposition_lifecycle.get("dispatch_attempted") is True
            )
            evidence["giver_reposition"] = {
                "geometry": reposition_geometry,
                "validated_plan": dict(reposition_plan.trajectory),
                "lifecycle": reposition_lifecycle,
                "trajectory": dict(
                    getattr(reposition_result, "trajectory", {}) or {}
                ),
            }
            if not (
                reposition_lifecycle.get("started") is True
                or reposition_lifecycle.get("physical_execution") is True
            ):
                raise ValueError(
                    f"giver reposition was not dispatched: "
                    f"{getattr(reposition_result, 'error', None)}"
                )
            # Re-observe after the carried reposition so the handover zone is
            # generated from the new giver pose and object location.
            await asyncio.sleep(0.05)
            selected = None
            for index in range(18):
                # Every planning attempt captures a new snapshot after the last
                # IK/plan. Do not reuse a scene that aged while a solver ran.
                request = await asyncio.to_thread(self.owner.capture, operation)
                zone, proposals = generate_zone_targets(request)
                evidence["target_region"] = zone.evidence()
                evidence["candidate_count"] = len(proposals)
                if index >= len(proposals):
                    break
                candidate = proposals[index]
                participants = (request.donor.robot_id, request.receiver.robot_id)
                targets = candidate.targets(request)
                if reposition_selected is not None:
                    # The giver has already been repositioned with the brick.
                    # Keep that measured joint state while MoveIt solves only
                    # the receiver's independent approach target; asking the
                    # giver to solve a second near-identical TCP pose is what
                    # previously reintroduced the piper_0 IK failure.
                    targets[request.donor.robot_id] = {
                        "maintain_pose": True,
                        "joint_hold_tolerance_rad": (
                            request.profile.joint_hold_tolerance_rad
                        ),
                    }
                motion = FreeSpaceMotionRequest(
                    motion_id=f"{operation.operation_id}:zone-plan-{index}",
                    session_robots=request.session_robots,
                    participants=participants,
                    scene_snapshot=request.scene_snapshot,
                    scene_version=request.scene_version,
                    targets=targets,
                    start_state=request.start_state,
                    timeout_s=8.0,
                    physical_facts={robot: {"role": role} for robot, role in
                                    zip(participants, ("giver", "receiver"))},
                )
                planned = await asyncio.to_thread(
                    plan, motion.to_coordinated_request(plan_only=True)
                )
                attempt = {"index": index, "proposal": asdict(candidate),
                           "targets": targets, "passed": planned.passed,
                           "error": planned.error}
                attempts.append(attempt)
                if planned.passed:
                    selected = planned
                    evidence["selected_index"] = index
                    evidence["independent_targets"] = targets
                    evidence["selected_proposal"] = asdict(candidate)
                    break
            if selected is None:
                errors = [attempt["error"] for attempt in attempts]
                raise ValueError(f"no paired zone plan succeeded: {errors}")
            # IK and a collision-checked combined path have succeeded. Reuse
            # that joint endpoint with a freshly measured joint start instead
            # of solving both pose goals a second time with different seeds.
            joint_targets = _joint_targets_from_trajectory(
                selected.trajectory, participants
            )
            evidence["validated_plan"] = dict(selected.trajectory)
            motion = replace(motion, motion_id=motion_id, targets=joint_targets,
                             scene_snapshot=request.scene_snapshot,
                             scene_version=request.scene_version,
                             start_state=request.start_state, timeout_s=30.0)
            self._motion_ids[operation.operation_id] = motion_id
            result = await asyncio.to_thread(
                execute, motion.to_coordinated_request(plan_only=False)
            )
            lifecycle = (
                dict(result.lifecycle)
                if isinstance(getattr(result, "lifecycle", None), Mapping)
                else {}
            )
            # Direct mode is exploratory: dispatch is the completion boundary.
            # Physical arrival/release safety remain recorded in lifecycle evidence.
            passed = bool(
                lifecycle.get("prepared") is True
                and lifecycle.get("started") is True
                and lifecycle.get("physical_execution") is True
            )
            motion_sent = bool(
                lifecycle.get("started") is True
                or lifecycle.get("physical_execution") is True
                or lifecycle.get("dispatch_attempted") is True
            )
            return OperationExecution(
                operation_id=operation.operation_id,
                robot_id=operation.executor_robot_id,
                status=OperationStatus.CONFIRMED if passed else OperationStatus.FAILED,
                passed=passed,
                terminal=passed,
                motion_sent=motion_sent,
                evidence={
                    **evidence,
                    "participants": list(participants),
                    "trajectory": dict(getattr(result, "trajectory", {}) or {}),
                    "lifecycle": lifecycle,
                    "receiver_holding_follow": "deferred_to_observation",
                    "donor_release_feedback": "deferred_to_observation",
                    "owner_transfer_committed": "deferred_to_observation",
                },
                reason=(
                    None
                    if passed
                    else f"DIRECT_HANDOVER_FAILED: {getattr(result, 'error', None)}"
                ),
            )
        except Exception as error:  # noqa: BLE001 - preserve runtime evidence
            return OperationExecution(
                operation_id=operation.operation_id,
                robot_id=operation.executor_robot_id,
                status=OperationStatus.FAILED,
                passed=False,
                terminal=False,
                motion_sent=motion_sent,
                evidence=evidence,
                reason=f"DIRECT_HANDOVER_FAILED: {type(error).__name__}: {error}",
            )
        finally:
            self._motion_ids.pop(operation.operation_id, None)

    async def cancel(self, operation_id: str) -> Mapping[str, object]:
        motion_id = self._motion_ids.get(operation_id, operation_id)
        cancel = getattr(self.owner.session, "cancel_free_space", None)
        if not callable(cancel):
            return {
                "operation_id": operation_id,
                "terminal": False,
                "safe_to_release": False,
                "reason": "persistent coordinated cancellation is unavailable",
            }
        raw = await asyncio.to_thread(cancel, motion_id)
        return dict(raw) if isinstance(raw, Mapping) else {
            "operation_id": operation_id,
            "terminal": False,
            "safe_to_release": False,
            "reason": "invalid cancellation evidence",
        }


class DirectSimulationHandover(LiveSimulationHandover):
    """Reuse post-lift capture, replacing staged execution with one direct call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.direct_backend = DirectHandoverBackend(self)
        self.backend = self.direct_backend
