"""Operation-scoped PICK orchestration with strict physical evidence gates."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.isaac_backend import (
    RuntimeOperation,
    attached_planning_scene_operation,
    gripper_operation,
    planning_scene_operation,
    tcp_pose_operation,
)
from rocobrick.multi_robot_scheduler import (
    JSONValue,
    OperationExecution,
    OperationStatus,
)
from rocobrick.moveit_contract import GripperTarget
from rocobrick.planning_scene import PlanningSceneUpdate, box_from_aabb, scene_digest
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.observation import (
    PickObservation,
    parse_pick_observation,
    verify_pick_following,
)
from rocobrick.skills.targets import pick_targets, world_position_to_base
from rocobrick.transit_adapter import TransitContext


class PickObservationTransport(Protocol):
    """Read one robot's live scene without sending motion."""

    async def observe(self, robot_id: str) -> Mapping[str, object]:
        """Return one measured scene for the named robot."""


class PickMotionTransport(Protocol):
    """Dispatch one robot-scoped runtime operation."""

    def run_motion_operation(
        self, robot_id: str, operation: RuntimeOperation
    ) -> Mapping[str, object]:
        """Run one runtime operation and return its strict record."""

    def cancel_motion_operation(self, operation_id: str) -> Mapping[str, object]:
        """Cancel one currently active runtime request."""


@dataclass(frozen=True)
class PickAdmission:
    """Immutable operation-scoped target and measured scene sample."""

    operation_id: str
    robot_id: str
    physical_instance_id: str
    sample: PickObservation


@dataclass(frozen=True)
class _MotionEvidence:
    """Strict normalized result of one PICK phase dispatch."""

    passed: bool
    terminal: bool
    safe: bool
    motion_sent: bool
    evidence: Mapping[str, JSONValue]
    reason: str | None


class PickHandler:
    """Execute PICK phases and confirm holding only from live follow evidence."""

    def __init__(
        self,
        observation_transport: PickObservationTransport,
        ledger: OperationStateLedger,
        *,
        session_id: str,
        manifest_sha256: str,
        motion_transport: PickMotionTransport | None = None,
        context: TransitContext | None = None,
        poll_interval_s: float = 0.2,
    ) -> None:
        """Bind one session identity, ledger, motion transport and calibration."""
        self._observations = observation_transport
        self._ledger = ledger
        self._session_id = session_id
        self._manifest_sha256 = manifest_sha256
        self._motion = motion_transport
        self._context = context
        self._poll_interval_s = float(poll_interval_s)
        self._active_requests: dict[str, str] = {}

    async def preflight(self, operation: RobotOperation) -> PickAdmission:
        """Validate target identity, ownership, arrival and fresh scene."""
        if operation.skill != "PICK" or operation.target is None:
            raise ValueError("PICK target is required")
        target = operation.target.values
        if set(target) != {"physical_instance_id"}:
            raise ValueError("PICK target must name exactly one physical instance")
        instance_id = target["physical_instance_id"]
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("PICK physical instance id is invalid")
        robot_id = operation.executor_robot_id
        owner = self._ledger.owner_of_object(instance_id)
        if owner is not None:
            raise ValueError(f"PICK object is already protected by {owner}")
        if any(key.startswith("holding:") for key in self._ledger.snapshot(robot_id)):
            raise ValueError("PICK robot already has confirmed holding")
        document = await self._observations.observe(robot_id)
        sample = parse_pick_observation(
            document,
            robot_id=robot_id,
            physical_instance_id=instance_id,
            session_id=self._session_id,
            manifest_sha256=self._manifest_sha256,
        )
        if "robot_at_pick_region" in operation.preconditions:
            arrival = self._ledger.observe(robot_id, "robot_at_pick_region")
            if (
                arrival is None
                or arrival.evidence.get("physical_instance_id") != instance_id
            ):
                raise ValueError("PICK robot has no confirmed arrival for target")
        return PickAdmission(operation.operation_id, robot_id, instance_id, sample)

    @staticmethod
    def _base_aabb(
        sample: PickObservation, context: TransitContext
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Transform all world AABB corners and rebuild a base-frame AABB."""
        corners = [
            (x, y, z)
            for x in (sample.aabb_min[0], sample.aabb_max[0])
            for y in (sample.aabb_min[1], sample.aabb_max[1])
            for z in (sample.aabb_min[2], sample.aabb_max[2])
        ]
        points = [world_position_to_base(point, context.base) for point in corners]
        return (
            tuple(min(point[index] for point in points) for index in range(3)),
            tuple(max(point[index] for point in points) for index in range(3)),
        )

    def _planning_payload(
        self,
        admission: PickAdmission,
        context: TransitContext,
        sample: PickObservation | None = None,
    ) -> dict[str, object]:
        """Build live target and configured ground collision boxes."""
        sample = sample or admission.sample
        minimum = tuple(float(value) for value in sample.aabb_min)
        maximum = tuple(float(value) for value in sample.aabb_max)
        object_id = admission.physical_instance_id
        frame_id = f"{admission.robot_id}_world"
        boxes = [
            box_from_aabb(
                object_id=object_id,
                frame_id=frame_id,
                minimum_m=minimum,
                maximum_m=maximum,
            )
        ]
        if context.ground is not None:
            ground = context.ground
            half_x, half_y = (size / 2.0 for size in ground.size_xy_m)
            boxes.append(
                box_from_aabb(
                    object_id=f"{admission.robot_id}_ground",
                    frame_id=frame_id,
                    minimum_m=(
                        ground.center_xy_m[0] - half_x,
                        ground.center_xy_m[1] - half_y,
                        ground.top_z_m - ground.thickness_m,
                    ),
                    maximum_m=(
                        ground.center_xy_m[0] + half_x,
                        ground.center_xy_m[1] + half_y,
                        ground.top_z_m,
                    ),
                )
            )
        links = tuple(sorted(
            (
                (object_id, f"{admission.robot_id}_gripper_link1"),
                (object_id, f"{admission.robot_id}_gripper_link2"),
            )
        ))
        update = PlanningSceneUpdate(
            scene_id=f"{admission.operation_id}-pick-scene",
            boxes=tuple(boxes),
            expected_scene_digest="0" * 64,
            allowed_collision_pairs=links,
        )
        digest = scene_digest(update)
        update = PlanningSceneUpdate(
            scene_id=update.scene_id,
            boxes=update.boxes,
            expected_scene_digest=digest,
            allowed_collision_pairs=update.allowed_collision_pairs,
        )
        return asdict(update)

    async def _dispatch(
        self, operation: RobotOperation, phase: str, runtime_operation: RuntimeOperation
    ) -> _MotionEvidence:
        """Dispatch one uniquely named phase and normalize strict lifecycle evidence."""
        if self._motion is None:
            return _MotionEvidence(False, True, False, False, {}, "motion transport unavailable")
        self._active_requests[operation.operation_id] = runtime_operation.request_id
        try:
            record = await asyncio.to_thread(
                self._motion.run_motion_operation,
                operation.executor_robot_id,
                runtime_operation,
            )
        except Exception as error:
            return _MotionEvidence(False, False, False, True, {}, f"{phase}: {error}")
        finally:
            self._active_requests.pop(operation.operation_id, None)
        raw_result = record.get("result") if isinstance(record, Mapping) else None
        execution = raw_result.get("execution") if isinstance(raw_result, Mapping) else None
        passed = raw_result.get("passed") is True if isinstance(raw_result, Mapping) else False
        non_motion_phase = getattr(runtime_operation.kind, "value", str(runtime_operation.kind)) in {
            "APPLY_PLANNING_SCENE",
            "ATTACH_PLANNING_SCENE",
        }
        if non_motion_phase:
            # Planning-scene updates have no action goal and therefore no
            # safe_to_release field.  A returned result is a terminal, safe
            # boundary; success still requires the runtime's passed/readback
            # evidence below.
            terminal = isinstance(raw_result, Mapping)
            safe = terminal
        else:
            terminal = execution.get("terminal") is True if isinstance(execution, Mapping) else False
            safe = execution.get("safe_to_release") is True if isinstance(execution, Mapping) else False
        sent = record.get("motion_sent") is True if isinstance(record, Mapping) else True
        evidence: dict[str, JSONValue] = {
            "phase": phase,
            "request_id": runtime_operation.request_id,
            "runtime": dict(raw_result) if isinstance(raw_result, Mapping) else {},
        }
        return _MotionEvidence(
            passed and terminal and safe,
            terminal,
            safe,
            sent,
            evidence,
            None if passed and terminal and safe else f"{phase}: runtime evidence failed",
        )

    @staticmethod
    def _close_feedback(
        evidence: Mapping[str, JSONValue],
    ) -> Mapping[str, JSONValue]:
        """Validate measured gripper feedback without claiming contact.

        A successful gripper action is only a mechanical feedback gate.
        Contact/holding is confirmed later by the lifted brick following the
        TCP, because the current scene contract exposes no force sensor.
        """
        runtime = evidence.get("runtime")
        execution = runtime.get("execution") if isinstance(runtime, Mapping) else None
        details = (
            execution.get("details")
            if isinstance(execution, Mapping)
            else None
        )
        if not isinstance(execution, Mapping) or not isinstance(details, Mapping):
            raise ValueError("close gripper runtime details are unavailable")
        measured = details.get("measured_opening_m")
        window = details.get("expected_opening_m")
        post_state = details.get("post_result_state_messages")
        accepted = execution.get("goal_accepted") is True
        terminal = execution.get("terminal") is True
        safe = execution.get("safe_to_release") is True
        status = execution.get("action_status") == "SUCCEEDED"
        if (
            isinstance(measured, bool)
            or not isinstance(measured, (int, float))
            or not isinstance(window, (list, tuple))
            or len(window) != 2
            or not all(isinstance(value, (int, float)) for value in window)
            or not isinstance(post_state, int)
            or post_state < 1
            or not (float(window[0]) <= float(measured) <= float(window[1]))
            or not all((accepted, terminal, safe, status))
        ):
            raise ValueError("close gripper feedback did not prove the commanded opening")
        return {
            "goal_accepted": True,
            "action_status": "SUCCEEDED",
            "terminal": True,
            "safe_to_release": True,
            "measured_opening_m": float(measured),
            "expected_opening_m": [float(window[0]), float(window[1])],
            "post_result_state_messages": post_state,
            "contact_confirmed": False,
            "contact_confirmation_deferred": "lift_following",
        }

    async def cancel(self, operation_id: str) -> Mapping[str, object]:
        """Cancel the currently active phase request for one operation."""
        request_id = self._active_requests.get(operation_id)
        if request_id is None or self._motion is None:
            return {"attempted": False, "terminal": False, "safe": False}
        return await asyncio.to_thread(self._motion.cancel_motion_operation, request_id)

    @staticmethod
    def _failed(
        operation: RobotOperation,
        reason: str,
        evidence: Mapping[str, JSONValue],
        *,
        terminal: bool,
        motion_sent: bool,
    ) -> OperationExecution:
        return OperationExecution(
            operation_id=operation.operation_id,
            robot_id=operation.executor_robot_id,
            status=OperationStatus.FAILED,
            passed=False,
            terminal=terminal,
            motion_sent=motion_sent,
            evidence=dict(evidence),
            reason=reason,
        )

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        """Run PICK phases and commit holding only after follow verification."""
        try:
            admission = await self.preflight(operation)
        except (ValueError, RuntimeError) as error:
            return self._failed(
                operation,
                f"PICK_PREFLIGHT_REJECTED: {error}",
                {},
                terminal=True,
                motion_sent=False,
            )
        if self._context is None or self._motion is None:
            return self._failed(
                operation,
                "PICK_PLANNING_SCENE_READBACK_UNAVAILABLE",
                {
                    "target_identity_checked": {
                        "operation_id": admission.operation_id,
                        "robot_id": admission.robot_id,
                        "physical_instance_id": admission.physical_instance_id,
                        "session_id": admission.sample.session_id,
                        "manifest_sha256": admission.sample.manifest_sha256,
                        "sequence": admission.sample.sequence,
                        "sim_time_s": admission.sample.sim_time_s,
                    }
                },
                terminal=True,
                motion_sent=False,
            )
        motion_sent = False
        evidence: dict[str, JSONValue] = {}
        try:
            context = self._context
            targets = pick_targets(
                admission.sample.aabb_min,
                admission.sample.aabb_max,
                context.base,
                context.provenance,
                context.params,
            )
            evidence.update({
                "target_identity_checked": {
                    "operation_id": admission.operation_id,
                    "robot_id": admission.robot_id,
                    "physical_instance_id": admission.physical_instance_id,
                    "session_id": admission.sample.session_id,
                    "manifest_sha256": admission.sample.manifest_sha256,
                    "sequence": admission.sample.sequence,
                    "sim_time_s": admission.sample.sim_time_s,
                }
            })
            planning_sample = admission.sample
            planning = planning_scene_operation(
                self._planning_payload(admission, context, planning_sample),
                60.0,
                f"{operation.operation_id}:plan_grasp",
            )
            result = await self._dispatch(operation, "plan_grasp", planning)
            evidence.update(result.evidence)
            if not result.passed:
                return self._failed(operation, result.reason or "plan_grasp failed", evidence,
                                    terminal=result.terminal, motion_sent=result.motion_sent)
            phases = (
                ("open_gripper", gripper_operation(
                    GripperTarget(
                        opening_m=targets.reach_opening_m,
                        duration_s=context.params.gripper_duration_s,
                        goal_tolerance=context.params.gripper_goal_tolerance,
                        expected_opening_m=targets.reach_expected_window_m,
                    ), 60.0, f"{operation.operation_id}:open_gripper")),
                ("reach_pregrasp", tcp_pose_operation(targets.pregrasp, 60.0,
                    f"{operation.operation_id}:reach_pregrasp")),
                ("approach", tcp_pose_operation(targets.approach, 60.0,
                    f"{operation.operation_id}:approach")),
                ("close_gripper", gripper_operation(
                    GripperTarget(
                        opening_m=targets.close_opening_m,
                        duration_s=context.params.gripper_duration_s,
                        goal_tolerance=context.params.gripper_goal_tolerance,
                        expected_opening_m=targets.close_expected_window_m,
                    ), 60.0, f"{operation.operation_id}:close_gripper")),
            )
            for phase, runtime_operation in phases:
                result = await self._dispatch(operation, phase, runtime_operation)
                evidence.update(result.evidence)
                motion_sent = motion_sent or result.motion_sent
                if not result.passed:
                    return self._failed(operation, result.reason or f"{phase} failed", evidence,
                                        terminal=result.terminal, motion_sent=motion_sent)
                if phase == "close_gripper":
                    try:
                        evidence["gripper_close_feedback_confirmed"] = (
                            self._close_feedback(result.evidence)
                        )
                    except ValueError as error:
                        return self._failed(
                            operation,
                            f"GRIPPER_FEEDBACK_UNCONFIRMED: {error}",
                            evidence,
                            terminal=result.terminal,
                            motion_sent=motion_sent,
                        )
                if phase == "reach_pregrasp":
                    self._ledger.confirm(
                        admission.robot_id,
                        "robot_at_pick_region",
                        {
                            "physical_instance_id": admission.physical_instance_id,
                            "phase": "reach_pregrasp",
                            "evidence": dict(result.evidence),
                        },
                    )
            grasp_sample = parse_pick_observation(
                await self._observations.observe(admission.robot_id),
                robot_id=admission.robot_id,
                physical_instance_id=admission.physical_instance_id,
                session_id=self._session_id,
                manifest_sha256=self._manifest_sha256,
            )
            evidence["grasp_observation"] = asdict(grasp_sample)
            attach_payload = self._planning_payload(admission, context, grasp_sample)
            attach = attached_planning_scene_operation(
                {
                    "scene_id": f"{operation.operation_id}-attach",
                    "object_id": admission.physical_instance_id,
                    "link_name": f"{admission.robot_id}_rocobrick_tcp",
                    "frame_id": f"{admission.robot_id}_world",
                    "minimum_m": list(grasp_sample.aabb_min),
                    "maximum_m": list(grasp_sample.aabb_max),
                    "geometry_digest": attach_payload["boxes"][0]["geometry_digest"],
                    "touch_links": [
                        f"{admission.robot_id}_gripper_link1",
                        f"{admission.robot_id}_gripper_link2",
                    ],
                }, 60.0, f"{operation.operation_id}:attach",
            )
            result = await self._dispatch(operation, "attach", attach)
            evidence.update(result.evidence)
            if not result.passed:
                return self._failed(operation, result.reason or "attach failed", evidence,
                                    terminal=result.terminal, motion_sent=motion_sent)
            lift = tcp_pose_operation(targets.lift, 60.0, f"{operation.operation_id}:lift")
            result = await self._dispatch(operation, "lift", lift)
            evidence.update(result.evidence)
            motion_sent = motion_sent or result.motion_sent
            if not result.passed:
                return self._failed(operation, result.reason or "lift failed", evidence,
                                    terminal=result.terminal, motion_sent=motion_sent)
            lift_start = parse_pick_observation(
                await self._observations.observe(admission.robot_id),
                robot_id=admission.robot_id,
                physical_instance_id=admission.physical_instance_id,
                session_id=self._session_id,
                manifest_sha256=self._manifest_sha256,
            )
            evidence["lift_start_observation"] = asdict(lift_start)
            samples = [lift_start]
            for _ in range(2):
                await asyncio.sleep(self._poll_interval_s)
                sample = parse_pick_observation(
                    await self._observations.observe(admission.robot_id),
                    robot_id=admission.robot_id,
                    physical_instance_id=admission.physical_instance_id,
                    session_id=self._session_id,
                    manifest_sha256=self._manifest_sha256,
                )
                samples.append(sample)
            holding = verify_pick_following(
                samples,
                position_tolerance_m=context.params.hold_follow_tolerance_m,
                orientation_tolerance_rad=context.params.orientation_tolerance_rad,
                minimum_lift_m=context.params.lift_height_m * 0.5,
            )
            evidence["holding_confirmed"] = holding
            evidence["gripper_feedback_confirmed"] = True
            evidence["lift_confirmed"] = True
            self._ledger.confirm_holding(
                admission.robot_id, admission.physical_instance_id, evidence
            )
            return OperationExecution(
                operation_id=operation.operation_id,
                robot_id=operation.executor_robot_id,
                status=OperationStatus.CONFIRMED,
                passed=True,
                terminal=True,
                motion_sent=motion_sent,
                evidence=evidence,
                reason="PICK_CONFIRMED",
            )
        except (ValueError, RuntimeError) as error:
            return self._failed(
                operation,
                f"PICK_FAILED: {error}",
                evidence,
                terminal=not motion_sent,
                motion_sent=motion_sent,
            )
