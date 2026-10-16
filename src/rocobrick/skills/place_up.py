"""Operation-scoped PLACE_UP for a lower brick inserted into a supported upper.

The placement robot owns the moving LOWER brick. The supporting robot owns a
non-grasping ``supporting:<upper>`` fact established by SUPPORT_TOP. MoveIt is
used for collision-checked approach and supervised LIN contact. Optional
measured-object feedback bounds each insertion step; exact connection or a
physical safety stop ends motion. Release and retreat are proven separately.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

from rocobrick.adapters.moveit.transit import TransitContext
from rocobrick.coordinated_session import (
    FreeSpaceMotionEvidence,
    FreeSpaceMotionRequest,
)
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.isaac_backend import (
    RuntimeOperation,
    detached_planning_scene_operation,
    gripper_operation,
    guarded_tcp_operation,
    planning_scene_operation,
)
from rocobrick.moveit_contract import GripperTarget
from rocobrick.multi_robot_scheduler import (
    JSONValue,
    OperationExecution,
    OperationStatus,
)
from rocobrick.planning_scene import PlanningSceneUpdate, box_from_aabb, scene_digest
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.contact import inspect_contact
from rocobrick.skills.observation import (
    PlaceObservation,
    Pose,
    _pose,
    parse_place_observation,
    verify_place_bricks,
)
from rocobrick.skills.placement_geometry import (
    compose,
    interface_place_up_targets,
    inverse,
)
from rocobrick.skills.successor_motion import move_boundary
from rocobrick.skills.support_evidence import verify_support_contact
from rocobrick.skills.targets import (
    _quat_rotate_vector,
    _tcp_pose_target,
    world_orientation_to_base,
    world_position_to_base,
)
from rocobrick.unified_dual_arm_runtime import SuccessorPrePhaseEvidence


def _native_press_ready(measured: Mapping[str, object]) -> bool:
    """Require native interface geometry before allowing bounded compression.

    Returns:
        Whether a fresh identified diagnostic satisfies every geometric gate.
    """
    thresholds = measured.get("native_assembly_thresholds", {})
    diagnostics = measured.get("native_assembly_diagnostics", [])
    if not isinstance(thresholds, Mapping) or not isinstance(diagnostics, list):
        return False
    if thresholds.get("enabled") is not True:
        return False
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        names = ("relative_distance", "position_error", "tilt", "yaw_error")
        limits = (
            "distance_tolerance",
            "max_penetration",
            "position_tolerance",
            "z_angle_tolerance",
            "yaw_tolerance",
        )
        values = [diagnostic.get(name) for name in names] + [
            thresholds.get(name) for name in limits
        ]
        if not all(
            type(value) in (int, float) and math.isfinite(value) for value in values
        ):
            continue
        (
            distance,
            position,
            tilt,
            yaw,
            tolerance,
            penetration,
            planar,
            angular,
            yaw_limit,
        ) = values
        if (
            -penetration <= distance <= tolerance
            and 0 <= position <= planar
            and abs(tilt) <= angular
            and abs(yaw) <= yaw_limit
        ):
            return True
    return False


class PlaceUpObservationTransport(Protocol):
    """Read one robot's live scene without sending motion."""

    async def observe(self, robot_id: str) -> Mapping[str, object]:
        """Return one measured scene for the named robot."""


class PlaceUpFreeSpaceTransport(Protocol):
    """Dispatch the dual-arm free-space approach only."""

    async def execute_free_space(
        self, request: FreeSpaceMotionRequest
    ) -> FreeSpaceMotionEvidence:
        """Execute the skill-owned pre-contact approach."""

    async def free_space_context(
        self, participants: tuple[str, ...]
    ) -> tuple[Mapping[str, object], int, Mapping[str, float]]:
        """Read one shared multi-robot scene with the complete start state."""


class PlaceUpMotionTransport(Protocol):
    """Dispatch one robot-scoped runtime operation."""

    def run_motion_operation(
        self, robot_id: str, operation: RuntimeOperation
    ) -> Mapping[str, object]:
        """Run one runtime operation and return its strict record."""

    def cancel_motion_operation(self, operation_id: str) -> Mapping[str, object]:
        """Cancel one currently active runtime request."""


@dataclass(frozen=True)
class PlaceUpAdmission:
    """Immutable upward pair target and measured upper/lower sample."""

    operation_id: str
    robot_id: str
    support_robot_id: str
    connection_id: str
    upper_physical_instance_id: str
    lower_physical_instance_id: str
    sample: PlaceObservation


@dataclass(frozen=True)
class _MotionEvidence:
    """Strict normalized result of one upward placement phase."""

    passed: bool
    terminal: bool
    safe: bool
    motion_sent: bool
    evidence: Mapping[str, JSONValue]
    reason: str | None


class PlaceUpHandler:
    """Execute upward insertion and commit it only after separated proofs."""

    def __init__(
        self,
        observation_transport: PlaceUpObservationTransport,
        ledger: OperationStateLedger,
        *,
        session_id: str,
        manifest_sha256: str,
        motion_transport: PlaceUpMotionTransport | None = None,
        free_space_transport: PlaceUpFreeSpaceTransport | None = None,
        context: TransitContext | None = None,
        poll_interval_s: float = 0.2,
    ) -> None:
        """Bind one session identity, ledger, motion transport and calibration."""
        self._observations = observation_transport
        self._ledger = ledger
        self._session_id = session_id
        self._manifest_sha256 = manifest_sha256
        self._motion = motion_transport
        self._free_space = free_space_transport
        self._context = context
        self._poll_interval_s = float(poll_interval_s)
        self._active_requests: dict[str, str] = {}
        self._cancelled_operations: set[str] = set()
        self._pre_phases: dict[str, SuccessorPrePhaseEvidence] = {}
        self._active_boundaries: dict[str, str] = {}

    def bind_pre_phase(
        self, operation_id: str, evidence: SuccessorPrePhaseEvidence
    ) -> None:
        """Bind validated successor arrival without rewriting the input plan."""
        if (
            not isinstance(evidence, SuccessorPrePhaseEvidence)
            or not evidence.passed
            or operation_id not in evidence.successor_operation_ids
        ):
            raise ValueError("PLACE_UP pre-phase receipt is not bound")
        self._pre_phases[operation_id] = evidence

    async def execute_free_space_phase(
        self,
        operation: RobotOperation,
        *,
        scene_snapshot: Mapping[str, object],
        scene_version: int,
        targets: Mapping[str, Mapping[str, object]],
        start_state: Mapping[str, float],
        session_robots: tuple[str, ...] | None = None,
    ) -> FreeSpaceMotionEvidence:
        """Execute pre-contact dual-arm approach, never insertion or release.

        Returns:
            Bound physical free-space completion evidence.
        """
        if self._free_space is None:
            raise RuntimeError("PLACE_UP free-space transport is unavailable")
        session_ids = session_robots or tuple(
            getattr(self._free_space, "robot_ids", operation.participants)
        )
        request = FreeSpaceMotionRequest(
            motion_id=f"{operation.operation_id}:free-space",
            session_robots=tuple(session_ids),
            participants=tuple(operation.participants),
            scene_snapshot=scene_snapshot,
            scene_version=scene_version,
            targets=targets,
            start_state=start_state,
            timeout_s=60.0,
        )
        return await self._free_space.execute_free_space(request)

    async def preflight(self, operation: RobotOperation) -> PlaceUpAdmission:
        """Validate the upward pair, holding state and supporting fact.

        Returns:
            The admitted upward operation with a fresh co-sampled observation.
        """
        if operation.skill != "PLACE_UP" or operation.target is None:
            raise ValueError("PLACE_UP target is required")
        target = operation.target.values
        expected = {
            "connection_id",
            "upper_physical_instance_id",
            "lower_physical_instance_id",
            "support_robot_id",
        }
        if set(target) != expected:
            raise ValueError("PLACE_UP target shape is invalid")
        upper = target["upper_physical_instance_id"]
        lower = target["lower_physical_instance_id"]
        support_robot = target["support_robot_id"]
        if not all(
            isinstance(value, str) and value
            for value in (target["connection_id"], upper, lower, support_robot)
        ):
            raise ValueError("PLACE_UP target values must be nonempty strings")
        if upper == lower or support_robot == operation.executor_robot_id:
            raise ValueError("PLACE_UP pair or support robot is invalid")
        if self._ledger.owner_of_object(lower) != operation.executor_robot_id:
            raise ValueError(
                f"PLACE_UP lower part {lower!r} is not confirmed held by "
                f"{operation.executor_robot_id!r}"
            )
        if self._ledger.observe(support_robot, f"supporting:{upper}") is None:
            raise ValueError(
                f"PLACE_UP upper part {upper!r} is not actively supported by "
                f"{support_robot!r}"
            )
        sample = await self._observe(operation.executor_robot_id, upper, lower)
        return PlaceUpAdmission(
            operation.operation_id,
            operation.executor_robot_id,
            support_robot,
            target["connection_id"],
            upper,
            lower,
            sample,
        )

    async def _observe(self, robot_id: str, upper: str, lower: str) -> PlaceObservation:
        """Read one strict upward pair observation.

        Returns:
            The live co-sampled upper/lower/TCP observation.
        """
        return parse_place_observation(
            await self._observations.observe(robot_id),
            robot_id=robot_id,
            upper_physical_instance_id=upper,
            lower_physical_instance_id=lower,
            session_id=self._session_id,
            manifest_sha256=self._manifest_sha256,
        )

    async def _guard_scene(
        self, robot_id: str, support_robot_id: str
    ) -> Mapping[str, object]:
        """Read the shared scene the insert guard's proof needs.

        The up-mode contact guard proves the SUPPORT robot's pad reaction while
        checking the executor's bricks, so it needs one instant carrying both
        robots' blocks and the global PhysX pad report. A robot-scoped
        observation cannot serve it: by design it carries only the requesting
        robot's own contact records, so the peer's proof could never find them
        and the insertion failed closed before moving (live evidence,
        task-c-top-offset13j-headless 2026-10-14: ``support_contact_lost`` while
        piper_1's own slice showed the support pressing at 5.21 N).

        Returns:
            The shared multi-robot scene snapshot covering both robots.
        """
        if self._free_space is None:
            raise ValueError("place-up contact guard requires the shared scene")
        snapshot, _, _ = await self._free_space.free_space_context(
            (robot_id, support_robot_id)
        )
        return snapshot

    async def _regrip_before_insert(
        self, operation: RobotOperation, sample: PlaceObservation
    ) -> _MotionEvidence | None:
        """Tighten the grip on the held brick before the upward insertion.

        The insertion carries its whole load through pad friction: the support
        robot's press on the upper brick transfers to the held lower brick at
        the stud engagement, so a grip that carried the transport can still
        creep under it (live evidence, task-c-top-offset13l/13p-headless: the
        brick slid 3.0-3.2 mm in the jaws at the engagement and the guard
        stopped the insertion).  Commanding a further
        ``place_up_regrip_closing_m`` of closure presses the already touching
        pads harder -- unlike the one-shot close, this cannot eject the brick,
        because the pads are already gripping it.

        Returns:
            The regrip dispatch evidence, or None when it is disabled.
        """
        closing = self._context.params.place_up_regrip_closing_m
        if closing <= 0.0:
            return None
        if self._motion is None:
            raise ValueError("pre-insert regrip requires the motion transport")
        measured = float(sample.lower.opening_m)
        target = max(0.0, measured - closing)
        window = (
            max(0.0, measured - self._context.params.contact_window_slop_m),
            measured + self._context.params.contact_window_slop_m,
        )
        regrip = gripper_operation(
            GripperTarget(
                opening_m=target,
                duration_s=self._context.params.gripper_duration_s,
                goal_tolerance=self._context.params.gripper_goal_tolerance,
                expected_opening_m=window,
            ),
            60.0,
            f"{operation.operation_id}:regrip",
        )
        return await self._dispatch(operation, "regrip", regrip)

    async def _dispatch(
        self, operation: RobotOperation, phase: str, runtime_operation: RuntimeOperation
    ) -> _MotionEvidence:
        """Dispatch one phase and normalize strict lifecycle evidence.

        Returns:
            The action's independently observed lifecycle proof.
        """
        if operation.operation_id in self._cancelled_operations:
            return _MotionEvidence(False, True, True, False, {}, "operation cancelled")
        if self._motion is None:
            return _MotionEvidence(
                False, True, False, False, {}, "motion transport unavailable"
            )
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
        raw = record.get("result") if isinstance(record, Mapping) else None
        execution = raw.get("execution") if isinstance(raw, Mapping) else None
        passed = raw.get("passed") is True if isinstance(raw, Mapping) else False
        non_motion = getattr(
            runtime_operation.kind, "value", str(runtime_operation.kind)
        ) in {
            "APPLY_PLANNING_SCENE",
            "ATTACH_PLANNING_SCENE",
            "DETACH_PLANNING_SCENE",
        }
        terminal = (
            isinstance(raw, Mapping)
            if non_motion
            else (
                execution.get("terminal") is True
                if isinstance(execution, Mapping)
                else False
            )
        )
        safe = (
            isinstance(raw, Mapping)
            if non_motion
            else (
                execution.get("safe_to_release") is True
                if isinstance(execution, Mapping)
                else False
            )
        )
        command_messages = (
            execution.get("command_messages")
            if isinstance(execution, Mapping)
            else None
        )
        sent = (
            record.get("motion_sent") is True
            or type(command_messages) is int
            and command_messages > 0
            or isinstance(raw, Mapping)
            and raw.get("evidence", {}).get("motion_sent") is True
        )
        evidence: dict[str, JSONValue] = {
            "phase": phase,
            "request_id": runtime_operation.request_id,
            "runtime": dict(raw) if isinstance(raw, Mapping) else {},
        }
        return _MotionEvidence(
            passed and terminal and safe,
            terminal,
            safe,
            sent,
            evidence,
            None
            if passed and terminal and safe
            else f"{phase}: runtime evidence failed",
        )

    async def cancel(self, operation_id: str) -> Mapping[str, object]:
        """Cancel the active upward phase request.

        Returns:
            The actual active action or participant group's cancellation.
        """
        self._cancelled_operations.add(operation_id)
        motion_id = self._active_boundaries.get(operation_id)
        if motion_id is not None:
            return await self._free_space.cancel_free_space(motion_id)
        request_id = self._active_requests.get(operation_id)
        if request_id is None or self._motion is None:
            return {"attempted": False, "terminal": False, "safe": False}
        return await asyncio.to_thread(self._motion.cancel_motion_operation, request_id)

    def _feedback_target(self, sample, guard, offset_m):
        """Derive a TCP goal from the live brick/TCP relation, not the old grasp.

        Returns:
            Calibrated target for a bounded offset along the live stud axis.
        """
        supported = _pose(guard["supported_pose"], "supported upper")
        live_seat = compose(
            compose(sample.upper.object_pose, inverse(supported)),
            _pose(guard["desired_lower"], "desired lower"),
        )
        normal = _quat_rotate_vector(
            sample.upper.object_pose.orientation_xyzw, (0, 0, 1)
        )
        desired = Pose(
            tuple(live_seat.position_xyz[i] + normal[i] * offset_m for i in range(3)),
            live_seat.orientation_xyzw,
        )
        measured_grasp = compose(
            inverse(sample.lower.object_pose), sample.lower.tcp_pose
        )
        tcp = compose(desired, measured_grasp)
        return _tcp_pose_target(
            world_position_to_base(tcp.position_xyz, self._context.base),
            world_orientation_to_base(tcp.orientation_xyzw, self._context.base),
            self._context.provenance,
            self._context.params,
        )

    async def _feedback_insert(self, operation, admission, guard, evidence):
        """Align off-contact, then advance only from fresh measured object progress.

        Returns:
            Last terminal action evidence and explicit connection/safety stop.
        """
        step_m = self._context.params.place_up_feedback_step_m
        feedback = {"step_m": step_m, "alignment_standoff_m": 0.008, "steps": []}
        evidence["insertion_feedback"] = feedback
        last = _MotionEvidence(False, True, True, False, {}, "no insertion motion")
        stalled = 0
        axial_commanded = 0.0
        compression_commanded = 0.0
        loop_deadline = asyncio.get_running_loop().time() + 240.0
        for index in range(32):
            if asyncio.get_running_loop().time() >= loop_deadline:
                return last, "insertion_feedback_deadline"
            scene = await self._guard_scene(
                admission.robot_id, admission.support_robot_id
            )
            reason, before = inspect_contact(scene, guard)
            feedback["last_observation"] = before
            if reason is not None:
                return last, reason
            sample = parse_place_observation(
                scene,
                robot_id=admission.robot_id,
                upper_physical_instance_id=admission.upper_physical_instance_id,
                lower_physical_instance_id=admission.lower_physical_instance_id,
                session_id=self._session_id,
                manifest_sha256=self._manifest_sha256,
            )
            aligning = index < 2
            near_contact = _native_press_ready(before)
            force = before.get("contact_force_n")
            compressing = near_contact and force is not None and abs(force) > 0.05
            if aligning:
                if (
                    abs(before["seat_axial_error_m"] + 0.008) <= 0.001
                    and before["seat_lateral_error_m"] <= 0.001
                    and before["seat_angle_rad"] <= math.radians(1)
                ):
                    continue
                offset = -0.008
            else:
                if before["seat_lateral_error_m"] > 0.001 or before[
                    "seat_angle_rad"
                ] > math.radians(3 if compressing else 1):
                    return last, "insertion_alignment_not_confirmed"
                offset = min(
                    before["seat_axial_error_m"] + min(step_m, 0.00025)
                    if near_contact
                    else before["seat_axial_error_m"] + step_m,
                    self._context.params.place_press_m,
                )
                if before["seat_axial_error_m"] >= self._context.params.place_press_m:
                    return last, "insertion_depth_bound_reached"
            target = self._feedback_target(sample, guard, offset)
            suffix = f"align_up_{index}" if aligning else f"insert_up_step_{index - 2}"
            last = await self._dispatch(
                operation,
                suffix,
                guarded_tcp_operation(
                    target, guard, 120.0, f"{operation.operation_id}:{suffix}"
                ),
            )
            details = (
                last.evidence.get("runtime", {}).get("execution", {}).get("details", {})
            )
            recorded = {
                "phase": suffix,
                "before": before,
                "target_offset_m": offset,
                "motion": dict(last.evidence),
                "bounded_compression": compressing,
            }
            feedback["steps"].append(recorded)
            stop = details.get("contact_stop_reason")
            if not last.terminal or not last.safe:
                return last, stop or "insertion_action_terminal_unknown"
            if stop is not None or not last.passed:
                return last, stop or "insertion_step_failed"
            after_scene = await self._guard_scene(
                admission.robot_id, admission.support_robot_id
            )
            reason, after = inspect_contact(after_scene, guard)
            recorded["after"] = after
            feedback["last_observation"] = after
            if after["sequence"] <= before["sequence"]:
                return last, "insertion_feedback_stale"
            if reason is not None:
                return last, reason
            if not aligning:
                normal = _quat_rotate_vector(
                    sample.upper.object_pose.orientation_xyzw, (0, 0, 1)
                )
                after_sample = parse_place_observation(
                    after_scene,
                    robot_id=admission.robot_id,
                    upper_physical_instance_id=admission.upper_physical_instance_id,
                    lower_physical_instance_id=admission.lower_physical_instance_id,
                    session_id=self._session_id,
                    manifest_sha256=self._manifest_sha256,
                )
                tcp_advance = sum(
                    normal[i]
                    * (
                        after_sample.lower.tcp_pose.position_xyz[i]
                        - sample.lower.tcp_pose.position_xyz[i]
                    )
                    for i in range(3)
                )
                object_advance = (
                    after["seat_axial_error_m"] - before["seat_axial_error_m"]
                )
                recorded["tcp_advance_m"] = tcp_advance
                recorded["object_advance_m"] = object_advance
                axial_commanded += max(0.0, tcp_advance)
                if compressing:
                    compression_commanded += max(0.0, tcp_advance)
                    recorded["compression_commanded_m"] = compression_commanded
                    if compression_commanded > 0.002:
                        return last, "insertion_compression_budget_exhausted"
                commanded_step = offset - before["seat_axial_error_m"]
                stalled = (
                    stalled + 1
                    if tcp_advance > commanded_step * 0.5
                    and object_advance < commanded_step * 0.2
                    else 0
                )
                if stalled >= 2 and not compressing:
                    return last, "insertion_object_not_following"
                if axial_commanded > 0.008 + self._context.params.place_press_m + 0.003:
                    return last, "insertion_command_budget_exhausted"
        return last, "insertion_step_budget_exhausted"

    def _world_payload(
        self,
        admission: PlaceUpAdmission,
        sample: PlaceObservation,
        suffix: str,
        shared: Mapping[str, object],
    ) -> dict[str, object]:
        """Build a measured world planning scene for the upward pair.

        Returns:
            Whole-world geometry and phase-scoped contact permissions.
        """
        if self._context is None:
            raise ValueError("PLACE_UP context unavailable")
        frame_id = "dual_world"
        # Applying a contact ACM diff must not delete the baseplate or any
        # unrelated structure/supply obstacle from the shared MoveIt world.
        boxes = [
            box_from_aabb(
                object_id=identity,
                frame_id=frame_id,
                minimum_m=brick["aabb_min"],
                maximum_m=brick["aabb_max"],
            )
            for identity, brick in shared["bricks"].items()
            if identity != admission.lower_physical_instance_id
        ]
        environment = shared.get("environment_colliders")
        if isinstance(environment, Mapping) and environment:
            boxes.extend(
                box_from_aabb(
                    object_id=f"environment:{identity}",
                    frame_id=frame_id,
                    minimum_m=brick["aabb_min"],
                    maximum_m=brick["aabb_max"],
                )
                for identity, brick in environment.items()
            )
        else:
            ground = shared["ground"]
            boxes.append(
                box_from_aabb(
                    object_id="unified_ground",
                    frame_id=frame_id,
                    minimum_m=ground["aabb_min"],
                    maximum_m=ground["aabb_max"],
                )
            )
        pairs = (
            tuple(
                sorted(
                    (
                        admission.lower_physical_instance_id,
                        admission.upper_physical_instance_id,
                    )
                )
            ),
        )
        update = PlanningSceneUpdate(
            scene_id=f"{admission.operation_id}-{suffix}",
            boxes=tuple(boxes),
            expected_scene_digest="0" * 64,
            allowed_collision_pairs=pairs,
        )
        digest = scene_digest(update)
        return asdict(
            PlanningSceneUpdate(
                scene_id=update.scene_id,
                boxes=update.boxes,
                expected_scene_digest=digest,
                allowed_collision_pairs=update.allowed_collision_pairs,
            )
        )

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        """Run upward transport, insertion, release, and collision-proven retreat.

        Returns:
            Terminal operation evidence with separate phase records.
        """
        motion_sent = False
        terminal_safe = True
        release_attempted = False
        evidence: dict[str, JSONValue] = {}
        try:
            admission = await self.preflight(operation)
            if self._context is None or self._motion is None:
                raise ValueError("PLACE_UP context or motion transport unavailable")
            if self._free_space is None:
                raise ValueError("PLACE_UP unified transport unavailable")
            sample = admission.sample
            pair_key = "|".join(
                sorted(
                    (
                        admission.upper_physical_instance_id,
                        admission.lower_physical_instance_id,
                    )
                )
            )
            interfaces = sample.interfaces
            pairs = interfaces.get("pairs") if isinstance(interfaces, Mapping) else None
            interface = pairs.get(pair_key) if isinstance(pairs, Mapping) else None
            if not isinstance(interface, Mapping):
                raise ValueError("PLACE_UP interface geometry unavailable")
            connection_pose = _pose(interface.get("lower_to_upper"), "connection pose")
            targets, desired_lower, held = interface_place_up_targets(
                sample.upper.object_pose,
                connection_pose,
                sample.lower.object_pose,
                sample.lower.tcp_pose,
                sample.lower.opening_m,
                self._context.base,
                self._context.provenance,
                self._context.params,
                yaw_symmetries=tuple(
                    interface.get("allowed_upper_yaw_quarter_turns", (0,))
                ),
            )
            evidence["place_up_geometry"] = {
                "source": "stud_hole_interface_and_measured_lower_grasp",
                "interface": dict(interface),
                "desired_lower_pose": asdict(desired_lower),
                "lower_to_tcp": asdict(held),
                "contact_tcp_world_z_m": targets.contact_world_z_m,
            }
            receipt = self._pre_phases.pop(operation.operation_id, None)
            if receipt is not None:
                evidence["pre_place_up"] = {
                    "source": "successor_pre_phase",
                    "transition_id": receipt.transition_id,
                    **dict(receipt.evidence),
                }
            else:
                approach = await move_boundary(
                    self._free_space,
                    operation,
                    self._context,
                    targets.preplace,
                    phase="pre_place_up",
                    active_requests=self._active_boundaries,
                )
                evidence["pre_place_up"] = dict(approach.evidence)
                motion_sent = approach.started
                terminal_safe = approach.terminal and approach.safe_to_release
                if not approach.passed:
                    raise RuntimeError(approach.reason or "PLACE_UP pre-place failed")
            # Contact permission is enabled only AFTER collision-proven arrival.
            contact_scene, _, _ = await self._free_space.free_space_context(
                tuple(operation.participants)
            )
            planning = planning_scene_operation(
                self._world_payload(admission, sample, "plan_place_up", contact_scene),
                60.0,
                f"{operation.operation_id}:plan_place_up",
            )
            result = await self._dispatch(operation, "plan_place_up", planning)
            evidence.update(result.evidence)
            motion_sent = motion_sent or result.motion_sent
            terminal_safe = result.terminal and result.safe
            if not result.passed:
                raise RuntimeError(result.reason or "plan_place_up failed")
            regrip_sample = await self._observe(
                admission.robot_id,
                admission.upper_physical_instance_id,
                admission.lower_physical_instance_id,
            )
            regrip = await self._regrip_before_insert(operation, regrip_sample)
            if regrip is not None:
                evidence["regrip"] = dict(regrip.evidence)
                motion_sent = motion_sent or regrip.motion_sent
                terminal_safe = regrip.terminal and regrip.safe
                if not regrip.passed:
                    raise RuntimeError(regrip.reason or "pre-insert regrip failed")
            aligned = await self._observe(
                admission.robot_id,
                admission.upper_physical_instance_id,
                admission.lower_physical_instance_id,
            )
            # Anchor contact guards to the actual post-transport grasp, not
            # the old pose that existed before public-space movement.
            held = compose(inverse(aligned.lower.object_pose), aligned.lower.tcp_pose)
            feedback_enabled = self._context.params.place_up_feedback_step_m > 0
            if feedback_enabled:
                # Keep the admitted interface/yaw branch, but anchor its seat
                # to the same fresh upper frame as the connection proof.
                desired_lower = compose(
                    compose(
                        aligned.upper.object_pose, inverse(sample.upper.object_pose)
                    ),
                    desired_lower,
                )
                evidence["place_up_geometry"]["feedback_desired_lower_pose"] = asdict(
                    desired_lower
                )
            verify_support_contact(
                await self._observations.observe(admission.support_robot_id),
                robot_id=admission.support_robot_id,
                upper_id=admission.upper_physical_instance_id,
                session_id=self._session_id,
                manifest_sha256=self._manifest_sha256,
            )
            guard = {
                "robot_id": admission.robot_id,
                "support_robot_id": admission.support_robot_id,
                "session_id": self._session_id,
                "manifest_sha256": self._manifest_sha256,
                "upper_id": admission.upper_physical_instance_id,
                "lower_id": admission.lower_physical_instance_id,
                "supported_pose": asdict(aligned.upper.object_pose),
                "held_transform": asdict(held),
                "desired_lower": asdict(desired_lower),
                "supported_position_m": 0.002,
                "supported_angle_rad": math.radians(3),
                # Continuous insertion retains the measured-feedback trial's
                # custody envelope; faster motion never loosens slip bounds.
                "grasp_position_m": 0.003,
                "grasp_angle_rad": math.radians(3),
                "force_limit_n": 8.0,
                "mode": "insert_up",
            }
            evidence["contact_guard"] = guard
            reason, measured = inspect_contact(
                await self._guard_scene(admission.robot_id, admission.support_robot_id),
                guard,
            )
            if reason == "connection_confirmed":
                connected = True
            elif reason is not None:
                raise ValueError(f"contact guard stopped upward insertion: {reason}")
            else:
                if feedback_enabled:
                    result, feedback_stop = await self._feedback_insert(
                        operation, admission, guard, evidence
                    )
                    evidence["insertion_feedback"]["stop_reason"] = feedback_stop
                else:
                    result = await self._dispatch(
                        operation,
                        "insert_up",
                        guarded_tcp_operation(
                            targets.contact,
                            guard,
                            120.0,
                            f"{operation.operation_id}:insert_up",
                        ),
                    )
                evidence["insertion_up"] = dict(result.evidence)
                motion_sent = motion_sent or result.motion_sent
                terminal_safe = result.terminal and result.safe
                if not result.terminal or not result.safe:
                    raise RuntimeError(
                        "upward insertion action has unknown terminal safety"
                    )
                runtime = result.evidence.get("runtime", {})
                details = runtime.get("execution", {}).get("details", {})
                stop = (
                    feedback_stop
                    if feedback_enabled
                    else details.get("contact_stop_reason")
                )
                connected = stop == "connection_confirmed"
                if not result.passed and not connected:
                    # A rejected/failed primitive invalidates this session.
                    # Do not mask its root error with another scene request,
                    # or submit a recovery action to a failed session.
                    raise RuntimeError(
                        result.reason or f"upward insertion primitive failed: {stop}"
                    )
                if not connected:
                    await asyncio.sleep(self._poll_interval_s)
                    settle_reason, _ = inspect_contact(
                        await self._guard_scene(
                            admission.robot_id, admission.support_robot_id
                        ),
                        guard,
                    )
                    connected = settle_reason == "connection_confirmed"
                if not connected:
                    # No press retries or release under an unconnected load.
                    # Withdraw once only after the insertion actually stopped.
                    unload_guard = {**guard, "mode": "unload_up"}
                    # Recovery retains the previously validated gross custody
                    # envelope; no release or success claim occurs here.
                    unload_guard["grasp_position_m"] = 0.008
                    unload_guard["grasp_angle_rad"] = 0.15
                    withdrawn = await self._dispatch(
                        operation,
                        "unload_up",
                        guarded_tcp_operation(
                            targets.preplace,
                            unload_guard,
                            120.0,
                            f"{operation.operation_id}:unload_up",
                        ),
                    )
                    evidence["failure_withdrawal"] = dict(withdrawn.evidence)
                    motion_sent = motion_sent or withdrawn.motion_sent
                    terminal_safe = withdrawn.terminal and withdrawn.safe
                    if not withdrawn.passed:
                        raise RuntimeError(
                            withdrawn.reason or "upward contact withdrawal failed"
                        )
                    withdrawal_scene = await self._guard_scene(
                        admission.robot_id, admission.support_robot_id
                    )
                    withdrawal_reason, withdrawal_sample = inspect_contact(
                        withdrawal_scene, unload_guard
                    )
                    if withdrawal_reason is not None:
                        raise ValueError(
                            f"withdrawal proof failed: {withdrawal_reason}"
                        )
                    evidence["withdrawal_confirmed"] = withdrawal_sample
                    raise ValueError(
                        f"upward insertion did not confirm connection: {stop}"
                    )
            connection_samples = [
                await self._observe(
                    admission.robot_id,
                    admission.upper_physical_instance_id,
                    admission.lower_physical_instance_id,
                )
            ]
            for _ in range(2):
                await asyncio.sleep(self._poll_interval_s)
                connection_samples.append(
                    await self._observe(
                        admission.robot_id,
                        admission.upper_physical_instance_id,
                        admission.lower_physical_instance_id,
                    )
                )
            evidence["connection_confirmed"] = verify_place_bricks(
                connection_samples,
                position_tolerance_m=self._context.params.still_tolerance_m,
                orientation_tolerance_rad=self._context.params.follow_orientation_tolerance_rad,
                require_connected=True,
            )
            release = gripper_operation(
                GripperTarget(
                    opening_m=targets.release_opening_m,
                    duration_s=self._context.params.gripper_duration_s,
                    goal_tolerance=self._context.params.gripper_goal_tolerance,
                    expected_opening_m=targets.release_expected_window_m,
                ),
                60.0,
                f"{operation.operation_id}:release",
            )
            release_attempted = True
            result = await self._dispatch(operation, "release", release)
            evidence.update(result.evidence)
            motion_sent = motion_sent or result.motion_sent
            terminal_safe = result.terminal and result.safe
            if not result.passed:
                raise RuntimeError(result.reason or "release failed")
            release_samples = [
                await self._observe(
                    admission.robot_id,
                    admission.upper_physical_instance_id,
                    admission.lower_physical_instance_id,
                )
            ]
            for _ in range(2):
                await asyncio.sleep(self._poll_interval_s)
                release_samples.append(
                    await self._observe(
                        admission.robot_id,
                        admission.upper_physical_instance_id,
                        admission.lower_physical_instance_id,
                    )
                )
            evidence["release_confirmed"] = verify_place_bricks(
                release_samples,
                position_tolerance_m=self._context.params.still_tolerance_m,
                orientation_tolerance_rad=self._context.params.follow_orientation_tolerance_rad,
                require_connected=True,
            )
            if not all(
                targets.release_expected_window_m[0]
                <= item.lower.opening_m
                <= targets.release_expected_window_m[1]
                for item in release_samples
            ):
                raise ValueError("PLACE_UP release opening not physically confirmed")
            detach = await self._dispatch(
                operation,
                "detach",
                detached_planning_scene_operation(
                    {
                        "scene_id": f"{operation.operation_id}-detach",
                        "object_id": admission.lower_physical_instance_id,
                        "link_name": f"{admission.robot_id}_rocobrick_tcp",
                        "operation": "detach",
                    },
                    60.0,
                    f"{operation.operation_id}:detach",
                ),
            )
            evidence["detach"] = dict(detach.evidence)
            if not detach.passed:
                raise RuntimeError(detach.reason or "detach failed")
            self._ledger.revoke(
                admission.robot_id,
                f"holding:{admission.lower_physical_instance_id}",
                "measured release after exact connection",
            )
            retreat = await move_boundary(
                self._free_space,
                operation,
                self._context,
                targets.retreat,
                phase="retreat_up",
                active_requests=self._active_boundaries,
            )
            evidence["retreat_motion"] = dict(retreat.evidence)
            motion_sent = motion_sent or retreat.started
            terminal_safe = retreat.terminal and retreat.safe_to_release
            if not retreat.passed:
                raise RuntimeError(retreat.reason or "PLACE_UP retreat failed")
            final_samples = []
            for index in range(3):
                if index:
                    await asyncio.sleep(self._poll_interval_s)
                final_samples.append(
                    await self._observe(
                        admission.robot_id,
                        admission.upper_physical_instance_id,
                        admission.lower_physical_instance_id,
                    )
                )
                verify_support_contact(
                    await self._observations.observe(admission.support_robot_id),
                    robot_id=admission.support_robot_id,
                    upper_id=admission.upper_physical_instance_id,
                    session_id=self._session_id,
                    manifest_sha256=self._manifest_sha256,
                )
            evidence["retreat_confirmed"] = verify_place_bricks(
                final_samples,
                position_tolerance_m=self._context.params.still_tolerance_m,
                orientation_tolerance_rad=self._context.params.follow_orientation_tolerance_rad,
                require_connected=True,
            )
            departure = (
                release_samples[-1].lower.tcp_pose.position_xyz[2]
                - final_samples[-1].lower.tcp_pose.position_xyz[2]
            )
            if departure < self._context.params.place_retreat_height_m * 0.5:
                raise ValueError("PLACE_UP actual downward retreat not confirmed")
            evidence["retreat_confirmed"]["actual_downward_tcp_distance_m"] = departure
            evidence["terminal_safe"] = True
            # Support remained constrained through release and departure; its
            # role can end now without adding a fabricated robot movement.
            self._ledger.revoke(
                admission.support_robot_id,
                f"supporting:{admission.upper_physical_instance_id}",
                "PLACE_UP retreat physically complete",
            )
            self._ledger.confirm(
                admission.robot_id,
                f"placed:{admission.lower_physical_instance_id}",
                {
                    "connection_id": admission.connection_id,
                    "upper_physical_instance_id": admission.upper_physical_instance_id,
                    "lower_physical_instance_id": admission.lower_physical_instance_id,
                    "connection_confirmed": dict(evidence["connection_confirmed"]),
                    "release_confirmed": dict(evidence["release_confirmed"]),
                    "retreat_confirmed": dict(evidence["retreat_confirmed"]),
                },
            )
            return OperationExecution(
                operation_id=operation.operation_id,
                robot_id=operation.executor_robot_id,
                status=OperationStatus.CONFIRMED,
                passed=True,
                terminal=True,
                motion_sent=motion_sent,
                evidence=evidence,
                reason="PLACE_UP_CONFIRMED",
            )
        except (ValueError, RuntimeError) as error:
            if (
                release_attempted
                and self._ledger.observe(
                    operation.executor_robot_id,
                    f"holding:{admission.lower_physical_instance_id}",
                )
                is not None
            ):
                self._ledger.invalidate_if_stale(
                    operation.executor_robot_id,
                    admission.lower_physical_instance_id,
                    "PLACE_UP release evidence incomplete",
                )
            return OperationExecution(
                operation_id=operation.operation_id,
                robot_id=operation.executor_robot_id,
                status=OperationStatus.FAILED,
                passed=False,
                terminal=terminal_safe,
                motion_sent=motion_sent,
                evidence=evidence,
                reason=f"PLACE_UP_FAILED: {error}",
            )
