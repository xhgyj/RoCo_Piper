"""Operation-scoped PLACE_DOWN orchestration with separated physical evidence.

Connection, release and retreat are proven by three INDEPENDENT evidence
paths before any assembled state is published:

- connection: the live bricksim ``are_parts_connected`` verdict for the
  exact pair, read from advancing scene observations while both bricks stay
  within tolerance of their first post-descent poses;
- release: measured gripper opening inside the commanded window AND the
  connected, undisturbed stack persisting across the opening;
- retreat: a real TCP rise with the connection still true and both bricks
  still stable afterwards.

The ledger's holding confirmation is required BEFORE any motion (the
carrying gate) and is revoked only after all three proofs exist; a failed
PLACE_DOWN therefore keeps the part resource retained, never released.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Protocol

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
    tcp_pose_operation,
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
    _pose,
    parse_place_observation,
    verify_place_bricks,
)
from rocobrick.skills.placement_geometry import (
    compose,
    interface_place_targets,
    inverse,
    pose_error,
)
from rocobrick.skills.targets import _quat_rotate_vector, world_position_to_base
from rocobrick.transit_adapter import TransitContext


class PlaceObservationTransport(Protocol):
    """Read one robot's live scene without sending motion."""

    async def observe(self, robot_id: str) -> Mapping[str, object]:
        """Return one measured scene for the named robot."""


class PlaceDownFreeSpaceTransport(Protocol):
    """Dispatch the free-space approach to the measured preplace target."""

    async def execute_free_space(
        self, request: FreeSpaceMotionRequest
    ) -> FreeSpaceMotionEvidence:
        """Execute the skill-owned pre-contact approach."""


class PlaceMotionTransport(Protocol):
    """Dispatch one robot-scoped runtime operation."""

    def run_motion_operation(
        self, robot_id: str, operation: RuntimeOperation
    ) -> Mapping[str, object]:
        """Run one runtime operation and return its strict record."""

    def cancel_motion_operation(self, operation_id: str) -> Mapping[str, object]:
        """Cancel one currently active runtime request."""


@dataclass(frozen=True)
class PlaceAdmission:
    """Immutable operation-scoped pair target and measured scene sample."""

    operation_id: str
    robot_id: str
    connection_id: str
    upper_physical_instance_id: str
    lower_physical_instance_id: str
    sample: PlaceObservation


@dataclass(frozen=True)
class _MotionEvidence:
    """Strict normalized result of one PLACE_DOWN phase dispatch."""

    passed: bool
    terminal: bool
    safe: bool
    motion_sent: bool
    evidence: Mapping[str, JSONValue]
    reason: str | None


def _quaternion_angle(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Return the angular distance between two xyzw quaternions in radians."""
    dot = abs(sum(a * b for a, b in zip(left, right)))
    return 2.0 * math.acos(min(1.0, dot))


class PlaceDownHandler:
    """Execute PLACE_DOWN and confirm placement only from live evidence."""

    def __init__(
        self,
        observation_transport: PlaceObservationTransport,
        ledger: OperationStateLedger,
        *,
        session_id: str,
        manifest_sha256: str,
        motion_transport: PlaceMotionTransport | None = None,
        free_space_transport: PlaceDownFreeSpaceTransport | None = None,
        context: TransitContext | None = None,
        poll_interval_s: float = 0.2,
        probe_contact: bool = False,
    ) -> None:
        """Bind one session identity, ledger, motion transport and calibration.

        ``probe_contact`` runs the validation variant: the supervised descent
        stops at the FIRST measured interface contact and the operation then
        fails closed after a verified withdrawal. It never publishes a placed
        state.
        """
        self._observations = observation_transport
        self._ledger = ledger
        self._session_id = session_id
        self._manifest_sha256 = manifest_sha256
        self._motion = motion_transport
        self._free_space = free_space_transport
        self._context = context
        self._poll_interval_s = float(poll_interval_s)
        self._probe_contact = bool(probe_contact)
        self._active_requests: dict[str, str] = {}
        self._cancelled_operations: set[str] = set()

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
        """Execute pre-contact placement approach only."""
        if self._free_space is None:
            raise RuntimeError("PLACE_DOWN free-space transport is unavailable")
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

    async def preflight(self, operation: RobotOperation) -> PlaceAdmission:
        """Validate the pair target, the carrying state and a fresh scene.

        Returns:
            The admitted operation target with its measured first sample.
        """
        if operation.skill != "PLACE_DOWN" or operation.target is None:
            raise ValueError("PLACE_DOWN target is required")
        target = operation.target.values
        if set(target) != {
            "connection_id",
            "upper_physical_instance_id",
            "lower_physical_instance_id",
        }:
            raise ValueError(
                "PLACE_DOWN target must name exactly the connection and pair"
            )
        connection_id = target["connection_id"]
        upper_id = target["upper_physical_instance_id"]
        lower_id = target["lower_physical_instance_id"]
        for name, value in (
            ("connection_id", connection_id),
            ("upper_physical_instance_id", upper_id),
            ("lower_physical_instance_id", lower_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"PLACE_DOWN {name} is invalid")
        if upper_id == lower_id:
            raise ValueError("PLACE_DOWN upper and lower parts are identical")
        robot_id = operation.executor_robot_id
        owner = self._ledger.owner_of_object(upper_id)
        if (
            owner != robot_id
            or self._ledger.observe(robot_id, f"holding:{upper_id}") is None
        ):
            raise ValueError(
                f"PLACE_DOWN upper part is not confirmed held by {robot_id} "
                f"(owner: {owner!r}); carrying must be proven before motion"
            )
        if "robot_at_place_region" in operation.preconditions:
            arrival = self._ledger.observe(robot_id, "robot_at_place_region")
            if arrival is None:
                raise ValueError(
                    "PLACE_DOWN has no confirmed place-region arrival; the "
                    "direct plan carries the transport itself"
                )
        sample = await self._observe(robot_id, upper_id, lower_id)
        return PlaceAdmission(
            operation.operation_id,
            robot_id,
            connection_id,
            upper_id,
            lower_id,
            sample,
        )

    async def _observe(
        self, robot_id: str, upper_id: str, lower_id: str
    ) -> PlaceObservation:
        """Read one strict placement observation for the named pair.

        Returns:
            The parsed two-brick sample with the live connection verdict.
        """
        return parse_place_observation(
            await self._observations.observe(robot_id),
            robot_id=robot_id,
            upper_physical_instance_id=upper_id,
            lower_physical_instance_id=lower_id,
            session_id=self._session_id,
            manifest_sha256=self._manifest_sha256,
        )

    def _world_payload(
        self,
        admission: PlaceAdmission,
        context: TransitContext,
        sample: PlaceObservation,
        *,
        include_upper: bool,
        scene_suffix: str,
    ) -> dict[str, object]:
        """Build the measured world boxes and the authorized place contact.

        Returns:
            The strict ``PlanningSceneUpdate`` payload document.
        """
        frame_id = "dual_world"
        boxes = [
            box_from_aabb(
                object_id=admission.lower_physical_instance_id,
                frame_id=frame_id,
                minimum_m=tuple(float(value) for value in sample.lower.aabb_min),
                maximum_m=tuple(float(value) for value in sample.lower.aabb_max),
            )
        ]
        if include_upper:
            boxes.append(
                box_from_aabb(
                    object_id=admission.upper_physical_instance_id,
                    frame_id=frame_id,
                    minimum_m=tuple(float(value) for value in sample.upper.aabb_min),
                    maximum_m=tuple(float(value) for value in sample.upper.aabb_max),
                )
            )
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
        # The authorized contact is the carried (attached) upper brick
        # touching the lower world brick at the final placement; the floor
        # stays a hard constraint for every robot link.  Pairs use the
        # canonical strictly-ordered form of the planning-scene contract.
        pairs = (
            tuple(
                sorted(
                    (
                        admission.upper_physical_instance_id,
                        admission.lower_physical_instance_id,
                    )
                )
            ),
        )
        update = PlanningSceneUpdate(
            scene_id=f"{admission.operation_id}-{scene_suffix}",
            boxes=tuple(boxes),
            expected_scene_digest="0" * 64,
            allowed_collision_pairs=pairs,
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
        """Dispatch one uniquely named phase and normalize strict lifecycle evidence.

        Returns:
            The normalized lifecycle evidence of this one phase.
        """
        if operation.operation_id in self._cancelled_operations:
            return _MotionEvidence(
                False, True, True, False, {}, "operation cancelled before phase"
            )
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
        raw_result = record.get("result") if isinstance(record, Mapping) else None
        execution = (
            raw_result.get("execution") if isinstance(raw_result, Mapping) else None
        )
        passed = (
            raw_result.get("passed") is True
            if isinstance(raw_result, Mapping)
            else False
        )
        non_motion_phase = getattr(
            runtime_operation.kind, "value", str(runtime_operation.kind)
        ) in {
            "APPLY_PLANNING_SCENE",
            "ATTACH_PLANNING_SCENE",
        }
        if non_motion_phase:
            terminal = isinstance(raw_result, Mapping)
            safe = terminal
        else:
            terminal = (
                execution.get("terminal") is True
                if isinstance(execution, Mapping)
                else False
            )
            safe = (
                execution.get("safe_to_release") is True
                if isinstance(execution, Mapping)
                else False
            )
        command_messages = (
            execution.get("command_messages")
            if isinstance(execution, Mapping)
            else None
        )
        runtime_evidence = (
            raw_result.get("evidence") if isinstance(raw_result, Mapping) else None
        )
        sent = (
            record.get("motion_sent") is True
            or (type(command_messages) is int and command_messages > 0)
            or (
                isinstance(runtime_evidence, Mapping)
                and runtime_evidence.get("motion_sent") is True
            )
        )
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
            None
            if passed and terminal and safe
            else f"{phase}: runtime evidence failed",
        )

    @staticmethod
    def _release_feedback(
        evidence: Mapping[str, JSONValue],
    ) -> Mapping[str, JSONValue]:
        """Validate measured gripper release feedback without claiming release.

        Mirrors the PICK close-feedback rule: a plain SUCCEEDED needs the
        measured opening inside the commanded window with fresh post-result
        states; a contact-limited cancellation is accepted only with safe
        terminal cancellation, settled feedback and an in-window opening.
        The physical release itself is confirmed later by the connected,
        undisturbed stack outliving the opened gripper.

        Returns:
            The JSON-safe release feedback evidence summary.
        """
        runtime = evidence.get("runtime")
        execution = runtime.get("execution") if isinstance(runtime, Mapping) else None
        details = execution.get("details") if isinstance(execution, Mapping) else None
        if not isinstance(execution, Mapping) or not isinstance(details, Mapping):
            raise ValueError("release gripper runtime details are unavailable")
        measured = details.get("measured_opening_m")
        window = details.get("expected_opening_m")
        post_state = details.get("post_result_state_messages")
        accepted = execution.get("goal_accepted") is True
        terminal = execution.get("terminal") is True
        safe = execution.get("safe_to_release") is True
        action_status = execution.get("action_status")
        contact_limited = details.get("contact_limited") is True
        cancel = details.get("cancel")
        contact_opening = details.get("contact_opening_m")
        status_ok = action_status == "SUCCEEDED"
        if contact_limited:
            status_ok = (
                action_status in ("CANCELED", "CANCELLED")
                and isinstance(cancel, Mapping)
                and cancel.get("terminal") is True
                and cancel.get("safe") is True
                and details.get("gripper_settled") is True
                and not isinstance(contact_opening, bool)
                and isinstance(contact_opening, (int, float))
                and math.isfinite(contact_opening)
            )
        if (
            isinstance(measured, bool)
            or not isinstance(measured, (int, float))
            or not math.isfinite(measured)
            or not isinstance(window, (list, tuple))
            or len(window) != 2
            or not all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                for value in window
            )
            or isinstance(post_state, bool)
            or not isinstance(post_state, int)
            or post_state < 1
            or not (float(window[0]) <= float(measured) <= float(window[1]))
            or not all((accepted, terminal, safe, status_ok))
            or (
                contact_limited
                and not (float(window[0]) <= contact_opening <= float(window[1]))
            )
        ):
            raise ValueError(
                "release gripper feedback did not prove the commanded opening"
            )
        return {
            "goal_accepted": True,
            "action_status": action_status,
            "contact_limited": contact_limited,
            "measured_opening_m": float(measured),
            "expected_opening_m": [float(window[0]), float(window[1])],
            "post_result_state_messages": post_state,
            "release_confirmation_deferred": "connected_stack_outlives_opening",
        }

    async def cancel(self, operation_id: str) -> Mapping[str, object]:
        """Cancel the currently active phase request for one operation.

        Returns:
            The transport cancel evidence, or an inert no-op record.
        """
        self._cancelled_operations.add(operation_id)
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

    async def _unload_failed_contact(
        self, operation, admission, targets, guard, evidence
    ):
        """Withdraw an unconnected held object with ONE supervised ascent.

        The withdrawal is the contact controller's own reverse mode (guard
        ``unload``): it is cancelled in flight if the connection appears, the
        held relation slips or the interface force stays high. Success still
        requires measured clearance, holding and support restoration.

        Returns:
            True only after measured clearance, holding and support restoration.
        """
        records = []
        evidence.setdefault("unload_attempts", []).append({"steps": records})
        evidence["failure_unload"] = {"passed": False, "steps": records}
        if operation.operation_id in self._cancelled_operations:
            return False
        context = self._context
        counters = getattr(self, "_unload_counts", {})
        count = counters.get(operation.operation_id, 0)
        counters[operation.operation_id] = count + 1
        self._unload_counts = counters
        try:
            start = await self._observe(
                admission.robot_id,
                admission.upper_physical_instance_id,
                admission.lower_physical_instance_id,
            )
            if start.connection:
                return False  # Never pull a connected brick with closed fingers.
            normal = _quat_rotate_vector(
                admission.sample.lower.object_pose.orientation_xyzw, (0.0, 0.0, 1.0)
            )
            unload_guard = {**guard, "mode": "unload"}
            # One continuous ascent past the verified-clearance bound. The
            # commanded rise is deeper than the 6 mm clearance proof so it
            # still clears under uncompensated gravity droop; the per-step
            # command lead of the retired microstep loop is replaced by the
            # runtime's live unload guard cancelling this action.
            target_world = tuple(
                start.upper.tcp_pose.position_xyz[i] + normal[i] * 0.012
                for i in range(3)
            )
            target = replace(
                targets.contact,
                position=world_position_to_base(target_world, context.base),
                position_tolerance_m=0.005,
            )
            phase = f"failure_unload_{count}_0"
            result = await self._dispatch(
                operation,
                phase,
                guarded_tcp_operation(
                    target,
                    unload_guard,
                    20.0,
                    f"{operation.operation_id}:{phase}",
                ),
            )
            records.append(dict(result.evidence))
            runtime_record = result.evidence.get("runtime", {})
            details = runtime_record.get("execution", {}).get("details", {})
            stop = details.get("contact_stop_reason")
            if (
                stop is not None
                or not result.passed
                or not result.terminal
                or not result.safe
            ):
                evidence["failure_unload"]["reason"] = (
                    stop or result.reason or "unknown unload terminal"
                )
                return False
            latest = await self._observe(
                admission.robot_id,
                admission.upper_physical_instance_id,
                admission.lower_physical_instance_id,
            )
            if latest.connection:
                return False
            rise = sum(
                (
                    latest.upper.tcp_pose.position_xyz[i]
                    - start.upper.tcp_pose.position_xyz[i]
                )
                * normal[i]
                for i in range(3)
            )
            support_p, support_a = pose_error(
                latest.lower.object_pose, admission.sample.lower.object_pose
            )
            held = compose(
                latest.upper.tcp_pose,
                inverse(_pose(guard["held_transform"], "held")),
            )
            slip_p, slip_a = pose_error(latest.upper.object_pose, held)
            passed = (
                rise >= 0.006
                and support_p <= guard["support_position_m"]
                and support_a <= guard["support_angle_rad"]
                and slip_p <= guard["grasp_position_m"]
                and slip_a <= guard["grasp_angle_rad"]
            )
            evidence["failure_unload"].update(
                passed=passed,
                tcp_rise_m=rise,
                support_position_m=support_p,
                support_angle_rad=support_a,
                grasp_position_m=slip_p,
                grasp_angle_rad=slip_a,
            )
            return passed
        except (ValueError, RuntimeError) as error:
            evidence["failure_unload"]["reason"] = str(error)
            return False

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        """Run PLACE_DOWN phases and commit placement only after all proofs.

        Returns:
            ``CONFIRMED`` only with connection, release and retreat evidence.
        """
        try:
            admission = await self.preflight(operation)
        except (ValueError, RuntimeError) as error:
            return self._failed(
                operation,
                f"PLACE_DOWN_PREFLIGHT_REJECTED: {error}",
                {},
                terminal=True,
                motion_sent=False,
            )
        if self._context is None or self._motion is None:
            return self._failed(
                operation,
                "PLACE_DOWN_PLANNING_SCENE_READBACK_UNAVAILABLE",
                {
                    "target_identity_checked": {
                        "operation_id": admission.operation_id,
                        "robot_id": admission.robot_id,
                        "connection_id": admission.connection_id,
                        "upper_physical_instance_id": (
                            admission.upper_physical_instance_id
                        ),
                        "lower_physical_instance_id": (
                            admission.lower_physical_instance_id
                        ),
                        "session_id": admission.sample.upper.session_id,
                        "manifest_sha256": admission.sample.upper.manifest_sha256,
                        "sequence": admission.sample.upper.sequence,
                        "sim_time_s": admission.sample.upper.sim_time_s,
                    }
                },
                terminal=True,
                motion_sent=False,
            )
        motion_sent = False
        contact_entered = False
        evidence: dict[str, JSONValue] = {}
        context = self._context
        try:
            interfaces = admission.sample.interfaces
            if not isinstance(interfaces, Mapping):
                raise ValueError("placement interface geometry is unavailable")
            pairs = interfaces.get("pairs")
            pair_key = "|".join(
                sorted(
                    (
                        admission.upper_physical_instance_id,
                        admission.lower_physical_instance_id,
                    )
                )
            )
            interface = pairs.get(pair_key) if isinstance(pairs, Mapping) else None
            if not isinstance(interface, Mapping) or (
                interface.get("upper_physical_instance_id")
                != admission.upper_physical_instance_id
                or interface.get("lower_physical_instance_id")
                != admission.lower_physical_instance_id
            ):
                raise ValueError("placement interface pair identity mismatch")
            connection_pose = _pose(interface.get("lower_to_upper"), "connection pose")
            targets, desired_upper, held_transform = interface_place_targets(
                admission.sample.lower.object_pose,
                connection_pose,
                admission.sample.upper.object_pose,
                admission.sample.upper.tcp_pose,
                admission.sample.upper.opening_m,
                context.base,
                context.provenance,
                context.params,
                yaw_symmetries=tuple(
                    interface.get("allowed_upper_yaw_quarter_turns", (0,))
                ),
            )
            evidence["place_geometry"] = {
                "source": "stud_hole_interface_and_measured_grasp",
                "interface": dict(interface),
                "desired_upper_pose": asdict(desired_upper),
                "upper_to_tcp": asdict(held_transform),
                "lower_center_world_xy_m": [
                    (
                        float(admission.sample.lower.aabb_min[0])
                        + float(admission.sample.lower.aabb_max[0])
                    )
                    / 2.0,
                    (
                        float(admission.sample.lower.aabb_min[1])
                        + float(admission.sample.lower.aabb_max[1])
                    )
                    / 2.0,
                ],
                "lower_top_world_z_m": float(admission.sample.lower.aabb_max[2]),
                "upper_height_m": (
                    float(admission.sample.upper.aabb_max[2])
                    - float(admission.sample.upper.aabb_min[2])
                ),
                "contact_tcp_world_z_m": targets.contact_world_z_m,
                "press_m": context.params.place_press_m,
                "closing_yaw_deg": targets.closing_yaw_deg,
            }
            evidence.update(
                {
                    "target_identity_checked": {
                        "operation_id": admission.operation_id,
                        "robot_id": admission.robot_id,
                        "connection_id": admission.connection_id,
                        "upper_physical_instance_id": (
                            admission.upper_physical_instance_id
                        ),
                        "lower_physical_instance_id": (
                            admission.lower_physical_instance_id
                        ),
                        "session_id": admission.sample.upper.session_id,
                        "manifest_sha256": admission.sample.upper.manifest_sha256,
                        "sequence": admission.sample.upper.sequence,
                        "sim_time_s": admission.sample.upper.sim_time_s,
                    }
                }
            )
            planning = planning_scene_operation(
                self._world_payload(
                    admission,
                    context,
                    admission.sample,
                    include_upper=False,
                    scene_suffix="plan_place",
                ),
                60.0,
                f"{operation.operation_id}:plan_place",
            )
            result = await self._dispatch(operation, "plan_place", planning)
            evidence.update(result.evidence)
            if not result.passed:
                return self._failed(
                    operation,
                    result.reason or "plan_place failed",
                    evidence,
                    terminal=result.terminal,
                    motion_sent=result.motion_sent,
                )
            robot_id = admission.robot_id
            upper_id = admission.upper_physical_instance_id
            lower_id = admission.lower_physical_instance_id

            async def observe() -> PlaceObservation:
                return await self._observe(robot_id, upper_id, lower_id)

            async def align_interface(label: str, first: PlaceObservation) -> None:
                """Measure the stud-interface error and correct it laterally.

                The descent gate is the stud/hole engagement scale, NOT the
                coarse transport tolerance: a held brick further out laterally
                cannot seat. The correction shifts the measured TCP by the
                measured lateral offset in free space (the grasp is treated as
                rigid for this small move); the axial error is untouched — the
                guarded descent owns the approach axis. Two corrections are
                bounded; a residual offset fails closed before any contact.
                """
                corrections = evidence.setdefault("place_alignment_corrections", [])
                aligned = first
                for index in range(3):
                    normal_local = _quat_rotate_vector(
                        aligned.lower.object_pose.orientation_xyzw, (0.0, 0.0, 1.0)
                    )
                    nominal_upper = compose(aligned.lower.object_pose, connection_pose)
                    selected_local = compose(
                        inverse(
                            compose(admission.sample.lower.object_pose, connection_pose)
                        ),
                        desired_upper,
                    )
                    expected_upper = compose(nominal_upper, selected_local)
                    delta = tuple(
                        aligned.upper.object_pose.position_xyz[i]
                        - expected_upper.position_xyz[i]
                        for i in range(3)
                    )
                    axial = sum(delta[i] * normal_local[i] for i in range(3))
                    lateral_vec = tuple(
                        delta[i] - axial * normal_local[i] for i in range(3)
                    )
                    lateral = math.sqrt(sum(value * value for value in lateral_vec))
                    _, angle = pose_error(aligned.upper.object_pose, expected_upper)
                    evidence["place_alignment_measured"] = {
                        "lateral_error_m": lateral,
                        "axial_error_m": axial,
                        "orientation_error_rad": angle,
                        "expected_upper_pose": asdict(expected_upper),
                        "measured_upper_pose": asdict(aligned.upper.object_pose),
                    }
                    if angle > context.params.place_alignment_orientation_rad:
                        raise ValueError(
                            "held brick orientation is not aligned to the "
                            f"interface: {angle:.5f} rad"
                        )
                    if lateral <= context.params.place_interface_align_tolerance_m:
                        return
                    if index >= 2:
                        raise ValueError(
                            "held brick is not aligned to the interface: "
                            f"lateral={lateral:.5f} m exceeds the stud "
                            "engagement bound"
                        )
                    target_world = tuple(
                        aligned.upper.tcp_pose.position_xyz[i] - lateral_vec[i]
                        for i in range(3)
                    )
                    phase = f"{label}_align_correct_{index}"
                    corrected = await self._dispatch(
                        operation,
                        phase,
                        tcp_pose_operation(
                            replace(
                                targets.contact,
                                position=world_position_to_base(
                                    target_world, context.base
                                ),
                                # Free-space tolerance: the correction only
                                # has to deliver its measured lateral delta;
                                # uncompensated droop is mostly axial and the
                                # loop closes on re-measurement, not command.
                                position_tolerance_m=(
                                    context.params.position_tolerance_m
                                ),
                            ),
                            15.0,
                            f"{operation.operation_id}:{phase}",
                        ),
                    )
                    corrections.append(
                        {
                            "phase": phase,
                            "lateral_error_m": lateral,
                            "correction_m": list(lateral_vec),
                            "runtime": dict(corrected.evidence),
                        }
                    )
                    if not corrected.passed:
                        raise ValueError(
                            f"interface alignment correction failed: {corrected.reason}"
                        )
                    aligned = await observe()

            contact_entered = False
            connected = False
            normal = _quat_rotate_vector(
                admission.sample.lower.object_pose.orientation_xyzw, (0.0, 0.0, 1.0)
            )
            guard = {
                "robot_id": admission.robot_id,
                "session_id": self._session_id,
                "manifest_sha256": self._manifest_sha256,
                "upper_id": admission.upper_physical_instance_id,
                "lower_id": admission.lower_physical_instance_id,
                "support_pose": asdict(admission.sample.lower.object_pose),
                "held_transform": asdict(held_transform),
                "desired_upper": asdict(desired_upper),
                # The support ANGLE bound (3 degrees) is the deformation
                # guard that the historic 10-degree destruction violated; the
                # position bound allows the compliant engagement sweep the
                # studs need while studs ride hole rims before seating.
                "support_position_m": 0.002,
                "support_angle_rad": math.radians(3),
                "grasp_position_m": 0.003,
                "grasp_angle_rad": 0.15,
                "force_limit_n": 8.0,
                "mode": "probe" if self._probe_contact else "insert",
            }
            evidence["contact_guard"] = guard
            phases = (
                (
                    "transport",
                    tcp_pose_operation(
                        targets.preplace, 60.0, f"{operation.operation_id}:transport"
                    ),
                ),
                (
                    "approach_clearance",
                    tcp_pose_operation(
                        replace(
                            targets.contact,
                            position=world_position_to_base(
                                tuple(
                                    compose(desired_upper, held_transform).position_xyz[
                                        i
                                    ]
                                    + normal[i] * 0.006
                                    for i in range(3)
                                ),
                                context.base,
                            ),
                        ),
                        60.0,
                        f"{operation.operation_id}:approach_clearance",
                    ),
                ),
            )
            for phase, runtime_operation in phases:
                result = await self._dispatch(operation, phase, runtime_operation)
                evidence.update(result.evidence)
                motion_sent = motion_sent or result.motion_sent
                if not result.passed:
                    return self._failed(
                        operation,
                        result.reason or f"{phase} failed",
                        evidence,
                        terminal=result.terminal,
                        motion_sent=motion_sent,
                    )
                if phase == "transport":
                    # The held brick must still sit in its measured grasp, and
                    # must be aligned to the stud interface to stud-scale
                    # tolerance before any descent is authorized.
                    aligned = await observe()
                    predicted_upper = compose(
                        aligned.upper.tcp_pose, inverse(held_transform)
                    )
                    following_position, following_angle = pose_error(
                        aligned.upper.object_pose, predicted_upper
                    )
                    if (
                        following_position > context.params.hold_follow_tolerance_m
                        or following_angle
                        > context.params.holding_orientation_tolerance_rad
                    ):
                        raise ValueError(
                            "carried brick lost its measured grasp transform"
                        )
                    await align_interface("transport", aligned)
            contact_entered = True
            connection_samples = [await observe()]
            insertion_start = connection_samples[0].upper.tcp_pose.position_xyz
            measured: Mapping[str, object] | None = None
            seated_tcp = compose(desired_upper, held_transform).position_xyz
            # ONE continuous descent from clearance to a commanded bound
            # BELOW the seated depth. The live monitor inside the runtime
            # cancels the action the moment the exact connection forms or any
            # physical guard (support/slip/force) fires, so the deep bound
            # absorbs gravity droop and grip compliance WITHOUT gravity
            # feed-forward. No press retries, no replanning during contact,
            # no lateral correction under load.
            scene = await self._observations.observe(robot_id)
            reason, measured = inspect_contact(scene, guard)
            if reason == "connection_confirmed":
                connected = True
            elif reason is not None:
                raise ValueError(f"contact guard stopped insertion: {reason}")
            else:
                target_world = tuple(
                    seated_tcp[i] - normal[i] * context.params.place_press_m
                    for i in range(3)
                )
                travel = sum(
                    (insertion_start[i] - target_world[i]) * normal[i] for i in range(3)
                )
                if travel > 0.0125:
                    raise ValueError("bounded insertion travel exhausted")
                target = replace(
                    targets.contact,
                    position=world_position_to_base(target_world, context.base),
                    position_tolerance_m=0.0003,
                )
                result = await self._dispatch(
                    operation,
                    "insert_descent",
                    guarded_tcp_operation(
                        target, guard, 20.0, f"{operation.operation_id}:insert_descent"
                    ),
                )
                motion_sent = motion_sent or result.motion_sent
                evidence["insertion_descent"] = dict(result.evidence)
                if not result.terminal or not result.safe:
                    evidence["recovery_prohibited"] = "unknown action terminal"
                    raise RuntimeError("insertion action has unknown terminal safety")
                runtime = result.evidence.get("runtime", {})
                details = runtime.get("execution", {}).get("details", {})
                trace = details.get("contact_trace")
                if isinstance(trace, list) and trace and isinstance(trace[-1], Mapping):
                    measured = trace[-1]
                reason = details.get("contact_stop_reason")
                if reason == "connection_confirmed":
                    connected = True
                elif reason == "probe_contact_detected":
                    evidence["contact_probe"] = {
                        "stop_reason": reason,
                        "observation": measured,
                    }
                else:
                    # A non-connection stop can race the connection forming
                    # during the cancellation settle (live evidence: a
                    # support-deflection cancel at 0.97 N connected moments
                    # later). Re-inspect once before declaring failure; the
                    # guard order (support before connection) keeps a still-
                    # deflected support from being read as success.
                    await asyncio.sleep(self._poll_interval_s)
                    settle_scene = await self._observations.observe(robot_id)
                    settle_reason, settle_measured = inspect_contact(
                        settle_scene, guard
                    )
                    measured = settle_measured
                    if settle_reason == "connection_confirmed":
                        connected = True
                    else:
                        # Still no connection: withdraw once (outer failure
                        # path) and fail closed. No retries.
                        raise ValueError(
                            "insertion did not confirm the exact connection: "
                            f"{reason or 'descent bound reached without a verdict'}"
                        )
            if not connected:
                if "contact_probe" in evidence:
                    raise ValueError(
                        "contact probe stopped at the first interface contact; "
                        "controlled placement was not attempted"
                    )
                raise ValueError("bounded insertion did not confirm exact connection")
            connection_samples = [await observe()]
            for _ in range(2):
                await asyncio.sleep(self._poll_interval_s)
                connection_samples.append(await observe())
            evidence["exact_interface_confirmed"] = measured
            evidence["connection_confirmed"] = verify_place_bricks(
                connection_samples,
                position_tolerance_m=context.params.still_tolerance_m,
                orientation_tolerance_rad=(
                    context.params.follow_orientation_tolerance_rad
                ),
                require_connected=True,
            )
            release = gripper_operation(
                GripperTarget(
                    opening_m=targets.release_opening_m,
                    duration_s=context.params.gripper_duration_s,
                    goal_tolerance=context.params.gripper_goal_tolerance,
                    expected_opening_m=targets.release_expected_window_m,
                ),
                60.0,
                f"{operation.operation_id}:release",
            )
            result = await self._dispatch(operation, "release", release)
            evidence.update(result.evidence)
            motion_sent = motion_sent or result.motion_sent
            if not result.passed:
                return self._failed(
                    operation,
                    result.reason or "release failed",
                    evidence,
                    terminal=result.terminal,
                    motion_sent=motion_sent,
                )
            evidence["gripper_release_feedback_confirmed"] = self._release_feedback(
                result.evidence
            )
            release_samples = [await observe()]
            for _ in range(2):
                await asyncio.sleep(self._poll_interval_s)
                release_samples.append(await observe())
            for sample in release_samples:
                support_position, support_angle = pose_error(
                    sample.lower.object_pose, admission.sample.lower.object_pose
                )
                placed_position, placed_angle = pose_error(
                    sample.upper.object_pose, desired_upper
                )
                if (
                    max(support_position, placed_position)
                    > context.params.place_align_tolerance_m
                    or max(support_angle, placed_angle)
                    > context.params.follow_orientation_tolerance_rad
                ):
                    raise ValueError(
                        "released stack did not return to its interface target"
                    )
            evidence["release_confirmed"] = verify_place_bricks(
                release_samples,
                position_tolerance_m=context.params.still_tolerance_m,
                orientation_tolerance_rad=(
                    context.params.follow_orientation_tolerance_rad
                ),
                require_connected=True,
            )
            detach = detached_planning_scene_operation(
                {
                    "scene_id": f"{operation.operation_id}-detach",
                    "object_id": upper_id,
                    "link_name": f"{robot_id}_rocobrick_tcp",
                    "operation": "detach",
                },
                60.0,
                f"{operation.operation_id}:detach",
            )
            result = await self._dispatch(operation, "detach", detach)
            evidence.update(result.evidence)
            if not result.passed:
                return self._failed(
                    operation,
                    result.reason or "detach failed",
                    evidence,
                    terminal=result.terminal,
                    motion_sent=motion_sent,
                )
            placed_sample = await observe()
            world_update = planning_scene_operation(
                self._world_payload(
                    admission,
                    context,
                    placed_sample,
                    include_upper=True,
                    scene_suffix="placed",
                ),
                60.0,
                f"{operation.operation_id}:world_placed",
            )
            result = await self._dispatch(operation, "world_placed", world_update)
            evidence.update(result.evidence)
            if not result.passed:
                return self._failed(
                    operation,
                    result.reason or "world_placed failed",
                    evidence,
                    terminal=result.terminal,
                    motion_sent=motion_sent,
                )
            evidence["handoff_required"] = True
            evidence["retreat_deferred"] = {
                "reason": "next transport owns departure from the placement interface",
                "anchor": "last_release_observation",
            }
            if not self._ledger.revoke(
                robot_id, f"holding:{upper_id}", "PLACE_DOWN_CONFIRMED"
            ):
                raise ValueError("the confirmed holding fact vanished before release")
            self._ledger.confirm(
                robot_id,
                f"placed:{upper_id}",
                {
                    "connection_id": admission.connection_id,
                    "upper_physical_instance_id": upper_id,
                    "lower_physical_instance_id": lower_id,
                    "connection_confirmed": dict(evidence["connection_confirmed"]),
                    "release_confirmed": dict(evidence["release_confirmed"]),
                    "handoff_required": True,
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
                reason="PLACE_DOWN_CONFIRMED",
            )
        except (ValueError, RuntimeError) as error:
            unloaded = False
            if (
                contact_entered
                and "targets" in locals()
                and "guard" in locals()
                and "recovery_prohibited" not in evidence
            ):
                unloaded = await self._unload_failed_contact(
                    operation, admission, targets, guard, evidence
                )
            return self._failed(
                operation,
                f"PLACE_DOWN_FAILED: {error}",
                evidence,
                terminal=unloaded or not motion_sent,
                motion_sent=motion_sent,
            )
