"""Physical HANDOVER orchestration and atomic owner adoption."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from rocobrick.coordinated_session import (
    FreeSpaceMotionEvidence,
    FreeSpaceMotionRequest,
)
from rocobrick.execution.operation_state_ledger import (
    ConfirmedState,
    OperationStateLedger,
)
from rocobrick.gripper import logical_opening_to_pad_separation
from rocobrick.isaac_backend import (
    OperationKind,
    RuntimeOperation,
    attached_planning_scene_operation,
    detached_planning_scene_operation,
    gripper_operation,
    gripper_target_from_payload,
    transfer_attached_planning_scene_operation,
)
from rocobrick.multi_robot_scheduler import (
    JSONValue,
    OperationExecution,
    OperationStatus,
    is_json_safe,
)
from rocobrick.planning_scene import box_geometry_digest
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.contracts import HandoverEvidence, HandoverRequest
from rocobrick.skills.observation import PickObservation, parse_pick_observation
from rocobrick.skills.pick import (
    FINGER_TIP_BELOW_TCP_M,
    FLOOR_CONTACT_MARGIN_M,
    PICK_CONTACT_SURFACE_CLEARANCE_M,
    _tcp_relative_aabb,
)
from rocobrick.skills.targets import PickTargets, RobotBaseFrame, world_position_to_base


def receiver_precontact_target(
    robot_id: str,
    targets: PickTargets,
    *,
    aabb_min: tuple[float, float, float],
    aabb_max: tuple[float, float, float],
    base: RobotBaseFrame,
    ground_top_z_m: float | None = None,
) -> Mapping[str, JSONValue]:
    """Bind the measured object boundary to the receiver's namespaced model.

    Returns:
        A base-local TCP target, not the receiver's current idle pose.
    """
    pose = targets.precontact_boundary
    center = tuple((low + high) / 2.0 for low, high in zip(aabb_min, aabb_max))
    contact_z = center[2] + PICK_CONTACT_SURFACE_CLEARANCE_M
    if ground_top_z_m is not None:
        contact_z = max(
            contact_z,
            ground_top_z_m + FINGER_TIP_BELOW_TCP_M + FLOOR_CONTACT_MARGIN_M,
        )
    # Use the same calibrated pad-midpoint clearance as the successful PICK
    # path, not the generic 15 mm standoff where pads cannot reach the brick.
    position = world_position_to_base((center[0], center[1], contact_z), base)
    return {
        "tcp_pose": {
            "position": list(position),
            "orientation_xyzw": list(pose.orientation_xyzw),
            "frame_id": f"{robot_id}_base_link",
            "tip_link": f"{robot_id}_rocobrick_tcp",
            "runtime_model_sha256": pose.runtime_model_sha256,
            "source_sha256": pose.source_sha256,
            "position_tolerance_m": pose.position_tolerance_m,
            "orientation_tolerance_rad": pose.orientation_tolerance_rad,
        }
    }


def require_receiver_contact_geometry(
    sample: PickObservation, *, tolerance_m: float
) -> None:
    """Reject distant/static objects as co-holding proof before donor release.

    Geometry is only a necessary condition: gripper feedback and subsequent
    follow evidence remain mandatory. The calibrated TCP is the finger-pad
    contact midpoint, so the object must overlap it in the two pad-plane axes.

    Raises:
        ValueError: If the receiver's pads cannot be touching this object.
    """
    minimum, maximum = _tcp_relative_aabb(sample)
    half_opening = logical_opening_to_pad_separation(sample.opening_m) / 2.0
    if any(
        minimum[axis] > tolerance_m or maximum[axis] < -tolerance_m
        for axis in (0, 2)
    ) or (
        minimum[1] > -half_opening + tolerance_m
        or maximum[1] < half_opening - tolerance_m
    ):
        raise ValueError("HANDOVER receiver contact geometry is not confirmed")


def receiver_attachment_from_sample(
    request: HandoverRequest, snapshot: Mapping[str, object]
) -> Mapping[str, JSONValue]:
    """Bind transfer geometry to the observed receiver after its approach.

    Returns:
        Receiver-relative measured geometry with a recomputed digest.
    """
    sample = parse_pick_observation(
        snapshot,
        robot_id=request.receiver_robot_id,
        physical_instance_id=request.physical_instance_id,
        session_id=request.scene_snapshot.get("session_id"),
        manifest_sha256=request.scene_snapshot.get("manifest_sha256"),
    )
    minimum, maximum = _tcp_relative_aabb(sample)
    frame_id = f"{request.receiver_robot_id}_rocobrick_tcp"
    return {
        **request.receiver_attach,
        "frame_id": frame_id,
        "link_name": frame_id,
        "minimum_m": list(minimum),
        "maximum_m": list(maximum),
        "geometry_digest": box_geometry_digest(
            object_id=request.physical_instance_id,
            frame_id=frame_id,
            minimum_m=minimum,
            maximum_m=maximum,
        ),
    }


class HandoverCommitTransport(Protocol):
    """Ledger boundary used by the physical HANDOVER executor."""

    def transfer_holding(
        self, request: HandoverRequest, evidence: HandoverEvidence
    ) -> ConfirmedState:
        """Commit a proven donor-to-receiver holding transfer."""


class HandoverFreeSpaceTransport(Protocol):
    """Unified combined-session free-space execution boundary."""

    async def execute_free_space(
        self, request: FreeSpaceMotionRequest
    ) -> FreeSpaceMotionEvidence:
        """Execute one receiver-only target in the complete session."""


class HandoverMotionTransport(Protocol):
    """Existing per-robot gripper and planning-scene operation boundary."""

    def run_motion_operation(
        self, robot_id: str, operation: RuntimeOperation
    ) -> Mapping[str, object]:
        """Execute one validated runtime operation."""


class HandoverObservationTransport(Protocol):
    """Co-sampled donor/receiver/object evidence boundary."""

    async def observe(self, request: HandoverRequest) -> Mapping[str, object]:
        """Return one identified advancing HANDOVER observation."""

    def verify_co_hold(
        self, request: HandoverRequest, samples: tuple[Mapping[str, object], ...]
    ) -> Mapping[str, JSONValue]:
        """Prove donor and receiver simultaneously hold the object."""

    def verify_receiver_holding(
        self, request: HandoverRequest, samples: tuple[Mapping[str, object], ...]
    ) -> Mapping[str, JSONValue]:
        """Prove the receiver follows the object after donor release."""


@dataclass(frozen=True)
class HandoverCommit:
    """Result of one atomic ledger adoption."""

    evidence: HandoverEvidence
    receiver_state: ConfirmedState


class HandoverOperationBackend:
    """Adapt one physical HANDOVER handler to the scheduler result contract."""

    def __init__(
        self,
        handler: "HandoverHandler",
        request_factory: Callable[[RobotOperation], HandoverRequest],
    ) -> None:
        """Bind the physical handler and a live request builder."""
        self._handler = handler
        self._request_factory = request_factory

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        """Build, execute and atomically commit one HANDOVER operation.

        Returns:
            A strict scheduler result retaining failed physical evidence.
        """
        try:
            request = self._request_factory(operation)
            evidence = await self._handler.execute(request)
            if not evidence.ready_to_commit:
                return self._result(operation, evidence, committed=False)
            committed = self._handler.commit(request, evidence)
            return self._result(operation, committed.evidence, committed=True)
        except Exception as error:  # noqa: BLE001 - operation must fail closed
            return OperationExecution(
                operation_id=operation.operation_id,
                robot_id=operation.executor_robot_id,
                status=OperationStatus.FAILED,
                passed=False,
                terminal=False,
                motion_sent=True,
                evidence={},
                reason=f"HANDOVER_BACKEND_FAILED: {error}",
            )

    @staticmethod
    def _result(
        operation: RobotOperation,
        evidence: HandoverEvidence,
        *,
        committed: bool,
    ) -> OperationExecution:
        """Convert physical evidence into the strict scheduler result.

        Returns:
            A committed transfer or an explicit failed operation result.
        """
        details = dict(evidence.evidence)
        details.update(
            {
                "receiver_free_space": evidence.receiver_free_space,
                "receiver_grasp_feedback": evidence.receiver_grasp_feedback,
                "co_hold_confirmed": evidence.co_hold_confirmed,
                "scene_transfer_committed": evidence.scene_transfer_committed,
                "donor_release_feedback": evidence.donor_release_feedback,
                "receiver_holding_follow": evidence.receiver_holding_follow,
                "terminal": evidence.terminal,
                "safe_to_release": evidence.safe_to_release,
                "owner_transfer_committed": committed,
            }
        )
        free = evidence.evidence.get("receiver_free_space")
        lifecycle = free.get("lifecycle") if isinstance(free, Mapping) else None
        no_dispatch = (
            isinstance(lifecycle, Mapping)
            and lifecycle.get("dispatch_attempted") is False
            and lifecycle.get("physical_execution") is False
            and lifecycle.get("terminal") is True
            and lifecycle.get("safe_to_release") is True
            and "receiver_grasp_feedback" not in evidence.evidence
        )
        return OperationExecution(
            operation_id=operation.operation_id,
            robot_id=operation.executor_robot_id,
            status=(
                OperationStatus.CONFIRMED
                if committed
                else OperationStatus.FAILED
            ),
            passed=committed,
            terminal=evidence.terminal,
            motion_sent=not no_dispatch,
            evidence=details,
            reason=evidence.reason,
        )


class HandoverHandler:
    """Execute HANDOVER phases and commit ownership only after physical proof."""

    def __init__(
        self,
        ledger: HandoverCommitTransport | OperationStateLedger,
        *,
        free_space: HandoverFreeSpaceTransport | None = None,
        motion: HandoverMotionTransport | None = None,
        observations: HandoverObservationTransport | None = None,
        poll_interval_s: float = 0.2,
    ) -> None:
        """Bind ledger and optional physical transports."""
        self._ledger = ledger
        self._free_space = free_space
        self._motion = motion
        self._observations = observations
        self._poll_interval_s = float(poll_interval_s)

    def commit(
        self, request: HandoverRequest, evidence: HandoverEvidence
    ) -> HandoverCommit:
        """Atomically adopt receiver ownership and return committed evidence.

        Returns:
            The committed evidence and receiver holding fact.
        """
        if not evidence.matches(request):
            raise ValueError("handover evidence does not match request")
        if not evidence.ready_to_commit:
            raise ValueError("handover physical evidence is incomplete")
        receiver_state = self._ledger.transfer_holding(request, evidence)
        return HandoverCommit(evidence.committed(), receiver_state)

    async def execute(self, request: HandoverRequest) -> HandoverEvidence:
        """Run physical HANDOVER phases without silently committing ownership.

        Returns:
            Complete physical evidence, with owner transfer still uncommitted.
        """
        if (
            self._free_space is None
            or self._motion is None
            or self._observations is None
        ):
            return self._failed(request, "HANDOVER physical transports unavailable")
        key = f"holding:{request.physical_instance_id}"
        if (
            self._ledger.owner_of_object(request.physical_instance_id)
            != request.donor_robot_id
            or self._ledger.observe(request.donor_robot_id, key) is None
        ):
            return self._failed(request, "HANDOVER donor holding is not confirmed")
        evidence: dict[str, JSONValue] = {}
        release_attempted = False
        transfer_attempted = False
        try:
            free = await self._execute_free_space(request)
            evidence["receiver_free_space"] = dict(free.evidence)
            if not free.passed:
                return self._failed(
                    request,
                    free.reason or "receiver free-space approach failed",
                    evidence,
                    terminal=free.terminal,
                    safe_to_release=free.safe_to_release,
                )
            receiver_grasp = await self._dispatch_gripper(
                request.receiver_robot_id,
                request.receiver_gripper,
                request,
                "receiver_grasp",
            )
            evidence["receiver_grasp_feedback"] = receiver_grasp
            samples = await self._samples(request)
            co_hold = self._observations.verify_co_hold(request, samples)
            self._require_proof(co_hold, "co-hold")
            evidence["co_hold_confirmed"] = dict(co_hold)
            transfer_attempted = True
            evidence["scene_transfer_attempted"] = True
            scene_transfer = await self._dispatch_transfer(request, samples[-1])
            evidence["scene_transfer_committed"] = scene_transfer
            release_attempted = True
            evidence["donor_release_attempted"] = True
            donor_release = await self._dispatch_gripper(
                request.donor_robot_id,
                request.donor_release_gripper,
                request,
                "donor_release",
            )
            evidence["donor_release_feedback"] = donor_release
            follow_samples = await self._samples(request)
            receiver_follow = self._observations.verify_receiver_holding(
                request, follow_samples
            )
            self._require_proof(receiver_follow, "receiver holding follow")
            evidence["receiver_holding_follow"] = dict(receiver_follow)
            return HandoverEvidence(
                operation_id=request.operation_id,
                donor_robot_id=request.donor_robot_id,
                receiver_robot_id=request.receiver_robot_id,
                physical_instance_id=request.physical_instance_id,
                receiver_free_space=True,
                receiver_grasp_feedback=True,
                co_hold_confirmed=True,
                scene_transfer_committed=True,
                donor_release_feedback=True,
                receiver_holding_follow=True,
                terminal=True,
                safe_to_release=True,
                owner_transfer_committed=False,
                evidence=evidence,
            )
        except Exception as error:  # noqa: BLE001 - fail closed after any phase
            if release_attempted or transfer_attempted:
                self._ledger.mark_handover_uncertain(
                    request,
                    f"HANDOVER partial transfer: {error}",
                    evidence,
                )
            return self._failed(
                request,
                f"HANDOVER_FAILED: {error}",
                evidence,
                terminal=False,
                safe_to_release=False,
            )

    async def _execute_free_space(
        self, request: HandoverRequest
    ) -> FreeSpaceMotionEvidence:
        motion = FreeSpaceMotionRequest(
            motion_id=f"{request.operation_id}:receiver_free_space",
            session_robots=request.session_robots,
            participants=(request.receiver_robot_id,),
            scene_snapshot=request.scene_snapshot,
            scene_version=request.scene_version,
            targets={request.receiver_robot_id: request.receiver_target},
            start_state=request.start_state,
            timeout_s=request.timeout_s,
            physical_facts={},
        )
        return await self._free_space.execute_free_space(motion)

    async def _samples(
        self, request: HandoverRequest
    ) -> tuple[Mapping[str, object], ...]:
        samples = [await self._observations.observe(request)]
        for _ in range(request.follow_samples - 1):
            await asyncio.sleep(self._poll_interval_s)
            samples.append(await self._observations.observe(request))
        if any(
            not isinstance(sample, Mapping) or not is_json_safe(sample)
            for sample in samples
        ):
            raise ValueError("HANDOVER observation samples are not JSON-safe")
        return tuple(samples)

    async def _dispatch_gripper(
        self,
        robot_id: str,
        payload: Mapping[str, JSONValue],
        request: HandoverRequest,
        phase: str,
    ) -> Mapping[str, JSONValue]:
        target = gripper_target_from_payload(payload)
        operation = gripper_operation(
            target,
            request.timeout_s,
            f"{request.operation_id}:{phase}",
        )
        return await self._dispatch(robot_id, operation, phase)

    async def _dispatch_transfer(
        self, request: HandoverRequest, snapshot: Mapping[str, object]
    ) -> Mapping[str, JSONValue]:
        """Dispatch one explicit session-level attached-scene transfer.

        Returns:
            Committed scene-transfer evidence bound to the live geometry.
        """
        transfer_id = f"{request.operation_id}:planning-scene-transfer"
        attach = receiver_attachment_from_sample(request, snapshot)
        donor_detach = request.donor_detach
        payload = {
            "transfer_id": transfer_id,
            "object_id": request.physical_instance_id,
            "donor_robot_id": request.donor_robot_id,
            "donor_link_name": str(donor_detach["link_name"]),
            "receiver_robot_id": request.receiver_robot_id,
            "receiver_link_name": str(attach["link_name"]),
            "frame_id": str(attach["frame_id"]),
            "minimum_m": attach["minimum_m"],
            "maximum_m": attach["maximum_m"],
            "geometry_digest": attach["geometry_digest"],
            "touch_links": attach["touch_links"],
            "expected_before": {
                "donor_attached": True,
                "receiver_attached": False,
            },
        }
        operation = transfer_attached_planning_scene_operation(
            payload, request.timeout_s, transfer_id
        )
        record = await asyncio.to_thread(
            self._motion.run_motion_operation,
            request.receiver_robot_id,
            operation,
        )
        raw = record.get("result") if isinstance(record, Mapping) else None
        execution = raw.get("execution") if isinstance(raw, Mapping) else None
        if (
            not isinstance(raw, Mapping)
            or raw.get("passed") is not True
            or not isinstance(execution, Mapping)
            or execution.get("transfer_status") != "COMMITTED"
            or execution.get("terminal") is not True
            or execution.get("safe_to_release") is not True
        ):
            raise ValueError(
                f"scene transfer did not commit: {raw!r}"
            )
        return {
            "transfer_id": transfer_id,
            "runtime": dict(raw),
            "transfer_status": "COMMITTED",
        }

    async def _dispatch_scene(
        self,
        robot_id: str,
        payload: Mapping[str, JSONValue],
        request: HandoverRequest,
        phase: str,
    ) -> Mapping[str, JSONValue]:
        operation = (
            attached_planning_scene_operation
            if payload.get("operation") != "detach"
            else detached_planning_scene_operation
        )(payload, request.timeout_s, f"{request.operation_id}:{phase}")
        return await self._dispatch(robot_id, operation, phase)

    async def _dispatch(
        self, robot_id: str, operation: RuntimeOperation, phase: str
    ) -> Mapping[str, JSONValue]:
        record = await asyncio.to_thread(
            self._motion.run_motion_operation, robot_id, operation
        )
        raw = record.get("result") if isinstance(record, Mapping) else None
        if not isinstance(raw, Mapping) or raw.get("passed") is not True:
            raise ValueError(f"{phase} runtime evidence failed")
        execution = raw.get("execution")
        if operation.kind is OperationKind.GRIPPER_TARGET:
            if not isinstance(execution, Mapping) or not all(
                execution.get(name) is True
                for name in ("terminal", "safe_to_release", "goal_accepted")
            ):
                raise ValueError(f"{phase} gripper terminal evidence failed")
        return {"phase": phase, "runtime": dict(raw)}

    @staticmethod
    def _require_proof(proof: Mapping[str, JSONValue], label: str) -> None:
        if not isinstance(proof, Mapping) or proof.get("confirmed") is not True:
            raise ValueError(f"{label} evidence is not confirmed")

    @staticmethod
    def _failed(
        request: HandoverRequest,
        reason: str,
        evidence: Mapping[str, JSONValue] | None = None,
        *,
        terminal: bool = True,
        safe_to_release: bool = False,
    ) -> HandoverEvidence:
        """Build failure evidence without claiming any physical transfer.

        Returns:
            Strict failure evidence bound to the request.
        """
        return HandoverEvidence(
            operation_id=request.operation_id,
            donor_robot_id=request.donor_robot_id,
            receiver_robot_id=request.receiver_robot_id,
            physical_instance_id=request.physical_instance_id,
            receiver_free_space=False,
            receiver_grasp_feedback=False,
            co_hold_confirmed=False,
            scene_transfer_committed=False,
            donor_release_feedback=False,
            receiver_holding_follow=False,
            terminal=terminal,
            safe_to_release=safe_to_release,
            owner_transfer_committed=False,
            evidence=dict(evidence or {}),
            reason=reason,
        )


__all__ = [
    "HandoverCommit",
    "HandoverCommitTransport",
    "HandoverFreeSpaceTransport",
    "HandoverHandler",
    "HandoverMotionTransport",
    "HandoverObservationTransport",
    "HandoverOperationBackend",
]
