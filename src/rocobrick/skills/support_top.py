"""SUPPORT_TOP: hold a TCP support pose above a fixed upper brick.

This skill is deliberately NOT PICK. The support robot does not close its
fingers and does not claim ownership of the upper brick. It moves its TCP to a
measured support pose and maintains a downward support role while PLACE_UP
inserts a separate lower brick from below. The ledger fact is
``supporting:<upper>`` rather than ``holding:<upper>``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

from rocobrick.coordinated_session import (
    FreeSpaceMotionEvidence,
    FreeSpaceMotionRequest,
)
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.isaac_backend import (
    RuntimeOperation,
    guarded_tcp_operation,
    planning_scene_operation,
)
from rocobrick.multi_robot_scheduler import (
    JSONValue,
    OperationExecution,
    OperationStatus,
)
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.observation import PickObservation, Pose, parse_pick_observation
from rocobrick.skills.placement_geometry import pose_error, support_top_target
from rocobrick.skills.successor_motion import contact_scene_payload, move_boundary
from rocobrick.skills.support_evidence import verify_support_contact
from rocobrick.transit_adapter import TransitContext
from rocobrick.unified_dual_arm_runtime import SuccessorPrePhaseEvidence


class SupportObservationTransport(Protocol):
    """Read one robot's live scene without sending motion."""

    async def observe(self, robot_id: str) -> Mapping[str, object]:
        """Return one measured scene for the named robot."""


class SupportFreeSpaceTransport(Protocol):
    """Dispatch one skill-neutral support approach."""

    async def execute_free_space(
        self, request: FreeSpaceMotionRequest
    ) -> FreeSpaceMotionEvidence:
        """Execute support approach targets."""


class SupportMotionTransport(Protocol):
    """Dispatch one robot-scoped runtime operation."""

    def run_motion_operation(
        self, robot_id: str, operation: RuntimeOperation
    ) -> Mapping[str, object]:
        """Run one runtime operation and return its strict record."""

    def cancel_motion_operation(self, operation_id: str) -> Mapping[str, object]:
        """Cancel one currently active runtime request."""


@dataclass(frozen=True)
class SupportAdmission:
    """One fixed upper brick and its first measured support sample."""

    operation_id: str
    robot_id: str
    placement_robot_id: str
    connection_id: str
    upper_physical_instance_id: str
    lower_physical_instance_id: str
    sample: Pose
    sample_aabb_min: tuple[float, float, float]
    sample_aabb_max: tuple[float, float, float]


class SupportTopHandler:
    """Maintain a non-grasping downward support pose for one upper brick."""

    def __init__(
        self,
        observation_transport: SupportObservationTransport,
        ledger: OperationStateLedger,
        *,
        session_id: str,
        manifest_sha256: str,
        motion_transport: SupportMotionTransport | None = None,
        free_space_transport: SupportFreeSpaceTransport | None = None,
        context: TransitContext | None = None,
        poll_interval_s: float = 0.2,
    ) -> None:
        """Bind identity, observation, motion and calibrated TCP context."""
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
        """Consume only a validated receipt belonging to this successor."""
        if (
            not isinstance(evidence, SuccessorPrePhaseEvidence)
            or not evidence.passed
            or operation_id not in evidence.successor_operation_ids
        ):
            raise ValueError("support pre-phase receipt is not bound")
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
        """Execute only the free-space approach, never support contact.

        Returns:
            Bound physical free-space completion evidence.
        """
        if self._free_space is None:
            raise RuntimeError("SUPPORT_TOP free-space transport is unavailable")
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

    async def preflight(self, operation: RobotOperation) -> SupportAdmission:
        """Validate the support pair and admit one fresh upper pose sample.

        Returns:
            The support admission with the upper brick's measured pose.
        """
        if operation.skill != "SUPPORT_TOP" or operation.target is None:
            raise ValueError("SUPPORT_TOP target is required")
        target = operation.target.values
        expected = {
            "connection_id",
            "upper_physical_instance_id",
            "lower_physical_instance_id",
            "placement_robot_id",
        }
        if set(target) != expected:
            raise ValueError("SUPPORT_TOP target shape is invalid")
        upper = target["upper_physical_instance_id"]
        lower = target["lower_physical_instance_id"]
        placement_robot = target["placement_robot_id"]
        if not all(
            isinstance(value, str) and value
            for value in (target["connection_id"], upper, lower, placement_robot)
        ):
            raise ValueError("SUPPORT_TOP target values must be nonempty strings")
        if upper == lower or placement_robot == operation.executor_robot_id:
            raise ValueError("SUPPORT_TOP robot roles or pair are invalid")
        sample = parse_pick_observation(
            await self._observations.observe(operation.executor_robot_id),
            robot_id=operation.executor_robot_id,
            physical_instance_id=upper,
            session_id=self._session_id,
            manifest_sha256=self._manifest_sha256,
        )
        return SupportAdmission(
            operation.operation_id,
            operation.executor_robot_id,
            placement_robot,
            target["connection_id"],
            upper,
            lower,
            sample.object_pose,
            sample.aabb_min,
            sample.aabb_max,
        )

    async def _dispatch(
        self, operation: RobotOperation, phase: str, runtime_operation: RuntimeOperation
    ) -> Mapping[str, JSONValue]:
        """Dispatch one support pose and return strict runtime evidence.

        Returns:
            The runtime phase's independently observed lifecycle proof.
        """
        if operation.operation_id in self._cancelled_operations:
            raise RuntimeError("operation cancelled before phase")
        if self._motion is None:
            raise RuntimeError("support motion transport unavailable")
        self._active_requests[operation.operation_id] = runtime_operation.request_id
        try:
            record = await asyncio.to_thread(
                self._motion.run_motion_operation,
                operation.executor_robot_id,
                runtime_operation,
            )
        except Exception as error:
            # An unanswered motion request is not evidence that it stopped.
            return {
                "phase": phase,
                "request_id": runtime_operation.request_id,
                "motion_sent": True,
                "runtime": {
                    "passed": False,
                    "execution": {"terminal": False, "safe_to_release": False},
                    "error": f"support dispatch failed: {error}",
                },
            }
        finally:
            self._active_requests.pop(operation.operation_id, None)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"{phase}: invalid motion transport record")
        return {
            "phase": phase,
            "request_id": runtime_operation.request_id,
            "runtime": dict(record.get("result") or {}),
            "motion_sent": record.get("motion_sent") is True,
        }

    async def cancel(self, operation_id: str) -> Mapping[str, object]:
        """Cancel the active support pose operation.

        Returns:
            Actual cancellation evidence from the active execution boundary.
        """
        self._cancelled_operations.add(operation_id)
        motion_id = self._active_boundaries.get(operation_id)
        if motion_id is not None:
            return await self._free_space.cancel_free_space(motion_id)
        request_id = self._active_requests.get(operation_id)
        if request_id is None or self._motion is None:
            return {"attempted": False, "terminal": False, "safe": False}
        return await asyncio.to_thread(self._motion.cancel_motion_operation, request_id)

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        """Move to support pose and confirm stable non-grasping support.

        Returns:
            ``CONFIRMED`` only after one support pose action and three
            advancing upper-brick observations prove the upper remains stable.
        """
        motion_sent = False
        terminal_safe = True
        evidence: dict[str, JSONValue] = {}
        try:
            admission = await self.preflight(operation)
            if self._context is None or self._motion is None:
                raise ValueError("support context or motion transport unavailable")
            base = self._context.base
            params = self._context.params
            if self._free_space is None:
                raise ValueError("support combined transport unavailable")
            sample = parse_pick_observation(
                await self._observations.observe(admission.robot_id),
                robot_id=admission.robot_id,
                physical_instance_id=admission.upper_physical_instance_id,
                session_id=self._session_id,
                manifest_sha256=self._manifest_sha256,
            )
            receipt = self._pre_phases.pop(operation.operation_id, None)
            if receipt is None:
                approach = support_top_target(
                    sample, base, self._context.provenance, params, approach=True
                )
                result = await move_boundary(
                    self._free_space,
                    operation,
                    self._context,
                    approach,
                    phase="support_approach",
                    active_requests=self._active_boundaries,
                )
                evidence["free_space_approach"] = dict(result.evidence)
                motion_sent = result.started
                terminal_safe = result.terminal and result.safe_to_release
                if not result.passed:
                    raise ValueError(result.reason or "support approach failed")
            else:
                evidence["successor_pre_phase"] = dict(receipt.evidence)
            target = support_top_target(
                sample, base, self._context.provenance, params, approach=False
            )
            contact_scene, _, _ = await self._free_space.free_space_context(
                (admission.robot_id, admission.placement_robot_id)
            )
            contact_pairs = [
                (
                    admission.upper_physical_instance_id,
                    f"{admission.robot_id}_gripper_link{index}",
                )
                for index in (1, 2)
            ]
            scene_record = await self._dispatch(
                operation,
                "support_contact_scene",
                planning_scene_operation(
                    contact_scene_payload(
                        contact_scene,
                        scene_id=f"{operation.operation_id}-contact",
                        attached_ids=(admission.lower_physical_instance_id,),
                        allowed_pairs=contact_pairs,
                    ),
                    60.0,
                    f"{operation.operation_id}:support_contact_scene",
                ),
            )
            evidence["support_contact_scene"] = scene_record
            if scene_record["runtime"].get("passed") is not True:
                raise ValueError("support contact scene readback failed")
            guard = {
                "robot_id": admission.robot_id,
                "session_id": self._session_id,
                "manifest_sha256": self._manifest_sha256,
                "upper_id": admission.upper_physical_instance_id,
                "supported_pose": asdict(admission.sample),
                "supported_position_m": 0.002,
                "supported_angle_rad": 0.05236,
                "force_limit_n": 8.0,
                "mode": "support_top",
            }
            record = await self._dispatch(
                operation,
                "support_contact",
                guarded_tcp_operation(
                    target, guard, 60.0, f"{operation.operation_id}:support_contact"
                ),
            )
            evidence["support_pose"] = record
            motion_sent = motion_sent or record["motion_sent"] is True
            execution = record["runtime"].get("execution") or {}
            terminal_safe = (
                execution.get("terminal") is True
                and execution.get("safe_to_release") is True
            )
            if not _passed(record):
                raise ValueError("support contact motion failed terminal gates")
            stop = (execution.get("details") or {}).get("contact_stop_reason")
            if stop not in (None, "support_contact_confirmed"):
                raise ValueError(f"support contact stopped: {stop}")
            samples: list[PickObservation] = []
            contacts = []
            for index in range(3):
                if index:
                    await asyncio.sleep(self._poll_interval_s)
                observed, proof = verify_support_contact(
                    await self._observations.observe(admission.robot_id),
                    robot_id=admission.robot_id,
                    upper_id=admission.upper_physical_instance_id,
                    session_id=self._session_id,
                    manifest_sha256=self._manifest_sha256,
                )
                if samples and (
                    observed.sequence <= samples[-1].sequence
                    or observed.sim_time_s <= samples[-1].sim_time_s
                ):
                    raise ValueError("support samples did not advance")
                samples.append(observed)
                contacts.append(proof)
            evidence["support_pose_confirmed"] = {
                "confirmed": True,
                "samples": contacts,
            }
            max_position = max(
                pose_error(item.object_pose, admission.sample)[0] for item in samples
            )
            max_angle = max(
                pose_error(item.object_pose, admission.sample)[1] for item in samples
            )
            if (
                max_position > params.place_align_tolerance_m
                or max_angle > params.follow_orientation_tolerance_rad
            ):
                raise ValueError(
                    "supported upper brick moved during support verification"
                )
            evidence["support_stability_confirmed"] = {
                "samples": len(samples),
                "first_sequence": samples[0].sequence,
                "last_sequence": samples[-1].sequence,
                "max_position_error_m": max_position,
                "max_orientation_error_rad": max_angle,
            }
            evidence["handoff_required"] = True
            evidence["terminal_safe"] = True
            evidence["support_hold"] = {
                "mode": "maintain_until_next_transport_handoff",
                "upper_physical_instance_id": admission.upper_physical_instance_id,
                "connection_id": admission.connection_id,
            }
            self._ledger.confirm(
                admission.robot_id,
                f"supporting:{admission.upper_physical_instance_id}",
                {
                    "connection_id": admission.connection_id,
                    "placement_robot_id": admission.placement_robot_id,
                    "upper_physical_instance_id": admission.upper_physical_instance_id,
                    "lower_physical_instance_id": admission.lower_physical_instance_id,
                    "mode": "non_grasping_tcp_support",
                    "support_contact_confirmed": contacts[-1],
                    "upper_pose": asdict(samples[-1].object_pose),
                    "tcp_pose": asdict(samples[-1].tcp_pose),
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
                reason="SUPPORT_TOP_CONFIRMED",
            )
        except (ValueError, RuntimeError) as error:
            return _failed(
                operation,
                f"SUPPORT_TOP_FAILED: {error}",
                evidence,
                terminal=terminal_safe,
                motion_sent=motion_sent,
            )


def _passed(record: Mapping[str, JSONValue]) -> bool:
    """Check a phase record's strict terminal safety.

    Returns:
        Whether execution passed and physically terminated safely.
    """
    runtime = record.get("runtime")
    execution = runtime.get("execution") if isinstance(runtime, Mapping) else None
    return bool(
        isinstance(runtime, Mapping)
        and runtime.get("passed") is True
        and isinstance(execution, Mapping)
        and execution.get("terminal") is True
        and execution.get("safe_to_release") is True
    )


def _failed(
    operation: RobotOperation,
    reason: str,
    evidence: Mapping[str, JSONValue],
    *,
    terminal: bool,
    motion_sent: bool,
) -> OperationExecution:
    """Build one fail-closed support result.

    Returns:
        Failure evidence without inventing support or stopped motion.
    """
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
