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
from rocobrick.isaac_backend import RuntimeOperation, tcp_pose_operation
from rocobrick.multi_robot_scheduler import (
    JSONValue,
    OperationExecution,
    OperationStatus,
)
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.observation import Pose, parse_pick_observation
from rocobrick.skills.placement_geometry import pose_error
from rocobrick.skills.targets import (
    _tcp_pose_target,
    world_orientation_to_base,
    world_position_to_base,
)
from rocobrick.transit_adapter import TransitContext


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
        """Execute only the free-space approach, never support contact."""
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
        """Dispatch one support pose and return strict runtime evidence."""
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
        """Cancel the active support pose operation."""
        self._cancelled_operations.add(operation_id)
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
        evidence: dict[str, JSONValue] = {}
        try:
            admission = await self.preflight(operation)
            if self._context is None or self._motion is None:
                raise ValueError("support context or motion transport unavailable")
            sample = admission.sample
            base = self._context.base
            params = self._context.params
            # Place the TCP just above the measured upper surface, then move
            # downward by support_press_m so PhysX supplies a bounded support
            # reaction. This is a pose-only action; gripper state is untouched.
            top_z = max(
                admission.sample_aabb_max[2], admission.sample.position_xyz[2]
            )
            world = (
                (admission.sample_aabb_min[0] + admission.sample_aabb_max[0]) / 2.0,
                (admission.sample_aabb_min[1] + admission.sample_aabb_max[1]) / 2.0,
                top_z - params.support_press_m,
            )
            target = _tcp_pose_target(
                world_position_to_base(world, base),
                world_orientation_to_base(sample.orientation_xyzw, base),
                self._context.provenance,
                params,
            )
            if self._free_space is not None:
                scene, scene_version, start_state = (
                    await self._free_space.free_space_context(
                        tuple(operation.participants)
                    )
                )
                free_target = {
                    operation.executor_robot_id: {
                        "tcp_pose": {
                            "position": tuple(target.position),
                            "orientation_xyzw": tuple(target.orientation_xyzw),
                            "frame_id": f"{operation.executor_robot_id}_base_link",
                            "tip_link": f"{operation.executor_robot_id}_rocobrick_tcp",
                            "runtime_model_sha256": target.runtime_model_sha256,
                            "source_sha256": target.source_sha256,
                            "position_tolerance_m": target.position_tolerance_m,
                            "orientation_tolerance_rad": target.orientation_tolerance_rad,
                        }
                    }
                }
                free_result = await self.execute_free_space_phase(
                    operation,
                    scene_snapshot=scene,
                    scene_version=scene_version,
                    targets=free_target,
                    start_state=start_state,
                )
                evidence["free_space_approach"] = dict(free_result.evidence)
                if not free_result.passed:
                    raise ValueError(
                        free_result.reason or "support free-space approach failed"
                    )
            record = await self._dispatch(
                operation,
                "support_pose",
                tcp_pose_operation(
                    target, 60.0, f"{operation.operation_id}:support_pose"
                ),
            )
            evidence["support_pose"] = record
            motion_sent = bool(record.get("motion_sent"))
            if not _passed(record):
                raise ValueError("support pose action did not pass terminal gates")
            samples = [
                parse_pick_observation(
                    await self._observations.observe(operation.executor_robot_id),
                    robot_id=operation.executor_robot_id,
                    physical_instance_id=admission.upper_physical_instance_id,
                    session_id=self._session_id,
                    manifest_sha256=self._manifest_sha256,
                )
            ]
            for _ in range(2):
                await asyncio.sleep(self._poll_interval_s)
                samples.append(
                    parse_pick_observation(
                        await self._observations.observe(operation.executor_robot_id),
                        robot_id=operation.executor_robot_id,
                        physical_instance_id=admission.upper_physical_instance_id,
                        session_id=self._session_id,
                        manifest_sha256=self._manifest_sha256,
                    )
                )
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
                terminal=not motion_sent,
                motion_sent=motion_sent,
            )


def _passed(record: Mapping[str, JSONValue]) -> bool:
    """Return whether a phase record proves passed terminal safety."""
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
    """Build one fail-closed support result."""
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
