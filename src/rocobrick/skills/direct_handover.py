"""One persistent paired HANDOVER preparation path.

The preparation motion starts directly from the measured post-lift state. It
moves both arms on one combined time axis: the giver carries and reorients the
brick while the receiver approaches. Contact and custody transfer are handled
by a later operation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace

from rocobrick.adapters.moveit.handover_live import LiveSimulationHandover
from rocobrick.coordinated_moveit_adapter import (
    coordinated_operation_from_joint_trajectory,
)
from rocobrick.coordinated_session import FreeSpaceMotionRequest
from rocobrick.multi_robot_scheduler import OperationExecution, OperationStatus
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.coordinated_handover import CoordinatedHandoverPreparation
from rocobrick.skills.handover_planning import HandoverPlanningRequest
from rocobrick.skills.handover_region import preparation_region
from rocobrick.skills.handover_zone import zone_from_request


def _joint_targets_from_trajectory(
    trajectory: Mapping[str, object],
    participants: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    names = trajectory.get("joint_names")
    points = trajectory.get("points")
    if (
        not isinstance(names, Sequence)
        or isinstance(names, (str, bytes))
        or not isinstance(points, Sequence)
        or isinstance(points, (str, bytes))
        or not points
    ):
        raise ValueError("combined plan has no joint trajectory")
    endpoint = points[-1]
    if not isinstance(endpoint, Mapping):
        raise ValueError("combined plan endpoint is malformed")
    positions = endpoint.get("positions")
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
        raise ValueError("combined plan endpoint has no positions")
    joints = dict(zip(names, positions, strict=True))
    return {
        robot: {
            "joint_positions": {
                f"{robot}_joint{i}": joints[f"{robot}_joint{i}"]
                for i in range(1, 7)
            }
        }
        for robot in participants
    }


def _started_and_physical(lifecycle: Mapping[str, object]) -> bool:
    return bool(
        lifecycle.get("started") is True
        and (
            lifecycle.get("physical_execution") is True
            or lifecycle.get("dispatch_attempted") is True
        )
    )


def _arrivals_ok(lifecycle: Mapping[str, object]) -> bool:
    participants = lifecycle.get("participants")
    if not isinstance(participants, Mapping):
        return True
    seen = []
    for evidence in participants.values():
        if not isinstance(evidence, Mapping):
            continue
        if "arrival_confirmed" in evidence:
            seen.append(evidence.get("arrival_confirmed") is True)
    return all(seen) if seen else True


def _readiness_error(error: object) -> bool:
    return "combined_joint_state_complete" in str(error or "")


class DirectHandoverBackend:
    """Execute one paired preparation and stop at the measured arrival."""

    def __init__(self, owner: "DirectSimulationHandover"):
        """Bind the live handover owner and its cancellation ledger."""
        self.owner = owner
        self._motion_ids: dict[str, str] = {}

    @staticmethod
    def _participants(request: HandoverPlanningRequest) -> tuple[str, ...]:
        """Use the session's canonical robot order for worker identity checks.

        Returns:
            The donor and receiver in canonical session order.
        """
        selected = {request.donor.robot_id, request.receiver.robot_id}
        return tuple(robot for robot in request.session_robots if robot in selected)

    async def _plan_candidate(
        self,
        operation: RobotOperation,
        request: HandoverPlanningRequest,
        targets: Mapping[str, Mapping[str, object]],
        motion_id: str,
    ):
        participants = self._participants(request)
        motion = FreeSpaceMotionRequest(
            motion_id=motion_id,
            session_robots=request.session_robots,
            participants=participants,
            scene_snapshot=request.scene_snapshot,
            scene_version=request.scene_version,
            targets=targets,
            start_state=request.start_state,
            timeout_s=8.0,
            physical_facts={
                request.donor.robot_id: {
                    "role": "giver",
                    "holding": request.physical_instance_id,
                },
                request.receiver.robot_id: {"role": "receiver"},
            },
        )
        plan = await asyncio.to_thread(
            self.owner.session.execute_successor_pre_phase,
            motion.to_coordinated_request(plan_only=True),
        )
        return motion, plan

    async def _execute_validated(
        self,
        operation: RobotOperation,
        request: HandoverPlanningRequest,
        motion: FreeSpaceMotionRequest,
        plan: object,
        motion_id: str,
    ):
        participants = self._participants(request)
        live = await asyncio.to_thread(self.owner.capture, operation)
        executable = replace(
            motion,
            motion_id=motion_id,
            targets=_joint_targets_from_trajectory(plan.trajectory, participants),
            scene_snapshot=live.scene_snapshot,
            scene_version=live.scene_version,
            start_state=live.start_state,
            timeout_s=30.0,
        )
        self._motion_ids[request.operation_id] = motion_id
        result = await asyncio.to_thread(
            self.owner.session.execute_coordinated_pre_phase,
            executable.to_coordinated_request(plan_only=False),
        )
        return result, executable, live

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        """Plan and execute the paired preparation without contact actions.

        Returns:
            Strict operation evidence for the preparation-only boundary.
        """
        evidence: dict[str, object] = {
            "path": "paired_handover_preparation_only",
            "completion_scope": "preparation_only",
            "handover_complete": False,
            "receiver_holding_follow": "deferred_to_observation",
            "donor_release_feedback": "deferred_to_observation",
            "owner_transfer_committed": "deferred_to_observation",
        }
        motion_sent = False
        try:
            plan = getattr(self.owner.session, "execute_successor_pre_phase", None)
            execute = getattr(self.owner.session, "execute_coordinated_pre_phase", None)
            if not callable(plan) or not callable(execute):
                raise RuntimeError(
                    "persistent combined planning/execution is unavailable"
                )

            initial = await asyncio.to_thread(self.owner.capture, operation)
            prep_zone = zone_from_request(initial)
            # Preparation is a continuous coupled region.  Let the combined
            # MoveIt sampler choose both arm joint configurations instead of
            # solving two exact Cartesian poses independently with the
            # measured lift state as a seed.  The giver remains a moving
            # participant: its object-relative grasp is carried by the
            # region metadata, while the receiver gets a broad precontact
            # neighborhood and unconstrained preparation orientation.
            initial_region = preparation_region(initial)
            prep_attempts: list[dict[str, object]] = []
            evidence.update(
                preparation_region=prep_zone.evidence(),
                preparation_target=initial_region["giver_pose"],
                preparation_candidate_count=1,
                preparation_attempts=prep_attempts,
                initial_start_state=dict(initial.start_state),
            )
            prep_selected = None
            prep_motion = None
            prep_request = initial
            for retry in range(3):
                request = (
                    initial
                    if retry == 0
                    else await asyncio.to_thread(self.owner.capture, operation)
                )
                retry_region = preparation_region(request)
                prep_targets = {
                    request.donor.robot_id: {"tcp_region": dict(retry_region)},
                    request.receiver.robot_id: {"tcp_region": dict(retry_region)},
                }
                motion, planned = await self._plan_candidate(
                    operation,
                    request,
                    prep_targets,
                    f"{operation.operation_id}:handover-preparation:region-{retry}",
                )
                prep_attempts.append(
                    {
                        "mode": "tcp_region",
                        "retry": retry,
                        "region": dict(retry_region),
                        "passed": planned.passed,
                        "error": planned.error,
                    }
                )
                if planned.passed:
                    prep_selected = planned
                    prep_motion = motion
                    prep_request = request
                    evidence["preparation_selected_mode"] = "tcp_region"
                    break
                # MoveIt can reject one combined query transiently while the
                # persistent scene/controller lane is settling. Reuse the
                # existing bounded attempts for every recorded planning
                # failure; the final error remains visible in evidence.
                if retry < 2:
                    await asyncio.sleep(0.05)
            if prep_selected is None or prep_motion is None:
                raise ValueError("no paired preparation plan succeeded")

            prep_result, _, executed_request = await self._execute_validated(
                operation,
                prep_request,
                prep_motion,
                prep_selected,
                f"{operation.operation_id}:handover-preparation",
            )
            prep_lifecycle = dict(prep_result.lifecycle)
            prep_motion_sent = _started_and_physical(prep_lifecycle)
            motion_sent = motion_sent or prep_motion_sent
            evidence["preparation"] = {
                "validated_plan": dict(prep_selected.trajectory),
                "trajectory": dict(prep_result.trajectory),
                "lifecycle": prep_lifecycle,
                "passed": bool(
                    prep_motion_sent
                    and _arrivals_ok(prep_lifecycle)
                    and prep_lifecycle.get("prepared") is True
                ),
            }
            if not (
                prep_motion_sent
                and _arrivals_ok(prep_lifecycle)
                and prep_lifecycle.get("prepared") is True
            ):
                raise ValueError(
                    f"paired preparation did not start/arrive: {prep_result.error}"
                )

            preparation = self._admit_executed_preparation(
                operation, executed_request, prep_result
            )
            formal = getattr(self.owner, "formal_backend", None)
            continue_after_preparation = getattr(
                formal, "execute_after_preparation", None
            )
            if not callable(continue_after_preparation):
                raise RuntimeError("formal HANDOVER continuation is unavailable")
            evidence["preparation_only"] = False
            evidence["continuation"] = "formal_from_measured_preparation_arrival"
            preparation_execution = OperationExecution(
                operation_id=operation.operation_id,
                robot_id=operation.executor_robot_id,
                status=OperationStatus.CONFIRMED,
                passed=True,
                terminal=True,
                motion_sent=motion_sent,
                evidence=evidence,
                reason=None,
            )
            return await continue_after_preparation(
                operation, preparation, preparation_execution
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

    def _admit_executed_preparation(
        self,
        operation: RobotOperation,
        request: HandoverPlanningRequest,
        planned: object,
    ) -> CoordinatedHandoverPreparation:
        """Bind the proven Direct MoveIt trajectory for formal continuation.

        Returns:
            A preparation contract whose trajectory has already been executed.
        """
        trajectory = getattr(planned, "trajectory", None)
        if not isinstance(trajectory, Mapping):
            raise ValueError("paired preparation has no combined trajectory")
        joint_names = trajectory.get("joint_names")
        points = trajectory.get("points")
        if not isinstance(joint_names, Sequence) or isinstance(
            joint_names, (str, bytes)
        ):
            raise ValueError("paired preparation trajectory has no joint binding")
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            raise ValueError("paired preparation trajectory has no time grid")
        participants = (request.donor.robot_id, request.receiver.robot_id)
        joint_binding = {
            robot: tuple(f"{robot}_joint{i}" for i in range(1, 7))
            for robot in participants
        }
        rendezvous = coordinated_operation_from_joint_trajectory(
            coordination_id=f"{operation.operation_id}:direct-preparation",
            scene_digest=request.scene_snapshot["scene_digest"],
            participants=participants,
            joint_names=joint_names,
            points=points,
            joint_binding=joint_binding,
            start_at_sim_time_s=0.0,
            synchronization_tolerance_s=request.profile.max_sync_skew_s,
        )
        start = dict(request.start_state)
        return CoordinatedHandoverPreparation(
            request=request,
            rendezvous=rendezvous,
            receiver_open=(start, dict(start)),
            capabilities=self.owner.factory.execution_capabilities,
            donor_holding_window=self.owner.donor_window_for(operation),
            phase_timeout_s=self.owner.factory.phase_timeout_s,
            simulation_authorization=self.owner.authorization,
        )

    async def cancel(self, operation_id: str) -> Mapping[str, object]:
        """Cancel the owned preparation motion and return terminal evidence.

        Returns:
            JSON-safe cancellation evidence for the owned motion.
        """
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
    """Reuse post-lift capture for one paired preparation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Create the live owner and its paired preparation backend."""
        super().__init__(*args, **kwargs)
        self.formal_backend = self.backend
        self.backend = DirectHandoverBackend(self)
