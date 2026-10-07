"""Physical, no-motion lease continuation from the existing PICK/lift.

This is neither an invented TRANSIT nor a new operation. It refreshes the
giver's existing physical evidence while the scheduler still owns all leases.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Protocol

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.multi_robot_scheduler import to_json_value
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.handover_observation import (
    HandoverObservation,
    parse_handover_observation,
    require_giver_retention_sample,
    verify_giver_holding,
)
from rocobrick.skills.handover_profile import HandoverProfile, finite
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.unified_dual_arm_runtime import (
    SuccessorPrePhaseEvidence,
    SuccessorPrePhaseRequest,
    failed_pre_phase,
)


class StationaryHandoverSession(Protocol):
    """Idle read-only session boundary; no controller methods are exposed."""

    def observe_shared_scene(self) -> Mapping[str, object]:
        """Return current shared physics, planning version and full joint state."""
        ...


class StationaryHandoverPolicy(Protocol):
    """Captured policy and the original giver's calibrated grasp window."""

    def profile_for(self, operation: RobotOperation) -> HandoverProfile:
        """Resolve the exact external operation profile."""
        ...

    def donor_window_for(self, operation: RobotOperation) -> tuple[float, float]:
        """Return the giver's own captured calibration, never receiver defaults."""
        ...


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode(),
    ).hexdigest()


class StationaryHandoverContinuation:
    """Admit a stationary lease adoption only from fresh actual retained grasp."""

    def __init__(
        self,
        session: StationaryHandoverSession,
        ledger: OperationStateLedger,
        policy: StationaryHandoverPolicy,
    ) -> None:
        """Bind existing services without creating a runtime or motion command."""
        self.session, self.ledger, self.policy = session, ledger, policy
        self._issued: dict[str, tuple[str, str, Mapping[str, float]]] = {}

    def _context(self, request: SuccessorPrePhaseRequest):
        op = request.next_operation
        if op is None or op.skill != "HANDOVER" or request.terminal_cleanup:
            raise ValueError("stationary continuation requires external HANDOVER")
        target = op.target.values if op.target else {}
        donor, receiver, physical = (
            target.get(k)
            for k in ("donor_robot_id", "receiver_robot_id", "physical_instance_id")
        )
        if any(
            not isinstance(v, str) or not v for v in (donor, receiver, physical)
        ) or (
            donor == receiver
            or set(request.participants) != {donor, receiver}
            or op.executor_robot_id != receiver
            or set(op.participants) != {donor, receiver}
            or dict(op.role_bindings)
            != {"executor": receiver, "donor": donor, "receiver": receiver}
            or self.ledger.owner_of_object(physical) != donor
            or any(self.ledger.uncertain_keys(r) for r in (donor, receiver))
        ):
            raise ValueError("stationary continuation roles/custody do not match")
        key = f"holding:{physical}"
        fact = self.ledger.observe(donor, key)
        if fact is None or self.ledger.observe(receiver, key) is not None:
            raise ValueError(
                "stationary continuation requires exclusive confirmed giver"
            )
        robots = request.scene_snapshot.document["robots"]
        if robots[donor]["facts"].get(key) != dict(fact.evidence):
            raise ValueError("stationary snapshot differs from original PICK ledger")
        for previous in request.predecessor_operations:
            if previous.skill == "PICK" and previous.executor_robot_id == donor:
                proof = fact.evidence.get("holding_confirmed")
                if fact.evidence.get("lift_confirmed") is not True or (
                    fact.evidence.get("gripper_feedback_confirmed") is not True
                    or not isinstance(proof, Mapping)
                    or proof.get("robot_id") != donor
                    or proof.get("physical_instance_id") != physical
                    or proof.get("session_id")
                    != request.scene_snapshot.document["session_id"]
                    or proof.get("manifest_sha256")
                    != request.scene_snapshot.document["manifest_sha256"]
                ):
                    raise ValueError(
                        "PICK continuation lacks actual lift/holding proof"
                    )
                finite(proof.get("lift_m"), "measured original lift", positive=True)
                if (
                    finite(proof.get("last_sim_time_s"), "PICK final time")
                    > (request.scene_snapshot.document["sim_time_s"])
                ):
                    raise ValueError("stationary scene predates completed PICK")
        profile = self.policy.profile_for(op)
        if profile.profile_id != op.parameters.get("grasp_profile_id"):
            raise ValueError("stationary continuation changed external profile")
        window = self.policy.donor_window_for(op)
        admitted = parse_handover_observation(
            request.scene_snapshot.document,
            donor_id=donor,
            receiver_id=receiver,
            physical_instance_id=physical,
            session_id=request.scene_snapshot.document["session_id"],
            manifest_sha256=request.scene_snapshot.document["manifest_sha256"],
        )
        expected = compose(inverse(admitted.donor.object_pose), admitted.donor.tcp_pose)
        return op, profile, window, admitted, expected

    async def _read(self, request, profile, admitted, expected, window):
        shared = await asyncio.wait_for(
            asyncio.to_thread(self.session.observe_shared_scene),
            finite(
                request.next_operation.parameters.get("co_hold_timeout_s"),
                "stationary query timeout",
                positive=True,
            ),
        )
        snapshot, joints = shared.get("scene_snapshot"), shared.get("start_state")
        if (
            not isinstance(snapshot, Mapping)
            or not isinstance(joints, Mapping)
            or not joints
        ):
            raise ValueError("stationary continuation lacks complete physics/joints")
        now = finite(shared.get("observed_sim_time_s"), "current simulation clock")
        sample_time = finite(snapshot.get("sim_time_s"), "stationary observation time")
        if not 0 <= now - sample_time <= profile.max_scene_age_s or (
            shared.get("planning_scene_version") != request.scene_version
            or snapshot.get("environment_colliders")
            != request.scene_snapshot.document.get("environment_colliders")
        ):
            raise ValueError(
                "stationary continuation scene/version changed or is stale"
            )
        sample = parse_handover_observation(
            snapshot,
            donor_id=admitted.donor.robot_id,
            receiver_id=admitted.receiver.robot_id,
            physical_instance_id=admitted.donor.physical_instance_id,
            session_id=admitted.donor.session_id,
            manifest_sha256=admitted.donor.manifest_sha256,
        )
        require_giver_retention_sample(
            sample,
            profile=profile,
            donor_window=window,
            expected_object_to_donor_tcp=expected,
        )
        for role in ("donor", "receiver"):
            a, b = getattr(admitted, role), getattr(sample, role)
            error = pose_error(a.tcp_pose, b.tcp_pose)
            if (
                error[0] > profile.position_tolerance_m
                or error[1] > profile.orientation_tolerance_rad
            ):
                raise ValueError("stationary lease continuation observed robot motion")
        error = pose_error(admitted.donor.object_pose, sample.donor.object_pose)
        if (
            error[0] > profile.observation_noise_m
            or error[1] > profile.relative_orientation_tolerance_rad
        ):
            raise ValueError("stationary continuation object moved")
        complete = {n: finite(v, f"measured joint {n}") for n, v in joints.items()}
        for robot in request.participants:
            for i in range(1, 7):
                finite(
                    complete.get(f"{robot}_joint{i}"), "complete stationary arm joint"
                )
        return sample, complete

    async def admit_stationary(
        self,
        request: SuccessorPrePhaseRequest,
    ) -> SuccessorPrePhaseEvidence:
        """Refresh retained custody and authorize no-motion lease adoption.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        try:
            op, profile, window, admitted, expected = self._context(request)
            samples: list[HandoverObservation] = []
            initial: Mapping[str, float] | None = None
            deadline = asyncio.get_running_loop().time() + finite(
                op.parameters.get("co_hold_timeout_s"),
                "continuation timeout",
                positive=True,
            )
            count = op.parameters.get("follow_samples")
            if type(count) is not int or count < 3:
                raise ValueError("continuation requires exact follow sample count")
            previous = admitted.donor
            while len(samples) < count or (
                samples[-1].donor.sim_time_s - samples[0].donor.sim_time_s
                < profile.min_observation_span_s
            ):
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("stationary custody admission timed out")
                sample, joints = await self._read(
                    request, profile, admitted, expected, window
                )
                if sample.donor.sequence <= previous.sequence or not (
                    0
                    < sample.donor.sim_time_s - previous.sim_time_s
                    <= profile.max_observation_gap_s
                ):
                    raise ValueError(
                        "stationary custody samples repeated or discontinuous"
                    )
                initial = joints if initial is None else initial
                if set(joints) != set(initial) or any(
                    abs(joints[n] - initial[n]) > profile.joint_hold_tolerance_rad
                    for n in initial
                ):
                    raise ValueError("stationary custody measured full state drifted")
                samples.append(sample)
                previous = sample.donor
            proof = verify_giver_holding(
                samples,
                profile=profile,
                follow_samples=count,
                donor_window=window,
                expected_object_to_donor_tcp=expected,
            )
            # Preserve original PICK lift/holding proof and provenance.
            facts = {
                r: dict(request.scene_snapshot.document["robots"][r]["facts"])
                for r in request.participants
            }
            key = f"holding:{admitted.donor.physical_instance_id}"
            facts[admitted.donor.robot_id][key] = {
                **facts[admitted.donor.robot_id][key],
                "stationary_retention": proof,
            }
            evidence = SuccessorPrePhaseEvidence(
                request.transition_id,
                request.predecessor_operation_ids,
                request.successor_operation_ids,
                request.participants,
                request.resources,
                request.lease_owners,
                request.scene_digest,
                request.scene_version,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                to_json_value(facts),
                to_json_value(
                    {
                        "stationary_custody_confirmed": True,
                        "motion_sent": False,
                        "physical_instance_id": admitted.donor.physical_instance_id,
                        "donor_robot_id": admitted.donor.robot_id,
                        "receiver_robot_id": admitted.receiver.robot_id,
                        "retention_proof": proof,
                    }
                ),
                transition_mode="stationary_handover",
            )
            if len(self._issued) >= 128 or request.transition_id in self._issued:
                raise ValueError("stationary admission identity replay/cache limit")
            self._issued[request.transition_id] = (
                _sha(asdict(request)),
                _sha(asdict(evidence)),
                dict(initial),
            )
            return evidence
        except Exception as error:
            return failed_pre_phase(request, f"STATIONARY_HANDOVER_REJECTED: {error}")

    async def validate_stationary(
        self,
        request: SuccessorPrePhaseRequest,
        evidence: SuccessorPrePhaseEvidence,
    ) -> bool:
        """Recheck live retained custody before the scheduler adopts any lease.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        try:
            issued = self._issued.pop(request.transition_id, None)
            if (
                issued is None
                or not evidence.matches(request)
                or not evidence.passed
                or (issued[:2] != (_sha(asdict(request)), _sha(asdict(evidence))))
            ):
                return False
            _, profile, window, admitted, expected = self._context(request)
            sample, joints = await self._read(
                request, profile, admitted, expected, window
            )
            proof = evidence.evidence["retention_proof"]
            return (
                sample.donor.sequence > proof["last_sequence"]
                and 0
                < sample.donor.sim_time_s - proof["last_sim_time_s"]
                <= profile.max_observation_gap_s
                and set(joints) == set(issued[2])
                and all(
                    abs(joints[n] - issued[2][n]) <= profile.joint_hold_tolerance_rad
                    for n in joints
                )
            )
        except Exception:
            return False
