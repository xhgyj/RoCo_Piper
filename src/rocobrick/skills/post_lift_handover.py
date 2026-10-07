"""Capture HANDOVER planning input after the existing PICK confirms its lift.

This is an explicit continuation adapter, not a new PICK/LIFT implementation
or an operation-plan repair. Generic HANDOVER may have other holding sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.handover_planning import (
    HandoverPlanningRequest,
    HandoverRobotContext,
    ObjectGraspGeometry,
)
from rocobrick.skills.handover_profile import HandoverProfile, finite


class PostLiftSceneSource(Protocol):
    """Shared read-only observation from the already running session."""

    def observe_shared_scene(self) -> Mapping[str, object]:
        """Return fresh scene, measured full joints, version and current time."""
        ...


def capture_post_lift_handover(
    operation: RobotOperation,
    *,
    ledger: OperationStateLedger,
    session: PostLiftSceneSource,
    contexts: Mapping[str, HandoverRobotContext],
    object_geometry: ObjectGraspGeometry,
    profile: HandoverProfile,
) -> HandoverPlanningRequest:
    """Capture after confirmed PICK lift, never from effects or setup state.

    Returns:
        A planning-only request retaining the exact PICK evidence and live grasp.
    """
    target = operation.target.values if operation.target is not None else {}
    donor = target.get("donor_robot_id")
    receiver = target.get("receiver_robot_id")
    physical = target.get("physical_instance_id")
    if (
        operation.skill != "HANDOVER"
        or any(not isinstance(v, str) or not v for v in (donor, receiver, physical))
        or donor == receiver
    ):
        raise ValueError("post-lift continuation requires exact HANDOVER identities")
    if (
        operation.executor_robot_id != receiver
        or dict(operation.role_bindings)
        != {"executor": receiver, "donor": donor, "receiver": receiver}
        or set(operation.participants) != {donor, receiver}
    ):
        raise ValueError(
            "post-lift roles/participants differ from the external operation"
        )
    if operation.parameters.get("grasp_profile_id") != profile.profile_id:
        raise ValueError("post-lift operation profile differs from captured policy")
    if ledger.owner_of_object(physical) != donor:
        raise ValueError("PICK lift has not confirmed exclusive giver ownership")
    fact = ledger.observe(donor, f"holding:{physical}")
    if fact is None or ledger.uncertain_keys(donor) or ledger.uncertain_keys(receiver):
        raise ValueError("post-lift custody is missing or uncertain")
    proof = fact.evidence.get("holding_confirmed")
    if fact.evidence.get("lift_confirmed") is not True or (
        fact.evidence.get("gripper_feedback_confirmed") is not True
        or not isinstance(proof, Mapping)
        or proof.get("robot_id") != donor
        or proof.get("physical_instance_id") != physical
    ):
        raise ValueError(
            "post-lift continuation requires actual PICK lift/holding evidence"
        )
    finite(proof.get("last_sim_time_s"), "PICK final holding time")
    finite(proof.get("lift_m"), "PICK measured lift", positive=True)
    if type(proof.get("samples")) is not int or proof["samples"] < 3:
        raise ValueError("PICK holding proof lacks measured following samples")
    # Do not read/setup a scene until the completed lift is admitted.
    shared = session.observe_shared_scene()
    snapshot = shared.get("scene_snapshot")
    joints = shared.get("start_state")
    if not isinstance(snapshot, Mapping) or not isinstance(joints, Mapping):
        raise ValueError("post-lift shared scene/full measured joints unavailable")
    if snapshot.get("session_id") != proof.get("session_id") or (
        snapshot.get("manifest_sha256") != proof.get("manifest_sha256")
        or finite(snapshot.get("sim_time_s"), "post-lift sample time")
        < proof["last_sim_time_s"]
    ):
        raise ValueError("post-lift sample predates PICK or belongs to another session")
    holding = {
        "holding_confirmed": True,
        "robot_id": donor,
        "physical_instance_id": physical,
        "source": "confirmed_pick_lift_ledger",
        "pick_evidence": dict(fact.evidence),
    }
    return HandoverPlanningRequest(
        operation.operation_id,
        physical,
        ledger.robot_ids,
        contexts[donor],
        contexts[receiver],
        snapshot,
        shared.get("planning_scene_version"),
        joints,
        holding,
        object_geometry,
        profile,
        operation.parameters.get("follow_samples"),
        operation.parameters.get("co_hold_timeout_s"),
        shared.get("observed_sim_time_s"),
    )
