"""Exact PICK-lift continuation admission, without runtime or motion."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_handover_joint_planning import planning_request
from test_robot_operation_plan_and_scheduler import handover_operation

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.robot_operation_plan import OperationTarget, RobotOperation
from rocobrick.skills.post_lift_handover import capture_post_lift_handover


def continuation():
    """Build a piper_0-to-piper_1 operation and completed synthetic lift proof.

    Returns:
        Operation, scene source, ledger and explicit planning dependencies.
    """
    source = planning_request()
    raw = handover_operation("after-original-pick-lift", "piper_0", "piper_1")
    raw["parameters"]["grasp_profile_id"] = source.profile.profile_id
    raw["target"] = OperationTarget(raw["target"])
    for key in (
        "participants",
        "preconditions",
        "effects",
        "resources",
        "evidence_requirements",
    ):
        raw[key] = tuple(raw[key])
    operation = RobotOperation(**raw)
    ledger = OperationStateLedger(source.session_robots)
    proof = {
        "robot_id": "piper_0",
        "physical_instance_id": "brick-1",
        "session_id": source.scene_snapshot["session_id"],
        "manifest_sha256": source.scene_snapshot["manifest_sha256"],
        "last_sim_time_s": 1.0,
        "samples": 4,
        "lift_m": 0.08,
    }
    ledger.confirm_holding(
        "piper_0",
        "brick-1",
        {
            "lift_confirmed": True,
            "gripper_feedback_confirmed": True,
            "holding_confirmed": proof,
        },
    )
    reads = []

    def observe():
        reads.append("post-lift")
        return {
            "scene_snapshot": source.scene_snapshot,
            "planning_scene_version": source.scene_version,
            "start_state": source.start_state,
            "observed_sim_time_s": source.observed_sim_time_s,
        }

    arguments = {
        "ledger": ledger,
        "session": SimpleNamespace(observe_shared_scene=observe),
        "contexts": {c.robot_id: c for c in (source.donor, source.receiver)},
        "object_geometry": source.object_geometry,
        "profile": source.profile,
    }
    return operation, arguments, reads


def test_original_pick_proof_and_lift_state_are_retained():
    """A complete measured lift precedes scene capture and bottom planning."""
    operation, arguments, reads = continuation()
    request = capture_post_lift_handover(operation, **arguments)
    assert reads == ["post-lift"]
    assert request.donor.robot_id == "piper_0"
    assert request.receiver.robot_id == "piper_1"
    assert (
        request.giver_holding_evidence["pick_evidence"]
        == arguments["ledger"].observe("piper_0", "holding:brick-1").evidence
    )
    assert request.operation_id == operation.operation_id


@pytest.mark.parametrize(
    "missing", ["lift", "gripper", "following", "identity", "samples", "displacement"]
)
def test_incomplete_pick_cannot_trigger_post_lift_capture(missing):
    """Planned effects and partial grasp evidence do not authorize continuation."""
    operation, arguments, reads = continuation()
    ledger = arguments["ledger"]
    proof = dict(ledger.observe("piper_0", "holding:brick-1").evidence)
    holding = dict(proof["holding_confirmed"])
    if missing == "lift":
        proof["lift_confirmed"] = False
    elif missing == "gripper":
        proof["gripper_feedback_confirmed"] = False
    elif missing == "following":
        proof["holding_confirmed"] = True
    elif missing == "identity":
        holding["robot_id"] = "piper_1"
    elif missing == "samples":
        holding["samples"] = 1
    else:
        holding["lift_m"] = 0.0
    if missing != "following":
        proof["holding_confirmed"] = holding
    ledger.confirm_holding("piper_0", "brick-1", proof)
    with pytest.raises(ValueError):
        capture_post_lift_handover(operation, **arguments)
    assert reads == []


def test_holding_effect_without_ledger_cannot_start_handover():
    """External operation effects cannot replace a completed original PICK."""
    operation, arguments, reads = continuation()
    arguments["ledger"] = OperationStateLedger(("piper_0", "piper_1"))
    operation = replace(operation, preconditions=("holding:brick-1",))
    with pytest.raises(ValueError, match="exclusive"):
        capture_post_lift_handover(operation, **arguments)
    assert reads == []


@pytest.mark.parametrize("change", ["session_id", "manifest_sha256", "sim_time_s"])
def test_post_lift_scene_must_match_completed_pick(change):
    """Earlier or foreign snapshots cannot define the HANDOVER starting grasp."""
    operation, arguments, _ = continuation()
    import copy

    shared = arguments["session"].observe_shared_scene()
    scene = copy.deepcopy(shared["scene_snapshot"])
    scene[change] = 0.1 if change == "sim_time_s" else "foreign"
    arguments["session"] = SimpleNamespace(
        observe_shared_scene=lambda: {**shared, "scene_snapshot": scene}
    )
    with pytest.raises(ValueError, match="predates"):
        capture_post_lift_handover(operation, **arguments)
