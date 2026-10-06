import asyncio
"""Tests for strict physical transition identity and admission."""

from dataclasses import replace

import pytest
from pre_phase_helpers import request, snapshot, success

from rocobrick.isaac_backend import gripper_operation
from rocobrick.moveit_contract import GripperTarget
from rocobrick.planning_scene import box_geometry_digest
from rocobrick.unified_dual_arm_runtime import SharedSceneSnapshot


def test_pre_phase_evidence_requires_every_lifecycle_gate():
    """Planning, preparation, start and arrival are independent proofs."""
    item = request()
    evidence = success(item)
    assert evidence.passed and evidence.matches(item)
    for gate in (
        "planned",
        "prepared",
        "started",
        "arrival_confirmed",
        "terminal",
        "safe",
        "adoption_ready",
    ):
        assert not replace(evidence, **{gate: False}).passed
        with pytest.raises(ValueError):
            replace(evidence, **{gate: 1})


@pytest.mark.parametrize(
    "field,value",
    [
        ("transition_id", "foreign"),
        ("successor_operation_ids", ("foreign",)),
        ("participants", ("piper_1",)),
        ("scene_digest", "b" * 64),
        ("scene_version", 2),
    ],
)
def test_wrong_bindings_never_match(field, value):
    """A truthy result cannot replace the original scene or operation binding."""
    item = request()
    values = {field: value}
    if field == "participants":
        values["final_facts"] = {"piper_1": {}}
    assert not replace(success(item), **values).matches(item)


def test_shared_snapshot_rejects_staleness_and_unknown_physical_state():
    """Admission never assumes empty/home from absent evidence."""
    item = snapshot()
    with pytest.raises(ValueError, match="stale"):
        replace(item, observed_sim_time_s=2.0)
    for flag in ("authorized", "terminal_safe", "holding_known", "supporting_known"):
        document = {
            **item.document,
            "robots": {"piper_0": {**item.document["robots"]["piper_0"], flag: False}},
        }
        with pytest.raises(ValueError, match="unknown"):
            SharedSceneSnapshot(document, 1.1, 0.5)


def _handover_request():
    from rocobrick.skills.contracts import HandoverRequest

    receiver_gripper = gripper_operation(
        GripperTarget(0.01, 1.0, 0.001, (0.0, 0.02)),
        30.0,
        "receiver-grasp",
    ).payload
    donor_release = gripper_operation(
        GripperTarget(0.08, 1.0, 0.001, (0.07, 0.1)),
        30.0,
        "donor-release",
    ).payload
    minimum = (-0.01, -0.01, -0.01)
    maximum = (0.01, 0.01, 0.01)
    digest = box_geometry_digest(
        object_id="brick-1",
        frame_id="piper_1_rocobrick_tcp",
        minimum_m=minimum,
        maximum_m=maximum,
    )
    return HandoverRequest(
        operation_id="handover-1",
        donor_robot_id="piper_0",
        receiver_robot_id="piper_1",
        physical_instance_id="brick-1",
        session_robots=("piper_0", "piper_1"),
        scene_snapshot={"sequence": 3, "scene_digest": "a" * 64},
        scene_version=2,
        start_state={"piper_0_joint1": 0.0, "piper_1_joint1": 0.0},
        receiver_target={
            "tcp_pose": {
                "position": [0.0, 0.0, 0.1],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "frame_id": "piper_1_base_link",
                "tip_link": "piper_1_rocobrick_tcp",
                "runtime_model_sha256": "a" * 64,
                "source_sha256": "b" * 64,
                "position_tolerance_m": 0.002,
                "orientation_tolerance_rad": 0.1,
            }
        },
        receiver_gripper=receiver_gripper,
        donor_release_gripper=donor_release,
        receiver_attach={
            "scene_id": "handover-1-receiver-attach",
            "object_id": "brick-1",
            "link_name": "piper_1_rocobrick_tcp",
            "frame_id": "piper_1_rocobrick_tcp",
            "minimum_m": list(minimum),
            "maximum_m": list(maximum),
            "geometry_digest": digest,
            "touch_links": ["piper_1_gripper_link1", "piper_1_gripper_link2"],
        },
        donor_detach={
            "scene_id": "handover-1-donor-detach",
            "object_id": "brick-1",
            "link_name": "piper_0_rocobrick_tcp",
            "operation": "detach",
        },
        timeout_s=30.0,
    )


def _handover_evidence(*, committed=False, **overrides):
    from rocobrick.skills.contracts import HandoverEvidence

    values = {
        "operation_id": "handover-1",
        "donor_robot_id": "piper_0",
        "receiver_robot_id": "piper_1",
        "physical_instance_id": "brick-1",
        "receiver_free_space": True,
        "receiver_grasp_feedback": True,
        "co_hold_confirmed": True,
        "scene_transfer_committed": True,
        "donor_release_feedback": True,
        "receiver_holding_follow": True,
        "terminal": True,
        "safe_to_release": True,
        "owner_transfer_committed": committed,
        "evidence": {"samples": 3},
    }
    values.update(overrides)
    return HandoverEvidence(**values)


def test_handover_contract_requires_distinct_robots_and_real_booleans():
    """Reject same-robot transfers and truthy non-boolean gates."""
    with pytest.raises(ValueError, match="different robots"):
        replace(_handover_request(), receiver_robot_id="piper_0")
    with pytest.raises(ValueError, match="real bool"):
        replace(_handover_evidence(), co_hold_confirmed=1)
    assert _handover_evidence().ready_to_commit
    assert not _handover_evidence().passed


def test_handover_commit_atomically_moves_holding_owner():
    """Move the confirmed owner only after every physical gate passes."""
    from rocobrick.execution.operation_state_ledger import OperationStateLedger
    from rocobrick.skills.handover import HandoverHandler

    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm_holding("piper_0", "brick-1", {"holding_confirmed": True})
    result = HandoverHandler(ledger).commit(_handover_request(), _handover_evidence())

    assert result.evidence.passed
    assert ledger.owner_of_object("brick-1") == "piper_1"
    assert ledger.observe("piper_0", "holding:brick-1") is None
    assert ledger.observe("piper_1", "holding:brick-1") == result.receiver_state


def test_handover_partial_evidence_preserves_donor_owner():
    """Keep the donor lease when release evidence is incomplete."""
    from rocobrick.execution.operation_state_ledger import OperationStateLedger
    from rocobrick.skills.handover import HandoverHandler

    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm_holding("piper_0", "brick-1", {"holding_confirmed": True})
    with pytest.raises(ValueError, match="incomplete"):
        HandoverHandler(ledger).commit(
            _handover_request(),
            _handover_evidence(donor_release_feedback=False),
        )

    assert ledger.owner_of_object("brick-1") == "piper_0"
    assert ledger.observe("piper_0", "holding:brick-1") is not None
    assert ledger.observe("piper_1", "holding:brick-1") is None


def test_handover_unknown_state_blocks_both_robots_until_reconciled():
    """An unknown post-release transfer cannot be guessed or reused."""
    from rocobrick.execution.operation_state_ledger import OperationStateLedger

    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm_holding("piper_0", "brick-1", {"holding_confirmed": True})
    record = ledger.mark_handover_uncertain(
        _handover_request(), "receiver follow missing", {"scene": "unknown"}
    )

    assert record.physical_instance_id == "brick-1"
    assert ledger.owner_of_object("brick-1") == "piper_0"
    assert ledger.uncertain_handover("brick-1") == record
    assert ledger.observe("piper_0", "holding:brick-1") is None
    assert ledger.observe("piper_1", "holding:brick-1") is None
    assert "holding:brick-1" in ledger.uncertain_keys("piper_0")
    assert "holding:brick-1" in ledger.uncertain_keys("piper_1")

    ledger.reconcile_handover(
        "brick-1",
        owner_robot_id="piper_1",
        evidence={"reconciled": True, "measured": True},
    )
    assert ledger.owner_of_object("brick-1") == "piper_1"
    assert ledger.observe("piper_1", "holding:brick-1") is not None
    assert ledger.uncertain_handover("brick-1") is None


def test_handover_operation_backend_adapts_and_commits_scheduler_result():
    """The formal HANDOVER backend exposes committed physical evidence."""
    from types import SimpleNamespace

    from rocobrick.robot_operation_plan import OperationTarget, RobotOperation
    from rocobrick.skills.handover import HandoverOperationBackend

    request = _handover_request()
    operation = RobotOperation(
        operation_id=request.operation_id,
        source_task_id="task-1",
        skill="HANDOVER",
        executor_robot_id=request.receiver_robot_id,
        participants=request.participants,
        role_bindings={
            "donor": request.donor_robot_id,
            "receiver": request.receiver_robot_id,
        },
        target=OperationTarget(
            {
                "physical_instance_id": request.physical_instance_id,
                "donor_robot_id": request.donor_robot_id,
                "receiver_robot_id": request.receiver_robot_id,
            }
        ),
        parameters={"co_hold_timeout_s": 10.0, "follow_samples": 3},
        preconditions=(),
        effects=("holding_transferred",),
        resources=(
            f"robot:{request.donor_robot_id}",
            f"robot:{request.receiver_robot_id}",
            f"part:{request.physical_instance_id}",
        ),
        evidence_requirements=(),
    )

    class FakeHandler:
        async def execute(self, received):
            assert received is request
            return _handover_evidence()

        def commit(self, received, evidence):
            assert received is request
            assert evidence.ready_to_commit
            return SimpleNamespace(evidence=_handover_evidence(committed=True))

    result = asyncio.run(
        HandoverOperationBackend(
            FakeHandler(), lambda received: request
        ).execute(operation)
    )

    assert result.status.value == "CONFIRMED"
    assert result.passed is True
    assert result.terminal is True
    assert result.motion_sent is True
    assert result.evidence["scene_transfer_committed"] is True
    assert result.evidence["owner_transfer_committed"] is True
