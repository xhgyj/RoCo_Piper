"""Formal HANDOVER safety regressions; no ROS/Isaac processes are started."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_coordinated_handover_lifecycle import fixture, operation
from test_coordinated_ros_runtime import pose, request
from test_handoff_contract import _handover_request
from test_handover_joint_planning import planning_request, snapshot
from test_persistent_ros_worker import ROBOTS, TOKEN, pump

from rocobrick.adapters.moveit.handover_live import LiveSimulationHandover
from rocobrick.coordinated_session_entry import perform_coordinated_plan
from rocobrick.execution.ros_session_worker import SCHEMA, SessionWorker, write_atomic
from rocobrick.planning_scene import box_geometry_digest
from rocobrick.skills.direct_handover import DirectHandoverBackend
from rocobrick.skills.handover_observation import (
    parse_handover_observation,
    verify_co_hold,
)
from rocobrick.skills.handover_planning import calibrated_target
from rocobrick.skills.placement_geometry import compose, inverse
from rocobrick.skills.targets import world_orientation_to_base, world_position_to_base


@pytest.mark.parametrize("robot", ROBOTS)
def test_receiver_target_is_measured_object_boundary_in_its_own_base(robot):
    """Retain frame/provenance binding, not the retired side-grasp coordinate."""
    item = planning_request()
    ctx = item.donor if item.donor.robot_id == robot else item.receiver
    world = item.observation(robot).tcp_pose
    target = calibrated_target(world, ctx, item.profile)["tcp_pose"]
    assert target["frame_id"] == f"{robot}_base_link"
    assert target["tip_link"] == f"{robot}_rocobrick_tcp"
    assert target["position"] == pytest.approx(
        world_position_to_base(world.position_xyz, ctx.base)
    )
    assert target["orientation_xyzw"] == pytest.approx(
        world_orientation_to_base(world.orientation_xyzw, ctx.base)
    )
    assert target["runtime_model_sha256"] == ctx.provenance.runtime_model_sha256
    assert target["source_sha256"] == ctx.provenance.source_sha256
    assert target["position_tolerance_m"] == item.profile.position_tolerance_m


@pytest.mark.parametrize("field,value", [
    ("frame_id", "base_link"), ("frame_id", "piper_0_base_link"),
    ("tip_link", "piper_0_rocobrick_tcp"),
])
def test_handover_contract_rejects_default_or_foreign_receiver_frame(field, value):
    """Foreign receiver frame is refused before any ROS request."""
    item = _handover_request()
    tcp = {**item.receiver_target["tcp_pose"], field: value}
    with pytest.raises(ValueError, match="frame/tip"):
        replace(item, receiver_target={"tcp_pose": tcp})


@pytest.mark.parametrize("axis", range(3))
def test_distant_stationary_object_cannot_prove_receiver_contact(axis):
    """Stationary relative pose/opening without actual pad contact proves nothing."""
    req = planning_request()
    samples = []
    for seq in range(2, 6):
        raw = snapshot(seq)
        raw["robots"][req.receiver.robot_id]["tcp_pose"]["position_xyz"][axis] += 0.3
        raw["pad_object_contacts"]["contacts"] = [
            c for c in raw["pad_object_contacts"]["contacts"]
            if c["robot_id"] != req.receiver.robot_id
        ]
        samples.append(parse_handover_observation(
            raw, donor_id=req.donor.robot_id, receiver_id=req.receiver.robot_id,
            physical_instance_id=req.physical_instance_id, session_id="session-1",
            manifest_sha256="c" * 64,
        ))
    expected = compose(inverse(samples[0].donor.object_pose), samples[0].donor.tcp_pose)
    with pytest.raises(ValueError, match="both identified pads"):
        verify_co_hold(samples, profile=req.profile, follow_samples=4,
                       donor_window=(0.015, 0.017), receiver_window=(0.015, 0.017),
                       expected_object_to_donor_tcp=expected)


def test_cohold_geometry_failure_never_transfers_scene_or_releases_donor():
    """Reject missing contact before scene transfer or donor release."""
    prepared, ledger, transport, backend = fixture(defect="no_receiver_contacts")
    result = asyncio.run(backend.execute(operation(prepared)))
    assert not result.passed
    assert "donor_open" not in transport.calls
    assert "scene_transfer" not in transport.calls
    assert ledger.owner_of_object("brick-1") == prepared.identity.donor_robot_id
    assert ledger.observe(
        prepared.identity.receiver_robot_id, "holding:brick-1"
    ) is None


def test_handover_no_dispatch_failure_does_not_claim_motion():
    """Refuse a foreign executor without dispatch, with terminal-safe evidence."""
    prepared, ledger, transport, backend = fixture()
    wrong = replace(
        operation(prepared), executor_robot_id=prepared.identity.donor_robot_id
    )
    result = asyncio.run(backend.execute(wrong))
    assert not result.motion_sent and not result.passed and result.terminal
    assert result.evidence["safe_to_release"] is True
    assert not transport.calls
    assert ledger.observe(
        prepared.identity.donor_robot_id, "holding:brick-1"
    ) is not None


@pytest.mark.parametrize("phase", ["scene_transfer", "donor_open"])
def test_attempted_transfer_failure_invalidates_confirmed_holder(phase):
    """Unknown scene/release completion invalidates both confirmed holding facts."""
    prepared, ledger, transport, backend = fixture(fail=phase)
    result = asyncio.run(backend.execute(operation(prepared)))
    assert not result.passed and result.motion_sent
    assert result.evidence["safe_to_release"] is False
    assert transport.cancelled
    assert ledger.uncertain_handover("brick-1") is not None
    assert ledger.observe("piper_0", "holding:brick-1") is None
    assert ledger.observe("piper_1", "holding:brick-1") is None


def _live_transfer_fixture():
    """Bind the real transfer method to an offline session boundary.

    Returns:
        The prepared program, bound owner and recorded dispatch calls.
    """
    prepared, _, _, _ = fixture()
    calls = []

    def dispatch(robot, op):
        calls.append((robot, op))
        return {"result": {"passed": True, "execution": {
            "transfer_status": "COMMITTED", "donor_after": [], "receiver_after": [{}],
            "terminal": True, "safe_to_release": True,
        }}}

    owner = SimpleNamespace(
        authorization=SimpleNamespace(session_id="session-1", manifest_sha256="c" * 64),
        session=SimpleNamespace(run_motion_operation=dispatch),
        _receipt=lambda *args: args,
    )
    return prepared, owner, calls


def test_transfer_attachment_uses_post_approach_receiver_tcp():
    """Exercise real live transfer from current TCP, including its digest API."""
    prepared, owner, calls = _live_transfer_fixture()
    raw = snapshot(4)
    asyncio.run(LiveSimulationHandover._transfer(
        owner, prepared, "scene_transfer", {"scene_snapshot": raw},
    ))
    assert len(calls) == 1
    _, op = calls[0]
    assert op.payload["frame_id"] == "piper_0_rocobrick_tcp"
    assert op.payload["minimum_m"] == pytest.approx((-0.008, -0.008, -0.0055))
    assert op.payload["maximum_m"] == pytest.approx((0.008, 0.008, 0.0055))
    assert op.payload["geometry_digest"] == box_geometry_digest(
        object_id="brick-1", frame_id="piper_0_rocobrick_tcp",
        minimum_m=op.payload["minimum_m"], maximum_m=op.payload["maximum_m"],
    )


@pytest.mark.parametrize(
    "key,value", [("session_id", "foreign"), ("manifest_sha256", "e" * 64)]
)
def test_transfer_attachment_rejects_foreign_provenance_before_dispatch(key, value):
    """Old attachment helper's provenance guard now exercises the real live path."""
    prepared, owner, calls = _live_transfer_fixture()
    raw = deepcopy(snapshot(4))
    raw[key] = value
    with pytest.raises(ValueError):
        asyncio.run(LiveSimulationHandover._transfer(
            owner, prepared, "scene_transfer", {"scene_snapshot": raw},
        ))
    assert not calls


def test_direct_backend_pre_dispatch_failure_is_fail_closed_not_terminal_safe():
    """Record Direct's conservative nonterminal result, not old backend equivalence."""
    owner = SimpleNamespace(session=SimpleNamespace())
    backend = DirectHandoverBackend(owner)
    op = SimpleNamespace(operation_id="direct-no-dispatch", executor_robot_id="piper_0")
    result = asyncio.run(backend.execute(op))
    assert not result.passed and not result.motion_sent and not result.terminal
    assert result.evidence["handover_complete"] is False
    assert "persistent combined planning/execution is unavailable" in result.reason
    assert backend._motion_ids == {}


def test_receiver_contact_boundary_keeps_the_pick_floor_tip_clearance(tmp_path):
    """Retire the PICK z clamp, but retain actual full-model floor rejection."""
    from test_handover_geometry import native_scene

    from rocobrick.adapters.moveit.handover_geometry import IsolatedHandoverGeometry

    scene, item = native_scene(tmp_path)
    assert scene.check(item.start_state, holder=item.donor.robot_id).valid
    captured = deepcopy(item.scene_snapshot)
    captured["environment_colliders"]["floor"] = {
        "aabb_min": [-2.0, -2.0, -0.1], "aabb_max": [2.0, 2.0, 0.1],
    }
    obstructed = replace(item, scene_snapshot=captured)
    blocked = IsolatedHandoverGeometry(scene.files, obstructed)
    assert not blocked.check(obstructed.start_state, holder=item.donor.robot_id).valid


def _planning_fixture():
    """Create an admitted physical request and a ROS-free combined planner.

    Returns:
        The request and fake session with all planning gates available.
    """
    arms = {robot: tuple(f"{robot}_joint{i}" for i in range(1, 7)) for robot in ROBOTS}
    names = [joint for robot in ROBOTS for joint in arms[robot]]
    item = replace(
        request({robot: pose(robot, 0.1) for robot in ROBOTS}),
        plan_only=False,
        scene_snapshot={"sequence": 1, "sim_time_s": 1.0},
        physical_facts={robot: {} for robot in ROBOTS},
    )
    session = SimpleNamespace(
        robot_ids=ROBOTS,
        runtime_for=lambda robot: SimpleNamespace(arm_joint_names=arms[robot]),
        coordinated_readiness=lambda req: {
            "coordination_id": req.coordination_id,
            "participants": list(ROBOTS), "scene_digest": req.scene_digest,
            "scene_sequence": 1, "scene_sim_time_s": 1.0,
            "move_group_action": "/dual_arm/move_action", "planning_group": "dual_arm",
            "components": {"clock_increasing": True},
        },
        sync_unified_scene=lambda req: {
            "readback_confirmed": True, "snapshot_digest": req.scene_digest,
            "scene_version": 1,
        },
        unified_joint_targets=lambda req: {
            robot: {joint: 0.05 for joint in arms[robot]} for robot in ROBOTS
        },
        unified_start_state=lambda: {joint: 0.0 for joint in names},
        # The sim-arbiter clearance gate reads the combined robot description
        # the planner used.  This double declares no collidable bodies, so the
        # gate has nothing to measure; the real session serves the latched
        # /dual_arm/robot_description.
        unified_robot_description=lambda: '<robot name="dual_stub"><link name="dual_world"/><link name="stub_link"/>'
        '<joint name="stub_joint" type="fixed"><parent link="dual_world"/>'
        '<child link="stub_link"/></joint></robot>',
        coordinated_plan_runtime=lambda: SimpleNamespace(
            plan=lambda req, timeout_s: {
                "joint_names": tuple(names),
                "points": (
                    {"time_from_start_s": 0.0, "positions": tuple(0.0 for _ in names)},
                    {"time_from_start_s": 1.0, "positions": tuple(0.05 for _ in names)},
                ),
            }
        ),
        validate_unified_scene=lambda proof: True,
    )
    return item, session


@pytest.mark.parametrize("stage", ["ik", "plan", "readback"])
def test_failure_before_executor_has_explicit_no_dispatch_proof(stage):
    """IK and planning errors fail the run but can stop the worker orderly."""
    item, session = _planning_fixture()
    calls = []

    def fail(*args, **kwargs):
        raise RuntimeError("injected planning failure")

    if stage == "ik":
        session.unified_joint_targets = fail
    elif stage == "plan":
        session.coordinated_plan_runtime = lambda: SimpleNamespace(plan=fail)
    else:
        session.validate_unified_scene = lambda proof: False
    result = perform_coordinated_plan(session, item, lambda op: calls.append(op))
    assert not result.passed
    assert not calls
    assert result.lifecycle["dispatch_attempted"] is False
    assert result.lifecycle["physical_execution"] is False
    assert result.lifecycle["terminal"] is True
    assert result.lifecycle["safe_to_release"] is True


@pytest.mark.parametrize("invalid_return", [False, True])
def test_exception_or_invalid_evidence_after_executor_is_unsafe(invalid_return):
    """A callback may have prepared motion before raising or returning junk."""
    item, session = _planning_fixture()
    calls = []

    def execute(op):
        calls.append(op)
        if invalid_return:
            return None
        raise RuntimeError("injected after dispatch")

    result = perform_coordinated_plan(session, item, execute)
    assert calls
    assert not result.passed
    assert result.lifecycle["dispatch_attempted"] is True
    assert result.lifecycle["terminal"] is False
    assert result.lifecycle["safe_to_release"] is False


def test_worker_planning_failure_blocks_admission_but_allows_orderly_stop(tmp_path):
    """A known no-dispatch failure is distinct from an uncertain robot stop."""
    item, session = _planning_fixture()

    def fail(req):
        raise RuntimeError("injected IK failure")

    session.unified_joint_targets = fail
    session.execute_coordinated_pre_phase = lambda req: perform_coordinated_plan(
        session, req, lambda op: pytest.fail("must never execute")
    )
    worker = SessionWorker(tmp_path, TOKEN, ROBOTS, session)
    write_atomic(
        tmp_path / "requests" / ("1" * 32 + ".json"),
        {
            "schema": SCHEMA, "token": TOKEN,
            "request": {
                "schema": "rocobrick-coordinated-execute/v1",
                "request_id": "physical-1", "request": item.serialize(),
            },
        },
    )
    pump(worker, lambda: "1" * 32 in worker.seen and not worker.coordinated_active)
    assert worker.failed
    assert not worker.unsafe
    assert worker.owners == {}
    worker.stopping.set()
    assert worker.run() is True
