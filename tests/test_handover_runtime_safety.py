"""Offline regressions for HANDOVER targets and no-dispatch shutdown proof."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_coordinated_ros_runtime import pose, request
from test_handoff_contract import _handover_request
from test_persistent_ros_worker import ROBOTS, TOKEN, pump

from rocobrick.coordinated_session import FreeSpaceMotionEvidence
from rocobrick.coordinated_session_entry import perform_coordinated_plan
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.execution.ros_session_worker import SCHEMA, SessionWorker, write_atomic
from rocobrick.planning_scene import box_geometry_digest
from rocobrick.skills.handover import (
    HandoverHandler,
    HandoverOperationBackend,
    receiver_attachment_from_sample,
    receiver_precontact_target,
    require_receiver_contact_geometry,
)
from rocobrick.skills.observation import PickObservation, Pose
from rocobrick.skills.targets import (
    RobotBaseFrame,
    SkillParams,
    TcpProvenance,
    pick_targets,
)


@pytest.mark.parametrize("robot", ROBOTS)
def test_receiver_target_is_measured_object_boundary_in_its_own_base(robot):
    """Rotated/translated bases preserve both pose numbers and frame identity."""
    base = RobotBaseFrame((0.25, -0.25, 0.0), (0.0, 0.0, 2**-0.5, 2**-0.5))
    params = SkillParams()
    targets = pick_targets(
        (-0.09, 0.02, 0.10),
        (-0.07, 0.04, 0.12),
        base,
        TcpProvenance("a" * 64, "b" * 64),
        params,
    )
    target = receiver_precontact_target(
        robot, targets,
        aabb_min=(-0.09, 0.02, 0.10), aabb_max=(-0.07, 0.04, 0.12), base=base,
    )["tcp_pose"]
    assert target["frame_id"] == f"{robot}_base_link"
    assert target["tip_link"] == f"{robot}_rocobrick_tcp"
    assert target["position"] == pytest.approx((0.28, 0.33, 0.1135))
    assert target["orientation_xyzw"] == pytest.approx(
        targets.precontact_boundary.orientation_xyzw
    )
    assert target["runtime_model_sha256"] == "a" * 64
    assert target["source_sha256"] == "b" * 64
    assert target["position_tolerance_m"] == params.position_tolerance_m


def test_receiver_contact_boundary_keeps_the_pick_floor_tip_clearance():
    """Transfer planning retains the successful PICK path's floor margin."""
    from rocobrick.skills.pick import FINGER_TIP_BELOW_TCP_M, FLOOR_CONTACT_MARGIN_M

    base = RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    minimum, maximum = (-0.01, -0.01, 0.0), (0.01, 0.01, 0.01)
    targets = pick_targets(
        minimum, maximum, base, TcpProvenance("a" * 64, "b" * 64), SkillParams()
    )
    target = receiver_precontact_target(
        "piper_1", targets, aabb_min=minimum, aabb_max=maximum, base=base,
        ground_top_z_m=0.0,
    )["tcp_pose"]
    assert target["position"][2] == pytest.approx(
        FINGER_TIP_BELOW_TCP_M + FLOOR_CONTACT_MARGIN_M
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_id", "base_link"),
        ("frame_id", "piper_0_base_link"),
        ("tip_link", "piper_0_rocobrick_tcp"),
    ],
)
def test_handover_contract_rejects_default_or_foreign_receiver_frame(field, value):
    """The production frame regression is refused before calling ROS IK."""
    item = _handover_request()
    tcp = {**item.receiver_target["tcp_pose"], field: value}
    with pytest.raises(ValueError, match="frame/tip"):
        replace(item, receiver_target={"tcp_pose": tcp})


def _sample(tcp_position=(0.0, 0.0, 0.1)):
    """Return an identified cube between horizontal finger pads."""
    return PickObservation(
        "piper_1", "brick-1", "live-session", "c" * 64, 4, 2.0,
        Pose((0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0)),
        Pose(tcp_position, (0.0, 0.0, 0.0, 1.0)),
        0.015, (-0.01, -0.01, 0.09), (0.01, 0.01, 0.11),
    )


@pytest.mark.parametrize(
    "position", [(0.3, 0.0, 0.1), (0.0, 0.3, 0.1), (0.0, 0.0, 0.24)]
)
def test_distant_stationary_object_cannot_prove_receiver_contact(position):
    """Stationary relative pose and plausible opening alone are insufficient."""
    with pytest.raises(ValueError, match="contact geometry"):
        require_receiver_contact_geometry(_sample(position), tolerance_m=0.005)
    require_receiver_contact_geometry(_sample(), tolerance_m=0.005)


def test_cohold_geometry_failure_never_transfers_scene_or_releases_donor():
    """Failure at the co-hold gate preserves the confirmed physical owner."""
    item = replace(_handover_request(), scene_snapshot=_snapshot(_sample()))
    calls = []

    class FreeSpace:
        async def execute_free_space(self, req):
            """Return an arrived free-space lifecycle, not holding proof."""
            return FreeSpaceMotionEvidence(
                req.motion_id, req.participants,
                planned=True, prepared=True, started=True, arrival_confirmed=True,
                terminal=True, safe_to_release=True,
                participant_evidence={
                    robot: {"arrival_confirmed": True} for robot in req.participants
                },
                evidence={}, reason=None,
            )

    class Motion:
        def run_motion_operation(self, robot, operation):
            """Return successful gripper feedback and record every dispatch."""
            calls.append((robot, operation.kind.value))
            return {"result": {"passed": True, "execution": {
                "terminal": True, "safe_to_release": True, "goal_accepted": True,
            }}}

    class Observations:
        async def observe(self, req):
            """Return a stationary object far from the receiver."""
            return _snapshot(_sample((0.3, 0.0, 0.1)))

        def verify_co_hold(self, req, samples):
            """Apply the same mandatory geometry guard as live observations."""
            require_receiver_contact_geometry(
                _sample((0.3, 0.0, 0.1)), tolerance_m=0.005
            )
            pytest.fail("distant object cannot prove co-hold")

    ledger = OperationStateLedger(ROBOTS)
    ledger.confirm_holding("piper_0", "brick-1", {"holding_confirmed": True})
    result = asyncio.run(HandoverHandler(
        ledger, free_space=FreeSpace(), motion=Motion(), observations=Observations(),
        poll_interval_s=0.0,
    ).execute(item))
    assert not result.ready_to_commit
    assert "contact geometry" in result.reason
    assert len(calls) == 1 and calls[0][0] == "piper_1"
    assert ledger.owner_of_object("brick-1") == "piper_0"
    assert "scene_transfer_committed" not in result.evidence
    assert "donor_release_feedback" not in result.evidence


def test_handover_no_dispatch_failure_does_not_claim_motion():
    """A planning refusal is terminal-safe but is never a successful transfer."""
    from test_handoff_contract import _handover_evidence

    evidence = _handover_evidence(
        receiver_free_space=False, receiver_grasp_feedback=False,
        co_hold_confirmed=False, scene_transfer_committed=False,
        donor_release_feedback=False, receiver_holding_follow=False,
        evidence={"receiver_free_space": {"lifecycle": {
            "dispatch_attempted": False, "physical_execution": False,
            "terminal": True, "safe_to_release": True,
        }}},
    )
    operation = SimpleNamespace(operation_id="handover-1", executor_robot_id="piper_1")
    result = HandoverOperationBackend._result(operation, evidence, committed=False)
    assert not result.motion_sent
    assert not result.passed
    assert result.terminal
    assert result.evidence["owner_transfer_committed"] is False


@pytest.mark.parametrize("failure", ["scene-transfer", "release"])
def test_attempted_transfer_failure_invalidates_confirmed_holder(failure):
    """Timeouts cannot leave the old ownership record as physical proof."""
    class Handler(HandoverHandler):
        async def _execute_free_space(self, req):
            return FreeSpaceMotionEvidence(
                req.operation_id, (req.receiver_robot_id,), True, True, True,
                True, True, True,
                {req.receiver_robot_id: {"arrival_confirmed": True}}, {}, None,
            )

        async def _dispatch_gripper(self, robot, payload, req, phase):
            if phase == "donor_release":
                raise TimeoutError("release result unavailable after dispatch")
            return {"passed": True}

        async def _samples(self, req):
            return ({"sequence": 4},) * req.follow_samples

        async def _dispatch_transfer(self, req, sample):
            if failure == "scene-transfer":
                raise TimeoutError("scene result unavailable after request")
            return {"committed": True}

    class Observations:
        def verify_co_hold(self, req, samples):
            return {"confirmed": True}

    item = _handover_request()
    ledger = OperationStateLedger(ROBOTS)
    ledger.confirm_holding("piper_0", "brick-1", {"holding_confirmed": True})
    result = asyncio.run(Handler(
        ledger, free_space=object(), motion=object(),
        observations=Observations(), poll_interval_s=0.0,
    ).execute(item))
    assert not result.ready_to_commit
    assert not result.terminal and not result.safe_to_release
    assert result.evidence["scene_transfer_attempted"] is True
    assert (result.evidence.get("donor_release_attempted") is True) == (
        failure == "release"
    )
    assert ledger.uncertain_handover("brick-1") is not None
    assert ledger.observe("piper_0", "holding:brick-1") is None
    assert ledger.observe("piper_1", "holding:brick-1") is None


def _snapshot(sample):
    """Serialize one observation using the strict live scene schema.

    Returns:
        One co-sampled scene snapshot with exact provenance.
    """
    return {
        "schema": "rocobrick-scene-state/v1",
        "session_id": sample.session_id,
        "manifest_sha256": sample.manifest_sha256,
        "scene_digest": "a" * 64,
        "sequence": sample.sequence,
        "sim_time_s": sample.sim_time_s,
        "robots": {
            sample.robot_id: {
                "robot_id": sample.robot_id,
                "sequence": sample.sequence,
                "sim_time_s": sample.sim_time_s,
                "opening_m": sample.opening_m,
                "tcp_pose": {
                    "position_xyz": list(sample.tcp_pose.position_xyz),
                    "orientation_xyzw": list(sample.tcp_pose.orientation_xyzw),
                },
            }
        },
        "bricks": {
            sample.physical_instance_id: {
                "aabb_min": list(sample.aabb_min),
                "aabb_max": list(sample.aabb_max),
                "pose": {
                    "position_xyz": list(sample.object_pose.position_xyz),
                    "orientation_xyzw": list(sample.object_pose.orientation_xyzw),
                },
            }
        },
    }


def test_transfer_attachment_uses_post_approach_receiver_tcp():
    """Pre-motion receiver-relative boxes must not survive receiver movement."""
    initial = _snapshot(_sample((0.3, 0.0, 0.24)))
    item = replace(_handover_request(), scene_snapshot=initial)
    attach = receiver_attachment_from_sample(item, _snapshot(_sample()))
    assert attach["minimum_m"] == pytest.approx((-0.01, -0.01, -0.01))
    assert attach["maximum_m"] == pytest.approx((0.01, 0.01, 0.01))
    assert attach["geometry_digest"] == box_geometry_digest(
        object_id="brick-1", frame_id="piper_1_rocobrick_tcp",
        minimum_m=attach["minimum_m"], maximum_m=attach["maximum_m"],
    )
    foreign = _snapshot(_sample())
    foreign["session_id"] = "foreign"
    with pytest.raises(ValueError, match="provenance"):
        receiver_attachment_from_sample(item, foreign)


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
