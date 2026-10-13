"""Continuous grasp-region contracts without Isaac/ROS startup or motion."""

import asyncio
import hashlib
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_coordinated_ros_runtime import Factory, pose, runtime
from test_pick_handler import Motion, Observer, document, operation
from test_place_down_handler import context

from rocobrick.coordinated_session import CoordinatedSessionRequest
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.moveit_contract import JOINT_NAMES
from rocobrick.readiness import state_is_complete
from rocobrick.skills.grasp_region import (
    CALIBRATION_PATH,
    admit_grasp_endpoint,
    calibrated_pad_geometry,
    place_up_grasp_region,
    planned_grasp_endpoints,
    targets_from_grasp_endpoint,
    validate_grasp_region,
    validate_grasp_scene,
)
from rocobrick.skills.observation import Pose, parse_pick_observation
from rocobrick.skills.pick import PickHandler
from rocobrick.skills.targets import pick_targets


def sample():
    """Return a measured object/TCP fixture.

    Returns:
        The co-sampled identified brick observation.
    """
    return parse_pick_observation(
        document(),
        robot_id="piper_0",
        physical_instance_id="brick-a",
        session_id="session-1",
        manifest_sha256="a" * 64,
    )


def region():
    """Inject explicit pad geometry rather than bypassing its production pin.

    Returns:
        A side-pinch region, independent of Task C IDs and slot positions.
    """
    value = place_up_grasp_region(
        sample(),
        context(),
        geometry={
            "pad_vertices_tcp": [[-0.0127, 0.0, -0.0048], [0.0127, 0.0, 0.0052]],
            "flange_to_tcp_m": 0.1305896,
        },
    )
    value["ground_top_world_m"] = 0.0
    return value


def endpoint():
    """Return a side entry that keeps its complete pad envelope off the floor.

    Returns:
        Actual TCP pose/opening, not a manufactured holding observation.
    """
    return replace(
        sample(),
        tcp_pose=Pose((0.01, 0.01, 0.018), (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))),
        opening_m=0.06,
    )


def query(targets, participants=("piper_0",)):
    """Bind combined measured joints and calibration to mixed goals.

    Returns:
        A planning request under the existing session envelope.
    """
    return CoordinatedSessionRequest(
        coordination_id="region-1",
        robots=("piper_0", "piper_1"),
        participants=participants,
        scene_digest="scene-1",
        planning_group="dual_arm",
        plan_only=True,
        timeout_s=30.0,
        targets=targets,
        start_state={
            **{
                f"{robot}_{name}": 0.0
                for robot in ("piper_0", "piper_1")
                for name in JOINT_NAMES
            },
            "piper_0_joint1": 0.1,
            "piper_1_joint1": 0.2,
            "piper_0_gripper": 0.06,
            "piper_1_gripper": 0.06,
        },
        scene_snapshot={
            **document(),
            "robots": {
                robot: {
                    **document()["robots"]["piper_0"],
                    "robot_id": robot,
                    "opening_m": 0.06,
                }
                for robot in ("piper_0", "piper_1")
            },
            "bricks": document()["bricks"],
            "tcp_provenance": {
                "runtime_model_sha256": context().provenance.runtime_model_sha256,
                "source_sha256": context().provenance.source_sha256,
            },
        },
    )


@pytest.mark.parametrize("plan_only", [True, False])
def test_session_entry_preserves_region_without_outer_ik_or_endpoint_fk(plan_only):
    """Leave grasp verification to measured arrival, not planned endpoint FK."""
    from rocobrick.coordinated_session_entry import perform_coordinated_plan

    captured = []
    executed = []
    item = replace(
        query(
            {"piper_0": {"grasp_region": region()}, "piper_1": pose("piper_1", 0.1)},
            participants=("piper_0", "piper_1"),
        ),
        physical_facts={"piper_0": {}, "piper_1": {}},
        plan_only=plan_only,
    )

    def plan(value, timeout_s):
        captured.append(value)
        return {
            "joint_names": ["piper_0_joint1", "piper_1_joint1"],
            "points": [
                {"time_from_start_s": 0.0, "positions": [0.1, 0.2]},
                {"time_from_start_s": 1.0, "positions": [0.15, 0.3]},
            ],
        }

    session = SimpleNamespace(
        robot_ids=item.robots,
        coordinated_readiness=lambda value: {
            "coordination_id": value.coordination_id,
            "scene_digest": value.scene_digest,
            "scene_sequence": 10,
            "scene_sim_time_s": 100.0,
            "participants": list(value.participants),
            "move_group_action": "/dual_arm/move_action",
            "planning_group": "dual_arm",
            "components": {"scene": True, "move_group": True},
        },
        coordinated_plan_runtime=lambda: SimpleNamespace(plan=plan),
        sync_unified_scene=lambda value: {
            "readback_confirmed": True,
            "snapshot_digest": value.scene_digest,
        },
        validate_unified_scene=lambda proof: True,
        unified_start_state=lambda: item.start_state,
        unified_joint_targets=lambda value: pytest.fail(
            "outer IK must not consume region or exact mixed peer"
        ),
        runtime_for=lambda robot: SimpleNamespace(
            arm_joint_names=(f"{robot}_joint1",),
            compute_tcp_fk=lambda state, timeout_s: pytest.fail(
                "planned endpoint FK must not gate grasp execution"
            ),
        ),
    )

    def execute(value):
        executed.append(value)
        return {
            "prepared": True,
            "started": True,
            "terminal": True,
            "safe_to_release": True,
        }

    result = perform_coordinated_plan(
        session, item, participants_executor=None if plan_only else execute
    )
    assert result.passed, result.error
    assert captured[0].targets == item.targets
    assert result.lifecycle["physical_execution"] is (not plan_only)
    assert len(executed) == (0 if plan_only else 1)
    assert "grasp_endpoints" not in result.trajectory
    assert result.trajectory["points"][-1]["positions"] == [0.15, 0.3]


def test_region_has_no_locked_quaternion_or_host_candidate_list():
    """Verify region has no locked quaternion or host candidate list."""
    value = region()
    validate_grasp_region(value)
    assert "orientation_xyzw" not in value
    assert "candidates" not in value
    assert value["physical_instance_id"] == "brick-a"


def test_region_binds_measured_object_pose_in_shared_snapshot():
    """Reject target geometry silently diverging from the shared world."""
    validate_grasp_scene(region(), document())
    changed = document()
    changed["bricks"]["brick-a"]["pose"]["position_xyz"][0] += 0.01
    with pytest.raises(ValueError, match="shared snapshot"):
        validate_grasp_scene(region(), changed)


def test_two_grasp_regions_share_one_combined_sampler_binding():
    """Two moving arms receive independent regions in a single dual-arm goal."""
    second = {
        **region(),
        "robot_id": "piper_1",
        "frame_id": "piper_1_base_link",
        "tip_link": "piper_1_rocobrick_tcp",
        "physical_instance_id": "brick-b",
    }
    item = query(
        {"piper_0": {"grasp_region": region()}, "piper_1": {"grasp_region": second}},
        participants=("piper_0", "piper_1"),
    )
    goal = runtime()._build_goal(item, Factory())
    constraints = goal.request.goal_constraints[0]
    assert len(goal.request.goal_constraints) == 1
    assert constraints.name.startswith("rocobrick-grasp-region/v1;2;")
    assert len(constraints.position_constraints) == 2
    assert constraints.joint_constraints == []
    assert goal.request.path_constraints.joint_constraints == []


def test_pinned_pad_geometry_uses_existing_calibration_not_new_offsets():
    """Verify pinned pad geometry uses existing calibration not new offsets."""
    sha = hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest()
    geometry = calibrated_pad_geometry(CALIBRATION_PATH, sha)
    points = geometry["pad_vertices_tcp"]
    assert max(p[0] for p in points) - min(p[0] for p in points) == pytest.approx(
        0.02542976, abs=1e-7
    )
    assert max(p[2] for p in points) == pytest.approx(0.0052104, abs=1e-7)
    with pytest.raises(ValueError, match="digest"):
        calibrated_pad_geometry(CALIBRATION_PATH, "0" * 64)


def test_actual_endpoint_drives_close_width_lift_and_orientation():
    """Verify actual endpoint drives close width lift and orientation."""
    ctx = context()
    observed = endpoint()
    original = pick_targets(
        observed.aabb_min, observed.aabb_max, ctx.base, ctx.provenance, ctx.params
    )
    result = targets_from_grasp_endpoint(region(), observed, ctx, original)
    assert result.brick_span_m == pytest.approx(0.02)
    assert result.lift.orientation_xyzw == observed.tcp_pose.orientation_xyzw
    assert result.lift.position[2] - result.precontact_boundary.position[
        2
    ] == pytest.approx(ctx.params.lift_height_m)
    assert result.close_opening_m < 0.02


@pytest.mark.parametrize(
    "defect", ["floor", "top", "closing_axis", "brick_moved", "not_open", "identity"]
)
def test_arrival_is_not_a_grasp_certificate_and_rejects_unsafe_endpoints(defect):
    """Verify arrival is not a grasp certificate and rejects unsafe endpoints."""
    observed = endpoint()
    if defect == "floor":
        observed = replace(
            observed,
            tcp_pose=replace(observed.tcp_pose, position_xyz=(0.01, 0.01, 0.01)),
        )
    elif defect == "top":
        observed = replace(
            observed,
            tcp_pose=replace(observed.tcp_pose, orientation_xyzw=(0.0, 1.0, 0.0, 0.0)),
        )
    elif defect == "closing_axis":
        observed = replace(
            observed,
            tcp_pose=replace(
                observed.tcp_pose,
                orientation_xyzw=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            ),
        )
    elif defect == "brick_moved":
        observed = replace(
            observed,
            object_pose=replace(observed.object_pose, position_xyz=(0.02, 0.0, 0.01)),
        )
    elif defect == "not_open":
        observed = replace(observed, opening_m=0.015)
    else:
        observed = replace(observed, robot_id="piper_1")
    with pytest.raises(ValueError):
        admit_grasp_endpoint(region(), observed)


def test_region_plus_nonparticipant_hold_is_one_combined_goal():
    """Verify region plus nonparticipant hold is one combined goal."""
    item = query({"piper_0": {"grasp_region": region()}})
    goal = runtime()._build_goal(item, Factory())
    assert goal.request.group_name == "dual_arm"
    assert len(goal.request.goal_constraints) == 1
    combined = goal.request.goal_constraints[0]
    assert combined.name.startswith("rocobrick-grasp-region/v1;1;piper_0;")
    assert len(combined.position_constraints) == 1
    assert [j.joint_name for j in combined.joint_constraints] == ["piper_1_joint1"]
    assert [j.joint_name for j in goal.request.path_constraints.joint_constraints] == [
        "piper_1_joint1"
    ]
    assert goal.planning_options.plan_only is True


def test_region_can_share_one_goal_with_a_moving_peer_tcp():
    """Verify region can share one goal with a moving peer tcp."""
    peer = pose("piper_1", 0.1)
    peer["tcp_pose"].update(
        runtime_model_sha256=context().provenance.runtime_model_sha256,
        source_sha256=context().provenance.source_sha256,
    )
    item = query(
        {"piper_0": {"grasp_region": region()}, "piper_1": peer},
        participants=("piper_0", "piper_1"),
    )
    combined = runtime()._build_goal(item, Factory()).request.goal_constraints[0]
    assert len(combined.position_constraints) == 2
    assert len(combined.orientation_constraints) == 2
    assert combined.joint_constraints == []


def test_implicit_hold_is_checked_on_every_returned_waypoint():
    """Verify implicit hold is checked on every returned waypoint."""
    item = query({"piper_0": {"grasp_region": region()}})
    with pytest.raises(ValueError, match="whole-path joint hold"):
        runtime()._validate_held_joints(
            item,
            ["piper_0_joint1", "piper_1_joint1"],
            [{"positions": [0.1, 0.2]}, {"positions": [0.1, 0.25]}],
        )


def test_region_rejects_fixed_joint_override_and_wrong_calibration():
    """Verify region rejects fixed joint override and wrong calibration."""
    item = query(
        {
            "piper_0": {
                "grasp_region": region(),
                "joint_positions": {"piper_0_joint1": 0.1},
            }
        }
    )
    with pytest.raises(ValueError, match="cannot carry fixed"):
        runtime()._build_goal(item, Factory())
    value = {**region(), "source_sha256": "0" * 64}
    with pytest.raises(ValueError, match="calibration"):
        runtime()._build_goal(query({"piper_0": {"grasp_region": value}}), Factory())


def test_active_pick_region_cancels_through_combined_boundary():
    """Verify active pick region cancels through combined boundary."""
    calls = []

    async def cancel(motion_id):
        calls.append(motion_id)
        return {"attempted": True, "terminal": True, "safe": True}

    handler = PickHandler(
        SimpleNamespace(),
        SimpleNamespace(),
        session_id="session-1",
        manifest_sha256="a" * 64,
        free_space_transport=SimpleNamespace(cancel_free_space=cancel),
    )
    handler._active_free_space[operation().operation_id] = "combined-region-motion"
    result = asyncio.run(handler.cancel(operation().operation_id))
    assert calls == ["combined-region-motion"]
    assert result["safe"] is True


def test_planned_endpoint_fk_rejects_broad_goal_that_is_not_a_grasp():
    """A missing/custom-sampler fallback cannot move toward an invalid grasp."""
    item = query({"piper_0": {"grasp_region": region()}})
    session = SimpleNamespace(
        runtime_for=lambda robot: SimpleNamespace(
            compute_tcp_fk=lambda state, timeout_s: (
                (0.01, 0.01, 0.01),
                (0.0, 1.0, 0.0, 0.0),
            )
        )
    )
    with pytest.raises(ValueError):
        planned_grasp_endpoints(
            session, item, ["piper_0_joint1"], [{"positions": [0.1]}]
        )


@pytest.mark.parametrize("robot", ["piper_0", "piper_1"])
def test_planned_endpoint_fk_filters_mimic_parent_and_preserves_final_joints(robot):
    """Use production FK admission, not a permissive pose stub."""
    value = {
        **region(),
        "robot_id": robot,
        "frame_id": f"{robot}_base_link",
        "tip_link": f"{robot}_rocobrick_tcp",
    }
    item = query({robot: {"grasp_region": value}}, participants=(robot,))
    calls = []

    def strict_fk(state, timeout_s):
        calls.append(state)
        assert state_is_complete(state)
        assert set(state) == set(JOINT_NAMES)
        assert "gripper" not in state
        assert state["joint1"] == pytest.approx(0.35)
        return endpoint().tcp_pose.position_xyz, endpoint().tcp_pose.orientation_xyzw

    session = SimpleNamespace(
        runtime_for=lambda owner: SimpleNamespace(compute_tcp_fk=strict_fk)
    )
    result = planned_grasp_endpoints(
        session, item, [f"{robot}_joint1"], [{"positions": [0.35]}]
    )
    assert len(calls) == 1
    assert item.start_state[f"{robot}_gripper"] == 0.06
    assert result[robot]["physical_holding_confirmed"] is False


def test_planned_endpoint_fk_rejects_missing_real_joint_without_calling_service():
    """Mimic filtering must not conceal an incomplete actual joint state."""
    item = query({"piper_0": {"grasp_region": region()}})
    state = dict(item.start_state)
    state.pop("piper_0_gripper_joint2")
    item = replace(item, start_state=state)
    session = SimpleNamespace(
        runtime_for=lambda owner: SimpleNamespace(
            compute_tcp_fk=lambda *args, **kwargs: pytest.fail(
                "incomplete FK state must not be sent"
            )
        )
    )
    with pytest.raises(ValueError, match="missing joints.*gripper_joint2"):
        planned_grasp_endpoints(
            session, item, ["piper_0_joint1"], [{"positions": [0.35]}]
        )


def test_cancelled_coroutine_retains_addressable_physical_request():
    """Async task cancellation is not proof that the combined goal stopped."""

    async def execute(request):
        raise asyncio.CancelledError

    handler = PickHandler(
        SimpleNamespace(),
        SimpleNamespace(),
        session_id="session-1",
        manifest_sha256="a" * 64,
        free_space_transport=SimpleNamespace(execute_free_space=execute),
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            handler.execute_free_space_phase(
                operation(),
                scene_snapshot={},
                scene_version=1,
                start_state={},
                targets={"piper_0": {"grasp_region": region()}},
                session_robots=("piper_0", "piper_1"),
                phase_id="region",
            )
        )
    assert handler._active_free_space[operation().operation_id] == "pick-1:region"


def test_place_up_pick_handler_sends_region_once_and_closes_at_actual_endpoint():
    """Preserve normal PICK phases without host pose scans or extra waypoints."""
    shared = document()
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm(
        "piper_0", "robot_at_pick_region", {"physical_instance_id": "brick-a"}
    )
    phases, requests = [], []

    async def capture(participants):
        return shared, 1, {"piper_0_joint1": 0.0, "piper_1_joint1": 0.0}

    handler = PickHandler(
        Observer(shared),
        ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=Motion(),
        free_space_transport=SimpleNamespace(
            robot_ids=("piper_0", "piper_1"), free_space_context=capture
        ),
        context=context(),
    )

    async def dispatch(item, phase, value):
        phases.append(phase)
        requests.append((phase, value))
        if phase == "open_gripper":
            shared["robots"]["piper_0"]["opening_m"] = value.payload["opening_m"]
        return SimpleNamespace(
            passed=phase != "close_gripper",
            terminal=True,
            safe=True,
            motion_sent=phase == "open_gripper",
            evidence={},
            reason="stop before lift",
        )

    async def approach(item, **kwargs):
        phases.append(kwargs["phase_id"])
        assert set(kwargs["targets"]["piper_0"]) == {"grasp_region"}
        selected = endpoint().tcp_pose
        shared["robots"]["piper_0"]["tcp_pose"] = {
            "position_xyz": selected.position_xyz,
            "orientation_xyzw": selected.orientation_xyzw,
        }
        return SimpleNamespace(
            planned=True,
            prepared=True,
            started=True,
            arrival_confirmed=True,
            terminal=True,
            safe_to_release=True,
            evidence={},
            motion_id="pick-1:precontact_boundary",
            participants=("piper_0",),
            reason=None,
        )

    handler._dispatch = dispatch
    handler.execute_free_space_phase = approach
    item = replace(operation(), parameters={"successor_skill": "PLACE_UP"})
    result = asyncio.run(handler.execute(item))
    assert phases == [
        "plan_grasp",
        "open_gripper",
        "precontact_boundary",
        "allow_grasp_contact",
        "close_gripper",
    ], result.reason
    close = next(value for phase, value in requests if phase == "close_gripper")
    assert close.payload["opening_m"] == pytest.approx(
        0.02 - context().params.squeeze_m
    )
    assert (
        result.evidence["successor_grasp"]["mode"] == "moveit_continuous_grasp_region"
    )
    assert result.evidence["lift_geometry"]["selected_orientation_xyzw"] == list(
        endpoint().tcp_pose.orientation_xyzw
    )
    assert ledger.owner_of_object("brick-a") is None
    assert not hasattr(handler, "_select_successor_grasp")
