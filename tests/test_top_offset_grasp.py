"""Vertical offset grasp geometry and PICK phase regression without simulation."""

import asyncio
import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_coordinated_ros_runtime import Factory, pose, runtime
from test_pick_handler import Motion, Observer, document, operation
from test_place_down_handler import context

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.plan.structure_scene import build_structure_scene_plan_from_files
from rocobrick.robot_operation_plan import parse_robot_operation_plan
from rocobrick.skills.observation import Pose, parse_pick_observation
from rocobrick.skills.pick import PickHandler
from rocobrick.skills.placement_geometry import compose
from rocobrick.skills.targets import (
    _quat_mul,
    _quat_rotate_vector,
    pick_targets,
)
from rocobrick.skills.top_offset_grasp import (
    TopOffsetBinding,
    top_offset_bindings,
    top_offset_targets,
    verify_top_offset_arrival,
)

ROOT = Path(__file__).parents[1]


def binding():
    """Return one exposed negative-Y half of a 4x4 brick.

    Returns:
        Structural binding independent of world placement.
    """
    return TopOffsetBinding(
        "brick-a", "brick-b", 1, -1, 0.016, 0.032, (0.032, 0.032), 0.5
    )


def observed(shared):
    """Parse an identified co-sampled test scene.

    Returns:
        Strict measured PICK observation.
    """
    return parse_pick_observation(
        shared,
        robot_id="piper_0",
        physical_instance_id="brick-a",
        session_id="session-1",
        manifest_sha256="a" * 64,
    )


def test_real_task_c_has_half_overlap_and_negative_y_free_strip():
    """Compute overlap from original JSON, not the mismatched live interface pose."""
    payload = json.loads(
        (
            ROOT / "artifacts/scene/type1-c-scaled2-place-up-robot-operation-plan.json"
        ).read_text()
    )
    structure, _ = build_structure_scene_plan_from_files(
        ROOT / "tasks/type1_scaled2/C/structure_start.json",
        ROOT / "tasks/type1_scaled2/C/structure_goal.json",
        payload,
    )
    before = copy.deepcopy(payload)
    bindings = top_offset_bindings(
        structure,
        parse_robot_operation_plan(payload),
        json.loads((ROOT / "config/lego_library.json").read_text()),
    )
    item = bindings["type1-c-scaled2-piper0-pick"]
    assert item.free_axis == 1 and item.free_sign == -1
    assert item.free_length_m == 0.016
    assert item.overlap_fraction == 0.5
    assert item.footprint_m == (0.032, 0.032)
    assert payload == before


def test_contact_is_downward_and_offset_survives_world_yaw_and_height():
    """Use brick-local empty-half geometry and measured elevated support height."""
    shared = document()
    sample = observed(shared)
    sample = replace(
        sample,
        object_pose=Pose((0.35, 0.06, 0.08), (0, 0, 2**-0.5, 2**-0.5)),
        aabb_min=(0.334, 0.044, 0.08),
        aabb_max=(0.366, 0.076, 0.1009),
    )
    ctx = context()
    original = pick_targets(
        sample.aabb_min, sample.aabb_max, ctx.base, ctx.provenance, ctx.params
    )
    targets, contact, proof = top_offset_targets(binding(), sample, ctx, original)
    assert contact.position_xyz == pytest.approx((0.358, 0.06, 0.09395))
    assert _quat_rotate_vector(contact.orientation_xyzw, (0, 0, 1)) == pytest.approx(
        (0, 0, -1)
    )
    assert targets.precontact_boundary.position == pytest.approx(contact.position_xyz)
    assert proof["orientation_branches"] == 2
    assert proof["approach_opening_m"] == 0.08
    assert proof["place_up_compatibility_confirmed"] is False
    verify_top_offset_arrival(sample, replace(sample, tcp_pose=contact), contact)
    rolled = Pose(
        contact.position_xyz,
        _quat_mul(contact.orientation_xyzw, (0.0, 0.0, 1.0, 0.0)),
    )
    verify_top_offset_arrival(sample, replace(sample, tcp_pose=rolled), contact)
    diagonal = Pose(
        contact.position_xyz,
        _quat_mul(
            contact.orientation_xyzw,
            (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)),
        ),
    )
    with pytest.raises(ValueError, match="closing axis"):
        verify_top_offset_arrival(sample, replace(sample, tcp_pose=diagonal), contact)
    tilt = 0.35
    tilted = Pose(
        contact.position_xyz,
        _quat_mul(
            (0.0, math.sin(tilt / 2), 0.0, math.cos(tilt / 2)),
            contact.orientation_xyzw,
        ),
    )
    with pytest.raises(ValueError, match="not downward"):
        verify_top_offset_arrival(sample, replace(sample, tcp_pose=tilted), contact)
    with pytest.raises(ValueError, match="brick moved"):
        verify_top_offset_arrival(
            sample,
            replace(
                sample,
                object_pose=Pose(
                    (0.36, 0.06, 0.08), sample.object_pose.orientation_xyzw
                ),
            ),
            contact,
        )


@pytest.mark.parametrize("push_brick", [False, True])
def test_pick_submits_only_offset_pose_and_retains_arrival_gate(push_brick):
    """Use strict arrival evidence and refuse early contact permission on a push."""
    shared = document()
    ledger = OperationStateLedger(("piper_0", "piper_1"))
    ledger.confirm(
        "piper_0", "robot_at_pick_region", {"physical_instance_id": "brick-a"}
    )
    phases, motions = [], []

    async def capture(participants):
        return shared, 1, {"piper_0_joint1": 0.0, "piper_1_joint1": 0.0}

    handler = PickHandler(
        Observer(shared),
        ledger,
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=Motion(),
        context=context(),
        top_offset_grasps={"pick-1": binding()},
        free_space_transport=SimpleNamespace(
            robot_ids=("piper_0", "piper_1"), free_space_context=capture
        ),
    )

    async def dispatch(item, phase, value):
        phases.append(phase)
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

    async def move(item, **kwargs):
        phase = kwargs["phase_id"]
        phases.append(phase)
        target = kwargs["targets"]["piper_0"]
        motions.append(target)
        assert "grasp_region" not in target
        assert target["hold_session_peers"] is True
        assert set(kwargs["targets"]) == {"piper_0"}
        assert "vertical_approach" not in target
        assert target["moveit_pose_goal"] is True
        tcp = target["tcp_pose"]
        base = Pose(context().base.position_xyz, context().base.orientation_xyzw)
        measured = compose(
            base, Pose(tuple(tcp["position"]), tuple(tcp["orientation_xyzw"]))
        )
        shared["robots"]["piper_0"]["tcp_pose"] = {
            "position_xyz": measured.position_xyz,
            "orientation_xyzw": measured.orientation_xyzw,
        }
        if push_brick and phase == "precontact_boundary":
            shared["bricks"]["brick-a"]["pose"]["position_xyz"][0] += 0.01
        return SimpleNamespace(
            planned=True,
            prepared=True,
            started=True,
            arrival_confirmed=True,
            terminal=True,
            safe_to_release=True,
            evidence={},
            motion_id=f"pick-1:{phase}",
            participants=("piper_0",),
            reason=None,
        )

    handler._dispatch = dispatch
    handler.execute_free_space_phase = move
    result = asyncio.run(
        handler.execute(
            replace(operation(), parameters={"successor_skill": "PLACE_UP"})
        )
    )
    if push_brick:
        assert phases == [
            "plan_grasp",
            "open_gripper",
            "precontact_boundary",
        ]
        assert "brick moved" in result.reason
        assert result.motion_sent is True
        assert ledger.owner_of_object("brick-a") is None
        return
    assert phases == [
        "plan_grasp",
        "open_gripper",
        "precontact_boundary",
        "allow_grasp_contact",
        "close_gripper",
    ], result.reason
    assert result.evidence["successor_grasp"]["mode"] == "vertical_offset"
    assert (
        result.evidence["lift_geometry"]["source"] == "measured_vertical_offset_grasp"
    )
    assert len(motions) == 1
    assert motions[0]["tcp_pose"]["position_tolerance_m"] == 0.002
    assert (
        motions[0]["tcp_pose"]["orientation_tolerance_rad"]
        == context().params.orientation_tolerance_rad
    )
    assert shared["robots"]["piper_0"]["opening_m"] == pytest.approx(0.08)
    assert ledger.owner_of_object("brick-a") is None


def test_vertical_trial_submits_two_axis_aligned_branches_and_holds_peer():
    """The unsent arm stays fixed; each OR'd branch keeps the closing aligned."""
    from rocobrick.coordinated_session import CoordinatedSessionRequest

    target = pose("piper_0", 0.1)
    target["hold_session_peers"] = True
    target["moveit_pose_goal"] = True
    item = CoordinatedSessionRequest(
        coordination_id="vertical-1",
        robots=("piper_0", "piper_1"),
        participants=("piper_0",),
        scene_digest="scene-1",
        planning_group="dual_arm",
        plan_only=True,
        timeout_s=30,
        targets={"piper_0": target},
        start_state={"piper_0_joint1": 0.0, "piper_1_joint1": 0.2},
    )
    current = runtime()
    goal = current._build_goal(item, Factory())
    branches = goal.request.goal_constraints
    assert len(branches) == 2
    orientations = []
    for branch in branches:
        assert [c.joint_name for c in branch.joint_constraints] == ["piper_1_joint1"]
        orientation = next(
            constraint
            for constraint in branch.orientation_constraints
            if constraint.link_name == "piper_0_rocobrick_tcp"
        )
        assert orientation.absolute_x_axis_tolerance == pytest.approx(0.015)
        assert orientation.absolute_y_axis_tolerance == pytest.approx(0.015)
        assert orientation.absolute_z_axis_tolerance == pytest.approx(0.015)
        orientations.append(orientation.orientation)
    assert (
        orientations[1].x,
        orientations[1].y,
        orientations[1].z,
        orientations[1].w,
    ) == pytest.approx(
        (orientations[0].y, -orientations[0].x, orientations[0].w, -orientations[0].z)
    )
    assert [c.joint_name for c in goal.request.path_constraints.joint_constraints] == [
        "piper_1_joint1"
    ]
    with pytest.raises(ValueError, match="whole-path joint hold"):
        current._validate_held_joints(
            item,
            ["piper_0_joint1", "piper_1_joint1"],
            [{"positions": [0, 0.2]}, {"positions": [0.1, 0.3]}],
        )


def test_carrier_pose_goal_stays_a_single_orientation_branch():
    """A brick carried off the tool axis must not offer the roll flip."""
    from rocobrick.coordinated_session import CoordinatedSessionRequest

    target = pose("piper_0", 0.1)
    target["hold_session_peers"] = True
    target["moveit_pose_goal"] = True
    target["roll_flip_equivalent"] = False
    target["tcp_pose"]["orientation_xyzw"] = (0.5, 0.5, 0.5, 0.5)
    peer = pose("piper_1", 0.2)
    peer["moveit_pose_goal"] = True
    item = CoordinatedSessionRequest(
        coordination_id="support-prephase-1",
        robots=("piper_0", "piper_1"),
        participants=("piper_0", "piper_1"),
        scene_digest="scene-1",
        planning_group="dual_arm",
        plan_only=True,
        timeout_s=30,
        targets={"piper_0": target, "piper_1": peer},
        start_state={"piper_0_joint1": 0.0, "piper_1_joint1": 0.0},
    )
    goal = runtime()._build_goal(item, Factory())
    branches = goal.request.goal_constraints
    assert len(branches) == 2
    for branch in branches:
        carrier = next(
            constraint
            for constraint in branch.orientation_constraints
            if constraint.link_name == "piper_0_rocobrick_tcp"
        )
        assert (
            carrier.orientation.x,
            carrier.orientation.y,
            carrier.orientation.z,
            carrier.orientation.w,
        ) == pytest.approx((0.5, 0.5, 0.5, 0.5))
    supports = [
        next(
            constraint.orientation
            for constraint in branch.orientation_constraints
            if constraint.link_name == "piper_1_rocobrick_tcp"
        )
        for branch in branches
    ]
    assert (
        supports[1].x,
        supports[1].y,
        supports[1].z,
        supports[1].w,
    ) == pytest.approx((supports[0].y, -supports[0].x, supports[0].w, -supports[0].z))
