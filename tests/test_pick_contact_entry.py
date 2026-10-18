"""Handler-owned entry safety; no ROS or simulator startup."""

import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_pick_grasp_region import Factory, endpoint, query, region, runtime
from test_pick_handler import document

from rocobrick.adapters.moveit.pick_entry import vertical_path
from rocobrick.clearance.fk import UrdfChain
from rocobrick.skills.observation import Pose
from rocobrick.skills.pick_contact import inspect_pick_contact, validate_pick_guard


def guard_scene():
    """Build a deterministic offline fixture.

    Returns:
        The fixture used for strict safety admission checks.
    """
    selected = endpoint().tcp_pose
    start = replace(
        selected,
        position_xyz=(*selected.position_xyz[:2], selected.position_xyz[2] + 0.003),
    )
    guard = dict(
        mode="pick_entry",
        robot_id="piper_0",
        session_id="session-1",
        manifest_sha256="a" * 64,
        region=region(),
        start=asdict(start),
        end=asdict(selected),
    )
    scene = document()
    scene["robots"]["piper_0"]["tcp_pose"] = asdict(start)
    scene["robots"]["piper_0"]["opening_m"] = region()["approach_opening_m"]
    scene["pad_object_contacts"] = {
        **{
            k: scene[k]
            for k in ("session_id", "manifest_sha256", "sequence", "sim_time_s")
        },
        "schema": "rocobrick-pad-object-contacts/v1",
        "source": "physx_contact_report",
        "contacts": [],
    }
    return guard, scene


def test_open_entry_is_not_contact_or_holding_proof():
    """Reject unsafe entry without dispatch or a planning-service fallback."""
    guard, scene = guard_scene()
    assert inspect_pick_contact(scene, guard)[0] is None
    scene["pad_object_contacts"]["contacts"] = [
        {"robot_id": "piper_0", "normal_force_n": 0.01}
    ]
    assert inspect_pick_contact(scene, guard)[0] == "pick_premature_contact"


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("brick", "pick_object_disturbed"),
        ("opening", "pick_opening_drift"),
        ("lateral", "pick_entry_tracking_error"),
        ("axial", "pick_entry_axial_bound"),
    ],
)
def test_entry_live_guards(defect, reason):
    """Reject unsafe entry without dispatch or a planning-service fallback."""
    guard, scene = guard_scene()
    if defect == "brick":
        scene["bricks"]["brick-a"]["pose"]["position_xyz"][0] += 0.002
    elif defect == "opening":
        scene["robots"]["piper_0"]["opening_m"] -= 0.002
    else:
        pose = scene["robots"]["piper_0"]["tcp_pose"]
        pose["position_xyz"] = list(pose["position_xyz"])
        # Both offsets exceed the live bounds (10 mm lateral, 3 mm above the
        # commanded start), while staying inside them for the servo residual.
        pose["position_xyz"][0 if defect == "lateral" else 2] += (
            0.02 if defect == "lateral" else 0.004
        )
    assert inspect_pick_contact(scene, guard)[0] == reason


def test_entry_requires_same_step_contact_sensor_even_with_zero_force():
    """Reject unsafe entry without dispatch or a planning-service fallback."""
    guard, scene = guard_scene()
    scene["pad_object_contacts"]["sequence"] -= 1
    with pytest.raises(ValueError, match="missing/stale"):
        inspect_pick_contact(scene, guard)


def test_final_entry_cannot_be_long_or_rotate():
    """Reject unsafe entry without dispatch or a planning-service fallback."""
    guard, _ = guard_scene()
    guard["start"]["position_xyz"] = [
        *guard["end"]["position_xyz"][:2],
        guard["end"]["position_xyz"][2] + 0.006,
    ]
    with pytest.raises(ValueError, match="bounded vertical"):
        validate_pick_guard(guard)


def model():
    """Build a deterministic offline fixture.

    Returns:
        The fixture used for strict safety admission checks.
    """
    text = '<robot name="test"><link name="root"/>'
    prior = "root"
    for i, axis in enumerate(("1 0 0", "0 1 0", "0 0 1") * 2, 1):
        link = "piper_0_rocobrick_tcp" if i == 6 else f"link{i}"
        kind = "prismatic" if i <= 3 else "revolute"
        text += (
            f'<link name="{link}"/><joint name="piper_0_joint{i}" type="{kind}">'
            f'<parent link="{prior}"/><child link="{link}"/><axis xyz="{axis}"/>'
            '<limit lower="-1" upper="1" effort="10" velocity="1"/></joint>'
        )
        prior = link
    return text + "</robot>"


def test_local_ik_certifies_commanded_joint_sequence_and_preserves_start():
    """Reject unsafe entry without dispatch or a planning-service fallback."""
    state = {f"piper_0_joint{i}": (0.2 if i == 3 else 0.0) for i in range(1, 7)}
    start, end = Pose((0, 0, 0.2), (0, 0, 0, 1)), Pose((0, 0, 0.197), (0, 0, 0, 1))
    result = vertical_path(model(), "piper_0", state, start, end)
    assert result["points"][0]["positions"] == list(state.values())
    chain = UrdfChain(model())
    for point in result["points"]:
        fk = chain.forward(dict(zip(result["joint_names"], point["positions"])))[
            "piper_0_rocobrick_tcp"
        ]
        assert np.linalg.norm(fk[:2, 3]) < 0.00005
        assert np.allclose(fk[:3, :3], np.eye(3), atol=0.0005)
    assert fk[2, 3] == pytest.approx(0.197, abs=0.00005)
    with pytest.raises(ValueError, match="deadline"):
        vertical_path(model(), "piper_0", state, start, end, expired=lambda: True)
    with pytest.raises(ValueError, match="disagrees"):
        vertical_path(
            model(), "piper_0", state, replace(start, position_xyz=(0, 0, 0.21)), end
        )
    with pytest.raises(ValueError, match="orientation left"):
        vertical_path(
            model(),
            "piper_0",
            state,
            start,
            replace(end, orientation_xyzw=(1, 0, 0, 0)),
        )


def test_endpoint_selection_marker_only_admitted_on_plan_only_lane():
    """Reject unsafe entry without dispatch or a planning-service fallback."""
    target = {
        "piper_0": {"grasp_region": region(), "pick_endpoint_selection_only": True}
    }
    item = replace(query(target), plan_only=True)
    goal = runtime()._build_goal(item, Factory())
    assert goal.request.pipeline_id == "rocobrick_region"
    with pytest.raises(ValueError, match="cannot carry"):
        runtime()._build_goal(replace(item, plan_only=False), Factory())


def test_whole_arm_world_gate_includes_other_bricks_and_all_colliders(monkeypatch):
    """Reject unsafe entry without dispatch or a planning-service fallback."""
    from rocobrick.clearance.hulls import box_hull
    from rocobrick.clearance.pick_entry import pick_entry_world_clearance

    hull = box_hull("stub", np.array([-0.01] * 3), np.array([0.01] * 3))
    monkeypatch.setattr(
        "rocobrick.clearance.pick_entry.load_hulls",
        lambda: SimpleNamespace(
            link_names=("base_link", "link3"), hull=lambda name: hull
        ),
    )
    description = (
        '<robot name="stub">'
        '<link name="world"/>'
        '<link name="piper_0_link3"/>'
        '<joint name="j" type="fixed">'
        '<parent link="world"/>'
        '<child link="piper_0_link3"/>'
        '<origin xyz="0 0 0.2"/>'
        "</joint>"
        "</robot>"
    )
    request = SimpleNamespace(
        targets={"piper_0": {}},
        start_state={},
        scene_snapshot={
            "bricks": {
                "other-brick": {
                    "aabb_min": [-0.02, -0.02, 0.18],
                    "aabb_max": [0.02, 0.02, 0.22],
                }
            },
            "environment_colliders": {},
        },
    )
    trajectory = {"joint_names": ["unused"], "points": [{"positions": [0.0]}]}
    with pytest.raises(ValueError, match="link3 vs other-brick"):
        pick_entry_world_clearance(description, request, trajectory)
    request.scene_snapshot["environment_colliders"] = request.scene_snapshot.pop(
        "bricks"
    )
    request.scene_snapshot["bricks"] = {}
    with pytest.raises(ValueError, match="link3 vs other-brick"):
        pick_entry_world_clearance(description, request, trajectory)


def test_local_path_preserves_self_collision_rejection(monkeypatch):
    """Do not trade away self-collision checking when removing MoveGroup."""
    from rocobrick.clearance.hulls import box_hull
    from rocobrick.clearance.pick_entry import pick_entry_world_clearance

    hull = box_hull("stub", np.array([-0.01] * 3), np.array([0.01] * 3))
    monkeypatch.setattr(
        "rocobrick.clearance.pick_entry.load_hulls",
        lambda: SimpleNamespace(
            link_names=("base_link", "link3"), hull=lambda name: hull
        ),
    )
    model_text = (
        '<robot name="stub"><link name="piper_0_base_link"/>'
        '<link name="piper_0_link3"/><joint name="j" type="fixed">'
        '<parent link="piper_0_base_link"/><child link="piper_0_link3"/>'
        "</joint></robot>"
    )
    request = SimpleNamespace(
        targets={"piper_0": {}},
        start_state={},
        scene_snapshot={
            "bricks": {"remote": {"aabb_min": [2, 2, 2], "aabb_max": [3, 3, 3]}},
            "environment_colliders": {},
        },
    )
    trajectory = {"joint_names": ["unused"], "points": [{"positions": [0.0]}]}
    with pytest.raises(ValueError, match="PICK_ENTRY_SELF_COLLISION"):
        pick_entry_world_clearance(model_text, request, trajectory)


@pytest.mark.parametrize("angle", [0.016, 0.03, 0.08])
def test_local_ik_holds_measured_orientation_not_ideal_hover(angle):
    """A nonideal measured start is valid; do not rotate back to a sampled goal."""
    state = {f"piper_0_joint{i}": 0.0 for i in range(1, 7)}
    state["piper_0_joint3"] = 0.2
    state["piper_0_joint5"] = angle
    measured = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
    start = Pose((0, 0, 0.2), measured)
    end = replace(start, position_xyz=(0, 0, 0.197))
    trajectory = vertical_path(model(), "piper_0", state, start, end)
    chain = UrdfChain(model())
    expected = chain.forward(state)["piper_0_rocobrick_tcp"][:3, :3]
    for point in trajectory["points"]:
        actual = chain.forward(dict(zip(trajectory["joint_names"], point["positions"])))
        assert np.allclose(
            actual["piper_0_rocobrick_tcp"][:3, :3], expected, atol=0.0005
        )


def test_measured_orientation_reference_does_not_disable_live_tracking_guard():
    """Track the measured start, while still rejecting drift during entry."""
    guard, scene = guard_scene()
    measured = (
        0.0,
        math.sin(math.pi / 4 + 0.03 / 2),
        0.0,
        math.cos(math.pi / 4 + 0.03 / 2),
    )
    for key in ("start", "end"):
        guard[key]["orientation_xyzw"] = measured
    scene["robots"]["piper_0"]["tcp_pose"]["orientation_xyzw"] = measured
    assert inspect_pick_contact(scene, guard)[0] is None
    scene["robots"]["piper_0"]["tcp_pose"]["orientation_xyzw"] = (
        0.0,
        math.sin(math.pi / 4 + 0.12 / 2),
        0.0,
        math.cos(math.pi / 4 + 0.12 / 2),
    )
    assert inspect_pick_contact(scene, guard)[0] == "pick_entry_tracking_error"


def test_measured_servo_residual_admits_the_live_arm_without_admitting_drift():
    """Admit the measured controller residual, not a 1 mm ideal pose.

    The 2026-10-18 headless run commanded a vertical descent from a hover that
    arrived 2.76 mm off, and the arm settled 4.1 mm / 10.3 mrad short of a
    requested pose.  Those numbers must pass the live guards; larger ones must
    not.
    """
    guard, scene = guard_scene()
    measured = (
        0.0,
        math.sin(math.pi / 4 + 0.0103 / 2),
        0.0,
        math.cos(math.pi / 4 + 0.0103 / 2),
    )
    start, end = (dict(guard[key]) for key in ("start", "end"))
    start["position_xyz"] = [end["position_xyz"][0] + 0.00276, *end["position_xyz"][1:]]
    start["position_xyz"][2] = end["position_xyz"][2] + 0.003
    for pose in (start, end):
        pose["orientation_xyzw"] = measured
    guard["start"], guard["end"] = start, end
    assert validate_pick_guard(guard)["mode"] == "pick_entry"
    scene["robots"]["piper_0"]["tcp_pose"] = {
        "position_xyz": [
            end["position_xyz"][0] + 0.00413,
            end["position_xyz"][1] + 0.00059,
            end["position_xyz"][2] + 0.00307,
        ],
        "orientation_xyzw": measured,
    }
    assert inspect_pick_contact(scene, guard)[0] is None
    # Beyond the correlated corridor the same guards still refuse.
    guard["end"] = {
        **end,
        "position_xyz": [start["position_xyz"][0] + 0.02, *start["position_xyz"][1:]],
    }
    with pytest.raises(ValueError, match="bounded vertical"):
        validate_pick_guard(guard)


@pytest.mark.parametrize("boundary_drift", [0.005, 0.06])
def test_handler_does_not_rebase_orientation_at_phase_boundary(boundary_drift):
    """Freeze measured hover through final entry, rejecting intervening drift."""
    import asyncio

    from test_pick_grasp_region import context, operation

    from rocobrick.skills.pick import PickHandler
    from rocobrick.skills.targets import pick_targets

    guard, scene = guard_scene()
    selected = endpoint().tcp_pose
    hover = replace(
        selected,
        position_xyz=(*selected.position_xyz[:2], selected.position_xyz[2] + 0.06),
    )
    scene["robots"]["piper_0"]["tcp_pose"] = asdict(hover)
    ctx = context()
    observed = endpoint()
    original = pick_targets(
        observed.aabb_min, observed.aabb_max, ctx.base, ctx.provenance, ctx.params
    ).precontact_boundary
    dispatched = []

    async def observe(robot):
        return scene

    async def dispatch(item, phase, value):
        dispatched.append(value.payload["guard"])
        scene["robots"]["piper_0"]["tcp_pose"] = dict(value.payload["guard"]["end"])
        scene["robots"]["piper_0"]["tcp_pose"]["orientation_xyzw"] = (
            0.0,
            math.sin(math.pi / 4 + boundary_drift / 2),
            0.0,
            math.cos(math.pi / 4 + boundary_drift / 2),
        )
        return SimpleNamespace(evidence={}, passed=True, terminal=True, safe=True)

    handler = SimpleNamespace(
        _observations=SimpleNamespace(observe=observe),
        _context=ctx,
        _dispatch=dispatch,
        _session_id="session-1",
        _manifest_sha256="a" * 64,
    )
    admission = SimpleNamespace(robot_id="piper_0", physical_instance_id="brick-a")
    evidence = {}
    task = PickHandler._enter_region_contact(
        handler,
        operation(),
        admission,
        guard["region"],
        selected,
        hover,
        original,
        evidence,
    )
    if boundary_drift > 0.03:
        with pytest.raises(ValueError, match="bounded vertical"):
            asyncio.run(task)
        assert len(dispatched) == 1
    else:
        asyncio.run(task)
        assert len(dispatched) == 2
        assert dispatched[1]["end"]["orientation_xyzw"] == hover.orientation_xyzw
    assert evidence["contact_orientation_reference"]["source"] == "measured_hover"
