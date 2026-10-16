"""Measured-object insertion feedback, without ROS or robot motion."""

import asyncio
import copy
import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_place_down_handler import context
from test_task_c_skill_continuation import FreeSpace, Scene, upward_operation

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.ros_moveit_runtime import RosMoveItRobotSession
from rocobrick.skills.contact import inspect_contact
from rocobrick.skills.observation import _pose
from rocobrick.skills.place_up import PlaceUpHandler, _native_press_ready
from rocobrick.skills.placement_geometry import compose, inverse
from rocobrick.skills.targets import SkillParams, _quat_mul
from rocobrick.unified_readiness import UnifiedDualArmReadiness


def setup_feedback(mode="follow"):
    """Create a physically scripted pair and its admitted grasp anchor.

    Returns:
        Handler, scene, motion, guard and operation-scoped admission.
    """
    scene = Scene()
    upper = _pose(scene.data["bricks"]["brick-upper"]["pose"], "upper")
    pair = next(iter(scene.data["placement_interfaces"]["pairs"].values()))
    desired = compose(upper, inverse(_pose(pair["lower_to_upper"], "interface")))
    current = replace(
        desired,
        position_xyz=(*desired.position_xyz[:2], desired.position_xyz[2] - 0.012),
    )
    grasp = _pose(
        {"position_xyz": [0, 0, 0.015], "orientation_xyzw": [0, 0, 0, 1]}, "grasp"
    )
    scene.data["bricks"]["brick-lower"]["pose"] = asdict(current)
    scene.data["robots"]["piper_0"]["tcp_pose"] = asdict(compose(current, grasp))
    if mode.startswith("compress"):
        pair.update(stud_path="/lower", hole_path="/upper", contact_force_n=0.0)
        scene.data["placement_interfaces"]["thresholds"] = {
            "enabled": True,
            "distance_tolerance": 0.001,
            "max_penetration": 0.005,
            "position_tolerance": 0.002,
            "z_angle_tolerance": 0.087,
            "yaw_tolerance": 0.087,
            "required_force": 1.0,
        }
    guard = {
        "robot_id": "piper_0",
        "support_robot_id": "piper_1",
        "session_id": "session-1",
        "manifest_sha256": "a" * 64,
        "upper_id": "brick-upper",
        "lower_id": "brick-lower",
        "supported_pose": asdict(upper),
        "desired_lower": asdict(desired),
        "held_transform": asdict(grasp),
        "supported_position_m": 0.002,
        "supported_angle_rad": math.radians(3),
        "grasp_position_m": 0.003,
        "grasp_angle_rad": math.radians(3),
        "force_limit_n": 8.0,
        "mode": "insert_up",
    }

    class FeedbackMotion:
        """Apply commanded poses, or model a brick that stops following."""

        def __init__(self):
            """Track dispatched targets and the scripted physical mode."""
            self.calls = []

        def run_motion_operation(self, robot, item):
            """Return terminal evidence after applying the scripted outcome.

            Returns:
                Synthetic action lifecycle with optional exact connection.
            """
            self.calls.append(item)
            target = item.payload["target"]
            tcp = _pose(
                {
                    "position_xyz": target["position"],
                    "orientation_xyzw": target["orientation_xyzw"],
                },
                "target",
            )
            scene.data["robots"][robot]["tcp_pose"] = asdict(tcp)
            inserting = "insert_up_step" in item.request_id
            if mode != "stall" or not inserting:
                lower = compose(tcp, inverse(grasp))
                if mode.startswith("compress"):
                    unclamped_z = lower.position_xyz[2]
                    contact_z = desired.position_xyz[2] - 0.0017
                    lower = replace(
                        lower,
                        position_xyz=(
                            *lower.position_xyz[:2],
                            min(unclamped_z, contact_z),
                        ),
                    )
                    force = -700 * max(0.0, unclamped_z - contact_z)
                    pair["contact_force_n"] = force
                    scene.data["placement_interfaces"]["diagnostics"] = [
                        {
                            "stud_path": "/lower",
                            "hole_path": "/upper",
                            "stud_interface": pair["stud_interface"],
                            "hole_interface": pair["hole_interface"],
                            "relative_distance": contact_z - lower.position_xyz[2],
                            "position_error": 0,
                            "tilt": 0,
                            "yaw_error": 0,
                            "projected_force": force,
                        }
                    ]
                if mode == "tilt" and inserting:
                    angle = math.radians(2)
                    lower = replace(
                        lower,
                        orientation_xyzw=_quat_mul(
                            (math.sin(angle / 2), 0, 0, math.cos(angle / 2)),
                            lower.orientation_xyzw,
                        ),
                    )
                scene.data["bricks"]["brick-lower"]["pose"] = asdict(lower)
            stop = None
            lower = _pose(scene.data["bricks"]["brick-lower"]["pose"], "lower")
            if (
                mode == "follow"
                and lower.position_xyz[2] >= desired.position_xyz[2] - 0.0005
            ) or (mode == "compress" and abs(pair.get("contact_force_n", 0)) >= 1.0):
                scene.data["bricks"]["brick-lower"]["pose"] = asdict(desired)
                scene.data["robots"][robot]["tcp_pose"] = asdict(
                    compose(desired, grasp)
                )
                for key in scene.data["connections"]:
                    scene.data["connections"][key] = True
                pair["actual_connection"] = {
                    "stud_interface": pair["stud_interface"],
                    "hole_interface": pair["hole_interface"],
                    "lower_to_upper": pair["lower_to_upper"],
                }
                stop = "connection_confirmed"
            return {
                "motion_sent": True,
                "result": {
                    "passed": True,
                    "execution": {
                        "terminal": True,
                        "safe_to_release": True,
                        "details": {"contact_stop_reason": stop},
                    },
                },
            }

    motion = FeedbackMotion()
    ctx = context()
    ctx = replace(ctx, params=replace(ctx.params, place_up_feedback_step_m=0.001))
    handler = PlaceUpHandler(
        scene,
        OperationStateLedger(("piper_0", "piper_1")),
        session_id="session-1",
        manifest_sha256="a" * 64,
        motion_transport=motion,
        free_space_transport=FreeSpace(scene),
        context=ctx,
        poll_interval_s=0,
    )
    admission = SimpleNamespace(
        robot_id="piper_0",
        support_robot_id="piper_1",
        upper_physical_instance_id="brick-upper",
        lower_physical_instance_id="brick-lower",
    )
    return handler, scene, motion, guard, admission


def test_feedback_advances_in_short_steps_and_requires_native_connection():
    """Endpoint motion alone is not a substitute for the connection proof."""
    handler, _, motion, guard, admission = setup_feedback()
    evidence = {}
    result, reason = asyncio.run(
        handler._feedback_insert(upward_operation(), admission, guard, evidence)
    )
    assert result.terminal and result.safe
    assert reason == "connection_confirmed"
    steps = evidence["insertion_feedback"]["steps"]
    insertion = [step for step in steps if step["phase"].startswith("insert_up_step")]
    assert len(insertion) > 2
    for step in insertion:
        assert (
            step["target_offset_m"] - step["before"]["seat_axial_error_m"] <= 0.00100001
        )
    assert motion.calls[0].request_id.endswith(":align_up_0")


def test_feedback_stops_when_tcp_advances_without_the_brick():
    """Two non-following steps stop before consuming the eight-mm old slip budget."""
    handler, _, motion, guard, admission = setup_feedback("stall")
    evidence = {}
    result, reason = asyncio.run(
        handler._feedback_insert(upward_operation(), admission, guard, evidence)
    )
    assert result.terminal and result.safe
    assert reason == "insertion_object_not_following"
    assert sum("insert_up_step" in item.request_id for item in motion.calls) == 2
    assert (
        evidence["insertion_feedback"]["last_observation"]["grasp_position_m"] < 0.003
    )


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("compress", "connection_confirmed"),
        ("compress_never", "insertion_compression_budget_exhausted"),
    ],
)
def test_contact_compression_is_small_bounded_and_requires_native_success(
    mode, expected
):
    """A contacted brick may load elastically, but is never released on force alone."""
    handler, _, _, guard, admission = setup_feedback(mode)
    evidence = {}
    _, reason = asyncio.run(
        handler._feedback_insert(upward_operation(), admission, guard, evidence)
    )
    assert reason == expected
    compression = [
        step
        for step in evidence["insertion_feedback"]["steps"]
        if step["bounded_compression"]
    ]
    assert compression
    assert all(
        step["target_offset_m"] - step["before"]["seat_axial_error_m"] <= 0.00025001
        for step in compression
    )


def test_feedback_stops_before_pushing_a_tilted_brick_again():
    """A two-degree seat error is not another authorized upward press."""
    handler, _, motion, guard, admission = setup_feedback("tilt")
    _, reason = asyncio.run(
        handler._feedback_insert(upward_operation(), admission, guard, {})
    )
    assert reason == "insertion_alignment_not_confirmed"
    assert sum("insert_up_step" in item.request_id for item in motion.calls) == 1


def test_support_force_target_requires_top_contact_not_a_first_touch():
    """Force-target stopping remains distinct from sustained support proof."""
    _, scene, _, _, _ = setup_feedback()
    guard = {
        "robot_id": "piper_1",
        "session_id": "session-1",
        "manifest_sha256": "a" * 64,
        "upper_id": "brick-upper",
        "supported_pose": scene.data["bricks"]["brick-upper"]["pose"],
        "supported_position_m": 0.002,
        "supported_angle_rad": 0.05236,
        "force_limit_n": 8.0,
        "target_force_n": 1.5,
        "mode": "support_top",
    }
    data = asyncio.run(scene.observe("piper_1"))
    contacts = data["pad_object_contacts"]["contacts"]
    relevant = [item for item in contacts if item["robot_id"] == "piper_1"]
    assert relevant
    for force, expected in [
        (0.12, None),
        (1.6, "support_contact_confirmed"),
        (8.1, "support_force_exceeded"),
    ]:
        for item in relevant:
            item["normal_force_n"] = force / len(relevant)
        reason, _ = inspect_contact(data, guard)
        assert reason == expected


def test_native_diagnostics_are_pair_filtered_and_missing_force_is_explicit():
    """The trace retains actual native rejection data without inventing a force."""
    _, scene, _, guard, _ = setup_feedback()
    data = asyncio.run(scene.observe("piper_0"))
    pair = next(iter(data["placement_interfaces"]["pairs"].values()))
    pair.update(stud_path="/lower", hole_path="/upper", contact_force_n=None)
    relevant = {
        "stud_path": "/lower",
        "hole_path": "/upper",
        "stud_interface": pair["stud_interface"],
        "hole_interface": pair["hole_interface"],
        "accepted": False,
        "projected_force": 0.5,
    }
    data["placement_interfaces"]["diagnostics"] = [
        relevant,
        {**relevant, "hole_path": "/unrelated"},
    ]
    data["placement_interfaces"]["thresholds"] = {"required_force": 1.0}
    _, measured = inspect_contact(data, guard)
    assert measured["native_assembly_diagnostics"] == [relevant]
    assert measured["native_assembly_thresholds"] == {"required_force": 1.0}
    assert measured["interface_force_available"] is False


def test_seat_error_is_not_world_z_aabb_overlap():
    """Even a negative AABB gap does not identify the commanded interface depth."""
    _, scene, _, guard, _ = setup_feedback()
    scene.data["bricks"]["brick-lower"]["pose"] = copy.deepcopy(guard["desired_lower"])
    scene.data["robots"]["piper_0"]["tcp_pose"] = asdict(
        compose(
            _pose(guard["desired_lower"], "seat"),
            _pose(guard["held_transform"], "grasp"),
        )
    )
    data = asyncio.run(scene.observe("piper_0"))
    data["bricks"]["brick-lower"]["aabb_max"][2] = (
        data["bricks"]["brick-upper"]["aabb_min"][2] + 0.0045
    )
    _, measured = inspect_contact(data, guard)
    assert measured["contact_gap_m"] == pytest.approx(-0.0045)
    assert measured["seat_axial_error_m"] == pytest.approx(0, abs=1e-12)
    assert measured["seat_lateral_error_m"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    "field,value",
    [
        ("relative_distance", 0.002),
        ("position_error", 0.003),
        ("tilt", 0.1),
        ("yaw_error", 0.1),
        ("tilt", float("nan")),
    ],
)
def test_compression_requires_native_geometric_admission(field, value):
    """Contact compliance never excuses a wrong or unmeasured interface."""
    measured = {
        "native_assembly_thresholds": {
            "enabled": True,
            "distance_tolerance": 0.001,
            "max_penetration": 0.005,
            "position_tolerance": 0.002,
            "z_angle_tolerance": 0.087,
            "yaw_tolerance": 0.087,
        },
        "native_assembly_diagnostics": [
            {
                "relative_distance": 0,
                "position_error": 0.0001,
                "tilt": 0.01,
                "yaw_error": 0,
            }
        ],
    }
    assert _native_press_ready(measured)
    measured["native_assembly_diagnostics"][0][field] = value
    assert not _native_press_ready(measured)
    assert not _native_press_ready({})


@pytest.mark.parametrize("step", [-0.001, 0.003, float("nan"), True])
def test_feedback_step_is_bounded_and_numeric(step):
    """Invalid feedback settings fail before runtime admission."""
    with pytest.raises(ValueError):
        SkillParams(place_up_feedback_step_m=step)


def test_unified_readiness_observes_state_convergence_without_waiving_gates():
    """Adjacent subscription ticks get a bounded resample, never a motion retry."""
    robots = ("piper_0", "piper_1")
    ready = UnifiedDualArmReadiness(
        robots, True, True, True, True, True, {robot: True for robot in robots}
    )
    reports = [replace(ready, combined_joint_state_complete=False), ready]
    clock = [0.0]
    calls = []

    def observe(request):
        calls.append(request)
        return reports.pop(0)

    session = SimpleNamespace(
        robot_ids=robots,
        unified_readiness=observe,
        _last_unified_timing={},
        _clock_fn=lambda: clock[0],
        _spin_fn=lambda: None,
        _sleep_fn=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    request = SimpleNamespace(
        robots=robots,
        timeout_s=2.0,
        participants=robots,
        coordination_id="state-sync",
        scene_digest="a" * 64,
        scene_snapshot={"sequence": 1, "sim_time_s": 0.0},
    )
    result = RosMoveItRobotSession.coordinated_readiness(session, request)
    assert len(calls) == 2
    assert result["components"]["combined_joint_state_complete"] is True
    reports[:] = [
        replace(ready, combined_joint_state_complete=False, scene_snapshot_fresh=False)
    ]
    calls.clear()
    with pytest.raises(RuntimeError, match="scene_snapshot_fresh"):
        RosMoveItRobotSession.coordinated_readiness(session, request)
    assert len(calls) == 1
