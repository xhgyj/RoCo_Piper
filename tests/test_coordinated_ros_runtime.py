"""Tests for combined MoveIt TCP pose goal construction."""

from types import SimpleNamespace

import pytest

from rocobrick.coordinated_ros_runtime import CoordinatedMoveItPlanRuntime
from rocobrick.coordinated_session import CoordinatedSessionRequest


class Factory:
    """Minimal fake MoveIt message factory for goal construction."""

    def move_group_goal_factory(self):
        return SimpleNamespace(request=None, planning_options=SimpleNamespace())

    def motion_plan_request_factory(self):
        return SimpleNamespace(goal_constraints=[])

    def constraints_factory(self):
        return SimpleNamespace(
            joint_constraints=[], position_constraints=[], orientation_constraints=[]
        )

    def joint_constraint_factory(self):
        return SimpleNamespace(
            joint_name="",
            position=0.0,
            tolerance_above=0.0,
            tolerance_below=0.0,
            weight=0.0,
        )

    def position_constraint_factory(self):
        return SimpleNamespace(
            header=SimpleNamespace(frame_id=""),
            link_name="",
            constraint_region=SimpleNamespace(primitives=[], primitive_poses=[]),
            weight=0.0,
        )

    def orientation_constraint_factory(self):
        return SimpleNamespace(
            header=SimpleNamespace(frame_id=""),
            link_name="",
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
            absolute_x_axis_tolerance=0.0,
            absolute_y_axis_tolerance=0.0,
            absolute_z_axis_tolerance=0.0,
            weight=0.0,
        )

    def solid_primitive_factory(self):
        return SimpleNamespace(SPHERE=2, type=0, dimensions=[])

    def pose_factory(self):
        return SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )


def request(targets):
    return CoordinatedSessionRequest(
        coordination_id="coord-1",
        robots=("piper_0", "piper_1"),
        scene_digest="scene-1",
        planning_group="dual_arm",
        plan_only=True,
        timeout_s=30.0,
        targets=targets,
    )


def runtime():
    return CoordinatedMoveItPlanRuntime(
        object(),
        joint_binding={"piper_0": ("piper_0_joint1",), "piper_1": ("piper_1_joint1",)},
        factories=Factory(),
        clock=lambda: 0.0,
        spin=lambda: None,
        sleep=lambda _: None,
    )


def pose(robot, x):
    return {
        "tcp_pose": {
            "position": (x, 0.2, 0.3),
            "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
            "tip_link": f"{robot}_rocobrick_tcp",
            "frame_id": f"{robot}_base_link",
            "runtime_model_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "position_tolerance_m": 0.005,
            "orientation_tolerance_rad": 0.015,
        }
    }


def test_build_goal_uses_prefixed_tcp_constraints_for_both_arms():
    goal = runtime()._build_goal(
        request({"piper_0": pose("piper_0", 0.1), "piper_1": pose("piper_1", -0.1)}),
        Factory(),
    )
    assert goal.request.group_name == "dual_arm"
    constraints = goal.request.goal_constraints[0]
    assert [item.link_name for item in constraints.position_constraints] == [
        "piper_0_rocobrick_tcp",
        "piper_1_rocobrick_tcp",
    ]
    assert [item.header.frame_id for item in constraints.orientation_constraints] == [
        "piper_0_base_link",
        "piper_1_base_link",
    ]
    assert goal.planning_options.plan_only is True


def test_build_goal_rejects_missing_geometry_target():
    with pytest.raises(ValueError, match="exact tcp_pose or joint_positions"):
        runtime()._build_goal(
            request({"piper_0": {"kind": "pre_support"}, "piper_1": pose("piper_1", 0.0)}),
            Factory(),
        )
