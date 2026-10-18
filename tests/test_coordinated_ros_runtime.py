"""Tests for combined MoveIt TCP pose goal construction."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from rocobrick.coordinated_ros_runtime import (
    CombinedMoveItPlanningError,
    CoordinatedMoveItPlanRuntime,
)
from rocobrick.coordinated_session import CoordinatedSessionRequest


class Factory:
    """Minimal fake MoveIt message factory for goal construction."""

    def move_group_goal_factory(self):
        return SimpleNamespace(request=None, planning_options=SimpleNamespace())

    def motion_plan_request_factory(self):
        return SimpleNamespace(
            goal_constraints=[],
            pipeline_id="",
            start_state=SimpleNamespace(
                joint_state=SimpleNamespace(name=[], position=[])
            ),
        )

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


def test_combined_pose_query_budget_and_single_and_goal_are_preserved():
    """Both TCP targets form one dual-arm goal within the remaining budget."""
    query = replace(
        request({"piper_0": pose("piper_0", 0.1), "piper_1": pose("piper_1", -0.1)}),
        timeout_s=3.0,
    )
    goal = runtime()._build_goal(query, Factory())
    assert goal.request.allowed_planning_time == 3.0
    assert len(goal.request.goal_constraints) == 1
    constraints = goal.request.goal_constraints[0]
    assert constraints.joint_constraints == []
    assert len(constraints.position_constraints) == 2
    assert len(constraints.orientation_constraints) == 2


def test_combined_runtime_preserves_internal_moveit_error_code(monkeypatch):
    """An actual action failure remains typed across the worker transport."""
    current = runtime()
    wrapper = SimpleNamespace(
        status=6,
        result=SimpleNamespace(
            error_code=SimpleNamespace(val=99999),
            planned_trajectory=SimpleNamespace(
                joint_trajectory=SimpleNamespace(joint_names=[], points=[])
            ),
        ),
    )
    handle = SimpleNamespace(
        accepted=True, get_result_async=lambda: SimpleNamespace(result=lambda: wrapper)
    )
    current._client = SimpleNamespace(
        send_goal_async=lambda _: SimpleNamespace(result=lambda: handle)
    )
    monkeypatch.setattr(current, "register", lambda: None)
    monkeypatch.setattr(current, "ready", lambda: True)
    monkeypatch.setattr(current, "_wait", lambda *_: True)
    factories = SimpleNamespace(
        goal_status_type=lambda: SimpleNamespace(
            STATUS_SUCCEEDED=4, STATUS_CANCELED=5, STATUS_ABORTED=6
        )
    )
    monkeypatch.setattr(current, "_resolve_factories", lambda: factories)
    monkeypatch.setattr(current, "_build_goal", lambda *_: None)
    with pytest.raises(CombinedMoveItPlanningError) as failure:
        current.plan(
            request({"piper_0": pose("piper_0", 0.0), "piper_1": pose("piper_1", 0.0)})
        )
    assert failure.value.moveit_error_code == 99999
    assert current._goal_handle is None and current._result_future is None


def test_build_goal_rejects_missing_geometry_target():
    with pytest.raises(ValueError, match="exact tcp_pose or joint_positions"):
        runtime()._build_goal(
            request(
                {"piper_0": {"kind": "pre_support"}, "piper_1": pose("piper_1", 0.0)}
            ),
            Factory(),
        )


def test_joint_goal_tolerance_is_carried_into_actual_planner_constraints():
    """The planner cannot aim at a looser endpoint than HANDOVER admission."""
    targets = {
        r: {"joint_positions": {r + "_joint1": 0.1}, "joint_goal_tolerance_rad": 0.002}
        for r in ("piper_0", "piper_1")
    }
    goal = runtime()._build_goal(request(targets), Factory())
    assert all(
        c.tolerance_above == 0.002 and c.tolerance_below == 0.002
        for c in goal.request.goal_constraints[0].joint_constraints
    )


@pytest.mark.parametrize("invalid", [True, 0, 0.02, float("nan")])
def test_invalid_joint_goal_tolerance_is_rejected(invalid):
    """Unknown, zero, nonfinite or widened goal tolerances are refused."""
    targets = {
        r: {
            "joint_positions": {r + "_joint1": 0.1},
            "joint_goal_tolerance_rad": invalid,
        }
        for r in ("piper_0", "piper_1")
    }
    expected = (
        "JSON-safe"
        if isinstance(invalid, float) and invalid != invalid
        else "joint goal tolerance"
    )
    with pytest.raises(ValueError, match=expected):
        runtime()._build_goal(request(targets), Factory())


def held_request():
    """Return one moving arm and one explicitly measured whole-path hold."""
    return replace(
        request(
            {
                "piper_0": {
                    "maintain_pose": True,
                    "joint_positions": {"piper_0_joint1": 0.2},
                },
                "piper_1": pose("piper_1", 0.0),
            }
        ),
        start_state={"piper_0_joint1": 0.2, "piper_1_joint1": 0.0},
    )


def test_maintained_arm_has_real_path_constraints_not_only_endpoint_goal():
    """A giver held during final approach cannot wander between endpoints."""
    goal = runtime()._build_goal(held_request(), Factory())
    path = goal.request.path_constraints.joint_constraints
    assert len(path) == 1
    assert path[0].joint_name == "piper_0_joint1"
    assert path[0].position == 0.2
    assert path[0].tolerance_above == 0.002
    final = goal.request.goal_constraints[0].joint_constraints[0]
    assert final is not path[0]
    assert final.position == path[0].position


def test_maintained_arm_rejects_a_different_goal_and_missing_measured_start():
    """Maintaining pose is not permission to replace the live initial state."""
    item = held_request()
    with pytest.raises(ValueError, match="different joint goal"):
        runtime()._build_goal(
            replace(
                item,
                targets={
                    **item.targets,
                    "piper_0": {
                        "maintain_pose": True,
                        "joint_positions": {"piper_0_joint1": 0.3},
                    },
                },
            ),
            Factory(),
        )
    with pytest.raises(ValueError, match="measured start"):
        runtime()._build_goal(replace(item, start_state={}), Factory())


def test_trajectory_hold_is_checked_at_interior_waypoints():
    """Matching first/last points cannot conceal an interior giver excursion."""
    names = ("piper_0_joint1", "piper_1_joint1")
    points = [
        {"positions": (0.2, 0.0)},
        {"positions": (0.25, 0.1)},
        {"positions": (0.2, 0.2)},
    ]
    with pytest.raises(ValueError, match="whole-path joint hold"):
        runtime()._validate_held_joints(held_request(), names, points)
    points[1] = {"positions": (0.201, 0.1)}
    runtime()._validate_held_joints(held_request(), names, points)


def test_non_region_goal_keeps_default_pipeline():
    """Only region requests opt into the mandatory binding planner."""
    goal = runtime()._build_goal(
        request({"piper_0": pose("piper_0", 0.1), "piper_1": pose("piper_1", 0.2)}),
        Factory(),
    )
    assert goal.request.pipeline_id == ""
