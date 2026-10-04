"""Tests for the plan-only dual-arm coordinator."""

import pytest

from rocobrick.coordinated_moveit import CoordinatedMoveItError
from rocobrick.coordinated_session import CoordinatedSessionRequest
from rocobrick.dual_arm_coordinator import DualArmPlanOnlyCoordinator


def request() -> CoordinatedSessionRequest:
    """Build one combined planning-only request."""
    return CoordinatedSessionRequest(
        coordination_id="coord-1",
        robots=("piper_0", "piper_1"),
        scene_digest="scene-1",
        planning_group="dual_arm",
        plan_only=True,
        timeout_s=30.0,
        targets={
            "piper_0": {"kind": "pre_support", "start_at_sim_time_s": 100.0},
            "piper_1": {"kind": "pre_place", "start_at_sim_time_s": 100.0},
        },
    )


def readiness() -> dict:
    """Build one passing group readiness document."""
    return {
        "coordination_id": "coord-1",
        "scene_digest": "scene-1",
        "scene_sequence": 10,
        "scene_sim_time_s": 100.0,
        "participants": ["piper_0", "piper_1"],
        "move_group_action": "/dual_arm/move_action",
        "planning_group": "dual_arm",
        "components": {"scene": True, "move_group": True},
    }


def coordinator() -> DualArmPlanOnlyCoordinator:
    """Build the exact two-arm joint binding."""
    return DualArmPlanOnlyCoordinator(
        {
            "piper_0": tuple(f"piper_0_joint{index}" for index in range(1, 7)),
            "piper_1": tuple(f"piper_1_joint{index}" for index in range(1, 7)),
        }
    )


def points() -> list[dict]:
    """Build a complete combined two-arm joint trajectory."""
    names = [
        *(f"piper_0_joint{index}" for index in range(1, 7)),
        *(f"piper_1_joint{index}" for index in range(1, 7)),
    ]
    return [
        {"time_from_start_s": 0.0, "positions": [0.0] * len(names)},
        {"time_from_start_s": 1.0, "positions": [0.1] * len(names)},
    ], names


def test_plan_only_coordinator_admits_readiness_and_plan() -> None:
    """Readiness and a full combined trajectory produce a no-motion object."""
    item = coordinator()
    item.admit_readiness(readiness(), request())
    trajectory, names = points()
    operation = item.admit_plan(request(), joint_names=names, points=trajectory)
    assert operation.coordination_id == "coord-1"
    assert [part.robot_id for part in operation.trajectories] == ["piper_0", "piper_1"]


def test_plan_only_coordinator_rejects_foreign_joint() -> None:
    """A foreign joint cannot enter the combined plan admission."""
    item = coordinator()
    trajectory, names = points()
    names[-1] = "piper_2_joint6"
    with pytest.raises(CoordinatedMoveItError, match="missing joints"):
        item.admit_plan(request(), joint_names=names, points=trajectory)
