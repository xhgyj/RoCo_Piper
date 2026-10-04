"""Tests for splitting one combined MoveIt trajectory by robot."""

import pytest

from rocobrick.coordinated_moveit import CoordinatedMoveItError
from rocobrick.coordinated_moveit_adapter import coordinated_operation_from_joint_trajectory


def test_combined_trajectory_splits_into_two_robot_trajectories() -> None:
    """Every robot receives only its own prefixed arm joints."""
    names = ("piper_0_joint1", "piper_1_joint1")
    result = coordinated_operation_from_joint_trajectory(
        coordination_id="coord-1",
        scene_digest="scene-1",
        participants=("piper_0", "piper_1"),
        joint_names=names,
        points=[
            {"time_from_start_s": 0.0, "positions": [0.0, 0.1]},
            {"time_from_start_s": 1.0, "positions": [0.2, 0.3]},
        ],
        joint_binding={
            "piper_0": ("piper_0_joint1",),
            "piper_1": ("piper_1_joint1",),
        },
        start_at_sim_time_s=10.0,
    )
    assert result.trajectories[0].points[-1].positions == (0.2,)
    assert result.trajectories[1].points[-1].positions == (0.3,)


def test_combined_trajectory_rejects_missing_robot_joint() -> None:
    """A partial combined result cannot be executed as coordinated motion."""
    with pytest.raises(CoordinatedMoveItError, match="missing joints"):
        coordinated_operation_from_joint_trajectory(
            coordination_id="coord-1",
            scene_digest="scene-1",
            participants=("piper_0", "piper_1"),
            joint_names=("piper_0_joint1",),
            points=[
                {"time_from_start_s": 0.0, "positions": [0.0]},
                {"time_from_start_s": 1.0, "positions": [0.2]},
            ],
            joint_binding={
                "piper_0": ("piper_0_joint1",),
                "piper_1": ("piper_1_joint1",),
            },
            start_at_sim_time_s=10.0,
        )
