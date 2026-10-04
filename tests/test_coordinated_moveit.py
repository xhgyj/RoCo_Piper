"""Offline contract tests for coordinated dual-arm trajectories."""

import pytest

from rocobrick.coordinated_moveit import (
    CoordinatedMoveItError,
    CoordinatedOperation,
    CoordinatedRobotTrajectory,
    CoordinatedTrajectoryPoint,
    validate_joint_binding,
)


def trajectory(robot: str, names: tuple[str, ...]) -> CoordinatedRobotTrajectory:
    """Build one valid two-point participant trajectory."""
    return CoordinatedRobotTrajectory(
        robot,
        names,
        (
            CoordinatedTrajectoryPoint(0.0, tuple(0.0 for _ in names)),
            CoordinatedTrajectoryPoint(1.0, tuple(0.1 for _ in names)),
        ),
    )


def test_coordinated_operation_requires_all_participants_and_unique_joints() -> None:
    """Both arms must be represented on one shared time axis."""
    operation = CoordinatedOperation(
        "coord-1",
        "scene-digest",
        ("piper_0", "piper_1"),
        (
            trajectory("piper_0", ("piper_0_joint1", "piper_0_joint2")),
            trajectory("piper_1", ("piper_1_joint1", "piper_1_joint2")),
        ),
        10.0,
    )
    assert operation.participants == ("piper_0", "piper_1")


def test_coordinated_operation_rejects_missing_participant() -> None:
    """A single-arm trajectory cannot masquerade as a coordinated operation."""
    with pytest.raises(CoordinatedMoveItError, match="one trajectory"):
        CoordinatedOperation(
            "coord-1",
            "scene-digest",
            ("piper_0", "piper_1"),
            (trajectory("piper_0", ("piper_0_joint1",)),),
            10.0,
        )


def test_coordinated_operation_rejects_non_monotonic_time() -> None:
    """The shared trajectory clock must be strictly increasing."""
    with pytest.raises(CoordinatedMoveItError, match="increase"):
        CoordinatedRobotTrajectory(
            "piper_0",
            ("piper_0_joint1",),
            (
                CoordinatedTrajectoryPoint(0.0, (0.0,)),
                CoordinatedTrajectoryPoint(0.0, (0.1,)),
            ),
        )


def test_joint_binding_rejects_foreign_or_missing_joints() -> None:
    """Foreign robot joints cannot enter a participant trajectory."""
    item = trajectory("piper_0", ("piper_0_joint1", "piper_0_joint2"))
    validate_joint_binding(item, ("piper_0_joint2", "piper_0_joint1"))
    with pytest.raises(CoordinatedMoveItError, match="joint binding"):
        validate_joint_binding(item, ("piper_0_joint1", "piper_0_joint3"))
