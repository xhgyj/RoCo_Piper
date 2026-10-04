"""Strict contracts for coordinated dual-arm MoveIt planning/execution.

This module is transport-free. It prevents a pair of independent robot
operations from being mistaken for one coordinated trajectory: every
participant, joint name, timestamp, scene digest and terminal result must be
explicitly present and robot-scoped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


class CoordinatedMoveItError(ValueError):
    """A coordinated operation violates its frozen contract."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CoordinatedMoveItError(f"{label} must be a non-empty string")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoordinatedMoveItError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise CoordinatedMoveItError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class CoordinatedTrajectoryPoint:
    """One joint waypoint on the shared coordinated time axis."""

    time_from_start_s: float
    positions: tuple[float, ...]

    def __post_init__(self) -> None:
        time_value = _finite(self.time_from_start_s, "time_from_start_s")
        if time_value < 0:
            raise CoordinatedMoveItError("time_from_start_s must be non-negative")
        if not self.positions:
            raise CoordinatedMoveItError("positions must not be empty")
        positions = tuple(_finite(value, "positions[]") for value in self.positions)
        object.__setattr__(self, "time_from_start_s", time_value)
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class CoordinatedRobotTrajectory:
    """One participant's complete arm trajectory."""

    robot_id: str
    joint_names: tuple[str, ...]
    points: tuple[CoordinatedTrajectoryPoint, ...]

    def __post_init__(self) -> None:
        robot = _text(self.robot_id, "robot_id")
        joints = tuple(_text(name, "joint_names[]") for name in self.joint_names)
        if not joints or len(joints) != len(set(joints)):
            raise CoordinatedMoveItError("joint_names must be unique and non-empty")
        points = tuple(self.points)
        if len(points) < 2:
            raise CoordinatedMoveItError("trajectory requires at least two points")
        if any(len(point.positions) != len(joints) for point in points):
            raise CoordinatedMoveItError("point position length does not match joints")
        times = tuple(point.time_from_start_s for point in points)
        if times[0] != 0.0 or any(right <= left for left, right in zip(times, times[1:])):
            raise CoordinatedMoveItError("trajectory times must start at zero and increase")
        object.__setattr__(self, "robot_id", robot)
        object.__setattr__(self, "joint_names", joints)
        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class CoordinatedOperation:
    """A prepared multi-robot trajectory sharing one scene and clock."""

    coordination_id: str
    scene_digest: str
    participants: tuple[str, ...]
    trajectories: tuple[CoordinatedRobotTrajectory, ...]
    start_at_sim_time_s: float
    synchronization_tolerance_s: float = 0.02

    def __post_init__(self) -> None:
        coordination = _text(self.coordination_id, "coordination_id")
        digest = _text(self.scene_digest, "scene_digest")
        participants = tuple(_text(robot, "participants[]") for robot in self.participants)
        if not participants or len(participants) != len(set(participants)):
            raise CoordinatedMoveItError("participants must be unique and non-empty")
        trajectories = tuple(self.trajectories)
        if {item.robot_id for item in trajectories} != set(participants):
            raise CoordinatedMoveItError("one trajectory is required for every participant")
        start = _finite(self.start_at_sim_time_s, "start_at_sim_time_s")
        tolerance = _finite(self.synchronization_tolerance_s, "synchronization_tolerance_s")
        if start < 0 or tolerance <= 0:
            raise CoordinatedMoveItError("start time/tolerance is out of bounds")
        joint_names = [name for item in trajectories for name in item.joint_names]
        if len(joint_names) != len(set(joint_names)):
            raise CoordinatedMoveItError("participant joint names must be globally unique")
        object.__setattr__(self, "coordination_id", coordination)
        object.__setattr__(self, "scene_digest", digest)
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "trajectories", trajectories)
        object.__setattr__(self, "start_at_sim_time_s", start)
        object.__setattr__(self, "synchronization_tolerance_s", tolerance)


def validate_joint_binding(
    trajectory: CoordinatedRobotTrajectory,
    expected_joint_names: Sequence[str],
) -> None:
    """Require one trajectory to name exactly its robot's expected joints."""
    expected = tuple(_text(name, "expected_joint_names[]") for name in expected_joint_names)
    if set(trajectory.joint_names) != set(expected) or len(trajectory.joint_names) != len(expected):
        raise CoordinatedMoveItError(
            f"{trajectory.robot_id} trajectory joint binding does not match expected joints"
        )
