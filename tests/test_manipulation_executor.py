"""Tests for the unified plan-and-execute manipulation boundary."""

from __future__ import annotations

import asyncio

import numpy as np
from scipy.spatial.transform import Rotation

from rocobrick.backends.base import RobotState
from rocobrick.execution import ActionStatus, ManipulationExecutor
from rocobrick.planning import (
    GraspRegion,
    GroundedAction,
    ObjectGeometry,
    SceneGeometry,
)
from rocobrick.skills import ManipulationAction, ManipulationSkillType


class _Robot:
    def __init__(self, robot_id: str, reachable: bool = True) -> None:
        self._robot_id = robot_id
        self._reachable = reachable
        self._q = np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.02, -0.02])
        self.commands: list[np.ndarray] = []

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def home_configuration(self) -> np.ndarray:
        return self._q.copy()

    @property
    def arm_configuration_indices(self) -> tuple[int, ...]:
        return (0, 1, 2, 3, 4, 5)

    @property
    def gripper_configuration_indices(self) -> tuple[int, ...]:
        return (6, 7)

    def read_state(self) -> RobotState:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(self._q[3:6]).as_matrix()
        pose[:3, 3] = self._q[:3]
        return RobotState(self._q.copy(), pose, self._q[6:].copy())

    def solve_ik(self, world_t_tcp: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        if not self._reachable:
            return None
        result = seed.copy()
        result[:3] = world_t_tcp[:3, 3]
        result[3:6] = Rotation.from_matrix(world_t_tcp[:3, :3]).as_rotvec()
        return result

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Return the test robot TCP pose.

        Returns:
            Pose encoded directly by the first six test joints.
        """
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        pose[:3, 3] = q[:3]
        return pose

    def command_configuration(self, q: np.ndarray) -> None:
        command = np.asarray(q, dtype=np.float64).copy()
        if abs(command[6]) < 0.016 and abs(command[7]) < 0.016:
            command[6], command[7] = 0.016, -0.016
        self._q = command
        self.commands.append(command.copy())

    def configuration_is_safe(self, q: np.ndarray) -> bool:
        return bool(np.isfinite(q).all())

    def with_gripper(
        self, q: np.ndarray, object_width: float, closed: bool
    ) -> np.ndarray:
        result = q.copy()
        opening = object_width * 0.5 + 0.006
        result[6] = 0.0 if closed else opening
        result[7] = 0.0 if closed else -opening
        return result


class _World:
    def __init__(self, robot: _Robot) -> None:
        self._robot = robot
        self._pose = np.eye(4)
        self._attached = False
        self._object_t_tcp = np.eye(4)

    def object_pose(self, object_id: str) -> np.ndarray:
        assert object_id == "brick"
        return self._pose.copy()

    async def advance(self, steps: int = 1) -> None:
        if not self._attached and self._robot.read_state().gripper_width <= 0.032001:
            self._attached = True
            self._object_t_tcp = (
                np.linalg.inv(self._pose) @ self._robot.read_state().tcp_world
            )
        if self._attached:
            self._pose = self._robot.read_state().tcp_world @ np.linalg.inv(
                self._object_t_tcp
            )


class _Grounder:
    def ground(self, action: ManipulationAction) -> GroundedAction:
        pose = np.eye(4)
        extents = np.array([0.008, 0.016, 0.006])
        target = ObjectGeometry(
            action.object_id,
            pose,
            np.eye(4),
            extents,
            (GraspRegion(np.eye(4), extents),),
        )
        return GroundedAction(SceneGeometry(target, ()))


def test_pick_uses_only_the_assigned_robot() -> None:
    """The executor never searches or falls back to another robot."""
    assigned = _Robot("assigned")
    other = _Robot("other")
    executor = ManipulationExecutor(
        {"assigned": assigned, "other": other}, _World(assigned), _Grounder()
    )
    action = ManipulationAction(
        "pick-1", ("assigned",), ManipulationSkillType.PICK, "brick"
    )
    result = asyncio.run(executor.execute(action))
    assert result.status is ActionStatus.SUCCESS
    assert result.held_by == "assigned"
    assert result.steps == result.metrics.simulation_steps
    assert result.metrics.waypoints > 0
    assert result.metrics.control_iterations > result.metrics.waypoints
    assert result.metrics.simulation_steps == result.metrics.control_iterations * 2
    assert assigned.commands
    assert other.commands == []


def test_planning_failure_sends_no_robot_command() -> None:
    """An unreachable assigned robot fails before any execution side effect."""
    assigned = _Robot("assigned", reachable=False)
    other = _Robot("other", reachable=True)
    executor = ManipulationExecutor(
        {"assigned": assigned, "other": other}, _World(assigned), _Grounder()
    )
    action = ManipulationAction(
        "pick-2", ("assigned",), ManipulationSkillType.PICK, "brick"
    )
    result = asyncio.run(executor.execute(action))
    assert result.status is ActionStatus.PLANNING_FAILED
    assert assigned.commands == []
    assert other.commands == []
