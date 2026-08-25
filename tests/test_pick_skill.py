"""Tests for backend-independent Pick composition."""

from __future__ import annotations

import asyncio

import numpy as np

from rocobrick.backends.base import RobotState
from rocobrick.controllers.motion import (
    CartesianController,
    IKController,
    MotionConfig,
    TrajectoryController,
)
from rocobrick.execution.skill_registry import SkillRegistry
from rocobrick.execution.types import ExecutionError, FailureCode
from rocobrick.primitives.base import PrimitiveContext
from rocobrick.primitives.gripper import Grasp
from rocobrick.safety.checks import CollisionCheck, GraspStabilityCheck
from rocobrick.skills.base_skill import ManipulationAction, ManipulationSkillType
from rocobrick.skills.pick import GraspCandidate, PickRequest, PickSkill


class _FakeRobot:
    def __init__(self) -> None:
        self._q = np.array([0.0, 0.0, 0.10, 0.02, -0.02])
        self.commands: list[np.ndarray] = []
        self.requested_commands: list[np.ndarray] = []
        self.unreachable_x = 0.5
        self.deflect_during_grasp = False

    @property
    def robot_id(self) -> str:
        return "robot_1"

    @property
    def home_configuration(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.10, 0.02, -0.02])

    @property
    def arm_configuration_indices(self) -> tuple[int, ...]:
        return (0, 1, 2)

    @property
    def gripper_configuration_indices(self) -> tuple[int, ...]:
        return (3, 4)

    def read_state(self) -> RobotState:
        tcp = np.eye(4)
        tcp[:3, 3] = self._q[:3]
        return RobotState(self._q.copy(), tcp, self._q[3:].copy())

    def solve_ik(
        self, world_t_tcp: np.ndarray, seed: np.ndarray
    ) -> np.ndarray | None:
        if float(world_t_tcp[0, 3]) > self.unreachable_x:
            return None
        result = seed.copy()
        result[:3] = world_t_tcp[:3, 3]
        return result

    def command_configuration(self, q: np.ndarray) -> None:
        command = np.asarray(q, dtype=np.float64).copy()
        self.requested_commands.append(command.copy())
        # Simulate rigid finger contact on a 16 mm object.
        if abs(command[3]) < 0.008 and abs(command[4]) < 0.008:
            command[3] = 0.008
            command[4] = -0.008
            if self.deflect_during_grasp:
                command[0] += 0.01
        self._q = command
        self.commands.append(command.copy())

    def configuration_is_safe(self, q: np.ndarray) -> bool:
        return bool(np.isfinite(q).all() and np.max(np.abs(q)) <= 1.0)

    def with_gripper(
        self, q: np.ndarray, object_width: float, closed: bool
    ) -> np.ndarray:
        result = q.copy()
        opening = object_width * 0.5 + 0.006
        result[3] = 0.0 if closed else opening
        result[4] = 0.0 if closed else -opening
        return result


class _FakeWorld:
    def __init__(self, robot: _FakeRobot, follow: bool = True) -> None:
        self._robot = robot
        self._follow = follow
        self._pose = np.eye(4)
        self._attached = False
        self._object_t_tcp = np.eye(4)

    def object_pose(self, object_id: str) -> np.ndarray:
        assert object_id == "brick_a"
        return self._pose.copy()

    async def advance(self, steps: int = 1) -> None:
        assert steps > 0
        state = self._robot.read_state()
        if not self._attached and state.gripper_width <= 0.016001:
            self._attached = True
            self._object_t_tcp = np.linalg.inv(self._pose) @ state.tcp_world
        if self._attached and self._follow:
            self._pose = state.tcp_world @ np.linalg.inv(self._object_t_tcp)


def _pose(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


def _candidate(x: float = 0.0) -> GraspCandidate:
    return GraspCandidate(
        world_t_pregrasp_tcp=_pose(x, 0.0, 0.06),
        world_t_grasp_tcp=_pose(x, 0.0, 0.0),
        grasp_axis=0,
        grasp_width=0.016,
    )


def test_registry_exposes_six_skills_with_only_pick_available() -> None:
    """Phase one declares the complete vocabulary without fake implementations."""
    registry = SkillRegistry.phase_one()
    assert len(ManipulationSkillType) == 6
    registry.require_available(ManipulationSkillType.PICK)
    for skill_type in ManipulationSkillType:
        expected = skill_type is ManipulationSkillType.PICK
        assert registry.capability(skill_type).available is expected


def test_handover_requires_two_assigned_robots() -> None:
    """Only the coordinated skill reserves two robot resources."""
    ManipulationAction(
        ("giver", "receiver"), ManipulationSkillType.HANDOVER, "brick_a"
    )
    try:
        ManipulationAction(("giver",), ManipulationSkillType.HANDOVER, "brick_a")
    except ValueError as exc:
        assert "requires 2" in str(exc)
    else:
        raise AssertionError("single-robot Handover must be rejected")


def test_pick_returns_lifted_held_object() -> None:
    """Pick composes public primitives and stops above the source object."""
    robot = _FakeRobot()
    world = _FakeWorld(robot)
    result = asyncio.run(
        PickSkill.create(robot, world).execute(
            PickRequest("brick_a", (_candidate(),))
        )
    )
    assert result.held.object_id == "brick_a"
    assert result.held.robot_id == "robot_1"
    assert result.held.grasp_axis == 0
    assert result.lifted_object_pose[2, 3] >= 0.05
    assert result.steps > 0


def test_pick_skips_unreachable_candidate_before_motion() -> None:
    """Grounded candidates are probed before the selected grasp executes."""
    robot = _FakeRobot()
    world = _FakeWorld(robot)
    result = asyncio.run(
        PickSkill.create(robot, world).execute(
            PickRequest("brick_a", (_candidate(0.8), _candidate(0.0)))
        )
    )
    assert result.held.object_id == "brick_a"
    assert all(float(command[0]) < 0.5 for command in robot.commands)


def test_pick_reports_object_that_does_not_follow_lift() -> None:
    """A stationary object cannot satisfy the Pick postcondition."""
    robot = _FakeRobot()
    world = _FakeWorld(robot, follow=False)
    try:
        asyncio.run(
            PickSkill.create(robot, world).execute(
                PickRequest("brick_a", (_candidate(),))
            )
        )
    except ExecutionError as exc:
        assert exc.code is FailureCode.SLIPPED
    else:
        raise AssertionError("Pick must reject an object that does not lift")


def test_grasp_holds_captured_arm_pose_during_finger_contact() -> None:
    """Finger actuation rejects contact-induced arm drift in its command state."""
    robot = _FakeRobot()
    robot._q[0] = 0.1
    robot.deflect_during_grasp = True
    world = _FakeWorld(robot)
    config = MotionConfig()
    collision = CollisionCheck(robot)
    grasp_check = GraspStabilityCheck(robot, world)
    ik = IKController(robot, collision)
    context = PrimitiveContext(
        robot,
        world,
        TrajectoryController(robot, world, collision, config),
        CartesianController(
            robot, world, ik, collision, grasp_check, config
        ),
        collision,
    )
    asyncio.run(Grasp(context).execute(0.016))
    close_commands = [
        command
        for command in robot.requested_commands
        if abs(command[3]) < 0.008 and abs(command[4]) < 0.008
    ]
    assert close_commands
    assert all(float(command[0]) == 0.1 for command in close_commands)
