"""Tests for direction-parameterized Assemble composition."""

from __future__ import annotations

import asyncio

import numpy as np

from rocobrick.backends.base import RobotState
from rocobrick.controllers.motion import MotionConfig
from rocobrick.execution.skill_registry import SkillRegistry
from rocobrick.execution.types import ExecutionError, FailureCode, HeldObject
from rocobrick.safety import ForceGuard, PredicateSuccessCheck
from rocobrick.skills.assemble import AssembleRequest, AssembleSkill
from rocobrick.skills.base_skill import ManipulationSkillType


class _AssemblyRobot:
    def __init__(self, initial_z: float) -> None:
        self._q = np.array([0.02, -0.02, initial_z, 0.008, -0.008])
        self.commands: list[np.ndarray] = []
        self.release_started_after_success: list[bool] = []
        self.success_probe = lambda: False

    @property
    def robot_id(self) -> str:
        return "robot_1"

    @property
    def home_configuration(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.1, 0.02, -0.02])

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
        result = seed.copy()
        result[:3] = world_t_tcp[:3, 3]
        return result

    def command_configuration(self, q: np.ndarray) -> None:
        command = np.asarray(q, dtype=np.float64).copy()
        if abs(command[3]) < 0.008 and abs(command[4]) < 0.008:
            command[3] = 0.008
            command[4] = -0.008
        opening = abs(command[3]) + abs(command[4]) > 0.016001
        if opening:
            self.release_started_after_success.append(self.success_probe())
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


class _AssemblyWorld:
    def __init__(self, robot: _AssemblyRobot) -> None:
        self._robot = robot
        self._pose = robot.read_state().tcp_world.copy()
        self.events: list[str] = []

    def object_pose(self, object_id: str) -> np.ndarray:
        assert object_id == "brick_a"
        return self._pose.copy()

    async def advance(self, steps: int = 1) -> None:
        assert steps > 0
        if self._robot.read_state().gripper_width <= 0.016001:
            self._pose = self._robot.read_state().tcp_world.copy()


class _DirectionalSuccess:
    def __init__(
        self, robot: _AssemblyRobot, direction: float, threshold: float
    ) -> None:
        self._robot = robot
        self._direction = direction
        self._threshold = threshold
        self.ever_satisfied = False

    def is_satisfied(self) -> bool:
        z = float(self._robot.read_state().tcp_world[2, 3])
        reached = z * self._direction >= self._threshold * self._direction
        self.ever_satisfied |= reached
        return self.ever_satisfied

    def conflict(self) -> str | None:
        return None

    def require_satisfied(self, stage: str) -> None:
        if not self.is_satisfied():
            raise ExecutionError(
                FailureCode.VERIFICATION_FAILED, stage, "not connected"
            )


class _ConstantWrench:
    def __init__(self, force_z: float) -> None:
        self._wrench = np.array([0.0, 0.0, force_z, 0.0, 0.0, 0.0])

    def read_wrench_world(self) -> np.ndarray:
        return self._wrench.copy()


def _pose(z: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


def _held() -> HeldObject:
    return HeldObject("brick_a", "robot_1", np.eye(4), 0, 0.016)


def _motion_config() -> MotionConfig:
    return MotionConfig(
        position_tolerance=0.0002,
        rotation_tolerance=np.deg2rad(1.0),
        max_arm_step=0.01,
        timeout_steps=100,
        settle_steps=1,
        translation_step=0.004,
        translation_step_held=0.004,
        command_lead=0.01,
        command_lead_held=0.01,
        command_lead_ramp_steps=1,
    )


def _run_direction(direction: float) -> tuple[_AssemblyRobot, _DirectionalSuccess]:
    preassembly_z = -direction * 0.02
    robot = _AssemblyRobot(preassembly_z)
    world = _AssemblyWorld(robot)
    success = _DirectionalSuccess(robot, direction, direction * 0.001)
    robot.success_probe = success.is_satisfied
    result = asyncio.run(
        AssembleSkill.create(robot, world, _motion_config()).execute(
            AssembleRequest(
                held=_held(),
                world_t_preassembly_tcp=_pose(
                    preassembly_z, x=0.02, y=-0.02
                ),
                world_t_goal_tcp=_pose(0.0),
                insertion_direction_world=np.array([0.0, 0.0, direction]),
                success_check=success,
                approach_clearance=0.004,
                insertion_distance=0.008,
                insertion_step=0.001,
                retreat_distance=0.01,
                max_insert_steps=12,
            )
        )
    )
    assert result.steps > 0
    return robot, success


def test_phase_two_enables_both_parameterized_place_skills() -> None:
    """Place-Up and Place-Down share one available implementation phase."""
    registry = SkillRegistry.phase_two()
    for skill_type in ManipulationSkillType:
        expected = skill_type in {
            ManipulationSkillType.PICK,
            ManipulationSkillType.PLACE_DOWN,
            ManipulationSkillType.PLACE_UP,
        }
        assert registry.capability(skill_type).available is expected


def test_place_up_and_down_share_opposite_direction_execution() -> None:
    """The same skill inserts and retreats correctly for both axis signs."""
    for direction in (-1.0, 1.0):
        robot, success = _run_direction(direction)
        assert success.ever_satisfied
        assert robot.release_started_after_success
        assert all(robot.release_started_after_success)
        final_z = float(robot.read_state().tcp_world[2, 3])
        assert final_z * direction < 0.0


def test_unsatisfied_insert_never_releases_object() -> None:
    """Exhausted insertion cannot bypass semantic verification."""
    robot = _AssemblyRobot(-0.02)
    world = _AssemblyWorld(robot)
    success = PredicateSuccessCheck(lambda: False)
    try:
        asyncio.run(
            AssembleSkill.create(robot, world, _motion_config()).execute(
                AssembleRequest(
                    _held(),
                    _pose(-0.02, x=0.02, y=-0.02),
                    _pose(0.0),
                    np.array([0.0, 0.0, 1.0]),
                    success,
                    approach_clearance=0.004,
                    insertion_distance=0.002,
                    insertion_step=0.001,
                    max_insert_steps=2,
                )
            )
        )
    except ExecutionError as exc:
        assert exc.code is FailureCode.CONTACT_FAILED
    else:
        raise AssertionError("unverified insertion must fail")
    assert not robot.release_started_after_success
    assert robot.read_state().gripper_width <= 0.016001


def test_force_guard_stops_insert_before_release() -> None:
    """Excess axial force terminates contact motion while retaining grasp."""
    robot = _AssemblyRobot(-0.02)
    world = _AssemblyWorld(robot)
    try:
        asyncio.run(
            AssembleSkill.create(robot, world, _motion_config()).execute(
                AssembleRequest(
                    _held(),
                    _pose(-0.02, x=0.02, y=-0.02),
                    _pose(0.0),
                    np.array([0.0, 0.0, 1.0]),
                    PredicateSuccessCheck(lambda: False),
                    approach_clearance=0.004,
                    wrench_source=_ConstantWrench(9.0),
                    force_guard=ForceGuard(8.0, 6.0),
                )
            )
        )
    except ExecutionError as exc:
        assert exc.code is FailureCode.EXCESS_FORCE
    else:
        raise AssertionError("excess force must stop InsertPress")
    assert not robot.release_started_after_success
