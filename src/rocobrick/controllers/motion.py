"""Joint and Cartesian controllers used by shared motion primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from rocobrick.backends.base import RobotBackend, WorldModel
from rocobrick.execution.types import ExecutionError, FailureCode, OperationResult
from rocobrick.safety.checks import (
    CollisionCheck,
    GraspStabilityCheck,
    GraspTracking,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MotionConfig:
    """Free-space motion limits retained from the original expert."""

    position_tolerance: float = 0.005
    rotation_tolerance: float = np.deg2rad(5.0)
    joint_tolerance: float = 0.04
    max_arm_step: float = 0.02
    max_gripper_step: float = 0.001
    timeout_steps: int = 600
    settle_steps: int = 3
    translation_step: float = 0.003
    translation_step_held: float = 0.002
    command_lead: float = 0.008
    command_lead_held: float = 0.004
    command_lead_ramp_steps: int = 15


def pose_error(actual: FloatArray, target: FloatArray) -> tuple[float, float]:
    """Return translation and rotation error between two transforms."""
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    rotation = float(
        Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude()
    )
    return position, rotation


class IKController:
    """Verified inverse-kinematics boundary for shared primitives."""

    def __init__(self, robot: RobotBackend, collision: CollisionCheck):
        """Bind a robot backend and its safety checker."""
        self._robot = robot
        self._collision = collision

    def solve(
        self, world_t_tcp: FloatArray, seed: FloatArray, stage: str
    ) -> FloatArray:
        """Return a verified safe solution or raise a structured failure."""
        solution = self._robot.solve_ik(world_t_tcp, seed)
        if solution is None:
            raise ExecutionError(
                FailureCode.UNREACHABLE, stage, "no verified IK solution"
            )
        self._collision.require_safe(solution, stage)
        return solution


class TrajectoryController:
    """Feedback tracking for one joint-space target."""

    def __init__(
        self,
        robot: RobotBackend,
        world: WorldModel,
        collision: CollisionCheck,
        config: MotionConfig,
    ):
        """Bind feedback, command, safety, and motion limits."""
        self._robot = robot
        self._world = world
        self._collision = collision
        self._config = config

    async def move_to(
        self,
        goal_q: FloatArray,
        stage: str,
        world_t_tcp: FloatArray | None = None,
    ) -> OperationResult:
        """Track a joint target until its postcondition settles.

        Returns:
            Completion steps and final Cartesian errors.
        """
        goal = np.asarray(goal_q, dtype=np.float64)
        settled = 0
        position_error = 0.0
        rotation_error = 0.0
        for step in range(1, self._config.timeout_steps + 1):
            state = self._robot.read_state()
            delta = goal - state.q
            limits = np.full(delta.shape, self._config.max_arm_step)
            limits[list(self._robot.gripper_configuration_indices)] = (
                self._config.max_gripper_step
            )
            command = state.q + np.clip(delta, -limits, limits)
            self._collision.require_safe(command, stage)
            self._robot.command_configuration(command)
            await self._world.advance(2)

            state = self._robot.read_state()
            if world_t_tcp is None:
                errors = np.abs(
                    state.q[list(self._robot.arm_configuration_indices)]
                    - goal[list(self._robot.arm_configuration_indices)]
                )
                reached = bool(
                    errors.size
                    and float(np.max(errors)) <= self._config.joint_tolerance
                )
            else:
                position_error, rotation_error = pose_error(
                    state.tcp_world, world_t_tcp
                )
                reached = (
                    position_error <= self._config.position_tolerance
                    and rotation_error <= self._config.rotation_tolerance
                )
            settled = settled + 1 if reached else 0
            if settled >= self._config.settle_steps:
                print(
                    f"[motion] {stage}: {step} steps, "
                    f"position={position_error:.4f} m, "
                    f"rotation={np.rad2deg(rotation_error):.2f} deg",
                    flush=True,
                )
                return OperationResult(step, position_error, rotation_error)
        raise ExecutionError(
            FailureCode.TIMEOUT,
            stage,
            f"state did not settle after {self._config.timeout_steps} steps",
        )


class CartesianController:
    """Incremental straight Cartesian tracking with continuous IK."""

    def __init__(
        self,
        robot: RobotBackend,
        world: WorldModel,
        ik: IKController,
        collision: CollisionCheck,
        grasp_check: GraspStabilityCheck,
        config: MotionConfig,
    ):
        """Bind control, world feedback, and safety dependencies."""
        self._robot = robot
        self._world = world
        self._ik = ik
        self._collision = collision
        self._grasp_check = grasp_check
        self._config = config

    async def move_to(
        self,
        world_t_tcp: FloatArray,
        stage: str,
        grasp_width: float,
        closed: bool,
        tracking: GraspTracking | None = None,
        allow_vertical_settling: bool = False,
    ) -> OperationResult:
        """Track one straight Cartesian segment and enforce postconditions.

        Returns:
            Completion steps and final Cartesian errors.
        """
        target = np.asarray(world_t_tcp, dtype=np.float64)
        settled = 0
        commanded_tcp: FloatArray | None = None
        position_error = 0.0
        rotation_error = 0.0
        for step in range(1, self._config.timeout_steps + 1):
            state = self._robot.read_state()
            if tracking is not None:
                self._grasp_check.require_stable(
                    tracking, stage, allow_vertical_settling
                )
            if commanded_tcp is None:
                commanded_tcp = state.tcp_world.copy()
            position_error, rotation_error = pose_error(state.tcp_world, target)
            reached = (
                position_error <= self._config.position_tolerance
                and rotation_error <= self._config.rotation_tolerance
            )
            settled = settled + 1 if reached else 0
            if settled >= self._config.settle_steps:
                print(
                    f"[motion] {stage}: {step} steps, "
                    f"position={position_error:.4f} m, "
                    f"rotation={np.rad2deg(rotation_error):.2f} deg",
                    flush=True,
                )
                return OperationResult(step, position_error, rotation_error)

            incremental = commanded_tcp.copy()
            translation = target[:3, 3] - commanded_tcp[:3, 3]
            translation_norm = float(np.linalg.norm(translation))
            translation_step = (
                self._config.translation_step_held
                if closed
                else self._config.translation_step
            )
            if translation_norm > translation_step:
                translation *= translation_step / translation_norm
            incremental[:3, 3] += translation

            lead = incremental[:3, 3] - state.tcp_world[:3, 3]
            lead_norm = float(np.linalg.norm(lead))
            lead_limit = (
                self._config.command_lead_held
                if closed
                else self._config.command_lead
            ) * min(1.0, step / self._config.command_lead_ramp_steps)
            if lead_norm > lead_limit:
                incremental[:3, 3] = state.tcp_world[:3, 3] + (
                    lead * (lead_limit / lead_norm)
                )

            local_rotation = Rotation.from_matrix(
                state.tcp_world[:3, :3].T @ target[:3, :3]
            ).as_rotvec()
            rotation_norm = float(np.linalg.norm(local_rotation))
            rotation_limit = np.deg2rad(3.0 if closed else 5.0)
            if rotation_norm > rotation_limit:
                local_rotation *= rotation_limit / rotation_norm
            incremental[:3, :3] = (
                state.tcp_world[:3, :3]
                @ Rotation.from_rotvec(local_rotation).as_matrix()
            )
            commanded_tcp = incremental.copy()
            q_target = self._ik.solve(incremental, state.q, stage)
            q_target = self._robot.with_gripper(
                q_target, grasp_width, closed=closed
            )
            delta_q = q_target - state.q
            for index in self._robot.arm_configuration_indices:
                delta_q[index] = np.clip(
                    delta_q[index],
                    -self._config.max_arm_step,
                    self._config.max_arm_step,
                )
            for index in self._robot.gripper_configuration_indices:
                delta_q[index] = np.clip(
                    delta_q[index],
                    -self._config.max_gripper_step,
                    self._config.max_gripper_step,
                )
            command = state.q + delta_q
            self._collision.require_safe(command, stage)
            self._robot.command_configuration(command)
            await self._world.advance(2)
        raise ExecutionError(
            FailureCode.TIMEOUT,
            stage,
            f"Cartesian target did not settle after "
            f"{self._config.timeout_steps} steps "
            f"(position={position_error:.4f} m, "
            f"rotation={np.rad2deg(rotation_error):.2f} deg)",
        )
