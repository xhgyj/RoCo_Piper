"""Backend-independent Assemble skill composed from shared primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rocobrick.backends.base import RobotBackend, WorldModel
from rocobrick.controllers.motion import (
    CartesianController,
    IKController,
    MotionConfig,
    TrajectoryController,
    pose_error,
)
from rocobrick.execution.types import (
    ExecutionError,
    ExecutionMetrics,
    FailureCode,
    HeldObject,
    OperationResult,
)
from rocobrick.primitives.align import Align, AlignRequest, _unit_vector
from rocobrick.primitives.approach import Approach, ApproachRequest
from rocobrick.primitives.base import PrimitiveContext
from rocobrick.primitives.insert_press import (
    InsertPress,
    InsertPressRequest,
    WrenchSource,
)
from rocobrick.primitives.move import Move, MoveRequest
from rocobrick.primitives.release import Release
from rocobrick.primitives.retreat import Retreat, RetreatRequest
from rocobrick.safety.checks import (
    CollisionCheck,
    ForceGuard,
    GraspStabilityCheck,
    SuccessCheck,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AssembleRequest:
    """Grounded geometric parameters for Place-Down or Place-Up."""

    held: HeldObject
    world_t_preassembly_tcp: FloatArray
    world_t_goal_tcp: FloatArray
    insertion_direction_world: FloatArray
    success_check: SuccessCheck
    alignment_rotation_hint_goal: FloatArray | None = None
    approach_clearance: float = 0.005
    insertion_distance: float = 0.012
    insertion_step: float = 0.0001
    retreat_distance: float = 0.04
    max_insert_steps: int = 180
    wrench_source: WrenchSource | None = None
    force_guard: ForceGuard | None = None
    preassembly_position_tolerance: float = 0.003
    preassembly_rotation_tolerance: float = np.deg2rad(3.0)
    max_alignment_rotation_step: float = np.deg2rad(1.0)

    def __post_init__(self) -> None:
        """Validate grounded geometry before any robot command."""
        for name, transform in (
            ("world_t_preassembly_tcp", self.world_t_preassembly_tcp),
            ("world_t_goal_tcp", self.world_t_goal_tcp),
        ):
            value = np.asarray(transform, dtype=np.float64)
            if value.shape != (4, 4) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 4x4 transform")
        _unit_vector(self.insertion_direction_world, "insertion_direction_world")
        if self.alignment_rotation_hint_goal is not None:
            _unit_vector(
                self.alignment_rotation_hint_goal,
                "alignment_rotation_hint_goal",
            )
        if (
            self.approach_clearance <= 0.0
            or self.insertion_distance <= 0.0
            or self.insertion_step <= 0.0
            or self.retreat_distance <= 0.0
            or self.preassembly_position_tolerance <= 0.0
            or self.preassembly_rotation_tolerance <= 0.0
            or self.max_alignment_rotation_step <= 0.0
        ):
            raise ValueError("assembly distances must be positive")
        if self.max_insert_steps <= 0:
            raise ValueError("max_insert_steps must be positive")


@dataclass(frozen=True)
class AssembleResult:
    """Successful assembly metrics and final retreated TCP pose."""

    metrics: ExecutionMetrics
    final_tcp_world: FloatArray

    @property
    def steps(self) -> int:
        """Return physical simulation steps."""
        return self.metrics.simulation_steps


class AssembleSkill:
    """Compose motion, mating, verification, release, and retreat."""

    def __init__(self, context: PrimitiveContext):
        """Bind reusable primitives without task or simulator state."""
        self._context = context
        self._move = Move(context)
        self._align = Align(context)
        self._approach = Approach(context)
        self._insert_press = InsertPress(context)
        self._release = Release(context)
        self._retreat = Retreat(context)

    @classmethod
    def create(
        cls,
        robot: RobotBackend,
        world: WorldModel,
        config: MotionConfig | None = None,
    ) -> AssembleSkill:
        """Construct an Assemble skill from backend-independent dependencies.

        Returns:
            Fully wired single-robot Assemble skill.
        """
        motion_config = config or MotionConfig()
        collision = CollisionCheck(robot)
        grasp_check = GraspStabilityCheck(robot, world)
        ik = IKController(robot, collision)
        trajectory = TrajectoryController(robot, world, collision, motion_config)
        cartesian = CartesianController(
            robot,
            world,
            ik,
            collision,
            grasp_check,
            motion_config,
        )
        return cls(PrimitiveContext(robot, world, trajectory, cartesian, collision))

    async def execute(self, request: AssembleRequest) -> AssembleResult:
        """Execute one direction-parameterized assembly operation.

        Returns:
            Total controller steps and the final retreat pose.
        """
        if request.held.robot_id != self._context.robot.robot_id:
            raise ValueError(
                "held object belongs to "
                f"{request.held.robot_id}, not {self._context.robot.robot_id}"
            )
        direction = _unit_vector(
            request.insertion_direction_world, "insertion_direction_world"
        )
        preassembly_q = self._context.robot.solve_ik(
            request.world_t_preassembly_tcp,
            self._context.robot.read_state().q,
        )
        if preassembly_q is None:
            raise ExecutionError(
                FailureCode.UNREACHABLE,
                "preassembly",
                "no verified IK solution",
            )
        current_error = pose_error(
            self._context.robot.read_state().tcp_world,
            request.world_t_preassembly_tcp,
        )
        if (
            current_error[0] <= request.preassembly_position_tolerance
            and current_error[1] <= request.preassembly_rotation_tolerance
        ):
            move_result = OperationResult(0, *current_error)
        else:
            move_result = await self._move.execute(
                MoveRequest(
                    preassembly_q,
                    "preassembly",
                    request.world_t_preassembly_tcp,
                    request.held,
                )
            )
        align_result = await self._align.execute(
            AlignRequest(
                request.world_t_goal_tcp,
                direction,
                request.held,
                success_check=request.success_check,
                rotation_hint_goal=request.alignment_rotation_hint_goal,
                max_rotation_step=request.max_alignment_rotation_step,
            )
        )
        if request.success_check.is_satisfied():
            approach_result = OperationResult(0)
            insert_result = OperationResult(0)
        else:
            approach_tcp = np.asarray(request.world_t_goal_tcp, dtype=np.float64).copy()
            approach_tcp[:3, 3] -= direction * request.approach_clearance
            approach_result = await self._approach.execute(
                ApproachRequest(
                    approach_tcp,
                    request.held.grasp_width,
                    "assembly_approach",
                    request.held,
                    request.success_check,
                )
            )
            insert_result = await self._insert_press.execute(
                InsertPressRequest(
                    direction,
                    request.held,
                    request.success_check,
                    request.insertion_distance,
                    request.insertion_step,
                    request.max_insert_steps,
                    wrench_source=request.wrench_source,
                    force_guard=request.force_guard,
                )
            )
        request.success_check.require_satisfied("assemble_verify")
        release_result = await self._release.execute(
            request.held.grasp_width, request.success_check
        )
        released_tcp = self._context.robot.read_state().tcp_world
        retreat_tcp = released_tcp.copy()
        retreat_tcp[:3, 3] -= direction * request.retreat_distance
        retreat_result = await self._retreat.execute(
            RetreatRequest(
                retreat_tcp,
                request.held.grasp_width,
                stage="assembly_retreat",
            )
        )
        request.success_check.require_satisfied("assembly_retreat")
        control_iterations = sum(
            result.steps
            for result in (
                move_result,
                align_result,
                approach_result,
                insert_result,
                release_result,
                retreat_result,
            )
        )
        return AssembleResult(
            ExecutionMetrics(
                control_iterations=control_iterations,
                simulation_steps=control_iterations * 2,
            ),
            self._context.robot.read_state().tcp_world,
        )
