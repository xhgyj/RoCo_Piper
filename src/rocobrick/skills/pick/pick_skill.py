"""Backend-independent Pick skill composed from shared primitives."""

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
)
from rocobrick.execution.types import (
    ExecutionError,
    FailureCode,
    HeldObject,
)
from rocobrick.primitives.approach import Approach, ApproachRequest
from rocobrick.primitives.base import PrimitiveContext
from rocobrick.primitives.gripper import Grasp
from rocobrick.primitives.move import Move, MoveRequest
from rocobrick.primitives.retreat import Retreat, RetreatRequest
from rocobrick.safety.checks import CollisionCheck, GraspStabilityCheck

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GraspCandidate:
    """One grounded side grasp considered by Pick."""

    world_t_pregrasp_tcp: FloatArray
    world_t_grasp_tcp: FloatArray
    grasp_axis: int
    grasp_width: float

    def __post_init__(self) -> None:
        """Validate geometry before invoking a controller."""
        for name, transform in (
            ("world_t_pregrasp_tcp", self.world_t_pregrasp_tcp),
            ("world_t_grasp_tcp", self.world_t_grasp_tcp),
        ):
            value = np.asarray(transform, dtype=np.float64)
            if value.shape != (4, 4) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 4x4 transform")
        if self.grasp_axis not in (0, 1):
            raise ValueError("grasp_axis must be 0 or 1")
        if self.grasp_width <= 0.0:
            raise ValueError("grasp_width must be positive")


@dataclass(frozen=True)
class PickRequest:
    """Grounded object and grasp candidates for one Pick action."""

    object_id: str
    candidates: tuple[GraspCandidate, ...]
    lift_distance: float = 0.06

    def __post_init__(self) -> None:
        """Require an object, candidate set, and positive lift."""
        if not self.object_id:
            raise ValueError("object_id cannot be empty")
        if not self.candidates:
            raise ValueError("Pick requires at least one grasp candidate")
        if self.lift_distance <= 0.0:
            raise ValueError("lift_distance must be positive")


@dataclass(frozen=True)
class PickResult:
    """Successful Pick output consumed by later skills."""

    held: HeldObject
    steps: int
    initial_object_pose: FloatArray
    lifted_object_pose: FloatArray


@dataclass(frozen=True)
class _SelectedGrasp:
    candidate: GraspCandidate
    q_pregrasp: FloatArray


class PickSkill:
    """Compose Move, Approach, Grasp, and Retreat into a stable Pick."""

    def __init__(
        self,
        context: PrimitiveContext,
        ik: IKController,
    ):
        """Bind reusable primitives without binding task or BrickSim state."""
        self._context = context
        self._ik = ik
        self._move = Move(context)
        self._approach = Approach(context)
        self._grasp = Grasp(context)
        self._retreat = Retreat(context)

    @classmethod
    def create(
        cls,
        robot: RobotBackend,
        world: WorldModel,
        config: MotionConfig | None = None,
    ) -> PickSkill:
        """Construct a Pick skill from backend-independent dependencies.

        Returns:
            Fully wired single-robot Pick skill.
        """
        motion_config = config or MotionConfig()
        collision = CollisionCheck(robot)
        grasp_check = GraspStabilityCheck(robot, world)
        ik = IKController(robot, collision)
        trajectory = TrajectoryController(
            robot, world, collision, motion_config
        )
        cartesian = CartesianController(
            robot,
            world,
            ik,
            collision,
            grasp_check,
            motion_config,
        )
        context = PrimitiveContext(
            robot, world, trajectory, cartesian, collision
        )
        return cls(context, ik)

    async def execute(self, request: PickRequest) -> PickResult:
        """Pick, lift, and verify the held-object contract.

        Returns:
            Held-object state and Pick execution metrics.
        """
        initial_object = self._context.world.object_pose(request.object_id)
        lift_direction = initial_object[:3, 2]
        selected = self._select_candidate(request, lift_direction)
        candidate = selected.candidate
        pregrasp_q = self._context.robot.with_gripper(
            selected.q_pregrasp,
            candidate.grasp_width,
            closed=False,
        )
        move_result = await self._move.execute(
            MoveRequest(
                pregrasp_q,
                "pregrasp",
                candidate.world_t_pregrasp_tcp,
            )
        )
        open_result = await self._grasp.prepare(candidate.grasp_width)
        approach_result = await self._approach.execute(
            ApproachRequest(
                candidate.world_t_grasp_tcp,
                candidate.grasp_width,
                "grasp_approach",
            )
        )
        grasp_result = await self._grasp.execute(candidate.grasp_width)

        state = self._context.robot.read_state()
        grasped_object = self._context.world.object_pose(request.object_id)
        held = HeldObject(
            request.object_id,
            self._context.robot.robot_id,
            np.linalg.inv(grasped_object) @ state.tcp_world,
            candidate.grasp_axis,
            candidate.grasp_width,
        )
        lift_tcp = state.tcp_world.copy()
        lift_tcp[:3, 3] += lift_direction * request.lift_distance
        retreat_result = await self._retreat.execute(
            RetreatRequest(
                lift_tcp,
                candidate.grasp_width,
                held,
                "pick_lift",
                allow_vertical_settling=True,
            )
        )
        lifted_object = self._context.world.object_pose(request.object_id)
        actual_lift = float(
            np.dot(
                lifted_object[:3, 3] - initial_object[:3, 3], lift_direction
            )
        )
        if actual_lift < request.lift_distance * 0.55:
            raise ExecutionError(
                FailureCode.SLIPPED,
                "pick_lift",
                f"object did not follow the gripper: lift={actual_lift:.4f} m",
            )
        final_state = self._context.robot.read_state()
        held = HeldObject(
            request.object_id,
            self._context.robot.robot_id,
            np.linalg.inv(lifted_object) @ final_state.tcp_world,
            candidate.grasp_axis,
            candidate.grasp_width,
        )
        total_steps = sum(
            result.steps
            for result in (
                move_result,
                open_result,
                approach_result,
                grasp_result,
                retreat_result,
            )
        )
        return PickResult(held, total_steps, initial_object, lifted_object)

    def _select_candidate(
        self, request: PickRequest, lift_direction: FloatArray
    ) -> _SelectedGrasp:
        seed = self._context.robot.read_state().q
        for candidate in request.candidates:
            try:
                q_pregrasp = self._ik.solve(
                    candidate.world_t_pregrasp_tcp,
                    seed,
                    "pick_pregrasp_probe",
                )
                q_grasp = self._ik.solve(
                    candidate.world_t_grasp_tcp,
                    q_pregrasp,
                    "pick_grasp_probe",
                )
                lift_tcp = candidate.world_t_grasp_tcp.copy()
                lift_tcp[:3, 3] += lift_direction * request.lift_distance
                self._ik.solve(lift_tcp, q_grasp, "pick_lift_probe")
            except ExecutionError:
                continue
            return _SelectedGrasp(candidate, q_pregrasp)
        raise ExecutionError(
            FailureCode.UNREACHABLE,
            "pick",
            "no grasp candidate has a verified pregrasp/grasp/lift IK chain",
        )
