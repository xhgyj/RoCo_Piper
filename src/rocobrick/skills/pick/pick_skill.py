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
    ExecutionMetrics,
    FailureCode,
    HeldObject,
)
from rocobrick.planning.models import CartesianPath, GraspPlan
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
    metrics: ExecutionMetrics
    initial_object_pose: FloatArray
    lifted_object_pose: FloatArray
    acquisition_object_t_tcp: FloatArray

    @property
    def steps(self) -> int:
        """Return physical simulation steps."""
        return self.metrics.simulation_steps


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
        trajectory = TrajectoryController(robot, world, collision, motion_config)
        cartesian = CartesianController(
            robot,
            world,
            ik,
            collision,
            grasp_check,
            motion_config,
        )
        context = PrimitiveContext(robot, world, trajectory, cartesian, collision)
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
        acquisition_object_t_tcp = held.object_t_tcp.copy()
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
            np.dot(lifted_object[:3, 3] - initial_object[:3, 3], lift_direction)
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
        control_iterations = sum(
            result.steps
            for result in (
                move_result,
                open_result,
                approach_result,
                grasp_result,
                retreat_result,
            )
        )
        return PickResult(
            held,
            ExecutionMetrics(
                control_iterations=control_iterations,
                simulation_steps=control_iterations * 2,
            ),
            initial_object,
            lifted_object,
            acquisition_object_t_tcp,
        )

    async def execute_plan(self, plan: GraspPlan) -> PickResult:
        """Execute the exact continuous IK branch selected by the planner.

        Returns:
            Held-object state and Pick execution metrics.
        """
        if plan.robot_id != self._context.robot.robot_id:
            raise ValueError("grasp plan belongs to a different robot")
        initial_object = self._context.world.object_pose(plan.object_id)
        metrics = await self._execute_path(
            plan.pregrasp, plan.grasp_width, closed=False, stage="pregrasp"
        )
        opened = await self._grasp.prepare(plan.grasp_width)
        metrics += ExecutionMetrics(
            control_iterations=opened.steps,
            simulation_steps=opened.steps * 2,
        )
        metrics += await self._execute_path(
            plan.approach,
            plan.grasp_width,
            closed=False,
            stage="grasp_approach",
        )
        grasped = await self._grasp.execute(plan.grasp_width)
        metrics += ExecutionMetrics(
            control_iterations=grasped.steps,
            simulation_steps=grasped.steps * 2,
        )
        state = self._context.robot.read_state()
        grasped_object = self._context.world.object_pose(plan.object_id)
        acquisition = np.linalg.inv(grasped_object) @ state.tcp_world
        metrics += await self._execute_path(
            plan.lift, plan.grasp_width, closed=True, stage="pick_lift"
        )
        lifted_object = self._context.world.object_pose(plan.object_id)
        expected_lift = plan.lift.poses[-1][:3, 3] - plan.lift.poses[0][:3, 3]
        actual_lift = lifted_object[:3, 3] - initial_object[:3, 3]
        if float(np.dot(actual_lift, expected_lift)) < float(
            np.dot(expected_lift, expected_lift) * 0.55
        ):
            raise ExecutionError(
                FailureCode.SLIPPED,
                "pick_lift",
                "object did not follow the planned lift",
            )
        final_tcp = self._context.robot.read_state().tcp_world
        settled = np.linalg.inv(lifted_object) @ final_tcp
        return PickResult(
            HeldObject(
                plan.object_id,
                plan.robot_id,
                settled,
                plan.grasp_axis,
                plan.grasp_width,
            ),
            metrics,
            initial_object,
            lifted_object,
            acquisition,
        )

    async def _execute_path(
        self,
        path: CartesianPath,
        grasp_width: float,
        closed: bool,
        stage: str,
    ) -> ExecutionMetrics:
        """Command every preverified IK sample in one Cartesian path.

        Returns:
            Number of backend control cycles advanced.
        """
        waypoints = 0
        for configuration in path.configurations[1:]:
            command = self._context.robot.with_gripper(
                configuration, grasp_width, closed
            )
            self._context.collision.require_safe(command, stage)
            self._context.robot.command_configuration(command)
            await self._context.world.advance(2)
            waypoints += 1
        return ExecutionMetrics(
            waypoints=waypoints,
            control_iterations=waypoints,
            simulation_steps=waypoints * 2,
        )

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
