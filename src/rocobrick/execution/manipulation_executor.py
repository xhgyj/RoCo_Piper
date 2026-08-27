"""One-call planning and execution for upper-level manipulation actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from rocobrick.backends.base import RobotBackend, WorldModel
from rocobrick.execution.types import (
    ActionFailure,
    ActionStatus,
    ExecutionError,
    ExecutionMetrics,
    FailureCode,
    HeldObject,
    HeldObjectState,
    ManipulationResult,
)
from rocobrick.planning.models import GroundedAction
from rocobrick.planning.planners import AssemblyPlanner, GraspPlanner
from rocobrick.skills.assemble import AssembleRequest, AssembleSkill
from rocobrick.skills.base_skill import ManipulationAction, ManipulationSkillType
from rocobrick.skills.pick import PickSkill


class ActionGrounder(Protocol):
    """Domain adapter from symbolic IDs to geometry and semantic checks."""

    def ground(self, action: ManipulationAction) -> GroundedAction:
        """Ground one action without choosing or changing its robots."""
        ...


class ManipulationExecutor:
    """Plan and execute one action while preserving upper-level assignment."""

    def __init__(
        self,
        robots: Mapping[str, RobotBackend],
        world: WorldModel,
        grounder: ActionGrounder,
        grasp_planner: GraspPlanner | None = None,
        assembly_planner: AssemblyPlanner | None = None,
    ):
        """Bind robot resources, feedback, grounding, and pure planners."""
        self._robots = dict(robots)
        self._world = world
        self._grounder = grounder
        self._grasp_planner = grasp_planner or GraspPlanner()
        self._assembly_planner = assembly_planner or AssemblyPlanner()
        self._held: dict[str, HeldObjectState] = {}

    def held_state(self, robot_id: str) -> HeldObjectState | None:
        """Return the current held-object state for one robot."""
        return self._held.get(robot_id)

    async def execute(self, action: ManipulationAction) -> ManipulationResult:
        """Ground, plan, and execute one upper-planner action.

        Returns:
            A uniform terminal result; expected failures are not raised.
        """
        invalid = self._validate_action(action)
        if invalid is not None:
            return self._failure(action, ActionStatus.INVALID_ACTION, invalid)
        robot_id = action.robot_ids[0]
        robot = self._robots[robot_id]
        try:
            grounded = self._grounder.ground(action)
            if grounded.scene.target.object_id != action.object_id:
                raise ValueError("grounder returned geometry for a different object")
        except (KeyError, ValueError) as exc:
            return self._failure(
                action,
                ActionStatus.INVALID_ACTION,
                ActionFailure(FailureCode.INVALID_REQUEST, "grounding", str(exc)),
            )
        if action.skill_type is ManipulationSkillType.PICK:
            return await self._execute_pick(action, robot, grounded)
        return await self._execute_place_down(action, robot, grounded)

    def _validate_action(self, action: ManipulationAction) -> ActionFailure | None:
        if action.skill_type not in {
            ManipulationSkillType.PICK,
            ManipulationSkillType.PLACE_DOWN,
        }:
            return ActionFailure(
                FailureCode.INVALID_REQUEST,
                "validation",
                "skill "
                f"{action.skill_type.value} is outside the downward-assembly scope",
            )
        missing = [
            robot_id for robot_id in action.robot_ids if robot_id not in self._robots
        ]
        if missing:
            return ActionFailure(
                FailureCode.INVALID_REQUEST,
                "validation",
                f"unknown assigned robot(s): {missing}",
            )
        robot_id = action.robot_ids[0]
        held = self._held.get(robot_id)
        if action.skill_type is ManipulationSkillType.PICK and held is not None:
            return ActionFailure(
                FailureCode.INVALID_REQUEST,
                "validation",
                f"{robot_id} already holds {held.held.object_id}",
            )
        if action.skill_type is ManipulationSkillType.PLACE_DOWN:
            if action.goal_id is None:
                return ActionFailure(
                    FailureCode.INVALID_REQUEST,
                    "validation",
                    "place_down requires goal_id",
                )
            if held is None or held.held.object_id != action.object_id:
                return ActionFailure(
                    FailureCode.INVALID_REQUEST,
                    "validation",
                    f"{robot_id} does not hold {action.object_id}",
                )
        return None

    async def _execute_pick(
        self,
        action: ManipulationAction,
        robot: RobotBackend,
        grounded: GroundedAction,
    ) -> ManipulationResult:
        try:
            plan = self._grasp_planner.plan(
                robot, grounded.scene, grounded.assembly_goal
            )
        except (ExecutionError, ValueError) as exc:
            return self._planning_failure(action, exc)
        try:
            result = await PickSkill.create(robot, self._world).execute_plan(plan)
        except (ExecutionError, ValueError) as exc:
            return self._execution_failure(action, exc)
        state = HeldObjectState.from_pick(result.held, result.acquisition_object_t_tcp)
        self._held[robot.robot_id] = state
        return ManipulationResult(
            action.action_id,
            ActionStatus.SUCCESS,
            action.robot_ids,
            action.object_id,
            robot.robot_id,
            result.metrics,
        )

    async def _execute_place_down(
        self,
        action: ManipulationAction,
        robot: RobotBackend,
        grounded: GroundedAction,
    ) -> ManipulationResult:
        held_state = self._held[robot.robot_id]
        if grounded.assembly_goal is None:
            return self._failure(
                action,
                ActionStatus.INVALID_ACTION,
                ActionFailure(
                    FailureCode.INVALID_REQUEST,
                    "grounding",
                    "place_down grounding did not provide an assembly goal",
                ),
            )
        try:
            plan = self._assembly_planner.plan(
                robot, grounded.scene, held_state, grounded.assembly_goal
            )
        except (ExecutionError, ValueError) as exc:
            return self._planning_failure(action, exc)
        try:
            transport_metrics = await self._execute_transport(robot, held_state, plan)
            immutable_held = HeldObject(
                held_state.held.object_id,
                held_state.held.robot_id,
                held_state.acquisition_object_t_tcp,
                held_state.held.grasp_axis,
                held_state.held.grasp_width,
            )
            assembled = await AssembleSkill.create(robot, self._world).execute(
                AssembleRequest(
                    immutable_held,
                    plan.world_t_preassembly_tcp,
                    plan.world_t_goal_tcp,
                    plan.goal.insertion_direction_world,
                    plan.goal.success_check,
                    approach_clearance=max(
                        0.001,
                        float(
                            np.linalg.norm(
                                plan.world_t_goal_tcp[:3, 3]
                                - plan.world_t_preassembly_tcp[:3, 3]
                            )
                        ),
                    ),
                    insertion_distance=plan.insertion_distance,
                    insertion_step=plan.insertion_step,
                    retreat_distance=plan.retreat_distance,
                    max_alignment_rotation_step=plan.rotation_step,
                )
            )
        except (ExecutionError, ValueError) as exc:
            return self._execution_failure(action, exc)
        del self._held[robot.robot_id]
        return ManipulationResult(
            action.action_id,
            ActionStatus.SUCCESS,
            action.robot_ids,
            action.object_id,
            None,
            transport_metrics + assembled.metrics,
        )

    async def _execute_transport(self, robot, held_state, plan) -> ExecutionMetrics:
        waypoints = 0
        for configuration in plan.transport.path.configurations[1:]:
            command = robot.with_gripper(
                configuration, held_state.held.grasp_width, closed=True
            )
            if not robot.configuration_is_safe(command):
                raise ExecutionError(
                    FailureCode.COLLISION,
                    "assembly_transport",
                    "planned configuration became unsafe before execution",
                )
            robot.command_configuration(command)
            await self._world.advance(2)
            object_pose = self._world.object_pose(held_state.held.object_id)
            object_t_tcp = np.linalg.inv(object_pose) @ robot.read_state().tcp_world
            held_state.observe(object_t_tcp)
            if held_state.cumulative_position_drift > 0.012:
                raise ExecutionError(
                    FailureCode.SLIPPED,
                    "assembly_transport",
                    "cumulative grasp translation drift exceeded 12 mm",
                )
            if held_state.cumulative_rotation_drift > np.deg2rad(10.0):
                raise ExecutionError(
                    FailureCode.SLIPPED,
                    "assembly_transport",
                    "cumulative grasp rotation drift exceeded 10 degrees",
                )
            waypoints += 1
        return ExecutionMetrics(
            waypoints=waypoints,
            control_iterations=waypoints,
            simulation_steps=waypoints * 2,
        )

    def _failure(
        self,
        action: ManipulationAction,
        status: ActionStatus,
        failure: ActionFailure,
    ) -> ManipulationResult:
        held_by = next(
            (
                robot_id
                for robot_id in action.robot_ids
                if robot_id in self._held
                and self._held[robot_id].held.object_id == action.object_id
            ),
            None,
        )
        return ManipulationResult(
            action.action_id,
            status,
            action.robot_ids,
            action.object_id,
            held_by,
            ExecutionMetrics(),
            failure,
        )

    def _planning_failure(
        self, action: ManipulationAction, error: Exception
    ) -> ManipulationResult:
        failure = self._as_failure(error, "planning")
        return self._failure(action, ActionStatus.PLANNING_FAILED, failure)

    def _execution_failure(
        self, action: ManipulationAction, error: Exception
    ) -> ManipulationResult:
        failure = self._as_failure(error, "execution")
        held = self._held.get(action.robot_ids[0])
        return ManipulationResult(
            action.action_id,
            ActionStatus.EXECUTION_FAILED,
            action.robot_ids,
            action.object_id,
            None if held is None else action.robot_ids[0],
            ExecutionMetrics(),
            failure,
        )

    @staticmethod
    def _as_failure(error: Exception, stage: str) -> ActionFailure:
        if isinstance(error, ExecutionError):
            return ActionFailure(error.code, error.stage, error.detail)
        return ActionFailure(FailureCode.INVALID_REQUEST, stage, str(error))
