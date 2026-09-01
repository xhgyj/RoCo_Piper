"""Execution planning and generic multi-arm episode scheduling."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from rocobrick.execution.manipulation_executor import ManipulationExecutor
from rocobrick.execution.types import ActionStatus, ManipulationResult
from rocobrick.planning.task_planner import AssemblyTask, TaskPlan
from rocobrick.skills.base_skill import ManipulationAction, ManipulationSkillType
from rocobrick.task_config.episode import EpisodeConfig


@dataclass(frozen=True)
class ExecutableAssemblyTask:
    """Upper task plus resource-safe preparation and placement gates."""

    task: AssemblyTask
    prepare_after: tuple[str, ...]
    place_after: tuple[str, ...]
    return_home: bool


@dataclass(frozen=True)
class ExecutionPlan:
    """Execution-layer expansion without geometric robot trajectories."""

    episode_id: str
    shared_resource: str
    tasks: tuple[ExecutableAssemblyTask, ...]


@dataclass(frozen=True)
class TaskExecutionRecord:
    """Terminal action results for one assembly task."""

    task_id: str
    target_part_id: int
    assigned_arm: str
    pick: ManipulationResult | None
    place: ManipulationResult | None
    failure_reason: str | None

    @property
    def success(self) -> bool:
        """Return whether both manipulation actions succeeded."""
        return bool(
            self.pick is not None
            and self.pick.status is ActionStatus.SUCCESS
            and self.place is not None
            and self.place.status is ActionStatus.SUCCESS
            and self.failure_reason is None
        )


@dataclass(frozen=True)
class EpisodeRunResult:
    """Terminal result of a complete multi-arm episode."""

    episode_id: str
    success: bool
    completed_task_ids: tuple[str, ...]
    records: tuple[TaskExecutionRecord, ...]
    failure_reason: str | None


class ExecutionPlanner:
    """Compile task dependencies into configurable prefetch gates."""

    def compile(self, plan: TaskPlan, episode: EpisodeConfig) -> ExecutionPlan:
        """Preserve assignments while adding arm and workspace ordering.

        Returns:
            Scheduler-ready execution plan.
        """
        if plan.episode_id != episode.episode_id:
            raise ValueError(
                f"plan episode {plan.episode_id} does not match {episode.episode_id}"
            )
        known = {task.task_id for task in plan.tasks}
        if len(known) != len(plan.tasks):
            raise ValueError("task plan contains duplicate task IDs")
        previous_by_arm: dict[str, str] = {}
        previous_global: str | None = None
        last_by_arm = {
            arm_id: next(
                task.task_id
                for task in reversed(plan.tasks)
                if task.assigned_arm == arm_id
            )
            for arm_id in {task.assigned_arm for task in plan.tasks}
        }
        result = []
        for task in plan.tasks:
            if task.assigned_arm not in episode.available_arm_ids:
                raise ValueError(f"task {task.task_id} uses unavailable arm")
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(
                    f"task {task.task_id} has missing dependencies {sorted(missing)}"
                )
            prepare = set(task.depends_on)
            arm_previous = previous_by_arm.get(task.assigned_arm)
            if arm_previous is not None:
                prepare.add(arm_previous)
            place = set(task.depends_on)
            if previous_global is not None:
                place.add(previous_global)
                if not episode.allow_prefetch:
                    prepare.add(previous_global)
            result.append(
                ExecutableAssemblyTask(
                    task,
                    tuple(sorted(prepare)),
                    tuple(sorted(place)),
                    last_by_arm[task.assigned_arm] == task.task_id,
                )
            )
            previous_by_arm[task.assigned_arm] = task.task_id
            previous_global = task.task_id
        return ExecutionPlan(
            plan.episode_id, episode.shared_resource, tuple(result)
        )


class _SynchronizedWorld:
    """Serialize Isaac stepping while allowing independent arm coroutines."""

    def __init__(self, world: object):
        self._world = world
        self._step_lock = asyncio.Lock()

    def object_pose(self, object_id: str) -> np.ndarray:
        """Forward one object pose query.

        Returns:
            Current world transform.
        """
        return self._world.object_pose(object_id)

    async def advance(self, steps: int = 1) -> None:
        """Advance the shared simulator without overlapping another step."""
        async with self._step_lock:
            await self._world.advance(steps)


class MultiArmScheduler:
    """Execute an arbitrary-arm plan with one logical assembly resource."""

    def __init__(self, env, episode: EpisodeConfig):
        """Bind the simulator and immutable episode policies."""
        self._env = env
        self._episode = episode
        self._pause_gate = asyncio.Event()
        self._pause_gate.set()
        self._motion_lock = asyncio.Lock()
        self._workspace_lock = asyncio.Lock()

    def pause(self) -> None:
        """Stop dispatching new safe-boundary phases."""
        self._pause_gate.clear()

    def resume(self) -> None:
        """Allow safe-boundary phase dispatch to continue."""
        self._pause_gate.set()

    async def run(self, plan: ExecutionPlan) -> EpisodeRunResult:
        """Run all task coroutines and return deterministic records.

        Returns:
            Complete episode status and per-task action results.
        """
        if plan.episode_id != self._episode.episode_id:
            raise ValueError("execution plan belongs to another episode")
        from rocobrick.backends.bricksim import (
            BrickSimRobotBackend,
            BrickSimWorldModel,
        )
        from rocobrick.policy.bricksim_grounder import BrickSimActionGrounder

        robot_indices = {
            str(config.get("Name", f"robot_{index}")): index
            for index, config in enumerate(self._env.robot_configs)
        }
        missing = set(self._episode.available_arm_ids) - set(robot_indices)
        if missing:
            raise ValueError(f"episode robots are unavailable: {sorted(missing)}")
        robots = {
            robot_id: BrickSimRobotBackend(self._env, robot_indices[robot_id])
            for robot_id in self._episode.available_arm_ids
        }
        world = _SynchronizedWorld(BrickSimWorldModel(self._env))
        assembled: dict[int, str] = {
            int(part_id): str(path)
            for part_id, path in self._env.pre_placed_parts.items()
        }
        terminal = {item.task.task_id: asyncio.Event() for item in plan.tasks}
        records: dict[str, TaskExecutionRecord] = {}

        async def execute(item: ExecutableAssemblyTask) -> None:
            task = item.task
            await self._wait_for(item.prepare_after, terminal, records)
            await self._pause_gate.wait()
            if self._dependency_failed(item.prepare_after, records):
                self._log(task, "blocked", "failed_dependency_before_pick")
                records[task.task_id] = self._blocked_record(task)
                terminal[task.task_id].set()
                return
            target_path = str(self._env.to_place_placed[task.target_part_id])
            grounder = BrickSimActionGrounder(
                self._env,
                target_id=task.target_part_id,
                assembled_parts=assembled,
                connection_ids=task.connection_ids,
            )
            executor = ManipulationExecutor(
                {task.assigned_arm: robots[task.assigned_arm]}, world, grounder
            )
            goal_id = grounder.assembly_goal_id(target_path)
            pick_action = ManipulationAction(
                f"{task.task_id}-pick",
                (task.assigned_arm,),
                ManipulationSkillType.PICK,
                target_path,
                goal_id,
            )
            async with self._motion_lock:
                self._log(task, "motion", "acquired_for_pick")
                self._log(task, "pick", "started")
                pick = await self._execute_with_retries(executor, pick_action)
                self._log(task, "pick", self._action_status(pick))
            if pick.status is not ActionStatus.SUCCESS:
                records[task.task_id] = TaskExecutionRecord(
                    task.task_id,
                    task.target_part_id,
                    task.assigned_arm,
                    pick,
                    None,
                    self._failure_text(pick),
                )
                terminal[task.task_id].set()
                return

            await self._wait_for(item.place_after, terminal, records)
            await self._pause_gate.wait()
            if self._dependency_failed(item.place_after, records):
                self._log(task, "blocked", "failed_dependency_before_place")
                records[task.task_id] = TaskExecutionRecord(
                    task.task_id,
                    task.target_part_id,
                    task.assigned_arm,
                    pick,
                    None,
                    "an assembly dependency failed",
                )
                terminal[task.task_id].set()
                return
            async with self._workspace_lock:
                self._log(task, self._episode.shared_resource, "acquired")
                async with self._motion_lock:
                    self._log(task, "motion", "acquired_for_place")
                    place_action = ManipulationAction(
                        f"{task.task_id}-place-down",
                        (task.assigned_arm,),
                        ManipulationSkillType.PLACE_DOWN,
                        target_path,
                        goal_id,
                    )
                    self._log(task, "place_down", "started")
                    place = await self._execute_with_retries(
                        executor, place_action
                    )
                    self._log(task, "place_down", self._action_status(place))
                    if place.status is ActionStatus.SUCCESS:
                        path = self._env.to_place_placed.pop(task.target_part_id)
                        self._env.pre_placed_parts[task.target_part_id] = path
                        assembled[task.target_part_id] = str(path)
                        if item.return_home:
                            self._log(task, "return_home", "started")
                            await self._return_home(
                                robots[task.assigned_arm], world
                            )
                            self._log(task, "return_home", "success")
                        reason = None
                    else:
                        reason = self._failure_text(place)
            records[task.task_id] = TaskExecutionRecord(
                task.task_id,
                task.target_part_id,
                task.assigned_arm,
                pick,
                place,
                reason,
            )
            terminal[task.task_id].set()

        await asyncio.gather(*(execute(item) for item in plan.tasks))
        ordered = tuple(records[item.task.task_id] for item in plan.tasks)
        completed = tuple(record.task_id for record in ordered if record.success)
        failure = next(
            (record.failure_reason for record in ordered if not record.success), None
        )
        return EpisodeRunResult(
            plan.episode_id,
            len(completed) == len(plan.tasks),
            completed,
            ordered,
            failure,
        )

    async def _execute_with_retries(
        self, executor: ManipulationExecutor, action: ManipulationAction
    ) -> ManipulationResult:
        result = await executor.execute(action)
        for _ in range(self._episode.max_retries):
            if result.status in {ActionStatus.SUCCESS, ActionStatus.INVALID_ACTION}:
                break
            if (
                action.skill_type is ManipulationSkillType.PICK
                and result.held_by is not None
            ):
                break
            await self._pause_gate.wait()
            result = await executor.execute(action)
        return result

    @staticmethod
    async def _wait_for(
        dependencies: tuple[str, ...],
        terminal: Mapping[str, asyncio.Event],
        records: Mapping[str, TaskExecutionRecord],
    ) -> None:
        del records
        await asyncio.gather(*(terminal[task_id].wait() for task_id in dependencies))

    @staticmethod
    def _dependency_failed(
        dependencies: tuple[str, ...], records: Mapping[str, TaskExecutionRecord]
    ) -> bool:
        return any(not records[task_id].success for task_id in dependencies)

    @staticmethod
    def _blocked_record(task: AssemblyTask) -> TaskExecutionRecord:
        return TaskExecutionRecord(
            task.task_id,
            task.target_part_id,
            task.assigned_arm,
            None,
            None,
            "a preparation dependency failed",
        )

    @staticmethod
    def _failure_text(result: ManipulationResult) -> str:
        if result.failure is None:
            return result.status.value
        return (
            f"{result.status.value}:{result.failure.code.value}:"
            f"{result.failure.stage}:{result.failure.detail}"
        )

    @staticmethod
    def _action_status(result: ManipulationResult) -> str:
        if result.failure is None:
            return result.status.value
        return f"{result.status.value}:{result.failure.code.value}"

    @staticmethod
    def _log(task: AssemblyTask, phase: str, status: str) -> None:
        print(
            "[episode] "
            f"task={task.task_id} target={task.target_part_id} "
            f"arm={task.assigned_arm} phase={phase} status={status}",
            flush=True,
        )

    @staticmethod
    async def _return_home(robot: object, world: _SynchronizedWorld) -> None:
        current = robot.read_state().q
        target = robot.home_configuration
        indices = list(robot.arm_configuration_indices)
        distance = float(np.max(np.abs(target[indices] - current[indices])))
        waypoint_count = max(1, int(np.ceil(distance / 0.04)))
        for index in range(1, waypoint_count + 1):
            fraction = index / waypoint_count
            command = current * (1.0 - fraction) + target * fraction
            if not robot.configuration_is_safe(command):
                raise RuntimeError(f"{robot.robot_id} home path became unsafe")
            robot.command_configuration(command)
            await world.advance(2)
        for _ in range(30):
            robot.command_configuration(target)
            await world.advance(1)
        actual = robot.read_state().q
        error = float(np.max(np.abs(actual[indices] - target[indices])))
        if error > 0.04:
            raise RuntimeError(
                f"{robot.robot_id} failed to return home: joint error={error:.4f}"
            )
