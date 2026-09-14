"""BrickSim/Isaac Sim adapter for measured trajectory playback.

This module intentionally does not import or start Isaac Sim at import time.
The application, world, and articulations are injected by the run bootstrap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Protocol

from rocobrick.gripper import logical_gripper_to_fingers
from rocobrick.trajectory_execution import MeasuredState


class Articulation(Protocol):
    @property
    def dof_names(self) -> list[str] | tuple[str, ...]: ...
    def get_joint_positions(self) -> object: ...
    def apply_action(self, action: object) -> None: ...


class SimulationWorld(Protocol):
    @property
    def current_time(self) -> float: ...


class SimulationApplication(Protocol):
    async def next_update_async(self) -> None: ...


ActionFactory = Callable[[tuple[float, ...], tuple[int, ...]], object]
HoldingObserver = Callable[[str], str | None]
SafetyObserver = Callable[[str], bool]


@dataclass(frozen=True)
class RobotBinding:
    articulation: Articulation
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, str] = ("gripper_joint1", "gripper_joint2")


def isaac_action_factory(positions: tuple[float, ...], indices: tuple[int, ...]) -> object:
    """Create an Isaac action lazily after SimulationApp has initialized extensions."""
    from isaacsim.core.utils.types import ArticulationAction

    return ArticulationAction(joint_positions=positions, joint_indices=indices)


class BrickSimTrajectoryPort:
    """Measured-state owner for trajectory commands sent to BrickSim only."""

    def __init__(
        self,
        app: SimulationApplication,
        world: SimulationWorld,
        robots: Mapping[str, RobotBinding],
        holding_observer: HoldingObserver,
        safety_observer: SafetyObserver,
        *,
        action_factory: ActionFactory = isaac_action_factory,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._app = app
        self._world = world
        self._robots = dict(robots)
        self._holding = holding_observer
        self._safety = safety_observer
        self._action_factory = action_factory
        self._monotonic = monotonic
        self._last_step_wall_s = monotonic()
        self._indices: dict[str, dict[str, int]] = {}
        for robot_id, binding in self._robots.items():
            names = tuple(binding.articulation.dof_names)
            if len(names) != len(set(names)):
                raise ValueError(f"{robot_id} articulation has duplicate DOF names")
            lookup = {name: index for index, name in enumerate(names)}
            missing = (set(binding.arm_joint_names) | set(binding.gripper_joint_names)) - set(lookup)
            if missing:
                raise ValueError(f"{robot_id} is missing configured joints: {sorted(missing)}")
            self._indices[robot_id] = lookup

    def measured_gripper_opening(self, robot_id: str) -> float:
        first, second = self.measured_gripper_fingers(robot_id)
        opening = first - second
        if not -0.002 <= opening <= 0.102:
            raise ValueError("measured gripper opening is outside the physical range")
        return max(0.0, min(0.1, opening))

    def measured_gripper_fingers(self, robot_id: str) -> tuple[float, float]:
        binding = self._robots[robot_id]
        positions = binding.articulation.get_joint_positions()
        first, second = binding.gripper_joint_names
        return (
            float(positions[self._indices[robot_id][first]]),
            float(positions[self._indices[robot_id][second]]),
        )

    def command_gripper_opening(self, robot_id: str, opening_m: float) -> None:
        binding = self._robots[robot_id]
        positions = logical_gripper_to_fingers(opening_m)
        indices = tuple(self._indices[robot_id][name] for name in binding.gripper_joint_names)
        binding.articulation.apply_action(self._action_factory(positions, indices))

    def measured_state(self, robot_id: str) -> MeasuredState:
        binding = self._robots[robot_id]
        positions = binding.articulation.get_joint_positions()
        joints = tuple(
            float(positions[self._indices[robot_id][name]])
            for name in binding.arm_joint_names
        )
        return MeasuredState(
            simulation_time_s=float(self._world.current_time),
            joints=joints,
            feedback_age_s=max(0.0, self._monotonic() - self._last_step_wall_s),
            holding_part=self._holding(robot_id),
            safety_valid=self._safety(robot_id),
        )

    def command_joint_positions(
        self,
        robot_id: str,
        joint_names: tuple[str, ...],
        positions: tuple[float, ...],
    ) -> None:
        binding = self._robots[robot_id]
        expected = binding.arm_joint_names
        normalized = tuple(name.removeprefix(f"{robot_id}_") for name in joint_names)
        if normalized != expected:
            raise ValueError(
                f"trajectory joints for {robot_id} must exactly equal configured arm order"
            )
        indices = tuple(self._indices[robot_id][name] for name in expected)
        binding.articulation.apply_action(self._action_factory(positions, indices))

    async def step(self) -> None:
        await self._app.next_update_async()
        self._last_step_wall_s = self._monotonic()

    def controlled_stop(self, robot_id: str) -> None:
        state = self.measured_state(robot_id)
        binding = self._robots[robot_id]
        indices = tuple(self._indices[robot_id][name] for name in binding.arm_joint_names)
        binding.articulation.apply_action(self._action_factory(state.joints, indices))
