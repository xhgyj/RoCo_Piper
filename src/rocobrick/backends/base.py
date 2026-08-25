"""Protocols exposed to controllers and manipulation skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RobotState:
    """One backend-independent robot feedback sample."""

    q: FloatArray
    tcp_world: FloatArray
    gripper_positions: FloatArray

    @property
    def gripper_width(self) -> float:
        """Return the summed absolute finger displacement."""
        return float(np.sum(np.abs(self.gripper_positions)))


class RobotBackend(Protocol):
    """Robot operations required by shared controllers."""

    @property
    def robot_id(self) -> str:
        """Return the stable robot identifier."""
        ...

    @property
    def home_configuration(self) -> FloatArray:
        """Return a copy of the configured home state."""
        ...

    @property
    def arm_configuration_indices(self) -> tuple[int, ...]:
        """Return configuration indices controlled as arm joints."""
        ...

    @property
    def gripper_configuration_indices(self) -> tuple[int, ...]:
        """Return configuration indices controlled as gripper joints."""
        ...

    def read_state(self) -> RobotState:
        """Read the current robot state."""
        ...

    def solve_ik(
        self, world_t_tcp: FloatArray, seed: FloatArray
    ) -> FloatArray | None:
        """Return a verified IK solution, or None when unreachable."""
        ...

    def command_configuration(self, q: FloatArray) -> None:
        """Command one full robot configuration."""
        ...

    def configuration_is_safe(self, q: FloatArray) -> bool:
        """Return whether a configuration passes backend safety checks."""
        ...

    def with_gripper(
        self, q: FloatArray, object_width: float, closed: bool
    ) -> FloatArray:
        """Return a configuration with the requested gripper state."""
        ...


class WorldModel(Protocol):
    """World feedback required by reusable manipulation primitives."""

    def object_pose(self, object_id: str) -> FloatArray:
        """Return an object's current world transform."""
        ...

    async def advance(self, steps: int = 1) -> None:
        """Advance or wait for the requested control cycles."""
        ...
