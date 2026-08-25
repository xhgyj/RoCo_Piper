"""Dependencies shared by manipulation primitives."""

from __future__ import annotations

from dataclasses import dataclass

from rocobrick.backends.base import RobotBackend, WorldModel
from rocobrick.controllers.motion import CartesianController, TrajectoryController
from rocobrick.safety.checks import CollisionCheck


@dataclass(frozen=True)
class PrimitiveContext:
    """Controllers and feedback available to every primitive."""

    robot: RobotBackend
    world: WorldModel
    trajectory: TrajectoryController
    cartesian: CartesianController
    collision: CollisionCheck
