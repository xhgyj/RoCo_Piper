"""Free-space Move primitive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rocobrick.execution.types import OperationResult
from rocobrick.primitives.base import PrimitiveContext

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MoveRequest:
    """One free-space joint target and optional TCP postcondition."""

    goal_q: FloatArray
    stage: str
    world_t_tcp: FloatArray | None = None


class Move:
    """Move the robot through free space to a verified target."""

    def __init__(self, context: PrimitiveContext):
        """Bind shared primitive dependencies."""
        self._context = context

    async def execute(self, request: MoveRequest) -> OperationResult:
        """Execute the configured joint target.

        Returns:
            Controller completion metrics.
        """
        return await self._context.trajectory.move_to(
            request.goal_q, request.stage, request.world_t_tcp
        )
