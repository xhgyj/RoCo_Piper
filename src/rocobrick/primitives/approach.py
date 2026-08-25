"""Straight-line Approach primitive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rocobrick.execution.types import OperationResult
from rocobrick.primitives.base import PrimitiveContext

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ApproachRequest:
    """Target pose and gripper state for a straight approach."""

    world_t_tcp: FloatArray
    grasp_width: float
    stage: str = "approach"


class Approach:
    """Approach a target pose without task-specific semantics."""

    def __init__(self, context: PrimitiveContext):
        """Bind shared primitive dependencies."""
        self._context = context

    async def execute(self, request: ApproachRequest) -> OperationResult:
        """Execute a straight Cartesian approach with an open gripper.

        Returns:
            Controller completion metrics.
        """
        return await self._context.cartesian.move_to(
            request.world_t_tcp,
            request.stage,
            request.grasp_width,
            closed=False,
        )
