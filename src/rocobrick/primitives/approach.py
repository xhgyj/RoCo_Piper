"""Straight-line Approach primitive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rocobrick.execution.types import HeldObject, OperationResult
from rocobrick.primitives.base import PrimitiveContext
from rocobrick.safety.checks import GraspTracking, SuccessCheck

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ApproachRequest:
    """Target pose and gripper state for a straight approach."""

    world_t_tcp: FloatArray
    grasp_width: float
    stage: str = "approach"
    held: HeldObject | None = None
    success_check: SuccessCheck | None = None


class Approach:
    """Approach a target pose without task-specific semantics."""

    def __init__(self, context: PrimitiveContext):
        """Bind shared primitive dependencies."""
        self._context = context

    async def execute(self, request: ApproachRequest) -> OperationResult:
        """Execute a straight Cartesian approach.

        Returns:
            Controller completion metrics.
        """
        tracking = None
        if request.held is not None:
            tracking = GraspTracking(
                request.held.object_id,
                request.held.object_t_tcp,
                request.held.grasp_axis,
                request.held.grasp_width,
            )
        return await self._context.cartesian.move_to(
            request.world_t_tcp,
            request.stage,
            request.grasp_width,
            closed=request.held is not None,
            tracking=tracking,
            success_check=request.success_check,
        )
