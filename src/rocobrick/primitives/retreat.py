"""Straight-line Retreat primitive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rocobrick.execution.types import HeldObject, OperationResult
from rocobrick.primitives.base import PrimitiveContext
from rocobrick.safety.checks import GraspTracking

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RetreatRequest:
    """Target pose and optional held-object state for a safe retreat."""

    world_t_tcp: FloatArray
    grasp_width: float
    held: HeldObject | None = None
    stage: str = "retreat"
    allow_vertical_settling: bool = False


class Retreat:
    """Retreat along a previously grounded safe segment."""

    def __init__(self, context: PrimitiveContext):
        """Bind shared primitive dependencies."""
        self._context = context

    async def execute(self, request: RetreatRequest) -> OperationResult:
        """Execute the retreat and monitor an optional held object.

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
            allow_vertical_settling=request.allow_vertical_settling,
        )
