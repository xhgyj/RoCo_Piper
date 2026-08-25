"""Semantically guarded Release primitive."""

from __future__ import annotations

from rocobrick.execution.types import OperationResult
from rocobrick.primitives.base import PrimitiveContext
from rocobrick.primitives.gripper import Grasp
from rocobrick.safety.checks import SuccessCheck


class Release:
    """Open the gripper only while the task postcondition is active."""

    def __init__(self, context: PrimitiveContext):
        """Bind shared primitive dependencies."""
        self._gripper = Grasp(context)

    async def execute(
        self,
        object_width: float,
        success_check: SuccessCheck,
    ) -> OperationResult:
        """Verify, open, and verify the task postcondition again.

        Returns:
            Gripper completion metrics.
        """
        success_check.require_satisfied("release")
        result = await self._gripper.prepare(
            object_width, allow_constrained=True
        )
        success_check.require_satisfied("release")
        return result
