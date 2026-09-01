"""Skill execution contracts and registries."""

from typing import TYPE_CHECKING

from rocobrick.execution.types import (
    ActionFailure,
    ActionStatus,
    ExecutionError,
    ExecutionMetrics,
    FailureCode,
    HeldObject,
    HeldObjectState,
    ManipulationResult,
    OperationResult,
)

if TYPE_CHECKING:
    from rocobrick.execution.manipulation_executor import (
        ActionGrounder,
        ManipulationExecutor,
    )


def __getattr__(name: str) -> object:
    """Load the executor lazily to avoid backend/planning import cycles.

    Returns:
        Requested public executor class.
    """
    if name in {"ActionGrounder", "ManipulationExecutor"}:
        from rocobrick.execution.manipulation_executor import (
            ActionGrounder,
            ManipulationExecutor,
        )

        return {
            "ActionGrounder": ActionGrounder,
            "ManipulationExecutor": ManipulationExecutor,
        }[name]
    raise AttributeError(name)


__all__ = [
    "ActionFailure",
    "ActionGrounder",
    "ActionStatus",
    "ExecutionMetrics",
    "ExecutionError",
    "FailureCode",
    "HeldObject",
    "HeldObjectState",
    "ManipulationExecutor",
    "ManipulationResult",
    "OperationResult",
]
