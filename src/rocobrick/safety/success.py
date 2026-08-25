"""Reusable semantic success-check implementation."""

from __future__ import annotations

from collections.abc import Callable

from rocobrick.execution.types import ExecutionError, FailureCode


class PredicateSuccessCheck:
    """Adapt task-specific predicates to the shared SuccessCheck protocol."""

    def __init__(
        self,
        satisfied: Callable[[], bool],
        conflict: Callable[[], str | None] | None = None,
    ):
        """Store live success and optional conflict predicates."""
        self._satisfied = satisfied
        self._conflict = conflict or (lambda: None)

    def is_satisfied(self) -> bool:
        """Evaluate the live semantic postcondition.

        Returns:
            Whether the requested condition is currently active.
        """
        return bool(self._satisfied())

    def conflict(self) -> str | None:
        """Evaluate the live terminal-conflict predicate.

        Returns:
            Conflict detail, or None when no conflict exists.
        """
        return self._conflict()

    def require_satisfied(self, stage: str) -> None:
        """Reject a conflict or unsatisfied postcondition."""
        conflict = self.conflict()
        if conflict is not None:
            raise ExecutionError(FailureCode.VERIFICATION_FAILED, stage, conflict)
        if not self.is_satisfied():
            raise ExecutionError(
                FailureCode.VERIFICATION_FAILED,
                stage,
                "semantic success condition is not satisfied",
            )
