"""Unified dual-arm successor pre-phase contracts.

A successor pre-phase owns the complete free-space transition from the
predecessor's freshly admitted terminal state to the next skill's contact
boundary. It is not a separate user-visible operation and it never includes
the next skill's contact/release phase.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from rocobrick.robot_operation_plan import RobotOperation

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class SuccessorPrePhaseRequest:
    """One next-skill pre-phase transition request."""

    predecessor_operation_id: str
    successor_operation_ids: tuple[str, ...]
    predecessor_state: Mapping[str, JSONValue]
    successor_operations: tuple[RobotOperation, ...]
    participants: tuple[str, ...]
    scene_digest: str
    resources: tuple[str, ...]
    terminal_cleanup: bool = False

    def __post_init__(self) -> None:
        """Validate identity, participant coverage and scene provenance."""
        if not self.predecessor_operation_id:
            raise ValueError("predecessor_operation_id must be non-empty")
        if not self.successor_operation_ids and not self.terminal_cleanup:
            raise ValueError("successor operation ids are required unless terminal cleanup")
        if len(self.successor_operation_ids) != len(set(self.successor_operation_ids)):
            raise ValueError("successor_operation_ids must be unique and non-empty")
        if tuple(item.operation_id for item in self.successor_operations) != (
            self.successor_operation_ids
        ):
            raise ValueError("successor operation identities do not match")
        if not self.participants or len(self.participants) != len(set(self.participants)):
            raise ValueError("pre-phase participants must be unique and non-empty")
        if not self.scene_digest:
            raise ValueError("pre-phase scene_digest must be non-empty")
        if not self.resources or len(self.resources) != len(set(self.resources)):
            raise ValueError("pre-phase resources must be unique and non-empty")
        if not isinstance(self.predecessor_state, Mapping):
            raise ValueError("predecessor_state must be a mapping")


@dataclass(frozen=True)
class SuccessorPrePhaseEvidence:
    """Terminal-safe evidence returned by the unified runtime."""

    predecessor_operation_id: str
    successor_operation_ids: tuple[str, ...]
    participants: tuple[str, ...]
    scene_digest: str
    planned: bool
    executed: bool
    terminal: bool
    safe: bool
    resource_transfer_confirmed: bool
    evidence: Mapping[str, JSONValue]
    terminal_cleanup: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate explicit lifecycle booleans and JSON-safe evidence."""
        if not self.predecessor_operation_id:
            raise ValueError("predecessor_operation_id must be non-empty")
        if not self.successor_operation_ids and not self.terminal_cleanup:
            raise ValueError("successor_operation_ids must be non-empty")
        if not self.participants or len(self.participants) != len(set(self.participants)):
            raise ValueError("pre-phase participants must be unique")
        if not self.scene_digest:
            raise ValueError("pre-phase scene_digest must be non-empty")
        for name in (
            "planned",
            "executed",
            "terminal",
            "safe",
            "resource_transfer_confirmed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"pre-phase {name} must be a real bool")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("pre-phase evidence must be a mapping")
        if not _json_safe(self.evidence):
            raise ValueError("pre-phase evidence must be JSON-safe")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("pre-phase reason must be str or None")

    @property
    def passed(self) -> bool:
        """True only after planning, execution, safety and transfer proof."""
        return (
            self.planned
            and self.executed
            and self.terminal
            and self.safe
            and self.resource_transfer_confirmed
        )


def _json_safe(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _json_safe(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_json_safe(item) for item in value)
    return False


class UnifiedDualArmRuntime(Protocol):
    """Single runtime for combined planning, execution and resource transfer."""

    async def execute_pre_phase(
        self, request: SuccessorPrePhaseRequest
    ) -> SuccessorPrePhaseEvidence: ...

    async def cancel(self, predecessor_operation_id: str) -> Mapping[str, object]: ...


__all__ = [
    "SuccessorPrePhaseEvidence",
    "SuccessorPrePhaseRequest",
    "UnifiedDualArmRuntime",
]
