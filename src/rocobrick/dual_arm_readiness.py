"""Strict readiness evidence for the combined dual-arm MoveIt instance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from rocobrick.coordinated_moveit import CoordinatedMoveItError


@dataclass(frozen=True)
class DualArmReadiness:
    """All participant and planning-scene evidence for coordinated planning."""

    coordination_id: str
    scene_digest: str
    scene_sequence: int
    scene_sim_time_s: float
    participants: tuple[str, ...]
    move_group_action: str
    planning_group: str
    components: Mapping[str, bool]

    def __post_init__(self) -> None:
        """Validate identity, scene evidence, and readiness components."""
        if not isinstance(self.coordination_id, str) or not self.coordination_id:
            raise CoordinatedMoveItError("coordination_id must be non-empty")
        if not isinstance(self.scene_digest, str) or not self.scene_digest:
            raise CoordinatedMoveItError("scene_digest must be non-empty")
        if type(self.scene_sequence) is not int or self.scene_sequence < 0:
            raise CoordinatedMoveItError("scene_sequence must be non-negative int")
        if (
            not isinstance(self.scene_sim_time_s, (int, float))
            or self.scene_sim_time_s < 0
        ):
            raise CoordinatedMoveItError("scene_sim_time_s must be non-negative")
        participants = tuple(self.participants)
        if not participants or len(participants) != len(set(participants)):
            raise CoordinatedMoveItError("readiness participants must be unique")
        if not isinstance(self.move_group_action, str) or not self.move_group_action:
            raise CoordinatedMoveItError("move_group_action must be non-empty")
        if not isinstance(self.planning_group, str) or not self.planning_group:
            raise CoordinatedMoveItError("planning_group must be non-empty")
        if not isinstance(self.components, Mapping) or not all(
            isinstance(key, str) and isinstance(value, bool)
            for key, value in self.components.items()
        ):
            raise CoordinatedMoveItError("readiness components must be bool mapping")
        if not all(self.components.values()):
            raise CoordinatedMoveItError(
                "all coordinated readiness components must pass"
            )
        object.__setattr__(self, "participants", participants)


def require_group_readiness(
    document: Mapping[str, object],
    *,
    coordination_id: str,
    participants: Sequence[str],
    scene_digest: str,
) -> DualArmReadiness:
    """Parse one exact combined MoveIt readiness document.

    Returns:
        Frozen readiness evidence for the requested coordination identity.

    Raises:
        CoordinatedMoveItError: When identity, scene or readiness evidence does
            not match the request.
    """
    if not isinstance(document, Mapping):
        raise CoordinatedMoveItError("dual-arm readiness must be an object")
    if document.get("coordination_id") != coordination_id:
        raise CoordinatedMoveItError("dual-arm readiness coordination id mismatch")
    if tuple(document.get("participants", ())) != tuple(participants):
        raise CoordinatedMoveItError("dual-arm readiness participants mismatch")
    if document.get("scene_digest") != scene_digest:
        raise CoordinatedMoveItError("dual-arm readiness scene digest mismatch")
    return DualArmReadiness(
        coordination_id=coordination_id,
        scene_digest=scene_digest,
        scene_sequence=document.get("scene_sequence"),
        scene_sim_time_s=document.get("scene_sim_time_s"),
        participants=tuple(document.get("participants")),
        move_group_action=document.get("move_group_action"),
        planning_group=document.get("planning_group"),
        components=document.get("components"),
    )
