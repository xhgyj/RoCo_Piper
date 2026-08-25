"""Planner-visible manipulation skill contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ManipulationSkillType(Enum):
    """The six manipulation actions exposed to the task planner."""

    PICK = "pick"
    HANDOVER = "handover"
    PLACE_DOWN = "place_down"
    SUPPORT_BOTTOM = "support_bottom"
    PLACE_UP = "place_up"
    SUPPORT_TOP = "support_top"


@dataclass(frozen=True)
class ManipulationAction:
    """One symbolic planner action before geometric grounding."""

    robot_ids: tuple[str, ...]
    skill_type: ManipulationSkillType
    object_id: str | None = None
    goal_id: str | None = None

    def __post_init__(self) -> None:
        """Validate resource cardinality without selecting a robot."""
        expected = 2 if self.skill_type is ManipulationSkillType.HANDOVER else 1
        if len(self.robot_ids) != expected:
            raise ValueError(
                f"{self.skill_type.value} requires {expected} robot(s), "
                f"got {len(self.robot_ids)}"
            )
