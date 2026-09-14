"""Skill capability declarations; unsupported skills never degrade silently."""

from __future__ import annotations

from dataclasses import dataclass

from rocobrick.models import ActionType, SkillType


@dataclass(frozen=True)
class SkillCapability:
    skill: SkillType
    available: bool
    phases: tuple[ActionType, ...]


CAPABILITIES = {
    SkillType.PICK: SkillCapability(SkillType.PICK, True, (ActionType.PICK,)),
    SkillType.PLACE_DOWN: SkillCapability(
        SkillType.PLACE_DOWN,
        True,
        (
            ActionType.TRANSPORT,
            ActionType.ALIGN,
            ActionType.INSERT,
            ActionType.VERIFY_CONNECTION,
            ActionType.RELEASE,
            ActionType.RETREAT,
        ),
    ),
    SkillType.HANDOVER: SkillCapability(SkillType.HANDOVER, False, ()),
    SkillType.SUPPORT_BOTTOM: SkillCapability(SkillType.SUPPORT_BOTTOM, False, ()),
    SkillType.PLACE_UP: SkillCapability(SkillType.PLACE_UP, False, ()),
    SkillType.SUPPORT_TOP: SkillCapability(SkillType.SUPPORT_TOP, False, ()),
}
