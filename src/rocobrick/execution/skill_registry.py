"""Capability registry for planner-visible manipulation skills."""

from __future__ import annotations

from dataclasses import dataclass

from rocobrick.skills.base_skill import ManipulationSkillType


@dataclass(frozen=True)
class SkillCapability:
    """Availability and resource requirements for one skill."""

    skill_type: ManipulationSkillType
    available: bool
    robot_count: int


class SkillRegistry:
    """Expose the standard skill vocabulary and current rollout state."""

    def __init__(self, capabilities: tuple[SkillCapability, ...]):
        """Build a complete registry without duplicate skill entries."""
        self._capabilities = {
            capability.skill_type: capability for capability in capabilities
        }
        if len(self._capabilities) != len(capabilities):
            raise ValueError("skill registry contains duplicate entries")
        missing = set(ManipulationSkillType) - set(self._capabilities)
        if missing:
            names = sorted(item.value for item in missing)
            raise ValueError(f"skill registry is missing {names}")

    @classmethod
    def phase_one(cls) -> SkillRegistry:
        """Return the first rollout with only Pick available."""
        return cls(
            tuple(
                SkillCapability(
                    skill_type,
                    available=skill_type is ManipulationSkillType.PICK,
                    robot_count=(
                        2 if skill_type is ManipulationSkillType.HANDOVER else 1
                    ),
                )
                for skill_type in ManipulationSkillType
            )
        )

    def require_available(self, skill_type: ManipulationSkillType) -> None:
        """Reject unavailable capabilities before plan execution."""
        capability = self._capabilities[skill_type]
        if not capability.available:
            raise ValueError(f"skill {skill_type.value} is not available")

    def capability(self, skill_type: ManipulationSkillType) -> SkillCapability:
        """Return one immutable capability descriptor."""
        return self._capabilities[skill_type]
