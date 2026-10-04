"""Unified dual-arm readiness evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class UnifiedDualArmReadiness:
    """Readiness for one combined planner and its two controller participants."""

    robots: tuple[str, ...]
    clock_increasing: bool
    combined_joint_state_complete: bool
    scene_snapshot_fresh: bool
    scene_digest_match: bool
    combined_move_group_ready: bool
    controllers_ready: Mapping[str, bool]

    @property
    def passed(self) -> bool:
        """True only when all unified planning/execution gates hold."""
        return bool(
            self.robots
            and self.clock_increasing
            and self.combined_joint_state_complete
            and self.scene_snapshot_fresh
            and self.scene_digest_match
            and self.combined_move_group_ready
            and self.controllers_ready
            and all(self.controllers_ready.values())
        )

    def serialize(self) -> dict[str, object]:
        """Return JSON-safe unified readiness evidence."""
        return {
            "robots": list(self.robots),
            "passed": self.passed,
            "components": {
                "clock_increasing": self.clock_increasing,
                "combined_joint_state_complete": self.combined_joint_state_complete,
                "scene_snapshot_fresh": self.scene_snapshot_fresh,
                "scene_digest_match": self.scene_digest_match,
                "combined_move_group_ready": self.combined_move_group_ready,
                "controllers_ready": dict(self.controllers_ready),
            },
        }


__all__ = ["UnifiedDualArmReadiness"]
