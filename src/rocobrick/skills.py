"""Explicit Skill phase protocols; failures never imply a different action."""

from __future__ import annotations

from dataclasses import dataclass

from rocobrick.compat import StrEnum
from rocobrick.models import ActionType, FailureReason, SkillType


class PhaseStatus(StrEnum):
    READY = "READY"
    ACTIVE = "ACTIVE"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class SkillPhase:
    phase_id: str
    action: ActionType
    confirmation: str


@dataclass(frozen=True)
class SkillProgram:
    skill: SkillType
    phases: tuple[SkillPhase, ...]


PICK_PROGRAM = SkillProgram(
    SkillType.PICK,
    (
        SkillPhase("plan_grasp", ActionType.PICK, "collision_checked_plan"),
        SkillPhase("reach_pregrasp", ActionType.PICK, "measured_pose"),
        SkillPhase("approach", ActionType.PICK, "measured_pose"),
        SkillPhase("close_gripper", ActionType.PICK, "measured_gripper"),
        SkillPhase("confirm_grasp", ActionType.PICK, "observable_grasp"),
        SkillPhase("lift", ActionType.PICK, "measured_pose"),
        SkillPhase("confirm_holding", ActionType.PICK, "observable_holding"),
    ),
)

PLACE_DOWN_PROGRAM = SkillProgram(
    SkillType.PLACE_DOWN,
    (
        SkillPhase("transport", ActionType.TRANSPORT, "measured_pose"),
        SkillPhase("align", ActionType.ALIGN, "measured_alignment"),
        SkillPhase("insert_segments", ActionType.INSERT, "force_pose_envelope"),
        SkillPhase("confirm_connection", ActionType.VERIFY_CONNECTION, "observable_connection"),
        SkillPhase("release", ActionType.RELEASE, "measured_gripper"),
        SkillPhase("confirm_release", ActionType.RELEASE, "observable_release"),
        SkillPhase("publish_assembled", ActionType.VERIFY_CONNECTION, "connection_and_release"),
        SkillPhase("retreat", ActionType.RETREAT, "measured_pose"),
    ),
)

PROGRAMS = {SkillType.PICK: PICK_PROGRAM, SkillType.PLACE_DOWN: PLACE_DOWN_PROGRAM}


@dataclass(frozen=True)
class PhaseResult:
    phase: SkillPhase
    status: PhaseStatus
    reason: FailureReason | None = None

    def __post_init__(self) -> None:
        if self.status == PhaseStatus.FAILED and self.reason is None:
            raise ValueError("a failed phase requires an explicit reason")
        if self.status != PhaseStatus.FAILED and self.reason is not None:
            raise ValueError("only a failed phase may carry a failure reason")


def program_for(skill: SkillType) -> SkillProgram:
    try:
        return PROGRAMS[skill]
    except KeyError as exc:
        raise RuntimeError(f"CAPABILITY_UNAVAILABLE: {skill}") from exc
