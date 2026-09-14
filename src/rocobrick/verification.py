"""Separate observable execution confirmation from privileged truth validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ObservableConfirmation:
    grasped: bool
    holding: bool
    connected: bool
    released: bool
    retreated: bool
    feedback_age_s: float


@dataclass(frozen=True)
class TruthValidation:
    target_pose_valid: bool
    target_connection_valid: bool
    stable_for_s: float
    unexpected_contacts: tuple[str, ...]


class ObservableVerifier(Protocol):
    def sample(self, task_id: str) -> ObservableConfirmation: ...


class PrivilegedTruthVerifier(Protocol):
    def validate(self, task_id: str) -> TruthValidation: ...


def assembly_confirmed(observation: ObservableConfirmation) -> bool:
    return observation.connected and observation.released and observation.feedback_age_s <= 0.25


def final_truth_valid(validation: TruthValidation, settle_time_s: float = 2.0) -> bool:
    return (
        validation.target_pose_valid
        and validation.target_connection_valid
        and validation.stable_for_s >= settle_time_s
        and not validation.unexpected_contacts
    )
