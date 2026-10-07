"""Explicit simulation trial admission, separate from physical qualification."""

from __future__ import annotations

from dataclasses import dataclass, replace

from rocobrick.skills.handover_profile import (
    EXECUTION_CAPABILITIES,
    HandoverProfile,
    digest,
    require_handover_capabilities,
    resolve_handover_profile,
)


def simulation_profile(profile_id: str) -> HandoverProfile:
    """Use the user's bounded contact allowance without qualifying hardware.

    Returns:
        A planning profile with the explicit bounded simulation allowance.
    """
    return replace(
        resolve_handover_profile(profile_id),
        contact_position_tolerance_m=0.0005,
        max_candidates=1024,
        max_evaluations=1024,
        planning_timeout_s=120.0,
        receiver_lateral_offsets_m=(0.0, 0.004, -0.004, 0.008, -0.008),
        receiver_depth_offsets_m=(0.0, 0.002, 0.004),
    )


@dataclass(frozen=True)
class SimulationHandoverAuthorization:
    """One explicitly opted-in Isaac session/model/profile/manifest binding."""

    session_id: str
    manifest_sha256: str
    profile_sha256: str
    runtime_model_sha256: str
    finger_contact_policy_sha256: str | None = None

    def __post_init__(self) -> None:
        """Require auditable exact identities, never a global qualified flag."""
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("simulation session identity is required")
        for value in (
            self.manifest_sha256,
            self.profile_sha256,
            self.runtime_model_sha256,
        ):
            digest(value, "simulation authorization")
        if self.finger_contact_policy_sha256 is not None:
            digest(
                self.finger_contact_policy_sha256, "simulation finger contact policy"
            )

    def validate(self, profile, snapshot, capabilities) -> None:
        """Keep every runtime capability gate and bind the trial to its inputs."""
        require_handover_capabilities(profile, capabilities, physical=False)
        if not EXECUTION_CAPABILITIES <= set(capabilities):
            raise ValueError("simulation trial lacks complete execution capabilities")
        if (
            snapshot.get("session_id") != self.session_id
            or snapshot.get("manifest_sha256") != self.manifest_sha256
            or profile.sha256 != self.profile_sha256
            or profile.runtime_model_sha256 != self.runtime_model_sha256
        ):
            raise ValueError("simulation authorization belongs to different inputs")
