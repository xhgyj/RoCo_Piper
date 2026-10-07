"""Versioned, fail-closed policy for coordinated bottom handovers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROFILE_SCHEMA = "rocobrick-handover-profile/v1"
BOTTOM_PROFILE_ID = "handover-bottom-coordinated-v1"
PLANNING_CAPABILITIES = frozenset(
    {
        "combined_joint_planning",
        "continuous_path_validation",
        "combined_state_validity",
        "gripper_sweep_validation",
        "cartesian_approach_validation",
        "separation_validation",
    }
)
EXECUTION_CAPABILITIES = PLANNING_CAPABILITIES | frozenset(
    {
        "named_pad_contacts",
        "synchronized_start_feedback",
        "constrained_cartesian_execution",
        "provisional_custody",
        "prepared_motion_execution",
    }
)


def finite(value: object, label: str, *, positive: bool = False) -> float:
    """Validate a finite numeric policy value without accepting bool.

    Returns:
        The validated floating-point value.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(
            f"{label} must be a finite {'positive ' if positive else ''}number"
        )
    return float(value)


def digest(value: object, label: str) -> str:
    """Validate a lowercase SHA-256 binding.

    Returns:
        The validated digest.
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


@dataclass(frozen=True)
class HandoverProfile:
    """Frozen planning policy; qualification is never inferred from IK success."""

    profile_id: str
    runtime_model_sha256: str
    execution_qualified: bool
    forward_offsets_m: tuple[float, ...]
    object_heights_m: tuple[float, ...]
    object_yaws_rad: tuple[float, ...]
    object_tilts_rad: tuple[float, ...]
    receiver_yaws_rad: tuple[float, ...]
    receiver_tilts_rad: tuple[float, ...]
    precontact_distance_m: float
    contact_clearance_m: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    relative_position_tolerance_m: float
    relative_orientation_tolerance_rad: float
    observation_noise_m: float
    verification_distance_m: float
    min_observation_span_s: float
    max_observation_gap_s: float
    max_scene_age_s: float
    planning_timeout_s: float
    max_candidates: int
    max_evaluations: int
    max_sync_skew_s: float
    joint_hold_tolerance_rad: float
    contact_position_tolerance_m: float = 0.0001
    receiver_lateral_offsets_m: tuple[float, ...] = (0.0,)
    receiver_depth_offsets_m: tuple[float, ...] = (0.0,)

    def __post_init__(self) -> None:
        """Reject malformed grids, unsafe evidence budgets and mutable policy."""
        if self.profile_id != BOTTOM_PROFILE_ID:
            raise ValueError("unsupported coordinated handover profile identity")
        digest(self.runtime_model_sha256, "runtime_model_sha256")
        if type(self.execution_qualified) is not bool:
            raise ValueError("execution_qualified must be a real bool")
        grids = (
            "forward_offsets_m",
            "object_heights_m",
            "object_yaws_rad",
            "object_tilts_rad",
            "receiver_yaws_rad",
            "receiver_tilts_rad",
            "receiver_lateral_offsets_m",
            "receiver_depth_offsets_m",
        )
        for name in grids:
            raw = getattr(self, name)
            if not isinstance(raw, (list, tuple)) or not raw:
                raise ValueError(f"{name} must be a nonempty numeric grid")
            values = tuple(finite(v, name) for v in raw)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)
        grid_size = (
            (1 + len(self.forward_offsets_m) * len(self.object_heights_m))
            * len(self.object_yaws_rad)
            * len(self.object_tilts_rad)
            * len(self.receiver_yaws_rad)
            * len(self.receiver_tilts_rad)
            * len(self.receiver_lateral_offsets_m)
            * len(self.receiver_depth_offsets_m)
        )
        if grid_size > 65536:
            raise ValueError(
                "handover candidate grid exceeds bounded generation budget"
            )
        if any(v <= 0 for v in self.forward_offsets_m + self.object_heights_m):
            raise ValueError("workspace offsets and heights must be positive")
        if any(abs(v) > math.pi for v in self.object_yaws_rad):
            raise ValueError("object yaw must be bounded by pi")
        if any(abs(v) > math.pi / 3 for v in self.object_tilts_rad):
            raise ValueError("object tilt must be bounded by pi/3")
        if any(v < 0 or v >= math.pi / 2 for v in self.receiver_tilts_rad):
            raise ValueError("receiver tilt must preserve a bottom approach")
        if any(abs(v) > 0.01 for v in self.receiver_lateral_offsets_m):
            raise ValueError("receiver lateral offset exceeds 10 mm")
        if any(not 0 <= v <= 0.008 for v in self.receiver_depth_offsets_m):
            raise ValueError("receiver depth offset must be in [0, 8 mm]")
        # Box face normals: arbitrary yaw is not a proven contact surface.
        if any(
            abs(v / (math.pi / 2) - round(v / (math.pi / 2))) > 1e-8
            for v in self.receiver_yaws_rad
        ):
            raise ValueError("receiver yaw must select a box face normal")
        for name in (
            "precontact_distance_m",
            "contact_clearance_m",
            "position_tolerance_m",
            "orientation_tolerance_rad",
            "relative_position_tolerance_m",
            "relative_orientation_tolerance_rad",
            "observation_noise_m",
            "verification_distance_m",
            "min_observation_span_s",
            "max_observation_gap_s",
            "planning_timeout_s",
            "max_scene_age_s",
            "max_sync_skew_s",
            "joint_hold_tolerance_rad",
            "contact_position_tolerance_m",
        ):
            object.__setattr__(
                self, name, finite(getattr(self, name), name, positive=True)
            )
        for name in ("max_candidates", "max_evaluations"):
            v = getattr(self, name)
            if type(v) is not int or not 1 <= v <= 4096:
                raise ValueError(f"{name} must be an integer in [1, 4096]")
        if self.max_evaluations > self.max_candidates:
            raise ValueError("max_evaluations exceeds candidate budget")
        if self.verification_distance_m <= 3 * self.observation_noise_m:
            raise ValueError("verification motion must exceed three times the noise")
        if self.contact_clearance_m >= self.precontact_distance_m:
            raise ValueError("precontact distance must exceed contact clearance")
        if self.contact_position_tolerance_m > 0.0005:
            raise ValueError("contact position tolerance exceeds 0.5 mm")

    @property
    def sha256(self) -> str:
        """Return a canonical digest of the complete frozen policy."""
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    @classmethod
    def parse(cls, raw: object) -> HandoverProfile:
        """Parse an exact-key profile, never silently dropping constraints.

        Returns:
            A validated immutable profile.
        """
        keys = {f.name for f in fields(cls)} | {"schema"}
        optional = {
            "contact_position_tolerance_m",
            "receiver_lateral_offsets_m",
            "receiver_depth_offsets_m",
        }
        if not isinstance(raw, Mapping) or not keys - optional <= set(raw) <= keys:
            raise ValueError("handover profile keys are incomplete or unknown")
        if raw["schema"] != PROFILE_SCHEMA:
            raise ValueError("unsupported handover profile schema")
        return cls(**{name: raw[name] for name in set(raw) - {"schema"}})


def resolve_handover_profile(profile_id: str) -> HandoverProfile:
    """Resolve only an explicitly registered, packaged strategy.

    Returns:
        Its strict policy, including its qualification flag.
    """
    if profile_id != BOTTOM_PROFILE_ID:
        raise ValueError(f"UNSUPPORTED_HANDOVER_PROFILE: {profile_id!r}")
    path = Path(__file__).with_name("profiles") / f"{BOTTOM_PROFILE_ID}.json"
    return HandoverProfile.parse(json.loads(path.read_text(encoding="utf-8")))


def require_handover_capabilities(
    profile: HandoverProfile,
    available: object,
    *,
    physical: bool,
) -> None:
    """Reject unsupported phases before any controller or gripper dispatch."""
    if not isinstance(available, (list, tuple, set, frozenset)) or any(
        not isinstance(x, str) for x in available
    ):
        raise ValueError("handover capabilities must be explicit names")
    missing = (EXECUTION_CAPABILITIES if physical else PLANNING_CAPABILITIES) - set(
        available
    )
    if missing:
        raise ValueError(
            "HANDOVER_CAPABILITY_UNAVAILABLE: " + ", ".join(sorted(missing))
        )
    if physical and not profile.execution_qualified:
        raise ValueError("HANDOVER_PROFILE_NOT_QUALIFIED: offline planning only")
