"""Synchronized dual-arm target generation for exploratory HANDOVER planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ArmWaypoint:
    time_from_start_s: float
    giver: Mapping[str, object]
    receiver: Mapping[str, object]


def _lerp(a: Sequence[float], b: Sequence[float], u: float) -> list[float]:
    return [float(x + (y - x) * u) for x, y in zip(a, b)]


def synchronized_waypoints(
    *,
    giver_start: Mapping[str, object],
    giver_lift: Mapping[str, object],
    receiver_start: Mapping[str, object],
    receiver_precontact: Mapping[str, object],
    duration_s: float = 3.0,
    steps: int = 12,
) -> tuple[ArmWaypoint, ...]:
    """Interpolate both arm targets on one shared time axis.

    The function deliberately keeps giver and receiver targets separate. Each
    pose mapping must contain position_xyz and orientation_xyzw; the
    caller may attach calibrated provenance and robot-specific metadata.
    """
    if steps < 1 or duration_s <= 0.0:
        raise ValueError("steps must be positive and duration_s must be > 0")
    for name, pose in (("giver_start", giver_start), ("giver_lift", giver_lift),
                       ("receiver_start", receiver_start), ("receiver_precontact", receiver_precontact)):
        if not isinstance(pose, Mapping) or "position_xyz" not in pose or "orientation_xyzw" not in pose:
            raise ValueError(f"{name} must contain position_xyz and orientation_xyzw")
    out: list[ArmWaypoint] = []
    for i in range(steps + 1):
        u = i / steps
        out.append(ArmWaypoint(
            time_from_start_s=duration_s * u,
            giver={
                **giver_lift,
                "position_xyz": _lerp(giver_start["position_xyz"], giver_lift["position_xyz"], u),
            },
            receiver={
                **receiver_precontact,
                "position_xyz": _lerp(receiver_start["position_xyz"], receiver_precontact["position_xyz"], u),
            },
        ))
    return tuple(out)
