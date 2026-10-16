"""Validate and retime collision-planned contact trajectories before execution."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ContactTrajectoryTiming:
    """Verified timing conversion; positions and the planned path stay unchanged."""

    scale: float
    duration_s: float
    times_s: tuple[float, ...]


# Contact motions may not exceed this measured Cartesian speed. Two live
# bounds meet here: the pilz LIN ceiling (1 m/s x velocity scaling) would let
# a full insertion descent cross the support-deflection bound between two
# guard samples, and the PhysX position drives leave a viscous tracking lag
# (~0.4 rad per rad/s on joint5) that must stay inside the 0.01 rad controller
# path tolerance while moving. Live evidence 2026-10-02: at 5 mm/s the
# descent aborted at 0.024 rad lag; 1 mm/s keeps the worst joint near 0.005.
CONTACT_CARTESIAN_SPEED_MPS = 0.001
UPWARD_INSERTION_SPEED_MPS = 0.005


def contact_speed_for_mode(mode: str) -> float:
    """Select insertion speed without accelerating support or recovery.

    Returns:
        Five mm/s for upward insertion, the original cap for other modes.
    """
    return (
        UPWARD_INSERTION_SPEED_MPS
        if mode == "insert_up"
        else CONTACT_CARTESIAN_SPEED_MPS
    )


# Floor so microscopic corrections keep a bounded minimum cycle time.
CONTACT_MINIMUM_DURATION_S = 0.4


def contact_minimum_duration_s(
    distance_m: float, *, speed_mps: float = CONTACT_CARTESIAN_SPEED_MPS
) -> float:
    """Bound one contact motion's duration by its Cartesian speed cap.

    Returns:
        The larger of the floor duration and ``distance_m`` travelled at
        :data:`CONTACT_CARTESIAN_SPEED_MPS`.
    """
    if (
        isinstance(distance_m, bool)
        or not isinstance(distance_m, (int, float))
        or not math.isfinite(distance_m)
        or distance_m < 0
    ):
        raise ValueError("contact Cartesian distance must be finite, nonnegative")
    if (
        type(speed_mps) not in (int, float)
        or not math.isfinite(speed_mps)
        or not 0 < speed_mps <= UPWARD_INSERTION_SPEED_MPS
    ):
        raise ValueError("contact speed must be finite and in (0, 0.005] m/s")
    return max(
        CONTACT_MINIMUM_DURATION_S,
        float(distance_m) / speed_mps,
    )


def contact_trajectory_timing(
    times_s: Sequence[float], *, minimum_duration_s: float = 0.4
) -> ContactTrajectoryTiming:
    """Slow very short trajectories without changing their geometric path.

    Returns:
        Strict times and a non-shortening time scale for derivatives.
    """
    values = tuple(times_s)
    if len(values) < 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("contact trajectory requires finite nonnegative times")
    if any(b <= a for a, b in zip(values, values[1:])) or values[-1] <= 0:
        raise ValueError("contact trajectory times must strictly increase")
    if not math.isfinite(minimum_duration_s) or minimum_duration_s <= 0:
        raise ValueError("minimum contact duration must be positive")
    scale = max(1.0, minimum_duration_s / values[-1])
    return ContactTrajectoryTiming(
        scale, values[-1] * scale, tuple(float(value) * scale for value in values)
    )
