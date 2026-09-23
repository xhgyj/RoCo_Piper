"""Pure readiness predicates for the live MoveIt execution chain.

ROS-message-free checks over ordinary mappings/sequences, mirroring the P1
readiness phase: strictly increasing /clock samples, complete finite 8-DOF
states on both the Isaac and broadcaster topics that carry the SAME values,
and the required controllers active. Aggregation only; no ROS imports.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from rocobrick.moveit_contract import (
    EXACT_STATE_MATCH_TOLERANCE,
    JOINT_NAMES,
    REQUIRED_CONTROLLERS,
    AdmissionDecision,
)

# Optional reserved key carrying the velocity sequence next to the joint
# positions in a state mapping; when present it must have length 0 or 8.
VELOCITY_KEY = "velocity"


def _is_finite(value: object) -> bool:
    """Return True when the value is a finite int/float."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def state_is_complete(state: Mapping[str, object]) -> bool:
    """True for the exact 8 canonical joints with finite positions.

    The optional ``velocity`` entry, when present, must be a sized sequence of
    length 0 or 8. Any unknown key, missing joint or non-finite position makes
    the state incomplete.
    """
    keys = set(state)
    if keys - {VELOCITY_KEY} != set(JOINT_NAMES):
        return False
    if not all(_is_finite(state[name]) for name in JOINT_NAMES):
        return False
    velocity = state.get(VELOCITY_KEY)
    if velocity is not None:
        try:
            length = len(velocity)
        except TypeError:
            return False
        if length not in (0, len(JOINT_NAMES)):
            return False
    return True


def states_match(
    a: Mapping[str, float], b: Mapping[str, float], *, tolerance: float = 1.0e-4
) -> bool:
    """True when both states carry all 8 finite joints within tolerance."""
    if not _is_finite(tolerance) or tolerance < 0.0:
        return False
    for name in JOINT_NAMES:
        if name not in a or name not in b:
            return False
        if not (_is_finite(a[name]) and _is_finite(b[name])):
            return False
    return (
        max(abs(float(a[name]) - float(b[name])) for name in JOINT_NAMES) <= tolerance
    )


def controllers_are_active(states: Mapping[str, str]) -> bool:
    """True when the required three controllers report exactly "active".

    Extra controllers are allowed.
    """
    return all(states.get(name) == "active" for name in REQUIRED_CONTROLLERS)


def clock_is_increasing(values: Sequence[int]) -> bool:
    """True for at least two clock samples that strictly increase pairwise."""
    if len(values) < 2:
        return False
    try:
        return all(later > earlier for earlier, later in zip(values, values[1:]))
    except TypeError:
        return False


def readiness_decision(
    *,
    clock_values: Sequence[int],
    isaac_state: Mapping[str, object] | None,
    broadcaster_state: Mapping[str, float] | None,
    controllers: Mapping[str, str],
) -> AdmissionDecision:
    """Aggregate the readiness predicates into one admission decision.

    Aggregation only: no ROS, no I/O, no state.
    """
    checks: dict[str, bool] = {
        "clock_increasing": clock_is_increasing(clock_values),
        "isaac_state_complete": (
            isaac_state is not None and state_is_complete(isaac_state)
        ),
        "broadcaster_state_complete": broadcaster_state is not None
        and state_is_complete(broadcaster_state),
        "states_match": (
            isaac_state is not None
            and broadcaster_state is not None
            and states_match(
                isaac_state, broadcaster_state, tolerance=EXACT_STATE_MATCH_TOLERANCE
            )
        ),
        "controllers_active": controllers_are_active(controllers),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        return AdmissionDecision(
            accepted=False,
            reason="readiness failed: " + ", ".join(failed),
            details=checks,
        )
    return AdmissionDecision(accepted=True, reason="ok", details=checks)
