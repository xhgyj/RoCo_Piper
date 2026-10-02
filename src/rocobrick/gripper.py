"""Single logical AgileX gripper command mapped to BrickSim finger joints."""

from __future__ import annotations

import math

# The original-mounted distal contact faces coincide at logical zero.
# The measured separation/opening fit has unit slope and a sub-nanometre
# intercept; unlike the former proximal metal patch there is no 8 mm gap.
PIPER_L_PAD_SEPARATION_INTERCEPT_M = 0.0


def pad_separation_to_logical_opening(separation_m: float) -> float:
    """Return the logical opening whose physical pad separation is requested.

    Raises:
        ValueError: When the separation is not finite or no logical opening
            in the official range [0.0, 0.1] m produces it.
    """
    separation = float(separation_m)
    if not math.isfinite(separation):
        raise ValueError("pad separation must be finite")
    opening = separation - PIPER_L_PAD_SEPARATION_INTERCEPT_M
    if not 0.0 <= opening <= 0.1:
        raise ValueError(
            "pad separation is outside the graspable range "
            f"[{PIPER_L_PAD_SEPARATION_INTERCEPT_M:.3f}, 0.100] m: {separation_m!r}"
        )
    return opening


def logical_opening_to_pad_separation(opening_m: float) -> float:
    """Return the physical pad separation produced by one logical opening.

    Raises:
        ValueError: When the opening is not finite or outside [0.0, 0.1] m.
    """
    opening = float(opening_m)
    if not math.isfinite(opening) or not 0.0 <= opening <= 0.1:
        raise ValueError("logical gripper opening must be within [0.0, 0.1] m")
    return opening + PIPER_L_PAD_SEPARATION_INTERCEPT_M


def logical_gripper_to_fingers(opening_m: float) -> tuple[float, float]:
    """Return the symmetric finger positions of one logical opening."""
    if not 0.0 <= opening_m <= 0.1:
        raise ValueError("logical gripper opening must be within [0.0, 0.1] m")
    return opening_m * 0.5, opening_m * -0.5


def fingers_to_contact_opening(
    finger_1_m: float, finger_2_m: float
) -> float:
    """Return the physical jaw separation of two measured finger positions.

    Contact breaks nominal symmetry, so this measures the opening directly
    from the opposing positions instead of assuming the mimic relation.
    """
    values = (float(finger_1_m), float(finger_2_m))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("measured gripper fingers must be finite")
    # A zero value is retained as an invalid/missing feedback case.
    if values[0] <= 0.0 or values[1] >= 0.0:
        raise ValueError("measured gripper fingers do not identify opposing contact")
    opening = values[0] - values[1]
    if not 0.0 <= opening <= 0.1:
        raise ValueError("measured gripper opening is outside the official range")
    return opening


def fingers_to_logical_gripper(
    finger_1_m: float, finger_2_m: float, tolerance_m: float = 1e-4
) -> float:
    """Return the logical opening implied by two symmetric finger positions."""
    opening_1 = finger_1_m * 2.0
    opening_2 = finger_2_m * -2.0
    if abs(opening_1 - opening_2) > tolerance_m:
        raise ValueError(
            "finger measurements do not satisfy the official mimic relation"
        )
    opening = (opening_1 + opening_2) / 2.0
    if not 0.0 <= opening <= 0.1:
        raise ValueError("measured gripper opening is outside the official range")
    return opening
