"""Single logical AgileX gripper command mapped to BrickSim finger joints."""

from __future__ import annotations


def logical_gripper_to_fingers(opening_m: float) -> tuple[float, float]:
    if not 0.0 <= opening_m <= 0.1:
        raise ValueError("logical gripper opening must be within [0.0, 0.1] m")
    return opening_m * 0.5, opening_m * -0.5


def fingers_to_logical_gripper(finger_1_m: float, finger_2_m: float, tolerance_m: float = 1e-4) -> float:
    opening_1 = finger_1_m * 2.0
    opening_2 = finger_2_m * -2.0
    if abs(opening_1 - opening_2) > tolerance_m:
        raise ValueError("finger measurements do not satisfy the official mimic relation")
    opening = (opening_1 + opening_2) / 2.0
    if not 0.0 <= opening <= 0.1:
        raise ValueError("measured gripper opening is outside the official range")
    return opening
