"""Offline tests for the calibrated pad-separation <-> opening conversion."""

from __future__ import annotations

import math

import pytest

from rocobrick.gripper import (
    PIPER_L_PAD_SEPARATION_INTERCEPT_M,
    logical_gripper_to_fingers,
    logical_opening_to_pad_separation,
    pad_separation_to_logical_opening,
)


@pytest.mark.parametrize(
    ("separation_m", "opening_m"),
    [
        (PIPER_L_PAD_SEPARATION_INTERCEPT_M, 0.0),
        (0.016 + PIPER_L_PAD_SEPARATION_INTERCEPT_M, 0.016),
        (0.05, 0.05 - PIPER_L_PAD_SEPARATION_INTERCEPT_M),
        (0.1 + PIPER_L_PAD_SEPARATION_INTERCEPT_M, 0.1),
    ],
)
def test_pad_separation_converts_to_the_calibrated_opening(
    separation_m: float, opening_m: float
) -> None:
    """Pad separation and logical opening convert exactly, both directions."""
    assert pad_separation_to_logical_opening(separation_m) == pytest.approx(
        opening_m
    )
    assert logical_opening_to_pad_separation(opening_m) == pytest.approx(
        separation_m
    )
    # The round trip through the finger positions used by the runtime command
    # keeps the physical separation of the commanded grasp.
    finger_1, finger_2 = logical_gripper_to_fingers(opening_m)
    assert finger_1 - finger_2 == pytest.approx(opening_m)


@pytest.mark.parametrize("separation_m", [math.nan, math.inf, -1e-9, 0.1005, 0.1085])
def test_pad_separation_outside_the_graspable_range_fails(
    separation_m: float,
) -> None:
    """Separations no logical opening can produce fail closed."""
    with pytest.raises(ValueError):
        pad_separation_to_logical_opening(separation_m)


@pytest.mark.parametrize("opening_m", [math.nan, math.inf, -1e-9, 0.1001])
def test_opening_outside_the_official_range_fails(opening_m: float) -> None:
    """Openings outside the official range are rejected before conversion."""
    with pytest.raises(ValueError):
        logical_opening_to_pad_separation(opening_m)


def test_intercept_matches_the_calibration_document() -> None:
    """The intercept equals the frozen calibration's exact relation."""
    # artifacts/model-audit/piper_l_tcp_calibration.json fixes
    # expected_patch_separation_m == "opening + 0.0" (fit r2 == 1.0).
    assert PIPER_L_PAD_SEPARATION_INTERCEPT_M == 0.0
