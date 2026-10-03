"""Contact trajectory timing preserves collision-checked geometry."""

import pytest

from rocobrick.adapters.moveit.contact_trajectory import (
    CONTACT_CARTESIAN_SPEED_MPS,
    contact_minimum_duration_s,
    contact_trajectory_timing,
)


def test_short_trajectory_slows_without_shortening_or_skipping_points():
    """Retiming preserves all points and scales derivatives consistently."""
    result = contact_trajectory_timing((0.0, 0.01, 0.02))
    assert result.scale == 20.0
    assert result.times_s == (0.0, 0.2, 0.4)
    assert result.duration_s == 0.4


def test_long_trajectory_is_never_accelerated():
    """An already slow path retains its original timing."""
    assert contact_trajectory_timing((0.0, 0.5, 1.0)).scale == 1.0


@pytest.mark.parametrize(
    "times", [(0.0,), (0.0, 0.0), (0.2, 0.1), (0.0, float("nan")), (0.0, True)]
)
def test_invalid_time_sequence_fails_before_execution(times):
    """Duplicate, reordered or nonfinite points are refused."""
    with pytest.raises(ValueError):
        contact_trajectory_timing(times)


def test_full_insertion_distance_is_capped_by_the_contact_speed():
    """A 6 mm descent takes at least 6 s at the 1 mm/s contact cap."""
    assert contact_minimum_duration_s(0.0) == pytest.approx(0.4)
    assert contact_minimum_duration_s(0.006) == pytest.approx(
        0.006 / CONTACT_CARTESIAN_SPEED_MPS
    )
    assert contact_minimum_duration_s(0.006) > contact_minimum_duration_s(0.001)


@pytest.mark.parametrize("distance", [-0.001, float("inf"), True, "0.01"])
def test_nonfinite_or_negative_contact_distance_is_refused(distance):
    """The speed cap only applies to measured finite distances."""
    with pytest.raises(ValueError):
        contact_minimum_duration_s(distance)
