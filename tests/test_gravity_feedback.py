"""Finite bounded gravity feed-forward never changes finger efforts."""

import pytest

from rocobrick.gravity_feedback import arm_gravity_efforts


def test_arm_gravity_preserves_joint_order_and_zeroes_fingers():
    """Gravity model values map exactly to the identified arm DOFs."""
    assert arm_gravity_efforts(
        ("piper_1_joint2", "piper_1_gripper_joint1", "piper_1_joint5"),
        (-5.3, 1.4, -0.1),
    ) == (-5.3, 0.0, -0.1)


@pytest.mark.parametrize("values", [(11.0,), (float("nan"),), (True,), ()])
def test_invalid_or_unbounded_gravity_feedback_is_refused(values):
    """Missing or invalid model forces cannot silently command effort."""
    with pytest.raises(ValueError):
        arm_gravity_efforts(("piper_0_joint2",), values)
