from types import SimpleNamespace

import pytest

from rocobrick.ros_moveit_runtime import _merge_allowed_collision_pairs


class FakeEntry:
    def __init__(self, enabled=None):
        self.enabled = list(enabled or [])


class FakeMatrix:
    def __init__(self):
        self.entry_names = ["link_a", "link_b"]
        self.entry_values = [
            FakeEntry([True, True]),
            FakeEntry([True, True]),
        ]
        self.default_entry_names = ["default_pair"]
        self.default_entry_values = [False]


def test_merge_preserves_existing_acm_and_adds_only_requested_pairs():
    matrix = FakeMatrix()

    result = _merge_allowed_collision_pairs(
        matrix, (("brick-1", "gripper_link1"),), FakeEntry
    )

    assert result is matrix
    assert result.entry_names == ["link_a", "link_b", "brick-1", "gripper_link1"]
    assert [entry.enabled for entry in result.entry_values] == [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, False, True],
        [False, False, True, False],
    ]
    assert result.default_entry_names == ["default_pair"]
    assert result.default_entry_values == [False]


def test_merge_rejects_non_square_acm():
    matrix = SimpleNamespace(
        entry_names=["link_a", "link_b"],
        entry_values=[FakeEntry([False])],
    )

    with pytest.raises(ValueError, match="not square"):
        _merge_allowed_collision_pairs(matrix, (("brick-1", "gripper_link1"),), FakeEntry)
