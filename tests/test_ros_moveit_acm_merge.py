from types import SimpleNamespace

import pytest

from rocobrick.ros_moveit_runtime import _merge_allowed_collision_pairs, _collision_pair_allowed


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


def test_explicit_forbidden_pair_revokes_stale_contact_only():
    matrix = FakeMatrix()
    pair = ("brick-1", "gripper_link1")
    _merge_allowed_collision_pairs(matrix, (pair,), FakeEntry)
    result = _merge_allowed_collision_pairs(
        matrix, (), FakeEntry, forbidden_pairs=(pair,),
    )
    a, b = (result.entry_names.index(name) for name in pair)
    assert result.entry_values[a].enabled[b] is False
    assert result.entry_values[b].enabled[a] is False
    assert result.entry_values[0].enabled[1] is True
    assert result.default_entry_names == ["default_pair"]
    _merge_allowed_collision_pairs(matrix, (pair,), FakeEntry)
    assert matrix.entry_values[a].enabled[b] is True


def test_forbidden_permission_roundtrip_and_legacy_payload():
    from dataclasses import asdict, replace
    from rocobrick.isaac_backend import planning_scene_update_from_payload
    from rocobrick.planning_scene import PlanningSceneUpdate, scene_digest

    update = PlanningSceneUpdate("protected", (), "0" * 64,
        forbidden_collision_pairs=(("brick-1", "gripper_link1"),))
    update = replace(update, expected_scene_digest=scene_digest(update))
    assert planning_scene_update_from_payload(asdict(update)) == update
    allowed = replace(update, forbidden_collision_pairs=(),
                      allowed_collision_pairs=update.forbidden_collision_pairs)
    assert scene_digest(allowed) != scene_digest(update)
    legacy = PlanningSceneUpdate("legacy", (), "0" * 64)
    legacy = replace(legacy, expected_scene_digest=scene_digest(legacy))
    payload = asdict(legacy)
    payload.pop("forbidden_collision_pairs")
    assert planning_scene_update_from_payload(payload) == legacy
    with pytest.raises(ValueError, match="both allowed and forbidden"):
        replace(update, allowed_collision_pairs=update.forbidden_collision_pairs)


def test_readback_omitted_false_pair_remains_collision_protected():
    matrix = FakeMatrix()
    assert _collision_pair_allowed(matrix, "brick-new", "link_a") is False
    matrix.default_entry_names = ["brick-new"]
    matrix.default_entry_values = [True]
    assert _collision_pair_allowed(matrix, "brick-new", "link_a") is True
    matrix.default_entry_names.append("link_a")
    matrix.default_entry_values.append(False)
    assert _collision_pair_allowed(matrix, "brick-new", "link_a") is False


def test_readback_ros_boolean_values_are_not_compared_by_identity():
    matrix = FakeMatrix()
    matrix.entry_values = [FakeEntry([0, 0]), FakeEntry([0, 0])]
    assert _collision_pair_allowed(matrix, "link_a", "link_b") is False
    matrix.entry_values = [FakeEntry([0, 1]), FakeEntry([1, 0])]
    assert _collision_pair_allowed(matrix, "link_a", "link_b") is True
    matrix.entry_values[1].enabled[0] = 0
    with pytest.raises(ValueError, match="symmetric"):
        _collision_pair_allowed(matrix, "link_a", "link_b")
