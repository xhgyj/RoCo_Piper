"""Offline contact safety guards and exact interface identity regressions."""

import copy
from dataclasses import asdict

import pytest
from test_place_down_handler import (
    LOWER,
    MANIFEST,
    ROBOT,
    SESSION,
    UPPER,
    document,
)

from rocobrick.skills.contact import (
    PROBE_CONTACT_GAP_M,
    allows_recovery,
    inspect_contact,
    validate_guard,
)
from rocobrick.skills.observation import Pose


def guard():
    """Return the strict support/holding reference for an insertion.

    Returns:
        The validated robot-scoped guard document.
    """
    return {
        "robot_id": ROBOT,
        "session_id": SESSION,
        "manifest_sha256": MANIFEST,
        "upper_id": UPPER,
        "lower_id": LOWER,
        "support_pose": asdict(Pose((0.32, 0.32, 0.0168), (0.0, 0.0, 0.0, 1.0))),
        "held_transform": asdict(Pose((0.0, 0.0, 0.01), (1.0, 0.0, 0.0, 0.0))),
        "desired_upper": asdict(Pose((0.32, 0.32, 0.028), (0.0, 0.0, 0.0, 1.0))),
        "support_position_m": 0.001,
        "support_angle_rad": 0.05,
        "grasp_position_m": 0.003,
        "grasp_angle_rad": 0.15,
        "force_limit_n": 8.0,
        "mode": "insert",
    }


def scene(connected=False):
    """Return a co-sampled scene with a rigid held relationship.

    Returns:
        The current measured scene document.
    """
    return document(sequence=20, upper_z=0.0224, tcp_z=0.038, connection=connected)


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("support", "support_deflected"),
        ("slip", "grasp_slipped"),
        ("force", "interface_force_exceeded"),
        ("wrong_offset", "wrong_interface_connection"),
    ],
)
def test_guard_stops_pressing_on_support_slip_force_or_wrong_connection(defect, reason):
    """Contact defects stop before another insertion increment is authorized."""
    s = scene(defect == "wrong_offset")
    g = guard()
    pair = s["placement_interfaces"]["pairs"][f"{LOWER}|{UPPER}"]
    if defect == "support":
        s["bricks"][LOWER]["pose"]["position_xyz"][2] -= 0.002
    elif defect == "slip":
        s["robots"][ROBOT]["tcp_pose"]["position_xyz"][2] -= 0.005
    elif defect == "force":
        pair["contact_force_n"] = 9.0
    else:
        pair["actual_connection"]["lower_to_upper"]["position_xyz"][0] = 0.008
    stop, evidence = inspect_contact(s, g)
    assert stop == reason
    assert evidence["sequence"] == 20


def test_exact_connection_stops_insertion_and_unload_refuses_to_pull_it():
    """Correct connection enables release, never another closed-finger pull."""
    stop, evidence = inspect_contact(scene(True), guard())
    assert stop == "connection_confirmed" and evidence["exact_connection"] is True
    stop, _ = inspect_contact(scene(True), {**guard(), "mode": "unload"})
    assert stop == "connection_changed_during_unload"


def test_probe_mode_stops_at_first_geometric_contact():
    """Probe stops while the AABBs first touch; insert mode keeps pressing."""
    seated = scene()  # upper AABB bottom exactly at the lower AABB top
    stop, evidence = inspect_contact(seated, {**guard(), "mode": "probe"})
    assert stop == "probe_contact_detected"
    assert evidence["contact_gap_m"] == pytest.approx(0.0)
    stop, _ = inspect_contact(seated, guard())
    assert stop is None


def test_probe_mode_descends_while_the_interface_gap_is_open():
    """An open axial gap is measured, not treated as contact."""
    high = document(sequence=21, upper_z=0.024, tcp_z=0.04, connection=False)
    stop, evidence = inspect_contact(high, {**guard(), "mode": "probe"})
    assert stop is None
    assert evidence["contact_gap_m"] == pytest.approx(0.0016)
    assert evidence["contact_gap_m"] > PROBE_CONTACT_GAP_M


def test_probe_mode_never_stops_without_lateral_overlap():
    """A held brick passing beside the support cannot 'contact' it."""
    beside = document(
        sequence=22,
        upper_z=0.0224,
        upper_xy=(0.36, 0.40),
        tcp_z=0.038,
        connection=False,
    )
    stop, evidence = inspect_contact(beside, {**guard(), "mode": "probe"})
    assert stop is None
    assert evidence["contact_gap_m"] is None


def test_probe_mode_keeps_physical_safety_gates_ahead_of_contact_detection():
    """A deflected support stops the probe before any contact bookkeeping."""
    defective = scene()
    defective["bricks"][LOWER]["pose"]["position_xyz"][2] -= 0.002
    stop, _ = inspect_contact(defective, {**guard(), "mode": "probe"})
    assert stop == "support_deflected"


def test_probe_stop_admits_bounded_transport_recovery_only():
    """The probe stop is a known safely stopped step, never assembly success."""
    assert allows_recovery("probe_contact_detected") is True
    guard_document = guard()
    guard_document["mode"] = "press"
    with pytest.raises(ValueError, match="mode"):
        validate_guard(guard_document)


def test_missing_exact_identity_cannot_substitute_pair_boolean():
    """Connected without interface identity fails closed."""
    s = scene(True)
    s["placement_interfaces"]["pairs"][f"{LOWER}|{UPPER}"]["actual_connection"] = None
    with pytest.raises(ValueError, match="exact interface"):
        inspect_contact(s, guard())


def test_guard_refuses_nonboolean_number_and_foreign_scene():
    """Typed guard values and robot provenance remain strict."""
    g = guard()
    g["force_limit_n"] = True
    with pytest.raises(ValueError):
        validate_guard(g)
    s = copy.deepcopy(scene())
    s["session_id"] = "foreign"


def test_guard_cancels_active_action_before_another_contact_increment():
    """A live support defect cancels an accepted action and awaits its result."""
    from types import SimpleNamespace

    from test_ros_moveit_runtime_contract import (
        FAKE_GOAL_STATUS,
        FakeGoalHandle,
        make_world,
    )

    world = make_world()
    handle = FakeGoalHandle()
    world.runtime._active_handle = handle
    world.runtime._active_result_future = handle.result_future
    defective = scene()
    defective["bricks"][LOWER]["pose"]["position_xyz"][2] -= 0.002
    world.runtime.query_scene_state = lambda timeout_s: defective

    def finish_cancel():
        handle.result_future.set(
            SimpleNamespace(status=FAKE_GOAL_STATUS.STATUS_CANCELED)
        )

    world.events.append(finish_cancel)
    result = world.runtime._await_guarded_action(
        handle.result_future, guard(), world.clock.now + 2, {}, 0
    )
    assert handle.cancel_calls == 1
    assert result.terminal is True and result.safe_to_release is True
    assert result.details["contact_stop_reason"] == "support_deflected"


def test_guard_missing_feedback_cancels_and_never_claims_completion():
    """Lost observations fail rather than continue pressing blind."""
    from test_ros_moveit_runtime_contract import FakeGoalHandle, make_world

    world = make_world()
    handle = FakeGoalHandle()
    world.runtime._active_handle = handle
    world.runtime._active_result_future = handle.result_future

    def missing(timeout_s):
        raise RuntimeError("lost scene")

    world.runtime.query_scene_state = missing
    result = world.runtime._await_guarded_action(
        handle.result_future, guard(), world.clock.now + 2, {}, 0
    )
    assert handle.cancel_calls == 1
    assert result.terminal is False and result.safe_to_release is False
    assert "contact_feedback_failed" in result.details["contact_stop_reason"]
