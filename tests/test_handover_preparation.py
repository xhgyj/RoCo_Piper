"""Paired preparation is not final contact; local paths remain certified."""

import copy
import math
from dataclasses import replace

import pytest
from test_handover_joint_planning import planning_request
from test_handover_scene_planner import native_provider

from rocobrick.skills.handover_region import (
    LOCAL_ALIGNMENT_ANGLE_RAD,
    LOCAL_ALIGNMENT_DISTANCE_M,
    candidate_from_endpoint,
    contact_from_preparation,
    preparation_from_endpoint,
    preparation_region,
)
from rocobrick.skills.observation import Pose
from rocobrick.skills.placement_geometry import (
    compose,
    interpolate_pose,
    inverse,
    pose_error,
)
from rocobrick.skills.targets import _quat_rotate_vector


def preparation(request, *, yaw=0.4, tilt=0.3, giver_offset=0.0):
    """Build a broad arrival with both-arm adjustment and a non-final rotation.

    Returns:
        Bound paired preparation geometry, never an execution certificate.
    """
    observed = request.observation(request.donor.robot_id)
    giver = compose(
        Pose(
            (giver_offset, 0, 0), (0, 0, math.sin(giver_offset), math.cos(giver_offset))
        ),
        observed.tcp_pose,
    )
    obj = compose(giver, compose(inverse(observed.tcp_pose), observed.object_pose))
    orientation = compose(
        Pose((0, 0, 0), (0, 0, math.sin(yaw / 2), math.cos(yaw / 2))),
        Pose((0, 0, 0), (math.sin(tilt / 2), 0, 0, math.cos(tilt / 2))),
    ).orientation_xyzw
    region = preparation_region(request)
    receiver = compose(
        obj,
        Pose(
            (
                region["center"][0],
                region["center"][1],
                region["center"][2]
                - region["precontact_distance_m"]
                - region["contact_clearance_m"],
            ),
            orientation,
        ),
    )
    return preparation_from_endpoint(request, giver, receiver)


@pytest.mark.parametrize("yaw", [-2.1, -0.4, 0.3, 0.9, 2.4])
@pytest.mark.parametrize("tilt", [-0.4, 0.0, 0.35])
def test_preparation_accepts_nonfinal_pose_and_continuous_local_side_pinch(yaw, tilt):
    """Broad arrival may be tilted; final local geometry exposes stud face."""
    request = planning_request()
    original = copy.deepcopy(request.scene_snapshot)
    binding = request.binding_sha256
    arrival = preparation(request, yaw=yaw, tilt=tilt)
    if tilt:
        with pytest.raises(ValueError, match="STUD_FACE"):
            candidate_from_endpoint(request, arrival.giver, arrival.receiver)
    candidate = contact_from_preparation(request, arrival)
    distance, angle = pose_error(arrival.receiver, candidate.receiver_precontact)
    assert distance <= LOCAL_ALIGNMENT_DISTANCE_M
    assert angle <= LOCAL_ALIGNMENT_ANGLE_RAD
    local = compose(inverse(candidate.object_goal), candidate.receiver_contact)
    closing = _quat_rotate_vector(local.orientation_xyzw, (0, 1, 0))
    assert abs(closing[2]) < 1e-10
    giver_local = compose(inverse(candidate.object_goal), candidate.donor_goal)
    giver_closing = _quat_rotate_vector(giver_local.orientation_xyzw, (0, 1, 0))
    assert abs(sum(a * b for a, b in zip(closing, giver_closing))) < 1e-10
    assert request.scene_snapshot == original
    assert request.binding_sha256 == binding


def test_preparation_preserves_adjusted_giver_grasp_and_receiver_arrival():
    """Giver is free during preparation, not locked to measured lift pose."""
    request = planning_request()
    arrival = preparation(request, giver_offset=0.03)
    candidate = contact_from_preparation(request, arrival)
    assert arrival.giver != request.observation(request.donor.robot_id).tcp_pose
    assert candidate.donor_goal == arrival.giver
    assert (
        max(
            pose_error(
                compose(candidate.object_goal, candidate.object_to_donor_tcp),
                arrival.giver,
            )
        )
        < 1e-7
    )


def test_preparation_bounds_and_local_handler_budget_are_separate():
    """Broad admissible arrival never authorizes an unbounded correction."""
    request = planning_request()
    arrival = preparation(request)
    far = compose(arrival.object_pose, Pose((0.14, 0, 0), (0, 0, 0, 1)))
    arrival = preparation_from_endpoint(request, arrival.giver, far)
    candidate = contact_from_preparation(request, arrival)
    assert pose_error(arrival.receiver, candidate.receiver_precontact)[0] <= LOCAL_ALIGNMENT_DISTANCE_M
    with pytest.raises(ValueError, match="PREPARATION_RECEIVER"):
        preparation_from_endpoint(
            request, arrival.giver, replace(far, position_xyz=(10, 0, 0))
        )
    with pytest.raises(ValueError, match="PREPARATION_TRANSLATION"):
        preparation_from_endpoint(
            request, replace(arrival.giver, position_xyz=(10, 0, 0)), far
        )
    with pytest.raises(ValueError, match="foreign"):
        contact_from_preparation(request, replace(arrival, binding_sha256="a" * 64))


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.3, math.pi])
def test_pose_interpolation_keeps_shortest_rotation_and_exact_position(angle):
    """The Cartesian tube midpoint must include rotation, not freeze it."""
    first = Pose((0, 0, 0), (0, 0, 0, 1))
    last = Pose((0.08, 0, 0), (0, 0, -math.sin(angle / 2), -math.cos(angle / 2)))
    middle = interpolate_pose(first, last, 0.5)
    assert middle.position_xyz == pytest.approx((0.04, 0, 0))
    assert pose_error(first, middle)[1] == pytest.approx(angle / 2, abs=1e-12)
    assert pose_error(middle, last)[1] == pytest.approx(angle / 2, abs=1e-12)


def test_native_local_alignment_holds_giver_and_certifies_rotating_tube(tmp_path):
    """Use the real native kernel; no ROS, simulator or execution facade."""
    provider, request, candidate, _ = native_provider(tmp_path)
    current = dict(request.start_state)
    first = provider.geometry.tcp_pose(request.receiver.robot_id, current)
    target = compose(
        first, Pose((0, 0, 0), (0, 0, math.sin(0.04 / 2), math.cos(0.04 / 2)))
    )
    states = provider._cartesian(
        request,
        request.receiver.robot_id,
        current,
        first,
        target,
        deadline_s=10,
        receiver_touch=False,
        holder=request.donor.robot_id,
    )
    assert states is not None, provider._cartesian_failure
    assert len(states) >= 3
    for state in states:
        assert all(
            state[f"{request.donor.robot_id}_joint{i}"]
            == current[f"{request.donor.robot_id}_joint{i}"]
            for i in range(1, 7)
        )
    assert (
        pose_error(
            provider.geometry.tcp_pose(request.receiver.robot_id, states[-1]), target
        )[1]
        < request.profile.orientation_tolerance_rad
    )


def test_native_local_alignment_rejects_out_of_budget_before_ik(tmp_path):
    """No local solver or command may be reached on an excessive correction."""
    provider, request, candidate, _ = native_provider(tmp_path)
    current = dict(request.start_state)
    current[request.receiver.robot_id + "_joint1"] = 0.9
    assert (
        provider._receiver_alignment(
            request,
            candidate,
            current,
            deadline_s=10,
        )
        is None
    )
    assert provider._cartesian_failure == "LOCAL_ALIGNMENT_DISTANCE_EXCEEDED"


def test_region_builds_final_precontact_candidate_from_giver_target(tmp_path):
    """The final receiver target is derived from the selected giver target."""
    provider, request, _, _ = native_provider(tmp_path)
    region = preparation_region(request)
    giver = Pose(
        tuple(region["giver_pose"]["position_xyz"]),
        tuple(region["giver_pose"]["orientation_xyzw"]),
    )
    candidate = provider._final_candidate_from_giver(request, giver)
    assert candidate is not None
    position, rotation = pose_error(candidate.donor_goal, giver)
    assert position < 1e-9
    assert rotation < 1e-9
    local = compose(inverse(candidate.object_goal), candidate.receiver_precontact)
    closing = _quat_rotate_vector(local.orientation_xyzw, (0, 1, 0))
    giver_local = compose(inverse(candidate.object_goal), candidate.donor_goal)
    giver_closing = _quat_rotate_vector(giver_local.orientation_xyzw, (0, 1, 0))
    assert abs(sum(a * b for a, b in zip(closing, giver_closing))) < 1e-10
