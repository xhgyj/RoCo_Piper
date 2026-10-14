"""Shape-independent stud-hole target geometry regressions."""

import math

import pytest

from rocobrick.skills.observation import Pose
from rocobrick.skills.placement_geometry import (
    compose,
    interface_place_targets,
    interface_place_up_targets,
    inverse,
    pose_error,
)
from rocobrick.skills.targets import RobotBaseFrame, SkillParams, TcpProvenance
from rocobrick.tcp_contract import EXPECTED_RUNTIME_MODEL_SHA256


@pytest.mark.parametrize("height", [0.0032, 0.0096, 0.0192])
@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, math.pi])
@pytest.mark.parametrize("offset", [(0.0, 0.0), (0.008, -0.024), (-0.016, 0.04)])
def test_interface_target_uses_measured_grasp_not_aabb_or_symmetry(height, yaw, offset):
    """Offsets, heights and rotations compose without special brick assumptions."""
    lower = Pose((0.13, -0.2, 0.017), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
    interface = Pose(
        (*offset, height), (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    )
    upper = Pose((0.31, 0.02, 0.12), (0.0, 0.0, 0.0, 1.0))
    held = Pose((0.003, -0.001, 0.008), (1.0, 0.0, 0.0, 0.0))
    tcp = compose(upper, held)
    base = RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    provenance = TcpProvenance(EXPECTED_RUNTIME_MODEL_SHA256, "c" * 64)
    params = SkillParams()
    targets, desired, measured_held = interface_place_targets(
        lower,
        interface,
        upper,
        tcp,
        0.02,
        base,
        provenance,
        params,
    )
    assert pose_error(measured_held, held)[0] < 1e-10
    assert pose_error(desired, compose(lower, interface))[0] < 1e-10
    contact = Pose(targets.contact.position, targets.contact.orientation_xyzw)
    actual_upper = compose(contact, inverse(held))
    assert actual_upper.position_xyz[:2] == pytest.approx(desired.position_xyz[:2])
    assert actual_upper.position_xyz[2] == pytest.approx(
        desired.position_xyz[2] - params.place_press_m
    )
    assert pose_error(actual_upper, desired)[1] < 1e-7
    assert targets.release_expected_window_m[0] > 0.02


def test_tilted_base_offsets_the_entire_target_not_just_z():
    """A rotated base carries every component of the world standoff."""
    angle = 0.3
    base = RobotBaseFrame(
        (0.01, 0.02, 0.03), (math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2))
    )
    lower = Pose((0.1, 0.2, 0.03), (0.0, 0.0, 0.0, 1.0))
    upper = Pose((0.3, 0.2, 0.1), (0.0, 0.0, 0.0, 1.0))
    tcp = Pose((0.3, 0.2, 0.11), (1.0, 0.0, 0.0, 0.0))
    targets, _, _ = interface_place_targets(
        lower,
        Pose((0.024, 0.008, 0.0096), (0.0, 0.0, 0.0, 1.0)),
        upper,
        tcp,
        0.02,
        base,
        TcpProvenance(EXPECTED_RUNTIME_MODEL_SHA256, "c" * 64),
        SkillParams(),
    )
    difference = tuple(
        targets.preplace.position[i] - targets.contact.position[i] for i in range(3)
    )
    assert difference[1] == pytest.approx(
        (0.08 + SkillParams().place_press_m) * math.sin(angle)
    )


@pytest.mark.parametrize("symmetries,expected_angle", [((0,), math.pi), ((0, 2), 0.0)])
def test_only_explicit_asset_symmetry_can_change_nominal_yaw(
    symmetries, expected_angle
):
    """Unknown shapes retain nominal orientation; verified rectangles may flip."""
    lower = Pose((0.1, 0.2, 0.01), (0.0, 0.0, 1.0, 0.0))
    upper = Pose((0.3, 0.2, 0.12), (0.0, 0.0, 0.0, 1.0))
    tcp = Pose((0.3, 0.2, 0.13), (1.0, 0.0, 0.0, 0.0))
    _, desired, _ = interface_place_targets(
        lower,
        Pose((0.008, -0.024, 0.0096), (0.0, 0.0, 0.0, 1.0)),
        upper,
        tcp,
        0.02,
        RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TcpProvenance(EXPECTED_RUNTIME_MODEL_SHA256, "c" * 64),
        SkillParams(),
        yaw_symmetries=symmetries,
    )
    assert pose_error(desired, Pose(desired.position_xyz, upper.orientation_xyzw))[
        1
    ] == pytest.approx(expected_angle)
    assert desired.position_xyz == pytest.approx((0.092, 0.224, 0.0196))


def test_place_up_yaw_symmetry_keeps_tool_off_the_mated_upper_half():
    """Task C gantry: only the free-side yaw candidate may be seated."""
    upper = Pose((-0.048, -0.034, 0.1184), (0.0, 0.0, 1.0, 0.0))
    lower_to_upper = Pose(
        (0.0, 0.032, 0.0192),
        (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
    )
    measured_lower = Pose((0.352, 0.062, 0.110), (0.0, 0.0, 0.0, 1.0))
    held = Pose((0.0, -0.0072, 0.0148), (1.0, 0.0, 0.0, 0.0))
    measured_tcp = compose(measured_lower, held)
    targets, desired_lower, measured_held = interface_place_up_targets(
        upper,
        lower_to_upper,
        measured_lower,
        measured_tcp,
        0.0318,
        RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TcpProvenance(EXPECTED_RUNTIME_MODEL_SHA256, "c" * 64),
        SkillParams(),
        (0, 2),
    )
    assert pose_error(measured_held, held)[0] < 1e-10
    assert desired_lower.position_xyz == pytest.approx(
        (-0.016, -0.034, 0.0992), abs=1e-6
    )
    desired_tcp = compose(desired_lower, held)
    # The beam footprint ends at world x = -0.016; the tool must stay clear
    # of it, which only the free-side candidate realizes.
    assert desired_tcp.position_xyz[0] == pytest.approx(-0.0088, abs=1e-3)
    assert desired_tcp.position_xyz[0] > -0.016
    assert targets.preplace.position[2] == pytest.approx(0.034, abs=1e-3)
    assert targets.contact_world_z_m == pytest.approx(0.114, abs=1e-3)


def test_place_up_fails_closed_when_every_symmetry_faces_the_upper():
    """A grasp offset on the mated side with no clearing rotation is fatal."""
    upper = Pose((-0.048, -0.034, 0.1184), (0.0, 0.0, 1.0, 0.0))
    lower_to_upper = Pose(
        (0.0, 0.032, 0.0192),
        (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
    )
    measured_lower = Pose((0.352, 0.062, 0.110), (0.0, 0.0, 0.0, 1.0))
    toward_upper = Pose((0.0, 0.0072, 0.0148), (1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="no verified yaw symmetry"):
        interface_place_up_targets(
            upper,
            lower_to_upper,
            measured_lower,
            compose(measured_lower, toward_upper),
            0.0318,
            RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            TcpProvenance(EXPECTED_RUNTIME_MODEL_SHA256, "c" * 64),
            SkillParams(),
            (0,),
        )


def test_place_up_vertical_overlap_keeps_orientation_tiebreak():
    """A fully stacked connection has no free side, so ordering is unchanged."""
    upper = Pose((-0.048, -0.034, 0.1184), (0.0, 0.0, 1.0, 0.0))
    lower_to_upper = Pose(
        (0.0, 0.0, 0.0192),
        (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
    )
    measured_lower = Pose((0.352, 0.062, 0.110), (0.0, 0.0, 0.0, 1.0))
    held = Pose((0.0, -0.0072, 0.0148), (1.0, 0.0, 0.0, 0.0))
    targets, desired_lower, _ = interface_place_up_targets(
        upper,
        lower_to_upper,
        measured_lower,
        compose(measured_lower, held),
        0.0318,
        RobotBaseFrame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TcpProvenance(EXPECTED_RUNTIME_MODEL_SHA256, "c" * 64),
        SkillParams(),
        (0, 2),
    )
    assert desired_lower.position_xyz[:2] == pytest.approx((-0.048, -0.034))
    assert targets.contact_world_z_m == pytest.approx(0.1184 - 0.0192 + 0.0148)


@pytest.mark.parametrize(
    "quaternion",
    [
        (
            -0.011421002447605133,
            0.001834752270951867,
            0.004789809696376324,
            0.9999216198921204,
        ),
        (
            -0.011421273462474346,
            0.0018344814889132977,
            0.004789730999618769,
            0.9999216198921204,
        ),
        (
            -0.011421500705182552,
            0.0018346573924645782,
            0.0047898320481181145,
            0.9999216198921204,
        ),
    ],
)
@pytest.mark.parametrize("sign", [1, -1])
def test_pose_error_recorded_lift_quaternions_have_no_false_rotation(quaternion, sign):
    """Recorded sequences 73/74/75 retain their raw floats and equivalent signs."""
    measured = Pose((0.0, 0.0, 0.0), quaternion)
    expected = Pose((0.0, 0.0, 0.04), tuple(sign * value for value in quaternion))
    original = measured.orientation_xyzw
    distance, angle = pose_error(measured, expected)
    assert distance == pytest.approx(0.04)
    assert angle == 0.0
    assert measured.orientation_xyzw == original


@pytest.mark.parametrize(
    "angle", [0.0, 1e-10, 1e-8, 1e-7, 1e-5, 0.2, math.pi, 2 * math.pi - 0.2]
)
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize(
    "scales", [(1.0, 1.0), (1 - 1e-7, 1 + 1e-7), (1 + 1e-7, 1 - 1e-7)]
)
def test_pose_error_preserves_real_small_rotation_and_shortest_arc(angle, sign, scales):
    """Normalizing valid sensor roundoff must never erase a genuine rotation."""
    left = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, scales[0]))
    right = Pose(
        (0.0, 0.0, 0.0),
        (
            0.0,
            0.0,
            sign * scales[1] * math.sin(angle / 2),
            sign * scales[1] * math.cos(angle / 2),
        ),
    )
    shortest = min(angle, 2 * math.pi - angle)
    assert pose_error(left, right)[1] == pytest.approx(shortest, rel=1e-10, abs=1e-15)
    assert pose_error(right, left)[1] == pytest.approx(shortest, rel=1e-10, abs=1e-15)


@pytest.mark.parametrize(
    "quaternion",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, float("nan"), 1.0),
        (0.0, 0.0, float("inf"), 1.0),
        (0.0, 0.0, 0.0, 2.0),
        (False, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ],
)
@pytest.mark.parametrize("invalid_side", ["measured", "expected"])
def test_pose_error_rejects_invalid_quaternions_before_normalization(
    quaternion, invalid_side
):
    """Bad poses fail closed instead of being turned into valid orientations."""
    valid = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    invalid = Pose((0.0, 0.0, 0.0), quaternion)
    with pytest.raises(ValueError):
        pose_error(
            *((invalid, valid) if invalid_side == "measured" else (valid, invalid))
        )
