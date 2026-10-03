"""Shape-independent stud-hole target geometry regressions."""

import math

import pytest

from rocobrick.skills.observation import Pose
from rocobrick.skills.placement_geometry import (
    compose,
    interface_place_targets,
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
