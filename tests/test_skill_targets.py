"""Offline tests for the pure skill-target geometry (no ROS, no Isaac)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rocobrick.gripper import PIPER_L_PAD_SEPARATION_INTERCEPT_M
from rocobrick.moveit_contract import (
    TCP_LINK,
    TCP_STATUS_CALIBRATED_PROJECT,
)
from rocobrick.skill_targets import (
    PickTargets,
    RobotBaseFrame,
    SkillParams,
    TcpProvenance,
    brick_span_m,
    closing_yaw_deg,
    grasp_orientation_xyzw,
    load_robot_base_frame,
    load_skill_params,
    pick_targets,
    transit_target,
    world_position_to_base,
    world_orientation_to_base,
    _quat_mul,
    _quat_rotate_vector,
)

ROOT = Path(__file__).parents[1]
SKILL_PARAMS_JSON = ROOT / "config" / "skill_params.json"
USER_CONFIG_JSON = ROOT / "config" / "user_config.json"

HASH_A = "a" * 64
HASH_B = "b" * 64
PROVENANCE = TcpProvenance(
    runtime_model_sha256=HASH_A, source_sha256=HASH_B
)
SQRT_HALF = math.sqrt(0.5)  # 1e-9-exact sqrt(2)/2 for the pinned orientations

# A 2x4 brick footprint: dx = 0.032 > dy = 0.016 -> closing yaw 0, span dy.
WIDE_MIN = (1.0, 1.0, 0.0)
WIDE_MAX = (1.032, 1.016, 0.016)
# A 4x2 brick footprint: dy = 0.032 > dx = 0.016 -> closing yaw 90, span dx.
TALL_MIN = (1.0, 1.0, 0.0)
TALL_MAX = (1.016, 1.032, 0.016)

IDENTITY_BASE = RobotBaseFrame(
    position_xyz=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)
)


def rotated_base() -> RobotBaseFrame:
    """piper_0's real base: origin (0.25, -0.25, 0), 90 degrees about z."""
    return RobotBaseFrame(
        position_xyz=(0.25, -0.25, 0.0),
        orientation_xyzw=(0.0, 0.0, SQRT_HALF, SQRT_HALF),
    )


# ---------------------------------------------------------------------------
# Quaternion helpers and grasp orientation semantics
# ---------------------------------------------------------------------------


def test_quat_mul_composes_two_z_rotations() -> None:
    quarter: tuple[float, float, float, float] = (0.0, 0.0, SQRT_HALF, SQRT_HALF)
    half = _quat_mul(quarter, quarter)
    assert half == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-12)


def test_grasp_orientation_yaw_zero_is_identity_quaternion() -> None:
    assert grasp_orientation_xyzw(0.0) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_grasp_orientation_yaw_ninety_is_exact() -> None:
    assert grasp_orientation_xyzw(90.0) == pytest.approx(
        (SQRT_HALF, SQRT_HALF, 0.0, 0.0), abs=1e-9
    )


def test_grasp_orientation_points_the_tool_z_axis_down() -> None:
    for yaw in (0.0, 30.0, 90.0, 180.0):
        orientation = grasp_orientation_xyzw(yaw)
        assert _quat_rotate_vector(orientation, (0.0, 0.0, 1.0)) == pytest.approx(
            (0.0, 0.0, -1.0), abs=1e-12
        )


def test_grasp_orientation_keeps_the_tool_y_axis_horizontal() -> None:
    for yaw in (0.0, 30.0, 90.0, 270.0):
        orientation = grasp_orientation_xyzw(yaw)
        tool_y = _quat_rotate_vector(orientation, (0.0, 1.0, 0.0))
        assert tool_y[2] == pytest.approx(0.0, abs=1e-12)
        # The horizontal tool y axis is yaw-rotated: at yaw 0 it is -y, at
        # yaw 90 it lies along +x.
        if yaw == 0.0:
            assert tool_y == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)
        if yaw == 90.0:
            assert tool_y[0] == pytest.approx(1.0, abs=1e-12)


def test_grasp_orientation_is_normalized() -> None:
    for yaw in (-45.0, 0.0, 13.75, 90.0, 360.0):
        norm = math.sqrt(sum(value**2 for value in grasp_orientation_xyzw(yaw)))
        assert norm == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Closing-yaw and span rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("aabb_min", "aabb_max", "expected_yaw"),
    [
        (WIDE_MIN, WIDE_MAX, 0.0),  # dy (0.016) <= dx (0.032): straddle x
        (TALL_MIN, TALL_MAX, 90.0),  # dy (0.032) > dx (0.016): straddle y
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 180.0),  # square uses symmetric yaw
        ((0.0, 0.0, 0.0), (1.0, 1.00005, 1.0), 180.0),  # measurement noise is square
        ((0.0, 0.0, 0.0), (1.0, 1.0002, 1.0), 90.0),  # larger y uses quarter-turn
    ],
)
def test_closing_yaw_rule(
    aabb_min: tuple, aabb_max: tuple, expected_yaw: float
) -> None:
    assert closing_yaw_deg(aabb_min, aabb_max) == expected_yaw


@pytest.mark.parametrize(
    ("aabb_min", "aabb_max", "expected_span"),
    [
        (WIDE_MIN, WIDE_MAX, 0.016),  # yaw 0 -> dy, the extent under the closing axis
        (TALL_MIN, TALL_MAX, 0.016),  # yaw 90 -> dx
    ],
)
def test_brick_span_follows_the_closing_axis(
    aabb_min: tuple, aabb_max: tuple, expected_span: float
) -> None:
    assert brick_span_m(aabb_min, aabb_max) == pytest.approx(expected_span)


# ---------------------------------------------------------------------------
# World -> base transforms
# ---------------------------------------------------------------------------


def test_world_position_to_base_identity_base_is_the_point_itself() -> None:
    point = (1.0, -2.0, 3.0)
    assert world_position_to_base(point, IDENTITY_BASE) == point


def test_world_position_to_base_round_trip_with_rotated_base() -> None:
    base = rotated_base()
    local = (0.1, -0.05, 0.2)
    # piper_0's base is rotated +90 degrees about z: world = t + R(q) * local.
    world = (0.25 + -local[1], -0.25 + local[0], 0.0 + local[2])
    assert world_position_to_base(world, base) == pytest.approx(local)


def test_world_orientation_to_base_applies_inverse_base_rotation() -> None:
    base = rotated_base()
    assert world_orientation_to_base((1.0, 0.0, 0.0, 0.0), base) == pytest.approx(
        (SQRT_HALF, -SQRT_HALF, 0.0, 0.0), abs=1e-9
    )


# ---------------------------------------------------------------------------
# Derived TCP targets
# ---------------------------------------------------------------------------


def test_transit_target_height_stacks_standoff_and_extra() -> None:
    target = transit_target(
        WIDE_MIN, WIDE_MAX, IDENTITY_BASE, PROVENANCE, SkillParams()
    )
    center_x = (WIDE_MIN[0] + WIDE_MAX[0]) / 2.0
    center_y = (WIDE_MIN[1] + WIDE_MAX[1]) / 2.0
    center_z = (WIDE_MIN[2] + WIDE_MAX[2]) / 2.0
    params = SkillParams()
    assert target.frame_id == "base_link"
    assert target.position == pytest.approx(
        (
            center_x,
            center_y,
            center_z + params.pregrasp_standoff_m + params.transit_extra_height_m,
        )
    )
    assert target.orientation_xyzw == pytest.approx(
        grasp_orientation_xyzw(closing_yaw_deg(WIDE_MIN, WIDE_MAX))
    )
    assert target.tip_link == TCP_LINK
    assert target.tcp_status == TCP_STATUS_CALIBRATED_PROJECT
    assert target.runtime_model_sha256 == HASH_A
    assert target.source_sha256 == HASH_B
    assert target.position_tolerance_m == params.position_tolerance_m
    assert target.orientation_tolerance_rad == params.orientation_tolerance_rad


def test_transit_target_expresses_the_height_in_the_base_frame() -> None:
    params = SkillParams()
    target = transit_target(WIDE_MIN, WIDE_MAX, rotated_base(), PROVENANCE, params)
    center_z = (WIDE_MIN[2] + WIDE_MAX[2]) / 2.0
    # The rotated base has no z offset or tilt: only x/y change frame.
    assert target.position[2] == pytest.approx(
        center_z + params.pregrasp_standoff_m + params.transit_extra_height_m
    )
    assert target.orientation_xyzw == pytest.approx(
        world_orientation_to_base(
            grasp_orientation_xyzw(closing_yaw_deg(WIDE_MIN, WIDE_MAX)),
            rotated_base(),
        )
    )


def test_pick_targets_heights_windows_and_provenance() -> None:
    params = SkillParams()
    targets = pick_targets(WIDE_MIN, WIDE_MAX, IDENTITY_BASE, PROVENANCE, params)
    assert isinstance(targets, PickTargets)
    center_x = (WIDE_MIN[0] + WIDE_MAX[0]) / 2.0
    center_y = (WIDE_MIN[1] + WIDE_MAX[1]) / 2.0
    center_z = (WIDE_MIN[2] + WIDE_MAX[2]) / 2.0
    assert targets.pregrasp.position == pytest.approx(
        (center_x, center_y, center_z + params.pregrasp_standoff_m)
    )
    assert targets.approach.position == pytest.approx(
        (center_x, center_y, center_z + params.approach_clearance_m)
    )
    assert targets.lift.position == pytest.approx(
        (
            center_x,
            center_y,
            center_z + params.pregrasp_standoff_m + params.lift_height_m,
        )
    )
    assert targets.closing_yaw_deg == 0.0
    assert targets.brick_span_m == pytest.approx(0.016)
    # Grasp widths are physical pad separations converted to logical openings:
    # pads reach span + open_extra, press to span - squeeze, and the close
    # window centers on the logical opening whose pads sit at the span.
    assert targets.reach_opening_m == pytest.approx(
        0.016 + params.open_extra_m - PIPER_L_PAD_SEPARATION_INTERCEPT_M
    )
    assert targets.close_opening_m == pytest.approx(
        0.016 - params.squeeze_m - PIPER_L_PAD_SEPARATION_INTERCEPT_M
    )
    assert targets.reach_expected_window_m == pytest.approx(
        (
            0.016 + params.open_extra_m - PIPER_L_PAD_SEPARATION_INTERCEPT_M
            - params.free_window_slop_m,
            0.016 + params.open_extra_m - PIPER_L_PAD_SEPARATION_INTERCEPT_M
            + params.free_window_slop_m,
        )
    )
    assert targets.close_expected_window_m == pytest.approx(
        (
            0.016 - PIPER_L_PAD_SEPARATION_INTERCEPT_M
            - params.contact_window_slop_m,
            0.016 - PIPER_L_PAD_SEPARATION_INTERCEPT_M
            + params.contact_window_slop_m,
        )
    )
    for pose in (targets.pregrasp, targets.approach, targets.lift):
        assert pose.frame_id == "base_link"
        assert pose.tip_link == TCP_LINK
        assert pose.tcp_status == TCP_STATUS_CALIBRATED_PROJECT
        assert pose.runtime_model_sha256 == HASH_A
        assert pose.source_sha256 == HASH_B
        assert pose.position_tolerance_m == params.position_tolerance_m
        assert pose.orientation_tolerance_rad == params.orientation_tolerance_rad
        assert pose.orientation_xyzw == pytest.approx(
            grasp_orientation_xyzw(0.0)
        )


def test_pick_targets_rejects_a_brick_too_narrow_to_squeeze() -> None:
    """A brick no logical opening can squeeze fails closed at target build."""
    # With the distal faces coincident at zero, only a squeezed width below
    # zero is outside the graspable range.
    narrow_min = (1.0, 1.0, 0.0)
    narrow_max = (1.032, 1.003, 0.016)
    with pytest.raises(ValueError, match="graspable range"):
        pick_targets(
            narrow_min, narrow_max, IDENTITY_BASE, PROVENANCE, SkillParams()
        )


def test_pick_targets_xy_follows_the_rotated_base_frame() -> None:
    targets = pick_targets(
        WIDE_MIN, WIDE_MAX, rotated_base(), PROVENANCE, SkillParams()
    )
    center_x = (WIDE_MIN[0] + WIDE_MAX[0]) / 2.0
    center_y = (WIDE_MIN[1] + WIDE_MAX[1]) / 2.0
    expected_x, expected_y, _ = world_position_to_base(
        (center_x, center_y, 0.0), rotated_base()
    )
    for pose in (targets.pregrasp, targets.approach, targets.lift):
        assert pose.position[0] == pytest.approx(expected_x)
        assert pose.position[1] == pytest.approx(expected_y)


# ---------------------------------------------------------------------------
# SkillParams validation and strict skill_params.json loading
# ---------------------------------------------------------------------------


def test_skill_params_defaults_match_the_pinned_values() -> None:
    params = SkillParams()
    assert params.pregrasp_standoff_m == 0.08
    assert params.lift_height_m == 0.03
    assert params.transit_extra_height_m == 0.10
    assert params.open_extra_m == 0.02
    assert params.squeeze_m == 0.004
    assert params.contact_window_slop_m == 0.006
    assert params.free_window_slop_m == 0.01
    assert params.gripper_duration_s == 4.0
    assert params.gripper_goal_tolerance == 0.002
    assert params.wait_duration_s == 2.0
    assert params.position_tolerance_m == 0.005
    assert params.orientation_tolerance_rad == 0.015
    assert params.follow_orientation_tolerance_rad == 0.05
    assert params.still_tolerance_m == 0.01
    assert params.hold_follow_tolerance_m == 0.02
    assert params.scene_query_poll_s == 0.2


@pytest.mark.parametrize(
    "overrides",
    [
        {"pregrasp_standoff_m": 0.0},
        {"lift_height_m": -0.1},
        {"transit_extra_height_m": 0.0},
        {"open_extra_m": -0.01},
        {"squeeze_m": -1.0},
        {"contact_window_slop_m": 0.0},
        {"free_window_slop_m": 0.0},
        {"gripper_duration_s": 0.0},
        {"gripper_goal_tolerance": -0.5},
        {"wait_duration_s": 0.0},
        {"position_tolerance_m": 0.0},
        {"position_tolerance_m": 0.02},
        {"orientation_tolerance_rad": 0.0},
        {"orientation_tolerance_rad": 0.05},
        {"still_tolerance_m": 0.0},
        {"hold_follow_tolerance_m": 0.0},
        {"scene_query_poll_s": 0.0},
    ],
)
def test_skill_params_rejects_out_of_bounds_values(overrides: dict) -> None:
    with pytest.raises(ValueError):
        SkillParams(**overrides)


def test_skill_params_rejects_non_finite_and_non_numeric() -> None:
    with pytest.raises(ValueError):
        SkillParams(lift_height_m=float("inf"))
    with pytest.raises(ValueError):
        SkillParams(lift_height_m="0.1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SkillParams(lift_height_m=True)  # type: ignore[arg-type]


def test_load_skill_params_round_trips_a_valid_document(tmp_path: Path) -> None:
    defaults = {
        name: getattr(SkillParams(), name) for name in vars(SkillParams())
    }
    document = {**defaults, "pregrasp_standoff_m": 0.05, "wait_duration_s": 0.25}
    path = tmp_path / "params.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    params = load_skill_params(path)
    assert params.pregrasp_standoff_m == 0.05
    assert params.wait_duration_s == 0.25
    assert params.lift_height_m == SkillParams().lift_height_m


@pytest.mark.parametrize(
    "document",
    [
        {"lift_height_m": 0.1},  # missing every other key
        {"extra_key": 1.0},
        [],
        {},
    ],
)
def test_load_skill_params_rejects_key_set_deviations(
    tmp_path: Path, document: object
) -> None:
    path = tmp_path / "params.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_skill_params(path)


def test_load_skill_params_rejects_unreadable_and_non_finite(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_skill_params(tmp_path / "missing.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_skill_params(malformed)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(
        json.dumps(
            {
                **{
                    field: getattr(SkillParams(), field)
                    for field in SkillParams().__dataclass_fields__
                },
                "lift_height_m": float("inf"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_skill_params(nonfinite)


def test_config_skill_params_json_equals_the_defaults() -> None:
    assert SKILL_PARAMS_JSON.is_file()
    params = load_skill_params(SKILL_PARAMS_JSON)
    assert params == SkillParams()
    document = json.loads(SKILL_PARAMS_JSON.read_text(encoding="utf-8"))
    expected = {
        name: getattr(SkillParams(), name) for name in vars(SkillParams())
    }
    assert document == expected
    assert SKILL_PARAMS_JSON.read_text(encoding="utf-8").endswith("\n")


# ---------------------------------------------------------------------------
# load_robot_base_frame against the user_config.json structure
# ---------------------------------------------------------------------------


def minimal_user_config() -> dict:
    """The real piper_0 entry of config/user_config.json, minimally copied."""
    return {
        "Robot_Config": {
            "Robots": [
                {
                    "Name": "piper_0",
                    "Robot_Prim_Path": "/World/Robot_0",
                    "Joint_Order": [
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                        "gripper_joint1",
                        "gripper_joint2",
                    ],
                    "Robot_Base_Frame": {
                        "Position": [0.25, -0.25, 0.0],
                        "Orientation": [0.7071, 0, 0, 0.7071],
                    },
                }
            ]
        }
    }


def test_load_robot_base_frame_reads_the_real_user_config() -> None:
    base = load_robot_base_frame(USER_CONFIG_JSON, "piper_0")
    assert base.position_xyz == pytest.approx((0.25, -0.25, 0.0))
    # The config stores WXYZ; xyzw (0, 0, 0.7071, 0.7071) normalized.
    assert base.orientation_xyzw == pytest.approx(
        (0.0, 0.0, SQRT_HALF, SQRT_HALF), abs=1e-4
    )


def test_load_robot_base_frame_reads_a_minimal_config_fixture(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "user_config.json"
    config_path.write_text(json.dumps(minimal_user_config()), encoding="utf-8")
    base = load_robot_base_frame(config_path, "piper_0")
    assert base.position_xyz == (0.25, -0.25, 0.0)
    norm = math.sqrt(sum(value**2 for value in base.orientation_xyzw))
    assert norm == pytest.approx(1.0, abs=1e-12)
    assert base.orientation_xyzw[0] == pytest.approx(0.0, abs=1e-12)
    assert base.orientation_xyzw[1] == pytest.approx(0.0, abs=1e-12)
    assert base.orientation_xyzw[2] > 0.0
    assert base.orientation_xyzw[3] > 0.0


def test_load_robot_base_frame_rejects_missing_robot(tmp_path: Path) -> None:
    config_path = tmp_path / "user_config.json"
    config_path.write_text(json.dumps(minimal_user_config()), encoding="utf-8")
    with pytest.raises(ValueError, match="piper_9"):
        load_robot_base_frame(config_path, "piper_9")


def test_load_robot_base_frame_rejects_zero_norm_orientation(
    tmp_path: Path,
) -> None:
    config = minimal_user_config()
    config["Robot_Config"]["Robots"][0]["Robot_Base_Frame"]["Orientation"] = [
        0,
        0,
        0,
        0,
    ]
    config_path = tmp_path / "user_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="orientation"):
        load_robot_base_frame(config_path, "piper_0")


def test_load_robot_base_frame_rejects_non_finite_values(tmp_path: Path) -> None:
    config = minimal_user_config()
    config["Robot_Config"]["Robots"][0]["Robot_Base_Frame"]["Position"] = [
        0.25,
        -0.25,
        "NaN",
    ]
    config_path = tmp_path / "user_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Position"):
        load_robot_base_frame(config_path, "piper_0")
    with pytest.raises(ValueError):
        load_robot_base_frame(tmp_path / "missing.json", "piper_0")
