"""R1 MoveIt library-layer contract tests (pure, offline, no live ROS).

Pins the exact P1.B / P1.C behavior already verified live
(artifacts/phase1-real/joint-target.json and pose-target.json): frozen
contract types and constants, strict caller-side admission for joint and
link6 pose goals (never clamping), waypoint bounds, readiness predicates,
feedback metrics, the execution success predicate, and the absence of any
ROS/Isaac/tools import in the production modules (checked via AST).
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import math
from pathlib import Path

import pytest

from rocobrick.admission import (
    strict_action_admission,
    validate_gripper_target,
    validate_joint_target,
    validate_link6_pose_target,
    validate_trajectory_waypoints,
)
from rocobrick.feedback import (
    max_joint_drift,
    max_joint_error,
    orientation_error_rad,
    position_error_m,
    state_is_stable,
)
from rocobrick.moveit_contract import (
    ACTION_ARM_FOLLOW,
    ACTION_GRIPPER_FOLLOW,
    ARM_JOINT_NAMES,
    BASE_FRAME,
    EXECUTION_POSITION_LIMITS,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPENING_RANGE_M,
    JOINT_NAMES,
    LIMIT_EPSILON,
    MAX_ORIENTATION_TOLERANCE_RAD,
    MAX_POSITION_TOLERANCE_M,
    MIN_WAYPOINTS,
    MOVE_GROUP_ACTION_CANDIDATES,
    SERVICE_COMPUTE_FK,
    SERVICE_LIST_CONTROLLERS,
    SERVICE_PLAN,
    SUCCESS_CODE,
    TCP_STATUS,
    TIP_LINK,
    TOPIC_CLOCK,
    TOPIC_ISAAC_JOINT_COMMAND,
    TOPIC_ISAAC_JOINT_STATES,
    TOPIC_JOINT_STATES,
    AdmissionDecision,
    GripperTarget,
    JointTarget,
    Link6PoseTarget,
)
from rocobrick.moveit_execution import (
    ExecutionResult,
    ExecutionStatus,
    MoveItExecutor,
    is_success,
)
from rocobrick.moveit_requests import (
    build_joint_target,
    build_link6_pose_target,
    normalize_quaternion,
    validate_target_within_limits,
)
from rocobrick.readiness import (
    clock_is_increasing,
    controllers_are_active,
    readiness_decision,
    state_is_complete,
    states_match,
)

ROOT = Path(__file__).parents[1]
JOINT_ARTIFACT = ROOT / "artifacts" / "phase1-real" / "joint-target.json"
POSE_ARTIFACT = ROOT / "artifacts" / "phase1-real" / "pose-target.json"

PRODUCTION_MODULES = (
    "moveit_contract",
    "moveit_requests",
    "admission",
    "readiness",
    "feedback",
    "moveit_execution",
)
FORBIDDEN_IMPORT_ROOTS = {
    "rclpy",
    "moveit_msgs",
    "sensor_msgs",
    "isaacsim",
    "omni",
    "tools",
    "geometry_msgs",
    "action_msgs",
    "control_msgs",
    "shape_msgs",
    "rosgraph_msgs",
    "controller_manager_msgs",
    "builtin_interfaces",
}

ACTIVE_CONTROLLERS = {
    "joint_state_broadcaster": "active",
    "arm_controller": "active",
    "gripper_controller": "active",
}


def _artifact(path: Path) -> dict:
    """Load a phase1-real artifact or skip when it is absent."""
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def joint_artifact() -> dict:
    """Return the live-verified P1.B joint-target evidence."""
    return _artifact(JOINT_ARTIFACT)["probe"]


@pytest.fixture(scope="module")
def pose_artifact() -> dict:
    """Return the live-verified P1.C pose-target evidence."""
    return _artifact(POSE_ARTIFACT)["probe"]


@pytest.fixture
def live_baseline(joint_artifact) -> dict[str, float]:
    """Return the live 8-DOF baseline captured by the P1.B probe."""
    return dict(joint_artifact["baseline"])


@pytest.fixture
def live_target(joint_artifact) -> JointTarget:
    """Return the target admitted live by the P1.B probe."""
    return JointTarget(
        positions={name: float(joint_artifact["admission"]["target"][name]) for name in ARM_JOINT_NAMES}
    )


@pytest.fixture
def live_pose(pose_artifact) -> dict:
    """Return the live FK-derived target pose of the P1.C probe."""
    target_fk = pose_artifact["target_fk"]
    return {
        "frame_id": target_fk["frame_id"],
        "position": [float(target_fk["position"][key]) for key in ("x", "y", "z")],
        "orientation_xyzw": [
            float(target_fk["orientation"][key]) for key in ("x", "y", "z", "w")
        ],
    }


# --------------------------------------------------------------------- constants


def test_joint_names_are_the_exact_canonical_eight():
    assert JOINT_NAMES == (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "gripper_joint1",
        "gripper_joint2",
    )
    assert len(JOINT_NAMES) == 8


def test_arm_and_gripper_names_partition_joint_names():
    assert ARM_JOINT_NAMES == JOINT_NAMES[:6]
    assert GRIPPER_JOINT_NAMES == JOINT_NAMES[6:]
    assert tuple(ARM_JOINT_NAMES) + tuple(GRIPPER_JOINT_NAMES) == JOINT_NAMES


def test_execution_position_limits_are_the_exact_six_arm_limits(joint_artifact):
    expected = {
        name: tuple(limits) for name, limits in joint_artifact["execution_position_limits"].items()
    }
    assert dict(EXECUTION_POSITION_LIMITS) == {
        "joint1": (-2.6179938, 2.6179938),
        "joint2": (0.0, 3.1415926),
        "joint3": (-2.9670597, 0.0),
        "joint4": (-1.7453292, 1.7453292),
        "joint5": (-1.2217304, 1.2217304),
        "joint6": (-2.0943951, 2.0943951),
    }
    assert dict(EXECUTION_POSITION_LIMITS) == expected
    assert set(EXECUTION_POSITION_LIMITS) == set(ARM_JOINT_NAMES)


def test_execution_position_limits_mapping_is_readonly():
    with pytest.raises(TypeError):
        EXECUTION_POSITION_LIMITS["joint1"] = (0.0, 0.0)  # type: ignore[index]


def test_topic_names_match_the_pinned_contract():
    assert TOPIC_CLOCK == "/clock"
    assert TOPIC_ISAAC_JOINT_STATES == "/isaac_joint_states"
    assert TOPIC_JOINT_STATES == "/joint_states"
    assert TOPIC_ISAAC_JOINT_COMMAND == "/isaac_joint_command"


def test_action_and_service_names_match_the_live_evidence(joint_artifact, pose_artifact):
    assert MOVE_GROUP_ACTION_CANDIDATES == ("/move_action", "/move_group")
    assert MOVE_GROUP_ACTION_CANDIDATES == tuple(joint_artifact["actions"]["move_group_candidates"])
    assert ACTION_ARM_FOLLOW == "/arm_controller/follow_joint_trajectory"
    assert ACTION_ARM_FOLLOW == joint_artifact["actions"]["arm_follow"]
    assert SERVICE_LIST_CONTROLLERS == "/controller_manager/list_controllers"
    assert SERVICE_PLAN == joint_artifact["services"]["plan_candidates"][0]
    assert SERVICE_COMPUTE_FK == "/compute_fk"
    assert SERVICE_COMPUTE_FK == pose_artifact["services"]["compute_fk"]


def test_numeric_contract_constants():
    assert SUCCESS_CODE == 1
    assert MIN_WAYPOINTS == 2
    assert LIMIT_EPSILON == 1e-4


def test_pose_scope_constants():
    assert BASE_FRAME == "base_link"
    assert TIP_LINK == "link6"
    assert TCP_STATUS == "unresolved"
    assert MAX_POSITION_TOLERANCE_M == 0.01
    assert MAX_ORIENTATION_TOLERANCE_RAD == 0.03


def test_production_modules_import_without_ros():
    for name in PRODUCTION_MODULES:
        module = importlib.import_module(f"rocobrick.{name}")
        assert module is not None


@pytest.mark.parametrize("module_name", PRODUCTION_MODULES)
def test_production_modules_have_no_forbidden_imports(module_name):
    source = (ROOT / "src" / "rocobrick" / f"{module_name}.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots = {node.module.split(".")[0]}
        else:
            continue
        banned = roots & FORBIDDEN_IMPORT_ROOTS
        assert not banned, f"{module_name} imports forbidden modules: {sorted(banned)}"


# ------------------------------------------------------------ frozen dataclasses


def test_joint_target_is_frozen_with_defaults():
    target = JointTarget(positions={"joint1": 0.1})
    assert target.positions == {"joint1": 0.1}
    assert target.goal_tolerance == 0.01
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.goal_tolerance = 0.5  # type: ignore[misc]


def test_link6_pose_target_defaults_pin_the_pose_scope():
    target = Link6PoseTarget(
        frame_id="base_link",
        position=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert target.position_tolerance_m == 0.01
    assert target.orientation_tolerance_rad == 0.03
    assert target.tip_link == "link6"
    assert target.tcp_status == "unresolved"
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.tip_link = "tool0"  # type: ignore[misc]


def test_admission_decision_carries_reason_and_details():
    decision = AdmissionDecision(accepted=True, reason="ok", details={"margin": 0.03})
    assert decision.accepted is True
    assert decision.reason == "ok"
    assert decision.details == {"margin": 0.03}
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.accepted = False  # type: ignore[misc]


# ------------------------------------------------------------- moveit_requests


def test_build_joint_target_replays_the_live_admission(live_baseline, joint_artifact):
    deltas = {name: float(value) for name, value in joint_artifact["target_deltas"].items()}
    target, decision = build_joint_target(live_baseline, deltas)
    assert decision.accepted is True
    assert target is not None
    assert list(target.positions) == list(ARM_JOINT_NAMES)
    assert target.positions == pytest.approx(
        {name: float(value) for name, value in joint_artifact["admission"]["target"].items()},
        abs=1e-8,
    )
    assert target.goal_tolerance == 0.01
    assert decision.details["margin"] == 0.03
    assert decision.details["chosen_deltas"] == {"joint1": pytest.approx(0.2), "joint4": pytest.approx(0.15)}


def test_build_joint_target_flips_sign_near_a_bound_instead_of_clamping():
    current = {name: 0.0 for name in ARM_JOINT_NAMES}
    current["joint1"] = 2.5
    target, decision = build_joint_target(current, {"joint1": 0.5})
    assert decision.accepted is True
    assert target is not None
    assert target.positions["joint1"] == pytest.approx(2.0)
    assert decision.details["chosen_deltas"]["joint1"] == pytest.approx(-0.5)


def test_build_joint_target_rejects_when_neither_sign_fits():
    current = {name: 0.0 for name in ARM_JOINT_NAMES}
    target, decision = build_joint_target(current, {"joint1": 3.0})
    assert target is None
    assert decision.accepted is False
    assert decision.details["joint"] == "joint1"
    assert "joint1" in decision.reason


def test_build_joint_target_rejects_nonfinite_current():
    current = {name: 0.0 for name in ARM_JOINT_NAMES}
    current["joint2"] = float("nan")
    target, decision = build_joint_target(current, {"joint1": 0.1})
    assert target is None
    assert decision.accepted is False
    assert "joint2" in decision.reason


def test_build_joint_target_rejects_unknown_and_nonfinite_deltas(live_baseline):
    for deltas in ({"joint7": 0.1}, {"joint1": float("inf")}):
        target, decision = build_joint_target(live_baseline, deltas)
        assert target is None
        assert decision.accepted is False


def test_build_joint_target_rejects_missing_arm_joint(live_baseline):
    current = {name: value for name, value in live_baseline.items() if name != "joint6"}
    target, decision = build_joint_target(current, {})
    assert target is None
    assert decision.accepted is False
    assert "joint6" in decision.reason


def test_build_joint_target_rejects_out_of_bounds_unchanged_joint(live_baseline):
    current = dict(live_baseline)
    current["joint3"] = -3.5
    target, decision = build_joint_target(current, {})
    assert target is None
    assert decision.accepted is False
    assert "joint3" in decision.reason


def test_build_joint_target_accepts_unchanged_joint_exactly_on_its_bound():
    current = {name: 0.0 for name in ARM_JOINT_NAMES}
    current["joint2"] = 3.1415926
    target, decision = build_joint_target(current, {})
    assert decision.accepted is True
    assert target is not None
    assert target.positions["joint2"] == pytest.approx(3.1415926)


def test_normalize_quaternion_identity_and_scaling():
    assert normalize_quaternion((0.0, 0.0, 0.0, 1.0)) == (0.0, 0.0, 0.0, 1.0)
    assert normalize_quaternion((0.0, 0.0, 0.0, 2.0)) == (0.0, 0.0, 0.0, 1.0)
    normalized = normalize_quaternion((1.0, 2.0, 3.0, 4.0))
    assert math.sqrt(sum(value * value for value in normalized)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "quaternion",
    [
        (float("nan"), 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, float("inf")),
        (1e-9, 1e-9, 1e-9, 1e-9),
    ],
)
def test_normalize_quaternion_rejects_nonfinite_and_near_zero(quaternion):
    with pytest.raises(ValueError):
        normalize_quaternion(quaternion)


def test_normalize_quaternion_rejects_wrong_length():
    with pytest.raises(ValueError):
        normalize_quaternion((0.0, 0.0, 1.0))


def test_build_link6_pose_target_replays_the_live_fk_pose(pose_artifact, live_pose):
    target, decision = build_link6_pose_target(
        frame_id=live_pose["frame_id"],
        position=live_pose["position"],
        orientation_xyzw=live_pose["orientation_xyzw"],
    )
    assert decision.accepted is True
    assert target is not None
    assert target.frame_id == "base_link"
    assert target.tip_link == "link6"
    assert target.tcp_status == "unresolved"
    assert target.position == pytest.approx(
        tuple(float(pose_artifact["target_fk"]["position"][key]) for key in ("x", "y", "z")),
        abs=1e-12,
    )
    assert target.orientation_xyzw == pytest.approx(
        tuple(float(pose_artifact["target_fk"]["orientation"][key]) for key in ("x", "y", "z", "w")),
        abs=1e-9,
    )
    assert target.position_tolerance_m == 0.01
    assert target.orientation_tolerance_rad == 0.03


def test_build_link6_pose_target_normalizes_the_requested_quaternion(live_pose):
    scaled = [2.0 * value for value in live_pose["orientation_xyzw"]]
    target, decision = build_link6_pose_target(
        frame_id="base_link",
        position=live_pose["position"],
        orientation_xyzw=scaled,
    )
    assert decision.accepted is True
    assert target is not None
    assert target.orientation_xyzw == pytest.approx(tuple(live_pose["orientation_xyzw"]), abs=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_id": "world"},
        {"position": (0.1, 0.0, float("nan"))},
        {"position": (0.1, 0.0)},
        {"orientation_xyzw": (1e-9, 1e-9, 1e-9, 1e-9)},
        {"orientation_xyzw": (float("inf"), 0.0, 0.0, 1.0)},
        {"orientation_xyzw": (0.0, 0.0, 1.0)},
        {"position_tolerance_m": 0.02},
        {"position_tolerance_m": 0.0},
        {"orientation_tolerance_rad": 0.05},
        {"orientation_tolerance_rad": -0.01},
    ],
)
def test_build_link6_pose_target_rejects_inadmissible_input(live_pose, kwargs):
    frame_id = kwargs.get("frame_id", "base_link")
    position = kwargs.get("position", live_pose["position"])
    orientation = kwargs.get("orientation_xyzw", live_pose["orientation_xyzw"])
    target, decision = build_link6_pose_target(
        frame_id=frame_id,
        position=position,
        orientation_xyzw=orientation,
        position_tolerance_m=kwargs.get("position_tolerance_m", 0.01),
        orientation_tolerance_rad=kwargs.get("orientation_tolerance_rad", 0.03),
    )
    assert target is None
    assert decision.accepted is False
    assert decision.reason


def test_validate_target_within_limits_accepts_the_live_target(live_target):
    decision = validate_target_within_limits(live_target.positions)
    assert decision.accepted is True


def test_validate_target_within_limits_rejects_out_of_bounds_and_margin():
    outside = {name: 0.0 for name in ARM_JOINT_NAMES}
    outside["joint1"] = 3.0
    assert validate_target_within_limits(outside).accepted is False
    near = {name: 0.0 for name in ARM_JOINT_NAMES}
    assert validate_target_within_limits(near, margin=0.03).accepted is False


def test_validate_target_within_limits_requires_exact_arm_names(live_target):
    missing = {name: value for name, value in live_target.positions.items() if name != "joint6"}
    extra = {**live_target.positions, "gripper_joint1": 0.0}
    assert validate_target_within_limits(missing).accepted is False
    assert validate_target_within_limits(extra).accepted is False


def test_validate_target_within_limits_rejects_nonfinite_and_never_clamps(live_target):
    nonfinite = dict(live_target.positions)
    nonfinite["joint1"] = float("nan")
    assert validate_target_within_limits(nonfinite).accepted is False
    before = dict(live_target.positions)
    validate_target_within_limits(nonfinite)
    assert live_target.positions == before


# ------------------------------------------------------------------- admission


def test_validate_trajectory_waypoints_accepts_the_live_planned_shape(joint_artifact):
    names = joint_artifact["outbound"]["planned_trajectory"]["trajectory_joint_names"]
    points = [
        [float(joint_artifact["admission"]["target"][name]) for name in names]
        for _ in range(joint_artifact["outbound"]["planned_point_count"])
    ]
    decision = validate_trajectory_waypoints(names, points)
    assert decision.accepted is True
    assert decision.details["waypoint_count"] == joint_artifact["outbound"]["planned_point_count"]


def test_validate_trajectory_waypoints_normalizes_column_order_for_checking_only():
    shuffled = ("joint3", "joint1", "joint2", "joint5", "joint4", "joint6")
    values = {"joint1": 0.1, "joint2": 0.2, "joint3": -0.3, "joint4": 0.4, "joint5": 0.0, "joint6": -0.1}
    points = [[values[name] for name in shuffled] for _ in range(3)]
    decision = validate_trajectory_waypoints(shuffled, points)
    assert decision.accepted is True


def test_validate_trajectory_waypoints_rejects_fewer_than_two_points():
    point = [0.0] * 6
    decision = validate_trajectory_waypoints(ARM_JOINT_NAMES, [point])
    assert decision.accepted is False
    assert decision.details["waypoint_count"] == 1


def test_validate_trajectory_waypoints_rejects_values_beyond_the_limits():
    inside = [0.0] * 6
    outside = [0.0] * 6
    outside[0] = 3.0
    decision = validate_trajectory_waypoints(ARM_JOINT_NAMES, [inside, outside])
    assert decision.accepted is False
    assert decision.details["over_limits"]["joint1"] == pytest.approx(3.0 - 2.6179938)


def test_validate_trajectory_waypoints_allows_only_epsilon_slack():
    edge = [0.0] * 6
    edge[1] = 3.1415926 + 5e-5
    assert validate_trajectory_waypoints(ARM_JOINT_NAMES, [[0.0] * 6, edge]).accepted is True
    beyond = [0.0] * 6
    beyond[1] = 3.1415926 + 2e-4
    assert validate_trajectory_waypoints(ARM_JOINT_NAMES, [[0.0] * 6, beyond]).accepted is False


def test_validate_trajectory_waypoints_rejects_nonfinite_and_bad_names():
    nonfinite = [0.0] * 6
    nonfinite[3] = float("nan")
    assert validate_trajectory_waypoints(ARM_JOINT_NAMES, [[0.0] * 6, nonfinite]).accepted is False
    assert validate_trajectory_waypoints(("joint1",), [[0.0], [0.0]]).accepted is False
    grippy = tuple(ARM_JOINT_NAMES) + ("gripper_joint1",)
    assert validate_trajectory_waypoints(grippy, [[0.0] * 7, [0.0] * 7]).accepted is False


def test_validate_joint_target_accepts_the_live_target(live_target):
    decision = validate_joint_target(live_target)
    assert decision.accepted is True


def test_validate_joint_target_rejects_bad_tolerance_and_bounds(live_target):
    for tolerance in (0.0, -0.01, float("nan")):
        decision = validate_joint_target(dataclasses.replace(live_target, goal_tolerance=tolerance))
        assert decision.accepted is False
    outside = dict(live_target.positions)
    outside["joint4"] = 2.0
    assert validate_joint_target(dataclasses.replace(live_target, positions=outside)).accepted is False


def test_validate_link6_pose_target_accepts_the_live_pose_target(live_pose):
    target, _ = build_link6_pose_target(
        frame_id=live_pose["frame_id"],
        position=live_pose["position"],
        orientation_xyzw=live_pose["orientation_xyzw"],
    )
    assert target is not None
    assert validate_link6_pose_target(target).accepted is True


def test_validate_link6_pose_target_rejects_frame_link_tcp_and_normalization(live_pose):
    target, _ = build_link6_pose_target(
        frame_id=live_pose["frame_id"],
        position=live_pose["position"],
        orientation_xyzw=live_pose["orientation_xyzw"],
    )
    assert target is not None
    assert validate_link6_pose_target(dataclasses.replace(target, frame_id="world")).accepted is False
    assert validate_link6_pose_target(dataclasses.replace(target, tip_link="tool0")).accepted is False
    assert validate_link6_pose_target(dataclasses.replace(target, tcp_status="resolved")).accepted is False
    scaled = tuple(2.0 * value for value in target.orientation_xyzw)
    assert validate_link6_pose_target(
        dataclasses.replace(target, orientation_xyzw=scaled)
    ).accepted is False


def test_strict_action_admission_dispatches_by_target_type(live_target, live_pose):
    assert strict_action_admission(live_target).accepted is True
    pose_target, _ = build_link6_pose_target(
        frame_id=live_pose["frame_id"],
        position=live_pose["position"],
        orientation_xyzw=live_pose["orientation_xyzw"],
    )
    assert pose_target is not None
    assert strict_action_admission(pose_target).accepted is True


def test_strict_action_admission_rejects_unknown_types(live_target):
    for unknown in (object(), "joint1", None):
        decision = strict_action_admission(unknown)  # type: ignore[arg-type]
        assert decision.accepted is False
        assert decision.details["type"] == type(unknown).__name__


def test_strict_action_admission_rejects_out_of_bounds_before_any_send():
    outside = JointTarget(positions={**{name: 0.0 for name in ARM_JOINT_NAMES}, "joint1": 9.0})
    decision = strict_action_admission(outside)
    assert decision.accepted is False
    assert decision.details["excess"]["joint1"] == pytest.approx(9.0 - 2.6179938)


def gripper_target(**overrides) -> GripperTarget:
    values = dict(
        opening_m=0.02,
        duration_s=1.0,
        goal_tolerance=0.005,
        expected_opening_m=(0.0, 0.03),
    )
    values.update(overrides)
    return GripperTarget(**values)


def test_gripper_contract_constants_pin_the_official_opening_range():
    assert ACTION_GRIPPER_FOLLOW == "/gripper_controller/follow_joint_trajectory"
    assert GRIPPER_OPENING_RANGE_M == (0.0, 0.1)


def test_validate_gripper_target_accepts_the_canonical_target():
    decision = validate_gripper_target(gripper_target())
    assert decision.accepted is True
    assert decision.reason == "ok"
    assert decision.details["opening_m"] == pytest.approx(0.02)
    # The exact bounds of the official range admit inclusive.
    assert validate_gripper_target(gripper_target(opening_m=0.0)).accepted is True
    assert validate_gripper_target(gripper_target(opening_m=0.1)).accepted is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"opening_m": 0.11},
        {"opening_m": -0.001},
        {"opening_m": float("nan")},
        {"opening_m": float("inf")},
        {"duration_s": 0.0},
        {"duration_s": -1.0},
        {"duration_s": float("nan")},
        {"goal_tolerance": 0.0},
        {"goal_tolerance": -0.005},
        {"goal_tolerance": float("inf")},
        {"expected_opening_m": (0.05, 0.02)},
        {"expected_opening_m": (-0.01, 0.02)},
        {"expected_opening_m": (0.0, 0.11)},
        {"expected_opening_m": (0.01,)},
        {"expected_opening_m": (0.01, 0.02, 0.03)},
        {"expected_opening_m": (0.0, float("nan"))},
    ],
)
def test_validate_gripper_target_rejects_bad_bounds(overrides):
    decision = validate_gripper_target(gripper_target(**overrides))
    assert decision.accepted is False
    assert decision.details["problems"]


def test_validate_gripper_target_does_not_require_the_window_to_contain_opening():
    """Closing onto a brick commands less than the expected contact width."""
    decision = validate_gripper_target(
        gripper_target(opening_m=0.0, expected_opening_m=(0.028, 0.032))
    )
    assert decision.accepted is True


def test_strict_action_admission_dispatches_gripper_targets():
    assert strict_action_admission(gripper_target()).accepted is True
    rejected = strict_action_admission(gripper_target(opening_m=0.5))
    assert rejected.accepted is False
    assert "opening_m" in rejected.reason


# ------------------------------------------------------------------- readiness


def test_state_is_complete_accepts_the_live_baseline(live_baseline):
    assert state_is_complete(live_baseline) is True
    assert state_is_complete({**live_baseline, "velocity": [0.0] * 8}) is True
    assert state_is_complete({**live_baseline, "velocity": []}) is True


def test_state_is_complete_rejects_incomplete_or_malformed_states(live_baseline):
    missing = {name: value for name, value in live_baseline.items() if name != "joint6"}
    extra = {**live_baseline, "flange": 0.0}
    nonfinite = {**live_baseline, "joint5": float("nan")}
    assert state_is_complete(missing) is False
    assert state_is_complete(extra) is False
    assert state_is_complete(nonfinite) is False
    assert state_is_complete({**live_baseline, "velocity": [0.0] * 3}) is False
    assert state_is_complete({**live_baseline, "velocity": "full"}) is False
    assert state_is_complete({}) is False


def test_states_match_is_the_exact_broadcaster_check(live_baseline):
    assert states_match(live_baseline, live_baseline) is True
    shifted = dict(live_baseline)
    shifted["joint1"] += 5e-5
    assert states_match(live_baseline, shifted) is True
    assert states_match(live_baseline, shifted, tolerance=1e-5) is False
    drifted = dict(live_baseline)
    drifted["joint4"] += 1e-3
    assert states_match(live_baseline, drifted) is False
    assert states_match(live_baseline, {"joint1": 0.0}) is False


def test_controllers_are_active_requires_the_three_named_controllers():
    assert controllers_are_active(ACTIVE_CONTROLLERS) is True
    assert controllers_are_active({**ACTIVE_CONTROLLERS, "extra_controller": "active"}) is True
    assert controllers_are_active({name: state for name, state in ACTIVE_CONTROLLERS.items() if name != "arm_controller"}) is False
    assert controllers_are_active({**ACTIVE_CONTROLLERS, "gripper_controller": "inactive"}) is False
    assert controllers_are_active({}) is False


def test_clock_is_increasing_needs_two_strictly_increasing_samples():
    assert clock_is_increasing([1, 2, 3]) is True
    assert clock_is_increasing([1, 1]) is False
    assert clock_is_increasing([3, 2, 1]) is False
    assert clock_is_increasing([1]) is False
    assert clock_is_increasing([]) is False


def test_readiness_decision_accepts_the_live_shape(live_baseline):
    decision = readiness_decision(
        clock_values=[1_000, 2_000, 3_000],
        isaac_state=live_baseline,
        broadcaster_state=dict(live_baseline),
        controllers=ACTIVE_CONTROLLERS,
    )
    assert decision.accepted is True
    assert decision.reason == "ok"
    assert all(decision.details.values())


def test_readiness_decision_rejects_each_failed_component(live_baseline):
    def decide(**overrides):
        kwargs = {
            "clock_values": [1_000, 2_000],
            "isaac_state": live_baseline,
            "broadcaster_state": dict(live_baseline),
            "controllers": ACTIVE_CONTROLLERS,
        }
        kwargs.update(overrides)
        return readiness_decision(**kwargs)

    assert decide(clock_values=[2_000, 1_000]).accepted is False
    assert decide(clock_values=[1_000]).accepted is False
    stale = dict(live_baseline)
    stale["joint3"] += 1e-2
    assert decide(broadcaster_state=stale).accepted is False
    assert decide(broadcaster_state=None).accepted is False
    assert decide(isaac_state={"joint1": 0.0}).accepted is False
    assert decide(controllers={**ACTIVE_CONTROLLERS, "arm_controller": "inactive"}).accepted is False

    failed = decide(controllers={})
    assert "controllers_active" in failed.reason
    assert failed.details["controllers_active"] is False


# -------------------------------------------------------------------- feedback


def test_max_joint_error_replays_the_live_final_errors(joint_artifact):
    outbound = joint_artifact["outbound"]
    final = outbound["final_positions"]
    target = outbound["target"]
    error = max_joint_error(final, target)
    assert error == pytest.approx(max(outbound["final_errors"].values()), abs=1e-8)


def test_max_joint_error_rejects_missing_or_nonfinite_joints(live_baseline):
    target = {name: 0.0 for name in ARM_JOINT_NAMES}
    with pytest.raises(ValueError):
        max_joint_error({"joint1": 0.0}, target)
    with pytest.raises(ValueError):
        max_joint_error(live_baseline, {**target, "joint5": float("inf")})


def test_max_joint_drift_replays_the_live_gripper_drift(joint_artifact):
    baseline = joint_artifact["baseline"]
    returned = joint_artifact["return"]["final_positions"]
    drift = max_joint_drift(returned, baseline, names=GRIPPER_JOINT_NAMES)
    assert drift == pytest.approx(joint_artifact["return"]["gripper_drift"], abs=1e-8)
    full = max_joint_drift(returned, baseline)
    assert full == pytest.approx(
        max(abs(returned[name] - baseline[name]) for name in JOINT_NAMES), abs=1e-12
    )


def test_max_joint_drift_rejects_unknown_names(live_baseline):
    with pytest.raises(ValueError):
        max_joint_drift(live_baseline, live_baseline, names=("elbow",))


def test_position_error_m_replays_the_live_fk_verification(pose_artifact):
    verification = pose_artifact["outbound"]["fk_verification"]
    fk_pose = verification["fk_pose"]
    requested = pose_artifact["outbound"]["requested_pose"]
    error = position_error_m(
        [fk_pose["position"][key] for key in ("x", "y", "z")],
        [requested["position"][key] for key in ("x", "y", "z")],
    )
    assert error == pytest.approx(verification["position_error_m"], rel=1e-5)
    assert position_error_m((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)


@pytest.mark.parametrize("actual", [(0.0, 0.0), (0.0, 0.0, float("nan")), (0.0, 0.0, 0.0, 0.0)])
def test_position_error_m_rejects_malformed_positions(actual):
    with pytest.raises(ValueError):
        position_error_m(actual, (0.0, 0.0, 0.0))


def test_orientation_error_rad_replays_the_live_fk_verification(pose_artifact):
    verification = pose_artifact["outbound"]["fk_verification"]
    fk_pose = verification["fk_pose"]
    requested = pose_artifact["outbound"]["requested_pose"]
    error = orientation_error_rad(
        [fk_pose["orientation"][key] for key in ("x", "y", "z", "w")],
        [requested["orientation"][key] for key in ("x", "y", "z", "w")],
    )
    assert error == pytest.approx(verification["orientation_error_rad"], rel=1e-4)


def test_orientation_error_rad_basic_angles():
    identity = (0.0, 0.0, 0.0, 1.0)
    quarter = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert orientation_error_rad(identity, identity) == pytest.approx(0.0)
    assert orientation_error_rad(identity, tuple(-value for value in identity)) == pytest.approx(0.0)
    assert orientation_error_rad(identity, quarter) == pytest.approx(math.pi / 2)


@pytest.mark.parametrize(
    "quaternion",
    [(float("nan"), 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
)
def test_orientation_error_rad_rejects_malformed_quaternions(quaternion):
    with pytest.raises(ValueError):
        orientation_error_rad(quaternion, (0.0, 0.0, 0.0, 1.0))


def test_state_is_stable_uses_the_per_sample_settle_step(live_baseline):
    quiet = dict(live_baseline)
    quiet["joint1"] += 1e-4
    moving = dict(live_baseline)
    moving["joint4"] += 1e-3
    assert state_is_stable(live_baseline, quiet) is True
    assert state_is_stable(live_baseline, moving) is False
    assert state_is_stable(live_baseline, moving, max_step=2e-3) is True


def test_state_is_stable_rejects_malformed_states_and_steps(live_baseline):
    with pytest.raises(ValueError):
        state_is_stable({"joint1": 0.0}, live_baseline)
    with pytest.raises(ValueError):
        state_is_stable(live_baseline, {**live_baseline, "joint6": float("nan")})
    with pytest.raises(ValueError):
        state_is_stable(live_baseline, live_baseline, max_step=-1.0)
    with pytest.raises(ValueError):
        state_is_stable(live_baseline, live_baseline, max_step=float("inf"))


# ------------------------------------------------------------ moveit_execution


def test_execution_status_members():
    assert {status.value for status in ExecutionStatus} == {
        "ACCEPTED",
        "SUCCEEDED",
        "REJECTED",
        "CANCELED",
        "TIMED_OUT",
        "FAILED",
    }
    assert isinstance(ExecutionStatus.SUCCEEDED, str)


def test_execution_result_is_frozen():
    result = ExecutionResult(status=ExecutionStatus.SUCCEEDED)
    assert result.moveit_error_code is None
    assert result.planned_waypoints == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = ExecutionStatus.FAILED  # type: ignore[misc]


def test_is_success_requires_every_p1_gate():
    success = ExecutionResult(
        status=ExecutionStatus.SUCCEEDED, moveit_error_code=SUCCESS_CODE, planned_waypoints=10
    )
    assert is_success(success) is True
    assert is_success(
        dataclasses.replace(success, planned_waypoints=MIN_WAYPOINTS - 1)
    ) is False
    assert is_success(dataclasses.replace(success, moveit_error_code=-1)) is False
    assert is_success(dataclasses.replace(success, moveit_error_code=None)) is False
    assert is_success(dataclasses.replace(success, status=ExecutionStatus.ACCEPTED)) is False
    assert is_success(dataclasses.replace(success, status=ExecutionStatus.CANCELED)) is False
    assert is_success(
        ExecutionResult(
            status=ExecutionStatus.FAILED, moveit_error_code=-1, planned_waypoints=0
        )
    ) is False


def test_moveit_executor_protocol_is_runtime_checkable(live_target, live_pose):
    class FakeExecutor:
        def execute_joint(self, target, *, timeout_s):
            return ExecutionResult(status=ExecutionStatus.SUCCEEDED)

        def execute_link6_pose(self, target, *, timeout_s):
            return ExecutionResult(status=ExecutionStatus.REJECTED)

    class Incomplete:
        def execute_joint(self, target, *, timeout_s):
            return ExecutionResult(status=ExecutionStatus.SUCCEEDED)

    assert isinstance(FakeExecutor(), MoveItExecutor)
    assert not isinstance(Incomplete(), MoveItExecutor)
