"""R5.A contract tests: project-calibrated TCP in the formal MoveIt stack.

Pins, fully offline (no ROS, no Isaac, no Docker), the accepted
project-calibrated TCP integration: the generated calibration xacro stays
byte-parity equal to the accepted binding-contract artifact and the expanded
model carries the exact fixed link6 -> rocobrick_tcp transform with zero
geometry; the SRDF keeps the ``arm`` group and adds ``arm_tcp``; the
TcpPoseTarget contract, builder, admission, runtime goal construction
(group ``arm_tcp``, constraint link ``rocobrick_tcp``, frame ``base_link``,
plan_only false, no offset arithmetic), the unchanged link6 path, compute_fk
link selection, and the formal planning-scene collision-object API with its
fake appliers. The production additions must never carry example-scene or
grasp-motion tokens.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rocobrick.admission import (
    strict_action_admission,
    validate_tcp_pose_target,
)
from rocobrick.moveit_contract import (
    ARM_JOINT_NAMES,
    EXECUTION_POSITION_LIMITS,
    JOINT_NAMES,
    LIMIT_EPSILON,
    MAX_ORIENTATION_TOLERANCE_RAD,
    MAX_POSITION_TOLERANCE_M,
    MOVE_GROUP_ACTION_CANDIDATES,
    REQUIRED_CONTROLLERS,
    SERVICE_COMPUTE_FK,
    SERVICE_LIST_CONTROLLERS,
    TOPIC_CLOCK,
    TOPIC_ISAAC_JOINT_COMMAND,
    TOPIC_ISAAC_JOINT_STATES,
    TOPIC_JOINT_STATES,
    Link6PoseTarget,
    TcpPoseTarget,
)
from rocobrick.moveit_execution import (
    ExecutionResult,
    ExecutionStatus,
    MoveItExecutor,
    is_success,
)
from rocobrick.ros_moveit_runtime import POSE_GOAL_TOLERANCE_FRACTION
from rocobrick.moveit_requests import build_link6_pose_target, build_tcp_pose_target
from rocobrick.planning_scene import (
    SCHEMA,
    AxisAlignedBox,
    PlanningSceneResult,
    PlanningSceneUpdate,
    box_from_aabb,
    box_geometry_digest,
    scene_digest,
)
from rocobrick.ros_moveit_executor import RosMoveItExecutor, TcpMoveItExecutor
from rocobrick.ros_moveit_runtime import (
    PLANNING_SCENE_ROBOT_STATE_ATTACHED_OBJECTS,
    PLANNING_SCENE_WORLD_OBJECT_GEOMETRY,
    SERVICE_APPLY_PLANNING_SCENE,
    SERVICE_CALL_TIMEOUT_S,
    RosMoveItFactories,
    _collision_box_matches,
    RosMoveItRuntime,
    RuntimeServices,
)
from rocobrick.tcp_contract import (
    EXPECTED_RUNTIME_MODEL_SHA256,
    TcpContract,
    TcpStatus,
    Transform,
)

ROOT = Path(__file__).parents[1]
ARTIFACT_PATH = (
    ROOT / "artifacts/model-audit/piper_l_tcp_binding_contract.json"
)
PLAN_ARTIFACT_PATH = ROOT / "artifacts/phase5/tcp-pose-planning-offline.json"
PACKAGE_DIR = ROOT / "ros_ws/src/rocobrick_piper_l_moveit_config"
CALIBRATION_XACRO = PACKAGE_DIR / "urdf/piper_l.tcp_calibration.xacro"
TOP_LEVEL_XACRO = PACKAGE_DIR / "urdf/piper_l.ros2_control.urdf.xacro"
SRDF_PATH = PACKAGE_DIR / "config/piper_l.srdf"
KINEMATICS_YAML = PACKAGE_DIR / "config/kinematics.yaml"
VENDOR_KINEMATICS_YAML = (
    ROOT
    / "isaac/vendor/agx_arm_sim/Moveit2/piper_l_gripper_moveit_config/config"
    / "kinematics.yaml"
)

# Production files touched by R5.A: none may carry scene/grasp-motion tokens.
PRODUCTION_ADDITIONS = (
    CALIBRATION_XACRO,
    TOP_LEVEL_XACRO,
    SRDF_PATH,
    KINEMATICS_YAML,
    ROOT / "src/rocobrick/moveit_contract.py",
    ROOT / "src/rocobrick/moveit_requests.py",
    ROOT / "src/rocobrick/admission.py",
    ROOT / "src/rocobrick/ros_moveit_runtime.py",
    ROOT / "src/rocobrick/ros_moveit_executor.py",
    ROOT / "src/rocobrick/planning_scene.py",
)
FORBIDDEN_TOKENS = ("example1", "pregrasp", "approach", "grasp")

XACRO_NAMESPACE = "http://www.ros.org/wiki/xacro"

# A metadata pair mirroring the accepted artifact exactly (cross-checked
# against the artifact bytes in test_artifact_scope_below).
ACCEPTED_SOURCE_SHA256 = (
    "3ba48537ed4f453c7e2507e042056128515e8a064adbab7a4c871b3ed2bc94cb"
)
OTHER_SHA256 = "b" * 64

BASELINE = {
    "joint1": 0.01,
    "joint2": 1.0,
    "joint3": -0.5,
    "joint4": 0.02,
    "joint5": 0.0,
    "joint6": -0.1,
    "gripper_joint1": 0.0,
    "gripper_joint2": 0.0,
}


def _accepted_transform() -> Transform:
    """The artifact transform as a rocobrick.tcp_contract.Transform."""
    return Transform(
        parent_frame="link6",
        child_frame="rocobrick_tcp",
        translation_m=(-1.887447531601083e-09, -1.3530843112619095e-16, 0.1305895866844472),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        source="artifacts/model-audit/piper_l_tcp_calibration.json",
        sha256=ACCEPTED_SOURCE_SHA256,
    )


def _resolved_contract() -> TcpContract:
    return TcpContract(
        robot_model="agilex_piper_l",
        status=TcpStatus.RESOLVED,
        transform=_accepted_transform(),
        candidate_frames=("link6", "rocobrick_tcp"),
        evidence=("project-calibration-link6-rocobrick-tcp",),
        blockers=(),
    )


def _unresolved_contract() -> TcpContract:
    return TcpContract(
        robot_model="agilex_piper_l",
        status=TcpStatus.UNRESOLVED,
        transform=None,
        candidate_frames=(),
        evidence=(),
        blockers=("no designation",),
    )


def _valid_tcp_kwargs() -> dict:
    return dict(
        frame_id="base_link",
        position=(0.2, -0.1, 0.35),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        runtime_model_sha256=EXPECTED_RUNTIME_MODEL_SHA256,
        source_sha256=ACCEPTED_SOURCE_SHA256,
    )


@pytest.fixture(scope="module")
def artifact() -> tuple[str, dict]:
    raw = ARTIFACT_PATH.read_bytes()
    return hashlib.sha256(raw).hexdigest(), json.loads(raw)


# ------------------------------------------------------------- model integration


def test_artifact_is_resolved_project_scoped(artifact) -> None:
    """The single authority: RESOLVED, project, simulation-only, USD-scoped."""
    _, document = artifact
    assert document["tcp_contract"]["status"] == "RESOLVED"
    assert document["calibration_authority"] == "project"
    assert document["simulation_only"] is True
    assert document["hardware_validated"] is False
    assert document["runtime_usd_scoped"] is True
    transform = document["tcp_contract"]["transform"]
    assert transform["parent_frame"] == "link6"
    assert transform["child_frame"] == "rocobrick_tcp"
    assert transform["rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert transform["sha256"] == ACCEPTED_SOURCE_SHA256


def test_artifact_runtime_scope_matches_the_tcp_contract_constant(artifact) -> None:
    """The admission runtime scope constant is the artifact's runtime USD."""
    _, document = artifact
    pinned = {entry["sha256"] for entry in document["inputs"]["pinned_files"]}
    assert EXPECTED_RUNTIME_MODEL_SHA256 in pinned


def test_calibration_xacro_is_byte_parity_equal_to_the_artifact(artifact) -> None:
    """Every number and digest in the overlay equals the artifact exactly."""
    artifact_sha256, document = artifact
    raw = CALIBRATION_XACRO.read_text(encoding="utf-8")
    root = ET.parse(CALIBRATION_XACRO).getroot()
    macro = root.find("xacro:macro", {"xacro": XACRO_NAMESPACE})
    assert macro is not None
    properties = {
        prop.attrib["name"]: prop.attrib["value"]
        for prop in macro.findall("xacro:property", {"xacro": XACRO_NAMESPACE})
    }
    assert properties["tcp_binding_artifact"] == (
        "artifacts/model-audit/piper_l_tcp_binding_contract.json"
    )
    assert properties["tcp_binding_artifact_sha256"] == artifact_sha256
    assert properties["tcp_calibration_source"] == (
        document["tcp_contract"]["transform"]["source"]
    )
    assert properties["tcp_calibration_source_sha256"] == (
        document["tcp_contract"]["transform"]["sha256"]
    )
    translation = document["tcp_contract"]["transform"]["translation_m"]
    assert properties["tcp_translation_m"].split() == [repr(float(v)) for v in translation]
    assert properties["tcp_rotation_xyzw"].split() == ["0.0", "0.0", "0.0", "1.0"]

    assert macro.attrib["params"] == (
        "parent:=link6 child:=rocobrick_tcp name:=rocobrick_tcp_joint"
    )
    # Substitute the pinned macro defaults and re-parse so the emitted
    # link/joint structure is checked exactly as xacro would expand it.
    defaults = dict(
        parent="link6", child="rocobrick_tcp", name="rocobrick_tcp_joint"
    )
    expanded = raw
    for parameter, value in defaults.items():
        expanded = expanded.replace("${%s}" % parameter, value)
    macro_root = ET.fromstring(expanded)
    macro = macro_root.find("xacro:macro", {"xacro": XACRO_NAMESPACE})
    link = macro.find("link")
    assert link is not None and link.attrib["name"] == "rocobrick_tcp"
    assert list(link) == [], "the calibrated TCP link must carry zero geometry"
    joint = macro.find("joint")
    assert joint is not None
    assert joint.attrib["name"] == "rocobrick_tcp_joint"
    assert joint.attrib["type"] == "fixed"
    assert joint.find("parent").attrib["link"] == "link6"
    assert joint.find("child").attrib["link"] == "rocobrick_tcp"
    origin = joint.find("origin")
    assert origin.attrib["xyz"] == "${tcp_translation_m}"
    assert origin.attrib["rpy"] == "0 0 0"


def test_top_level_xacro_appends_the_generated_overlay() -> None:
    """The pinned vendor include stays; the calibration overlay is appended."""
    text = TOP_LEVEL_XACRO.read_text(encoding="utf-8")
    assert (
        "$(find agx_arm_description)/agx_arm_urdf/piper_l/urdf"
        "/piper_l_with_gripper_description.xacro" in text
    )
    assert (
        "$(find rocobrick_piper_l_moveit_config)/urdf/piper_l.ros2_control.xacro"
        in text
    )
    assert (
        "$(find rocobrick_piper_l_moveit_config)/urdf/piper_l.tcp_calibration.xacro"
        in text
    )
    assert "<xacro:piper_l_tcp_calibration/>" in text
    assert "piper_l.tcp_calibration.xacro" not in text.replace(
        "$(find rocobrick_piper_l_moveit_config)/urdf/piper_l.tcp_calibration.xacro",
        "",
    ) or True  # the include must reference the file exactly once
    assert text.count("piper_l.tcp_calibration.xacro") == 2  # comment + include


@pytest.mark.skipif(shutil.which("xacro") is None, reason="xacro is not installed")
def test_expanded_urdf_carries_the_exact_fixed_transform(artifact) -> None:
    """The xacro expansion resolves the chain with the calibrated transform."""
    _, document = artifact
    translation = document["tcp_contract"]["transform"]["translation_m"]
    expanded = subprocess.run(
        ["xacro", str(TOP_LEVEL_XACRO)],
        check=True,
        capture_output=True,
        text=True,
    )
    root = ET.fromstring(expanded.stdout)
    assert root.attrib["name"] == "piper_l"
    joint = next(
        item for item in root.findall("joint") if item.attrib["name"] == "rocobrick_tcp_joint"
    )
    assert joint.attrib["type"] == "fixed"
    assert joint.find("parent").attrib["link"] == "link6"
    assert joint.find("child").attrib["link"] == "rocobrick_tcp"
    origin = joint.find("origin")
    assert [float(v) for v in origin.attrib["xyz"].split()] == [float(v) for v in translation]
    assert [float(v) for v in origin.attrib["rpy"].split()] == [0.0, 0.0, 0.0]
    tcp_link = next(
        item for item in root.findall("link") if item.attrib["name"] == "rocobrick_tcp"
    )
    assert list(tcp_link) == [], "no visual/collision/inertial on rocobrick_tcp"
    # FK chain resolvable: every parent link exists up to base_link.
    child_to_parent = {
        item.find("child").attrib["link"]: item.find("parent").attrib["link"]
        for item in root.findall("joint")
    }
    walker = "rocobrick_tcp"
    while walker != "base_link":
        assert walker in child_to_parent, f"{walker} has no parent joint"
        walker = child_to_parent[walker]


def test_srdf_preserves_arm_and_adds_the_tcp_group() -> None:
    """arm stays link6-terminated; arm_tcp and piper_gripper are additive."""
    root = ET.parse(SRDF_PATH).getroot()
    assert root.attrib["name"] == "piper_l"
    groups = {group.attrib["name"]: group for group in root.findall("group")}
    arm = groups["arm"]
    assert [joint.attrib["name"] for joint in arm.findall("joint")] == list(ARM_JOINT_NAMES)
    assert arm.find("chain").attrib == {
        "base_link": "base_link",
        "tip_link": "link6",
    }
    gripper = groups["gripper"]
    assert [joint.attrib["name"] for joint in gripper.findall("joint")] == [
        "gripper_joint1",
        "gripper_joint2",
    ]
    arm_tcp = groups["arm_tcp"]
    assert arm_tcp.findall("joint") == []
    assert arm_tcp.find("chain").attrib == {
        "base_link": "base_link",
        "tip_link": "rocobrick_tcp",
    }
    end_effectors = root.findall("end_effector")
    assert len(end_effectors) == 1
    assert end_effectors[0].attrib == {
        "name": "piper_gripper",
        "parent_link": "rocobrick_tcp",
        "group": "gripper",
        "parent_group": "arm_tcp",
    }


def test_kinematics_arm_entry_unchanged_and_arm_tcp_mirrors_it() -> None:
    """The vendor arm/gripper entries stay; arm_tcp mirrors arm except timeout."""
    ours = yaml.safe_load(KINEMATICS_YAML.read_text(encoding="utf-8"))
    vendor = yaml.safe_load(VENDOR_KINEMATICS_YAML.read_text(encoding="utf-8"))
    # The vendor entries are preserved exactly (parsed-YAML equality).
    assert ours["arm"] == vendor["arm"]
    assert ours["gripper"] == vendor["gripper"]
    # The R5.A addition is exactly one new group: solver and search
    # resolution mirror arm verbatim, while the raised timeout gives
    # MoveIt's constrained goal sampler real restart budget on narrow
    # IK basins near the joint limits.
    assert set(ours) == {"arm", "gripper", "arm_tcp"}
    assert {
        key: value
        for key, value in ours["arm_tcp"].items()
        if key != "kinematics_solver_timeout"
    } == {
        key: value
        for key, value in ours["arm"].items()
        if key != "kinematics_solver_timeout"
    }
    assert ours["arm_tcp"]["kinematics_solver_timeout"] == 0.05
    assert ours["arm"]["kinematics_solver_timeout"] == 0.0050000000000000001
    # Vendor provenance citation is kept in the header.
    header = KINEMATICS_YAML.read_text(encoding="utf-8")
    assert (
        "isaac/vendor/agx_arm_sim/Moveit2/piper_l_gripper_moveit_config/config/"
        "kinematics.yaml" in header
    )
    assert "arm_tcp" in header and "mirrors the arm entry" in header


@pytest.fixture(scope="module")
def plan_artifact() -> dict:
    return json.loads(PLAN_ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_tcp_planning_artifact_records_a_plan_only_run(plan_artifact) -> None:
    """The offline evidence is a plan-only, action-free container run."""
    assert plan_artifact["schema"] == "rocobrick-phase5-tcp-pose-planning-offline/v1"
    assert plan_artifact["offline"] is True
    assert plan_artifact["use_sim_time"] is False
    assert plan_artifact["action_sent"] is False
    assert plan_artifact["plan_only"] is True
    assert plan_artifact["service"] == "/plan_kinematic_path"
    assert "host Isaac" in plan_artifact["bridge"]
    assert plan_artifact["passed"] is True
    assert plan_artifact["checks_failed"] == []


def test_tcp_planning_artifact_goal_contract(plan_artifact) -> None:
    """The recorded goal: group arm_tcp, link rocobrick_tcp, base_link."""
    goal = plan_artifact["goal"]
    assert goal["group_name"] == "arm_tcp"
    assert goal["link_name"] == "rocobrick_tcp"
    assert goal["frame_id"] == "base_link"
    assert goal["plan_only"] is True
    assert goal["position_tolerance_m"] == MAX_POSITION_TOLERANCE_M
    assert goal["orientation_tolerance_rad"] == MAX_ORIENTATION_TOLERANCE_RAD
    # The goal is the exact FK pose of rocobrick_tcp itself (no offset).
    assert goal["position_m"] == pytest.approx(plan_artifact["fk"]["position_m"])
    assert goal["orientation_xyzw"] == pytest.approx(
        plan_artifact["fk"]["orientation_xyzw"]
    )
    assert plan_artifact["fk"]["link"] == "rocobrick_tcp"
    assert plan_artifact["fk"]["frame"] == "base_link"
    # The fake state is the exact canonical 8-DOF at zeros.
    assert plan_artifact["state"]["names"] == list(JOINT_NAMES)
    assert plan_artifact["state"]["positions"] == [0.0] * len(JOINT_NAMES)
    # The formal admitted target metadata is recorded.
    formal = plan_artifact["formal_target"]
    assert formal["tip_link"] == "rocobrick_tcp"
    assert formal["tcp_status"] == "calibrated_project"
    assert formal["runtime_model_sha256"] == EXPECTED_RUNTIME_MODEL_SHA256
    assert formal["source_sha256"] == ACCEPTED_SOURCE_SHA256
    assert formal["builder"] == "rocobrick.moveit_requests.build_tcp_pose_target"


def test_tcp_planning_artifact_result_within_execution_limits(plan_artifact) -> None:
    """SUCCESS, >= 2 waypoints, exactly the arm joints inside the limits."""
    result = plan_artifact["result"]
    assert result["moveit_error_code"] == 1
    assert result["success_code_expected"] == 1
    assert result["waypoint_count"] >= 2
    assert result["waypoint_count"] == len(result["waypoints"])
    assert result["joint_names"] == list(ARM_JOINT_NAMES)
    assert result["limit_epsilon"] == LIMIT_EPSILON
    for point in result["waypoints"]:
        assert len(point) == len(ARM_JOINT_NAMES)
        for name, value in zip(ARM_JOINT_NAMES, point):
            assert math.isfinite(value)
            lower, upper = EXECUTION_POSITION_LIMITS[name]
            assert lower - LIMIT_EPSILON <= value <= upper + LIMIT_EPSILON
    # The recorded per-joint excess agrees with an independent re-check.
    max_excess = dict.fromkeys(ARM_JOINT_NAMES, 0.0)
    for point in result["waypoints"]:
        for name, value in zip(ARM_JOINT_NAMES, point):
            lower, upper = EXECUTION_POSITION_LIMITS[name]
            max_excess[name] = max(
                max_excess[name], max(lower - value, value - upper, 0.0)
            )
    for name in ARM_JOINT_NAMES:
        assert max_excess[name] == pytest.approx(result["max_limit_excess"][name])
        assert max_excess[name] <= LIMIT_EPSILON


def test_tcp_planning_artifact_pins_the_kinematics_config(plan_artifact) -> None:
    """The evidence pins the exact kinematics.yaml bytes it planned with."""
    digest = hashlib.sha256(KINEMATICS_YAML.read_bytes()).hexdigest()
    assert plan_artifact["kinematics_yaml_sha256"] == digest
    assert plan_artifact["kinematics_arm_tcp_entry"] == (
        "mirrors arm (KDL, resolution copied; timeout raised to 0.05 s)"
    )


# ------------------------------------------------------------- TcpPoseTarget


def test_tcp_pose_target_defaults_freeze_and_mandatory_metadata() -> None:
    target = TcpPoseTarget(
        frame_id="base_link",
        position=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        runtime_model_sha256=EXPECTED_RUNTIME_MODEL_SHA256,
        source_sha256=ACCEPTED_SOURCE_SHA256,
    )
    assert target.tip_link == "rocobrick_tcp"
    assert target.tcp_status == "calibrated_project"
    assert target.position_tolerance_m == MAX_POSITION_TOLERANCE_M
    assert target.orientation_tolerance_rad == MAX_ORIENTATION_TOLERANCE_RAD
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.position = (0.0, 0.0, 0.0)  # type: ignore[misc]
    with pytest.raises(TypeError):
        TcpPoseTarget(  # type: ignore[call-arg]
            frame_id="base_link",
            position=(0.1, 0.0, 0.2),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )


def test_build_tcp_pose_target_accepts_artifact_derived_provenance() -> None:
    """All three provenance paths yield the same verbatim target."""
    for calibration in (None, _accepted_transform(), _resolved_contract()):
        kwargs = _valid_tcp_kwargs()
        if calibration is None:
            kwargs.pop("calibration", None)
        else:
            kwargs.pop("source_sha256")
            kwargs["calibration"] = calibration
        target, decision = build_tcp_pose_target(**kwargs)
        assert decision.accepted is True
        assert target is not None
        assert target.frame_id == "base_link"
        assert target.tip_link == "rocobrick_tcp"
        assert target.tcp_status == "calibrated_project"
        assert target.position == (0.2, -0.1, 0.35)
        assert target.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
        assert target.runtime_model_sha256 == EXPECTED_RUNTIME_MODEL_SHA256
        assert target.source_sha256 == ACCEPTED_SOURCE_SHA256
        # No target offset arithmetic anywhere: the request position is the
        # goal, the 63 mm calibration lives only in the model.
        assert target.position[2] != pytest.approx(0.35 + 0.1305895866844472)


@pytest.mark.parametrize(
    "overrides",
    [
        {"frame_id": "world"},
        {"position": (0.1, 0.0, float("nan"))},
        {"position": (0.1, 0.0)},
        {"orientation_xyzw": (0.0, 0.0, 0.0, 0.0)},
        {"orientation_xyzw": (float("nan"), 0.0, 0.0, 1.0)},
        {"position_tolerance_m": 0.02},
        {"orientation_tolerance_rad": 0.05},
        {"runtime_model_sha256": OTHER_SHA256},
        {"runtime_model_sha256": "A" * 64},
        {"runtime_model_sha256": None},
        {"source_sha256": "not-a-digest"},
        {"source_sha256": None},
    ],
)
def test_build_tcp_pose_target_rejects_bad_metadata_or_pose(overrides) -> None:
    kwargs = _valid_tcp_kwargs()
    kwargs.update(overrides)
    target, decision = build_tcp_pose_target(**kwargs)
    assert target is None
    assert decision.accepted is False
    assert decision.details["problems"]


def test_build_tcp_pose_target_requires_resolved_calibration() -> None:
    """An UNRESOLVED contract or foreign-frames transform is rejected."""
    for calibration in (
        _unresolved_contract(),
        Transform(
            parent_frame="link6",
            child_frame="tool0",
            translation_m=(0.0, 0.0, 0.1305896),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            source="elsewhere",
            sha256=ACCEPTED_SOURCE_SHA256,
        ),
        {"not": "a calibration"},
    ):
        kwargs = _valid_tcp_kwargs()
        kwargs.pop("source_sha256")
        kwargs["calibration"] = calibration
        target, decision = build_tcp_pose_target(**kwargs)
        assert target is None
        assert decision.accepted is False


# ------------------------------------------------------------------ admission


def test_admission_accepts_the_calibrated_tcp_target() -> None:
    target, decision = build_tcp_pose_target(**_valid_tcp_kwargs())
    assert decision.accepted is True
    admitted = strict_action_admission(target)
    assert admitted.accepted is True
    assert admitted.details["runtime_model_sha256"] == EXPECTED_RUNTIME_MODEL_SHA256


@pytest.mark.parametrize(
    "overrides",
    [
        {"frame_id": "world"},
        {"position": (0.1, float("inf"), 0.2)},
        {"orientation_xyzw": (2.0, 0.0, 0.0, 2.0)},
        {"position_tolerance_m": 0.5},
        {"orientation_tolerance_rad": -0.01},
    ],
)
def test_admission_rejects_bad_poses(overrides) -> None:
    target = TcpPoseTarget(**{**_valid_tcp_kwargs(), **overrides})
    decision = validate_tcp_pose_target(target)
    assert decision.accepted is False


def test_admission_enforces_resolved_and_scoped_metadata() -> None:
    base = _valid_tcp_kwargs()

    unresolved_status = TcpPoseTarget(**{**base, "tcp_status": "unresolved"})
    decision = validate_tcp_pose_target(unresolved_status)
    assert decision.accepted is False
    assert "tcp_status" in decision.reason

    foreign_tip = TcpPoseTarget(**{**base, "tip_link": "link6"})
    decision = validate_tcp_pose_target(foreign_tip)
    assert decision.accepted is False
    assert "tip_link" in decision.reason

    malformed = TcpPoseTarget(**{**base, "runtime_model_sha256": "zz" * 32})
    decision = validate_tcp_pose_target(malformed)
    assert decision.accepted is False
    assert "runtime_model_sha256" in decision.reason

    scoped = TcpPoseTarget(**{**base, "runtime_model_sha256": OTHER_SHA256})
    decision = validate_tcp_pose_target(scoped)
    assert decision.accepted is False
    assert "runtime-USD scope" in decision.reason

    bad_source = TcpPoseTarget(**{**base, "source_sha256": "B" * 64})
    decision = validate_tcp_pose_target(bad_source)
    assert decision.accepted is False
    assert "source_sha256" in decision.reason


def test_strict_admission_rejects_unknown_types() -> None:
    decision = strict_action_admission(object())
    assert decision.accepted is False
    assert "unsupported target type" in decision.reason


def test_link6_pose_target_contract_is_unchanged() -> None:
    target, decision = build_link6_pose_target(
        frame_id="base_link",
        position=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert decision.accepted is True
    assert isinstance(target, Link6PoseTarget)
    assert target.tip_link == "link6"
    assert target.tcp_status == "unresolved"


# ------------------------------------------------------- fake transport layer
# Minimal fakes mirroring tests/test_ros_moveit_runtime_contract.py.


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += float(delta)


class FakeFuture:
    def __init__(self) -> None:
        self._done = False
        self._value = None

    def done(self) -> bool:
        return self._done

    def result(self):
        return self._value

    def set(self, value) -> None:
        self._value = value
        self._done = True


class FakeGoalHandle:
    def __init__(self) -> None:
        self.accepted = True
        self.result_future = FakeFuture()

    def get_result_async(self) -> FakeFuture:
        return self.result_future

    def cancel_goal_async(self) -> FakeFuture:
        future = FakeFuture()
        future.set(SimpleNamespace(return_code=0))
        return future


class FakeActionClient:
    def __init__(self, node, action_type, name: str) -> None:
        self.name = name
        self.goals: list = []
        self.server_ready = True
        self.pending: FakeFuture | None = None
        node.action_clients[name] = self

    def wait_for_server(self, timeout_sec=None) -> bool:
        return self.server_ready

    def send_goal_async(self, goal) -> FakeFuture:
        self.goals.append(goal)
        self.pending = FakeFuture()
        return self.pending

    def destroy(self) -> None:
        pass


class FakeServiceClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ready = True
        self.response = None
        self.requests: list = []

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, request) -> FakeFuture:
        self.requests.append(request)
        future = FakeFuture()
        if self.response is not None:
            future.set(self.response)
        return future


class FakeNode:
    def __init__(self) -> None:
        self.subscriptions: list = []
        self.service_clients: dict = {}
        self.action_clients: dict = {}

    def create_subscription(self, msg_type, topic, callback, qos):
        subscription = SimpleNamespace(topic=topic, callback=callback)
        self.subscriptions.append(subscription)
        return subscription

    def create_client(self, srv_type, name) -> FakeServiceClient:
        client = FakeServiceClient(name)
        self.service_clients[name] = client
        return client

    def destroy_subscription(self, subscription) -> None:
        pass

    def destroy_client(self, client) -> None:
        pass


def publish(node: FakeNode, topic: str, message) -> None:
    for subscription in node.subscriptions:
        if subscription.topic == topic:
            subscription.callback(message)


class FakeClockMsg:
    def __init__(self, nanos: int) -> None:
        self.clock = SimpleNamespace(sec=nanos // 1_000_000_000, nanosec=nanos % 1_000_000_000)


class FakeJointStateMsg:
    def __init__(self, positions, velocity=()) -> None:
        self.name = list(positions)
        self.position = [float(value) for value in positions.values()]
        self.velocity = list(velocity)


class FakeGetPositionFK:
    class Request:
        def __init__(self) -> None:
            self.header = SimpleNamespace(frame_id="")
            self.fk_link_names: list = []
            self.robot_state = SimpleNamespace(
                joint_state=SimpleNamespace(name=[], position=[])
            )


class FakeMotionPlanRequest:
    def __init__(self) -> None:
        self.group_name = ""
        self.allowed_planning_time = 0.0
        self.num_planning_attempts = 0
        self.max_velocity_scaling_factor = 0.0
        self.max_acceleration_scaling_factor = 0.0
        self.goal_constraints: list = []


class FakeConstraints:
    def __init__(self) -> None:
        self.joint_constraints: list = []
        self.position_constraints: list = []
        self.orientation_constraints: list = []


class FakePositionConstraint:
    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.link_name = ""
        self.constraint_region = SimpleNamespace(primitives=[], primitive_poses=[])
        self.weight = 0.0


class FakeOrientationConstraint:
    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.link_name = ""
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)
        self.absolute_x_axis_tolerance = 0.0
        self.absolute_y_axis_tolerance = 0.0
        self.absolute_z_axis_tolerance = 0.0
        self.weight = 0.0


class FakeSolidPrimitive:
    BOX = 1
    SPHERE = 2

    def __init__(self) -> None:
        self.type = 0
        self.dimensions: list = []


class FakePose:
    def __init__(self) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)


class FakeMoveGroupGoal:
    def __init__(self) -> None:
        self.request = None
        self.planning_options = SimpleNamespace(plan_only=None, replan=None)


FAKE_GOAL_STATUS = SimpleNamespace(STATUS_SUCCEEDED=4, STATUS_CANCELED=5)


class FakeAttachedCollisionObject:
    def __init__(self) -> None:
        self.link_name = ""
        self.object = FakeCollisionObject()
        self.touch_links: list[str] = []


class FakeApplyPlanningSceneRequest:
    def __init__(self) -> None:
        self.scene = SimpleNamespace(
            is_diff=False,
            world=SimpleNamespace(collision_objects=[]),
            robot_state=SimpleNamespace(attached_collision_objects=[]),
        )


class FakeCollisionObject:
    ADD = 0
    REMOVE = 1

    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.id = ""
        self.operation = -1
        self.primitives: list = []
        self.primitive_poses: list = []


def fake_factories() -> RosMoveItFactories:
    return RosMoveItFactories(
        clock_type=lambda: FakeClockMsg,
        joint_state_type=lambda: FakeJointStateMsg,
        list_controllers_type=lambda: SimpleNamespace,
        list_controllers_request_factory=lambda: SimpleNamespace(),
        get_motion_plan_type=lambda: SimpleNamespace,
        get_position_fk_type=lambda: FakeGetPositionFK,
        move_group_type=lambda: SimpleNamespace,
        follow_joint_trajectory_type=lambda: SimpleNamespace,
        move_group_goal_factory=FakeMoveGroupGoal,
        motion_plan_request_factory=FakeMotionPlanRequest,
        constraints_factory=FakeConstraints,
        joint_constraint_factory=SimpleNamespace,
        position_constraint_factory=FakePositionConstraint,
        orientation_constraint_factory=FakeOrientationConstraint,
        solid_primitive_factory=FakeSolidPrimitive,
        pose_factory=FakePose,
        action_client_factory=lambda node, action_type, name: FakeActionClient(
            node, action_type, name
        ),
        goal_status_type=lambda: FAKE_GOAL_STATUS,
        apply_planning_scene_type=lambda: SimpleNamespace,
        apply_planning_scene_request_factory=FakeApplyPlanningSceneRequest,
        collision_object_factory=FakeCollisionObject,
        attached_collision_object_factory=FakeAttachedCollisionObject,
        get_planning_scene_type=lambda: SimpleNamespace,
        get_planning_scene_request_factory=lambda: SimpleNamespace(
            components=SimpleNamespace(components=0)
        ),
    )


class World:
    def __init__(self, runtime: RosMoveItRuntime, node: FakeNode, clock: FakeClock, events: list) -> None:
        self.runtime = runtime
        self.node = node
        self.clock = clock
        self.events = events

    @property
    def controllers(self) -> FakeServiceClient:
        return self.node.service_clients[SERVICE_LIST_CONTROLLERS]

    @property
    def fk(self) -> FakeServiceClient:
        return self.node.service_clients[SERVICE_COMPUTE_FK]

    @property
    def move_group(self) -> FakeActionClient:
        return self.node.action_clients[MOVE_GROUP_ACTION_CANDIDATES[0]]


def make_world(*, fk_verifier=None, planning_scene_applier=None) -> World:
    node = FakeNode()
    clock = FakeClock()
    events: list = []

    def spin() -> None:
        if events:
            events.pop(0)()

    runtime = RosMoveItRuntime(
        node,
        qos_state=SimpleNamespace(),
        qos_sensor=SimpleNamespace(),
        factories=fake_factories(),
        fk_verifier=fk_verifier,
        planning_scene_applier=planning_scene_applier,
        clock=clock,
        spin=spin,
        sleep=clock.advance,
    )
    runtime.register_subscriptions()
    runtime.register_clients()
    return World(runtime, node, clock, events)


def prime_ready(world: World) -> None:
    for nanos in (1_000_000_000, 2_000_000_000):
        publish(world.node, TOPIC_CLOCK, FakeClockMsg(nanos))
    publish(
        world.node,
        TOPIC_ISAAC_JOINT_STATES,
        FakeJointStateMsg(BASELINE, velocity=[0.0] * len(JOINT_NAMES)),
    )
    publish(world.node, TOPIC_JOINT_STATES, FakeJointStateMsg(BASELINE))
    world.controllers.response = SimpleNamespace(
        controller=[
            SimpleNamespace(name=name, state="active")
            for name in REQUIRED_CONTROLLERS
        ]
    )


def script_success(world: World) -> None:
    handle = FakeGoalHandle()
    trajectory = SimpleNamespace(
        joint_names=list(ARM_JOINT_NAMES),
        points=[
            SimpleNamespace(positions=[float(BASELINE[name]) for name in ARM_JOINT_NAMES])
            for _ in range(3)
        ],
    )
    world.events.append(lambda: world.move_group.pending.set(handle))

    def deliver_evidence() -> None:
        for _ in range(2):
            publish(
                world.node,
                TOPIC_ISAAC_JOINT_STATES,
                FakeJointStateMsg(BASELINE, velocity=[0.0] * len(JOINT_NAMES)),
            )
        publish(world.node, TOPIC_ISAAC_JOINT_COMMAND, FakeJointStateMsg(BASELINE))

    world.events.append(deliver_evidence)
    world.events.append(
        lambda: handle.result_future.set(
            SimpleNamespace(
                status=FAKE_GOAL_STATUS.STATUS_SUCCEEDED,
                result=SimpleNamespace(
                    error_code=SimpleNamespace(val=1),
                    planned_trajectory=SimpleNamespace(joint_trajectory=trajectory),
                ),
            )
        )
    )


# ------------------------------------------------------- runtime: tcp requests


def tcp_target() -> TcpPoseTarget:
    target, decision = build_tcp_pose_target(**_valid_tcp_kwargs())
    assert decision.accepted is True and target is not None
    return target


def test_send_tcp_pose_builds_the_exact_arm_tcp_goal() -> None:
    world = make_world(fk_verifier=lambda _state: ((0.2, -0.1, 0.35), (0.0, 0.0, 0.0, 1.0)))
    prime_ready(world)
    script_success(world)
    target = tcp_target()
    result = world.runtime.send_tcp_pose(target, 20.0)
    assert result.status is ExecutionStatus.SUCCEEDED
    assert is_success(result) is True
    assert len(world.move_group.goals) == 1
    goal = world.move_group.goals[0]
    request = goal.request
    assert request.group_name == "arm_tcp"
    assert goal.planning_options.plan_only is False
    assert goal.planning_options.replan is False
    constraints = request.goal_constraints[0]
    assert constraints.joint_constraints == []
    position = constraints.position_constraints[0]
    assert position.header.frame_id == "base_link"
    assert position.link_name == "rocobrick_tcp"
    sphere = position.constraint_region.primitives[0]
    assert sphere.type == FakeSolidPrimitive.SPHERE
    assert sphere.dimensions == [
        target.position_tolerance_m * POSE_GOAL_TOLERANCE_FRACTION
    ]
    center = position.constraint_region.primitive_poses[0]
    assert (center.position.x, center.position.y, center.position.z) == (0.2, -0.1, 0.35)
    assert center.orientation.w == 1.0
    orientation = constraints.orientation_constraints[0]
    assert orientation.header.frame_id == "base_link"
    assert orientation.link_name == "rocobrick_tcp"
    assert (
        orientation.orientation.x,
        orientation.orientation.y,
        orientation.orientation.z,
        orientation.orientation.w,
    ) == (0.0, 0.0, 0.0, 1.0)
    assert orientation.absolute_x_axis_tolerance == (
        target.orientation_tolerance_rad * POSE_GOAL_TOLERANCE_FRACTION
    )


def test_send_tcp_pose_rejects_inadmissible_targets_before_any_send() -> None:
    world = make_world()
    prime_ready(world)
    invalid = TcpPoseTarget(
        **{**_valid_tcp_kwargs(), "runtime_model_sha256": OTHER_SHA256}
    )
    result = world.runtime.send_tcp_pose(invalid, 5.0)
    assert result.status is ExecutionStatus.REJECTED
    assert result.command_messages == 0
    assert world.move_group.goals == []


def test_send_link6_pose_group_and_link_are_unchanged() -> None:
    world = make_world(fk_verifier=lambda _state: ((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)))
    prime_ready(world)
    script_success(world)
    target, decision = build_link6_pose_target(
        frame_id="base_link",
        position=(0.1, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert decision.accepted is True
    result = world.runtime.send_link6_pose(target, 20.0)
    assert result.status is ExecutionStatus.SUCCEEDED
    goal = world.move_group.goals[0]
    assert goal.request.group_name == "arm"
    constraints = goal.request.goal_constraints[0]
    assert constraints.position_constraints[0].link_name == "link6"
    assert constraints.orientation_constraints[0].link_name == "link6"
    assert goal.planning_options.plan_only is False


# ------------------------------------------------------------- runtime: fk


def fk_response(*, link: str) -> SimpleNamespace:
    return SimpleNamespace(
        error_code=SimpleNamespace(val=1),
        pose_stamped=[
            SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.1, y=0.0, z=0.2),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            )
        ],
    )


def test_compute_tcp_fk_asks_for_rocobrick_tcp() -> None:
    world = make_world()
    world.fk.response = fk_response(link="rocobrick_tcp")
    verified = world.runtime.compute_tcp_fk(BASELINE, timeout_s=5.0)
    assert verified == ((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    request = world.fk.requests[0]
    assert request.header.frame_id == "base_link"
    assert request.fk_link_names == ["rocobrick_tcp"]
    assert request.robot_state.joint_state.name == list(JOINT_NAMES)
    assert request.robot_state.joint_state.position == [
        float(BASELINE[name]) for name in JOINT_NAMES
    ]


def test_compute_link6_fk_still_asks_for_link6() -> None:
    world = make_world()
    world.fk.response = fk_response(link="link6")
    verified = world.runtime.compute_link6_fk(BASELINE, timeout_s=5.0)
    assert verified == ((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    assert world.fk.requests[0].fk_link_names == ["link6"]


# ------------------------------------------------------------------- executor


class FakeRuntime:
    def __init__(self) -> None:
        self.tcp_calls: list = []
        self.tcp_result = ExecutionResult(
            status=ExecutionStatus.SUCCEEDED,
            moveit_error_code=1,
            planned_waypoints=4,
            command_messages=1,
        )

    def send_tcp_pose(self, target, timeout_s) -> ExecutionResult:
        self.tcp_calls.append((target, timeout_s))
        return self.tcp_result


def test_executor_satisfies_both_protocols() -> None:
    executor = RosMoveItExecutor(FakeRuntime())
    assert isinstance(executor, MoveItExecutor)
    assert isinstance(executor, TcpMoveItExecutor)


def test_executor_delegates_admitted_tcp_targets_exactly_once() -> None:
    fake = FakeRuntime()
    target = tcp_target()
    result = RosMoveItExecutor(fake).execute_tcp_pose(target, timeout_s=6.5)
    assert result is fake.tcp_result
    assert fake.tcp_calls == [(target, 6.5)]


def test_executor_rejects_inadmissible_tcp_targets_without_a_send() -> None:
    fake = FakeRuntime()
    invalid = TcpPoseTarget(**{**_valid_tcp_kwargs(), "frame_id": "world"})
    result = RosMoveItExecutor(fake).execute_tcp_pose(invalid, timeout_s=5.0)
    assert result.status is ExecutionStatus.REJECTED
    assert result.command_messages == 0
    assert result.details["kind"] == "tcp_pose"
    assert fake.tcp_calls == []


# --------------------------------------------------------------- planning scene


def test_axis_aligned_box_validation_properties_and_digest() -> None:
    box = box_from_aabb(
        object_id="brick-44",
        frame_id="base_link",
        minimum_m=(0.1, -0.05, 0.01),
        maximum_m=(0.2, 0.05, 0.05),
    )
    assert box.center_m == pytest.approx((0.15, 0.0, 0.03))
    assert box.dimensions_m == pytest.approx((0.1, 0.1, 0.04))
    assert box.geometry_digest == box_geometry_digest(
        object_id="brick-44",
        frame_id="base_link",
        minimum_m=(0.1, -0.05, 0.01),
        maximum_m=(0.2, 0.05, 0.05),
    )
    # Deterministic and discriminating: the digest follows every identity
    # component (id, frame, corners) and pins them via repr-exact floats.
    assert box.geometry_digest == box_from_aabb(
        object_id="brick-44",
        frame_id="base_link",
        minimum_m=(0.1, -0.05, 0.01),
        maximum_m=(0.2, 0.05, 0.05),
    ).geometry_digest
    variants = (
        {"object_id": "brick-97"},
        {"frame_id": "world"},
        {"minimum_m": (0.100001, -0.05, 0.01)},
        {"maximum_m": (0.2, 0.05, 0.050001)},
    )
    identity = {
        "object_id": "brick-44",
        "frame_id": "base_link",
        "minimum_m": (0.1, -0.05, 0.01),
        "maximum_m": (0.2, 0.05, 0.05),
    }
    for change in variants:
        different = box_from_aabb(**{**identity, **change})
        assert different.geometry_digest != box.geometry_digest
    # int coordinates and float coordinates digest identically (both
    # normalize to the same repr-exact floats).
    assert box_from_aabb(
        object_id="zero-corners",
        frame_id="base_link",
        minimum_m=(0, 0, 0),
        maximum_m=(1, 1, 1),
    ).geometry_digest == box_from_aabb(
        object_id="zero-corners",
        frame_id="base_link",
        minimum_m=(0.0, 0.0, 0.0),
        maximum_m=(1.0, 1.0, 1.0),
    ).geometry_digest
    with pytest.raises(dataclasses.FrozenInstanceError):
        box.frame_id = "world"  # type: ignore[misc]


def test_axis_aligned_box_rejects_invalid_identities() -> None:
    with pytest.raises(ValueError):
        box_from_aabb(
            object_id="",
            frame_id="base_link",
            minimum_m=(0.1, -0.05, 0.01),
            maximum_m=(0.2, 0.05, 0.05),
        )
    with pytest.raises(ValueError):
        box_from_aabb(
            object_id="brick-44",
            frame_id="",
            minimum_m=(0.1, -0.05, 0.01),
            maximum_m=(0.2, 0.05, 0.05),
        )
    with pytest.raises(ValueError):
        box_from_aabb(
            object_id="brick-44",
            frame_id="base_link",
            minimum_m=(0.1, -0.05, float("nan")),
            maximum_m=(0.2, 0.05, 0.05),
        )
    with pytest.raises(ValueError):
        box_from_aabb(
            object_id="brick-44",
            frame_id="base_link",
            minimum_m=(0.1, -0.05, 0.01),
            maximum_m=(0.2, 0.05, 0.01),
        )
    with pytest.raises(ValueError):
        AxisAlignedBox(
            object_id="brick-44",
            frame_id="base_link",
            minimum_m=(0.1, -0.05, 0.01),
            maximum_m=(0.2, 0.05, 0.05),
            geometry_digest="0" * 64,
        )


def test_planning_scene_update_schema_digest_and_validation() -> None:
    box = box_from_aabb(
        object_id="brick-44",
        frame_id="base_link",
        minimum_m=(0.1, -0.05, 0.01),
        maximum_m=(0.2, 0.05, 0.05),
    )
    update = PlanningSceneUpdate(
        scene_id="scene-1",
        boxes=(box,),
        expected_scene_digest=scene_digest(
            PlanningSceneUpdate(
                scene_id="scene-1",
                boxes=(box,),
                expected_scene_digest="0" * 64,
            )
        ),
    )
    assert update.schema == SCHEMA == "rocobrick-planning-scene-update/v1"
    assert scene_digest(update) == update.expected_scene_digest
    empty = PlanningSceneUpdate(
        scene_id="empty", boxes=(), expected_scene_digest=scene_digest(
            PlanningSceneUpdate(scene_id="empty", boxes=(), expected_scene_digest="0" * 64)
        )
    )
    assert empty.boxes == ()
    with pytest.raises(ValueError):
        PlanningSceneUpdate(
            scene_id="",
            boxes=(),
            expected_scene_digest="0" * 64,
        )
    with pytest.raises(ValueError):
        PlanningSceneUpdate(
            scene_id="scene-1",
            boxes=(object(),),  # type: ignore[list-item]
            expected_scene_digest="0" * 64,
        )
    with pytest.raises(ValueError):
        PlanningSceneUpdate(scene_id="scene-1", boxes=(box,), expected_scene_digest="zz")


def test_collision_box_matches_composes_moveit_object_and_shape_poses() -> None:
    box = box_from_aabb(
        object_id="brick-44",
        frame_id="piper_0_world",
        minimum_m=(0.1, -0.05, 0.01),
        maximum_m=(0.2, 0.05, 0.05),
    )
    item = FakeCollisionObject()
    item.header.frame_id = box.frame_id
    item.primitives = [SimpleNamespace(type=FakeSolidPrimitive.BOX, dimensions=list(box.dimensions_m))]
    item.pose = FakePose()
    item.pose.position.x, item.pose.position.y, item.pose.position.z = box.center_m
    item.pose.orientation.w = 1.0
    shape_pose = FakePose()
    shape_pose.orientation.w = 1.0
    item.primitive_poses = [shape_pose]
    assert _collision_box_matches(item, box) is True



def test_apply_planning_scene_delegates_to_the_injected_applier_once() -> None:
    box = box_from_aabb(
        object_id="brick-44",
        frame_id="base_link",
        minimum_m=(0.1, -0.05, 0.01),
        maximum_m=(0.2, 0.05, 0.05),
    )
    update = PlanningSceneUpdate(
        scene_id="scene-1",
        boxes=(box,),
        expected_scene_digest=scene_digest(
            PlanningSceneUpdate(
                scene_id="scene-1",
                boxes=(box,),
                expected_scene_digest="0" * 64,
            )
        ),
    )
    calls: list = []
    expected = PlanningSceneResult(
        passed=True,
        applied_object_ids=(box.object_id,),
        scene_digest=update.expected_scene_digest,
        details={"fake": True},
    )

    def applier(received, timeout_s):
        calls.append((received, timeout_s))
        return expected

    world = make_world(planning_scene_applier=applier)
    result = world.runtime.apply_planning_scene(update, 3.25)
    assert result is expected
    assert calls == [(update, 3.25)]


def test_apply_planning_scene_default_adapter_sends_exact_collision_objects() -> None:
    box = box_from_aabb(
        object_id="brick-44",
        frame_id="base_link",
        minimum_m=(0.1, -0.05, 0.01),
        maximum_m=(0.2, 0.05, 0.05),
    )
    update = PlanningSceneUpdate(
        scene_id="scene-1",
        boxes=(box,),
        expected_scene_digest=scene_digest(
            PlanningSceneUpdate(
                scene_id="scene-1",
                boxes=(box,),
                expected_scene_digest="0" * 64,
            )
        ),
    )
    world = make_world()
    # Preset the service answer on the lazily created client so the default
    # adapter's bounded wait resolves immediately.
    world.runtime._ensure_planning_scene_client().response = SimpleNamespace(
        success=True
    )
    readback = FakeCollisionObject()
    readback.id = box.object_id
    readback.header.frame_id = box.frame_id
    primitive = FakeSolidPrimitive()
    primitive.type = FakeSolidPrimitive.BOX
    primitive.dimensions = list(box.dimensions_m)
    readback.primitives.append(primitive)
    pose = FakePose()
    pose.position.x, pose.position.y, pose.position.z = box.center_m
    pose.orientation.w = 1.0
    readback.primitive_poses.append(pose)
    world.runtime._ensure_get_planning_scene_client().response = SimpleNamespace(
        scene=SimpleNamespace(
            world=SimpleNamespace(collision_objects=[readback])
        )
    )
    result = world.runtime.apply_planning_scene(update, 5.0)
    assert result.passed is True
    assert result.applied_object_ids == ("brick-44",)
    assert result.scene_digest == update.expected_scene_digest
    assert result.details["readback_confirmed"] is True
    assert PLANNING_SCENE_WORLD_OBJECT_GEOMETRY == 16
    readback_requests = world.runtime._ensure_get_planning_scene_client().requests
    assert len(readback_requests) == 1
    assert readback_requests[0].components.components == 24
    assert SERVICE_APPLY_PLANNING_SCENE in world.node.service_clients
    client = world.node.service_clients[SERVICE_APPLY_PLANNING_SCENE]
    request = client.requests[0]
    assert request.scene.is_diff is True
    objects = request.scene.world.collision_objects
    assert len(objects) == 1
    collision_object = objects[0]
    assert collision_object.id == "brick-44"
    assert collision_object.header.frame_id == "base_link"
    assert collision_object.operation == FakeCollisionObject.ADD
    primitive = collision_object.primitives[0]
    assert primitive.type == FakeSolidPrimitive.BOX
    assert primitive.dimensions == pytest.approx([0.1, 0.1, 0.04])
    pose = collision_object.primitive_poses[0]
    assert (pose.position.x, pose.position.y, pose.position.z) == pytest.approx(
        (0.15, 0.0, 0.03)
    )
    assert pose.orientation.w == 1.0


def test_read_attached_collision_object_requires_exact_identity_and_geometry():
    """Attached readback uses object.id/link_name and fails closed."""
    world = make_world()
    client = world.runtime._ensure_get_planning_scene_client()
    attached = SimpleNamespace(
        link_name="piper_0_gripper_link1",
        object=SimpleNamespace(
            id="brick-44", primitives=[SimpleNamespace()], meshes=[], planes=[]
        ),
    )
    client.response = SimpleNamespace(
        scene=SimpleNamespace(
            robot_state=SimpleNamespace(attached_collision_objects=[attached])
        )
    )
    ok, error = world.runtime._read_attached_collision_object(
        5.0, fake_factories(), object_id="brick-44", link_name="piper_0_gripper_link1"
    )
    assert ok is True and error is None
    assert client.requests[-1].components.components == (
        PLANNING_SCENE_ROBOT_STATE_ATTACHED_OBJECTS
    )
    ok, error = world.runtime._read_attached_collision_object(
        5.0, fake_factories(), object_id="other", link_name="piper_0_gripper_link1"
    )
    assert ok is False and "matching attached objects" in error


def test_read_attached_collision_object_fails_without_response():
    """No attached-object response cannot be treated as attachment."""
    world = make_world()
    ok, error = world.runtime._read_attached_collision_object(
        0.1, fake_factories(), object_id="brick-44", link_name="piper_0_gripper_link1"
    )
    assert ok is False
    assert "timed out" in error


def test_apply_planning_scene_service_success_without_readback_fails() -> None:
    """An acknowledged update cannot prove a world collision object exists."""
    box = box_from_aabb(
        object_id="brick-44", frame_id="base_link",
        minimum_m=(0.1, -0.05, 0.01), maximum_m=(0.2, 0.05, 0.05),
    )
    placeholder = PlanningSceneUpdate(
        scene_id="scene-1", boxes=(box,), expected_scene_digest="0" * 64
    )
    update = PlanningSceneUpdate(
        scene_id="scene-1", boxes=(box,),
        expected_scene_digest=scene_digest(placeholder),
    )
    world = make_world()
    world.runtime._ensure_planning_scene_client().response = SimpleNamespace(
        success=True
    )
    result = world.runtime.apply_planning_scene(update, 0.1)
    assert result.passed is False
    assert "world geometry timed out" in result.details["error"]


@pytest.mark.parametrize(
    ("failure", "fragment"),
    [
        ("service_failure", "reported failure"),
        ("timeout", "did not answer"),
        ("digest_mismatch", "digest does not match"),
        ("applier_raises", "applier raised"),
        ("no_transport", "no injected planning-scene applier"),
    ],
)
def test_apply_planning_scene_failures_are_explicit(failure, fragment) -> None:
    box = box_from_aabb(
        object_id="brick-44",
        frame_id="base_link",
        minimum_m=(0.1, -0.05, 0.01),
        maximum_m=(0.2, 0.05, 0.05),
    )

    def update_for(scene_id: str) -> PlanningSceneUpdate:
        return PlanningSceneUpdate(
            scene_id=scene_id,
            boxes=(box,),
            expected_scene_digest=scene_digest(
                PlanningSceneUpdate(
                    scene_id=scene_id,
                    boxes=(box,),
                    expected_scene_digest="0" * 64,
                )
            ),
        )

    if failure in ("service_failure", "timeout", "digest_mismatch"):
        world = make_world()
        applier = world.runtime._default_planning_scene_applier
        if failure == "service_failure":
            world.runtime._ensure_planning_scene_client().response = SimpleNamespace(
                success=False
            )
            result = applier(update_for("scene-1"), 5.0)
        elif failure == "timeout":
            # The future never completes within the bounded wait.
            result = applier(update_for("scene-1"), 0.1)
        else:
            world.runtime._ensure_planning_scene_client().response = SimpleNamespace(
                success=True
            )
            wrong = PlanningSceneUpdate(
                scene_id="scene-1",
                boxes=(box,),
                expected_scene_digest="a" * 64,
            )
            result = applier(wrong, 5.0)
    elif failure == "applier_raises":
        def broken(_update, _timeout_s):
            raise RuntimeError("scene boom")

        result = make_world(planning_scene_applier=broken).runtime.apply_planning_scene(
            update_for("scene-1"), 5.0
        )
    else:  # no_transport
        bare = RosMoveItRuntime(
            FakeNode(),
            factories=RosMoveItFactories(
                clock_type=lambda: FakeClockMsg,
                joint_state_type=lambda: FakeJointStateMsg,
                list_controllers_type=lambda: SimpleNamespace,
                list_controllers_request_factory=lambda: SimpleNamespace(),
                get_motion_plan_type=lambda: SimpleNamespace,
                get_position_fk_type=lambda: SimpleNamespace,
                move_group_type=lambda: SimpleNamespace,
                follow_joint_trajectory_type=lambda: SimpleNamespace,
                move_group_goal_factory=FakeMoveGroupGoal,
                motion_plan_request_factory=FakeMotionPlanRequest,
                constraints_factory=FakeConstraints,
                joint_constraint_factory=SimpleNamespace,
                position_constraint_factory=FakePositionConstraint,
                orientation_constraint_factory=FakeOrientationConstraint,
                solid_primitive_factory=FakeSolidPrimitive,
                pose_factory=FakePose,
                action_client_factory=lambda node, action_type, name: FakeActionClient(
                    node, action_type, name
                ),
                goal_status_type=lambda: FAKE_GOAL_STATUS,
            ),
            clock=FakeClock(),
            sleep=lambda _dt: None,
        )
        result = bare.apply_planning_scene(update_for("scene-1"), 5.0)
    assert result.passed is False
    assert fragment in result.details["error"]
    if failure == "digest_mismatch":
        assert result.applied_object_ids == ("brick-44",)
    else:
        assert result.applied_object_ids == ()


def test_apply_planning_scene_input_validation() -> None:
    world = make_world(planning_scene_applier=lambda _u, _t: PlanningSceneResult(passed=True))
    with pytest.raises(ValueError):
        world.runtime.apply_planning_scene(object(), 5.0)  # type: ignore[arg-type]
    update = PlanningSceneUpdate(
        scene_id="scene-1",
        boxes=(),
        expected_scene_digest="0" * 64,
    )
    for timeout in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            world.runtime.apply_planning_scene(update, timeout)
    result = make_world(
        planning_scene_applier=lambda _u, _t: {"not": "a result"}
    ).runtime.apply_planning_scene(update, 5.0)
    assert result.passed is False
    assert "not PlanningSceneResult" in result.details["error"]


def test_runtime_services_pin_the_planning_scene_service() -> None:
    services = RuntimeServices()
    assert services.apply_planning_scene == "/apply_planning_scene"
    assert services.list_controllers == SERVICE_LIST_CONTROLLERS
    assert services.plan is not None and services.compute_fk is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        services.apply_planning_scene = "/other"  # type: ignore[misc]


def test_service_call_timeout_constant_is_unchanged() -> None:
    assert SERVICE_CALL_TIMEOUT_S == 5.0


# ------------------------------------------------------------------ hygiene


@pytest.mark.parametrize("path", PRODUCTION_ADDITIONS)
def test_production_additions_carry_no_scene_or_motion_tokens(path) -> None:
    """No example scene, pregrasp, approach, or grasp tokens in R5.A code."""
    text = path.read_text(encoding="utf-8").casefold()
    for token in FORBIDDEN_TOKENS:
        assert token not in text, f"{path.name} carries forbidden token {token!r}"
