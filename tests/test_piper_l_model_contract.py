"""Offline contract tests for the P0.4 minimal Piper_L model audit.

These tests never touch the network, git, or Isaac Sim. They validate that
``artifacts/model-audit/piper_l_model_contract.json`` records the pinned
official model data (six arm joints, gripper mimic mapping, planning groups,
TCP facts, license caveat) and that composed-USD values appear only with
recorded extraction evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from rocobrick.gripper import fingers_to_logical_gripper, logical_gripper_to_fingers

ROOT = Path(__file__).parents[1]
ARTIFACT_PATH = ROOT / "artifacts/model-audit/piper_l_model_contract.json"
MANIFEST_PATH = ROOT / "third_party/manifest.lock.yaml"
PIPER_L_URDF = (
    ROOT
    / "ros_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf"
    / "piper_l/urdf/piper_l_description.urdf"
)
PIPER_L_GRIPPER_XACRO = (
    ROOT
    / "ros_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf"
    / "piper_l/urdf/piper_l_with_gripper_description.xacro"
)
COMPOSED_URDF = ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf"
EXECUTION_URDF = (
    ROOT / "isaac/vendor/agx_arm_sim/agx_arm_description/urdf/piper_l_gripper_d435.urdf"
)
EXECUTION_USD = (
    ROOT / "isaac/vendor/agx_arm_sim/agx_arm_description/urdf/USD"
    / "piper_l_gripper_d435/piper_l_gripper_d435.usd"
)
SRDF_PATH = (
    ROOT
    / "isaac/vendor/agx_arm_sim/Moveit2/piper_l_gripper_moveit_config/config"
    / "piper_l.srdf"
)
MOVEIT_DIR = (
    ROOT / "isaac/vendor/agx_arm_sim/Moveit2/piper_l_gripper_moveit_config/config"
)

ARM_JOINT_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["gripper_joint1", "gripper_joint2"]
EXPECTED_DOF_ORDER = ARM_JOINT_ORDER + GRIPPER_JOINTS
AGX_ARM_URDF_PIN = "f6642ce0d7872c686f29c99e9e10cd23d1d49313"
AGX_ARM_SIM_GITLINK = "1ee5659d02b33b9379fd647a2e5647800d67f0f4"
PINNED_RUNTIME_HASHES = {
    "urdf": "8a5e4a77c53947e234f7f8a15db14e30ca85eb2c9f7ae3dc1784f07057626b0d",
    "usd_top": "a781d52f9d55291fca9c8f2695abf9e311640d996314d272f47639d5323e2375",
    "piper_l_gripper_d435_base.usd": "14b24829e7ec1179fe437f89222a9a7f586369e56b1ed010772b5887c3663b50",
    "piper_l_gripper_d435_physics.usd": "7cd338a94a00d50d3f799ca39f66d1fad515b280dcca5b1b893dbe4953e35888",
    "piper_l_gripper_d435_robot.usd": "41660f0300863d9bd8a675937afec9a62ecd174fc789c4c7a5d3553741b48f00",
    "piper_l_gripper_d435_sensor.usd": "5fbd085f92be9940d861a16a5b31ccacbb031f196695c07e6764b35a8c3011fb",
}

# Planning source: the pinned agx_arm_urdf submodule used by official MoveIt.
PLANNING_LIMITS = {
    "joint1": (-2.6179938, 2.6179938),
    "joint2": (0.0, 3.1415926),
    "joint3": (-2.9670597, 0.0),
    "joint4": (-2.2165681, 2.2165681),
    "joint5": (-1.5620696, 1.5620696),
    "joint6": (-2.0943951, 2.0943951),
}
EXECUTION_LIMITS = {
    **PLANNING_LIMITS,
    "joint4": (-1.7453292, 1.7453292),
    "joint5": (-1.2217304, 1.2217304),
}


@pytest.fixture
def artifact() -> dict:
    """Return the model contract artifact parsed with json.loads."""
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def manifest() -> dict:
    """Return the P0.1 third-party lock parsed with yaml.safe_load."""
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _urdf_joints(path: Path) -> dict[str, dict]:
    root = ET.parse(path).getroot()
    joints: dict[str, dict] = {}
    for element in root.findall("joint"):
        axis = element.find("axis")
        limit = element.find("limit")
        mimic = element.find("mimic")
        entry = {
            "type": element.attrib["type"],
            "axis": (
                [float(v) for v in axis.attrib["xyz"].split()]
                if axis is not None
                else None
            ),
            "limit": {
                key: float(limit.attrib[key])
                for key in ("lower", "upper", "effort", "velocity")
                if limit is not None and key in limit.attrib
            },
            "mimic": dict(mimic.attrib) if mimic is not None else None,
        }
        joints[element.attrib["name"]] = entry
    return joints


def test_artifact_exists_with_schema_phase_and_model(artifact: dict) -> None:
    """The artifact must declare the P0.4 Piper_L model contract schema."""
    assert artifact["schema"] == "rocobrick-piper-l-model-contract/v1"
    assert artifact["phase"] == "P0.4"
    assert artifact["model"] == "agilex_piper_l"
    assert artifact["offline"] is True


def test_provenance_pins_match_manifest_and_required_commits(
    artifact: dict, manifest: dict
) -> None:
    """Both checkouts must be pinned to the manifest commits, verified."""
    pinned = artifact["provenance"]["pinned_repositories"]
    assert pinned["agx_arm_ros"]["commit"] == (
        manifest["repositories"]["agx_arm_ros"]["commit"]
    ) == "b9ad14de2a6eb818a2d206fbd1adba48343662f7"
    assert pinned["agx_arm_sim"]["commit"] == (
        manifest["repositories"]["agx_arm_sim"]["commit"]
    ) == "f8cd8b147c75d59e14f90fb0646770eefa268ed0"
    assert pinned["agx_arm_ros"]["head_verified"] is True
    assert pinned["agx_arm_sim"]["head_verified"] is True


def test_agx_arm_urdf_submodule_pin_is_explicit(artifact: dict) -> None:
    """The agx_arm_urdf submodule pin must be recorded and verified."""
    submodule = artifact["provenance"]["submodules"]["agx_arm_urdf_in_agx_arm_ros"]
    assert submodule["pin"] == AGX_ARM_URDF_PIN
    assert submodule["recorded_gitlink"] == AGX_ARM_URDF_PIN
    assert submodule["checked_out_head"] == AGX_ARM_URDF_PIN
    assert submodule["pin_verified"] is True
    gitmodules = (
        ROOT / "ros_ws/src/agx_arm_ros/.gitmodules"
    ).read_text(encoding="utf-8")
    assert "agx_arm_urdf.git" in gitmodules
    assert submodule["url"].endswith("agx_arm_urdf.git")


def test_agx_arm_sim_gitlink_recorded_but_not_the_model_source(
    artifact: dict,
) -> None:
    """The vendor sim gitlink must be recorded as uninitialized."""
    submodule = artifact["provenance"]["submodules"]["agx_arm_urdf_in_agx_arm_sim"]
    assert submodule["recorded_gitlink"] == AGX_ARM_SIM_GITLINK
    assert submodule["initialized"] is False


@pytest.mark.parametrize(
    "relative",
    [
        "ros_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf"
        "/piper_l/urdf/piper_l_description.urdf",
        "ros_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf"
        "/piper_l/urdf/piper_l_with_gripper_description.xacro",
        "robot_assets/piper_l/piper_l_gripper_d435.urdf",
        "robot_assets/piper_l/USD/piper_l_gripper_d435.usd",
        "isaac/vendor/agx_arm_sim/Moveit2/piper_l_gripper_moveit_config"
        "/config/piper_l.srdf",
    ],
)
def test_recorded_sha256_matches_checkout(artifact: dict, relative: str) -> None:
    """Every recorded hash must equal the on-disk file hash."""
    hashes = artifact["provenance"]["file_sha256"]
    assert relative in hashes
    assert hashes[relative] == _sha256(ROOT / relative)


def test_pinned_vendor_assets_match_runtime_bytes(artifact: dict) -> None:
    """Runtime URDF/USD and every USD sublayer must equal the pinned sim files."""
    identity = artifact["provenance"]["runtime_vendor_identity"]
    assert set(identity["usd_configuration_layers"]) == {
        "piper_l_gripper_d435_base.usd",
        "piper_l_gripper_d435_physics.usd",
        "piper_l_gripper_d435_robot.usd",
        "piper_l_gripper_d435_sensor.usd",
    }
    assert identity["urdf"]["vendor_path"] == str(EXECUTION_URDF.relative_to(ROOT))
    assert identity["usd_top"]["vendor_path"] == str(EXECUTION_USD.relative_to(ROOT))
    for key, pair in (
        ("urdf", identity["urdf"]),
        ("usd_top", identity["usd_top"]),
        *identity["usd_configuration_layers"].items(),
    ):
        runtime_path = ROOT / pair["runtime_path"]
        vendor_path = ROOT / pair["vendor_path"]
        assert pair["byte_identical"] is True
        assert runtime_path.read_bytes() == vendor_path.read_bytes()
        assert (
            pair["runtime_sha256"]
            == pair["vendor_sha256"]
            == _sha256(runtime_path)
            == PINNED_RUNTIME_HASHES[key]
        )


def test_joint_limit_conflict_and_execution_policy(artifact: dict) -> None:
    """Both pinned sources and the joint4/5 intersection must be explicit."""
    conflict = artifact["limit_conflict"]
    assert conflict["policy"] == "intersection_of_pinned_urdf_limits"
    assert conflict["planning_urdf"]["sha256"] == _sha256(PIPER_L_URDF)
    assert conflict["execution_urdf"]["sha256"] == _sha256(EXECUTION_URDF)
    assert conflict["execution_usd"]["sha256"] == _sha256(EXECUTION_USD)
    assert [item["joint"] for item in conflict["conflicts"]] == ["joint4", "joint5"]
    planning = _urdf_joints(PIPER_L_URDF)
    execution = _urdf_joints(EXECUTION_URDF)
    for item in conflict["conflicts"]:
        name = item["joint"]
        assert item["agx_arm_urdf_rad"] == [
            planning[name]["limit"]["lower"],
            planning[name]["limit"]["upper"],
        ] == list(PLANNING_LIMITS[name])
        assert item["agx_arm_sim_rad"] == [
            execution[name]["limit"]["lower"],
            execution[name]["limit"]["upper"],
        ] == list(EXECUTION_LIMITS[name])
    for name, limits in EXECUTION_LIMITS.items():
        assert conflict["execution_limits_rad"][name] == list(limits)
    assert "Pending MoveIt integration" in conflict["planning_enforcement"]


def test_arm_joints_exact_order_types_and_axes(artifact: dict) -> None:
    """The contract must list exactly six revolute arm joints in URDF order."""
    arm = artifact["arm_joints"]
    assert arm["count"] == 6
    assert arm["order"] == ARM_JOINT_ORDER
    assert [joint["name"] for joint in arm["joints"]] == ARM_JOINT_ORDER
    for joint in arm["joints"]:
        assert joint["type"] == "revolute"
        assert joint["axis"] == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("name", ARM_JOINT_ORDER)
def test_arm_joints_official_limits_exact(artifact: dict, name: str) -> None:
    """Limits must equal the official URDF values, not rounded variants."""
    joints = {joint["name"]: joint for joint in artifact["arm_joints"]["joints"]}
    lower, upper = PLANNING_LIMITS[name]
    assert joints[name]["limit"]["lower"] == lower
    assert joints[name]["limit"]["upper"] == upper
    assert joints[name]["limit"]["effort"] == 100.0
    assert joints[name]["limit"]["velocity"] == 5.0


def test_arm_joints_match_pinned_urdf_file(artifact: dict) -> None:
    """Recorded arm joints must match an independent parse of the URDF."""
    urdf = _urdf_joints(PIPER_L_URDF)
    for joint in artifact["arm_joints"]["joints"]:
        source = urdf[joint["name"]]
        assert joint["type"] == source["type"]
        assert joint["axis"] == source["axis"]
        assert joint["limit"] == source["limit"]
        assert joint["source"]["sha256"] == _sha256(PIPER_L_URDF)


def test_gripper_mapping_matches_official_mimic_relations(
    artifact: dict,
) -> None:
    """Mimic multipliers must be the official +0.5/-0.5 with range [0, 0.1]."""
    gripper = artifact["gripper"]
    assert gripper["logical_master_joint"]["opening_range_m"] == [0.0, 0.1]
    assert gripper["logical_master_joint"]["child_link_has_geometry"] is False
    fingers = {finger["name"]: finger for finger in gripper["fingers"]}
    assert set(fingers) == set(GRIPPER_JOINTS)
    assert fingers["gripper_joint1"]["mimic"]["multiplier"] == pytest.approx(0.5)
    assert fingers["gripper_joint2"]["mimic"]["multiplier"] == pytest.approx(-0.5)
    assert fingers["gripper_joint1"]["limit"]["lower"] == 0.0
    assert fingers["gripper_joint1"]["limit"]["upper"] == pytest.approx(0.05)
    assert fingers["gripper_joint2"]["limit"]["lower"] == pytest.approx(-0.05)
    assert fingers["gripper_joint2"]["limit"]["upper"] == 0.0


def test_logical_to_finger_mapping_matches_rocobrick(artifact: dict) -> None:
    """The recorded mapping rule must be the one rocobrick implements."""
    mapping = artifact["gripper"]["logical_to_finger_mapping"]
    assert mapping["implementation"] == "rocobrick.gripper.logical_gripper_to_fingers"
    for opening in (0.0, 0.02, 0.05, 0.1):
        finger1, finger2 = logical_gripper_to_fingers(opening)
        assert finger1 == pytest.approx(opening * 0.5)
        assert finger2 == pytest.approx(opening * -0.5)
        assert fingers_to_logical_gripper(finger1, finger2) == pytest.approx(opening)
    with pytest.raises(ValueError, match="within"):
        logical_gripper_to_fingers(0.15)


def test_usd_expected_dof_order_matches_config_and_composed_urdf(
    artifact: dict,
) -> None:
    """The expected 8-DOF order must match config and URDF declaration order."""
    assert artifact["usd"]["expected_dof_order"] == EXPECTED_DOF_ORDER
    config = json.loads((ROOT / "config/user_config.json").read_text(encoding="utf-8"))
    for robot in config["Robot_Config"]["Robots"]:
        assert robot["Joint_Order"] == EXPECTED_DOF_ORDER
        recorded = artifact["usd"]["expected_dof_order_evidence"][
            "config_joint_order"
        ][robot["Name"]]
        assert recorded == EXPECTED_DOF_ORDER
    composed = ET.parse(COMPOSED_URDF).getroot()
    movable = [
        joint.attrib["name"]
        for joint in composed.findall("joint")
        if joint.attrib["type"] in {"revolute", "continuous", "prismatic"}
    ]
    assert artifact["usd"]["expected_dof_order_evidence"][
        "urdf_movable_declaration_order"
    ] == movable
    assert movable == EXPECTED_DOF_ORDER


def test_usd_composed_values_only_with_extraction_evidence(artifact: dict) -> None:
    """Composed-USD claims must exist only when extraction succeeded."""
    usd = artifact["usd"]
    extraction = usd["extraction"]
    evidenced = usd["evidenced_movable_joints"]
    if extraction["status"] != "ok":
        assert evidenced == []
        assert usd["composed_stage_facts"] is None
        return
    assert extraction["method"] == "pxr_usd_python"
    assert len(evidenced) == 8
    assert [joint["name"] for joint in evidenced] == EXPECTED_DOF_ORDER
    stage_facts = usd["composed_stage_facts"]
    assert stage_facts["default_prim"].startswith("/")
    assert stage_facts["up_axis"] == "Z"
    for joint in evidenced:
        assert joint["usd_path"].startswith(stage_facts["default_prim"] + "/")


def test_usd_movable_joints_agree_with_execution_limits(artifact: dict) -> None:
    """USD limits (deg for revolute) match the pinned execution URDF."""
    joints = {
        joint["name"]: joint
        for joint in artifact["usd"]["evidenced_movable_joints"]
    }
    if not joints:
        pytest.skip("USD extraction was unavailable for this artifact")
    for name, (lower, upper) in EXECUTION_LIMITS.items():
        assert joints[name]["usd_type"] == "PhysicsRevoluteJoint"
        assert joints[name]["units"] == "deg"
        assert math.isclose(
            math.radians(joints[name]["lower_limit"]), lower, abs_tol=1e-6
        )
        assert math.isclose(
            math.radians(joints[name]["upper_limit"]), upper, abs_tol=1e-6
        )
        assert joints[name]["execution_urdf"]["sha256"] == _sha256(EXECUTION_URDF)
        assert joints[name]["planning_urdf_limits_rad"] == list(PLANNING_LIMITS[name])
        assert joints[name]["matches_execution_urdf_limits"] is True
        assert joints[name]["within_planning_urdf_limits"] is True
    for name in GRIPPER_JOINTS:
        assert joints[name]["usd_type"] == "PhysicsPrismaticJoint"
        assert joints[name]["units"] == "m"
    assert joints["gripper_joint1"]["lower_limit"] == pytest.approx(0.0)
    assert joints["gripper_joint1"]["upper_limit"] == pytest.approx(0.05)
    assert joints["gripper_joint2"]["lower_limit"] == pytest.approx(-0.05)
    assert joints["gripper_joint2"]["upper_limit"] == pytest.approx(0.0)
    assert all(
        joints[name]["matches_official_urdf_limits"] for name in GRIPPER_JOINTS
    )


def test_usd_runtime_dof_order_is_not_claimed(artifact: dict) -> None:
    """The artifact must never claim a runtime DOF order read from USD."""
    runtime = artifact["usd"]["runtime_dof_order"]
    assert runtime["stored_in_usd"] is False
    assert "dof_names" in runtime["authoritative_source"]
    assert "not a value read from the USD" in runtime["note"]


def test_isaac_bridge_topics_match_official_xacro(artifact: dict) -> None:
    """Bridge topics must equal the official TopicBasedSystem contract."""
    topics = artifact["planning"]["isaac_bridge_topics"]
    assert topics["hardware_plugin"] == "topic_based_ros2_control/TopicBasedSystem"
    assert topics["joint_states"] == "/isaac_joint_states"
    assert topics["joint_commands"] == "/isaac_joint_command"
    assert topics["message_type"] == "sensor_msgs/JointState"
    xacro = (MOVEIT_DIR / "piper_l.ros2_control.xacro").read_text(encoding="utf-8")
    assert '<param name="joint_commands_topic">/isaac_joint_command</param>' in xacro
    assert '<param name="joint_states_topic">/isaac_joint_states</param>' in xacro
    assert topics["source"]["sha256"] == _sha256(MOVEIT_DIR / "piper_l.ros2_control.xacro")
    # The P0.5 bridge must use exactly these topic names.
    bridge = (ROOT / "isaac" / "p05_host_bridge.py").read_text(encoding="utf-8")
    assert 'TOPIC_JOINT_STATES = "/isaac_joint_states"' in bridge
    assert 'TOPIC_JOINT_COMMANDS = "/isaac_joint_command"' in bridge


def test_planning_groups_match_official_srdf(artifact: dict) -> None:
    """Groups must equal the official SRDF arm and gripper definitions."""
    srdf = ET.parse(SRDF_PATH).getroot()
    assert artifact["planning"]["robot_name"] == srdf.attrib["name"] == "piper_l"
    groups = artifact["planning"]["groups"]
    assert set(groups) == {"arm", "gripper"}
    assert groups["arm"]["joints"] == ARM_JOINT_ORDER
    assert groups["arm"]["chains"] == [
        {"base_link": "base_link", "tip_link": "link6"}
    ]
    assert groups["gripper"]["joints"] == GRIPPER_JOINTS
    srdf_groups = {
        group.attrib["name"]: [j.attrib["name"] for j in group.findall("joint")]
        for group in srdf.findall("group")
    }
    assert groups["arm"]["joints"] == srdf_groups["arm"]
    assert groups["gripper"]["joints"] == srdf_groups["gripper"]


def test_controllers_match_official_moveit_and_ros2_configs(
    artifact: dict,
) -> None:
    """Controllers must match the official MoveIt/ros2_control configs."""
    moveit = yaml.safe_load(
        (MOVEIT_DIR / "moveit_controllers.yaml").read_text(encoding="utf-8")
    )
    ros2 = yaml.safe_load(
        (MOVEIT_DIR / "ros2_controllers.yaml").read_text(encoding="utf-8")
    )
    assert artifact["planning"]["moveit_controllers"] == moveit
    assert artifact["planning"]["ros2_controllers"] == ros2
    moveit_joints = moveit["moveit_simple_controller_manager"]
    assert moveit_joints["arm_controller"]["joints"] == ARM_JOINT_ORDER
    assert moveit_joints["gripper_controller"]["joints"] == GRIPPER_JOINTS
    assert ros2["controller_manager"]["ros__parameters"]["arm_controller"] == {
        "type": "joint_trajectory_controller/JointTrajectoryController"
    }
    assert ros2["arm_controller"]["ros__parameters"]["joints"] == ARM_JOINT_ORDER
    assert (
        ros2["gripper_controller"]["ros__parameters"]["joints"] == GRIPPER_JOINTS
    )


def test_tcp_is_recorded_as_unresolved(artifact: dict) -> None:
    """TCP status must be unresolved with concrete named facts."""
    tcp = artifact["tcp"]
    assert tcp["status"] == "unresolved"
    assert tcp["facts"]
    joined = " ".join(tcp["facts"])
    assert "no tool0" in joined or "no tool0 or dedicated TCP frame" in joined
    assert "link6" in joined


def test_license_caveat_records_unresolved_licensing(artifact: dict) -> None:
    """The artifact must carry the license caveat and unconfirmed assets."""
    license_facts = artifact["license"]
    caveat = license_facts["caveat"]
    assert "not fully confirmed" in caveat
    assert "MIT" in caveat
    assert "SolidWorks" in caveat
    sim_license = license_facts["agx_arm_sim_license"]
    assert sim_license["standalone_license_file"] is False
    assert sim_license["assets_license_confirmed"] is False
    assert not (ROOT / "isaac/vendor/agx_arm_sim/LICENSE").exists()


def test_robot_assets_snapshot_drift_is_recorded(artifact: dict) -> None:
    """The runtime asset copy is tracked but differs from the MoveIt model."""
    snapshot = artifact["provenance"]["robot_assets_snapshot"]
    assert snapshot["version_controlled"] is True
    assert snapshot["has_mimic_declarations"] is False
    assert snapshot["has_flange_link"] is False
    assert snapshot["differs_from_pinned_submodule"] is True
    assert "wider joint4/joint5 limits" in snapshot["note"]
