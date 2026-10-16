"""A3 offline contract: dual-robot profile names and the namespaced launch.

No ROS, no Docker, no Isaac: these tests pin the per-robot endpoint/profile
naming contract and the generation of the two disjoint MoveIt models from the
pinned single-robot sources. The ``xacro``-dependent end-to-end test runs the
real expansion when ``xacro`` is on PATH (for example inside the ROS image)
and is skipped on a bare host.

The generated names are cross-checked against
``rocobrick.robot_runtime_profile`` — the A5 routing contract — so a drift of
either side fails here.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from rocobrick.moveit_contract import TCP_LINK as CONTRACT_TCP_LINK
from rocobrick.robot_runtime_profile import (
    RobotRuntimeProfile,
    assert_profiles_disjoint,
)

ROOT = Path(__file__).parents[1]
PACKAGE_DIR = ROOT / "ros_ws/src/rocobrick_piper_l_moveit_config"
LAUNCH_DIR = PACKAGE_DIR / "launch"
HELPER_PATH = LAUNCH_DIR / "dual_robot_config.py"
DUAL_LAUNCH_PATH = LAUNCH_DIR / "dual_move_group.launch.py"
SINGLE_LAUNCH_PATH = LAUNCH_DIR / "move_group.launch.py"
ARTIFACT_PATH = ROOT / "artifacts/model-audit/piper_l_tcp_binding_contract.json"

_ROBOT_IDS = ("piper_0", "piper_1")


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "rocobrick_dual_robot_config", HELPER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helper():
    return _load_helper()


def _synthetic_model(tcp_xyz: str) -> str:
    """Minimal single-robot expansion with the pinned reference structure."""
    arm = []
    for index in range(1, 7):
        parent = "base_link" if index == 1 else f"link{index - 1}"
        arm.append(
            f'<link name="link{index}"/>'
            f'<joint name="joint{index}" type="revolute">'
            f'<parent link="{parent}"/><child link="link{index}"/>'
            f'<axis xyz="0 0 1"/>'
            f'<limit lower="-2.0" upper="2.0" effort="10" velocity="5"/>'
            f"</joint>"
        )
    control_joints = "".join(
        f'<joint name="{name}"><command_interface name="position"/>'
        f'<state_interface name="position">'
        f'<param name="initial_value">0</param></state_interface>'
        f'<state_interface name="velocity"/></joint>'
        for name in (
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper_joint1",
            "gripper_joint2",
        )
    )
    return (
        '<?xml version="1.0"?><robot name="piper_l">'
        '<link name="world"/>'
        '<joint name="world_to_base_link" type="fixed">'
        '<parent link="world"/><child link="base_link"/>'
        '<origin xyz="0 0 0" rpy="0 0 0"/>'
        "</joint>"
        '<link name="base_link"/>' + "".join(arm) + '<link name="flange_link"/>'
        '<joint name="flange_joint" type="fixed"><parent link="link6"/>'
        '<child link="flange_link"/></joint>'
        '<link name="gripper_base"/>'
        '<joint name="gripper_base_joint" type="fixed"><parent link="flange_link"/>'
        '<child link="gripper_base"/><origin xyz="0 0 0.0045" rpy="0 0 0"/></joint>'
        '<link name="gripper_link"/>'
        '<joint name="gripper" type="prismatic"><parent link="gripper_base"/>'
        '<child link="gripper_link"/><axis xyz="0 0 1"/>'
        '<limit lower="0" upper="0.1" effort="10" velocity="3"/></joint>'
        '<link name="gripper_link1"/>'
        '<joint name="gripper_joint1" type="prismatic"><parent link="gripper_base"/>'
        '<child link="gripper_link1"/><axis xyz="0 0 1"/>'
        '<origin xyz="0 0 0.138" rpy="1.5707963 0 0"/>'
        '<limit lower="0" upper="0.05" effort="10" velocity="3"/>'
        '<mimic joint="gripper" multiplier="0.5" offset="0.0"/></joint>'
        '<link name="gripper_link2"/>'
        '<joint name="gripper_joint2" type="prismatic"><parent link="gripper_base"/>'
        '<child link="gripper_link2"/><axis xyz="0 0 -1"/>'
        '<origin xyz="0 0 0.138" rpy="1.5707963 0 -3.1415926"/>'
        '<limit lower="-0.05" upper="0" effort="10" velocity="3"/>'
        '<mimic joint="gripper" multiplier="-0.5" offset="0.0"/></joint>'
        # Wrist-camera chain mirrored from urdf/piper_l.camera_stand.xacro:
        # the SRDF collision entries reference these links.
        '<link name="camera_stand_link"/>'
        '<joint name="camera_stand_joint" type="fixed"><parent link="gripper_base"/>'
        '<child link="camera_stand_link"/>'
        '<origin xyz="-0.034 -0.001 0.022" rpy="0 0 0"/></joint>'
        '<link name="d435_camera_link"/>'
        '<joint name="d435_camera_joint" type="fixed">'
        '<parent link="camera_stand_link"/>'
        '<child link="d435_camera_link"/>'
        '<origin xyz="-0.031 -0.028 0.0018" rpy="0 -1.22 0"/></joint>'
        '<link name="camera_bottom_screw_frame"/>'
        '<joint name="camera_link_joint" type="fixed">'
        '<parent link="d435_camera_link"/>'
        '<child link="camera_bottom_screw_frame"/>'
        '<origin xyz="0.010600000000000002 0.0175 0.0125" rpy="0 0 0"/></joint>'
        '<link name="camera_link"/>'
        '<joint name="camera_joint" type="fixed">'
        '<parent link="camera_bottom_screw_frame"/>'
        '<child link="camera_link"/><origin xyz="0 0 0" rpy="0 0 0"/></joint>'
        '<ros2_control name="PiperLIsaacTopicBasedSystem" type="system"><hardware>'
        "<plugin>topic_based_ros2_control/TopicBasedSystem</plugin>"
        '<param name="joint_commands_topic">/isaac_joint_command</param>'
        '<param name="joint_states_topic">/isaac_joint_states</param>'
        "</hardware>" + control_joints + "</ros2_control>"
        '<link name="rocobrick_tcp"/>'
        '<joint name="rocobrick_tcp_joint" type="fixed"><parent link="link6"/>'
        f'<child link="rocobrick_tcp"/><origin xyz="{tcp_xyz}" rpy="0 0 0"/></joint>'
        "</robot>"
    )


def _generate(helper, output_dir: Path, expanded_urdf: str, robot_ids=_ROBOT_IDS):
    return helper.generate_dual_robot_configs(
        robot_ids=robot_ids,
        output_dir=output_dir,
        package_dir=PACKAGE_DIR,
        xacro_runner=lambda _path: expanded_urdf,
    )


def _assert_generated_pair(helper, configs) -> None:
    """Shared assertions over a generated dual-robot configuration pair."""
    assert tuple(config.robot_id for config in configs) == _ROBOT_IDS
    seen: dict[str, str] = {}
    for config in configs:
        prefix = f"{config.robot_id}_"
        root = ET.parse(config.urdf).getroot()
        links = {element.get("name") for element in root.findall("link")}
        joints = {element.get("name") for element in root.findall("joint")}
        assert links and joints
        # Every link/joint (hence every TF frame id) is prefixed: no /tf
        # collision between the two robot_state_publishers.
        assert all(name.startswith(prefix) for name in links | joints)
        params = {
            param.get("name"): param.text
            for param in root.findall("ros2_control/hardware/param")
        }
        assert params == {
            "joint_commands_topic": f"/{config.robot_id}/isaac_joint_command",
            "joint_states_topic": f"/{config.robot_id}/isaac_joint_states",
        }
        assert [
            joint.get("name") for joint in root.findall("ros2_control/joint")
        ] == list(config.endpoints.joint_names)
        tcp_joint = next(
            joint
            for joint in root.findall("joint")
            if joint.get("name") == prefix + helper.TCP_JOINT
        )
        assert tcp_joint.get("type") == "fixed"
        # Planning origins match the original runtime mounting without
        # reversing either finger.
        gripper_origins = {
            joint.get("name"): (
                joint.find("origin").get("xyz"),
                joint.find("origin").get("rpy"),
            )
            for joint in root.findall("joint")
            if joint.get("name")
            in {
                prefix + "gripper_base_joint",
                prefix + "gripper_joint1",
                prefix + "gripper_joint2",
            }
        }
        assert gripper_origins == {
            prefix + "gripper_base_joint": ("0 0 0", "0 0 0"),
            prefix + "gripper_joint1": ("0 0 0.1358", "1.5707963 0 0"),
            prefix + "gripper_joint2": ("0 0 0.1358", "1.5707963 0 -3.1415926"),
        }
        assert tcp_joint.find("parent").get("link") == prefix + helper.LINK6
        assert tcp_joint.find("child").get("link") == prefix + helper.TCP_LINK
        tcp_links = [
            element
            for element in root.findall("link")
            if element.get("name") == prefix + helper.TCP_LINK
        ]
        assert len(tcp_links) == 1
        assert list(tcp_links[0]) == [], (
            "the calibrated TCP link must stay geometry-free"
        )
        for name in links | joints:
            owner = seen.setdefault(name, config.robot_id)
            assert owner == config.robot_id, f"models share the name {name!r}"

        srdf = ET.parse(config.srdf).getroot()
        groups = {group.get("name"): group for group in srdf.findall("group")}
        assert set(groups) == {prefix + group for group in helper.SRDF_GROUPS}
        chain = groups[config.endpoints.planning_group].find("chain")
        assert chain.attrib == {
            "base_link": prefix + helper.BASE_LINK,
            "tip_link": prefix + helper.TCP_LINK,
        }
        for joint in srdf.iter("joint"):
            assert joint.get("name") in joints
        for link in srdf.iter("link"):
            assert link.get("name") in links
        for entry in srdf.findall("disable_collisions"):
            assert entry.get("link1") in links
            assert entry.get("link2") in links
        for state in srdf.findall("group_state"):
            assert state.get("group") in groups

        kinematics = yaml.safe_load(config.kinematics.read_text(encoding="utf-8"))
        assert set(kinematics) == {
            prefix + helper.ARM_GROUP,
            prefix + helper.GRIPPER_GROUP,
            prefix + helper.TCP_GROUP,
        }
        limits = yaml.safe_load(config.joint_limits.read_text(encoding="utf-8"))[
            "joint_limits"
        ]
        assert tuple(limits) == config.endpoints.joint_names
        controllers = yaml.safe_load(
            config.moveit_controllers.read_text(encoding="utf-8")
        )
        simple = controllers["moveit_simple_controller_manager"]
        assert simple["controller_names"] == ["arm_controller", "gripper_controller"]
        assert (
            tuple(simple["arm_controller"]["joints"])
            == config.endpoints.joint_names[:6]
        )
        assert (
            tuple(simple["gripper_controller"]["joints"])
            == config.endpoints.joint_names[6:]
        )
        ns = config.endpoints.namespace
        ros2_controllers = yaml.safe_load(
            config.ros2_controllers.read_text(encoding="utf-8")
        )
        assert set(ros2_controllers) == {
            f"{ns}/controller_manager",
            f"{ns}/arm_controller",
            f"{ns}/gripper_controller",
        }
        assert (
            tuple(ros2_controllers[f"{ns}/arm_controller"]["ros__parameters"]["joints"])
            == (config.endpoints.joint_names[:6])
        )

    profiles = tuple(
        RobotRuntimeProfile.from_robot_id(config.robot_id) for config in configs
    )
    assert_profiles_disjoint(profiles)
    for config, profile in zip(configs, profiles):
        endpoints = config.endpoints
        assert endpoints.namespace == profile.namespace
        assert endpoints.joint_names == profile.joint_names
        assert endpoints.base_frame == profile.base_frame
        assert endpoints.tip_link == profile.tip_link
        assert endpoints.planning_group == profile.planning_group
        assert endpoints.link6_link == profile.link6_link
        assert endpoints.link6_planning_group == profile.link6_planning_group
        assert endpoints.clock == profile.clock
        assert endpoints.isaac_joint_states == profile.isaac_joint_states
        assert endpoints.joint_states == profile.joint_states
        assert endpoints.isaac_joint_command == profile.isaac_joint_command
        assert endpoints.move_group_action == profile.move_group_action
        assert endpoints.arm_follow == profile.arm_follow
        assert endpoints.gripper_follow == profile.gripper_follow
        assert endpoints.list_controllers == profile.list_controllers
        assert endpoints.plan == profile.plan
        assert endpoints.compute_fk == profile.compute_fk


# ------------------------------------------------------------------ profile


def test_profile_tip_link_is_the_calibrated_tcp_not_link6() -> None:
    for robot_id in _ROBOT_IDS:
        profile = RobotRuntimeProfile.from_robot_id(robot_id)
        assert profile.tip_link == f"{robot_id}_{CONTRACT_TCP_LINK}"
        assert "link6" not in profile.tip_link
        assert profile.planning_group == f"{robot_id}_arm_tcp"
        assert profile.base_frame == f"{robot_id}_base_link"
        # The legacy raw-flange chain is still addressable, never conflated
        # with the calibrated skill TCP.
        assert profile.link6_link == f"{robot_id}_link6"
        assert profile.link6_planning_group == f"{robot_id}_arm"
    canonical = RobotRuntimeProfile.canonical()
    assert canonical.tip_link == CONTRACT_TCP_LINK
    assert canonical.planning_group == "arm_tcp"
    assert canonical.link6_link == "link6"
    assert canonical.link6_planning_group == "arm"


@pytest.mark.parametrize(
    "robot_id", ["", "piper 0", "piper/0", "0piper", "piper-0", "piper.0"]
)
def test_profile_rejects_non_ros_safe_robot_ids(robot_id: str) -> None:
    with pytest.raises(ValueError):
        RobotRuntimeProfile.from_robot_id(robot_id)


def test_profile_with_the_legacy_field_set_derives_the_link6_names() -> None:
    """A5 regression: profiles built without the A3 link6 fields stay valid.

    The fields were added to the dataclass while ``canonical()`` was being
    updated in parallel, which transiently raised ``TypeError`` for callers
    holding the pre-A3 field set. Both factory outputs must stay constructible
    from that field set, with the legacy names derived identically.
    """
    references = (
        RobotRuntimeProfile.canonical(),
        RobotRuntimeProfile.canonical("piper_9"),
        *(
            RobotRuntimeProfile.from_robot_id(robot_id)
            for robot_id in _ROBOT_IDS
        ),
    )
    for reference in references:
        legacy_kwargs = {
            key: value
            for key, value in dataclasses.asdict(reference).items()
            if key not in ("link6_link", "link6_planning_group")
        }
        rebuilt = RobotRuntimeProfile(**legacy_kwargs)
        assert rebuilt == reference
        assert rebuilt.link6_link == reference.link6_link
        assert rebuilt.link6_planning_group == reference.link6_planning_group
        assert dataclasses.replace(reference, link6_link="custom_link").link6_link == (
            "custom_link"
        )
    assert RobotRuntimeProfile.canonical().link6_link == "link6"
    assert RobotRuntimeProfile.canonical().link6_planning_group == "arm"
    assert RobotRuntimeProfile.from_robot_id("piper_1").link6_link == "piper_1_link6"


def test_profile_namespaces_share_only_the_global_clock() -> None:
    profiles = tuple(
        RobotRuntimeProfile.from_robot_id(robot_id) for robot_id in _ROBOT_IDS
    )
    assert_profiles_disjoint(profiles)
    assert {profile.clock for profile in profiles} == {"/clock"}
    for profile in profiles:
        for endpoint in (
            profile.isaac_joint_states,
            profile.joint_states,
            profile.isaac_joint_command,
            profile.move_group_action,
            profile.arm_follow,
            profile.gripper_follow,
            profile.list_controllers,
            profile.plan,
            profile.compute_fk,
        ):
            assert endpoint.startswith(f"/{profile.robot_id}/")
    with pytest.raises(ValueError, match="duplicate robot id"):
        assert_profiles_disjoint((profiles[0], profiles[0]))


def test_host_bridge_frozen_names_are_consumed_by_the_profile() -> None:
    """A4 freeze: <robot_id>_<joint> names, per-robot isaac topics, one clock."""
    profile = RobotRuntimeProfile.from_robot_id("piper_1")
    assert profile.isaac_joint_states == "/piper_1/isaac_joint_states"
    assert profile.isaac_joint_command == "/piper_1/isaac_joint_command"
    assert profile.joint_states == "/piper_1/joint_states"
    assert profile.joint_names[0] == "piper_1_joint1"
    assert profile.joint_names[6] == "piper_1_gripper_joint1"


# ------------------------------------------------------------------- helper


def test_helper_endpoints_match_the_profile(helper) -> None:
    for robot_id in _ROBOT_IDS:
        endpoints = helper.robot_endpoints(robot_id)
        profile = RobotRuntimeProfile.from_robot_id(robot_id)
        assert endpoints.namespace == profile.namespace
        assert endpoints.joint_names == profile.joint_names
        assert endpoints.base_frame == profile.base_frame
        assert endpoints.tip_link == profile.tip_link
        assert endpoints.planning_group == profile.planning_group
        assert endpoints.link6_link == profile.link6_link
        assert endpoints.link6_planning_group == profile.link6_planning_group
        assert endpoints.clock == profile.clock
        assert endpoints.isaac_joint_states == profile.isaac_joint_states
        assert endpoints.joint_states == profile.joint_states
        assert endpoints.isaac_joint_command == profile.isaac_joint_command
        assert endpoints.move_group_action == profile.move_group_action
        assert endpoints.arm_follow == profile.arm_follow
        assert endpoints.gripper_follow == profile.gripper_follow
        assert endpoints.list_controllers == profile.list_controllers
        assert endpoints.plan == profile.plan
        assert endpoints.compute_fk == profile.compute_fk
    assert helper.CLOCK_TOPIC == "/clock"
    assert helper.TCP_LINK == CONTRACT_TCP_LINK
    assert helper.parse_robot_ids("piper_0, piper_1") == _ROBOT_IDS


@pytest.mark.parametrize(
    "value",
    [
        "",
        ",",
        "piper 0",
        "piper/0",
        "0piper",
        "piper-0",
        "piper_0,piper_0",
        "piper_0,,piper_0",
    ],
)
def test_helper_rejects_invalid_robot_id_lists(helper, value: str) -> None:
    with pytest.raises(ValueError):
        helper.parse_robot_ids(value)


def test_helper_has_no_ros_or_rocobrick_imports(helper) -> None:
    """The generator must import inside the ROS image (no host venv deps)."""
    tree = ast.parse(HELPER_PATH.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert modules <= {
        "__future__",
        "copy",
        "re",
        "subprocess",
        "xml",
        "dataclasses",
        "pathlib",
        "yaml",
    }


# --------------------------------------------------------------- generation


def test_synthetic_generation_produces_disjoint_models(helper, tmp_path) -> None:
    expanded = _synthetic_model(
        "-1.887447531601083e-09 -1.3530843112619095e-16 0.1305895866844472"
    )
    configs = _generate(helper, tmp_path / "configs", expanded)
    _assert_generated_pair(helper, configs)
    for config in configs:
        assert config.urdf.is_file()
        assert config.srdf.is_file()
        assert config.ros2_controllers.is_file()


def test_generation_fails_closed_without_the_calibrated_tcp(helper, tmp_path) -> None:
    expanded = _synthetic_model("0 0 0").replace("rocobrick_tcp", "not_the_tcp")
    with pytest.raises(ValueError, match="calibrated rocobrick_tcp"):
        _generate(helper, tmp_path / "configs", expanded)


def test_generation_fails_closed_on_a_relative_control_topic(helper, tmp_path) -> None:
    expanded = _synthetic_model("0 0 0").replace(
        "/isaac_joint_command", "isaac_joint_command"
    )
    with pytest.raises(ValueError, match="unexpected hardware topic"):
        _generate(helper, tmp_path / "configs", expanded)


def test_joint_limit_document_must_cover_exactly_the_eight_joints(helper) -> None:
    source = yaml.safe_load(
        (PACKAGE_DIR / "config/joint_limits.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="eight piper joints"):
        helper.build_robot_joint_limits(
            {"joint_limits": {"joint1": source["joint_limits"]["joint1"]}}, "piper_0"
        )


def test_moveit_controller_joint_sets_are_frozen(helper) -> None:
    source = yaml.safe_load(
        (PACKAGE_DIR / "config/moveit_controllers.yaml").read_text(encoding="utf-8")
    )
    source["moveit_simple_controller_manager"]["arm_controller"]["joints"] = ["joint1"]
    with pytest.raises(ValueError, match="joint set changed"):
        helper.build_robot_moveit_controllers(source, "piper_0")


def test_ros2_controller_documents_are_namespaced(helper) -> None:
    source = yaml.safe_load(
        (PACKAGE_DIR / "config/ros2_controllers.yaml").read_text(encoding="utf-8")
    )
    document = helper.build_robot_ros2_controllers(source, "piper_1")
    assert set(document) == {
        "/piper_1/controller_manager",
        "/piper_1/arm_controller",
        "/piper_1/gripper_controller",
    }
    assert document["/piper_1/arm_controller"]["ros__parameters"]["joints"] == [
        f"piper_1_joint{index}" for index in range(1, 7)
    ]


# ------------------------------------------------------------- real sources


@pytest.mark.skipif(shutil.which("xacro") is None, reason="xacro is not installed")
def test_real_pinned_sources_generate_disjoint_models(helper, tmp_path) -> None:
    configs = helper.generate_dual_robot_configs(
        robot_ids=_ROBOT_IDS,
        output_dir=tmp_path / "out",
        package_dir=PACKAGE_DIR,
    )
    _assert_generated_pair(helper, configs)
    translation = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))["tcp_contract"][
        "transform"
    ]["translation_m"]
    for config in configs:
        joint = next(
            joint
            for joint in ET.parse(config.urdf).getroot().findall("joint")
            if joint.get("name") == f"{config.robot_id}_rocobrick_tcp_joint"
        )
        origin = joint.find("origin")
        assert [float(value) for value in origin.get("xyz").split()] == [
            float(value) for value in translation
        ]
        assert [float(value) for value in origin.get("rpy").split()] == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------- launch




