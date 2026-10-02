"""Pure helpers that derive one namespaced MoveIt model/config per robot id.

The pinned model in this package is a single un-prefixed Piper_L: the vendor
description (``isaac/vendor/agx_arm_sim`` / the pinned agx_arm_urdf chain)
cannot be renamed in place, and two copies of it cannot share one ROS graph:

* ``robot_state_publisher`` publishes link frame ids on the global ``/tf`` and
  ``/tf_static`` topics, so identical link names collide even when the nodes
  are namespaced;
* ``move_group``/``controller_manager``/``ros2_control`` address joints,
  planning groups, and controller joints by the names inside the model.

The dual-robot stack therefore expands the pinned single-robot xacro once
(``urdf/piper_l.ros2_control.urdf.xacro``, which already appends the
project-calibrated ``rocobrick_tcp`` overlay and bakes the initial positions)
and derives, per robot id, a model whose links/joints/groups carry the
``<robot_id>_`` prefix, plus the matching SRDF and YAML configuration. The
``TopicBasedSystem`` hardware topics are rewritten to the robot namespace
``/<robot_id>/isaac_joint_command`` and ``/<robot_id>/isaac_joint_states``,
which is the frozen host-bridge contract (host bridge keeps a single node/
participant and one global ``/clock``; the DDS domain is shared).

Nothing here is ROS-specific beyond PyYAML: the module is importable offline
and ``tests/test_dual_robot_profile_launch.py`` pins every derived name equal
to ``rocobrick.robot_runtime_profile`` (the A5 routing contract).
"""

from __future__ import annotations

import copy
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_ROBOT_IDS: tuple[str, ...] = ("piper_0", "piper_1")

ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)
GRIPPER_JOINT_NAMES: tuple[str, ...] = ("gripper_joint1", "gripper_joint2")
JOINT_NAMES: tuple[str, ...] = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

# MoveIt/SRDF group names of the pinned single-robot model.
ARM_GROUP: str = "arm"
GRIPPER_GROUP: str = "gripper"
TCP_GROUP: str = "arm_tcp"
SRDF_GROUPS: tuple[str, ...] = (ARM_GROUP, GRIPPER_GROUP, TCP_GROUP)

BASE_LINK: str = "base_link"
LINK6: str = "link6"
TCP_LINK: str = "rocobrick_tcp"
TCP_JOINT: str = "rocobrick_tcp_joint"
# Legacy raw flange group/links of the vendor model (un-prefixed).
FLANGE_LINK: str = "flange_link"
GRIPPER_BASE_LINK: str = "gripper_base"
GRIPPER_LINK: str = "gripper_link"

CLOCK_TOPIC: str = "/clock"
ISAAC_JOINT_STATES: str = "isaac_joint_states"
ISAAC_JOINT_COMMAND: str = "isaac_joint_command"
JOINT_STATES: str = "joint_states"
MOVE_GROUP_ACTION: str = "move_action"
ARM_CONTROLLER: str = "arm_controller"
GRIPPER_CONTROLLER: str = "gripper_controller"
LIST_CONTROLLERS_SERVICE: str = "controller_manager/list_controllers"
PLAN_SERVICE: str = "plan_kinematic_path"
COMPUTE_FK_SERVICE: str = "compute_fk"

# The planning checkout includes a flange spacer and a different finger
# origin. Match the runtime URDF without changing either finger's orientation.
_PINNED_GRIPPER_ORIGINS: dict[str, tuple[str, str]] = {
    "gripper_base_joint": ("0 0 0.0045", "0 0 0"),
    "gripper_joint1": ("0 0 0.138", "1.5707963 0 0"),
    "gripper_joint2": ("0 0 0.138", "1.5707963 0 -3.1415926"),
}
_CORRECTED_GRIPPER_ORIGINS: dict[str, tuple[str, str]] = {
    "gripper_base_joint": ("0 0 0", "0 0 0"),
    "gripper_joint1": ("0 0 0.1358", "1.5707963 0 0"),
    "gripper_joint2": ("0 0 0.1358", "1.5707963 0 -3.1415926"),
}

REQUIRED_CONTROLLERS: tuple[str, ...] = (
    "joint_state_broadcaster",
    ARM_CONTROLLER,
    GRIPPER_CONTROLLER,
)
CONTROL_TOPIC_PARAMS: tuple[str, ...] = (
    "joint_commands_topic",
    "joint_states_topic",
)

SINGLE_ROBOT_XACRO: str = "urdf/piper_l.ros2_control.urdf.xacro"
SINGLE_ROBOT_SRDF: str = "config/piper_l.srdf"
SINGLE_ROBOT_KINEMATICS: str = "config/kinematics.yaml"
SINGLE_ROBOT_JOINT_LIMITS: str = "config/joint_limits.yaml"
SINGLE_ROBOT_MOVEIT_CONTROLLERS: str = "config/moveit_controllers.yaml"
SINGLE_ROBOT_ROS2_CONTROLLERS: str = "config/ros2_controllers.yaml"

_ROBOT_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# tag -> attributes that reference a URDF link/joint name.
_URDF_REFERENCES: dict[str, tuple[str, ...]] = {
    "link": ("name",),
    "joint": ("name",),
    "parent": ("link",),
    "child": ("link",),
    "mimic": ("joint",),
    "transmission": ("name",),
    "actuator": ("name",),
    "ros2_control": ("name",),
}

# tag -> attributes that reference an SRDF group/joint/link name. Group state
# names (``<group_state name=...>``) are per-group and stay unchanged.
_SRDF_REFERENCES: dict[str, tuple[str, ...]] = {
    "group": ("name",),
    "chain": ("base_link", "tip_link"),
    "joint": ("name",),
    "link": ("name",),
    "group_state": ("group",),
    "end_effector": ("name", "parent_link", "group", "parent_group"),
    "disable_collisions": ("link1", "link2"),
    "passive_joint": ("name",),
    "virtual_joint": ("name", "parent_frame", "child_link"),
}


def validate_robot_id(robot_id: str) -> str:
    """Return ``robot_id`` if it is a ROS-safe identifier, else raise."""
    if not isinstance(robot_id, str) or not _ROBOT_ID_PATTERN.match(robot_id):
        raise ValueError(f"robot id {robot_id!r} is not a ROS-safe identifier")
    return robot_id


def robot_prefix(robot_id: str) -> str:
    """Model-name prefix of one robot (``piper_0_``)."""
    return validate_robot_id(robot_id) + "_"


def robot_namespace(robot_id: str) -> str:
    """Absolute device namespace of one robot (``/piper_0``)."""
    return "/" + validate_robot_id(robot_id)


def parse_robot_ids(value: str) -> tuple[str, ...]:
    """Parse the ``robot_ids`` launch argument (comma-separated, unique)."""
    if not isinstance(value, str):
        raise ValueError("robot_ids must be a string")
    robot_ids = tuple(part.strip() for part in value.split(",") if part.strip())
    if not robot_ids:
        raise ValueError("robot_ids must list at least one robot id")
    if len(set(robot_ids)) != len(robot_ids):
        raise ValueError("robot_ids must be unique")
    for robot_id in robot_ids:
        validate_robot_id(robot_id)
    return robot_ids


def prefixed_joint_names(robot_id: str) -> tuple[str, ...]:
    """Joint names of the generated model (``piper_0_joint1`` ...)."""
    prefix = robot_prefix(robot_id)
    return tuple(prefix + name for name in JOINT_NAMES)


@dataclass(frozen=True)
class RobotEndpoints:
    """Every runtime name of one robot stack, derived from the robot id."""

    robot_id: str
    namespace: str
    joint_names: tuple[str, ...]
    base_frame: str
    tip_link: str
    planning_group: str
    link6_link: str
    link6_planning_group: str
    clock: str
    isaac_joint_states: str
    joint_states: str
    isaac_joint_command: str
    move_group_action: str
    arm_follow: str
    gripper_follow: str
    list_controllers: str
    plan: str
    compute_fk: str

    def serialize(self) -> dict[str, object]:
        """JSON-serializable form for logs/evidence."""
        return {
            "robot_id": self.robot_id,
            "namespace": self.namespace,
            "joint_names": list(self.joint_names),
            "base_frame": self.base_frame,
            "tip_link": self.tip_link,
            "planning_group": self.planning_group,
            "link6_link": self.link6_link,
            "link6_planning_group": self.link6_planning_group,
            "clock": self.clock,
            "isaac_joint_states": self.isaac_joint_states,
            "joint_states": self.joint_states,
            "isaac_joint_command": self.isaac_joint_command,
            "move_group_action": self.move_group_action,
            "arm_follow": self.arm_follow,
            "gripper_follow": self.gripper_follow,
            "list_controllers": self.list_controllers,
            "plan": self.plan,
            "compute_fk": self.compute_fk,
        }


def robot_endpoints(robot_id: str) -> RobotEndpoints:
    """Derive the frozen per-robot endpoint contract from the robot id."""
    ns = robot_namespace(robot_id)
    prefix = robot_prefix(robot_id)
    return RobotEndpoints(
        robot_id=robot_id,
        namespace=ns,
        joint_names=prefixed_joint_names(robot_id),
        base_frame=prefix + BASE_LINK,
        tip_link=prefix + TCP_LINK,
        planning_group=prefix + TCP_GROUP,
        link6_link=prefix + LINK6,
        link6_planning_group=prefix + ARM_GROUP,
        clock=CLOCK_TOPIC,
        isaac_joint_states=f"{ns}/{ISAAC_JOINT_STATES}",
        joint_states=f"{ns}/{JOINT_STATES}",
        isaac_joint_command=f"{ns}/{ISAAC_JOINT_COMMAND}",
        move_group_action=f"{ns}/{MOVE_GROUP_ACTION}",
        arm_follow=f"{ns}/{ARM_CONTROLLER}/follow_joint_trajectory",
        gripper_follow=f"{ns}/{GRIPPER_CONTROLLER}/follow_joint_trajectory",
        list_controllers=f"{ns}/{LIST_CONTROLLERS_SERVICE}",
        plan=f"{ns}/{PLAN_SERVICE}",
        compute_fk=f"{ns}/{COMPUTE_FK_SERVICE}",
    )


def _local_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def prefix_model(
    xml_text: str, prefix: str, references: dict[str, tuple[str, ...]]
) -> str:
    """Prefix every referenced name of a URDF/SRDF XML document.

    Only attributes listed in ``references`` are rewritten, so topics, frame
    origins, limits, and ``disable_collisions`` reasons stay byte-identical.
    """
    root = ET.fromstring(xml_text)
    for element in root.iter():
        for attribute in references.get(_local_tag(element), ()):
            value = element.get(attribute)
            if value is not None:
                element.set(attribute, prefix + value)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def align_gripper_geometry_to_runtime_asset(urdf_text: str) -> str:
    """Align planning origins with the original, unmodified runtime URDF.

    Fail closed on an unreviewed planning-model revision. Both finger
    orientations, axes, limits and mesh references remain unchanged.
    """
    root = ET.fromstring(urdf_text)
    joints = {element.get("name"): element for element in root.findall("joint")}
    for name, (pinned_xyz, pinned_rpy) in _PINNED_GRIPPER_ORIGINS.items():
        joint = joints.get(name)
        if joint is None:
            raise ValueError(f"pinned model lost joint {name!r}")
        origin = joint.find("origin")
        if origin is None:
            raise ValueError(f"joint {name!r} has no origin element")
        if origin.get("xyz") != pinned_xyz or origin.get("rpy") != pinned_rpy:
            raise ValueError(
                f"pinned origin of {name!r} changed: "
                f"xyz={origin.get('xyz')!r} rpy={origin.get('rpy')!r}"
            )
    for name, (xyz, rpy) in _CORRECTED_GRIPPER_ORIGINS.items():
        origin = joints[name].find("origin")
        origin.set("xyz", xyz)
        origin.set("rpy", rpy)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def namespace_control_topics(urdf_text: str, namespace: str) -> str:
    """Rewrite the ``TopicBasedSystem`` topics to the robot namespace.

    Fails closed when a topic is not a single-segment absolute name, so a
    model change can never silently leave both robots on one shared topic.
    """
    root = ET.fromstring(urdf_text)
    for param in root.iter("param"):
        if param.get("name") not in CONTROL_TOPIC_PARAMS:
            continue
        topic = (param.text or "").strip()
        if not topic.startswith("/") or topic.count("/") != 1:
            raise ValueError(f"unexpected hardware topic {topic!r}")
        param.text = f"{namespace}{topic}"
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def build_robot_urdf(expanded_urdf_text: str, robot_id: str) -> str:
    """Namespaced, prefixed model for one robot.

    The expanded single-robot model already carries the calibrated
    ``link6 -> rocobrick_tcp`` overlay and the baked initial positions; this
    only renames it and moves the hardware topics into the namespace.
    """
    namespaced = namespace_control_topics(expanded_urdf_text, robot_namespace(robot_id))
    return prefix_model(namespaced, robot_prefix(robot_id), _URDF_REFERENCES)


def build_robot_srdf(single_srdf_text: str, robot_id: str) -> str:
    """Prefixed SRDF (groups, chains, end effector, collision pairs)."""
    return prefix_model(single_srdf_text, robot_prefix(robot_id), _SRDF_REFERENCES)


def build_robot_joint_limits(document: dict, robot_id: str) -> dict:
    """Prefixed ``robot_description_planning`` document for one robot."""
    limits = document["joint_limits"]
    if set(limits) != set(JOINT_NAMES):
        raise ValueError("joint_limits.yaml must define exactly the eight piper joints")
    prefix = robot_prefix(robot_id)
    return {
        "joint_limits": {
            prefix + name: copy.deepcopy(limits[name]) for name in JOINT_NAMES
        }
    }


def build_robot_kinematics(document: dict, robot_id: str) -> dict:
    """Prefixed IK solver document (keys are the prefixed SRDF groups)."""
    if set(document) != set(SRDF_GROUPS):
        raise ValueError(
            "kinematics.yaml must define exactly the arm/gripper/arm_tcp groups"
        )
    prefix = robot_prefix(robot_id)
    return {prefix + group: copy.deepcopy(document[group]) for group in SRDF_GROUPS}


def build_robot_moveit_controllers(document: dict, robot_id: str) -> dict:
    """Trajectory-execution document with prefixed controller joints.

    Controller names stay un-prefixed: MoveIt resolves the relative action
    name under the namespaced ``move_group`` node, which is exactly the
    ``/<robot_id>/arm_controller/follow_joint_trajectory`` contract.
    """
    simple = document["moveit_simple_controller_manager"]
    if tuple(simple["controller_names"]) != (ARM_CONTROLLER, GRIPPER_CONTROLLER):
        raise ValueError("moveit_controllers.yaml controller set changed")
    result = copy.deepcopy(document)
    simple = result["moveit_simple_controller_manager"]
    prefix = robot_prefix(robot_id)
    for controller in (ARM_CONTROLLER, GRIPPER_CONTROLLER):
        expected = (
            ARM_JOINT_NAMES if controller == ARM_CONTROLLER else GRIPPER_JOINT_NAMES
        )
        joints = tuple(simple[controller]["joints"])
        if set(joints) != set(expected):
            raise ValueError(f"{controller} joint set changed")
        simple[controller]["joints"] = [prefix + joint for joint in joints]
    return result


def build_robot_ros2_controllers(document: dict, robot_id: str) -> dict:
    """Controller-manager parameter document for one namespaced stack.

    Top-level node keys are rewritten to the fully qualified namespaced node
    names (``/piper_0/controller_manager`` ...) so the generated file can be
    passed straight to the namespaced ``ros2_control_node``.
    """
    ns = robot_namespace(robot_id)
    prefix = robot_prefix(robot_id)
    manager = copy.deepcopy(document["controller_manager"])
    declared = tuple(manager["ros__parameters"])
    for controller in REQUIRED_CONTROLLERS:
        if controller not in declared:
            raise ValueError(f"ros2_controllers.yaml does not declare {controller}")
    result: dict = {f"{ns}/controller_manager": manager}
    for controller in (ARM_CONTROLLER, GRIPPER_CONTROLLER):
        params = copy.deepcopy(document[controller])
        expected = (
            ARM_JOINT_NAMES if controller == ARM_CONTROLLER else GRIPPER_JOINT_NAMES
        )
        joints = tuple(params["ros__parameters"]["joints"])
        if set(joints) != set(expected):
            raise ValueError(f"{controller} joint set changed")
        params["ros__parameters"]["joints"] = [prefix + joint for joint in joints]
        result[f"{ns}/{controller}"] = params
    return result


@dataclass(frozen=True)
class RobotConfigPaths:
    """Generated per-robot config files plus the derived endpoint contract."""

    robot_id: str
    directory: Path
    urdf: Path
    srdf: Path
    kinematics: Path
    joint_limits: Path
    moveit_controllers: Path
    ros2_controllers: Path
    endpoints: RobotEndpoints


def _write_yaml(path: Path, document: dict) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def generate_robot_configs(
    *,
    output_dir: Path,
    robot_id: str,
    expanded_urdf_text: str,
    single_srdf_text: str,
    config_dir: Path,
) -> RobotConfigPaths:
    """Write one robot's URDF/SRDF/YAML files under ``output_dir/<robot_id>``."""
    validate_robot_id(robot_id)
    directory = Path(output_dir) / robot_id
    directory.mkdir(parents=True, exist_ok=True)
    paths = RobotConfigPaths(
        robot_id=robot_id,
        directory=directory,
        urdf=directory / "model.urdf",
        srdf=directory / "model.srdf",
        kinematics=directory / "kinematics.yaml",
        joint_limits=directory / "joint_limits.yaml",
        moveit_controllers=directory / "moveit_controllers.yaml",
        ros2_controllers=directory / "ros2_controllers.yaml",
        endpoints=robot_endpoints(robot_id),
    )
    paths.urdf.write_text(
        build_robot_urdf(expanded_urdf_text, robot_id), encoding="utf-8"
    )
    paths.srdf.write_text(
        build_robot_srdf(single_srdf_text, robot_id), encoding="utf-8"
    )
    _write_yaml(
        paths.kinematics,
        build_robot_kinematics(
            yaml.safe_load(
                (config_dir / "kinematics.yaml").read_text(encoding="utf-8")
            ),
            robot_id,
        ),
    )
    _write_yaml(
        paths.joint_limits,
        build_robot_joint_limits(
            yaml.safe_load(
                (config_dir / "joint_limits.yaml").read_text(encoding="utf-8")
            ),
            robot_id,
        ),
    )
    _write_yaml(
        paths.moveit_controllers,
        build_robot_moveit_controllers(
            yaml.safe_load(
                (config_dir / "moveit_controllers.yaml").read_text(encoding="utf-8")
            ),
            robot_id,
        ),
    )
    _write_yaml(
        paths.ros2_controllers,
        build_robot_ros2_controllers(
            yaml.safe_load(
                (config_dir / "ros2_controllers.yaml").read_text(encoding="utf-8")
            ),
            robot_id,
        ),
    )
    return paths


def expand_single_robot_urdf(single_xacro: Path) -> str:
    """Expand the pinned single-robot xacro (runs ``xacro`` as a subprocess)."""
    completed = subprocess.run(
        ["xacro", str(single_xacro)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _model_names(
    urdf_text: str,
) -> tuple[set[str], set[str], dict[str, str], dict[str, str], set[str]]:
    root = ET.fromstring(urdf_text)
    links = {element.get("name") for element in root.findall("link")}
    joints = {element.get("name") for element in root.findall("joint")}
    children: dict[str, str] = {}
    parents: dict[str, str] = {}
    mimics: set[str] = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            parents[child.get("link")] = parent.get("link")
            children[joint.get("name")] = child.get("link")
        mimic = joint.find("mimic")
        if mimic is not None:
            mimics.add(mimic.get("joint"))
    return links, joints, children, parents, mimics


def assert_generated_models_consistent(configs: tuple[RobotConfigPaths, ...]) -> None:
    """Fail closed unless the generated models are self-consistent and disjoint.

    Checks per robot: the calibrated TCP frame exists and hangs off the raw
    ``link6`` through a fixed joint, every parent/child/mimic reference
    resolves, and the model exposes every profile joint. Across robots: no
    shared link or joint name.
    """
    if not configs:
        raise ValueError("no robot configs generated")
    seen: dict[str, str] = {}
    for config in configs:
        links, joints, _children, parents, mimics = _model_names(
            config.urdf.read_text(encoding="utf-8")
        )
        prefix = robot_prefix(config.robot_id)
        expected_links = {prefix + name for name in (BASE_LINK, LINK6, TCP_LINK)}
        if not expected_links <= links:
            raise ValueError(
                f"{config.robot_id}: generated model misses {expected_links - links}"
            )
        if not set(prefixed_joint_names(config.robot_id)) <= joints:
            raise ValueError(f"{config.robot_id}: generated model misses joints")
        if parents.get(prefix + TCP_LINK) != prefix + LINK6:
            raise ValueError(
                f"{config.robot_id}: {prefix}{TCP_LINK} is not a child of "
                f"{prefix}{LINK6}"
            )
        for name, parent in parents.items():
            if name not in links or parent not in links:
                raise ValueError(
                    f"{config.robot_id}: unresolved parent/child {name!r} -> {parent!r}"
                )
        if not mimics <= joints:
            raise ValueError(f"{config.robot_id}: unresolved mimic joint")
        for name in links | joints:
            owner = seen.setdefault(name, config.robot_id)
            if owner != config.robot_id:
                raise ValueError(f"generated models share the name {name!r}")


def generate_dual_robot_configs(
    *,
    robot_ids: tuple[str, ...],
    output_dir: Path,
    package_dir: Path,
    xacro_runner=expand_single_robot_urdf,
) -> tuple[RobotConfigPaths, ...]:
    """Generate every robot's files from the pinned single-robot sources.

    ``xacro_runner`` defaults to the ``xacro`` subprocess; offline tests can
    inject a stub so the generation path is covered without ROS installed.
    """
    if not robot_ids:
        raise ValueError("robot_ids must not be empty")
    output_dir = Path(output_dir)
    config_dir = Path(package_dir) / "config"
    expanded_urdf_text = align_gripper_geometry_to_runtime_asset(
        xacro_runner(Path(package_dir) / SINGLE_ROBOT_XACRO)
    )
    single_srdf_text = (config_dir / "piper_l.srdf").read_text(encoding="utf-8")
    if "rocobrick_tcp" not in expanded_urdf_text:
        raise ValueError(
            "the expanded model does not carry the calibrated rocobrick_tcp overlay"
        )
    configs = tuple(
        generate_robot_configs(
            output_dir=output_dir,
            robot_id=robot_id,
            expanded_urdf_text=expanded_urdf_text,
            single_srdf_text=single_srdf_text,
            config_dir=config_dir,
        )
        for robot_id in robot_ids
    )
    assert_generated_models_consistent(configs)
    return configs
