"""Pure generation of one combined dual-arm MoveIt model.

The existing dual launch creates two isolated robot stacks. This module builds
one collision/planning model from those already-prefixed per-robot URDF/SRDF
files without touching vendor assets or changing the single-robot generator.
The output is consumed by the coordinated MoveIt launch in a later layer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import yaml


class RobotConfigLike(Protocol):
    """Minimal generated robot config consumed by the combiner."""

    robot_id: str
    urdf: Path
    srdf: Path
    kinematics: Path
    joint_limits: Path
    moveit_controllers: Path


DUAL_ROOT_LINK = "dual_world"
DUAL_ARM_GROUP = "dual_arm"
DUAL_ARM_TCP_GROUP = "dual_arm_tcp"


@dataclass(frozen=True)
class DualArmConfigPaths:
    """Generated combined model/configuration files."""

    directory: Path
    urdf: Path
    srdf: Path
    kinematics: Path
    joint_limits: Path
    moveit_controllers: Path


def _parse_xml(path: Path) -> ET.Element:
    """Read and parse one XML configuration file."""
    return ET.fromstring(path.read_text(encoding="utf-8"))


def load_base_poses(path: Path, expected_sha256: str, robot_ids: tuple[str, ...]):
    """Load exact world base poses from the immutable runtime configuration.

    Returns:
        Robot positions and RPY rotations converted from configured WXYZ quaternions.
    """
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("runtime configuration hash mismatch")
    document = json.loads(raw)
    robots = document["Robot_Config"]["Robots"]
    entries = {item["Name"]: item for item in robots}
    if len(entries) != len(robots) or not set(robot_ids) <= set(entries):
        raise ValueError("runtime configuration robot set is invalid")
    result = {}
    for robot_id in robot_ids:
        base = entries[robot_id]["Robot_Base_Frame"]
        position = base["Position"]
        quaternion = base["Orientation"]
        if (
            len(position) != 3
            or len(quaternion) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in (*position, *quaternion)
            )
        ):
            raise ValueError("base pose must contain finite vectors")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 1e-12:
            raise ValueError("base orientation quaternion is zero")
        w, x, y, z = (value / norm for value in quaternion)
        rpy = (
            math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        )
        result[robot_id] = (tuple(position), rpy)
    return result


def combine_urdfs(
    configs: tuple[RobotConfigLike, ...],
    base_poses: Mapping[
        str, tuple[tuple[float, float, float], tuple[float, float, float]]
    ]
    | None = None,
) -> str:
    """Combine prefixed URDFs below one fixed ``dual_world`` root.

    Returns:
        A single URDF with all links/joints and one fixed root joint per robot.

    Raises:
        ValueError: If robot models are empty, duplicate names, or already
            contain a world/root link that would collide with the combined root.
    """
    if not configs:
        raise ValueError("dual model requires at least one robot config")
    root = ET.Element("robot", {"name": "rocobrick_dual_arm"})
    ET.SubElement(root, "link", {"name": DUAL_ROOT_LINK})
    links: set[str] = {DUAL_ROOT_LINK}
    joints: set[str] = set()
    for config in configs:
        model = _parse_xml(config.urdf)
        model_links = {item.get("name") for item in model.findall("link")}
        model_joints = {item.get("name") for item in model.findall("joint")}
        world_link = f"{config.robot_id}_world"
        world_joint = f"{config.robot_id}_world_to_base_link"
        if world_link in model_links:
            model_links.remove(world_link)
        root_element = next(
            (
                item
                for item in model.findall("joint")
                if item.get("name") == world_joint
            ),
            None,
        )
        if root_element is not None:
            model_joints.remove(world_joint)
        if None in model_links or None in model_joints:
            raise ValueError(f"{config.robot_id}: URDF contains unnamed link/joint")
        if links & model_links or joints & model_joints:
            raise ValueError(f"{config.robot_id}: combined URDF names are not unique")
        if DUAL_ROOT_LINK in model_links:
            raise ValueError(f"{config.robot_id}: URDF already owns {DUAL_ROOT_LINK}")
        for element in model.findall("link"):
            if element.get("name") != world_link:
                root.append(copy.deepcopy(element))
        for element in model.findall("joint"):
            if element.get("name") != world_joint:
                root.append(copy.deepcopy(element))
        base = f"{config.robot_id}_base_link"
        root_joint = f"{config.robot_id}_dual_world_joint"
        if root_joint in joints:
            raise ValueError(f"duplicate dual root joint {root_joint}")
        joint_element = ET.SubElement(
            root, "joint", {"name": root_joint, "type": "fixed"}
        )
        ET.SubElement(joint_element, "parent", {"link": DUAL_ROOT_LINK})
        ET.SubElement(joint_element, "child", {"link": base})
        xyz, rpy = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        if base_poses is not None:
            xyz, rpy = base_poses.get(config.robot_id, (xyz, rpy))
        ET.SubElement(
            joint_element,
            "origin",
            {
                "xyz": " ".join(str(value) for value in xyz),
                "rpy": " ".join(str(value) for value in rpy),
            },
        )
        links.update(model_links)
        joints.update(model_joints)
        joints.add(root_joint)
    return ET.tostring(root, encoding="unicode")


def combine_srdfs(configs: tuple[RobotConfigLike, ...]) -> str:
    """Combine prefixed SRDFs and add the `dual_arm` subgroup.

    Returns:
        One SRDF retaining each robot's groups and adding a joint-space
        ``dual_arm`` group plus a TCP subgroup.
    """
    if not configs:
        raise ValueError("dual SRDF requires at least one robot config")
    root = ET.Element("robot", {"name": "rocobrick_dual_arm"})
    for config in configs:
        source = _parse_xml(config.srdf)
        for child in source:
            if child.tag != "virtual_joint":
                root.append(copy.deepcopy(child))
    dual = ET.SubElement(root, "group", {"name": DUAL_ARM_GROUP})
    dual_tcp = ET.SubElement(root, "group", {"name": DUAL_ARM_TCP_GROUP})
    for config in configs:
        prefix = f"{config.robot_id}_"
        ET.SubElement(dual, "group", {"name": f"{prefix}arm_tcp"})
        ET.SubElement(dual_tcp, "group", {"name": f"{prefix}arm_tcp"})
    return ET.tostring(root, encoding="unicode")


def combine_kinematics(configs: tuple[RobotConfigLike, ...]) -> dict[str, object]:
    """Combine per-robot kinematics and add explicit joint-group metadata."""
    result: dict[str, object] = {}
    for config in configs:
        data = yaml.safe_load(config.kinematics.read_text(encoding="utf-8")) or {}
        for group, value in data.items():
            if str(group).endswith("gripper"):
                continue
            prefixed_group = str(group)
            if not prefixed_group.startswith(f"{config.robot_id}_"):
                prefixed_group = f"{config.robot_id}_{prefixed_group}"
            result[prefixed_group] = copy.deepcopy(value)
    # The joint-space subgroup does not need a multi-tip IK solver; it is used
    # for collision-checked joint trajectory planning. Pose goals remain per
    # arm and are composed by the coordinator.
    # ``dual_arm`` is a joint-space subgroup, so it intentionally has no
    # single-chain IK solver entry. Pose goals are solved per arm by the
    # coordinator before the joint-space joint-goal is planned.
    return result


def combine_joint_limits(configs: tuple[RobotConfigLike, ...]) -> dict[str, object]:
    """Combine the exact prefixed joint-limit documents."""
    result: dict[str, object] = {"joint_limits": {}}
    for config in configs:
        data = yaml.safe_load(config.joint_limits.read_text(encoding="utf-8")) or {}
        for joint, value in (data.get("joint_limits") or {}).items():
            name = str(joint)
            prefixed = (
                name
                if name.startswith(f"{config.robot_id}_")
                else f"{config.robot_id}_{name}"
            )
            if prefixed in result["joint_limits"]:
                raise ValueError(f"duplicate combined joint limit {prefixed}")
            result["joint_limits"][prefixed] = copy.deepcopy(value)
    return result


def combine_moveit_controllers(
    configs: tuple[RobotConfigLike, ...],
) -> dict[str, object]:
    """Declare each arm/gripper controller for the coordinated MoveIt model."""
    if not configs:
        raise ValueError("dual controllers require at least one robot config")
    result: dict[str, object] = {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {"controller_names": []},
    }
    names: list[str] = result["moveit_simple_controller_manager"]["controller_names"]
    for config in configs:
        data = (
            yaml.safe_load(config.moveit_controllers.read_text(encoding="utf-8")) or {}
        )
        source = data.get("moveit_simple_controller_manager") or {}
        for name in source.get("controller_names", []):
            controller = copy.deepcopy(source[name])
            combined_name = f"{config.robot_id}_{name}"
            joints = [
                joint
                if joint.startswith(f"{config.robot_id}_")
                else f"{config.robot_id}_{joint}"
                for joint in controller.get("joints", [])
            ]
            controller["joints"] = joints
            controller["action_ns"] = (
                f"/{config.robot_id}/{name}/follow_joint_trajectory"
            )
            names.append(combined_name)
            result["moveit_simple_controller_manager"][combined_name] = controller
    return result


def generate_dual_arm_configs(
    configs: tuple[RobotConfigLike, ...],
    output_dir: Path,
    base_poses: Mapping[
        str, tuple[tuple[float, float, float], tuple[float, float, float]]
    ]
    | None = None,
) -> DualArmConfigPaths:
    """Write one combined model/config directory from generated robot configs."""
    directory = Path(output_dir) / "dual_arm"
    directory.mkdir(parents=True, exist_ok=True)
    paths = DualArmConfigPaths(
        directory=directory,
        urdf=directory / "model.urdf",
        srdf=directory / "model.srdf",
        kinematics=directory / "kinematics.yaml",
        joint_limits=directory / "joint_limits.yaml",
        moveit_controllers=directory / "moveit_controllers.yaml",
    )
    paths.urdf.write_text(combine_urdfs(configs, base_poses), encoding="utf-8")
    paths.srdf.write_text(combine_srdfs(configs), encoding="utf-8")
    paths.kinematics.write_text(
        yaml.safe_dump(combine_kinematics(configs), sort_keys=False), encoding="utf-8"
    )
    paths.joint_limits.write_text(
        yaml.safe_dump(combine_joint_limits(configs), sort_keys=False), encoding="utf-8"
    )
    paths.moveit_controllers.write_text(
        yaml.safe_dump(combine_moveit_controllers(configs), sort_keys=False),
        encoding="utf-8",
    )
    return paths
