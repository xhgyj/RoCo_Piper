#!/usr/bin/env python3
"""Create one collision world from two copies of the official Piper L URDF."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

REFERENCES = {
    "link": ("name",),
    "joint": ("name",),
    "parent": ("link",),
    "child": ("link",),
    "mimic": ("joint",),
    "transmission": ("name",),
    "actuator": ("name",),
}


def prefix_tree(element: ET.Element, prefix: str) -> ET.Element:
    result = copy.deepcopy(element)
    for node in result.iter():
        for attribute in REFERENCES.get(node.tag, ()):
            if attribute in node.attrib:
                node.attrib[attribute] = prefix + node.attrib[attribute]
    return result


def append_camera(robot: ET.Element, prefix: str) -> None:
    stand = ET.SubElement(robot, "link", {"name": f"{prefix}camera_stand_link"})
    collision = ET.SubElement(stand, "collision")
    ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": "0.045 0.035 0.025"})
    stand_joint = ET.SubElement(robot, "joint", {"name": f"{prefix}camera_stand_joint", "type": "fixed"})
    ET.SubElement(stand_joint, "origin", {"xyz": "-0.034 -0.001 0.022", "rpy": "0 0 0"})
    ET.SubElement(stand_joint, "parent", {"link": f"{prefix}gripper_base"})
    ET.SubElement(stand_joint, "child", {"link": f"{prefix}camera_stand_link"})
    camera = ET.SubElement(robot, "link", {"name": f"{prefix}d435_camera_link"})
    camera_collision = ET.SubElement(camera, "collision")
    ET.SubElement(camera_collision, "origin", {"xyz": "0 -0.0175 0", "rpy": "0 0 0"})
    camera_geometry = ET.SubElement(camera_collision, "geometry")
    ET.SubElement(camera_geometry, "box", {"size": "0.02505 0.09 0.025"})
    camera_joint = ET.SubElement(robot, "joint", {"name": f"{prefix}d435_camera_joint", "type": "fixed"})
    ET.SubElement(camera_joint, "origin", {"xyz": "-0.031 -0.028 0.0018", "rpy": "0 -1.22 0"})
    ET.SubElement(camera_joint, "parent", {"link": f"{prefix}camera_stand_link"})
    ET.SubElement(camera_joint, "child", {"link": f"{prefix}d435_camera_link"})


def append_tcp(robot: ET.Element, prefix: str) -> None:
    ET.SubElement(robot, "link", {"name": f"{prefix}tcp_link"})
    joint = ET.SubElement(robot, "joint", {"name": f"{prefix}tcp_joint", "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0.1358", "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": f"{prefix}link6"})
    ET.SubElement(joint, "child", {"link": f"{prefix}tcp_link"})


def generate_urdf(single_path: Path, output: Path) -> None:
    source = ET.parse(single_path).getroot()
    robot = ET.Element("robot", {"name": "rocobrick_dual_piper_l"})
    ET.SubElement(robot, "link", {"name": "world"})
    bases = {
        "piper_0_": ("0.25 -0.25 0", "0 0 1.57079632679"),
        "piper_1_": ("-0.25 -0.25 0", "0 0 1.57079632679"),
    }
    for prefix, (xyz, rpy) in bases.items():
        for child in source:
            if child.tag == "link" and child.get("name") == "world":
                continue
            if child.tag == "joint" and child.get("name") == "world_to_base_link":
                continue
            if child.tag == "material" and any(item.tag == "material" and item.get("name") == child.get("name") for item in robot):
                continue
            robot.append(prefix_tree(child, prefix))
        base_joint = ET.SubElement(robot, "joint", {"name": f"{prefix}world_to_base", "type": "fixed"})
        ET.SubElement(base_joint, "origin", {"xyz": xyz, "rpy": rpy})
        ET.SubElement(base_joint, "parent", {"link": "world"})
        ET.SubElement(base_joint, "child", {"link": f"{prefix}base_link"})
        append_tcp(robot, prefix)
        append_camera(robot, prefix)
    ET.indent(robot)
    output.write_text(ET.tostring(robot, encoding="unicode") + "\n", encoding="utf-8")


def generate_srdf(urdf_path: Path, output: Path) -> None:
    robot = ET.Element("robot", {"name": "rocobrick_dual_piper_l"})
    for prefix in ("piper_0_", "piper_1_"):
        arm = ET.SubElement(robot, "group", {"name": f"{prefix}arm"})
        ET.SubElement(arm, "chain", {"base_link": f"{prefix}base_link", "tip_link": f"{prefix}tcp_link"})
        gripper = ET.SubElement(robot, "group", {"name": f"{prefix}gripper"})
        ET.SubElement(gripper, "joint", {"name": f"{prefix}gripper"})
        ET.SubElement(robot, "passive_joint", {"name": f"{prefix}gripper_joint1"})
        ET.SubElement(robot, "passive_joint", {"name": f"{prefix}gripper_joint2"})
        ET.SubElement(robot, "end_effector", {"name": f"{prefix}eef", "parent_link": f"{prefix}tcp_link", "group": f"{prefix}gripper", "parent_group": f"{prefix}arm"})
        home = ET.SubElement(robot, "group_state", {"name": "home", "group": f"{prefix}arm"})
        for number in range(1, 7):
            ET.SubElement(home, "joint", {"name": f"{prefix}joint{number}", "value": "0"})
    both = ET.SubElement(robot, "group", {"name": "dual_arms"})
    ET.SubElement(both, "group", {"name": "piper_0_arm"})
    ET.SubElement(both, "group", {"name": "piper_1_arm"})
    disabled: set[tuple[str, str, str]] = set()
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is not None and child is not None:
            disabled.add((parent.attrib["link"], child.attrib["link"], "Adjacent"))
    for prefix in ("piper_0_", "piper_1_"):
        for first, second in (
            ("gripper_link1", "gripper_link2"),
            ("gripper_link1", "flange_link"),
            ("gripper_link2", "flange_link"),
            ("gripper_base", "link6"),
            ("gripper_base", "link5"),
            ("flange_link", "link5"),
        ):
            disabled.add((prefix + first, prefix + second, "Never"))
    for first, second, reason in sorted(disabled):
        ET.SubElement(
            robot,
            "disable_collisions",
            {"link1": first, "link2": second, "reason": reason},
        )
    ET.indent(robot)
    output.write_text(ET.tostring(robot, encoding="unicode") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    args = parser.parse_args()
    generate_urdf(args.single, args.urdf)
    generate_srdf(args.urdf, args.srdf)


if __name__ == "__main__":
    main()
