"""Pure-numpy URDF chain forward kinematics for the container-side gate.

The ROS/MoveIt container runs Python 3.10 with numpy and without Pinocchio, so
the clearance gate carries its own kinematics.  It parses the very model URDF
the container generates for MoveIt (``dual-robot-config/dual_arm/model.urdf``)
and evaluates the parent->child chain, so it cannot disagree with the model
MoveIt itself planned on.  ``tests/test_clearance_gate.py`` pins the result
against Pinocchio on the repository model.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence

import numpy as np

_REVOLUTE = "revolute"
_CONTINUOUS = "continuous"
_PRISMATIC = "prismatic"
_FIXED = "fixed"


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("joint axis must be non-zero")
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one = 1.0 - c
    return np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ]
    )


def _homogeneous(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


class _Joint:
    __slots__ = ("name", "kind", "parent", "child", "origin", "axis", "mimic")

    def __init__(
        self,
        name: str,
        kind: str,
        parent: str,
        child: str,
        origin: np.ndarray,
        axis: np.ndarray,
        mimic: tuple[str, float, float] | None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.parent = parent
        self.child = child
        self.origin = origin
        self.axis = axis
        self.mimic = mimic


class UrdfChain:
    """Forward kinematics for one parsed URDF document.

    ``forward`` returns every link's world transform for a joint-value mapping.
    Joints missing from the mapping hold their zero position; mimic joints
    follow their source joint.  Mimic chains are resolved in declaration order.
    """

    def __init__(self, document: str) -> None:
        """Parse a URDF document into a chain and index its joints."""
        root = ET.fromstring(document)
        self._joints: dict[str, _Joint] = {}
        parents: dict[str, str] = {}
        for element in root.findall("joint"):
            name = element.get("name")
            kind = element.get("type")
            if not name or not kind:
                raise ValueError("joint requires a name and a type")
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError(f"joint {name} requires parent and child")
            origin = np.eye(4)
            origin_element = element.find("origin")
            if origin_element is not None:
                xyz = [float(v) for v in origin_element.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in origin_element.get("rpy", "0 0 0").split()]
                origin = _homogeneous(_rpy(*rpy), xyz)
            axis_element = element.find("axis")
            axis = np.array(
                [
                    float(v)
                    for v in (
                        axis_element.get("xyz", "1 0 0")
                        if axis_element is not None
                        else "1 0 0"
                    ).split()
                ]
            )
            mimic_element = element.find("mimic")
            mimic = None
            if mimic_element is not None:
                mimic = (
                    str(mimic_element.get("joint")),
                    float(mimic_element.get("multiplier", "1")),
                    float(mimic_element.get("offset", "0")),
                )
            if kind in (_REVOLUTE, _CONTINUOUS, _PRISMATIC, _FIXED):
                self._joints[name] = _Joint(
                    name,
                    kind,
                    parent.get("link", ""),
                    child.get("link", ""),
                    origin,
                    axis,
                    mimic,
                )
                parents[child.get("link", "")] = name
            else:
                raise ValueError(f"joint {name}: unsupported type {kind}")
        children = {joint.child for joint in self._joints.values()}
        roots = [
            link
            for link in {joint.parent for joint in self._joints.values()}
            if link not in children
        ]
        if len(roots) != 1:
            raise ValueError("URDF must describe exactly one kinematic tree")
        self.root = roots[0]
        self._chain: list[tuple[str, str]] = []
        frontier = [(self.root, None)]
        links = {self.root: None}
        while frontier:
            link, joint_name = frontier.pop()
            if joint_name is not None:
                self._chain.append((link, joint_name))
            for joint in self._joints.values():
                if joint.parent == link:
                    links[joint.child] = joint.name
                    frontier.append((joint.child, joint.name))
        self.link_names = tuple(links)

    def forward(self, values: Mapping[str, float]) -> dict[str, np.ndarray]:
        """Return world transforms for every link at ``values``.

        Returns:
            Mapping of link name to 4x4 homogeneous transform.
        """
        resolved = dict(values)
        for joint in self._joints.values():
            if joint.mimic is None or joint.name in resolved:
                continue
            source, multiplier, offset = joint.mimic
            if source in resolved:
                resolved[joint.name] = multiplier * resolved[source] + offset
        poses = {self.root: np.eye(4)}
        for link, joint_name in self._chain:
            joint = self._joints[joint_name]
            parent_pose = poses[joint.parent]
            value = (
            0.0
            if joint.kind == _FIXED
            else float(resolved.get(joint.name, 0.0))
        )
            if joint.kind == _FIXED:
                motion = np.eye(4)
            elif joint.kind == _PRISMATIC:
                motion = np.eye(4)
                motion[:3, 3] = joint.axis * value
            else:
                motion = _homogeneous(_axis_angle(joint.axis, value), (0.0, 0.0, 0.0))
            poses[link] = parent_pose @ joint.origin @ motion
        return poses
