#!/usr/bin/env python3
"""Fail fast when host Piper assets drift from the pinned planning model."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


JOINT_LIMITS = {
    "joint4": (-2.2165681, 2.2165681),
    "joint5": (-1.5620696, 1.5620696),
}


def limits(path: Path) -> dict[str, tuple[float, float]]:
    robot = ET.parse(path).getroot()
    result: dict[str, tuple[float, float]] = {}
    for joint in robot.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and joint.get("name") in JOINT_LIMITS:
            result[joint.get("name", "")] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
    return result


def validate(host_urdf: Path, official_urdf: Path) -> None:
    host = limits(host_urdf)
    official = limits(official_urdf)
    for joint, expected in JOINT_LIMITS.items():
        if official.get(joint) != expected:
            raise RuntimeError(
                f"pinned official asset has unexpected {joint} limits: {official.get(joint)}"
            )
        if host.get(joint) != expected:
            raise RuntimeError(f"host URDF has stale {joint} limits: {host.get(joint)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-urdf", type=Path, required=True)
    parser.add_argument("--official-urdf", type=Path, required=True)
    args = parser.parse_args()
    validate(args.host_urdf, args.official_urdf)


if __name__ == "__main__":
    main()
