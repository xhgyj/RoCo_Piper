#!/usr/bin/env python3
"""Fail fast when the runtime Piper URDF drifts from the execution model.

The runtime URDF must remain byte-identical to the pinned agx_arm_sim URDF.
The runtime USD and sublayers are checked separately by the P0.4 model audit.
The agx_arm_urdf planning model carries wider joint4/joint5 limits than the
agx_arm_sim URDF/USD, so the narrow intersection applies to execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


JOINT_LIMITS = {
    "joint4": (-1.7453292, 1.7453292),
    "joint5": (-1.2217304, 1.2217304),
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
    if host_urdf.read_bytes() != official_urdf.read_bytes():
        raise RuntimeError(
            f"host URDF is not byte-identical to the pinned agx_arm_sim URDF: "
            f"{host_urdf} vs {official_urdf}"
        )
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
