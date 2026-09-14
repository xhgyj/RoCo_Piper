#!/usr/bin/env python3
"""Author the pinned Piper joint limits into a USD layer.

Run this through Isaac Sim Python so the ``pxr`` modules are initialized.
USD angular limits are expressed in degrees.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

LIMITS_RAD = {
    "joint4": (-2.2165681, 2.2165681),
    "joint5": (-1.5620696, 1.5620696),
}


def synchronize(stage, output: Path) -> None:
    from pxr import UsdPhysics

    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))
    editable = type(stage).Open(str(output))
    found: set[str] = set()
    for prim in editable.Traverse():
        limits = LIMITS_RAD.get(prim.GetName())
        if limits is None:
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        joint.CreateLowerLimitAttr(math.degrees(limits[0]))
        joint.CreateUpperLimitAttr(math.degrees(limits[1]))
        found.add(prim.GetName())
    missing = set(LIMITS_RAD) - found
    if missing:
        raise RuntimeError(f"USD is missing Piper joints: {sorted(missing)}")
    editable.GetRootLayer().Save()


def main() -> None:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    from pxr import Usd

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        stage = Usd.Stage.Open(str(args.input))
        if stage is None:
            raise RuntimeError(f"cannot open USD stage {args.input}")
        synchronize(stage, args.output)
    finally:
        app.close()


if __name__ == "__main__":
    main()
