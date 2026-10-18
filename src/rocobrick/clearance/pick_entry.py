"""Whole moving-arm/world coverage for handler-owned local PICK paths."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from rocobrick.clearance.fk import UrdfChain
from rocobrick.clearance.gate import sample_trajectory
from rocobrick.clearance.hulls import box_hull, load_hulls, pair_distance


def pick_entry_world_clearance(model, request, trajectory):
    """Check every moving arm hull against all measured world AABBs.

    Returns:
        Coverage evidence. Static base contact is not a moving-link exemption.
    """
    robot = next(iter(request.targets))
    chain, hulls = UrdfChain(model), load_hulls()
    links = [name for name in hulls.link_names if name != "base_link"]
    worlds = {
        **request.scene_snapshot["bricks"],
        **request.scene_snapshot["environment_colliders"],
    }
    if not worlds:
        raise ValueError("PICK entry world coverage missing")
    boxes = [
        (name, box_hull(name, np.asarray(box["aabb_min"]), np.asarray(box["aabb_max"])))
        for name, box in worlds.items()
    ]
    points = [np.asarray(p["positions"]) for p in trajectory["points"]]
    count = 1 + sum(
        max(1, math.ceil(float(np.max(np.abs(b - a))) / 0.002))
        for a, b in zip(points, points[1:])
    )
    if count > 10000:
        raise ValueError("PICK entry world sampling budget exceeded")
    samples = sample_trajectory(
        [p.tolist() for p in points], max_substep_rad=0.002, max_samples=10000
    )
    # Use the same canonical per-robot exclusions as the combined SRDF generator.
    srdf = (
        Path(__file__).resolve().parents[3] / "ros_ws/src/"
        "rocobrick_piper_l_moveit_config/config/piper_l.srdf"
    )
    excluded = {
        frozenset((e.get("link1"), e.get("link2")))
        for e in ET.fromstring(srdf.read_text()).findall("disable_collisions")
    }
    self_pairs = [
        (a, b)
        for i, a in enumerate(hulls.link_names)
        for b in hulls.link_names[i + 1 :]
        if frozenset((a, b)) not in excluded
    ]
    checked = 0
    self_checked = 0
    for index, values in enumerate(samples):
        state = dict(request.start_state)
        state.update(zip(trajectory["joint_names"], values, strict=True))
        poses = chain.forward(state)
        for suffix in links:
            link = f"{robot}_{suffix}"
            if link not in poses:
                raise ValueError(f"PICK entry moving hull missing: {link}")
            for name, world in boxes:
                _, intersects = pair_distance(
                    hulls.hull(suffix), poses[link], world, np.eye(4)
                )
                checked += 1
                if intersects:
                    raise ValueError(
                        f"PICK_ENTRY_WORLD_COLLISION: {link} vs {name} at {index}"
                    )
        for a, b in self_pairs:
            first, second = f"{robot}_{a}", f"{robot}_{b}"
            if first not in poses or second not in poses:
                raise ValueError("PICK entry self-collision geometry missing")
            _, intersects = pair_distance(
                hulls.hull(a), poses[first], hulls.hull(b), poses[second]
            )
            self_checked += 1
            if intersects:
                raise ValueError(f"PICK_ENTRY_SELF_COLLISION: {a} vs {b} at {index}")
    return {
        "self_checked_pairs": self_checked,
        "self_collision_exclusions_source": str(srdf),
        "covered_links": links,
        "world_bodies": list(worlds),
        "samples": len(samples),
        "checked_pairs": checked,
        "contact_exemptions": [],
        "static_base_excluded": True,
    }
