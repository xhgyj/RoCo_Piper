"""Precontact gripper/world hull gate; no final-grasp contact exemptions."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from rocobrick.clearance.fk import UrdfChain
from rocobrick.clearance.gate import sample_trajectory
from rocobrick.clearance.hulls import box_hull, load_hulls, pair_distance


def grasp_approach_clearance(model: str, request, trajectory: Mapping) -> dict:
    """Check region-approach jaws/palm against target and feeding platforms.

    This is deliberately stage-specific: normal insertion/close requests do
    not carry a grasp region, and their existing contact policy is unchanged.
    Positive margins reuse the region's existing floor/contact thresholds.

    Returns:
        Bounded geometric coverage evidence; unsafe trajectories raise.
    """
    regions = {
        robot: target.get("grasp_region", target.get("grasp_approach_region"))
        for robot, target in request.targets.items()
        if isinstance(target, Mapping) and ("grasp_region" in target or "grasp_approach_region" in target)
    }
    if not regions:
        return {"applicable": False}
    chain = UrdfChain(model)
    proxy = load_hulls()
    points = [np.asarray(p["positions"], dtype=float) for p in trajectory["points"]]
    required_samples = 1 + sum(
        max(1, math.ceil(float(np.max(np.abs(b - a))) / 0.002))
        for a, b in zip(points, points[1:])
    )
    if required_samples > 10000:
        raise ValueError("grasp approach dense sampling budget exceeded")
    samples = sample_trajectory(
        [p["positions"] for p in trajectory["points"]],
        max_substep_rad=0.002,
        max_samples=10000,
    )
    worst = None
    checked = 0
    for robot, region in regions.items():
        links = [
            f"{robot}_{name}"
            for name in ("gripper_base", "gripper_link1", "gripper_link2")
        ]
        # Lightweight test descriptions carry no physical robot bodies.
        if not any(link in chain.link_names for link in links):
            if f"{robot}_base_link" in chain.link_names:
                raise ValueError(f"grasp approach gripper hulls missing: {robot}")
            continue
        snapshot = request.scene_snapshot
        identity = region["physical_instance_id"]
        brick = snapshot["bricks"][identity]
        worlds = [(identity, brick, region["contact_margin_m"])]
        for name, box in snapshot.get("environment_colliders", {}).items():
            if name.startswith("/World/feeding_platforms/"):
                worlds.append((name, box, region["floor_margin_m"]))
        if region["ground_top_world_m"] > 0.001 and not any(
            abs(box["aabb_max"][2] - region["ground_top_world_m"]) <= 0.001
            for _, box, _ in worlds[1:]
        ):
            raise ValueError("grasp approach support collider missing from live scene")
        hulls = [
            (
                name,
                box_hull(name, np.array(box["aabb_min"]), np.array(box["aabb_max"])),
                margin,
            )
            for name, box, margin in worlds
        ]
        for index, point in enumerate(samples):
            values = dict(request.start_state or {})
            values.update(zip(trajectory["joint_names"], point, strict=True))
            poses = chain.forward(values)
            for link in links:
                if link not in poses:
                    raise ValueError(f"grasp approach hull coverage missing: {link}")
                hull = proxy.hull(link[len(robot) + 1 :])
                for name, world_hull, margin in hulls:
                    distance, intersects = pair_distance(
                        hull, poses[link], world_hull, np.eye(4)
                    )
                    checked += 1
                    record = {
                        "robot_link": link,
                        "world_body": name,
                        "sample": index,
                        "distance_m": float(distance),
                        "intersecting": bool(intersects),
                        "required_margin_m": margin,
                    }
                    if worst is None or distance - margin < (
                        worst["distance_m"] - worst["required_margin_m"]
                    ):
                        worst = record
                    if intersects or distance < margin:
                        raise ValueError(
                            f"GRASP_APPROACH_WORLD_CLEARANCE: {link} vs {name} "
                            f"at sample {index}: {distance:.6f} m < {margin:.6f} m"
                        )
    return {
        "applicable": True,
        "checked_pairs": checked,
        "samples": len(samples),
        "worst": worst,
        "contact_exemptions": [],
        "arbiter": "isaac-convex-hull-proxy/v1",
    }
