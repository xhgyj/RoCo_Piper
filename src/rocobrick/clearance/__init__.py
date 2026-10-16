"""Simulator-arbiter collision proxy and clearance gating for the dual Piper.

The Isaac host bridge collides the convex hulls of the robot collision meshes
while MoveIt plans against the concave meshes; this package carries the hulls
the simulator actually uses and the checks that measure in that geometry.
"""

from rocobrick.clearance.fk import UrdfChain
from rocobrick.clearance.gate import (
    DEFAULT_FLOOR_M,
    AttachedBody,
    ClearanceReport,
    SimCollisionHulls,
    attachment_box,
    box_hull,
    gate_trajectory,
    region_clearance,
    robot_prefixes,
    sample_trajectory,
    trajectory_clearance,
)
from rocobrick.clearance.hulls import load_hulls

__all__ = [
    "DEFAULT_FLOOR_M",
    "AttachedBody",
    "ClearanceReport",
    "SimCollisionHulls",
    "UrdfChain",
    "attachment_box",
    "box_hull",
    "gate_trajectory",
    "load_hulls",
    "region_clearance",
    "robot_prefixes",
    "sample_trajectory",
    "trajectory_clearance",
]
