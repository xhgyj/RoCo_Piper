"""Trajectory-level clearance gating in the simulator's collision geometry.

The gate replays a commanded joint trajectory -- the very points MoveIt
returned -- through the arbiter hulls and the model's own kinematics, and
fails closed when the two robots would come closer than the floor.  It runs
between planning and dispatch, so a plan that MoveIt's concave-mesh check
accepted but the simulator's convex-hull colliders cannot execute is refused
before any motion is sent.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from rocobrick.clearance.fk import UrdfChain
from rocobrick.clearance.hulls import (
    DEFAULT_FLOOR_M,
    AttachedBody,
    Clearance,
    ClearanceReport,
    SimCollisionHulls,
    box_hull,
    load_hulls,
    region_clearance,
)

DEFAULT_MAX_SUBSTEP_RAD = 0.02
DEFAULT_MAX_SAMPLES = 600
_ARM_JOINT = re.compile(r"(?P<robot>.+)_joint(?P<index>[1-6])$")


def robot_prefixes(joint_names: Iterable[str]) -> tuple[str, ...]:
    """Return every robot prefix that owns a complete joint1..joint6 set.

    Returns:
        Sorted robot ids such as ``("piper_0", "piper_1")``.
    """
    names = set(joint_names)
    prefixes = set()
    for name in names:
        match = _ARM_JOINT.match(name)
        if match is None:
            continue
        robot = match.group("robot")
        if all(f"{robot}_joint{index}" in names for index in range(1, 7)):
            prefixes.add(robot)
    return tuple(sorted(prefixes))


def attachment_box(
    name: str,
    link: str,
    minimum: Sequence[float],
    maximum: Sequence[float],
) -> AttachedBody:
    """Return a box rigidly attached to one model link.

    The box spans ``minimum``..``maximum`` in the link frame.

    Returns:
        The attached body carrying an exact box hull.
    """
    return AttachedBody(
        name=name,
        link=link,
        local=np.eye(4),
        hull=box_hull(
            name,
            np.asarray(minimum, dtype=np.float64),
            np.asarray(maximum, dtype=np.float64),
        ),
    )


def sample_trajectory(
    points: Sequence[Sequence[float]],
    *,
    max_substep_rad: float = DEFAULT_MAX_SUBSTEP_RAD,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> list[np.ndarray]:
    """Sample a waypoint trajectory linearly in joint space.

    Sampling mirrors the interpolation the trajectory controller executes
    between waypoints; each substep moves no joint by more than
    ``max_substep_rad`` (bounded per segment), and the total is capped at
    ``max_samples``.

    Returns:
        The sampled joint vectors, endpoints included.
    """
    if not points:
        return []
    arrays = [np.asarray(point, dtype=np.float64) for point in points]
    if len(arrays) == 1:
        return [arrays[0]]
    counts: list[int] = []
    for index in range(len(arrays) - 1):
        delta = float(np.max(np.abs(arrays[index + 1] - arrays[index])))
        counts.append(
            1 if delta <= 0.0 else max(1, int(math.ceil(delta / max_substep_rad)))
        )
    budget = max_samples - 1
    if budget < len(counts):
        raise ValueError("trajectory has more segments than the sample budget")
    while sum(counts) > budget:
        largest = max(range(len(counts)), key=lambda index: counts[index])
        if counts[largest] <= 1:
            break
        counts[largest] -= 1
    samples = [arrays[0]]
    for index, count in enumerate(counts):
        start, end = arrays[index], arrays[index + 1]
        for step in range(1, count + 1):
            ratio = step / float(count)
            samples.append(start * (1.0 - ratio) + end * ratio)
    return samples


def trajectory_clearance(
    model_document: str,
    samples: Sequence[np.ndarray],
    joint_names: Sequence[str],
    *,
    static_joint_values: Mapping[str, float] | None = None,
    attached: dict[str, list[AttachedBody]] | None = None,
    floor_m: float = DEFAULT_FLOOR_M,
    hulls: SimCollisionHulls | None = None,
    robots: Sequence[str] | None = None,
) -> ClearanceReport:
    """Return the worst cross-robot clearance over sampled joint vectors.

    ``robots`` names the participants whose relative geometry is checked; it
    defaults to every robot prefix the trajectory's joint names identify.  Pass
    it explicitly when the trajectory commands only a subset of the joints (for
    example a gripper), so the parked peer still participates from
    ``static_joint_values``.
    """
    chain = UrdfChain(model_document)
    bound = tuple(robots) if robots is not None else robot_prefixes(joint_names)
    if not bound:
        raise ValueError("trajectory joint names do not bind any robot pair set")
    static = dict(static_joint_values or {})
    proxy = hulls if hulls is not None else load_hulls()
    worst: Clearance | None = None
    covered: dict[str, int] = {}
    for index, values in enumerate(samples):
        joint_values: dict[str, dict[str, float]] = {}
        for robot in bound:
            mapping = {
                name: float(value)
                for name, value in static.items()
                if name.startswith(f"{robot}_")
            }
            for name, value in zip(joint_names, values, strict=True):
                if name.startswith(f"{robot}_"):
                    mapping[name] = float(value)
            joint_values[robot] = mapping
        poses = {
            robot: {
                link: pose
                for link, pose in chain.forward(mapping).items()
                if link.startswith(f"{robot}_")
            }
            for robot, mapping in joint_values.items()
        }
        report = region_clearance(
            poses, hulls=proxy, attached=attached, floor_m=floor_m, sample=index
        )
        if report.covered:
            covered = report.covered
        if report.worst is None:
            continue
        if worst is None or report.worst.distance_m < worst.distance_m:
            worst = report.worst
    if worst is None:
        # No sampled configuration brought any pair inside the prefilter.
        return ClearanceReport(
            passed=True,
            floor_m=floor_m,
            worst=None,
            samples=len(samples),
            covered=covered,
            excluded=proxy.disabled_names,
        )
    return ClearanceReport(
        passed=worst.distance_m >= floor_m,
        floor_m=floor_m,
        worst=worst,
        samples=len(samples),
        covered=covered,
        excluded=proxy.disabled_names,
    )


def gate_trajectory(
    model_document: str,
    joint_names: Sequence[str],
    points: Sequence[Sequence[float]],
    *,
    static_joint_values: Mapping[str, float] | None = None,
    attached: dict[str, list[AttachedBody]] | None = None,
    floor_m: float = DEFAULT_FLOOR_M,
    max_substep_rad: float = DEFAULT_MAX_SUBSTEP_RAD,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    hulls: SimCollisionHulls | None = None,
    robots: Sequence[str] | None = None,
) -> ClearanceReport:
    """Replay a commanded trajectory and report the worst clearance.

    Returns:
        The verdict over the whole trajectory, with the worst pair and sample.
    """
    samples = sample_trajectory(
        points, max_substep_rad=max_substep_rad, max_samples=max_samples
    )
    return trajectory_clearance(
        model_document,
        samples,
        joint_names,
        static_joint_values=static_joint_values,
        attached=attached,
        floor_m=floor_m,
        hulls=hulls,
        robots=robots,
    )


__all__ = [
    "DEFAULT_FLOOR_M",
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_MAX_SUBSTEP_RAD",
    "AttachedBody",
    "ClearanceReport",
    "SimCollisionHulls",
    "attachment_box",
    "box_hull",
    "gate_trajectory",
    "region_clearance",
    "robot_prefixes",
    "sample_trajectory",
    "trajectory_clearance",
]
