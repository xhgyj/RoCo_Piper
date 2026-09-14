"""Greenfield, load-only scene bootstrap and initial-state verification.

The bootstrap owns no simulation objects after construction.  It asks a concrete
BrickSim adapter to create assets, then turns the measured result into the first
immutable execution snapshot.  No motion command is legal during this phase.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time
from typing import Protocol

from rocobrick.models import (
    AssemblyPlan,
    ExecutionSnapshot,
    Pose,
    Revisions,
    RobotSnapshot,
    SceneManifest,
)


class SceneBootstrapError(RuntimeError):
    """The physical scene does not match the validated run inputs."""


@dataclass(frozen=True)
class RobotSpawnSpec:
    robot_id: str
    prim_path: str
    usd_path: str
    base_pose: Pose
    arm_joint_names: tuple[str, ...]
    sensor_prim_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedRobot:
    robot_id: str
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]


@dataclass(frozen=True)
class LoadedPart:
    part_instance_id: str
    prim_path: str
    brick_id: int
    measured_pose: Pose


@dataclass(frozen=True)
class LoadedScene:
    robots: Mapping[str, LoadedRobot]
    parts: Mapping[str, LoadedPart]
    execution_snapshot: ExecutionSnapshot


class LoadOnlyScenePort(Protocol):
    """Narrow BrickSim boundary used before the execution layer is enabled."""

    async def open(self) -> None: ...

    async def spawn_robot(self, spec: RobotSpawnSpec) -> LoadedRobot: ...

    async def spawn_part(self, part_instance_id: str, brick_id: int, pose: Pose) -> LoadedPart: ...

    async def settle_and_measure(self) -> None: ...

    def measured_parts(self) -> Mapping[str, LoadedPart]: ...


def _pose_error(expected: Pose, measured: Pose) -> tuple[float, float]:
    if expected.frame_id != measured.frame_id:
        return math.inf, math.inf
    position = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(expected.position, measured.position))
    )
    dot = abs(sum(a * b for a, b in zip(expected.orientation_xyzw, measured.orientation_xyzw)))
    dot = min(1.0, max(-1.0, dot))
    angle = 2.0 * math.acos(dot)
    return position, angle


async def bootstrap_scene(
    port: LoadOnlyScenePort,
    plan: AssemblyPlan,
    manifest: SceneManifest,
    robot_specs: Mapping[str, RobotSpawnSpec],
    *,
    position_tolerance_m: float = 0.002,
    orientation_tolerance_rad: float = 0.035,
    monotonic=time.monotonic,
) -> LoadedScene:
    """Load and verify a scene without issuing any articulation action."""
    required_robots = {
        robot
        for task in plan.tasks.values()
        for robot in (task.primary_robot, task.supporting_robot)
        if robot is not None
    }
    if set(robot_specs) != required_robots:
        raise SceneBootstrapError(
            "robot configuration must exactly match robots referenced by the plan: "
            f"expected {sorted(required_robots)}, got {sorted(robot_specs)}"
        )
    missing_parts = {
        task.part_instance_id for task in plan.tasks.values()
    } - set(manifest.parts)
    if missing_parts:
        raise SceneBootstrapError(
            f"scene is missing planned physical instances: {sorted(missing_parts)}"
        )

    await port.open()
    robots: dict[str, LoadedRobot] = {}
    for robot_id in sorted(robot_specs):
        spec = robot_specs[robot_id]
        if spec.robot_id != robot_id or not spec.arm_joint_names:
            raise SceneBootstrapError(f"invalid robot specification for {robot_id}")
        loaded = await port.spawn_robot(spec)
        if loaded.robot_id != robot_id:
            raise SceneBootstrapError(f"adapter returned wrong robot id for {robot_id}")
        if loaded.joint_names != spec.arm_joint_names:
            raise SceneBootstrapError(f"measured joint order differs for {robot_id}")
        if len(loaded.joint_positions) != len(loaded.joint_names) or not all(
            math.isfinite(value) for value in loaded.joint_positions
        ):
            raise SceneBootstrapError(f"invalid measured joints for {robot_id}")
        robots[robot_id] = loaded

    parts: dict[str, LoadedPart] = {}
    for instance_id in sorted(manifest.parts):
        expected = manifest.parts[instance_id]
        loaded = await port.spawn_part(instance_id, expected.brick_id, expected.pose)
        if loaded.part_instance_id != instance_id or loaded.brick_id != expected.brick_id:
            raise SceneBootstrapError(f"adapter returned wrong part binding for {instance_id}")
        if not loaded.prim_path:
            raise SceneBootstrapError(f"part {instance_id} has no physical prim binding")
        parts[instance_id] = loaded

    await port.settle_and_measure()
    measured_parts = dict(port.measured_parts())
    if set(measured_parts) != set(parts):
        raise SceneBootstrapError("measured part set differs from spawned part bindings")
    for instance_id, loaded in measured_parts.items():
        expected = manifest.parts[instance_id]
        position_error, orientation_error = _pose_error(expected.pose, loaded.measured_pose)
        if position_error > position_tolerance_m or orientation_error > orientation_tolerance_rad:
            raise SceneBootstrapError(
                f"initial pose mismatch for {instance_id}: position={position_error:.6f} m, "
                f"orientation={orientation_error:.6f} rad"
            )

    sampled_at = monotonic()
    snapshot = ExecutionSnapshot(
        revisions=Revisions(world=1, robot_state=1, motion=0),
        robots={
            robot_id: RobotSnapshot(
                robot_id=robot_id,
                joints=robot.joint_positions,
                feedback_monotonic_s=sampled_at,
            )
            for robot_id, robot in robots.items()
        },
    )
    return LoadedScene(robots=robots, parts=measured_parts, execution_snapshot=snapshot)
