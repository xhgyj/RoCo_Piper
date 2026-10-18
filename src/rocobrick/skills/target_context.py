"""Immutable calibrated target contexts shared by every formal skill.

One :class:`TargetContext` per validated robot carries the robot base frame, the
authorized TCP provenance and the frozen numeric skill parameters read once per
run, plus the optional ground/collision geometry from the same configuration
snapshot.

No ROS, no Isaac, no runtime imports: this module only composes the strict
readers of :mod:`rocobrick.skills.targets` over the authorized run inputs.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rocobrick.feeding_platforms import FeedingPlatform, parse_feeding_platforms
from rocobrick.skills.targets import (
    RobotBaseFrame,
    SkillParams,
    TcpProvenance,
    load_robot_base_frame,
    load_skill_params,
)


@dataclass(frozen=True)
class GroundSpec:
    """Metric collision geometry for the simulated base plate."""

    center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    top_z_m: float
    thickness_m: float


@dataclass(frozen=True)
class TargetContext:
    """Calibration and skill parameters needed for one robot's targets."""

    base: RobotBaseFrame
    provenance: TcpProvenance
    params: SkillParams
    ground: GroundSpec | None = None
    feeding_platforms: tuple[FeedingPlatform, ...] = ()


def load_target_contexts(
    config_path: Path,
    skill_params_path: Path,
    robot_ids: Sequence[str],
    *,
    runtime_model_sha256: str,
    source_sha256: str,
) -> dict[str, TargetContext]:
    """Build immutable target contexts from the authorized run inputs.

    Target generation is intentionally downstream of TCP authorization: both
    provenance digests are required and validated before any target exists.
    The base frame and numeric skill parameters are read once for the run so
    every target is derived from the same configuration snapshot.

    Returns:
        One calibrated target context per validated robot.
    """
    if (
        not isinstance(runtime_model_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", runtime_model_sha256) is None
    ):
        raise ValueError("runtime_model_sha256 must be a lowercase SHA-256 digest")
    if (
        not isinstance(source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    params = load_skill_params(Path(skill_params_path))
    ground: GroundSpec | None = None
    raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if isinstance(raw_config, Mapping):
        task = raw_config.get("Task_Config")
        manipulation = raw_config.get("Manipulation_Config")
        if isinstance(task, Mapping) and isinstance(manipulation, Mapping):
            base_plate = task.get("Base_Plate")
            position = (
                base_plate.get("Position")
                if isinstance(base_plate, Mapping)
                else None
            )
            size = manipulation.get("Ground_Size")
            top = manipulation.get("Ground_Top_Z")
            thickness = manipulation.get("Ground_Thickness")
            if (
                isinstance(position, (list, tuple))
                and len(position) >= 2
                and isinstance(size, (list, tuple))
                and len(size) == 2
            ):
                values = (*position[:2], *size, top, thickness)
                if all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in values
                ):
                    if (
                        float(size[0]) > 0
                        and float(size[1]) > 0
                        and float(thickness) > 0
                    ):
                        ground = GroundSpec(
                            center_xy_m=(float(position[0]), float(position[1])),
                            size_xy_m=(float(size[0]), float(size[1])),
                            top_z_m=float(top),
                            thickness_m=float(thickness),
                        )
    environment = raw_config.get("Env_Config", {})
    configured_robots = raw_config.get("Robot_Config", {}).get("Robots", [])
    platforms = parse_feeding_platforms(
        environment.get("Feeding_Platforms"),
        tuple(robot["Name"] for robot in configured_robots),
    )
    contexts: dict[str, TargetContext] = {}
    for robot_id in robot_ids:
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robot_ids must contain non-empty strings")
        contexts[robot_id] = TargetContext(
            base=load_robot_base_frame(Path(config_path), robot_id),
            provenance=TcpProvenance(
                runtime_model_sha256=runtime_model_sha256,
                source_sha256=source_sha256,
            ),
            params=params,
            ground=ground,
            feeding_platforms=platforms,
        )
    return contexts


__all__ = ["GroundSpec", "TargetContext", "load_target_contexts"]
