"""Configuration contracts for one planner-driven BrickSim episode."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    items = _sequence(value, name)
    if len(items) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(items))


@dataclass(frozen=True)
class RobotPoseOverride:
    """Static robot base pose applied before the simulator starts."""

    robot_id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


@dataclass(frozen=True)
class ArmStagingConfig:
    """Initial loose-part pickup slots for one planner-visible arm."""

    robot_id: str
    parking_slots: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class StagingConfig:
    """Deterministic loose-part staging used by one episode."""

    initial_yaw_degrees: float
    settle_steps: int
    workspace_size: tuple[float, float, float]
    workspace_position: tuple[float, float, float]
    arms: tuple[ArmStagingConfig, ...]

    def arm(self, robot_id: str) -> ArmStagingConfig:
        """Return staging for one arm.

        Returns:
            Matching immutable staging configuration.
        """
        for arm in self.arms:
            if arm.robot_id == robot_id:
                return arm
        raise ValueError(f"episode has no staging configuration for {robot_id}")


@dataclass(frozen=True)
class EpisodeConfig:
    """Fully validated static input to planning and simulation."""

    episode_id: str
    source_path: Path
    system_config_path: Path
    user_config_path: Path
    initial_structure_path: Path
    goal_structure_path: Path
    available_arm_ids: tuple[str, ...]
    robot_overrides: tuple[RobotPoseOverride, ...]
    staging: StagingConfig
    shared_resource: str
    allow_prefetch: bool
    max_retries: int

    @classmethod
    def load(cls, path: str | Path) -> EpisodeConfig:
        """Load one episode relative to its containing directory.

        Returns:
            Validated episode configuration with absolute referenced paths.
        """
        source = Path(path).resolve()
        data = _mapping(json.loads(source.read_text(encoding="utf-8")), "episode")
        if data.get("schema") != "rocobrick/episode@1":
            raise ValueError("episode schema must be rocobrick/episode@1")
        root = source.parent
        scene = _mapping(data.get("scene"), "scene")
        task = _mapping(data.get("task"), "task")
        arms = tuple(
            _string(item, f"available_arms[{index}]")
            for index, item in enumerate(
                _sequence(data.get("available_arms"), "available_arms")
            )
        )
        if not arms or len(set(arms)) != len(arms):
            raise ValueError("available_arms must be non-empty and unique")

        robot_overrides = cls._robot_overrides(data.get("robots"), arms)
        staging = cls._staging(data.get("staging"), arms)
        execution = _mapping(data.get("execution"), "execution")
        retries = execution.get("max_retries", 0)
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("execution.max_retries must be a non-negative integer")
        allow_prefetch = execution.get("allow_prefetch", False)
        if not isinstance(allow_prefetch, bool):
            raise ValueError("execution.allow_prefetch must be a boolean")

        result = cls(
            episode_id=_string(data.get("episode_id"), "episode_id"),
            source_path=source,
            system_config_path=cls._path(
                root, scene.get("system_config"), "system_config"
            ),
            user_config_path=cls._path(root, scene.get("user_config"), "user_config"),
            initial_structure_path=cls._path(
                root, task.get("initial_structure"), "initial_structure"
            ),
            goal_structure_path=cls._path(
                root, task.get("goal_structure"), "goal_structure"
            ),
            available_arm_ids=arms,
            robot_overrides=robot_overrides,
            staging=staging,
            shared_resource=_string(
                execution.get("shared_resource"), "execution.shared_resource"
            ),
            allow_prefetch=allow_prefetch,
            max_retries=retries,
        )
        result._validate_files()
        return result

    @staticmethod
    def _path(root: Path, value: object, name: str) -> Path:
        return (root / _string(value, name)).resolve()

    @staticmethod
    def _robot_overrides(
        value: object, available_arms: tuple[str, ...]
    ) -> tuple[RobotPoseOverride, ...]:
        records = _mapping(value, "robots")
        unknown = set(records) - set(available_arms)
        if unknown:
            raise ValueError(
                "robot overrides contain unavailable arms: "
                f"{sorted(unknown)}"
            )
        result = []
        for robot_id, raw in records.items():
            record = _mapping(raw, f"robots.{robot_id}")
            result.append(
                RobotPoseOverride(
                    robot_id,
                    _vector(
                        record.get("base_position"),
                        3,
                        f"robots.{robot_id}.base_position",
                    ),
                    _vector(
                        record.get("base_orientation"),
                        4,
                        f"robots.{robot_id}.base_orientation",
                    ),
                )
            )
        return tuple(result)

    @staticmethod
    def _staging(value: object, available_arms: tuple[str, ...]) -> StagingConfig:
        record = _mapping(value, "staging")
        arm_records = _mapping(record.get("arms"), "staging.arms")
        if set(arm_records) != set(available_arms):
            raise ValueError("staging.arms must exactly match available_arms")
        arms = []
        for robot_id in available_arms:
            arm = _mapping(arm_records[robot_id], f"staging.arms.{robot_id}")
            slots = tuple(
                _vector(item, 2, f"staging.arms.{robot_id}.parking_slots[{index}]")
                for index, item in enumerate(
                    _sequence(
                        arm.get("parking_slots"),
                        f"staging.arms.{robot_id}.parking_slots",
                    )
                )
            )
            if not slots:
                raise ValueError(f"staging arm {robot_id} requires parking slots")
            arms.append(
                ArmStagingConfig(
                    robot_id,
                    slots,
                )
            )
        settle_steps = record.get("settle_steps", 30)
        if not isinstance(settle_steps, int) or settle_steps <= 0:
            raise ValueError("staging.settle_steps must be a positive integer")
        return StagingConfig(
            _number(record.get("initial_yaw_degrees"), "staging.initial_yaw_degrees"),
            settle_steps,
            _vector(record.get("workspace_size"), 3, "staging.workspace_size"),
            _vector(record.get("workspace_position"), 3, "staging.workspace_position"),
            tuple(arms),
        )

    def _validate_files(self) -> None:
        for path in (
            self.system_config_path,
            self.user_config_path,
            self.initial_structure_path,
            self.goal_structure_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
