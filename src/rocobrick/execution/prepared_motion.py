"""Frozen prepared-trajectory admission; execution never replans its goal.

Snapshot SHA binds the planning input. Live scene comparison deliberately
ignores advancing query sequence/time/contact buffers, and instead checks
freshness, complete measured joints, model, object poses and environment.
Hash identity is not itself collision proof or physical custody evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields

from rocobrick.coordinated_moveit import (
    CoordinatedOperation,
    CoordinatedRobotTrajectory,
    CoordinatedTrajectoryPoint,
)
from rocobrick.skills.handover_profile import digest, finite
from rocobrick.skills.observation import _pose, _vector
from rocobrick.skills.placement_geometry import pose_error

PREPARED_MOTION_SCHEMA = "rocobrick-prepared-motion/v1"


def _hash(raw: object) -> str:
    return hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SynchronizationPolicy:
    """Actual controller-clock and tracking evidence required for admission."""

    max_start_skew_s: float
    max_tracking_error_rad: float
    max_feedback_gap_s: float
    min_feedback_samples: int = 3

    def __post_init__(self) -> None:
        """Reject unknown, nonfinite or unlimited synchronization budgets."""
        for name in (
            "max_start_skew_s",
            "max_tracking_error_rad",
            "max_feedback_gap_s",
        ):
            finite(getattr(self, name), name, positive=True)
        if (
            self.max_start_skew_s > 0.02
            or self.max_tracking_error_rad > 0.01
            or (self.max_feedback_gap_s > 0.2)
        ):
            raise ValueError("prepared execution synchronization budget is too broad")
        if (
            type(self.min_feedback_samples) is not int
            or not 3 <= self.min_feedback_samples <= 64
        ):
            raise ValueError(
                "prepared execution requires a bounded feedback sample count"
            )


@dataclass(frozen=True)
class PreparedMotionRequest:
    """One verified trajectory and its complete post-lift scene/start binding."""

    operation: CoordinatedOperation
    robots: tuple[str, ...]
    scene_snapshot: Mapping[str, object]
    scene_version: int
    start_state: Mapping[str, float]
    policy: SynchronizationPolicy
    timeout_s: float
    max_scene_age_s: float = 0.5
    start_tolerance_rad: float = 0.002
    scene_position_tolerance_m: float = 0.001
    scene_orientation_tolerance_rad: float = 0.02
    _admitted_sha256: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        """Require an exact two-arm trajectory and immutable JSON inputs."""
        if not isinstance(self.operation, CoordinatedOperation) or not isinstance(
            self.policy, SynchronizationPolicy
        ):
            raise ValueError("prepared motion requires typed trajectory and policy")
        if self.robots != ("piper_0", "piper_1") or set(self.participants) != set(
            self.robots
        ):
            raise ValueError(
                "prepared HANDOVER motion must cover both complete Piper arms"
            )
        if self.operation.scene_digest != self.scene_snapshot.get("scene_digest"):
            raise ValueError("prepared trajectory belongs to another captured snapshot")
        digest(self.operation.scene_digest, "prepared snapshot")
        if type(self.scene_version) is not int or self.scene_version < 0:
            raise ValueError("prepared scene version is invalid")
        for name in (
            "timeout_s",
            "max_scene_age_s",
            "start_tolerance_rad",
            "scene_position_tolerance_m",
            "scene_orientation_tolerance_rad",
        ):
            finite(getattr(self, name), name, positive=True)
        if (
            self.max_scene_age_s > 0.5
            or self.start_tolerance_rad > 0.01
            or (
                self.scene_position_tolerance_m > 0.005
                or self.scene_orientation_tolerance_rad > 0.03
            )
        ):
            raise ValueError("prepared motion drift budget is too broad")
        if self.operation.synchronization_tolerance_s > self.policy.max_start_skew_s:
            raise ValueError(
                "trajectory synchronization budget exceeds execution policy"
            )
        snapshot = json.loads(json.dumps(self.scene_snapshot, allow_nan=False))
        object.__setattr__(self, "scene_snapshot", snapshot)
        object.__setattr__(self, "start_state", dict(self.start_state))
        object.__setattr__(self, "_admitted_sha256", _hash(self._payload()))
        self._validate_trajectory()

    def _payload(self) -> dict[str, object]:
        raw = asdict(self)
        raw.pop("_admitted_sha256", None)
        return raw

    @property
    def coordination_id(self) -> str:
        """Return the admitted common motion identity."""
        return self.operation.coordination_id

    @property
    def participants(self) -> tuple[str, ...]:
        """Return both exact trajectory participants."""
        return self.operation.participants

    @property
    def binding_sha256(self) -> str:
        """Return the trajectory/scene/policy/start digest, not a physical proof."""
        current = _hash(self._payload())
        if current != self._admitted_sha256:
            raise ValueError("prepared request mutated after admission")
        return current

    def _validate_trajectory(self) -> None:
        paths = self.operation.trajectories
        if len(paths) != 2 or len({t.robot_id for t in paths}) != 2:
            raise ValueError("prepared motion requires exactly one trajectory per arm")
        grids = []
        for path in paths:
            expected = tuple(f"{path.robot_id}_joint{i}" for i in range(1, 7))
            if path.joint_names != expected:
                raise ValueError(
                    "prepared motion requires canonical six-joint wire order"
                )
            grids.append(tuple(p.time_from_start_s for p in path.points))
            for name, position in zip(path.joint_names, path.points[0].positions):
                if abs(
                    finite(self.start_state.get(name), "prepared start joint")
                    - position
                ) > (self.start_tolerance_rad):
                    raise ValueError(
                        "prepared trajectory does not start at captured joints"
                    )
        if grids[0] != grids[1] or grids[0][-1] >= self.timeout_s:
            raise ValueError("prepared motion requires one bounded common time grid")
        for name, value in self.start_state.items():
            if not isinstance(name, str) or not name:
                raise ValueError("prepared start joint identity is invalid")
            finite(value, "prepared start joint")

    def validate_live(
        self,
        snapshot: Mapping[str, object],
        version: int,
        joints: Mapping[str, float],
        observed_sim_time_s: float,
    ) -> None:
        """Refuse stale state or real scene drift, not an advancing query counter."""
        self.binding_sha256
        time = finite(snapshot.get("sim_time_s"), "live snapshot time")
        age = finite(observed_sim_time_s, "live observed time") - time
        if not 0 <= age <= self.max_scene_age_s:
            raise ValueError("prepared live scene is stale or from a future clock")
        original_time = finite(
            self.scene_snapshot.get("sim_time_s"), "captured scene time"
        )
        if (
            time < original_time
            or type(version) is not int
            or version != self.scene_version
        ):
            raise ValueError("prepared scene clock/version changed")
        for key in ("schema", "session_id", "manifest_sha256", "tcp_provenance"):
            if (
                snapshot.get(key) != self.scene_snapshot.get(key)
                or snapshot.get(key) is None
            ):
                raise ValueError("prepared scene identity/model/calibration changed")
        if not self.scene_snapshot.get("environment_colliders") or (
            snapshot.get("environment_colliders")
            != self.scene_snapshot["environment_colliders"]
        ):
            raise ValueError("prepared environment geometry changed or is unavailable")
        if set(joints) != set(self.start_state):
            raise ValueError("prepared live joint coverage changed")
        for name, expected in self.start_state.items():
            if (
                abs(finite(joints[name], "live joint") - expected)
                > self.start_tolerance_rad
            ):
                raise ValueError(f"prepared live start drifted: {name}")
        for key in ("robots", "bricks"):
            old, new = self.scene_snapshot.get(key), snapshot.get(key)
            if (
                not isinstance(old, Mapping)
                or not isinstance(new, Mapping)
                or set(old) != set(new)
            ):
                raise ValueError(
                    "prepared complete scene object/robot coverage changed"
                )
            for identity, before in old.items():
                after = new[identity]
                if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                    raise ValueError("prepared live scene block is malformed")
                pose_key = "tcp_pose" if key == "robots" else "pose"
                position, rotation = pose_error(
                    _pose(before.get(pose_key), "captured pose"),
                    _pose(after.get(pose_key), "live pose"),
                )
                if (
                    position > self.scene_position_tolerance_m
                    or rotation > self.scene_orientation_tolerance_rad
                ):
                    raise ValueError(f"prepared live geometry drifted: {identity}")
                if key == "bricks":
                    for bounds in ("aabb_min", "aabb_max"):
                        if (
                            max(
                                abs(a - b)
                                for a, b in zip(
                                    _vector(before.get(bounds), 3, "captured bounds"),
                                    _vector(after.get(bounds), 3, "live bounds"),
                                )
                            )
                            > self.scene_position_tolerance_m
                        ):
                            raise ValueError("prepared collision envelope drifted")

    def serialize(self) -> dict[str, object]:
        """Return an exact-key mailbox envelope with a complete immutable binding."""
        return {
            "schema": PREPARED_MOTION_SCHEMA,
            **self._payload(),
            "binding_sha256": self.binding_sha256,
        }

    @classmethod
    def parse(cls, raw: object) -> PreparedMotionRequest:
        """Parse the exact prepared plan without inferring missing evidence.

        Returns:
            A fully validated request with recomputed binding.
        """
        expected = {f.name for f in fields(cls) if f.init} | {
            "schema",
            "binding_sha256",
        }
        if (
            not isinstance(raw, Mapping)
            or set(raw) != expected
            or raw.get("schema") != PREPARED_MOTION_SCHEMA
        ):
            raise ValueError("prepared motion envelope is incomplete or unknown")
        operation = raw["operation"]
        if not isinstance(operation, Mapping) or set(operation) != {
            f.name for f in fields(CoordinatedOperation)
        }:
            raise ValueError("prepared operation keys are incomplete or unknown")
        paths = []
        for path in operation["trajectories"]:
            if not isinstance(path, Mapping) or set(path) != {
                f.name for f in fields(CoordinatedRobotTrajectory)
            }:
                raise ValueError("prepared trajectory keys are incomplete or unknown")
            points = []
            for point in path["points"]:
                if not isinstance(point, Mapping) or set(point) != {
                    f.name for f in fields(CoordinatedTrajectoryPoint)
                }:
                    raise ValueError("prepared waypoint keys are incomplete or unknown")
                points.append(CoordinatedTrajectoryPoint(**point))
            paths.append(
                CoordinatedRobotTrajectory(
                    path["robot_id"], tuple(path["joint_names"]), tuple(points)
                )
            )
        op = CoordinatedOperation(**{**operation, "trajectories": tuple(paths)})
        policy = raw["policy"]
        if not isinstance(policy, Mapping) or set(policy) != {
            f.name for f in fields(SynchronizationPolicy)
        }:
            raise ValueError("prepared synchronization policy keys are invalid")
        value = cls(
            **{
                **{
                    n: raw[n]
                    for n in expected
                    - {"schema", "binding_sha256", "operation", "policy", "robots"}
                },
                "operation": op,
                "policy": SynchronizationPolicy(**policy),
                "robots": tuple(raw["robots"]),
            }
        )
        if value.binding_sha256 != raw["binding_sha256"]:
            raise ValueError("prepared trajectory binding mismatch")
        return value
