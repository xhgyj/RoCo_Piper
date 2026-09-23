"""Pure numeric feedback metrics for live MoveIt execution evidence.

Joint errors and drift, pose errors and at-rest stability over ordinary
mappings/sequences. Every malformed or non-finite input raises ValueError;
nothing is silently clamped. The error formulas mirror the live P1 probes
(max abs joint deviation, Euclidean position error, quaternion angular
error). No ROS imports.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from rocobrick.moveit_contract import ARM_JOINT_NAMES, JOINT_NAMES
from rocobrick.moveit_requests import normalize_quaternion


def _require_finite_joints(
    values: Mapping[str, float], names: Sequence[str], label: str
) -> dict[str, float]:
    """Return the named joints as finite floats; raise ValueError otherwise."""
    resolved: dict[str, float] = {}
    for name in names:
        if name not in values:
            raise ValueError(f"{label} is missing joint {name!r}")
        value = values[name]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label} joint {name!r} is not a finite number")
        resolved[name] = float(value)
    return resolved


def _require_finite_vector(
    values: Sequence[float], length: int, label: str
) -> tuple[float, ...]:
    """Return a fixed-length vector of finite floats; raise ValueError otherwise."""
    if len(values) != length:
        raise ValueError(
            f"{label} must have exactly {length} values, got {len(values)}"
        )
    resolved = []
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label}[{index}] is not a finite number")
        resolved.append(float(value))
    return tuple(resolved)


def max_joint_error(actual: Mapping[str, float], target: Mapping[str, float]) -> float:
    """Largest |actual - target| over the six canonical arm joints."""
    actual_values = _require_finite_joints(actual, ARM_JOINT_NAMES, "actual")
    target_values = _require_finite_joints(target, ARM_JOINT_NAMES, "target")
    return max(
        abs(actual_values[name] - target_values[name]) for name in ARM_JOINT_NAMES
    )


def max_joint_drift(
    actual: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    names: Sequence[str] | None = None,
) -> float:
    """Largest |actual - baseline| over ``names`` (default: all 8 joints)."""
    resolved_names = tuple(names) if names is not None else JOINT_NAMES
    unknown = [name for name in resolved_names if name not in JOINT_NAMES]
    if unknown:
        raise ValueError(f"unknown joint names: {unknown}")
    actual_values = _require_finite_joints(actual, resolved_names, "actual")
    baseline_values = _require_finite_joints(baseline, resolved_names, "baseline")
    return max(
        abs(actual_values[name] - baseline_values[name]) for name in resolved_names
    )


def position_error_m(actual_xyz: Sequence[float], target_xyz: Sequence[float]) -> float:
    """Euclidean distance (m) between two xyz positions."""
    actual = _require_finite_vector(actual_xyz, 3, "actual_xyz")
    target = _require_finite_vector(target_xyz, 3, "target_xyz")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, target)))


def orientation_error_rad(
    actual_xyzw: Sequence[float], target_xyzw: Sequence[float]
) -> float:
    """Rotation angle (rad, in [0, pi]) between two xyzw quaternions."""
    first = normalize_quaternion(_require_finite_vector(actual_xyzw, 4, "actual_xyzw"))
    second = normalize_quaternion(_require_finite_vector(target_xyzw, 4, "target_xyzw"))
    dot = sum(a * b for a, b in zip(first, second))
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(dot))))


def state_is_stable(
    previous: Mapping[str, float],
    current: Mapping[str, float],
    *,
    max_step: float = 5.0e-4,
) -> bool:
    """True when no canonical joint moved more than ``max_step`` rad.

    This is the per-sample at-rest check used before deriving targets from or
    sending goals to the live arm.
    """
    if not isinstance(max_step, (int, float)) or not math.isfinite(float(max_step)):
        raise ValueError(f"max_step must be a finite number, got {max_step}")
    if max_step < 0.0:
        raise ValueError(f"max_step must be non-negative, got {max_step}")
    before = _require_finite_joints(previous, JOINT_NAMES, "previous")
    after = _require_finite_joints(current, JOINT_NAMES, "current")
    return (
        max(abs(after[name] - before[name]) for name in JOINT_NAMES) <= float(max_step)
    )


__all__ = [
    "max_joint_drift",
    "max_joint_error",
    "orientation_error_rad",
    "position_error_m",
    "state_is_stable",
]
