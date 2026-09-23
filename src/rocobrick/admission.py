"""Pure caller-side admission for planned waypoints and MoveGroup goals.

Strict validation BEFORE any action send: MoveIt's documented out-of-bounds
goal clamping ("Assuming maximum bounds instead") is never relied on. No ROS
imports. Behavioral reference: the waypoint inspection and strict admission
gates of tools/p1_real_joint_probe.py and tools/p1_real_pose_probe.py.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from rocobrick.moveit_contract import (
    ARM_JOINT_NAMES,
    BASE_FRAME,
    EXECUTION_POSITION_LIMITS,
    LIMIT_EPSILON,
    MAX_ORIENTATION_TOLERANCE_RAD,
    MAX_POSITION_TOLERANCE_M,
    MIN_WAYPOINTS,
    QUAT_NORM_EPS,
    QUAT_NORM_MIN,
    TCP_LINK,
    TCP_STATUS,
    TCP_STATUS_CALIBRATED_PROJECT,
    TIP_LINK,
    AdmissionDecision,
    JointTarget,
    Link6PoseTarget,
    TcpPoseTarget,
)
from rocobrick.moveit_requests import validate_target_within_limits
from rocobrick.tcp_contract import EXPECTED_RUNTIME_MODEL_SHA256


def _is_sha256(value: object) -> bool:
    """True for a lowercase 64-character SHA-256 hex digest string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_finite(value: object) -> bool:
    """Return True when the value is a finite int/float."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _rejected(reason: str, details: Mapping[str, object]) -> AdmissionDecision:
    """Build a rejected decision with the given evidence."""
    return AdmissionDecision(accepted=False, reason=reason, details=details)


def validate_trajectory_waypoints(
    joint_names: Sequence[str],
    points: Sequence[Sequence[float]],
    *,
    epsilon: float = LIMIT_EPSILON,
) -> AdmissionDecision:
    """Admit a planned trajectory: exact arm joints, finite, inside limits.

    Column order may be normalized onto the canonical arm order for checking
    only. Requires at least MIN_WAYPOINTS points; every waypoint value must be
    finite and inside its execution limit plus ``epsilon`` slack.
    """
    if not _is_finite(epsilon) or epsilon < 0.0:
        raise ValueError(f"epsilon must be finite and non-negative, got {epsilon}")
    names = [str(name) for name in joint_names]
    if set(names) != set(ARM_JOINT_NAMES):
        return _rejected(
            "trajectory joint names must be exactly the six canonical arm joints",
            {
                "joint_names": tuple(names),
                "missing": tuple(sorted(set(ARM_JOINT_NAMES) - set(names))),
                "extra": tuple(sorted(set(names) - set(ARM_JOINT_NAMES))),
            },
        )
    if len(set(names)) != len(names):
        return _rejected(
            "trajectory joint names contain duplicates", {"joint_names": tuple(names)}
        )
    if len(points) < MIN_WAYPOINTS:
        return _rejected(
            f"trajectory needs at least {MIN_WAYPOINTS} waypoints, got {len(points)}",
            {"waypoint_count": len(points)},
        )

    indices = [names.index(name) for name in ARM_JOINT_NAMES]
    nonfinite = 0
    width_mismatches = 0
    max_excess = {name: 0.0 for name in ARM_JOINT_NAMES}
    for point in points:
        if len(point) != len(names):
            width_mismatches += 1
            continue
        for name, slot in zip(ARM_JOINT_NAMES, indices):
            value = point[slot]
            if not _is_finite(value):
                nonfinite += 1
                continue
            lower, upper = EXECUTION_POSITION_LIMITS[name]
            excess = max(lower - float(value), float(value) - upper, 0.0)
            max_excess[name] = max(max_excess[name], excess)
    problems: list[str] = []
    if width_mismatches:
        problems.append(f"{width_mismatches} waypoints have the wrong width")
    if nonfinite:
        problems.append(f"{nonfinite} waypoint values are non-finite")
    over = {name: excess for name, excess in max_excess.items() if excess > epsilon}
    if over:
        problems.append("waypoints outside the execution limits beyond epsilon")
    details: dict[str, object] = {
        "waypoint_count": len(points),
        "nonfinite": nonfinite,
        "width_mismatches": width_mismatches,
        "epsilon": epsilon,
        "max_limit_excess": dict(max_excess),
    }
    if problems:
        return _rejected("; ".join(problems), {**details, "over_limits": over})
    return AdmissionDecision(accepted=True, reason="ok", details=details)


def validate_joint_target(
    target: JointTarget, *, margin: float = 0.0
) -> AdmissionDecision:
    """Admit a JointTarget: canonical names, finite values, limits + margin."""
    if not _is_finite(margin) or margin < 0.0:
        raise ValueError(f"margin must be finite and non-negative, got {margin}")
    if not _is_finite(target.goal_tolerance) or float(target.goal_tolerance) <= 0.0:
        return _rejected(
            f"goal_tolerance must be finite and positive, got {target.goal_tolerance}",
            {"goal_tolerance": target.goal_tolerance},
        )
    limits_decision = validate_target_within_limits(target.positions, margin=margin)
    if not limits_decision.accepted:
        return limits_decision
    return AdmissionDecision(
        accepted=True,
        reason="ok",
        details={
            "goal_tolerance": float(target.goal_tolerance),
            "margin": margin,
            "joints": tuple(ARM_JOINT_NAMES),
        },
    )


def validate_link6_pose_target(target: Link6PoseTarget) -> AdmissionDecision:
    """Admit a Link6PoseTarget: exact frame/link, normalized pose, tolerances."""
    problems: list[str] = []
    if target.frame_id != BASE_FRAME:
        problems.append(
            f"frame_id must be exactly {BASE_FRAME!r}, got {target.frame_id!r}"
        )
    if target.tip_link != TIP_LINK:
        problems.append(
            f"tip_link must be exactly {TIP_LINK!r}, got {target.tip_link!r}"
        )
    if target.tcp_status != TCP_STATUS:
        problems.append(
            f"tcp_status must be exactly {TCP_STATUS!r}, got {target.tcp_status!r}"
        )
    if len(target.position) != 3 or not all(
        _is_finite(value) for value in target.position
    ):
        problems.append("position must be exactly 3 finite values")
    if len(target.orientation_xyzw) != 4 or not all(
        _is_finite(value) for value in target.orientation_xyzw
    ):
        problems.append("orientation must be exactly 4 finite components")
    else:
        norm = math.sqrt(sum(float(value) ** 2 for value in target.orientation_xyzw))
        if norm < QUAT_NORM_MIN:
            problems.append(
                f"near-zero quaternion (norm {norm:.3e} < {QUAT_NORM_MIN:.3e})"
            )
        elif abs(norm - 1.0) > QUAT_NORM_EPS:
            problems.append(f"quaternion is not normalized (norm {norm:.12f})")
    if not _is_finite(target.position_tolerance_m) or not (
        0.0 < float(target.position_tolerance_m) <= MAX_POSITION_TOLERANCE_M
    ):
        problems.append(
            f"position_tolerance_m must be finite in (0, {MAX_POSITION_TOLERANCE_M}], "
            f"got {target.position_tolerance_m}"
        )
    if not _is_finite(target.orientation_tolerance_rad) or not (
        0.0 < float(target.orientation_tolerance_rad) <= MAX_ORIENTATION_TOLERANCE_RAD
    ):
        problems.append(
            "orientation_tolerance_rad must be finite in (0, "
            f"{MAX_ORIENTATION_TOLERANCE_RAD}], got {target.orientation_tolerance_rad}"
        )
    if problems:
        return _rejected(
            "pose target rejected: " + "; ".join(problems),
            {"problems": tuple(problems)},
        )
    return AdmissionDecision(
        accepted=True,
        reason="ok",
        details={
            "frame_id": target.frame_id,
            "tip_link": target.tip_link,
            "tcp_status": target.tcp_status,
        },
    )


def validate_tcp_pose_target(target: TcpPoseTarget) -> AdmissionDecision:
    """Admit a TcpPoseTarget: calibrated metadata, exact frame/link, pose.

    The link6 admission contract applies verbatim (frame, tip link semantics
    swapped for the calibrated TCP, normalized pose, tolerances) plus the
    resolved/scoped provenance: ``tcp_status`` must be the project-calibrated
    status, and both SHA-256 fields must be lowercase digests with
    ``runtime_model_sha256`` equal to the frozen runtime-USD scope of the
    accepted project calibration.
    """
    problems: list[str] = []
    if target.frame_id != BASE_FRAME:
        problems.append(
            f"frame_id must be exactly {BASE_FRAME!r}, got {target.frame_id!r}"
        )
    if target.tip_link != TCP_LINK:
        problems.append(
            f"tip_link must be exactly {TCP_LINK!r}, got {target.tip_link!r}"
        )
    if target.tcp_status != TCP_STATUS_CALIBRATED_PROJECT:
        problems.append(
            f"tcp_status must be exactly {TCP_STATUS_CALIBRATED_PROJECT!r}, "
            f"got {target.tcp_status!r}"
        )
    if not _is_sha256(target.runtime_model_sha256):
        problems.append(
            f"runtime_model_sha256 must be a lowercase SHA-256 digest, got "
            f"{target.runtime_model_sha256!r}"
        )
    elif target.runtime_model_sha256 != EXPECTED_RUNTIME_MODEL_SHA256:
        problems.append(
            "runtime_model_sha256 is not the expected runtime-USD scope of "
            "the project calibration"
        )
    if not _is_sha256(target.source_sha256):
        problems.append(
            f"source_sha256 must be a lowercase SHA-256 digest, got "
            f"{target.source_sha256!r}"
        )
    if len(target.position) != 3 or not all(
        _is_finite(value) for value in target.position
    ):
        problems.append("position must be exactly 3 finite values")
    if len(target.orientation_xyzw) != 4 or not all(
        _is_finite(value) for value in target.orientation_xyzw
    ):
        problems.append("orientation must be exactly 4 finite components")
    else:
        norm = math.sqrt(sum(float(value) ** 2 for value in target.orientation_xyzw))
        if norm < QUAT_NORM_MIN:
            problems.append(
                f"near-zero quaternion (norm {norm:.3e} < {QUAT_NORM_MIN:.3e})"
            )
        elif abs(norm - 1.0) > QUAT_NORM_EPS:
            problems.append(f"quaternion is not normalized (norm {norm:.12f})")
    if not _is_finite(target.position_tolerance_m) or not (
        0.0 < float(target.position_tolerance_m) <= MAX_POSITION_TOLERANCE_M
    ):
        problems.append(
            f"position_tolerance_m must be finite in (0, {MAX_POSITION_TOLERANCE_M}], "
            f"got {target.position_tolerance_m}"
        )
    if not _is_finite(target.orientation_tolerance_rad) or not (
        0.0 < float(target.orientation_tolerance_rad) <= MAX_ORIENTATION_TOLERANCE_RAD
    ):
        problems.append(
            "orientation_tolerance_rad must be finite in (0, "
            f"{MAX_ORIENTATION_TOLERANCE_RAD}], got {target.orientation_tolerance_rad}"
        )
    if problems:
        return _rejected(
            "tcp pose target rejected: " + "; ".join(problems),
            {"problems": tuple(problems)},
        )
    return AdmissionDecision(
        accepted=True,
        reason="ok",
        details={
            "frame_id": target.frame_id,
            "tip_link": target.tip_link,
            "tcp_status": target.tcp_status,
            "runtime_model_sha256": target.runtime_model_sha256,
            "source_sha256": target.source_sha256,
        },
    )


def strict_action_admission(
    target: JointTarget | Link6PoseTarget | TcpPoseTarget,
) -> AdmissionDecision:
    """Final pre-send admission; rejects unknown target types outright.

    Joint targets are checked against the raw narrowed execution limits: the
    strict 0.03 margin discipline for moved joints is already enforced by
    :func:`rocobrick.moveit_requests.build_joint_target`, while unchanged
    joints may legitimately sit anywhere inside the limits (live P1 evidence).
    Pose targets (link6 and the project-calibrated TCP) are verified against
    the full pose contract. Nothing is clamped or corrected.
    """
    if isinstance(target, JointTarget):
        return validate_joint_target(target)
    if isinstance(target, Link6PoseTarget):
        return validate_link6_pose_target(target)
    if isinstance(target, TcpPoseTarget):
        return validate_tcp_pose_target(target)
    return _rejected(
        f"unsupported target type {type(target).__name__}",
        {"type": type(target).__name__},
    )


__all__ = [
    "strict_action_admission",
    "validate_joint_target",
    "validate_link6_pose_target",
    "validate_tcp_pose_target",
    "validate_trajectory_waypoints",
]
