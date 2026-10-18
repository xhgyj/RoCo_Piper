"""Bounded multi-start TCP IK seed for one arm's unified query.

The stock KDL backend behind the unified ``compute_ik`` service performs ONE
Newton descent per call, so restart diversity has to come from explicit seeds.
A hover pose that reorients the tool by a large angle is reachable, but the
measured state of a parked arm sits on the elbow-straight joint limit where the
Jacobian is rank deficient: a descent from there stalls millimetres and degrees
away, the uniform random seeds of the existing budget start too far away to
help, and the whole query fails with ``no seed converged`` even though the pose
is plainly inside the workspace.

This module supplies the missing candidate: the same bound-aware trust-region
least-squares the handover geometry already solves with, restarted from a small
deterministic grid of bent-arm postures and then from a reproducible random
budget, all under one wall-clock deadline.  The result is only a seed.  The
caller still requires the authoritative IK answer, so ``None`` simply leaves
the existing seed budget untouched; nothing here is certified or dispatched.
"""

from __future__ import annotations

import hashlib
import itertools
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping

import numpy as np

from rocobrick.clearance.fk import UrdfChain

ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 7))
# Deterministic bent-arm restarts: the elbow (joint3) and shoulder (joint2)
# spans are the ones that leave the stretched-at-the-limit parked posture, and
# joint5 tilts the wrist either way so a tool flip needs no wrist singularity.
RESTART_ELBOWS = (-1.6, -0.8, -2.4)
RESTART_SHOULDERS = (0.9, 0.3, 1.6)
RESTART_WRISTS = (0.8, -0.8, 0.0)
RANDOM_ATTEMPTS = 24
SOLVER_FUNCTION_EVALUATIONS = 200
SEED_POSITION_TOLERANCE_M = 5e-4
SEED_ANGLE_TOLERANCE_RAD = 5e-3
SEED_RANDOM_SEED_SPAN = 2**32
# Position and rotation residuals are mixed through this length so a bounded
# least-squares step trades them at a physical scale instead of per-axis units.
SEED_RESIDUAL_SCALE_M = 0.05
SEED_DEADLINE_S = 2.0


def _quaternion_matrix(quaternion) -> np.ndarray:
    """Return the XYZW rotation matrix of one quaternion."""
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("TCP orientation must be a finite XYZW quaternion")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("TCP orientation must be a non-zero quaternion")
    x, y, z, w = values / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _joint_limits(model: str, names) -> dict[str, tuple[float, float]]:
    """Read the exact model limits of the named joints.

    Returns:
        Mapping of joint name to its ``(lower, upper)`` limit in radians.
    """
    root = ET.fromstring(model)
    wanted = set(names)
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in wanted:
            continue
        element = joint.find("limit")
        if element is None:
            continue
        lower, upper = element.get("lower"), element.get("upper")
        if lower is None or upper is None:
            continue
        limits[str(name)] = (float(lower), float(upper))
    return limits


def _pose_residual(chain, state, names, goal, tcp_link):
    """Build the SE(3) residual of one arm posture against the goal.

    Returns:
        The residual function mapping six joint values to a six-vector.
    """
    rotation, position = goal[:3, :3], goal[:3, 3]

    def residual(values):
        tcp = chain.forward({**state, **dict(zip(names, values, strict=True))})[
            tcp_link
        ]
        difference = rotation @ tcp[:3, :3].T
        angular = (
            np.array(
                [
                    difference[2, 1] - difference[1, 2],
                    difference[0, 2] - difference[2, 0],
                    difference[1, 0] - difference[0, 1],
                ]
            )
            / 2.0
        )
        return np.r_[(position - tcp[:3, 3]) / SEED_RESIDUAL_SCALE_M, angular]

    return residual


def _starts(measured, lower, upper, seed):
    """Yield the deterministic restarts, then the reproducible random ones."""
    yield measured
    yield (lower + upper) / 2.0
    for elbow, shoulder, wrist in itertools.product(
        RESTART_ELBOWS, RESTART_SHOULDERS, RESTART_WRISTS
    ):
        start = measured.copy()
        start[1], start[2], start[4] = shoulder, elbow, wrist
        yield np.clip(start, lower, upper)
    generator = np.random.default_rng(seed)
    for _ in range(RANDOM_ATTEMPTS):
        yield generator.uniform(lower, upper)


def seed_tcp_ik(
    model: str,
    robot: str,
    state: Mapping[str, float],
    target: Mapping[str, object],
    *,
    deadline_s: float = SEED_DEADLINE_S,
) -> dict[str, float] | None:
    """Return one arm posture reaching the target TCP, as an IK seed.

    Args:
        model: The combined URDF document the planner itself plans on.
        robot: The participant whose arm is seeded.
        state: That participant's live, wire-named joint state.
        target: The serialized combined target carrying ``tcp_pose``.
        deadline_s: Wall-clock budget for the whole multi-start search.

    Returns:
        Wire-named arm joint positions inside the model limits, or ``None``
        when no restart reaches the pose.  ``None`` only leaves the caller's
        existing seed budget untouched.
    """
    pose = target.get("tcp_pose") if isinstance(target, Mapping) else None
    if not isinstance(pose, Mapping):
        return None
    if pose.get("frame_id") != f"{robot}_base_link":
        return None
    names = [f"{robot}_{name}" for name in ARM_JOINTS]
    tcp_link = f"{robot}_rocobrick_tcp"
    if not all(name in state for name in names):
        return None
    limits = _joint_limits(model, names)
    if set(limits) != set(names):
        return None
    chain = UrdfChain(model)
    poses = chain.forward(state)
    base = poses.get(f"{robot}_base_link")
    if base is None or poses.get(tcp_link) is None:
        return None
    goal = np.eye(4)
    goal[:3, :3] = base[:3, :3] @ _quaternion_matrix(pose.get("orientation_xyzw"))
    goal[:3, 3] = (
        base[:3, :3] @ np.asarray(pose.get("position"), dtype=float) + base[:3, 3]
    )
    lower = np.array([limits[name][0] for name in names])
    upper = np.array([limits[name][1] for name in names])
    measured = np.clip(np.array([float(state[name]) for name in names]), lower, upper)
    digest = hashlib.sha256(
        np.round(np.r_[goal[:3, :3].ravel(), goal[:3, 3]], 9).tobytes() + robot.encode()
    ).digest()
    restart_seed = int.from_bytes(digest[:4], "big") % SEED_RANDOM_SEED_SPAN
    residual = _pose_residual(chain, state, names, goal, tcp_link)
    from scipy.optimize import least_squares

    expires = time.monotonic() + deadline_s
    for start in _starts(measured, lower, upper, restart_seed):
        if time.monotonic() >= expires:
            return None
        try:
            solved = least_squares(
                residual,
                np.clip(start, lower, upper),
                bounds=(lower, upper),
                max_nfev=SOLVER_FUNCTION_EVALUATIONS,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
        error = residual(solved.x)
        if (
            float(np.linalg.norm(error[:3])) * SEED_RESIDUAL_SCALE_M
            <= SEED_POSITION_TOLERANCE_M
            and float(np.linalg.norm(error[3:])) <= SEED_ANGLE_TOLERANCE_RAD
        ):
            return {
                name: float(value) for name, value in zip(names, solved.x, strict=True)
            }
    return None
