"""Local URDF IK + supervised controller execution, without MoveGroup/IK service."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np

from rocobrick.clearance.fk import UrdfChain
from rocobrick.clearance.gate import sample_trajectory
from rocobrick.clearance.grasp_approach import grasp_approach_clearance
from rocobrick.clearance.pick_entry import pick_entry_world_clearance
from rocobrick.coordinated_session_entry import clearance_gate
from rocobrick.isaac_backend import tcp_pose_target_from_payload
from rocobrick.moveit_contract import BASE_FRAME
from rocobrick.skills.observation import _pose
from rocobrick.skills.pick_contact import inspect_pick_contact, validate_pick_guard
from rocobrick.skills.placement_geometry import compose, pose_error
from rocobrick.skills.targets import _quat_rotate_vector


def rotation(quaternion):
    """Return an XYZW rotation matrix."""
    return np.column_stack(
        [
            _quat_rotate_vector(quaternion, axis)
            for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        ]
    )


def vertical_path(model, robot, state, start, end, *, expired=lambda: False):
    """Solve a seeded, bounded local branch at half-millimetre intervals.

    Returns:
        Dense arm trajectory; residuals and branch jumps must pass separately.
    """
    chain = UrdfChain(model)
    names = [f"{robot}_joint{i}" for i in range(1, 7)]
    root = ET.fromstring(model)
    limits = {
        j.get("name"): (
            float(j.find("limit").get("lower")),
            float(j.find("limit").get("upper")),
        )
        for j in root.findall("joint")
        if j.get("name") in names
    }
    if set(limits) != set(names) or not all(name in state for name in names):
        raise ValueError("PICK IK requires complete live state/model limits")
    lower, upper = np.array([limits[n] for n in names]).T
    q = np.array([state[n] for n in names], dtype=float)
    if np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6):
        raise ValueError("PICK IK start outside model bounds")
    q = np.clip(q, lower, upper)
    goal_rotation = rotation(end.orientation_xyzw)

    def residual(values, goal):
        if expired():
            raise ValueError("PICK local IK deadline")
        current = chain.forward({**state, **dict(zip(names, values))})[
            f"{robot}_rocobrick_tcp"
        ]
        diff = goal_rotation @ current[:3, :3].T
        angle = math.acos(float(np.clip((np.trace(diff) - 1) / 2, -1, 1)))
        if angle > 0.1:
            raise ValueError("PICK IK orientation left local branch")
        angular = (
            np.array(
                [
                    diff[2, 1] - diff[1, 2],
                    diff[0, 2] - diff[2, 0],
                    diff[1, 0] - diff[0, 1],
                ]
            )
            / 2
        )
        return np.r_[goal - current[:3, 3], angular]

    start_error = residual(q, np.array(start.position_xyz))
    if (
        np.linalg.norm(start_error[:3]) > 0.001
        or np.linalg.norm(start_error[3:]) > 0.015
    ):
        raise ValueError("PICK live TCP disagrees with URDF FK")
    distance = math.dist(start.position_xyz, end.position_xyz)
    steps = max(1, math.ceil(distance / 0.0005))
    points = [{"positions": list(q), "time_from_start_s": 0.0}]
    for index in range(1, steps + 1):
        goal = np.array(start.position_xyz) + index / steps * (
            np.array(end.position_xyz) - start.position_xyz
        )
        prior = q.copy()
        for _ in range(80):
            error = residual(q, goal)
            if (
                np.linalg.norm(error[:3]) < 0.00005
                and np.linalg.norm(error[3:]) < 0.0005
            ):
                break
            jacobian = np.column_stack(
                [
                    (residual(q + np.eye(6)[j] * 1e-5, goal) - error) / 1e-5
                    for j in range(6)
                ]
            )
            delta = -jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + np.eye(6) * 1e-7, error
            )
            q = np.clip(
                q + delta * min(1.0, 0.02 / max(1e-12, float(np.max(np.abs(delta))))),
                lower,
                upper,
            )
        else:
            raise ValueError("PICK local IK residual did not converge")
        if np.max(np.abs(q - prior)) > 0.05:
            raise ValueError("PICK local IK branch jump")
        points.append(
            {
                "positions": list(q),
                "time_from_start_s": index / steps * max(0.4, distance / 0.001),
            }
        )
    # Certify the actual joint-interpolated command, not just ideal TCP knots.
    line = np.asarray(end.position_xyz) - np.asarray(start.position_xyz)
    for values in sample_trajectory(
        [p["positions"] for p in points], max_substep_rad=0.002, max_samples=10000
    ):
        fk = chain.forward({**state, **dict(zip(names, values))})[
            f"{robot}_rocobrick_tcp"
        ]
        offset = fk[:3, 3] - np.asarray(start.position_xyz)
        fraction = float(np.clip(offset @ line / (line @ line), 0, 1))
        lateral = np.linalg.norm(offset - fraction * line)
        angle = math.acos(
            float(np.clip((np.trace(goal_rotation @ fk[:3, :3].T) - 1) / 2, -1, 1))
        )
        if lateral > 0.001 or angle > 0.015:
            raise ValueError("PICK commanded joint interpolation leaves entry corridor")
        if expired():
            raise ValueError("PICK local IK deadline")
    return {"joint_names": names, "points": points}


def execute_pick_entry(runtime, payload, timeout_s):
    """Certify a local path and delegate only to the supervised arm controller.

    Returns:
        Strict terminal action evidence, with no planning-service fallback.
    """
    if set(payload) != {"target", "guard"}:
        raise ValueError("PICK entry payload keys mismatch")
    target = tcp_pose_target_from_payload(payload["target"])
    guard = validate_pick_guard(payload["guard"])
    robot = runtime.robot_id
    # The calibrated TCP target carries the repo-wide ``BASE_FRAME`` name; the
    # combined model spells the same frame per robot. Both name the robot base,
    # and the guard itself is expressed in the world frame the scene reports.
    if guard["robot_id"] != robot or target.frame_id not in {
        BASE_FRAME,
        f"{robot}_base_link",
    }:
        raise ValueError("PICK entry robot/frame mismatch")
    deadline = runtime._clock_fn() + timeout_s
    scene = runtime.query_scene_state(0.5)
    reason, _ = inspect_pick_contact(scene, guard)
    if reason:
        raise ValueError(reason)
    start, end = (_pose(guard[k], k) for k in ("start", "end"))
    base = _pose(guard["region"]["base_in_world"], "base")
    expected = compose(
        base,
        _pose(
            {
                "position_xyz": list(target.position),
                "orientation_xyzw": list(target.orientation_xyzw),
            },
            "target",
        ),
    )
    if max(pose_error(end, expected)) > 1e-6:
        raise ValueError("PICK entry target/guard mismatch")
    state = runtime._combined_start_state_reader()
    model = runtime._combined_robot_description_reader()
    raw = vertical_path(
        model, robot, state, start, end, expired=lambda: runtime._clock_fn() >= deadline
    )
    request = SimpleNamespace(
        targets={robot: {"grasp_region": guard["region"]}},
        scene_snapshot=scene,
        start_state=state,
        robots=tuple(
            sorted({name.split("_joint")[0] for name in state if "_joint" in name})
        ),
        physical_facts={},
    )
    geometry = grasp_approach_clearance(model, request, raw)
    world = pick_entry_world_clearance(model, request, raw)
    clearance = clearance_gate(
        SimpleNamespace(unified_robot_description=lambda: model), request, raw
    )
    if not clearance.passed:
        raise ValueError("PICK entry cross-robot clearance failed")
    reason, _ = inspect_pick_contact(runtime.query_scene_state(0.5), guard)
    if reason:
        raise ValueError(reason)
    if runtime._clock_fn() >= deadline:
        raise ValueError("PICK entry certification deadline")
    factories = runtime._resolve_factories()
    points = []
    for source in raw["points"]:
        point = factories.joint_trajectory_point_factory()
        point.positions = source["positions"]
        point.velocities = [0.0] * 6
        point.accelerations = [0.0] * 6
        point.time_from_start = runtime._duration_message(
            source["time_from_start_s"], factories
        )
        points.append(point)
    trajectory = SimpleNamespace(joint_names=raw["joint_names"], points=points)
    outcome = runtime._execute_contact_trajectory(
        trajectory, target, guard, deadline, len(points)
    )
    from dataclasses import replace

    return replace(
        outcome,
        details={
            **outcome.details,
            "path_source": "local_urdf_ik_no_movegroup",
            "grasp_approach_clearance": geometry,
            "moving_arm_world_clearance": world,
            "local_trajectory": raw,
        },
    )
