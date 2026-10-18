"""Local multi-start TCP IK seed contracts.

The unified ``compute_ik`` backend performs one Newton descent per seed, so a
parked arm sitting on the elbow-straight joint limit cannot reach a pose that
flips the tool, and the whole pick then fails before any motion is dispatched.
These tests pin the seed solver that fills that gap: it must actually reach a
reachable pose from the parked state, keep every value inside the model limits,
and decline anything it cannot reach instead of guessing.
"""

from __future__ import annotations

import copy
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rocobrick.clearance import UrdfChain
from rocobrick.clearance.local_ik import seed_tcp_ik
from rocobrick.ros_moveit_runtime import IK_SEED_ATTEMPTS

ROOT = Path(__file__).parents[1]
RUNTIME_URDF = ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf"
ROBOT = "piper_0"
ARM_JOINTS = tuple(f"{ROBOT}_joint{index}" for index in range(1, 7))
# The failing live request started from the parked, elbow-straight state: every
# arm joint at zero, which is exactly where a single Newton descent stalls.
PARKED = dict.fromkeys(ARM_JOINTS, 0.0)
BENT = {
    f"{ROBOT}_joint1": -0.4244,
    f"{ROBOT}_joint2": 1.3186,
    f"{ROBOT}_joint3": -0.8514,
    f"{ROBOT}_joint4": 0.1847,
    f"{ROBOT}_joint5": 0.8779,
    f"{ROBOT}_joint6": -0.5202,
}


def _prefixed_model() -> str:
    """Build the single prefixed arm model the container hands to MoveIt.

    Returns:
        The URDF document text with this robot's link and joint prefixes.
    """
    source = ET.fromstring(RUNTIME_URDF.read_text())
    lines = ['<?xml version="1.0"?>', f'<robot name="{ROBOT}">']
    for element in source.findall("link"):
        if element.get("name") == "world":
            continue
        clone = copy.deepcopy(element)
        clone.set("name", f"{ROBOT}_{element.get('name')}")
        lines.append(ET.tostring(clone, encoding="unicode"))
    # The container's generated model adds the calibrated TCP link.
    lines.append(f'<link name="{ROBOT}_rocobrick_tcp"/>')
    lines.append(
        f'<joint name="{ROBOT}_rocobrick_tcp_joint" type="fixed">'
        f'<parent link="{ROBOT}_link6"/><child link="{ROBOT}_rocobrick_tcp"/>'
        '<origin xyz="-1.887447531601083e-09 -1.3530843112619095e-16 '
        '0.1305895866844472" rpy="0 0 0"/></joint>'
    )
    for element in source.findall("joint"):
        if element.find("parent").get("link") == "world":
            continue
        clone = copy.deepcopy(element)
        clone.set("name", f"{ROBOT}_{element.get('name')}")
        clone.find("parent").set("link", f"{ROBOT}_{clone.find('parent').get('link')}")
        clone.find("child").set("link", f"{ROBOT}_{clone.find('child').get('link')}")
        mimic = clone.find("mimic")
        if mimic is not None:
            mimic.set("joint", f"{ROBOT}_{mimic.get('joint')}")
        lines.append(ET.tostring(clone, encoding="unicode"))
    lines.append("</robot>")
    return "\n".join(lines)


def _tcp_pose(model: str, joints: dict[str, float]) -> dict[str, object]:
    """Serialize the calibrated TCP pose of one posture in the base frame.

    Returns:
        The wire-shaped ``tcp_pose`` mapping the runtime targets carry.
    """
    from scipy.spatial.transform import Rotation

    poses = UrdfChain(model).forward(joints)
    base = np.linalg.inv(poses[f"{ROBOT}_base_link"])
    tcp = base @ poses[f"{ROBOT}_rocobrick_tcp"]
    quaternion = Rotation.from_matrix(tcp[:3, :3]).as_quat()
    return {
        "frame_id": f"{ROBOT}_base_link",
        "position": [float(value) for value in tcp[:3, 3]],
        "orientation_xyzw": [float(value) for value in quaternion],
    }


def _residual(model: str, joints: dict[str, float], pose: dict[str, object]) -> tuple:
    """Return the TCP position error and orientation error of one posture."""
    from scipy.spatial.transform import Rotation

    poses = UrdfChain(model).forward(joints)
    base = np.linalg.inv(poses[f"{ROBOT}_base_link"])
    tcp = base @ poses[f"{ROBOT}_rocobrick_tcp"]
    wanted = np.eye(4)
    wanted[:3, :3] = Rotation.from_quat(pose["orientation_xyzw"]).as_matrix()
    wanted[:3, 3] = np.asarray(pose["position"], dtype=float)
    position_error = float(np.linalg.norm(wanted[:3, 3] - tcp[:3, 3]))
    angle_error = float(
        Rotation.from_matrix(wanted[:3, :3].T @ tcp[:3, :3]).magnitude()
    )
    return position_error, angle_error


def test_seed_reaches_a_flipped_pose_from_the_parked_state() -> None:
    """The parked, elbow-straight start still yields a usable seed."""
    model = _prefixed_model()
    pose = _tcp_pose(model, BENT)
    seed = seed_tcp_ik(model, ROBOT, PARKED, {"tcp_pose": pose})
    assert seed is not None
    assert set(seed) == set(ARM_JOINTS)
    position_error, angle_error = _residual(model, seed, pose)
    assert position_error <= 1e-3
    assert angle_error <= 1e-2


def test_seed_keeps_every_value_inside_the_model_limits() -> None:
    """A seed is a posture the model itself can hold."""
    model = _prefixed_model()
    pose = _tcp_pose(model, BENT)
    seed = seed_tcp_ik(model, ROBOT, PARKED, {"tcp_pose": pose})
    assert seed is not None
    limits = {}
    for joint in ET.fromstring(model).findall("joint"):
        element = joint.find("limit")
        if element is not None and joint.get("name") in ARM_JOINTS:
            limits[joint.get("name")] = (
                float(element.get("lower")),
                float(element.get("upper")),
            )
    assert set(limits) == set(ARM_JOINTS)
    for joint, value in seed.items():
        lower, upper = limits[joint]
        assert lower <= value <= upper


def test_seed_declines_a_pose_in_another_frame() -> None:
    """A pose expressed for a different robot is never reinterpreted."""
    model = _prefixed_model()
    pose = dict(_tcp_pose(model, BENT), frame_id="piper_1_base_link")
    assert seed_tcp_ik(model, ROBOT, PARKED, {"tcp_pose": pose}) is None


def test_seed_declines_an_unreachable_pose_within_the_budget() -> None:
    """An out-of-workspace pose returns None instead of a guess."""
    model = _prefixed_model()
    pose = dict(_tcp_pose(model, BENT), position=[5.0, 0.0, 0.0])
    started = time.monotonic()
    assert seed_tcp_ik(model, ROBOT, PARKED, {"tcp_pose": pose}) is None
    assert time.monotonic() - started < 10.0


def _unified_session():
    """Build the unified session with fake arm runtimes for IK seeding.

    Returns:
        The session and the list collecting every ``solve_tcp_ik`` call.
    """
    from test_coordinated_ros_runtime import pose, request

    from rocobrick.ros_moveit_runtime import RosMoveItRobotSession

    canonical = {f"joint{index}": 0.02 * index for index in range(1, 7)}
    canonical.update({"gripper_joint1": 0.01, "gripper_joint2": -0.01})
    calls = []
    answer = {f"piper_0_joint{index}": 0.11 for index in range(1, 7)}

    def fake_ik(robot_id, target, wire_seed, timeout_s, extra_seeds=()):
        calls.append(
            {"robot": robot_id, "seed": dict(wire_seed), "extra": tuple(extra_seeds)}
        )
        return dict(answer)

    def runtime_for(robot):
        return SimpleNamespace(
            arm_joint_names=tuple(f"{robot}_joint{index}" for index in range(1, 7)),
            latest_joint_state=lambda: dict(canonical),
            solve_tcp_ik=fake_ik,
        )

    session = RosMoveItRobotSession.__new__(RosMoveItRobotSession)
    session.robot_ids = ("piper_0", "piper_1")
    session.runtimes = {robot: runtime_for(robot) for robot in session.robot_ids}
    session._joint_limits_cache = {
        f"{robot}_joint{index}": (-1.0, 1.0)
        for robot in session.robot_ids
        for index in range(1, 7)
    }
    items = request(
        {
            "piper_0": pose("piper_0", 0.1),
            "piper_1": {**pose("piper_1", -0.1), "maintain_pose": True},
        }
    )
    return session, items, calls


def test_local_seed_is_offered_first_to_the_unified_ik(monkeypatch) -> None:
    """The locally solved posture leads the restart budget, not trails it."""
    session, items, calls = _unified_session()
    session.unified_robot_description = lambda timeout: "<robot/>"
    solved_seed = dict.fromkeys(ARM_JOINTS, 0.25)
    monkeypatch.setattr(
        "rocobrick.clearance.local_ik.seed_tcp_ik",
        lambda model, robot, state, target, **_: dict(solved_seed),
    )
    session.unified_joint_targets(items)
    assert calls and calls[0]["extra"][0] == solved_seed
    assert len(calls[0]["extra"]) == IK_SEED_ATTEMPTS


def test_unavailable_local_seed_leaves_the_restart_budget_unchanged(
    monkeypatch,
) -> None:
    """A failing local solver never shrinks or blocks the existing budget."""
    session, items, calls = _unified_session()
    session.unified_robot_description = lambda timeout: "<robot/>"

    def explode(model, robot, state, target, **_):
        raise ValueError("no solution")

    monkeypatch.setattr("rocobrick.clearance.local_ik.seed_tcp_ik", explode)
    session.unified_joint_targets(items)
    assert calls and len(calls[0]["extra"]) == IK_SEED_ATTEMPTS - 1
    assert all(entry != dict.fromkeys(ARM_JOINTS, 0.25) for entry in calls[0]["extra"])


def test_silent_robot_description_skips_the_local_seed(monkeypatch) -> None:
    """Without the planner's own model the local solver is not consulted."""
    session, items, calls = _unified_session()
    session._robot_description_cache = None

    def silent(timeout):
        raise RuntimeError("combined description did not answer")

    session.unified_robot_description = silent
    consulted = []
    monkeypatch.setattr(
        "rocobrick.clearance.local_ik.seed_tcp_ik",
        lambda *args, **kwargs: consulted.append(args) or None,
    )
    session.unified_joint_targets(items)
    assert consulted == []
    assert calls and len(calls[0]["extra"]) == IK_SEED_ATTEMPTS - 1
