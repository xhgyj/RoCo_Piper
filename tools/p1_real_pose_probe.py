#!/usr/bin/env python3
"""P1.C container-side real pose-target probe (host Isaac bridge + MoveIt).

Runs INSIDE the rootless ROS Humble container (started by
``tools/smoke_p1_real_pose.sh`` together with the ``move_group`` stack) and
proves a real MoveIt POSE-target execution end to end, against the live host
bridge (isaac/p05_host_bridge.py) through the container-client DDS profile:

    MoveIt plan-and-execute (moveit_msgs/action/MoveGroup, plan_only false)
      -> arm_controller (FollowJointTrajectory)
      -> ros2_control TopicBasedSystem -> /isaac_joint_command
      -> host Isaac articulation -> live /isaac_joint_states
      -> /joint_states via joint_state_broadcaster.

Scope is the link6 ORIGIN only. The model contract explicitly records the TCP
as unresolved, so the tip frame of every goal and check is ``link6`` expressed
in ``base_link``; no TCP frame is created or referenced.

Phases, all wall-clock bounded:

1. READINESS (no motion before all pass): strictly increasing /clock samples;
   /isaac_joint_states with the exact 8-DOF names, finite positions and a
   velocity array of length 0 or 8 (>= 3 valid messages); /joint_states from
   the broadcaster carrying the SAME 8 finite values; /controller_manager/
   list_controllers reporting joint_state_broadcaster, arm_controller and
   gripper_controller active; the /arm_controller/follow_joint_trajectory
   action ready; the MoveGroup action (/move_action or /move_group, exact name
   recorded) ready; the /compute_fk service ready; then a fresh, complete,
   finite baseline captured at rest (>= 3 valid states plus consecutive
   stable samples before anything is derived from it).
2. FK-DERIVED TARGET: the target pose is computed by the live /compute_fk
   service (GetPositionFK, fk_link_names=["link6"], header.frame_id=
   "base_link") from the captured baseline joint state; no world coordinate is
   hardcoded anywhere in this probe. A tiny +0.005 m displacement along
   base_link x (fallback z) is applied ONLY when a bounded GetPositionIK
   preflight (collision aware, seeded with the baseline) proves the displaced
   pose reachable; otherwise the exact FK pose is used and the reason is
   recorded. The target is labeled tip_link=link6, tcp_status=unresolved.
3. STRICT GOAL ADMISSION (caller side, BEFORE any MoveGroup send): finite
   position, finite and near-zero-rejected quaternion, normalized quaternion,
   exact frame/link on both constraints, sphere radius and absolute orientation
   tolerances inside the pinned bounds, and NO simultaneous joint target. A
   single goal position constraint (SPHERE) and a single goal orientation
   constraint are built with moveit_msgs.msg types; MoveIt's documented goal
   clamping is NEVER relied on. An inadmissible goal fails before the action
   call.
4. OUTBOUND POSE EXECUTION: moveit_msgs/action/MoveGroup with plan_only
   false, replan false, empty start state, allowed planning time 5 s and
   velocity/acceleration scaling 0.1. The goal must be accepted, finish with
   terminal status SUCCEEDED and MoveIt error code SUCCESS (1) inside the wall
   deadline (an overrun is cancelled and recorded), the planned trajectory
   must have >= 2 points whose every normalized arm waypoint is finite and
   inside the narrowed execution limits. Convergence is proven with the LIVE
   state through /compute_fk: the final base_link -> link6 pose must stay
   within 0.01 m position error and 0.03 rad orientation angular error of the
   requested pose, with >= 2 post-command state messages and >= 1 observed
   /isaac_joint_command message, and the arm must be at rest
   (SETTLE_STABLE_SAMPLES consecutive stable samples).
5. RETURN TO BASELINE: a second REAL MoveGroup pose goal whose pose is the FK
   of the ORIGINAL baseline (never a hardcoded joint command); same gates and
   tolerances, final FK position/orientation verified against the baseline FK
   pose, and no continuing movement at the end (at-rest gate again).

The probe prints exactly one machine-readable JSON result line on stdout and
exits 0 only when every phase passed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

SCHEMA = "rocobrick-p1-real-pose-probe/v1"
PHASE = "P1.C"

# Standard ROS topics plus the official Isaac<->MoveIt topic contract from the
# pinned vendor ros2_control xacro (TopicBasedSystem): the states topic is
# plural, the command topic singular.
TOPIC_CLOCK = "/clock"
TOPIC_JOINT_STATES = "/isaac_joint_states"
TOPIC_BROADCAST_STATES = "/joint_states"
TOPIC_JOINT_COMMANDS = "/isaac_joint_command"

# Actions / services.
ACTION_ARM_FOLLOW = "/arm_controller/follow_joint_trajectory"
MOVE_GROUP_ACTION_CANDIDATES = ("/move_action", "/move_group")
SERVICE_LIST_CONTROLLERS = "/controller_manager/list_controllers"
SERVICE_FK = "/compute_fk"
# Live evidence: this pinned Humble move_group exposes the IK service under
# the "/compute_ik" name (the "/get_position_ik" alias exists in other
# MoveIt2 releases); the ready candidate is resolved and recorded.
SERVICE_IK_CANDIDATES = ("/compute_ik", "/get_position_ik")

REQUIRED_CONTROLLERS = ("joint_state_broadcaster", "arm_controller", "gripper_controller")

# Same order as the model contract in tests/test_piper_l_model_contract.py.
EXPECTED_DOF_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_joint1",
    "gripper_joint2",
)
ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINT_NAMES = ("gripper_joint1", "gripper_joint2")

# Narrowed EXECUTION limits (rad) = intersection of the pinned official
# sources, i.e. the model-contract table (tests/test_piper_l_model_contract.py)
# as enforced for PLANNING by
# ros_ws/src/rocobrick_piper_l_moveit_config/config/joint_limits.yaml.
# These are hardcoded here for waypoint admission: the probe never relies on
# MoveIt's out-of-bounds clamping ("Assuming maximum bounds instead").
EXECUTION_POSITION_LIMITS = {
    "joint1": (-2.6179938, 2.6179938),
    "joint2": (0.0, 3.1415926),
    "joint3": (-2.9670597, 0.0),
    "joint4": (-1.7453292, 1.7453292),
    "joint5": (-1.2217304, 1.2217304),
    "joint6": (-2.0943951, 2.0943951),
}

# Scope: the link6 ORIGIN only. The model contract explicitly records the TCP
# as unresolved (no tool0 or dedicated TCP frame exists in the pinned model),
# so no TCP frame is created, referenced, or corrected for.
TIP_LINK = "link6"
BASE_FRAME = "base_link"
PLANNING_GROUP = "arm"
TCP_STATUS = "unresolved"

# Live FK verification thresholds: the final base_link -> link6 pose must
# stay within 0.01 m position error and 0.03 rad orientation angular error of
# the requested pose. Never weakened.
POSITION_TOLERANCE_M = 0.01
ORIENTATION_TOLERANCE_RAD = 0.03
# Requested goal constraints are TIGHTER than the verification thresholds:
# the planner may legally use the whole constraint volume, so requesting the
# full 0.01 m / 0.03 rad would leave zero margin for execution noise (live
# evidence: a planned goal on the tolerance boundary produced a 0.0322 rad
# final angular error). Both stay inside the allowed bounds (<= 0.01 m,
# <= 0.03 rad) of the P1.C contract.
GOAL_POSITION_TOLERANCE_M = 0.005
GOAL_ORIENTATION_TOLERANCE_RAD = 0.015

# Tiny explicitly recorded displacement applied to the FK pose, ONLY when the
# bounded GetPositionIK preflight proves the displaced pose reachable.
POSE_DISPLACEMENT_M = 0.005
DISPLACEMENT_AXES = ("x", "z")

# Quaternion gates: reject nonfinite components and near-zero quaternions,
# then normalize and demand a normalized result.
QUAT_NORM_MIN = 1.0e-6
QUAT_NORM_EPS = 1.0e-6

# Numeric slack for floating-point noise only (rad / unitless).
LIMIT_EPSILON = 1.0e-4
# The IK-preflight solution must sit strictly inside the execution limits with
# this margin (rad), mirroring the joint-probe admission discipline.
ADMISSION_MARGIN = 0.03
# Max |/joint_states - /isaac_joint_states| while proving the broadcaster path.
EXACT_STATE_MATCH_TOLERANCE = 1.0e-4
# A state is stale when no valid message arrived within this window (s).
STATE_FRESH_SECONDS = 3.0

MIN_BASELINE_STATE_MESSAGES = 3
MIN_POST_COMMAND_STATE_MESSAGES = 2
MIN_COMMAND_MESSAGES = 1

# The arm must be AT REST before a pose is derived from it and before each
# MoveGroup goal is sent: MoveIt validates the planned start point against the
# live state with a 0.01 tolerance, so planning from a still-settling state
# aborts execution (live evidence: "start point deviates from current robot
# state more than 0.01"). At rest means this many consecutive state samples
# that move less than SETTLE_STABLE_DELTA_MAX between samples.
SETTLE_STABLE_SAMPLES = 8
SETTLE_STABLE_DELTA_MAX = 5.0e-4

# MoveGroup request contract.
ALLOWED_PLANNING_TIME = 5.0
NUM_PLANNING_ATTEMPTS = 10
MAX_VELOCITY_SCALING = 0.1
MAX_ACCELERATION_SCALING = 0.1
MIN_WAYPOINTS = 2
SUCCESS_CODE = 1  # moveit_msgs.msg.MoveItErrorCodes.SUCCESS
STATUS_SUCCEEDED = GoalStatus.STATUS_SUCCEEDED
ERROR_CODE_NAMES = {
    SUCCESS_CODE: "SUCCESS",
    -1: "PLANNING_FAILED",
    99999: "FAILURE",
}

# Live FK verification: recompute the tip pose from the settled live state via
# /compute_fk and demand the request tolerances against the requested pose.
FK_CALL_SPACING_S = 0.25
SERVICE_CALL_TIMEOUT_S = 5.0

qos_state = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.RELIABLE,
)
qos_sensor = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded probe arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--timeout", type=float, default=90.0, help="readiness phase bound (s)"
    )
    parser.add_argument(
        "--goal-timeout",
        type=float,
        default=90.0,
        help="per MoveGroup action wall deadline (s); an overrun is cancelled",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=60.0,
        help="per-goal live FK convergence bound (s)",
    )
    args = parser.parse_args(argv)
    if min(args.timeout, args.goal_timeout, args.settle_timeout) <= 0.0:
        parser.error("--timeout, --goal-timeout and --settle-timeout must be positive")
    return args


def finite(value: float) -> bool:
    """Return True when the value is a finite float."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def positions_by_name(msg: JointState) -> dict[str, float]:
    """Return the joint_state positions keyed by joint name."""
    return {str(name): float(value) for name, value in zip(msg.name, msg.position)}


def is_valid_state(state: dict[str, float], msg: JointState) -> bool:
    """Complete 8 names, finite positions, velocity length 0 or 8."""
    if set(state) != set(EXPECTED_DOF_NAMES):
        return False
    if not all(finite(value) for value in state.values()):
        return False
    if len(msg.velocity) not in (0, len(EXPECTED_DOF_NAMES)):
        return False
    return True


def clock_value(msg: Clock) -> int:
    """Return the clock message as integer nanoseconds."""
    return int(msg.clock.sec) * 1_000_000_000 + int(msg.clock.nanosec)


def max_deviation(final: dict[str, float], reference: dict[str, float], names) -> float:
    """Largest |final - reference| over the given joint names."""
    return max((abs(final[n] - reference[n]) for n in names if n in final and n in reference), default=float("inf"))


def max_step(states: list[dict[str, float]]) -> float:
    """Largest per-sample position change over the last two state samples."""
    if len(states) < 2:
        return float("inf")
    return max_deviation(states[-1], states[-2], EXPECTED_DOF_NAMES)


def euclidean(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def quat_norm(quaternion: dict) -> float:
    """Euclidean norm of a quaternion dict with x/y/z/w keys."""
    return math.sqrt(sum(float(quaternion[key]) ** 2 for key in ("x", "y", "z", "w")))


def quaternion_is_admissible(quaternion: dict) -> tuple[bool, str]:
    """Finite components and a non-near-zero norm; no clamping is applied."""
    for key in ("x", "y", "z", "w"):
        if not finite(quaternion.get(key)):
            return False, f"nonfinite quaternion component {key}"
    norm = quat_norm(quaternion)
    if norm < QUAT_NORM_MIN:
        return False, f"near-zero quaternion (norm {norm:.3e} < {QUAT_NORM_MIN:.3e})"
    return True, "ok"


def quat_normalize(quaternion: dict) -> dict:
    """Return a normalized copy of an admissible quaternion."""
    norm = quat_norm(quaternion)
    return {key: float(quaternion[key]) / norm for key in ("x", "y", "z", "w")}


def quat_angle_between(first: dict, second: dict) -> float:
    """Rotation angle (rad) between two quaternions, in [0, pi]."""
    a = quat_normalize(first)
    b = quat_normalize(second)
    dot = sum(a[key] * b[key] for key in ("x", "y", "z", "w"))
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(dot))))


def state_joint_message(state: dict[str, float]) -> JointState:
    """Full 8-DOF JointState message built from a live state sample."""
    message = JointState()
    message.name = list(EXPECTED_DOF_NAMES)
    message.position = [float(state[name]) for name in EXPECTED_DOF_NAMES]
    return message


def wait_future(future, deadline: float) -> bool:
    """Block (wall clock) until the future is done or the deadline passes."""
    while not future.done():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


class TopicMonitor:
    """Collects bounded state evidence from the subscriptions."""

    def __init__(self) -> None:
        self.clock_values: list[int] = []
        self.isaac_states: list[dict[str, float]] = []
        self.isaac_receipts: list[float] = []
        self.isaac_invalid = 0
        self.broadcast_states: list[dict[str, float]] = []
        self.command_count = 0
        self.command_invalid = 0
        self.command_unknown_names: list[str] = []
        self.command_min: float | None = None
        self.command_max: float | None = None

    def on_clock(self, msg: Clock) -> None:
        self.clock_values.append(clock_value(msg))

    def on_isaac_state(self, msg: JointState) -> None:
        state = positions_by_name(msg)
        if not is_valid_state(state, msg):
            self.isaac_invalid += 1
            return
        self.isaac_states.append(state)
        self.isaac_receipts.append(time.monotonic())

    def on_broadcast_state(self, msg: JointState) -> None:
        state = positions_by_name(msg)
        if set(state) == set(EXPECTED_DOF_NAMES) and all(
            finite(value) for value in state.values()
        ):
            self.broadcast_states.append(state)

    def on_command(self, msg: JointState) -> None:
        self.command_count += 1
        names = [str(name) for name in msg.name]
        values = [float(value) for value in msg.position]
        if len(names) != len(values) or not all(finite(value) for value in values):
            self.command_invalid += 1
            return
        for name in names:
            if name not in EXPECTED_DOF_NAMES and name not in self.command_unknown_names:
                self.command_unknown_names.append(name)
        if values:
            low, high = min(values), max(values)
            self.command_min = low if self.command_min is None else min(self.command_min, low)
            self.command_max = high if self.command_max is None else max(self.command_max, high)


def call_fk(fk_client, state: dict[str, float]) -> dict | None:
    """Bounded GetPositionFK: link6 origin in base_link from a live state."""
    if fk_client is None or not fk_client.service_is_ready():
        return None
    request = GetPositionFK.Request()
    request.header.frame_id = BASE_FRAME
    request.fk_link_names = [TIP_LINK]
    message = state_joint_message(state)
    request.robot_state.joint_state.name = list(message.name)
    request.robot_state.joint_state.position = list(message.position)
    future = fk_client.call_async(request)
    if not wait_future(future, time.monotonic() + SERVICE_CALL_TIMEOUT_S):
        return None
    response = future.result()
    if response is None or int(response.error_code.val) != SUCCESS_CODE:
        return None
    if len(response.pose_stamped) != 1:
        return None
    stamped = response.pose_stamped[0]
    pose = stamped.pose
    position = {
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "z": float(pose.position.z),
    }
    quaternion = {
        "x": float(pose.orientation.x),
        "y": float(pose.orientation.y),
        "z": float(pose.orientation.z),
        "w": float(pose.orientation.w),
    }
    if not all(finite(value) for value in position.values()):
        return None
    admissible, _reason = quaternion_is_admissible(quaternion)
    if not admissible:
        return None
    return {
        "frame_id": str(stamped.header.frame_id),
        "link_name": TIP_LINK,
        "position": position,
        "orientation": quat_normalize(quaternion),
        "quaternion_norm": round(quat_norm(quaternion), 12),
    }


def displaced_pose(pose: dict, axis: str, delta: float) -> dict:
    """Copy of a pose with a recorded displacement along one base_link axis."""
    shifted = {"position": dict(pose["position"]), "orientation": dict(pose["orientation"])}
    shifted["position"][axis] = float(shifted["position"][axis]) + float(delta)
    return shifted


def ik_preflight(ik_client, seed_state: dict[str, float], pose: dict) -> dict:
    """Bounded reachability proof for a pose via collision-aware GetPositionIK."""
    if ik_client is None or not ik_client.service_is_ready():
        return {"proved": False, "reason": "ik service not ready"}
    request = GetPositionIK.Request()
    request.ik_request.group_name = PLANNING_GROUP
    request.ik_request.ik_link_name = TIP_LINK
    request.ik_request.pose_stamped.header.frame_id = BASE_FRAME
    request.ik_request.pose_stamped.pose.position.x = float(pose["position"]["x"])
    request.ik_request.pose_stamped.pose.position.y = float(pose["position"]["y"])
    request.ik_request.pose_stamped.pose.position.z = float(pose["position"]["z"])
    for key in ("x", "y", "z", "w"):
        setattr(
            request.ik_request.pose_stamped.pose.orientation,
            key,
            float(pose["orientation"][key]),
        )
    message = state_joint_message(seed_state)
    request.ik_request.robot_state.joint_state.name = list(message.name)
    request.ik_request.robot_state.joint_state.position = list(message.position)
    request.ik_request.avoid_collisions = True
    request.ik_request.timeout = Duration(
        sec=int(SERVICE_CALL_TIMEOUT_S), nanosec=0
    )
    future = ik_client.call_async(request)
    if not wait_future(future, time.monotonic() + 2.0 * SERVICE_CALL_TIMEOUT_S):
        return {"proved": False, "reason": "ik call exceeded the wall deadline"}
    response = future.result()
    if response is None:
        return {"proved": False, "reason": "ik response missing"}
    if int(response.error_code.val) != SUCCESS_CODE:
        return {
            "proved": False,
            "reason": "ik error code {code}".format(code=int(response.error_code.val)),
            "error_code": int(response.error_code.val),
        }
    solution = {
        str(name): float(value)
        for name, value in zip(
            response.solution.joint_state.name, response.solution.joint_state.position
        )
    }
    arm = {name: solution[name] for name in ARM_JOINT_NAMES if name in solution}
    if set(arm) != set(ARM_JOINT_NAMES) or not all(finite(value) for value in arm.values()):
        return {"proved": False, "reason": "ik solution incomplete or nonfinite"}
    max_excess = 0.0
    for name, value in arm.items():
        lower, upper = EXECUTION_POSITION_LIMITS[name]
        max_excess = max(max_excess, lower + ADMISSION_MARGIN - value, value - (upper - ADMISSION_MARGIN), 0.0)
    if max_excess > 0.0:
        return {
            "proved": False,
            "reason": "ik solution outside the execution limits with margin",
            "max_margin_excess": round(max_excess, 9),
        }
    return {
        "proved": True,
        "solution_arm": {name: round(value, 9) for name, value in arm.items()},
        "error_code": SUCCESS_CODE,
    }


def build_pose_request(pose: dict) -> MotionPlanRequest:
    """Build the MoveGroup request for the link6 pose goal in base_link.

    Exactly one goal position constraint (SPHERE around the requested
    position) and one goal orientation constraint (absolute tolerances around
    the requested normalized quaternion); the start_state stays EMPTY:
    move_group substitutes the current state from its planning-scene monitor.
    """
    request = MotionPlanRequest()
    request.group_name = PLANNING_GROUP
    request.allowed_planning_time = ALLOWED_PLANNING_TIME
    request.num_planning_attempts = NUM_PLANNING_ATTEMPTS
    request.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
    request.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING
    goal = Constraints()

    position = PositionConstraint()
    position.header.frame_id = BASE_FRAME
    position.link_name = TIP_LINK
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [GOAL_POSITION_TOLERANCE_M]
    position.constraint_region.primitives.append(sphere)
    center = Pose()
    center.position.x = float(pose["position"]["x"])
    center.position.y = float(pose["position"]["y"])
    center.position.z = float(pose["position"]["z"])
    center.orientation.w = 1.0
    position.constraint_region.primitive_poses.append(center)
    position.weight = 1.0
    goal.position_constraints.append(position)

    orientation = OrientationConstraint()
    orientation.header.frame_id = BASE_FRAME
    orientation.link_name = TIP_LINK
    for key in ("x", "y", "z", "w"):
        setattr(orientation.orientation, key, float(pose["orientation"][key]))
    orientation.absolute_x_axis_tolerance = GOAL_ORIENTATION_TOLERANCE_RAD
    orientation.absolute_y_axis_tolerance = GOAL_ORIENTATION_TOLERANCE_RAD
    orientation.absolute_z_axis_tolerance = GOAL_ORIENTATION_TOLERANCE_RAD
    orientation.weight = 1.0
    goal.orientation_constraints.append(orientation)

    request.goal_constraints.append(goal)
    return request


def admit_pose_request(request: MotionPlanRequest, requested_pose: dict) -> dict:
    """Strict caller-side admission BEFORE the send; no MoveIt clamp reliance.

    Checks the ACTUAL message fields that will be sent: finite position,
    finite/near-zero-rejected/normalized quaternion, exact frame/link on both
    constraints, tolerance bounds, and no simultaneous joint target.
    """
    problems: list[str] = []
    if len(request.goal_constraints) != 1:
        return {"passed": False, "problems": ["exactly one goal constraint set required"]}
    goal = request.goal_constraints[0]
    no_joint_target = not goal.joint_constraints
    if not no_joint_target:
        problems.append("no simultaneous joint target allowed")

    radius = None
    max_abs_tolerance = None
    center_error = None
    quaternion_norm = None
    normalized = False
    if len(goal.position_constraints) != 1:
        problems.append("exactly one position constraint required")
    else:
        position = goal.position_constraints[0]
        if position.header.frame_id != BASE_FRAME:
            problems.append("position constraint frame must be " + BASE_FRAME)
        if position.link_name != TIP_LINK:
            problems.append("position constraint link must be " + TIP_LINK)
        if (
            len(position.constraint_region.primitives) != 1
            or position.constraint_region.primitives[0].type != SolidPrimitive.SPHERE
        ):
            problems.append("exactly one sphere primitive required")
        else:
            radius = float(position.constraint_region.primitives[0].dimensions[0])
            if not finite(radius) or radius <= 0.0 or radius > POSITION_TOLERANCE_M:
                problems.append(
                    f"sphere radius {radius} outside (0, {POSITION_TOLERANCE_M}]"
                )
        if len(position.constraint_region.primitive_poses) != 1:
            problems.append("exactly one sphere center required")
        else:
            region = position.constraint_region.primitive_poses[0]
            center = {
                "x": float(region.position.x),
                "y": float(region.position.y),
                "z": float(region.position.z),
            }
            if not all(finite(value) for value in center.values()):
                problems.append("nonfinite sphere center")
            else:
                center_error = round(
                    euclidean(
                        center["x"],
                        center["y"],
                        center["z"],
                        float(requested_pose["position"]["x"]),
                        float(requested_pose["position"]["y"]),
                        float(requested_pose["position"]["z"]),
                    ),
                    12,
                )
                if center_error > LIMIT_EPSILON:
                    problems.append("sphere center does not match the requested position")

    if len(goal.orientation_constraints) != 1:
        problems.append("exactly one orientation constraint required")
    else:
        orientation = goal.orientation_constraints[0]
        if orientation.header.frame_id != BASE_FRAME:
            problems.append("orientation constraint frame must be " + BASE_FRAME)
        if orientation.link_name != TIP_LINK:
            problems.append("orientation constraint link must be " + TIP_LINK)
        tolerances = {
            "x": float(orientation.absolute_x_axis_tolerance),
            "y": float(orientation.absolute_y_axis_tolerance),
            "z": float(orientation.absolute_z_axis_tolerance),
        }
        max_abs_tolerance = max(tolerances.values()) if all(
            finite(value) for value in tolerances.values()
        ) else None
        if max_abs_tolerance is None:
            problems.append("nonfinite orientation tolerance")
        elif not all(0.0 < value <= ORIENTATION_TOLERANCE_RAD for value in tolerances.values()):
            problems.append(
                f"orientation tolerances outside (0, {ORIENTATION_TOLERANCE_RAD}]"
            )
        sent_quaternion = {
            "x": float(orientation.orientation.x),
            "y": float(orientation.orientation.y),
            "z": float(orientation.orientation.z),
            "w": float(orientation.orientation.w),
        }
        admissible, reason = quaternion_is_admissible(sent_quaternion)
        if not admissible:
            problems.append("orientation constraint quaternion rejected: " + reason)
        else:
            quaternion_norm = round(quat_norm(sent_quaternion), 12)
            normalized = abs(quaternion_norm - 1.0) <= QUAT_NORM_EPS
            if not normalized:
                problems.append("orientation constraint quaternion is not normalized")
            angle = quat_angle_between(sent_quaternion, requested_pose["orientation"])
            if angle > 1.0e-6:
                problems.append("orientation constraint does not match the requested quaternion")

    requested_quaternion = dict(requested_pose["orientation"])
    admissible, reason = quaternion_is_admissible(requested_quaternion)
    if not admissible:
        problems.append("requested quaternion rejected: " + reason)
    if not all(
        finite(float(requested_pose["position"][key])) for key in ("x", "y", "z")
    ):
        problems.append("nonfinite requested position")

    return {
        "passed": not problems,
        "problems": problems,
        "frame": BASE_FRAME,
        "link": TIP_LINK,
        "no_joint_target": no_joint_target,
        "sphere_radius_m": radius,
        "max_orientation_tolerance_rad": max_abs_tolerance,
        "sphere_center_error_m": center_error,
        "quaternion_norm": quaternion_norm,
        "quaternion_normalized": normalized,
    }


def serialize_pose(pose: dict) -> dict:
    """Rounded, JSON-safe pose record."""
    return {
        "frame_id": BASE_FRAME,
        "link": TIP_LINK,
        "position": {key: round(float(pose["position"][key]), 9) for key in ("x", "y", "z")},
        "orientation": {
            key: round(float(pose["orientation"][key]), 12) for key in ("x", "y", "z", "w")
        },
    }


def serialize_request(request: MotionPlanRequest) -> dict:
    """Full request record for the artifact."""
    goal = request.goal_constraints[0]
    position = goal.position_constraints[0]
    orientation = goal.orientation_constraints[0]
    region = position.constraint_region.primitive_poses[0]
    return {
        "group_name": request.group_name,
        "plan_only": False,
        "allowed_planning_time_s": float(request.allowed_planning_time),
        "num_planning_attempts": int(request.num_planning_attempts),
        "max_velocity_scaling_factor": float(request.max_velocity_scaling_factor),
        "max_acceleration_scaling_factor": float(request.max_acceleration_scaling_factor),
        "joint_constraint_count": len(goal.joint_constraints),
        "position_constraint": {
            "frame_id": str(position.header.frame_id),
            "link_name": str(position.link_name),
            "primitive_type": "SPHERE",
            "sphere_radius_m": float(position.constraint_region.primitives[0].dimensions[0]),
            "center_position": {
                "x": round(float(region.position.x), 9),
                "y": round(float(region.position.y), 9),
                "z": round(float(region.position.z), 9),
            },
        },
        "orientation_constraint": {
            "frame_id": str(orientation.header.frame_id),
            "link_name": str(orientation.link_name),
            "orientation": {
                "x": round(float(orientation.orientation.x), 12),
                "y": round(float(orientation.orientation.y), 12),
                "z": round(float(orientation.orientation.z), 12),
                "w": round(float(orientation.orientation.w), 12),
            },
            "absolute_x_axis_tolerance": float(orientation.absolute_x_axis_tolerance),
            "absolute_y_axis_tolerance": float(orientation.absolute_y_axis_tolerance),
            "absolute_z_axis_tolerance": float(orientation.absolute_z_axis_tolerance),
        },
    }


def inspect_planned_trajectory(result) -> dict:
    """Check the planned trajectory: >=2 points, finite, inside the limits."""
    trajectory = result.planned_trajectory.joint_trajectory
    waypoint_names = [str(name) for name in trajectory.joint_names]
    point_count = len(trajectory.points)
    # Normalize every waypoint column order onto the canonical arm order.
    indices = [waypoint_names.index(name) for name in ARM_JOINT_NAMES if name in waypoint_names]
    names_normalized = [name for name in ARM_JOINT_NAMES if name in waypoint_names]
    nonfinite = 0
    max_excess = {name: 0.0 for name in names_normalized}
    for point in trajectory.points:
        positions = list(point.positions)
        for slot, name in zip(indices, names_normalized):
            if slot >= len(positions):
                nonfinite += 1
                continue
            value = float(positions[slot])
            if not finite(value):
                nonfinite += 1
                continue
            lower, upper = EXECUTION_POSITION_LIMITS[name]
            excess = max(lower - value, value - upper, 0.0)
            max_excess[name] = max(max_excess[name], excess)
    waypoints_within_limits = all(
        excess <= LIMIT_EPSILON for excess in max_excess.values()
    )
    return {
        "trajectory_joint_names": waypoint_names,
        "waypoint_count": point_count,
        "min_waypoints_met": point_count >= MIN_WAYPOINTS,
        "arm_joints_exact": set(waypoint_names) == set(ARM_JOINT_NAMES),
        "nonfinite_waypoints": nonfinite,
        "waypoint_max_limit_excess": {
            name: round(excess, 9) for name, excess in max_excess.items()
        },
        "waypoints_within_limits": bool(
            names_normalized and waypoints_within_limits and nonfinite == 0
        ),
    }


def execute_pose_goal(
    monitor: TopicMonitor,
    action_client,
    fk_client,
    request: MotionPlanRequest,
    requested_pose: dict,
    label: str,
    goal_timeout: float,
    settle_timeout: float,
) -> dict:
    """Send one plan-and-execute MoveGroup POSE goal and verify it via FK."""
    summary: dict = {
        "label": label,
        "requested_pose": serialize_pose(requested_pose),
        "request": serialize_request(request),
        "tip_link": TIP_LINK,
        "tcp_status": TCP_STATUS,
    }
    goal = MoveGroup.Goal()
    goal.request = request
    goal.planning_options.plan_only = False
    goal.planning_options.replan = False

    send_index = len(monitor.isaac_states)
    future = action_client.send_goal_async(goal)
    accepted_deadline = time.monotonic() + min(15.0, goal_timeout)
    if not wait_future(future, accepted_deadline) or not future.done():
        summary["error"] = f"{label}: goal response not received in time"
        return summary
    goal_handle = future.result()
    if not goal_handle.accepted:
        summary["error"] = f"{label}: goal rejected by move_group"
        summary["accepted"] = False
        return summary
    summary["accepted"] = True

    hard_deadline = time.monotonic() + goal_timeout
    result_future = goal_handle.get_result_async()
    if not wait_future(result_future, hard_deadline):
        summary["timed_out"] = True
        summary["cancel"] = {"attempted": True}
        try:
            cancel_future = goal_handle.cancel_goal_async()
            if wait_future(cancel_future, 10.0) and cancel_future.done():
                cancel_response = cancel_future.result()
                summary["cancel"]["return_code"] = int(cancel_response.return_code)
        except Exception as exc:  # cancellation is best effort, still a failure
            summary["cancel"]["error"] = str(exc)
        summary["error"] = f"{label}: action exceeded the {goal_timeout}s wall deadline (cancelled)"
        return summary

    action_result = result_future.result().result
    summary["status"] = int(result_future.result().status)
    summary["status_name"] = {
        GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
        GoalStatus.STATUS_CANCELED: "CANCELED",
        GoalStatus.STATUS_ABORTED: "ABORTED",
    }.get(summary["status"], f"UNKNOWN({summary['status']})")
    summary["error_code"] = int(action_result.error_code.val)
    summary["error_code_name"] = ERROR_CODE_NAMES.get(
        summary["error_code"], f"UNKNOWN({summary['error_code']})"
    )
    summary["planning_time_s"] = round(float(action_result.planning_time), 6)

    plan_check = inspect_planned_trajectory(action_result)
    summary["planned_trajectory"] = plan_check
    summary["planned_point_count"] = plan_check["waypoint_count"]

    action_ok = (
        summary["status"] == STATUS_SUCCEEDED
        and summary["error_code"] == SUCCESS_CODE
        and plan_check["min_waypoints_met"]
        and plan_check["arm_joints_exact"]
        and plan_check["waypoints_within_limits"]
    )

    # Bounded convergence wait, verified through the LIVE /compute_fk service:
    # the tip pose recomputed from the latest live state must stay inside the
    # request tolerances AND the arm must be at rest for SETTLE_STABLE_SAMPLES
    # consecutive state samples.
    settle_deadline = time.monotonic() + settle_timeout
    fk_result: dict | None = None
    fk_calls = 0
    fk_failures = 0
    last_fk_index = -1
    last_fk_time = 0.0
    stable_count = 0
    final_state: dict[str, float] | None = None
    min_observed_step = float("inf")
    while time.monotonic() < settle_deadline:
        states = monitor.isaac_states
        receipts = monitor.isaac_receipts
        if states and time.monotonic() - receipts[-1] <= STATE_FRESH_SECONDS:
            final_state = states[-1]
            observed_step = max_step(states)
            min_observed_step = min(min_observed_step, observed_step)
            needs_fk = (
                fk_result is None
                or (
                    len(states) != last_fk_index
                    and time.monotonic() - last_fk_time >= FK_CALL_SPACING_S
                )
            )
            if needs_fk:
                last_fk_index = len(states)
                last_fk_time = time.monotonic()
                fk_calls += 1
                fresh_fk = call_fk(fk_client, final_state)
                if fresh_fk is None:
                    fk_failures += 1
                    fk_result = None
                else:
                    fk_result = fresh_fk
                    fk_result["position_error_m"] = round(
                        euclidean(
                            fk_result["position"]["x"],
                            fk_result["position"]["y"],
                            fk_result["position"]["z"],
                            float(requested_pose["position"]["x"]),
                            float(requested_pose["position"]["y"]),
                            float(requested_pose["position"]["z"]),
                        ),
                        9,
                    )
                    fk_result["orientation_error_rad"] = round(
                        quat_angle_between(fk_result["orientation"], requested_pose["orientation"]),
                        9,
                    )
            within = bool(
                fk_result is not None
                and fk_result["position_error_m"] <= POSITION_TOLERANCE_M
                and fk_result["orientation_error_rad"] <= ORIENTATION_TOLERANCE_RAD
            )
            stable_count = stable_count + 1 if (
                within and observed_step <= SETTLE_STABLE_DELTA_MAX
            ) else 0
            if stable_count >= SETTLE_STABLE_SAMPLES:
                break
        else:
            stable_count = 0
        time.sleep(0.2)
    summary["settled_at_rest"] = bool(stable_count >= SETTLE_STABLE_SAMPLES)
    summary["final_max_step"] = (
        round(max_step(monitor.isaac_states), 9) if monitor.isaac_states else None
    )
    summary["min_observed_max_step"] = (
        round(min_observed_step, 9) if math.isfinite(min_observed_step) else None
    )

    verification = {
        "passed": bool(
            fk_result is not None
            and summary["settled_at_rest"]
            and fk_result["position_error_m"] <= POSITION_TOLERANCE_M
            and fk_result["orientation_error_rad"] <= ORIENTATION_TOLERANCE_RAD
        ),
        "fk_pose": (
            {
                "frame_id": fk_result["frame_id"],
                "position": {
                    key: round(float(fk_result["position"][key]), 9)
                    for key in ("x", "y", "z")
                },
                "orientation": {
                    key: round(float(fk_result["orientation"][key]), 12)
                    for key in ("x", "y", "z", "w")
                },
            }
            if fk_result is not None
            else None
        ),
        "position_error_m": fk_result["position_error_m"] if fk_result is not None else None,
        "orientation_error_rad": (
            fk_result["orientation_error_rad"] if fk_result is not None else None
        ),
        "fk_source_state": (
            {name: round(float(final_state[name]), 9) for name in EXPECTED_DOF_NAMES}
            if final_state is not None
            else None
        ),
        "fk_calls": fk_calls,
        "fk_failures": fk_failures,
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
    }
    summary["fk_verification"] = verification

    post_states = monitor.isaac_states[send_index:]
    summary["post_command_state_messages"] = len(post_states)
    summary["command_messages"] = monitor.command_count

    summary["action_and_plan_passed"] = bool(action_ok)
    summary["feedback_passed"] = bool(
        action_ok
        and verification["passed"]
        and len(post_states) >= MIN_POST_COMMAND_STATE_MESSAGES
        and monitor.command_count >= MIN_COMMAND_MESSAGES
    )
    summary["passed"] = summary["feedback_passed"]
    if not summary["passed"]:
        summary["error"] = summary.get(
            "error",
            f"{label}: live FK did not confirm the requested pose within tolerance",
        )
    return summary


def command_stats(monitor: TopicMonitor) -> dict:
    """Snapshot of the observed /isaac_joint_command evidence."""
    return {
        "count": monitor.command_count,
        "invalid": monitor.command_invalid,
        "unknown_names": list(monitor.command_unknown_names),
        "min_position": monitor.command_min,
        "max_position": monitor.command_max,
    }


def run_probe(args: argparse.Namespace) -> dict:
    """Run the bounded probe; return the result dict (never raises)."""
    result: dict = {
        "schema": SCHEMA,
        "phase": PHASE,
        "tip_link": TIP_LINK,
        "tcp_status": TCP_STATUS,
        "tcp_note": (
            "model contract: no tool0 or dedicated TCP frame exists in the "
            "pinned piper_l model, so the pose target addresses the link6 "
            "origin in base_link and the TCP stays unresolved"
        ),
        "planning_group": PLANNING_GROUP,
        "base_frame": BASE_FRAME,
        "topics": {
            "clock": TOPIC_CLOCK,
            "isaac_joint_states": TOPIC_JOINT_STATES,
            "broadcast_joint_states": TOPIC_BROADCAST_STATES,
            "joint_commands": TOPIC_JOINT_COMMANDS,
        },
        "actions": {
            "arm_follow": ACTION_ARM_FOLLOW,
            "move_group_candidates": list(MOVE_GROUP_ACTION_CANDIDATES),
        },
        "services": {
            "list_controllers": SERVICE_LIST_CONTROLLERS,
            "compute_fk": SERVICE_FK,
            "get_position_ik_candidates": list(SERVICE_IK_CANDIDATES),
        },
        "expected_dof_names": list(EXPECTED_DOF_NAMES),
        "execution_position_limits": {
            name: list(limits) for name, limits in EXECUTION_POSITION_LIMITS.items()
        },
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
        "goal_position_tolerance_m": GOAL_POSITION_TOLERANCE_M,
        "goal_orientation_tolerance_rad": GOAL_ORIENTATION_TOLERANCE_RAD,
        "pose_displacement_m": POSE_DISPLACEMENT_M,
        "displacement_axes": list(DISPLACEMENT_AXES),
        "quat_norm_min": QUAT_NORM_MIN,
        "quat_norm_eps": QUAT_NORM_EPS,
        "admission_margin": ADMISSION_MARGIN,
        "allowed_planning_time_s": ALLOWED_PLANNING_TIME,
        "scaling_factor": MAX_VELOCITY_SCALING,
        "settle_stable_samples": SETTLE_STABLE_SAMPLES,
        "settle_stable_delta_max": SETTLE_STABLE_DELTA_MAX,
        "exact_state_match_tolerance": EXACT_STATE_MATCH_TOLERANCE,
        "min_baseline_state_messages": MIN_BASELINE_STATE_MESSAGES,
        "readiness": {},
        "fk_request": None,
        "fk_source_state": None,
        "target_fk": None,
        "displacement": {"applied": False},
        "admission": {"passed": False},
        "outbound": {"passed": False},
        "return": {"passed": False},
        "passed": False,
    }

    rclpy.init()
    node = rclpy.create_node("p1_real_pose_probe")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        monitor = TopicMonitor()
        node.create_subscription(Clock, TOPIC_CLOCK, monitor.on_clock, qos_sensor)
        node.create_subscription(
            JointState, TOPIC_JOINT_STATES, monitor.on_isaac_state, qos_sensor
        )
        node.create_subscription(
            JointState, TOPIC_BROADCAST_STATES, monitor.on_broadcast_state, qos_state
        )
        node.create_subscription(
            JointState, TOPIC_JOINT_COMMANDS, monitor.on_command, qos_state
        )
        list_clients = node.create_client(ListControllers, SERVICE_LIST_CONTROLLERS)
        arm_action = ActionClient(node, FollowJointTrajectory, ACTION_ARM_FOLLOW)
        move_group_clients = {
            name: ActionClient(node, MoveGroup, name) for name in MOVE_GROUP_ACTION_CANDIDATES
        }
        fk_client = node.create_client(GetPositionFK, SERVICE_FK)
        ik_clients = {
            name: node.create_client(GetPositionIK, name) for name in SERVICE_IK_CANDIDATES
        }

        # ---------------------------------------------------- readiness phase
        controllers: dict | None = None
        controllers_future = None
        arm_action_ready = False
        move_group_name = None
        fk_ready = False
        ik_name = None
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if controllers is None and controllers_future is None and list_clients.service_is_ready():
                controllers_future = list_clients.call_async(ListControllers.Request())
            if controllers_future is not None and controllers_future.done():
                response = controllers_future.result()
                if response is not None:
                    controllers = {c.name: c.state for c in response.controller}
                controllers_future = None
            if not arm_action_ready:
                arm_action_ready = arm_action.wait_for_server(timeout_sec=0.5)
            if move_group_name is None:
                for name in MOVE_GROUP_ACTION_CANDIDATES:
                    if move_group_clients[name].wait_for_server(timeout_sec=0.5):
                        move_group_name = name
                        break
            fk_ready = fk_ready or fk_client.service_is_ready()
            if ik_name is None:
                for name in SERVICE_IK_CANDIDATES:
                    if ik_clients[name].service_is_ready():
                        ik_name = name
                        break
            states = monitor.isaac_states
            ready = (
                controllers is not None
                and all(controllers.get(name) == "active" for name in REQUIRED_CONTROLLERS)
                and arm_action_ready
                and move_group_name is not None
                and fk_ready
                and len(states) >= MIN_BASELINE_STATE_MESSAGES
                and bool(monitor.broadcast_states)
                and any(
                    later > earlier
                    for earlier, later in zip(monitor.clock_values, monitor.clock_values[1:])
                )
                and max_deviation(states[-1], monitor.broadcast_states[-1], EXPECTED_DOF_NAMES)
                <= EXACT_STATE_MATCH_TOLERANCE
            )
            if ready:
                break
            time.sleep(0.2)

        readiness = {
            "clock_messages": len(monitor.clock_values),
            "clock_increasing": any(
                later > earlier
                for earlier, later in zip(monitor.clock_values, monitor.clock_values[1:])
            ),
            "isaac_joint_state_messages": len(monitor.isaac_states),
            "isaac_invalid_state_messages": monitor.isaac_invalid,
            "broadcast_joint_state_messages": len(monitor.broadcast_states),
            "controllers": controllers,
            "controllers_active": bool(
                controllers
                and all(controllers.get(name) == "active" for name in REQUIRED_CONTROLLERS)
            ),
            "arm_follow_action_ready": arm_action_ready,
            "move_group_action_name": move_group_name,
            "fk_service_ready": fk_ready,
            "ik_service_name": ik_name,
            "ik_service_ready": ik_name is not None,
        }
        result["readiness"] = readiness
        baseline_ready = bool(
            readiness["clock_increasing"]
            and len(monitor.isaac_states) >= MIN_BASELINE_STATE_MESSAGES
            and readiness["broadcast_joint_state_messages"] > 0
            and readiness["controllers_active"]
            and readiness["arm_follow_action_ready"]
            and readiness["move_group_action_name"]
            and readiness["fk_service_ready"]
        )
        if not baseline_ready:
            result["error"] = "readiness phase failed: move_group stack, /compute_fk or live bridge not ready"
            return result

        # --------------------------------------------- at-rest baseline capture
        settle_deadline = time.monotonic() + args.settle_timeout
        stable_count = 0
        while time.monotonic() < settle_deadline:
            states = monitor.isaac_states
            receipts = monitor.isaac_receipts
            if states and time.monotonic() - receipts[-1] <= STATE_FRESH_SECONDS:
                stable_count = (
                    stable_count + 1 if max_step(states) <= SETTLE_STABLE_DELTA_MAX else 0
                )
                if stable_count >= SETTLE_STABLE_SAMPLES:
                    break
            else:
                stable_count = 0
            time.sleep(0.2)
        result["baseline_at_rest"] = bool(stable_count >= SETTLE_STABLE_SAMPLES)
        states = monitor.isaac_states
        receipts = monitor.isaac_receipts
        result["command_stats"] = command_stats(monitor)
        if not result["baseline_at_rest"] or not states:
            result["error"] = "the arm never settled at rest before the baseline capture"
            return result

        baseline = dict(states[-1])
        result["baseline"] = {name: round(float(baseline[name]), 9) for name in EXPECTED_DOF_NAMES}
        result["baseline_state_messages"] = len(states)
        fresh = time.monotonic() - receipts[-1] <= STATE_FRESH_SECONDS
        broadcast_match = (
            max_deviation(baseline, monitor.broadcast_states[-1], EXPECTED_DOF_NAMES)
            <= EXACT_STATE_MATCH_TOLERANCE
        )
        result["baseline_fresh"] = fresh
        result["baseline_matches_broadcast"] = broadcast_match
        if not (fresh and broadcast_match):
            result["error"] = (
                "baseline state stale/nonfinite/incomplete before the target derivation"
            )
            return result

        # ---------------------------------------------------- FK target source
        result["fk_request"] = {
            "service": SERVICE_FK,
            "frame_id": BASE_FRAME,
            "fk_link_names": [TIP_LINK],
            "joint_names": list(EXPECTED_DOF_NAMES),
        }
        result["fk_source_state"] = {
            name: round(float(baseline[name]), 9) for name in EXPECTED_DOF_NAMES
        }
        target_fk = call_fk(fk_client, baseline)
        result["target_fk"] = (
            {
                "frame_id": target_fk["frame_id"],
                "position": {
                    key: round(float(target_fk["position"][key]), 9)
                    for key in ("x", "y", "z")
                },
                "orientation": {
                    key: round(float(target_fk["orientation"][key]), 12)
                    for key in ("x", "y", "z", "w")
                },
                "quaternion_norm": target_fk["quaternion_norm"],
            }
            if target_fk is not None
            else None
        )
        if target_fk is None:
            result["error"] = "the /compute_fk target derivation from the baseline failed"
            return result

        # ------------------------------------------- displacement preflight
        # Tiny recorded displacement, applied ONLY when the bounded
        # GetPositionIK preflight proves the displaced pose reachable.
        ik_client = ik_clients[ik_name] if ik_name is not None else None
        displacement = {
            "applied": False,
            "axis": None,
            "delta_m": 0.0,
            "reason": "no preflight attempted",
            "attempts": [],
        }
        target_pose = {
            "position": dict(target_fk["position"]),
            "orientation": dict(target_fk["orientation"]),
        }
        for axis in DISPLACEMENT_AXES:
            candidate = displaced_pose(target_fk, axis, POSE_DISPLACEMENT_M)
            evidence = ik_preflight(ik_client, baseline, candidate)
            displacement["attempts"].append(
                {"axis": axis, "delta_m": POSE_DISPLACEMENT_M, **evidence}
            )
            if evidence["proved"]:
                displacement["applied"] = True
                displacement["axis"] = axis
                displacement["delta_m"] = POSE_DISPLACEMENT_M
                displacement["reason"] = (
                    f"+{POSE_DISPLACEMENT_M} m along base_link {axis} proved "
                    "reachable by the bounded collision-aware GetPositionIK preflight"
                )
                target_pose = candidate
                break
        if not displacement["applied"]:
            displacement["reason"] = (
                "exact FK pose used: no displacement axis was proved reachable"
                if displacement["attempts"]
                else "exact FK pose used: the GetPositionIK service was not ready"
            )
        result["displacement"] = displacement

        # ------------------------------------------------- strict admission
        request = build_pose_request(target_pose)
        admission = admit_pose_request(request, target_pose)
        result["admission"] = admission
        if not admission["passed"]:
            result["error"] = "pose goal admission failed: " + "; ".join(admission["problems"])
            return result

        # ------------------------------------------------- outbound execution
        outbound = execute_pose_goal(
            monitor,
            move_group_clients[move_group_name],
            fk_client,
            request,
            target_pose,
            "outbound",
            args.goal_timeout,
            args.settle_timeout,
        )
        result["outbound"] = outbound
        if not outbound.get("passed"):
            return result

        # ------------------------------------------------ return to baseline
        # The return target is the FK of the ORIGINAL baseline (the exact
        # baseline FK pose, no displacement), sent as a second REAL MoveGroup
        # pose goal - never a hardcoded joint command.
        return_request = build_pose_request(target_fk)
        return_admission = admit_pose_request(return_request, target_fk)
        result["return_admission"] = return_admission
        if not return_admission["passed"]:
            result["error"] = "return admission failed: " + "; ".join(return_admission["problems"])
            return result
        return_summary = execute_pose_goal(
            monitor,
            move_group_clients[move_group_name],
            fk_client,
            return_request,
            target_fk,
            "return",
            args.goal_timeout,
            args.settle_timeout,
        )
        result["return"] = return_summary
        result["command_stats"] = command_stats(monitor)
        result["host_correlation"] = {
            "command_topic": TOPIC_JOINT_COMMANDS,
            "probe_command_count": monitor.command_count,
            "host_evidence_file": "host-pose-bridge.json",
            "correlated_by": "tools/smoke_p1_real_pose.sh aggregates the host "
            "bridge totals (applied == received, clamped 0, no unknown names)",
            "ros_domain_id_env": os.environ.get("ROS_DOMAIN_ID"),
        }
        result["passed"] = bool(outbound["passed"] and return_summary["passed"])
        if not result["passed"]:
            result["error"] = result.get("error") or return_summary.get(
                "error", "return to the baseline pose did not converge within tolerance"
            )
        return result
    except Exception as exc:  # noqa: BLE001 - the probe must always print one line
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    """Run the bounded real pose probe; print one JSON result line."""
    args = parse_args(argv)
    result = run_probe(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
