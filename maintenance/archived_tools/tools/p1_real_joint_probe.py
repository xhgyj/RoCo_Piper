#!/usr/bin/env python3
"""P1.B container-side real joint-target probe (host Isaac bridge + MoveIt).

Runs INSIDE the rootless ROS Humble container (started by
``tools/smoke_p1_real_joint.sh`` together with the ``move_group`` stack) and
proves the REAL execution chain end to end, against the live host bridge
(isaac/p05_host_bridge.py) through the container-client DDS profile:

    MoveIt plan-and-execute (move_group action)
      -> arm_controller (FollowJointTrajectory)
      -> ros2_control TopicBasedSystem -> /isaac_joint_command
      -> host Isaac articulation -> live /isaac_joint_states
      -> /joint_states via joint_state_broadcaster.

Phases, all wall-clock bounded:

1. READINESS (no motion before all pass): strictly increasing /clock samples;
   /isaac_joint_states with the exact 8-DOF names, finite positions and a
   velocity array of length 0 or 8 (>= 3 valid messages); /joint_states from
   the broadcaster carrying the SAME 8 finite values (TopicBasedSystem ->
   broadcaster proof); /controller_manager/list_controllers reporting
   joint_state_broadcaster, arm_controller and gripper_controller active; the
   /arm_controller/follow_joint_trajectory action ready; the MoveGroup action
   server (/move_action or /move_group, exact name recorded) ready; the
   planning service recorded as a diagnostic; and only then a stable baseline
   observation (fresh, finite, complete).
2. STRICT GOAL ADMISSION (caller side, BEFORE any MoveGroup send): the target
   is built from the live baseline (joint1 +0.20 rad, joint4 +0.15 rad, all
   other joints stay at baseline) and must sit strictly inside the narrowed
   execution limits with a 0.03 rad margin; the opposite sign is used when the
   baseline sits near a bound (chosen direction recorded). MoveIt's documented
   goal clamping is NEVER relied on. An inadmissible target fails before the
   action call.
3. OUTBOUND MOVEGROUP ACTION: moveit_msgs/action/MoveGroup with plan_only
   false, replan false, empty start state, allowed planning time 5 s, scaling
   0.1 and joint1..joint6 goal constraints with tolerance 0.01. The goal must
   be accepted, finish with terminal status SUCCEEDED and MoveIt error code
   SUCCESS (1) inside the wall deadline (a deadline overrun is cancelled and
   recorded), the planned trajectory must have >= 2 points whose every
   normalized arm waypoint is finite and inside the narrow limits, and the
   LIVE /isaac_joint_states feedback must converge to the target (joint1 /
   joint4 <= 0.03 rad, remaining arm joints <= 0.04 rad, gripper drift
   <= 0.01) with >= 2 post-command state messages and >= 1 observed
   /isaac_joint_command message.
4. RETURN TO BASELINE: a second plan-and-execute action targeting the exact
   captured baseline; same gates and tolerances, final arm errors <= 0.03 rad
   and gripper drift <= 0.01, recorded separately.

The probe prints exactly one machine-readable JSON result line on stdout and
exits 0 only when every phase passed.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest
from moveit_msgs.srv import GetMotionPlan
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

SCHEMA = "rocobrick-p1-real-joint-probe/v1"
PHASE = "P1.B"

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
SERVICE_PLAN_CANDIDATES = ("/plan_kinematic_path", "/move_group/plan_kinematic_path")

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
# These are hardcoded here for CALLER-SIDE goal admission: the probe never
# relies on MoveIt's out-of-bounds clamping ("Assuming maximum bounds instead").
EXECUTION_POSITION_LIMITS = {
    "joint1": (-2.6179938, 2.6179938),
    "joint2": (0.0, 3.1415926),
    "joint3": (-2.9670597, 0.0),
    "joint4": (-1.7453292, 1.7453292),
    "joint5": (-1.2217304, 1.2217304),
    "joint6": (-2.0943951, 2.0943951),
}

# Safe small deltas applied to the live baseline (rad), execution limits below.
TARGET_DELTAS = {"joint1": 0.20, "joint4": 0.15}
# Every target must be strictly inside its limits with this margin (rad).
ADMISSION_MARGIN = 0.03

# Numeric slack for floating-point noise only (rad).
LIMIT_EPSILON = 1.0e-4
# Max |/joint_states - /isaac_joint_states| while proving the broadcaster path.
EXACT_STATE_MATCH_TOLERANCE = 1.0e-4
# A state is stale when no valid message arrived within this window (s).
STATE_FRESH_SECONDS = 3.0

MIN_BASELINE_STATE_MESSAGES = 3
MIN_POST_COMMAND_STATE_MESSAGES = 2
MIN_COMMAND_MESSAGES = 1

# The settled arm must also be AT REST before another MoveGroup goal is sent:
# MoveIt validates the planned start point against the live state with a
# 0.01 tolerance, so planning from a still-settling state aborts execution
# (live evidence: "start point deviates from current robot state more than
# 0.01"). At rest means this many consecutive state samples that are inside
# tolerance AND move less than SETTLE_STABLE_DELTA_MAX between samples.
SETTLE_STABLE_SAMPLES = 8
SETTLE_STABLE_DELTA_MAX = 5.0e-4

# MoveGroup request contract.
ALLOWED_PLANNING_TIME = 5.0
NUM_PLANNING_ATTEMPTS = 10
MAX_VELOCITY_SCALING = 0.1
MAX_ACCELERATION_SCALING = 0.1
GOAL_TOLERANCE = 0.01
MIN_WAYPOINTS = 2
SUCCESS_CODE = 1  # moveit_msgs.msg.MoveItErrorCodes.SUCCESS
STATUS_SUCCEEDED = GoalStatus.STATUS_SUCCEEDED
ERROR_CODE_NAMES = {
    SUCCESS_CODE: "SUCCESS",
    -1: "PLANNING_FAILED",
    99999: "FAILURE",
}

# Final live-feedback tolerances (rad), per the P1.B contract.
FINAL_TARGET_ERROR_MAX_MAIN = 0.03  # joint1 / joint4 vs target
FINAL_TARGET_ERROR_MAX_OTHER = 0.04  # remaining arm joints vs target
GRIPPER_DRIFT_MAX = 0.01  # gripper joints vs baseline
RETURN_ARM_ERROR_MAX = 0.03  # every arm joint vs the baseline target

# The main delta joints of the outbound target (tighter tolerance).
TARGET_MAIN_JOINTS = tuple(TARGET_DELTAS)

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
        default=45.0,
        help="post-goal live-feedback convergence bound (s)",
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


def build_request(targets: dict[str, float]) -> MotionPlanRequest:
    """Build the MoveGroup MotionPlanRequest for the arm joint targets.

    The start_state stays EMPTY: move_group substitutes the current state from
    its planning-scene monitor.
    """
    request = MotionPlanRequest()
    request.group_name = "arm"
    request.allowed_planning_time = ALLOWED_PLANNING_TIME
    request.num_planning_attempts = NUM_PLANNING_ATTEMPTS
    request.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
    request.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING
    goal = Constraints()
    for name in ARM_JOINT_NAMES:
        constraint = JointConstraint()
        constraint.joint_name = name
        constraint.position = float(targets[name])
        constraint.tolerance_above = GOAL_TOLERANCE
        constraint.tolerance_below = GOAL_TOLERANCE
        constraint.weight = 1.0
        goal.joint_constraints.append(constraint)
    request.goal_constraints.append(goal)
    return request


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


def wait_future(future, deadline: float) -> bool:
    """Block (wall clock) until the future is done or the deadline passes."""
    while not future.done():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def execute_move_group_goal(
    monitor: TopicMonitor,
    action_client,
    targets: dict[str, float],
    label: str,
    baseline: dict[str, float],
    goal_timeout: float,
    settle_timeout: float,
) -> dict:
    """Send one plan-and-execute MoveGroup goal and verify it end to end."""
    summary: dict = {"label": label, "target": {n: round(float(targets[n]), 9) for n in ARM_JOINT_NAMES}}
    goal = MoveGroup.Goal()
    goal.request = build_request(targets)
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

    # Bounded convergence wait on the LIVE Isaac feedback: inside tolerance
    # AND at rest for SETTLE_STABLE_SAMPLES consecutive state samples.
    settle_deadline = time.monotonic() + settle_timeout
    final = None
    stable_count = 0
    while time.monotonic() < settle_deadline:
        states = monitor.isaac_states
        receipts = monitor.isaac_receipts
        if states and time.monotonic() - receipts[-1] <= STATE_FRESH_SECONDS:
            final = states[-1]
            within = (
                max_deviation(final, targets, TARGET_MAIN_JOINTS) <= FINAL_TARGET_ERROR_MAX_MAIN
                and max_deviation(final, targets, [n for n in ARM_JOINT_NAMES if n not in TARGET_MAIN_JOINTS])
                <= FINAL_TARGET_ERROR_MAX_OTHER
                and max_deviation(final, baseline, GRIPPER_JOINT_NAMES) <= GRIPPER_DRIFT_MAX
            )
            stable_count = stable_count + 1 if (
                within and max_step(states) <= SETTLE_STABLE_DELTA_MAX
            ) else 0
            if stable_count >= SETTLE_STABLE_SAMPLES:
                break
        else:
            stable_count = 0
        time.sleep(0.2)
    summary["settled_at_rest"] = bool(stable_count >= SETTLE_STABLE_SAMPLES)
    if final is None and monitor.isaac_states:
        final = monitor.isaac_states[-1]
    summary["final_positions"] = (
        {name: round(float(final[name]), 9) for name in EXPECTED_DOF_NAMES}
        if final is not None
        else None
    )
    if final is not None:
        summary["final_errors"] = {
            name: round(abs(float(final[name]) - float(targets.get(name, final[name]))), 9)
            for name in ARM_JOINT_NAMES
        }
        summary["gripper_drift"] = round(
            max_deviation(final, baseline, GRIPPER_JOINT_NAMES), 9
        )
    post_states = monitor.isaac_states[send_index:]
    summary["post_command_state_messages"] = len(post_states)
    summary["command_messages"] = monitor.command_count

    summary["action_and_plan_passed"] = bool(action_ok)
    summary["feedback_passed"] = bool(
        action_ok
        and final is not None
        and summary["settled_at_rest"]
        and len(post_states) >= MIN_POST_COMMAND_STATE_MESSAGES
        and monitor.command_count >= MIN_COMMAND_MESSAGES
        and max_deviation(final, targets, TARGET_MAIN_JOINTS) <= FINAL_TARGET_ERROR_MAX_MAIN
        and max_deviation(
            final, targets, [n for n in ARM_JOINT_NAMES if n not in TARGET_MAIN_JOINTS]
        )
        <= FINAL_TARGET_ERROR_MAX_OTHER
        and max_deviation(final, baseline, GRIPPER_JOINT_NAMES) <= GRIPPER_DRIFT_MAX
    )
    summary["passed"] = summary["feedback_passed"]
    if not summary["passed"]:
        summary["error"] = summary.get(
            "error",
            f"{label}: live feedback did not converge to the target within tolerance",
        )
    return summary


def run_probe(args: argparse.Namespace) -> dict:
    """Run the bounded probe; return the result dict (never raises)."""
    result: dict = {
        "schema": SCHEMA,
        "phase": PHASE,
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
            "plan_candidates": list(SERVICE_PLAN_CANDIDATES),
        },
        "expected_dof_names": list(EXPECTED_DOF_NAMES),
        "execution_position_limits": {
            name: list(limits) for name, limits in EXECUTION_POSITION_LIMITS.items()
        },
        "target_deltas": dict(TARGET_DELTAS),
        "admission_margin": ADMISSION_MARGIN,
        "goal_tolerance": GOAL_TOLERANCE,
        "allowed_planning_time_s": ALLOWED_PLANNING_TIME,
        "scaling_factor": MAX_VELOCITY_SCALING,
        "final_error_max_main": FINAL_TARGET_ERROR_MAX_MAIN,
        "final_error_max_other": FINAL_TARGET_ERROR_MAX_OTHER,
        "gripper_drift_max": GRIPPER_DRIFT_MAX,
        "return_arm_error_max": RETURN_ARM_ERROR_MAX,
        "settle_stable_samples": SETTLE_STABLE_SAMPLES,
        "settle_stable_delta_max": SETTLE_STABLE_DELTA_MAX,
        "exact_state_match_tolerance": EXACT_STATE_MATCH_TOLERANCE,
        "min_baseline_state_messages": MIN_BASELINE_STATE_MESSAGES,
        "readiness": {},
        "admission": {"passed": False},
        "outbound": {"passed": False},
        "return": {"passed": False},
        "passed": False,
    }

    rclpy.init()
    node = rclpy.create_node("p1_real_joint_probe")
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
        plan_service_clients = {
            name: node.create_client(GetMotionPlan, name) for name in SERVICE_PLAN_CANDIDATES
        }

        # ---------------------------------------------------- readiness phase
        controllers: dict | None = None
        controllers_future = None
        arm_action_ready = False
        move_group_name = None
        plan_service_name = None
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
            if plan_service_name is None:
                for name in SERVICE_PLAN_CANDIDATES:
                    if plan_service_clients[name].service_is_ready():
                        plan_service_name = name
                        break
            states = monitor.isaac_states
            ready = (
                controllers is not None
                and all(controllers.get(name) == "active" for name in REQUIRED_CONTROLLERS)
                and arm_action_ready
                and move_group_name is not None
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
            "plan_service_name": plan_service_name,
        }
        result["readiness"] = readiness
        baseline_ready = bool(
            readiness["clock_increasing"]
            and len(monitor.isaac_states) >= MIN_BASELINE_STATE_MESSAGES
            and readiness["broadcast_joint_state_messages"] > 0
            and readiness["controllers_active"]
            and readiness["arm_follow_action_ready"]
            and readiness["move_group_action_name"]
        )
        if not baseline_ready:
            result["error"] = "readiness phase failed: move_group stack or live bridge not ready"
            return result

        # --------------------------------------------------- baseline capture
        states = monitor.isaac_states
        receipts = monitor.isaac_receipts
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
        # Command traffic is recorded on every post-baseline exit path.
        result["command_stats"] = {
            "count": monitor.command_count,
            "invalid": monitor.command_invalid,
            "unknown_names": list(monitor.command_unknown_names),
            "min_position": monitor.command_min,
            "max_position": monitor.command_max,
        }
        if not (fresh and broadcast_match):
            result["error"] = (
                "baseline state stale/nonfinite/incomplete before the goal send"
            )
            return result

        # ----------------------------------------------- strict goal admission
        targets: dict[str, float] = {}
        chosen: dict[str, float] = {}
        for name, delta in TARGET_DELTAS.items():
            lower, upper = EXECUTION_POSITION_LIMITS[name]
            base = baseline[name]
            candidate = None
            for sign in (1.0, -1.0):
                value = base + sign * delta
                if lower + ADMISSION_MARGIN < value < upper - ADMISSION_MARGIN:
                    candidate = value
                    chosen[name] = sign * delta
                    break
            if candidate is None:
                result["admission"] = {
                    "passed": False,
                    "error": (
                        f"{name}: no sign of delta {delta} keeps the target "
                        f"inside [{lower}, {upper}] with margin {ADMISSION_MARGIN}"
                    ),
                }
                result["error"] = result["admission"]["error"]
                return result
            targets[name] = candidate
        for name in ARM_JOINT_NAMES:
            if name not in targets:
                # Unchanged joints carry the exact live baseline; they must sit
                # inside the limits (the margin binds the moved targets only).
                lower, upper = EXECUTION_POSITION_LIMITS[name]
                value = baseline[name]
                if not lower <= value <= upper:
                    result["admission"] = {
                        "passed": False,
                        "error": (
                            f"{name}: baseline {value} outside its execution "
                            f"limits [{lower}, {upper}]"
                        ),
                    }
                    return result
                targets[name] = value
        result["admission"] = {
            "passed": True,
            "target": {name: round(float(targets[name]), 9) for name in ARM_JOINT_NAMES},
            "chosen_deltas": {name: round(float(value), 9) for name, value in chosen.items()},
            "margin": ADMISSION_MARGIN,
        }

        # ------------------------------------------------- outbound execution
        outbound = execute_move_group_goal(
            monitor,
            move_group_clients[move_group_name],
            targets,
            "outbound",
            baseline,
            args.goal_timeout,
            args.settle_timeout,
        )
        result["outbound"] = outbound
        if not outbound.get("passed"):
            return result

        # ------------------------------------------------ return to baseline
        return_summary = execute_move_group_goal(
            monitor,
            move_group_clients[move_group_name],
            {name: baseline[name] for name in ARM_JOINT_NAMES},
            "return",
            baseline,
            args.goal_timeout,
            args.settle_timeout,
        )
        # The return verdict uses the strict uniform arm tolerance.
        return_errors = return_summary.get("final_errors") or {}
        return_arm_ok = bool(
            return_errors
            and all(
                error <= RETURN_ARM_ERROR_MAX for error in return_errors.values()
            )
            and (return_summary.get("gripper_drift", float("inf")) <= GRIPPER_DRIFT_MAX)
        )
        return_summary["passed"] = bool(
            return_summary.get("action_and_plan_passed") and return_arm_ok
        )
        result["return"] = return_summary
        result["command_stats"] = {
            "count": monitor.command_count,
            "invalid": monitor.command_invalid,
            "unknown_names": list(monitor.command_unknown_names),
            "min_position": monitor.command_min,
            "max_position": monitor.command_max,
        }
        result["passed"] = bool(outbound["passed"] and return_summary["passed"])
        if not result["passed"]:
            result["error"] = result.get("error") or return_summary.get(
                "error", "return to baseline did not converge within tolerance"
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
    """Run the bounded real joint probe; print one JSON result line."""
    args = parse_args(argv)
    result = run_probe(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
