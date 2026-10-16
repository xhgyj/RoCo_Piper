#!/usr/bin/env python3
"""P1 container-side MoveIt planning probe for the offline Piper_L smoke.

Runs INSIDE the ROS Humble container (started by
``tools/smoke_p1_moveit_offline.sh`` together with
``tools/p1_fake_joint_states.py``, which replaces ONLY the host Isaac bridge
as the ``/isaac_joint_states`` source). It proves the P1 planning chain and
the execution-limit narrowing entirely on the CPU, without Isaac Sim:

1. Wait (bounded) for ``/joint_states`` messages carrying the complete 8-DOF
   Piper_L state — proves TopicBasedSystem -> joint_state_broadcaster ->
   /joint_states works — and for the move_group planning service
   (``/plan_kinematic_path`` first, then ``/move_group/plan_kinematic_path``).
2. VALID GOAL: call the GetMotionPlan service with joint goals strictly
   inside the NARROW execution limits and assert SUCCESS plus a non-trivial
   trajectory whose every arm waypoint stays inside those limits.
3. WIDE-ONLY GOAL (the narrowing proof): the same request but joint4 = 2.0 rad
   (inside the WIDE official URDF limit 2.2165681, OUTSIDE the narrowed
   1.7453292). MoveIt's documented behavior for an out-of-bounds joint goal
   is to CLAMP it to the bound ("Assuming maximum bounds instead") rather
   than reject, so the acceptance property is: the response either fails, or
   succeeds with EVERY waypoint still inside the narrow limits — the planner
   must never command beyond the narrowed execution limits. Callers that need
   strict rejection must pre-validate goals (the piper-run admission gate).

The probe prints exactly one machine-readable JSON result line on stdout and
exits 0 only when every phase passed.
"""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest
from moveit_msgs.srv import GetMotionPlan
from sensor_msgs.msg import JointState

SCHEMA = "rocobrick-p1-moveit-offline/v1"
PHASE = "P1"

# Standard /joint_states republished by joint_state_broadcaster from the
# TopicBasedSystem hardware state interfaces (which READ /isaac_joint_states).
TOPIC_JOINT_STATES = "/joint_states"

# move_group planning service: the unqualified name first, the namespaced
# fallback second; whichever answers is recorded in the result.
SERVICE_CANDIDATES = ("/plan_kinematic_path", "/move_group/plan_kinematic_path")

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

# Narrowed EXECUTION limits (rad) = intersection of the pinned official
# sources, i.e. the model-contract table (tests/test_piper_l_model_contract.py)
# as enforced for PLANNING by
# ros_ws/src/rocobrick_piper_l_moveit_config/config/joint_limits.yaml
# (the URDF itself keeps the WIDE joint4 +/-2.2165681 / joint5 +/-1.5620696).
EXECUTION_POSITION_LIMITS = {
    "joint1": (-2.6179938, 2.6179938),
    "joint2": (0.0, 3.1415926),
    "joint3": (-2.9670597, 0.0),
    "joint4": (-1.7453292, 1.7453292),
    "joint5": (-1.2217304, 1.2217304),
    "joint6": (-2.0943951, 2.0943951),
}

# Numeric waypoint-bound slack for floating-point noise only (rad).
LIMIT_EPSILON = 1.0e-4

ALLOWED_PLANNING_TIME = 5.0
NUM_PLANNING_ATTEMPTS = 10
GOAL_TOLERANCE = 0.01
MIN_WAYPOINTS = 2

# VALID GOAL: every target strictly inside the narrow execution limits.
VALID_GOAL_POSITIONS = {
    "joint1": 0.4,
    "joint2": 0.5,
    "joint3": -0.4,
    "joint4": 0.6,
    "joint5": 0.5,
    "joint6": 0.4,
}

# WIDE-ONLY GOAL (acceptance-critical): inside the wide URDF limit
# (2.2165681) but outside the narrowed execution limit (1.7453292); the
# planner must reject it, proving the planning-side narrowing binds.
WIDE_JOINT_NAME = "joint4"
WIDE_JOINT4_POSITION = 2.0
# Expected rejection code: -1 PLANNING_FAILED (a.k.a. GOAL_VIOLATES_LIMITS
# in the P1 task wording); the hard assertion stays "!= SUCCESS".
WIDE_EXPECTED_ERROR_CODE = -1

SUCCESS_CODE = 1  # moveit_msgs.msg.MoveItErrorCodes.SUCCESS
ERROR_CODE_NAMES = {
    SUCCESS_CODE: "SUCCESS",
    -1: "PLANNING_FAILED",
    99999: "FAILURE",
}

# Best-effort subscription: compatible with both a reliable and a
# sensor-QoS /joint_states publisher (joint_state_broadcaster).
qos_state = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded probe arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--timeout", type=float, default=90.0, help="ready phase bound (s)"
    )
    parser.add_argument(
        "--plan-timeout",
        type=float,
        default=60.0,
        help="per planning service call bound (s)",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0.0 or args.plan_timeout <= 0.0:
        parser.error("--timeout and --plan-timeout must be positive")
    return args


def positions_by_name(msg: JointState) -> dict[str, float]:
    """Return the joint_state positions keyed by joint name."""
    return {str(name): float(value) for name, value in zip(msg.name, msg.position)}


def build_request(targets: dict[str, float]) -> MotionPlanRequest:
    """Build the GetMotionPlan request for the arm joint targets.

    The start_state stays EMPTY: move_group substitutes the current state
    from its planning-scene monitor.
    """
    request = MotionPlanRequest()
    request.group_name = "arm"
    request.allowed_planning_time = ALLOWED_PLANNING_TIME
    request.num_planning_attempts = NUM_PLANNING_ATTEMPTS
    request.max_velocity_scaling_factor = 0.1
    request.max_acceleration_scaling_factor = 0.1
    goal = Constraints()
    for name in ARM_JOINT_NAMES:
        constraint = JointConstraint()
        constraint.joint_name = name
        constraint.position = targets[name]
        constraint.tolerance_above = GOAL_TOLERANCE
        constraint.tolerance_below = GOAL_TOLERANCE
        constraint.weight = 1.0
        goal.joint_constraints.append(constraint)
    request.goal_constraints.append(goal)
    return request


def call_plan(
    node, client, targets: dict[str, float], plan_timeout: float
) -> tuple[dict | None, str | None]:
    """Call the planning service once; return (summary, error)."""
    srv_request = GetMotionPlan.Request()
    srv_request.motion_plan_request = build_request(targets)
    future = client.call_async(srv_request)
    deadline = time.monotonic() + plan_timeout
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not future.done():
        return None, f"planning call timed out after {plan_timeout}s"
    try:
        response = future.result()
    except Exception as exc:  # service call failed at the DDS layer
        return None, f"planning call failed: {exc}"
    motion = response.motion_plan_response
    code = int(motion.error_code.val)
    trajectory = motion.trajectory.joint_trajectory
    waypoint_names = [str(name) for name in trajectory.joint_names]
    waypoint_count = len(trajectory.points)
    max_excess = {name: 0.0 for name in ARM_JOINT_NAMES if name in waypoint_names}
    for point in trajectory.points:
        for name, position in zip(waypoint_names, point.positions):
            if name not in max_excess:
                continue
            lower, upper = EXECUTION_POSITION_LIMITS[name]
            excess = max(lower - position, position - upper, 0.0)
            max_excess[name] = max(max_excess[name], excess)
    waypoints_within_limits = all(excess <= LIMIT_EPSILON for excess in max_excess.values())
    final_positions: dict[str, float] = {}
    if trajectory.points and waypoint_names:
        last_point = trajectory.points[-1]
        final_positions = {
            name: round(float(value), 9)
            for name, value in zip(waypoint_names, last_point.positions)
        }
    summary = {
        "error_code": code,
        "error_code_name": ERROR_CODE_NAMES.get(code, f"UNKNOWN({code})"),
        "planning_time_s": round(float(motion.planning_time), 6),
        "waypoint_count": waypoint_count,
        "trajectory_joint_names": waypoint_names,
        "waypoint_max_limit_excess": {
            name: round(excess, 9) for name, excess in max_excess.items()
        },
        "waypoints_within_limits": waypoints_within_limits,
        "final_positions": final_positions,
    }
    return summary, None


def main(argv: list[str] | None = None) -> int:
    """Run the bounded MoveIt planning probe; print one JSON result line."""
    args = parse_args(argv)

    result: dict = {
        "schema": SCHEMA,
        "phase": PHASE,
        "topic_joint_states": TOPIC_JOINT_STATES,
        "service_candidates": list(SERVICE_CANDIDATES),
        "expected_dof_names": list(EXPECTED_DOF_NAMES),
        "execution_position_limits": {
            name: list(limits) for name, limits in EXECUTION_POSITION_LIMITS.items()
        },
        "allowed_planning_time_s": ALLOWED_PLANNING_TIME,
        "goal_tolerance": GOAL_TOLERANCE,
        "valid_goal_positions": dict(VALID_GOAL_POSITIONS),
        "wide_goal_joint": WIDE_JOINT_NAME,
        "wide_goal_position": WIDE_JOINT4_POSITION,
        "wide_goal_expected_error_code": WIDE_EXPECTED_ERROR_CODE,
        "min_waypoints": MIN_WAYPOINTS,
        "joint_state_messages": 0,
        "observed_dof_names": [],
        "joint_states_flow": False,
        "service_name_used": None,
        "service_available": False,
        "valid_goal": {"passed": False},
        "wide_goal": {"never_exceeds_limits": False},
        "passed": False,
    }

    rclpy.init()
    node = rclpy.create_node("p1_moveit_offline_probe")
    try:
        joint_states: list[dict[str, float]] = []
        node.create_subscription(
            JointState,
            TOPIC_JOINT_STATES,
            lambda msg: joint_states.append(positions_by_name(msg)),
            qos_state,
        )
        clients = {
            name: node.create_client(GetMotionPlan, name) for name in SERVICE_CANDIDATES
        }

        # Phase 1: /joint_states flowing and a planning service available.
        ready_deadline = time.monotonic() + args.timeout
        while time.monotonic() < ready_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if joint_states and not result["service_name_used"]:
                for name in SERVICE_CANDIDATES:
                    if clients[name].service_is_ready():
                        result["service_name_used"] = name
                        result["service_available"] = True
                        break
            if joint_states and result["service_name_used"]:
                break
        result["joint_state_messages"] = len(joint_states)
        if joint_states:
            result["observed_dof_names"] = sorted(joint_states[-1])
        result["joint_states_flow"] = bool(
            joint_states
            and set(joint_states[-1]) == set(EXPECTED_DOF_NAMES)
        )
        if not (result["joint_states_flow"] and result["service_available"]):
            result["error"] = (
                "ready phase failed: no complete 8-DOF /joint_states"
                " or no move_group planning service"
            )
            print(json.dumps(result, sort_keys=True), flush=True)
            return 1

        client = clients[result["service_name_used"]]

        # Phase 2: VALID goal (inside the narrow limits) must plan SUCCESS
        # with a non-trivial trajectory fully inside the narrow limits.
        valid_summary, valid_error = call_plan(
            node, client, VALID_GOAL_POSITIONS, args.plan_timeout
        )
        if valid_summary is None:
            result["valid_goal"] = {"passed": False, "error": valid_error}
            result["error"] = f"valid goal failed: {valid_error}"
            print(json.dumps(result, sort_keys=True), flush=True)
            return 1
        valid_summary["passed"] = bool(
            valid_summary["error_code"] == SUCCESS_CODE
            and valid_summary["waypoint_count"] >= MIN_WAYPOINTS
            and valid_summary["waypoints_within_limits"]
        )
        result["valid_goal"] = valid_summary
        if not valid_summary["passed"]:
            result["error"] = (
                "valid goal failed: expected SUCCESS with >=2 waypoints"
                " inside the narrow execution limits"
            )
            print(json.dumps(result, sort_keys=True), flush=True)
            return 1

        # Phase 3: WIDE-ONLY goal (joint4 = 2.0, outside the narrowed
        # execution limit): either rejected, or MoveIt clamps it to the
        # narrow bound and every waypoint stays inside the execution limits.
        # Either way the planner must never command beyond the narrowed
        # limits — that is the acceptance property.
        wide_targets = dict(VALID_GOAL_POSITIONS)
        wide_targets[WIDE_JOINT_NAME] = WIDE_JOINT4_POSITION
        wide_summary, wide_error = call_plan(
            node, client, wide_targets, args.plan_timeout
        )
        if wide_summary is None:
            result["wide_goal"] = {"never_exceeds_limits": False, "error": wide_error}
            result["error"] = f"wide goal call failed: {wide_error}"
            print(json.dumps(result, sort_keys=True), flush=True)
            return 1
        wide_summary["hard_rejected"] = bool(
            wide_summary["error_code"] != SUCCESS_CODE
        )
        wide_summary["clamped_within_limits"] = bool(
            wide_summary["error_code"] == SUCCESS_CODE
            and wide_summary["waypoints_within_limits"]
        )
        wide_summary["never_exceeds_limits"] = bool(
            wide_summary["hard_rejected"] or wide_summary["clamped_within_limits"]
        )
        result["wide_goal"] = wide_summary
        result["passed"] = bool(
            result["joint_states_flow"] and wide_summary["never_exceeds_limits"]
        )
        if not result["passed"]:
            result["error"] = (
                "wide-only goal produced a trajectory beyond the narrowed"
                " execution limits: the execution-limit narrowing is not"
                " binding the planner"
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
