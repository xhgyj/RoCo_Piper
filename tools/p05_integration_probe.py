#!/usr/bin/env python3
"""P0.5 ROS container-side integration probe for the Isaac host bridge.

Runs INSIDE the rootless ROS Humble container (started by
``tools/smoke_p05_host_bridge.sh`` with the container Fast DDS profile and the
published 17651 TCP port). It is the external validation half of P0.5: the
host process only runs the Isaac bridge participant, so every ROS check
happens here.

Phases, all wall-clock bounded:

1. Subscribe ``/clock`` and ``/isaac_joint_states`` and wait until a joint_states
   message carries the complete 8-DOF Piper L state names and the clock is
   strictly increasing.
2. Publish a safe named ``sensor_msgs/JointState`` command on
   ``/isaac_joint_command``: a small positive delta on a subset of the six arm
   joints, clamped inside the official URDF limits.
3. Verify subsequent joint_states: commanded joints moved at least
   ``COMMAND_MOVE_MIN`` and every uncommanded joint stayed within
   ``IDLE_DELTA_MAX``.

The probe prints exactly one machine-readable JSON result line on stdout and
exits 0 only when every phase passed.
"""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

SCHEMA = "rocobrick-p05-integration-probe/v1"
PHASE = "P0.5"

TOPIC_CLOCK = "/clock"
# Official Isaac<->MoveIt topic contract (piper_l.ros2_control.xacro,
# TopicBasedSystem): plural states, singular command.
TOPIC_JOINT_STATES = "/isaac_joint_states"
TOPIC_JOINT_COMMANDS = "/isaac_joint_command"

QOS_DEPTH = 10
qos_reliable = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=QOS_DEPTH,
    reliability=ReliabilityPolicy.RELIABLE,
)

# same order as the model contract in tests/test_piper_l_model_contract.py.
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

# Command a small delta on a subset of the six arm joints only (never the
# gripper): the subscribed names drive the Articulation Controller, so every
# joint not listed here must keep idling within IDLE_DELTA_MAX.
COMMAND_JOINT_NAMES = ("joint1", "joint4")

# Execution limits (rad): the intersection of the pinned official sources,
# equal to the agx_arm_sim URDF/USD values (joint4/5 are narrower there than
# in the agx_arm_urdf submodule; see the model contract limit_conflict).
JOINT_LIMITS = {
    "joint1": (-2.6179938, 2.6179938),
    "joint2": (0.0, 3.1415926),
    "joint3": (-2.9670597, 0.0),
    "joint4": (-1.7453292, 1.7453292),
    "joint5": (-1.2217304, 1.2217304),
    "joint6": (-2.0943951, 2.0943951),
}

COMMAND_DELTA = 0.2
COMMAND_LIMIT_MARGIN = 0.05
COMMAND_MOVE_MIN = 0.05
IDLE_DELTA_MAX = 0.35
COMMAND_PERIOD_S = 0.1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded probe arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=float, default=60.0, help="ready phase bound (s)")
    parser.add_argument(
        "--drive-timeout", type=float, default=40.0, help="command/verify phase bound (s)"
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0.0 or args.drive_timeout <= 0.0:
        parser.error("--timeout and --drive-timeout must be positive")
    return args


def clock_value(msg: Clock) -> int:
    """Return the clock message as integer nanoseconds."""
    return int(msg.clock.sec) * 1_000_000_000 + int(msg.clock.nanosec)


def positions_by_name(msg: JointState) -> dict[str, float]:
    """Return the joint_state positions keyed by joint name."""
    return {str(name): float(value) for name, value in zip(msg.name, msg.position)}


def safe_targets(baseline: dict[str, float]) -> dict[str, float]:
    """Compute limit-clamped command targets for the commanded joints."""
    targets = {}
    for name in COMMAND_JOINT_NAMES:
        lower, upper = JOINT_LIMITS[name]
        target = baseline[name] + COMMAND_DELTA
        targets[name] = min(max(target, lower + COMMAND_LIMIT_MARGIN), upper - COMMAND_LIMIT_MARGIN)
    return targets


def main(argv: list[str] | None = None) -> int:
    """Run the bounded integration probe; print one JSON result line."""
    args = parse_args(argv)

    result: dict = {
        "schema": SCHEMA,
        "phase": PHASE,
        "topics": {
            "clock": TOPIC_CLOCK,
            "joint_states": TOPIC_JOINT_STATES,
            "joint_commands": TOPIC_JOINT_COMMANDS,
        },
        "expected_dof_names": list(EXPECTED_DOF_NAMES),
        "command_joint_names": list(COMMAND_JOINT_NAMES),
        "command_delta": COMMAND_DELTA,
        "command_move_min": COMMAND_MOVE_MIN,
        "idle_delta_max": IDLE_DELTA_MAX,
        "dof_names_ok": False,
        "clock_increasing": False,
        "command_published": False,
        "commanded_moved": False,
        "idle_within_limit": False,
        "passed": False,
    }

    rclpy.init()
    node = rclpy.create_node("p05_integration_probe")
    try:
        clock_values: list[int] = []
        joint_states: list[dict[str, float]] = []
        diag_messages = [0]
        node.create_subscription(Clock, TOPIC_CLOCK, lambda msg: clock_values.append(clock_value(msg)), qos_reliable)
        node.create_subscription(
            JointState,
            TOPIC_JOINT_STATES,
            lambda msg: joint_states.append(positions_by_name(msg)),
            qos_profile=qos_reliable,
        )
        node.create_subscription(
            JointState,
            "/joint_states_diag",
            lambda _msg: diag_messages.__setitem__(0, diag_messages[0] + 1),
            qos_profile=qos_reliable,
        )

        # Phase 1: complete 8-DOF state names and a strictly increasing clock.
        ready_deadline = time.monotonic() + args.timeout
        while time.monotonic() < ready_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            increasing = any(
                later > earlier for earlier, later in zip(clock_values, clock_values[1:])
            )
            if joint_states and set(joint_states[-1]) == set(EXPECTED_DOF_NAMES) and increasing:
                result["dof_names_ok"] = True
                result["clock_increasing"] = True
                break
        result["publisher_info"] = {
            TOPIC_CLOCK: [
                {"node_name": info.node_name, "topic_type": info.topic_type}
                for info in node.get_publishers_info_by_topic(TOPIC_CLOCK)
            ],
            TOPIC_JOINT_STATES: [
                {"node_name": info.node_name, "topic_type": info.topic_type}
                for info in node.get_publishers_info_by_topic(TOPIC_JOINT_STATES)
            ],
        }
        result["clock_messages"] = len(clock_values)
        result["joint_state_diag_messages"] = diag_messages[0]
        result["joint_state_messages"] = len(joint_states)
        if clock_values:
            result["clock_first_ns"] = clock_values[0]
            result["clock_last_ns"] = clock_values[-1]
        if joint_states:
            result["observed_dof_names"] = sorted(joint_states[-1])
        if not (result["dof_names_ok"] and result["clock_increasing"]):
            result["error"] = "ready phase failed: no complete 8-DOF joint_states or non-increasing clock"
            print(json.dumps(result, sort_keys=True), flush=True)
            return 1

        # Phase 2: publish the safe named command with a small clamped delta.
        baseline = dict(joint_states[-1])
        targets = safe_targets(baseline)
        command = JointState()
        command.name = list(COMMAND_JOINT_NAMES)
        command.position = [targets[name] for name in COMMAND_JOINT_NAMES]
        publisher = node.create_publisher(JointState, TOPIC_JOINT_COMMANDS, 10)
        result["command_positions"] = targets
        result["baseline"] = {name: baseline[name] for name in EXPECTED_DOF_NAMES}

        post_command_states: list[dict[str, float]] = []
        publish_deadline = time.monotonic() + args.drive_timeout
        next_publish = 0.0
        while time.monotonic() < publish_deadline:
            now = time.monotonic()
            if now >= next_publish:
                publisher.publish(command)
                result["command_published"] = True
                next_publish = now + COMMAND_PERIOD_S
            before = len(joint_states)
            rclpy.spin_once(node, timeout_sec=COMMAND_PERIOD_S / 2.0)
            if len(joint_states) > before:
                post_command_states.append(joint_states[-1])

        # Phase 3: commanded joints moved, uncommanded joints stayed bounded.
        if not post_command_states:
            result["error"] = "verify phase failed: no joint_states after the command"
            print(json.dumps(result, sort_keys=True), flush=True)
            return 1
        final = dict(post_command_states[-1])
        deltas = {
            name: round(final[name] - baseline[name], 6)
            for name in EXPECTED_DOF_NAMES
            if name in final and name in baseline
        }
        result["final_positions"] = {name: final.get(name) for name in EXPECTED_DOF_NAMES}
        result["deltas"] = deltas
        result["commanded_moved"] = all(
            name in deltas and abs(deltas[name]) >= COMMAND_MOVE_MIN
            for name in COMMAND_JOINT_NAMES
        )
        result["idle_within_limit"] = all(
            abs(deltas[name]) <= IDLE_DELTA_MAX
            for name in deltas
            if name not in COMMAND_JOINT_NAMES
        )
        result["passed"] = bool(
            result["command_published"] and result["commanded_moved"] and result["idle_within_limit"]
        )
        if not result["passed"]:
            result["error"] = "verify phase failed: commanded joints did not move or idle joints drifted"
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
