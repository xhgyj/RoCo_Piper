#!/usr/bin/env python3
"""P1 offline fake joint-states feeder for the container MoveIt smoke.

In production the ``/isaac_joint_states`` topic is fed by the host Isaac
bridge (isaac/p05_host_bridge.py publishing the simulated Piper_L state).
This feeder replaces ONLY that bridge for offline validation: it publishes
sensor_msgs/JointState on ``/isaac_joint_states`` at 30 Hz with the exact
8-DOF Piper_L names (joint1..joint6, gripper_joint1, gripper_joint2) held at
the package initial positions (ros_ws/src/rocobrick_piper_l_moveit_config/
config/initial_positions.yaml, all zeros; positions only, velocity zeros).

The lifetime is bounded by --duration seconds, after which the node shuts
down cleanly. It runs INSIDE the ROS container (started by
tools/smoke_p1_moveit_offline.sh); it never touches Isaac Sim or the GPU.
"""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from sensor_msgs.msg import JointState

# Official Isaac<->MoveIt topic contract (piper_l.ros2_control.xacro,
# TopicBasedSystem): plural states topic (the singular /isaac_joint_command
# is the WRITE direction and is never published by this feeder).
TOPIC_JOINT_STATES = "/isaac_joint_states"

PUBLISH_RATE_HZ = 30.0

# Same 8-DOF names/order as the model contract (tests/test_piper_l_model_contract.py)
# and the ros2_control macro (piper_l.ros2_control.xacro).
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

# Package initial positions (config/initial_positions.yaml): all zeros.
INITIAL_POSITIONS = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "gripper_joint1": 0.0,
    "gripper_joint2": 0.0,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded feeder arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="publishing lifetime bound (s) before the clean shutdown",
    )
    parser.add_argument("--rate", type=float, default=PUBLISH_RATE_HZ, help="publish rate (Hz)")
    args = parser.parse_args(argv)
    if args.duration <= 0.0 or args.rate <= 0.0:
        parser.error("--duration and --rate must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    """Publish the initial 8-DOF state at 30 Hz for --duration seconds."""
    args = parse_args(argv)
    published = 0

    rclpy.init()
    node = rclpy.create_node("p1_fake_joint_states")
    publisher = node.create_publisher(JointState, TOPIC_JOINT_STATES, 10)
    try:
        message = JointState()
        message.name = list(EXPECTED_DOF_NAMES)
        message.position = [INITIAL_POSITIONS[name] for name in EXPECTED_DOF_NAMES]
        message.velocity = [0.0] * len(EXPECTED_DOF_NAMES)

        period = 1.0 / args.rate
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline and rclpy.ok():
            message.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(message)
            published += 1
            rclpy.spin_once(node, timeout_sec=period)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(
        json.dumps(
            {
                "schema": "rocobrick-p1-fake-joint-states/v1",
                "topic": TOPIC_JOINT_STATES,
                "rate_hz": args.rate,
                "duration_s": args.duration,
                "published": published,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
