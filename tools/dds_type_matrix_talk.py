#!/usr/bin/env python3
"""Container-side control talker for the DDS type matrix test.

Publishes Clock, String, and JointState (empty and full) on the /tm/* topics
so container-to-container delivery can be validated without Isaac Sim.
"""
from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import String

DOF_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "gripper_joint1", "gripper_joint2",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    reliable = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    rclpy.init()
    node = rclpy.create_node("dds_type_matrix_talker")
    pub_clock = node.create_publisher(Clock, "/tm/clock", reliable)
    pub_string = node.create_publisher(String, "/tm/string", reliable)
    pub_empty = node.create_publisher(JointState, "/tm/js_empty", reliable)
    pub_full = node.create_publisher(JointState, "/tm/js_full", reliable)

    started = time.monotonic()
    period = 1.0 / args.hz
    published = 0
    while time.monotonic() - started < args.seconds:
        clock = Clock()
        clock.clock.sec = int(time.time())
        clock.clock.nanosec = int((time.time() % 1) * 1e9)
        pub_clock.publish(clock)

        text = String()
        text.data = f"tm-{published}"
        pub_string.publish(text)

        pub_empty.publish(JointState())

        full = JointState()
        full.name = list(DOF_NAMES)
        full.position = [0.0] * 8
        full.velocity = [0.0] * 8
        pub_full.publish(full)
        published += 1
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(period)
    node.destroy_node()
    rclpy.shutdown()
    print(
        json.dumps(
            {"schema": "rocobrick-dds-type-matrix-talker/v1", "published_each": published},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
