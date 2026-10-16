#!/usr/bin/env python3
"""Container-side listener counting messages per topic for the type matrix.

Runs in the ROS Humble container. Subscribes the four matrix topics with a
reliable QoS, counts messages for --seconds, prints one JSON line, exits.
Used by both the container-to-container control test and the host matrix test.
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

TOPICS = {
    "/tm/clock": Clock,
    "/tm/string": String,
    "/tm/js_empty": JointState,
    "/tm/js_full": JointState,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    reliable = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    rclpy.init()
    node = rclpy.create_node("dds_type_matrix_listener")
    counts = {topic: 0 for topic in TOPICS}
    last_payload = {}
    for topic, msg_type in TOPICS.items():
        def make_cb(topic=topic):
            def cb(msg):
                counts[topic] += 1
                if topic == "/tm/string":
                    last_payload[topic] = msg.data
                elif topic == "/tm/js_full" and counts[topic] == 1:
                    last_payload[topic] = list(msg.name)
            return cb
        node.create_subscription(msg_type, topic, make_cb(topic), reliable)

    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    print(
        json.dumps(
            {
                "schema": "rocobrick-dds-type-matrix-listener/v1",
                "seconds": args.seconds,
                "counts": counts,
                "first_js_full_names": last_payload.get("/tm/js_full"),
                "first_string": last_payload.get("/tm/string"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
