#!/usr/bin/env python3
"""Host-side DDS type matrix publisher using the Isaac bundled rclpy.

Boots headless Isaac Sim, enables isaacsim.ros2.bridge (which loads the
bundled Humble rclpy), then publishes Clock, String, JointState(empty) and
JointState(full, 8 DOF) on /tm/* through the solved TCPv4 host profile.
Prints a ready marker, publishes for --seconds, prints one JSON summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

BRIDGE_EXTENSION = "isaacsim.ros2.bridge"
BRIDGE_OGN_TYPES = frozenset(
    {
        "isaacsim.ros2.bridge.ROS2Context",
        "isaacsim.ros2.bridge.ROS2PublishClock",
        "isaacsim.ros2.bridge.ROS2PublishJointState",
        "isaacsim.ros2.bridge.ROS2SubscribeJointState",
    }
)
DOF_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "gripper_joint1", "gripper_joint2",
]
READY_MARKER = "dds_type_matrix_host_ready"


def _bridge_ready(og) -> bool:
    try:
        registered = og.get_registered_nodes()
        return BRIDGE_OGN_TYPES <= {str(item) for item in registered}
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.graph.core as og
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate(BRIDGE_EXTENSION, True)
        gate_deadline = time.monotonic() + 30.0
        gate = "registry_confirmed"
        while time.monotonic() < gate_deadline:
            app.update()
            if _bridge_ready(og):
                break
        else:
            gate = "timeout"

        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        rclpy.init()
        node = rclpy.create_node("dds_type_matrix_host")
        pub_clock = node.create_publisher(Clock, "/tm/clock", reliable)
        pub_string = node.create_publisher(String, "/tm/string", reliable)
        pub_empty = node.create_publisher(JointState, "/tm/js_empty", reliable)
        pub_full = node.create_publisher(JointState, "/tm/js_full", reliable)

        print(
            json.dumps({"stage": "ready", "marker": READY_MARKER, "gate": gate}),
            flush=True,
        )
        started = time.monotonic()
        period = 1.0 / args.hz
        counts = {t: 0 for t in ("/tm/clock", "/tm/string", "/tm/js_empty", "/tm/js_full")}
        seq = 0
        while time.monotonic() - started < args.seconds:
            clock = Clock()
            now = time.time()
            clock.clock.sec = int(now)
            clock.clock.nanosec = int((now % 1) * 1e9)
            pub_clock.publish(clock)

            text = String()
            text.data = f"tm-{seq}"
            pub_string.publish(text)

            pub_empty.publish(JointState())

            full = JointState()
            full.name = list(DOF_NAMES)
            full.position = [0.0] * 8
            full.velocity = [0.0] * 8
            pub_full.publish(full)

            for topic in counts:
                counts[topic] += 1
            seq += 1
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
        node.destroy_node()
        rclpy.shutdown()
        print(
            json.dumps(
                {"schema": "rocobrick-dds-type-matrix-host/v1", "gate": gate, "published_each": seq},
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
