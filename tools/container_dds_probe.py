from __future__ import annotations

import argparse
import json
import time

import rclpy
from std_msgs.msg import String


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", required=True)
    parser.add_argument("--expect", required=True)
    parser.add_argument("--timeout", type=float, default=35.0)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("rocobrick_container_dds_probe")
    publisher = node.create_publisher(String, "/rocobrick/dds/container_to_host", 10)
    received: list[str] = []
    node.create_subscription(String, "/rocobrick/dds/host_to_container", lambda msg: received.append(msg.data), 10)
    deadline = time.monotonic() + args.timeout
    success_deadline: float | None = None
    next_publish = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_publish:
            publisher.publish(String(data=args.publish))
            next_publish = now + 0.25
        rclpy.spin_once(node, timeout_sec=0.1)
        if args.expect in received and success_deadline is None:
            success_deadline = time.monotonic() + 2.0
        if success_deadline is not None and time.monotonic() >= success_deadline:
            break
    result = {"schema": "rocobrick-container-dds-probe/v1", "published": args.publish, "expected": args.expect, "received": received, "passed": args.expect in received}
    print(json.dumps(result, sort_keys=True), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
