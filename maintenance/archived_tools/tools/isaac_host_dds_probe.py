from __future__ import annotations

import argparse
import json
import sys
import time

from isaacsim import SimulationApp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", required=True)
    parser.add_argument("--expect", required=True)
    parser.add_argument("--timeout", type=float, default=35.0)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]

    app = SimulationApp({"headless": True})
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    app.update()

    import rclpy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("rocobrick_isaac_host_dds_probe")
    publisher = node.create_publisher(String, "/rocobrick/dds/host_to_container", 10)
    received: list[str] = []
    node.create_subscription(String, "/rocobrick/dds/container_to_host", lambda msg: received.append(msg.data), 10)
    deadline = time.monotonic() + args.timeout
    success_deadline: float | None = None
    next_publish = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_publish:
            publisher.publish(String(data=args.publish))
            next_publish = now + 0.25
        rclpy.spin_once(node, timeout_sec=0.1)
        app.update()
        if args.expect in received and success_deadline is None:
            success_deadline = time.monotonic() + 2.0
        if success_deadline is not None and time.monotonic() >= success_deadline:
            break
    result = {"schema": "rocobrick-host-dds-probe/v1", "published": args.publish, "expected": args.expect, "received": received, "passed": args.expect in received}
    print(json.dumps(result, sort_keys=True), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    app.close()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
