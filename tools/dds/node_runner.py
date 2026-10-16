#!/usr/bin/env python3
"""Single-context ROS 2 talker/listener used by the bounded DDS smoke test.

Deliberately runs ONE rclpy context per process (one DomainParticipant) so the
fixed participant ID from the generated Fast DDS profile stays valid. The
container side and the host side both execute this same file:

- container: ``python3 node_runner.py ...`` with the ROS Humble environment;
- host:      the Isaac Sim venv Python with the environment exported by
             ``tools/dds/isaac_host_env.sh`` (or ``--via-isaac`` to boot
             ``SimulationApp`` first, which also makes rclpy importable).

Payloads carry a role prefix so both directions are verified by content, not
just by discovery: a listener only accepts messages whose data starts with the
expected peer prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("talker", "listener", "both"), required=True)
    parser.add_argument("--topic", default="/rocobrick/dds_smoke")
    parser.add_argument("--count", type=int, default=5, help="messages to send / expect")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds before giving up")
    parser.add_argument("--publish-prefix", default="host", help="prefix embedded in published payloads")
    parser.add_argument("--expect-prefix", default=None, help="only accept payloads starting with this")
    parser.add_argument("--report", type=Path, default=None, help="write a JSON result report")
    parser.add_argument(
        "--via-isaac",
        action="store_true",
        help="boot Isaac SimulationApp (headless) before importing rclpy",
    )
    return parser.parse_args(argv)


def start_simulation_app() -> None:
    from isaacsim.simulation_app import SimulationApp  # noqa: F401  (boots Kit/carb)

    app = SimulationApp({"headless": True})
    globals()["_SIMULATION_APP"] = app


def run(args: argparse.Namespace) -> dict:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    result: dict = {
        "topic": args.topic,
        "mode": args.mode,
        "sent": 0,
        "received": [],
        "foreign_messages_seen": 0,
        "timeout": args.timeout,
        "expect_prefix": args.expect_prefix,
    }
    try:
        node = Node("rocobrick_dds_smoke_runner")
        received: list[str] = []
        foreign = []

        def on_message(msg: String) -> None:
            if args.expect_prefix is not None:
                if msg.data.startswith(args.expect_prefix):
                    received.append(msg.data)
                else:
                    foreign.append(msg.data)
            else:
                received.append(msg.data)

        subscription = None
        if args.mode in ("listener", "both"):
            subscription = node.create_subscription(String, args.topic, on_message, 10)
        publisher = None
        if args.mode in ("talker", "both"):
            publisher = node.create_publisher(String, args.topic, 10)

        # ROS 2 default durability is volatile: only payloads published while a
        # reader is matched are guaranteed delivered. The talker therefore
        # publishes cyclically for the whole window so a listener that matches
        # late still receives the payload; the listener exits as soon as it has
        # counted `count` peer messages.
        deadline = time.monotonic() + args.timeout
        index = 0
        # After the runner has counted `count` verified peer payloads it keeps
        # publishing one more full cycle so the peer (which exits on the same
        # condition) cannot miss the tail of the stream to exit-order jitter.
        extra_cycle_budget = None
        while time.monotonic() < deadline:
            if publisher is not None:
                publisher.publish(String(data=f"{args.publish_prefix}-{index}"))
                result["sent"] += 1
                index += 1
                if index >= args.count:
                    index = 0
            if subscription is not None:
                rclpy.spin_once(node, timeout_sec=0.1)
            else:
                time.sleep(0.1)
            if len(received) >= args.count and extra_cycle_budget is None:
                extra_cycle_budget = args.count
            if extra_cycle_budget is not None:
                extra_cycle_budget -= 1
                if extra_cycle_budget <= 0 and len(received) >= args.count:
                    break

        result["received"] = received
        result["foreign_messages_seen"] = len(foreign)
        result["foreign_sample"] = foreign[:3]
        result["ok"] = (args.mode == "listener" or result["sent"] >= args.count) and (
            args.expect_prefix is None or len(received) >= args.count
        )
    finally:
        rclpy.shutdown()

    return result


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    app = None
    if args.via_isaac:
        start_simulation_app()
        app = globals().get("_SIMULATION_APP")
    try:
        result = run(args)
    finally:
        if app is not None:
            app.close()
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
