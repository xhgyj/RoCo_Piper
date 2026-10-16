#!/usr/bin/env python3
"""P1.A host-side multi-participant DDS gate probe (single participant).

Runs on the physical host with the bundled Humble rclpy (tools/isaac-ros-env.sh)
and the host-listener Fast DDS profile. It is CPU-only: no Isaac Sim, no
SimulationApp, no GPU. Exactly one rclpy node participates; it publishes one
unique payload per container participant (``host-><id>|<nonce>``) and must
receive each participant's unique payload (``<id>->host|<nonce>``). Evidence is
per participant, never aggregate.
"""
from __future__ import annotations

import argparse
import json
import time

import rclpy
from std_msgs.msg import String

SCHEMA = "rocobrick-p1-host-gate-probe/v1"
DEFAULT_TOPIC = "/rocobrick/p1/dds/gate"


def parse_pairs(items: list[str], flag: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or not value or key in pairs:
            raise SystemExit(f"bad or duplicate {flag} entry: {item!r}")
        pairs[key] = value
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="append", default=[],
                        help="id=nonce; publishes host->id|nonce for each participant")
    parser.add_argument("--expect", action="append", default=[],
                        help="id=nonce; expects id->host|nonce from each participant")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--timeout", type=float, default=70.0)
    parser.add_argument("--grace", type=float, default=2.0,
                        help="keep publishing after all expectations are met")
    args = parser.parse_args()

    targets = parse_pairs(args.target, "--target")
    expects = parse_pairs(args.expect, "--expect")
    if not targets or set(targets) != set(expects):
        raise SystemExit("--target and --expect must describe the same participant set")

    published = {pid: f"host->{pid}|{nonce}" for pid, nonce in sorted(targets.items())}
    expected = {pid: f"{pid}->host|{nonce}" for pid, nonce in sorted(expects.items())}

    rclpy.init()
    node = rclpy.create_node("rocobrick_p1_host_gate_probe")
    publisher = node.create_publisher(String, args.topic, 10)
    received: dict[str, int] = {}

    def on_message(msg: String) -> None:
        received[msg.data] = received.get(msg.data, 0) + 1

    node.create_subscription(String, args.topic, on_message, 10)

    deadline = time.monotonic() + args.timeout
    sent = 0
    while time.monotonic() < deadline:
        for payload in published.values():
            publisher.publish(String(data=payload))
            sent += 1
        rclpy.spin_once(node, timeout_sec=0.1)
        if all(payload in received for payload in expected.values()):
            break
    hold_until = time.monotonic() + args.grace
    while time.monotonic() < hold_until:
        for payload in published.values():
            publisher.publish(String(data=payload))
            sent += 1
        rclpy.spin_once(node, timeout_sec=0.1)

    result = {
        "schema": SCHEMA,
        "role": "host-listener",
        "published": published,
        "expected": expected,
        "received": {
            pid: {"payload": payload, "count": received.get(payload, 0)}
            for pid, payload in expected.items()
        },
        "unexpected_received": sorted(
            set(received) - set(expected.values()) - set(published.values())
        ),
        "sent_count": sent,
        "passed": all(received.get(payload, 0) > 0 for payload in expected.values()),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
