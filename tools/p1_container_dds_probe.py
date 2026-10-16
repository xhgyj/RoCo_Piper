#!/usr/bin/env python3
"""P1.A container-side client-only DDS gate probe (one participant per process).

Every invocation is an independent container OS process with its own rclpy
context (never multiple nodes in one context) and its own client-only Fast DDS
profile (``fastdds-container-p<N>.xml`` pins a distinct participant ID). It
publishes its unique payload ``<id>->host|<nonce>`` and must receive the host
payloads addressed to it (``host-><id>|<nonce>``). Milestone JSON lines are
emitted (flushed) as each expected host payload arrives so per-round evidence
survives a later kill. ``--hold`` keeps a satisfied process alive until killed
or timed out; ``--gate-tag`` marks its command line for the smoke's pkill.
"""
from __future__ import annotations

import argparse
import json
import time

import rclpy
from std_msgs.msg import String

SCHEMA = "rocobrick-p1-container-gate-probe/v1"
DEFAULT_TOPIC = "/rocobrick/p1/dds/gate"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="participant label, e.g. p2")
    parser.add_argument("--participant-id", type=int, required=True,
                        help="Fast DDS participant ID pinned by its profile")
    parser.add_argument("--nonce", required=True, help="unique payload nonce")
    parser.add_argument("--expect-host", required=True,
                        help="comma-separated host nonces this process must receive")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--grace", type=float, default=2.0)
    parser.add_argument("--hold", action="store_true",
                        help="after expectations are met keep running until killed")
    parser.add_argument("--gate-tag", default="", help="unique kill marker")
    args = parser.parse_args()

    mine = f"{args.id}->host|{args.nonce}"
    wants = [f"host->{args.id}|{nonce}" for nonce in args.expect_host.split(",") if nonce]
    if not wants:
        raise SystemExit("no expected host payloads")

    rclpy.init()
    node = rclpy.create_node(f"rocobrick_p1_container_gate_{args.id}")
    publisher = node.create_publisher(String, args.topic, 10)
    received: dict[str, int] = {}

    def on_message(msg: String) -> None:
        received[msg.data] = received.get(msg.data, 0) + 1

    node.create_subscription(String, args.topic, on_message, 10)

    def emit(event: str, **extra: object) -> None:
        record = {
            "schema": SCHEMA,
            "event": event,
            "id": args.id,
            "participant_id": args.participant_id,
            "gate_tag": args.gate_tag,
        }
        record.update(extra)
        print(json.dumps(record, sort_keys=True), flush=True)

    emit("start", published=mine, expected_host_payloads=wants, hold=args.hold,
         timeout=args.timeout)

    deadline = time.monotonic() + args.timeout
    sent = 0
    satisfied_at: float | None = None
    announced: set[str] = set()
    while time.monotonic() < deadline:
        publisher.publish(String(data=mine))
        sent += 1
        rclpy.spin_once(node, timeout_sec=0.1)
        for payload in wants:
            # One flushed milestone per expected host payload so per-round
            # evidence survives a later kill (round-2 restart of --hold
            # processes) and spans multi-round processes.
            if received.get(payload, 0) > 0 and payload not in announced:
                announced.add(payload)
                emit("received", payload=payload, count=received[payload],
                     sent_count=sent)
        if all(payload in received for payload in wants):
            if args.hold:
                continue
            if satisfied_at is None:
                satisfied_at = time.monotonic()
            elif time.monotonic() - satisfied_at >= args.grace:
                break

    got_all = all(payload in received for payload in wants)
    emit("final", published=mine, sent_count=sent, expected_host_payloads=wants,
         received={payload: received.get(payload, 0) for payload in wants},
         unexpected_received=sorted(set(received) - set(wants) - {mine}),
         passed=got_all)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if got_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
