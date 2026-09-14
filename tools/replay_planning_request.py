#!/usr/bin/env python3
"""Replay a saved planning request without executing its trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpc
from google.protobuf.json_format import MessageToDict, Parse

from rocobrick.generated import planner_pb2, planner_pb2_grpc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("--planner", default="127.0.0.1:50052")
    parser.add_argument("--target-z", type=float)
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    request = Parse(args.request.read_text(), planner_pb2.PlanRequest())
    if args.target_z is not None:
        request.target.position_m[2] = args.target_z

    with grpc.insecure_channel(args.planner) as channel:
        response = planner_pb2_grpc.PlannerStub(channel).Plan(request, timeout=args.timeout)
    result = MessageToDict(response, preserving_proto_field_name=True)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
