#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m grpc_tools.protoc \
  -I "${repo_root}/proto/planner/v1" \
  --python_out="${repo_root}/src/rocobrick/generated" \
  --grpc_python_out="${repo_root}/src/rocobrick/generated" \
  "${repo_root}/proto/planner/v1/planner.proto"
sed -i 's/^import planner_pb2 as planner__pb2$/from . import planner_pb2 as planner__pb2/' \
  "${repo_root}/src/rocobrick/generated/planner_pb2_grpc.py"
