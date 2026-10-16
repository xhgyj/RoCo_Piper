#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/artifacts/phase0/rootless-dds-smoke.json}"
DOMAIN="${ROS_DOMAIN_ID:-73}"
HOST_PORT="${DDS_HOST_PORT:-17650}"
RUNTIME="$ROOT/.runtime/dds"
IMAGE="${ROS_IMAGE:-roco-piper-ros:humble}"
mkdir -p "$RUNTIME" "$(dirname "$OUT")"
# P1.A topology: the container profile is client-only (no listening port), so
# nothing container-side is published; the host listener owns TCP 17650 alone.
profiles_json="$($ROOT/.venv/bin/python "$ROOT/tools/generate_dds_profiles.py" --output-dir "$RUNTIME" --host-port "$HOST_PORT")"
work="$(mktemp -d)"
cleanup() { docker rm -f "${container_name:-}" >/dev/null 2>&1 || true; rm -rf "$work"; }
trap cleanup EXIT

run_round() {
  local round="$1"
  container_name="rocobrick-dds-smoke-${round}-$$"
  local host_payload="host-${round}-$$"
  local container_payload="container-${round}-$$"
  docker create --name "$container_name" \
    -e ROS_DOMAIN_ID="$DOMAIN" -e ROS_LOCALHOST_ONLY=0 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/dds/fastdds-container.xml \
    -v "$RUNTIME/fastdds-container.xml:/dds/fastdds-container.xml:ro" \
    -v "$ROOT/maintenance/archived_tools/tools/container_dds_probe.py:/probe.py:ro" \
    "$IMAGE" python3 /probe.py --publish "$container_payload" --expect "$host_payload" --timeout 60 >/dev/null
  docker start "$container_name" >/dev/null
  sleep 2
  ROS_DOMAIN_ID="$DOMAIN" FASTRTPS_DEFAULT_PROFILES_FILE="$RUNTIME/fastdds-host.xml" \
    timeout 90 bash "$ROOT/tools/isaac-ros-env.sh" "$ROOT/.venv/bin/python" \
    "$ROOT/maintenance/archived_tools/tools/isaac_host_dds_probe.py" --publish "$host_payload" \
    --expect "$container_payload" --timeout 60 >"$work/host-$round.log" 2>&1
  for _ in $(seq 1 20); do
    [[ "$(docker inspect "$container_name" --format '{{.State.Running}}')" == false ]] && break
    sleep 1
  done
  docker logs "$container_name" >"$work/container-$round.log" 2>&1
  docker rm -f "$container_name" >/dev/null
  container_name=""
  python3 - "$work/host-$round.log" "$work/container-$round.log" <<'PY'
import json, sys

def result(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line.startswith("{") and '"schema": "rocobrick-' in line:
            value = json.loads(line)
            if "passed" in value:
                return value
    raise SystemExit(f"probe result missing: {path}")
print(json.dumps({"host": result(sys.argv[1]), "container": result(sys.argv[2])}, sort_keys=True))
PY
}

round1="$(run_round 1)"
round2="$(run_round 2)"
image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
python3 - "$OUT" "$profiles_json" "$round1" "$round2" "$image_id" "$DOMAIN" <<'PY'
import json, sys
from datetime import datetime, timezone
out, profiles, one, two, image_id, domain = sys.argv[1:]
rounds = [json.loads(one), json.loads(two)]
report = {
    "schema": "rocobrick-rootless-dds-smoke/v1",
    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    "transport": "TCPv4",
    "ros_domain_id": int(domain),
    "image_id": image_id,
    "profiles": json.loads(profiles),
    "rounds": rounds,
    "container_restart_passed": len(rounds) == 2 and all(r["host"]["passed"] and r["container"]["passed"] for r in rounds),
    "host_isaac_runtime_used": True,
    "isaac_clock_tested": False,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(report, sort_keys=True))
raise SystemExit(0 if report["container_restart_passed"] else 1)
PY
