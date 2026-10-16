#!/usr/bin/env bash
# P0.5 bounded end-to-end smoke: host Isaac bridge + external ROS validation.
#
# The host process (isaac/p05_host_bridge.py) runs exactly ONE Fast DDS
# participant — a single in-process rclpy node created from the Humble rclpy
# bundled inside the Isaac install and wired by tools/isaac-ros-env.sh. The
# isaacsim.ros2.bridge Kit extension is NEVER enabled: enabling it would make
# the extension spawn additional in-process participants, and every
# participant built from the host profile binds TCP listening port 17650, so
# a second one aborts with "bind: Address already in use" (the live P0.5
# failure).
#
# Orchestration, every step bounded:
#   0. offline preflight model audit (runtime USD/URDF vs pinned
#      agx_arm_sim) so model drift fails before any GPU time is spent;
#   1. generate the solved rootless TCPv4 DDS profiles (.runtime/dds);
#   2. start the host bridge in the background under `timeout`;
#   3. wait for the p0.5_host_bridge_ready readiness marker in the host log;
#   4. start the rootless ROS container (container profile, published 17651
#      TCP port) running tools/p05_integration_probe.py: subscribe /clock and
#      /isaac_joint_states, verify the complete 8-DOF names and increasing
#      clock, publish a safe named /isaac_joint_command delta, verify commanded
#      movement and bounded idling of the uncommanded joints;
#   5. wait for the host to stop orderly (then INT, then KILL);
#   6. write the aggregate machine-readable artifact
#      artifacts/phase0.5/p05-integration.json and exit 0 only when the host
#      evidence and the external probe both passed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${P05_OUT_DIR:-$ROOT/artifacts/phase0.5}"
HOST_EVIDENCE="$OUT_DIR/host-bridge-smoke.json"
HOST_LOG="$OUT_DIR/host-bridge-smoke.log"
PROBE_LOG="$OUT_DIR/integration-probe.log"
AGGREGATE="$OUT_DIR/p05-integration.json"
PROFILES_JSON="$OUT_DIR/dds-profiles.json"
RUNTIME="$ROOT/.runtime/dds"
DOMAIN="${ROS_DOMAIN_ID:-73}"
HOST_PORT="${DDS_HOST_PORT:-17650}"
CONTAINER_PORT="${DDS_CONTAINER_PORT:-17651}"
IMAGE="${ROS_IMAGE:-roco-piper-ros:humble}"
HOST_TIMEOUT="${P05_SIM_TIMEOUT:-900}"
TIME_LIMIT="${P05_TIME_LIMIT:-300}"
MAX_STEPS="${P05_MAX_STEPS:-1000000}"
READY_TIMEOUT="${P05_READY_TIMEOUT:-60}"
HOLD_STEPS="${P05_HOLD_STEPS:-600}"
HOLD_SECONDS="${P05_HOLD_SECONDS:-90}"
READY_WAIT="${P05_READY_WAIT:-600}"
STOP_WAIT="${P05_STOP_WAIT:-150}"
PROBE_TIMEOUT="${P05_PROBE_TIMEOUT:-240}"
PROBE_READY_TIMEOUT="${P05_PROBE_READY_TIMEOUT:-90}"
PROBE_DRIVE_TIMEOUT="${P05_PROBE_DRIVE_TIMEOUT:-60}"
AUDIT_TIMEOUT="${P05_AUDIT_TIMEOUT:-120}"
MARKER="p0.5_host_bridge_ready"
CONTAINER_NAME="rocobrick-p05-probe-$$"

HOST_PID=""
mkdir -p "$RUNTIME" "$OUT_DIR"

host_alive() {
  # True while the host process runs (a reaped-but-unwaited zombie is dead).
  local stat
  stat="$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ')" || return 1
  [[ -n "$stat" && "$stat" != Z* ]]
}

stop_host() {
  # Wait for the host's own bounded self-exit; escalate INT then KILL.
  [[ -n "$HOST_PID" ]] || return 0
  host_alive "$HOST_PID" || return 0
  local waited
  for waited in $(seq 1 "$STOP_WAIT"); do
    host_alive "$HOST_PID" || return 0
    sleep 1
  done
  kill -INT "$HOST_PID" 2>/dev/null || true
  for waited in $(seq 1 30); do
    host_alive "$HOST_PID" || return 0
    sleep 1
  done
  kill -KILL "$HOST_PID" 2>/dev/null || true
}

cleanup() {
  stop_host || true
  if [[ -n "${CONTAINER_NAME:-}" ]]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

write_aggregate() {
  # Aggregate host evidence + external probe result into one artifact.
  "$ROOT/.venv/bin/python" - \
    "$AGGREGATE" "$HOST_EVIDENCE" "$PROBE_LOG" "$PROFILES_JSON" \
    "$marker_seen" "${host_status:-none}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

aggregate_path, host_path, probe_path, profiles_path, marker_seen, host_status = sys.argv[1:]

host = None
if Path(host_path).is_file():
    try:
        host = json.loads(Path(host_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        host = None

probe = None
try:
    lines = Path(probe_path).read_text(encoding="utf-8", errors="replace").splitlines()
except OSError:
    lines = []
for line in lines:
    line = line.strip()
    if line.startswith("{") and "rocobrick-p05-integration-probe/v1" in line:
        try:
            probe = json.loads(line)
        except ValueError:
            probe = None

profiles = None
if Path(profiles_path).is_file():
    try:
        profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        profiles = None

report = {
    "schema": "rocobrick-p05-integration/v1",
    "phase": "P0.5",
    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    "readiness_marker_seen": marker_seen == "1",
    "host_exit_status": host_status,
    "profiles": profiles,
    "host_bridge": host,
    "integration_probe": probe,
}
checks = {
    "readiness_marker_seen": report["readiness_marker_seen"],
    "host_evidence_passed": bool(host and host.get("passed") and host.get("ready")),
    "host_single_participant": bool(
        host
        and host.get("rclpy_node_count") == 1
        and host.get("omnigraph_created") is False
        and host.get("bridge_extension_enabled") is False
    ),
    "host_cleanup_orderly": bool(
        host
        and host.get("cleanup", {}).get("rclpy_shutdown") is True
        and host.get("cleanup", {}).get("world_stopped") is True
        and (
            host.get("cleanup", {}).get("app_closed") is True
            or host_status in {"0", "130", "143"}
        )
    ),
    "host_dofs_complete": bool(
        host and host.get("joint_dofs") == [
            "joint1", "joint2", "joint3", "joint4",
            "joint5", "joint6", "gripper_joint1", "gripper_joint2",
        ]
    ),
    "probe_passed": bool(probe and probe.get("passed")),
}
report["checks"] = checks
report["passed"] = all(checks.values())
Path(aggregate_path).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({"schema": report["schema"], "passed": report["passed"],
                  "checks": checks, "aggregate": aggregate_path}, sort_keys=True))
PY
}

# 0. Offline preflight: the runtime model must still match the pinned
#    agx_arm_sim audit; drift fails here before any GPU time is spent.
timeout --signal=INT --kill-after=15 "$AUDIT_TIMEOUT" \
  "$ROOT/.venv/bin/python" "$ROOT/tools/piper_l_model_audit.py" \
  --output "$OUT_DIR/preflight-model-audit.json" \
  >"$OUT_DIR/preflight-model-audit.log" 2>&1

# 1. Solved Fast DDS profiles (same generator as the validated DDS smoke).
"$ROOT/.venv/bin/python" "$ROOT/tools/generate_dds_profiles.py" \
  --output-dir "$RUNTIME" \
  --host-port "$HOST_PORT" --container-port "$CONTAINER_PORT" \
  >"$PROFILES_JSON"

# 2. Host bridge in the background, hard-bounded by `timeout`.
set +e
timeout --signal=INT --kill-after=30 "$HOST_TIMEOUT" \
  env ROS_DOMAIN_ID="$DOMAIN" \
  bash "$ROOT/tools/isaac-ros-env.sh" "$ROOT/.venv/bin/python" \
    "$ROOT/isaac/p05_host_bridge.py" \
    --max-steps "$MAX_STEPS" \
    --time-limit "$TIME_LIMIT" \
    --ready-timeout "$READY_TIMEOUT" \
    --hold-steps "$HOLD_STEPS" \
    --hold-seconds "$HOLD_SECONDS" \
    --evidence-out "$HOST_EVIDENCE" >"$HOST_LOG" 2>&1 &
HOST_PID=$!
set -e

# 3. Bounded wait for the readiness marker (never trust a silent bridge).
marker_seen=0
marker_deadline=$(( $(date +%s) + READY_WAIT ))
while (( $(date +%s) < marker_deadline )); do
  if grep -q "\"marker\": \"$MARKER\"" "$HOST_LOG" 2>/dev/null; then
    marker_seen=1
    break
  fi
  host_alive "$HOST_PID" || break
  sleep 2
done
host_status="running"
if [[ "$marker_seen" != 1 ]]; then
  stop_host
  set +e
  wait "$HOST_PID" 2>/dev/null
  host_status=$?
  HOST_PID=""
  set -e
  write_aggregate >/dev/null
  echo "p05 smoke FAILED: readiness marker not seen; see $HOST_LOG" >&2
  exit 1
fi

# 4. External probe in the rootless ROS container (bounded), 17651 published.
set +e
timeout --signal=INT --kill-after=15 "$PROBE_TIMEOUT" \
  docker run --rm --name "$CONTAINER_NAME" \
    -p "$CONTAINER_PORT:$CONTAINER_PORT/tcp" \
    -e ROS_DOMAIN_ID="$DOMAIN" -e ROS_LOCALHOST_ONLY=0 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/dds/fastdds-container.xml \
    -v "$RUNTIME/fastdds-container.xml:/dds/fastdds-container.xml:ro" \
    -v "$ROOT/tools/p05_integration_probe.py:/probe.py:ro" \
    "$IMAGE" python3 /probe.py \
    --timeout "$PROBE_READY_TIMEOUT" --drive-timeout "$PROBE_DRIVE_TIMEOUT" \
    >"$PROBE_LOG" 2>&1
probe_status=$?
set -e

# 5. Wait for the host's bounded self-exit; escalate to INT, then KILL.
stop_host
set +e
wait "$HOST_PID" 2>/dev/null
host_status=$?
HOST_PID=""
set -e

# 6. Aggregate artifact; exit 0 only when both sides passed.
write_aggregate
"$ROOT/.venv/bin/python" - "$AGGREGATE" "$probe_status" "$PROBE_LOG" <<'PY'
import json
import sys

aggregate_path, probe_status, probe_log = sys.argv[1:]
report = json.loads(open(aggregate_path, encoding="utf-8").read())
if not report["passed"]:
    failed = sorted(name for name, ok in report["checks"].items() if not ok)
    raise SystemExit(f"p05 smoke FAILED: checks {failed}; probe log {probe_log}")
print(json.dumps({"schema": report["schema"], "passed": True,
                  "marker": "p0.5_host_bridge_ready",
                  "aggregate": aggregate_path}, sort_keys=True))
PY
