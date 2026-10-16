#!/usr/bin/env bash
# P1.C bounded LIVE smoke: real MoveIt POSE-target execution (moveit_msgs/
# action/MoveGroup, plan_only false) through the rootless ROS container ->
# ros2_control TopicBasedSystem -> /isaac_joint_command -> host Isaac
# articulation, with live /isaac_joint_states feedback and /compute_fk proof.
#
# Topology (P1.A frozen contract + P1.B additive container mesh, preserved):
#   - host: isaac/p05_host_bridge.py via tools/isaac-ros-env.sh — the single
#     bundled-rclpy participant owning the boundary TCP listener 17650
#     (host-listener profile); it publishes /clock and /isaac_joint_states and
#     subscribes /isaac_joint_command.
#   - container: ONE rootless container, every participant a client-only TCPv4
#     client of the host listener (fastdds-container-mesh.xml: no TCP listening
#     port anywhere; the profile additionally carries a netns-local UDPv4
#     rendezvous so the several container processes — ros2_control_node,
#     move_group, robot_state_publisher, spawners, probe — discover each other
#     inside the namespace).
#
# Orchestration, every step bounded:
#   0. preflight: the P1.A gate artifact must exist and have passed all
#      checks; offline model audit; client-only+mesh profile generation;
#      GPU availability diagnostic (recorded, non-fatal); NO foreign process
#      is ever killed;
#   1. start the host bridge in the background under `timeout` with long
#      hold/time limits (host 900s outer timeout, bridge time limit 600s,
#      hold 240s — shorter than the stop grace so the bridge always
#      self-exits orderly, max steps 1000000); bounded wait for the
#      p0.5_host_bridge_ready marker;
#   2. start ONE rootless container (unique name; ros_ws/tools/profiles/out
#      mounts; container mesh profile): source ROS + workspace, launch
#      move_group with use_sim_time:=true, bounded readiness wait (planning
#      service AND /compute_fk), run the bounded pose probe
#      tools/p1_real_pose_probe.py, capture the controller listing and the
#      in-namespace TCP listener evidence, teardown the launch;
#   3. stop the host orderly (wait for its bounded self-exit, then INT, then
#      KILL);
#   4. aggregate artifacts/phase1-real/pose-target.json (schema
#      rocobrick-p1-real-pose-smoke/v1) embedding the probe result, the host
#      evidence, the generated profile report and bounded log tails; scan all
#      logs for transport/hardware/controller/state/NaN failures.
# The script exits 0 only when every check passed; a cleanup trap removes the
# container and stops the host even on early failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${P1_OUT_DIR:-$ROOT/artifacts/phase1-real}"
AGGREGATE="$OUT_DIR/pose-target.json"
HOST_EVIDENCE="$OUT_DIR/host-pose-bridge.json"
HOST_LOG="$OUT_DIR/host-pose-bridge.log"
PROFILES_JSON="$OUT_DIR/dds-profiles-pose.json"
CONTAINER_LOG="$OUT_DIR/container-pose.log"
P1A_ARTIFACT="${P1A_ARTIFACT:-$ROOT/artifacts/phase1/dds-multiprocess-gate.json}"
RUNTIME="$ROOT/.runtime/dds"
DOMAIN="${ROS_DOMAIN_ID:-73}"
HOST_PORT="${DDS_HOST_PORT:-17650}"
IMAGE="${ROS_IMAGE:-roco-piper-ros:humble}"
WS_DIR="${P1_WS_DIR:-$ROOT/ros_ws}"
MARKER="p0.5_host_bridge_ready"
CONTAINER_NAME="rocobrick-p1-real-pose-$$"

# Bounds (seconds / steps): host side. HOLD_SECONDS MUST stay shorter than
# STOP_WAIT: live evidence (two runs) shows Kit intercepts SIGINT itself and
# exits 0 at the C level, so an externally INTerrupted bridge never runs the
# python cleanup that writes the evidence file. With hold < stop-grace the
# bridge always finishes its hold and self-exits orderly (hold_completed,
# command totals, evidence JSON) before any escalation is needed.
HOST_TIMEOUT="${P1_HOST_TIMEOUT:-900}"
TIME_LIMIT="${P1_TIME_LIMIT:-600}"
READY_TIMEOUT="${P1_READY_TIMEOUT:-120}"
HOLD_SECONDS="${P1_HOLD_SECONDS:-240}"
# The bridge hold ends only when BOTH hold bounds are satisfied, so the step
# bound must correspond to the wall bound (~500 physics steps/s wall): with
# 1000000 steps the hold would run to the 600s time limit and the stop
# escalation would hit Kit's SIGINT-exit before any evidence is written.
HOLD_STEPS="${P1_HOLD_STEPS:-125000}"
MAX_STEPS="${P1_MAX_STEPS:-1000000}"
READY_WAIT="${P1_READY_WAIT:-420}"
STOP_WAIT="${P1_STOP_WAIT:-300}"
AUDIT_TIMEOUT="${P1_AUDIT_TIMEOUT:-120}"
# Bounds: container side.
BUILD_TIMEOUT="${P1_BUILD_TIMEOUT:-300}"
MOVE_GROUP_WAIT="${P1_MOVE_GROUP_WAIT:-150}"
PROBE_TIMEOUT="${P1_PROBE_TIMEOUT:-480}"
PROBE_READY_TIMEOUT="${P1_PROBE_READY_TIMEOUT:-120}"
PROBE_GOAL_TIMEOUT="${P1_PROBE_GOAL_TIMEOUT:-90}"
PROBE_SETTLE_TIMEOUT="${P1_PROBE_SETTLE_TIMEOUT:-60}"
CONTROLLER_WAIT="${P1_CONTROLLER_WAIT:-30}"
TOTAL_TIMEOUT="${P1_TOTAL_TIMEOUT:-780}"
TAIL_BYTES=20000

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
  [[ -n "${HOST_PID:-}" ]] || return 0
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
  set +e
  stop_host
  if [[ -n "${CONTAINER_NAME:-}" ]]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  set -e
}
trap cleanup EXIT

write_aggregate() {
  # Aggregate host evidence + probe result + profiles + logs into one artifact.
  "$ROOT/.venv/bin/python" - \
    "$AGGREGATE" "$OUT_DIR" "$HOST_EVIDENCE" "$PROFILES_JSON" "$P1A_ARTIFACT" \
    "$HOST_LOG" "$CONTAINER_LOG" "$marker_seen" "${host_status:-none}" \
    "$CONTAINER_NAME" "$TAIL_BYTES" "${container_status:-none}" <<'PY'
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

(aggregate_path, out_dir, host_path, profiles_path, p1a_path, host_log,
 container_log, marker_seen, host_status, container_name, tail_bytes,
 container_status
 ) = sys.argv[1:13]
out = Path(out_dir)
tail_bytes = int(tail_bytes)


def read_text(name: str) -> str:
    try:
        return (out / name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_text_abs(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def bounded_tail(name: str) -> str:
    return read_text(name)[-tail_bytes:]


def run_output(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "scan-failed"


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


host = load_json(Path(host_path))
profiles = load_json(Path(profiles_path))
p1a = load_json(Path(p1a_path))

probe = None
for line in read_text("pose-probe.jsonl").splitlines():
    line = line.strip()
    if line.startswith("{") and "rocobrick-p1-real-pose-probe/v1" in line:
        try:
            probe = json.loads(line)
        except ValueError:
            probe = None

mesh = (profiles or {}).get("profiles", {}).get("container-mesh")
mesh_xml_path = Path(mesh.get("path")) if mesh else None
mesh_xml = ""
if mesh_xml_path is not None:
    try:
        mesh_xml = re.sub(r"<!--.*?-->", "", mesh_xml_path.read_text(encoding="utf-8"),
                          flags=re.DOTALL)
    except OSError:
        mesh_xml = ""

# In-namespace TCP listener evidence captured inside the container. Only
# loopback-bound listeners (127.0.0.1 / ::1) are acceptable: they are
# unreachable from outside the network namespace and therefore cannot become
# a rootless-boundary listening endpoint. Anything else fails the check.
LISTEN_ADDRESSES = {"0100007F", "00000000000000000000000001000000"}
listener_rows = []
for line in read_text("net-tcp-listeners-pose.txt").splitlines():
    parts = line.split()
    if not parts or parts[0].startswith("#"):
        continue
    address = parts[1].split(":")[0].upper() if len(parts) > 1 else ""
    if address not in LISTEN_ADDRESSES:
        listener_rows.append(line.strip()[:300])

host_commands = ((host or {}).get("commands") or {})
host_cleanup = ((host or {}).get("cleanup") or {})

report = {
    "schema": "rocobrick-p1-real-pose-smoke/v1",
    "phase": "P1.C",
    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    "container": {
        "image": (read_text("image-id-pose").strip() or None),
        "name": container_name,
        "exit_status": container_status,
    },
    "gpu_diagnostic": read_text("gpu-diagnostic-pose.txt").strip().splitlines(),
    "readiness_marker_seen": marker_seen == "1",
    "host_exit_status": host_status,
    "profiles": profiles,
    "p1a_prerequisite": {
        "path": str(p1a_path),
        "schema": (p1a or {}).get("schema"),
        "passed": bool(p1a and p1a.get("passed") and all((p1a.get("checks") or {}).values())),
    },
    "host_bridge": host,
    "probe": probe,
    "controllers_listing": bounded_tail("controllers-pose.txt"),
    "net_tcp_listeners": listener_rows,
    "host_log_tail": read_text_abs(host_log)[-tail_bytes:],
    "container_log_tail": read_text_abs(container_log)[-tail_bytes:],
    "probe_log_tail": bounded_tail("pose-probe.log"),
    "move_group_log_tail": bounded_tail("move_group-pose.log.tail"),
}

# ------------------------------------------------------------------ log scan
# The bare "shm" substring is NOT scanned: Kit/PhysX CUDA error lines carry
# file paths that contain it without any DDS meaning ("shared memory" and the
# case-sensitive Fast DDS token "SHM" cover the transport-fallback case).
SCAN_PATTERNS = [
    "address already in use",
    "error binding",
    "binding at port",
    "shared memory",
    "udpv4",
    "udpv6",
    "multicast",
    "hardware init",
    "failed to activate",
    "activation failed",
    "stale",
]
SHM_PATTERN = "SHM"  # case-sensitive: the Fast DDS shared-memory token
NAN_PATTERN = re.compile(r"\bnan\b", re.IGNORECASE)
findings = []
scanned = []
scan_sources = {
    "host-pose-bridge.log": read_text_abs(host_log),
    "container-pose.log": read_text_abs(container_log),
    "pose-probe.log": read_text("pose-probe.log"),
    "move_group-pose.log.tail": read_text("move_group-pose.log.tail"),
}
for name, text in scan_sources.items():
    scanned.append(name)
    for pattern in SCAN_PATTERNS:
        for line in text.splitlines():
            if pattern in line.lower():
                findings.append({"file": name, "pattern": pattern,
                                 "line": line.strip()[:300]})
    for line in text.splitlines():
        if SHM_PATTERN in line:
            findings.append({"file": name, "pattern": "SHM",
                             "line": line.strip()[:300]})
        if NAN_PATTERN.search(line):
            findings.append({"file": name, "pattern": "nan", "line": line.strip()[:300]})
scan_ok = not findings

readiness = (probe or {}).get("readiness") or {}
outbound = (probe or {}).get("outbound") or {}
return_summary = (probe or {}).get("return") or {}
command_stats = (probe or {}).get("command_stats") or {}
outbound_fk = (outbound.get("fk_verification") or {})

checks = {
    "p1a_prerequisite": bool(p1a and p1a.get("passed")
                             and all((p1a.get("checks") or {}).values())),
    "host_marker": report["readiness_marker_seen"],
    "container_profile_client_only_no_listener": bool(
        mesh
        and mesh.get("role") == "container-client-mesh"
        and mesh.get("listening_ports") == []
        and mesh.get("initial_peers")
        and mesh.get("local_spdp_multicast")
        and "<listening_ports>" not in mesh_xml
        and "<initialPeersList>" in mesh_xml
        and "<transport_id>rocobrick_tcp</transport_id>" in mesh_xml
        and not listener_rows
    ),
    "readiness": bool(
        probe
        and readiness.get("clock_increasing")
        and readiness.get("isaac_joint_state_messages", 0) >= 3
        and readiness.get("broadcast_joint_state_messages", 0) > 0
        and readiness.get("controllers_active")
        and readiness.get("arm_follow_action_ready")
        and readiness.get("move_group_action_name")
        and readiness.get("fk_service_ready")
        and probe.get("baseline_fresh")
        and probe.get("baseline_matches_broadcast")
        and probe.get("baseline_at_rest")
    ),
    "probe_passed": bool(probe and probe.get("passed")),
    "pose_outbound_passed": bool(outbound.get("passed")),
    "pose_fk_verification": bool(
        outbound_fk.get("passed")
        and outbound_fk.get("position_error_m") is not None
        and outbound_fk.get("orientation_error_rad") is not None
        and outbound_fk["position_error_m"] <= 0.01
        and outbound_fk["orientation_error_rad"] <= 0.03
    ),
    "pose_return_passed": bool(return_summary.get("passed")),
    "command_observed": bool(
        command_stats.get("count", 0) >= 1
        and command_stats.get("invalid", 1) == 0
        and command_stats.get("unknown_names") == []
    ),
    "host_commands_applied": bool(
        host_commands.get("received", 0) >= 1
        and host_commands.get("applied") == host_commands.get("received")
        and host_commands.get("clamped_to_limits") == 0
        and host_commands.get("unknown_joint_names_ignored") == []
    ),
    "host_cleanup_orderly": bool(
        host
        and host.get("ready")
        and not host.get("error")
        and host_cleanup.get("rclpy_shutdown") is True
        and host_cleanup.get("world_stopped") is True
        and host_cleanup.get("timeline_stopped") is True
        and (host_cleanup.get("app_closed") is True
             or host_cleanup.get("app_close_initiated") is True)
        and (host.get("passed") is True or host.get("stopped") == "sigint_orderly_stop")
    ),
    "log_scan_clean": scan_ok,
    "bounded_and_no_leftovers": True,  # refined below after the leftover scan
}
report["log_scan"] = {"patterns": SCAN_PATTERNS + ["SHM", "nan"], "files": scanned,
                      "findings": findings, "passed": scan_ok}

# Leftover scan with a small retry: a freshly SIGINTed/reaped host process or
# an in-flight --rm container removal can still be visible for a few seconds
# before it disappears; only a persistent leftover fails the check.
leftover_containers = None
leftover_hosts = None
for _attempt in range(4):
    leftover_containers = run_output(
        ["docker", "ps", "-a", "--filter", "name=rocobrick-p1-real-pose-",
         "--format", "{{.Names}}"]
    ).splitlines()
    leftover_hosts = run_output(["pgrep", "-f", "p05_host_bridge"]).splitlines()
    if not leftover_containers and not leftover_hosts:
        break
    time.sleep(3)
report["leftovers"] = {"container": leftover_containers or None,
                       "host_process": leftover_hosts or None}
checks["bounded_and_no_leftovers"] = (
    not leftover_containers and not leftover_hosts
)

report["checks"] = checks
report["passed"] = all(checks.values())
Path(aggregate_path).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({"schema": report["schema"], "passed": report["passed"],
                  "checks": checks, "aggregate": aggregate_path}, sort_keys=True))
if not report["passed"]:
    failed = sorted(name for name, ok in checks.items() if not ok)
    raise SystemExit(f"p1 real pose smoke FAILED: checks {failed}; artifacts {out_dir}")
PY
}

# ---------------------------------------------------------------- preflight
# 0a. P1.A prerequisite: the multiprocess DDS gate must have passed everything.
"$ROOT/.venv/bin/python" - "$P1A_ARTIFACT" <<'PY'
import json, sys
from pathlib import Path
artifact = Path(sys.argv[1])
if not artifact.is_file():
    raise SystemExit(f"P1.A gate artifact missing: {artifact}")
gate = json.loads(artifact.read_text(encoding="utf-8"))
if gate.get("schema") != "rocobrick-p1-dds-multiprocess-gate/v1" \
    or gate.get("passed") is not True \
    or not all((gate.get("checks") or {}).values()):
    raise SystemExit(f"P1.A gate artifact did not pass all checks: {artifact}")
print(json.dumps({"preflight": "p1a_prerequisite", "passed": True}))
PY

# 0b. Offline model audit: fail before any GPU time on model drift.
timeout --signal=INT --kill-after=15 "$AUDIT_TIMEOUT" \
  "$ROOT/.venv/bin/python" "$ROOT/tools/piper_l_model_audit.py" \
  --output "$OUT_DIR/preflight-model-audit-pose.json" \
  >"$OUT_DIR/preflight-model-audit-pose.log" 2>&1

# 0c. GPU availability diagnostic (recorded only; the host run itself fails
#     honestly if the GPU is unusable). Never kills any foreign process.
if command -v nvidia-smi >/dev/null 2>&1; then
  timeout 20 nvidia-smi -L >"$OUT_DIR/gpu-diagnostic-pose.txt" 2>&1 || true
else
  printf 'nvidia-smi not found\n' >"$OUT_DIR/gpu-diagnostic-pose.txt"
fi

# 0d. Client-only profiles plus the P1.B additive container mesh profile.
"$ROOT/.venv/bin/python" "$ROOT/tools/generate_dds_profiles.py" \
  --output-dir "$RUNTIME" \
  --host-port "$HOST_PORT" --container-mesh --ros-domain-id "$DOMAIN" \
  >"$PROFILES_JSON"

# ------------------------------------------------------------- host bridge
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
  write_aggregate >/dev/null 2>&1 || true
  echo "p1 real pose smoke FAILED: readiness marker not seen; see $HOST_LOG" >&2
  exit 1
fi

# ---------------------------------------------------------------- container
set +e
timeout --signal=INT --kill-after=30 "$TOTAL_TIMEOUT" \
  docker run --rm -i --name "$CONTAINER_NAME" \
    -e ROS_DOMAIN_ID="$DOMAIN" \
    -e ROS_LOCALHOST_ONLY=0 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/dds/fastdds-container-mesh.xml \
    -e P1_BUILD_TIMEOUT="$BUILD_TIMEOUT" \
    -e P1_MOVE_GROUP_WAIT="$MOVE_GROUP_WAIT" \
    -e P1_PROBE_TIMEOUT="$PROBE_TIMEOUT" \
    -e P1_PROBE_READY_TIMEOUT="$PROBE_READY_TIMEOUT" \
    -e P1_PROBE_GOAL_TIMEOUT="$PROBE_GOAL_TIMEOUT" \
    -e P1_PROBE_SETTLE_TIMEOUT="$PROBE_SETTLE_TIMEOUT" \
    -e P1_CONTROLLER_WAIT="$CONTROLLER_WAIT" \
    -v "$WS_DIR:/workspace/ros_ws" \
    -v "$ROOT/maintenance/archived_tools/tools/p1_real_pose_probe.py:/workspace/tools/p1_real_pose_probe.py:ro" \
    -v "$RUNTIME:/dds:ro" \
    -v "$OUT_DIR:/artifacts" \
    "$IMAGE" bash -s <<'INNER' >"$CONTAINER_LOG" 2>&1
set -euo pipefail
# The ament setup scripts reference optionally-unset trace variables; source
# them with -u relaxed (same fix as docker/ros-entrypoint.sh).
set +u
source /opt/ros/humble/setup.bash
if [ ! -f /workspace/ros_ws/install/setup.bash ]; then
  echo "[p1-real-pose-inner] install dir missing; building the two packages"
  cd /workspace/ros_ws
  timeout "${P1_BUILD_TIMEOUT:-300}" \
    colcon build --packages-select agx_arm_description rocobrick_piper_l_moveit_config \
    --symlink-install
  cd /
fi
source /workspace/ros_ws/install/setup.bash
set -u
mkdir -p /artifacts

# 1. move_group stack in the background on the LIVE sim clock.
nohup ros2 launch rocobrick_piper_l_moveit_config move_group.launch.py \
  use_sim_time:=true >/tmp/p1-pose-move_group.log 2>&1 &
LAUNCH_PID=$!

# 2. Bounded wait for the move_group planning service AND the /compute_fk
#    service (both provided by move_group; the pose probe needs both).
ready=0
deadline=$(( $(date +%s) + ${P1_MOVE_GROUP_WAIT:-150} ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  services="$(ros2 service list 2>/dev/null || true)"
  if echo "$services" | grep -q plan_kinematic_path \
     && echo "$services" | grep -q compute_fk; then
    ready=1
    break
  fi
  kill -0 "$LAUNCH_PID" 2>/dev/null || break
  sleep 2
done
echo "[p1-real-pose-inner] move_group precondition ready=$ready"

# 3. The bounded pose probe: readiness -> at-rest baseline -> FK target ->
#    IK preflight -> admission -> execute -> FK verify -> return.
probe_status=0
timeout --signal=INT --kill-after=10 "${P1_PROBE_TIMEOUT:-480}" \
  python3 /workspace/tools/p1_real_pose_probe.py \
    --timeout "${P1_PROBE_READY_TIMEOUT:-120}" \
    --goal-timeout "${P1_PROBE_GOAL_TIMEOUT:-90}" \
    --settle-timeout "${P1_PROBE_SETTLE_TIMEOUT:-45}" \
    >/tmp/p1-pose-probe.jsonl 2>/tmp/p1-pose-probe.log || probe_status=$?
cp /tmp/p1-pose-probe.jsonl /artifacts/pose-probe.jsonl 2>/dev/null || true
cp /tmp/p1-pose-probe.log /artifacts/pose-probe.log 2>/dev/null || true
echo "[p1-real-pose-inner] probe_status=$probe_status"

# 4. Controller listing (DIAGNOSTIC ONLY: recorded in the evidence; the probe
#    itself already required all three controllers active before moving).
timeout "${P1_CONTROLLER_WAIT:-30}" ros2 control list_controllers \
  >/artifacts/controllers-pose.txt 2>&1 || true

# 5. In-namespace TCP listener evidence: the mesh profile must NOT listen.
{
  echo "# LISTEN rows of /proc/net/tcp(+6) inside the container (must be empty)"
  awk '$4 == "0A"' /proc/net/tcp /proc/net/tcp6 2>/dev/null || true
} >/artifacts/net-tcp-listeners-pose.txt

# 6. Teardown: kill the launch, then bounded log tails out.
kill -INT "$LAUNCH_PID" 2>/dev/null || true
sleep 3
kill -KILL "$LAUNCH_PID" 2>/dev/null || true
wait 2>/dev/null || true
tail -c 20000 /tmp/p1-pose-move_group.log >/artifacts/move_group-pose.log.tail 2>/dev/null || true
exit "$probe_status"
INNER
container_status=$?
set -e

# ------------------------------------------------------- host orderly stop
stop_host
set +e
wait "$HOST_PID" 2>/dev/null
host_status=$?
HOST_PID=""
set -e

docker image inspect "$IMAGE" --format '{{.Id}}' >"$OUT_DIR/image-id-pose" 2>/dev/null || true

# ---------------------------------------------------------------- artifact
write_aggregate
"$ROOT/.venv/bin/python" - "$AGGREGATE" <<'PY'
import json, sys

report = json.loads(open(sys.argv[1], encoding="utf-8").read())
if not report["passed"]:
    failed = sorted(name for name, ok in report["checks"].items() if not ok)
    raise SystemExit(f"p1 real pose smoke FAILED: checks {failed}; artifacts {report['container']['name']}")
probe = report.get("probe") or {}
outbound = probe.get("outbound") or {}
fk = outbound.get("fk_verification") or {}
print(json.dumps({
    "schema": report["schema"],
    "passed": True,
    "move_group_action": (probe.get("readiness") or {}).get("move_group_action_name"),
    "tip_link": probe.get("tip_link"),
    "tcp_status": probe.get("tcp_status"),
    "displacement": probe.get("displacement"),
    "outbound_passed": outbound.get("passed"),
    "outbound_fk_position_error_m": fk.get("position_error_m"),
    "outbound_fk_orientation_error_rad": fk.get("orientation_error_rad"),
    "return_passed": (probe.get("return") or {}).get("passed"),
    "host_commands": (report.get("host_bridge") or {}).get("commands"),
}, sort_keys=True))
PY
