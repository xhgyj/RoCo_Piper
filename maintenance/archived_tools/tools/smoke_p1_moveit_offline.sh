#!/usr/bin/env bash
# P1 bounded container-only MoveIt planning smoke (NO Isaac, NO GPU).
#
# Everything runs inside ONE roco-piper-ros:humble container sharing its
# default localhost DDS (no Fast DDS profile is passed): the fake joint-states
# feeder replaces ONLY the host Isaac bridge as the /isaac_joint_states
# source, the move_group stack launches with use_sim_time:=false, and the
# bounded probe validates the P1 planning chain plus the execution-limit
# narrowing (a joint4 = 2.0 rad wide-only goal must be rejected).
#
# Orchestration, every step bounded:
#   1. docker run --rm with a unique name, ./ros_ws mounted, executing the
#      inner bash sequence: source /opt/ros/humble + the ws install;
#      (re)build agx_arm_description + rocobrick_piper_l_moveit_config ONLY
#      if the ws install dir is missing; start the feeder in the background
#      (log file); start `ros2 launch rocobrick_piper_l_moveit_config
#      move_group.launch.py use_sim_time:=false` in the background (log
#      file); bounded wait for the probe preconditions; run the probe;
#      capture `ros2 control list_controllers` (DIAGNOSTIC ONLY: controller
#      activation failures are recorded, never fail this planning smoke);
#      teardown kills the background jobs;
#   2. always cleanup the container (trap);
#   3. write the aggregate artifact
#      artifacts/phase1/moveit-offline-planning.json embedding the probe
#      JSON + controller listing + bounded log tails. Aggregate passed =
#      probe passed (chain + valid goal planned + wide-only goal rejected).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${P1_OUT_DIR:-$ROOT/artifacts/phase1}"
AGGREGATE="$OUT_DIR/moveit-offline-planning.json"
CONTAINER_LOG="$OUT_DIR/container.log"
IMAGE="${ROS_IMAGE:-roco-piper-ros:humble}"
WS_DIR="${P1_WS_DIR:-$ROOT/ros_ws}"
DOMAIN="${ROS_DOMAIN_ID:-73}"
CONTAINER_NAME="rocobrick-p1-moveit-offline-$$"
TOTAL_TIMEOUT="${P1_TOTAL_TIMEOUT:-600}"
BUILD_TIMEOUT="${P1_BUILD_TIMEOUT:-300}"
PROBE_TIMEOUT="${P1_PROBE_TIMEOUT:-300}"
PROBE_READY_TIMEOUT="${P1_PROBE_READY_TIMEOUT:-90}"
PROBE_PLAN_TIMEOUT="${P1_PROBE_PLAN_TIMEOUT:-60}"
PRECONDITION_WAIT="${P1_PRECONDITION_WAIT:-120}"
CONTROLLER_WAIT="${P1_CONTROLLER_WAIT:-30}"
FEEDER_DURATION="${P1_FEEDER_DURATION:-420}"
TAIL_BYTES=20000

mkdir -p "$OUT_DIR"

cleanup() {
  # The container is --rm; if the host timeout killed the client first,
  # force-remove it so nothing keeps running.
  if [[ -n "${CONTAINER_NAME:-}" ]]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

set +e
timeout --signal=INT --kill-after=20 "$TOTAL_TIMEOUT" \
  docker run --rm -i --name "$CONTAINER_NAME" \
    -e ROS_DOMAIN_ID="$DOMAIN" \
    -e ROS_LOCALHOST_ONLY=0 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e P1_BUILD_TIMEOUT="$BUILD_TIMEOUT" \
    -e P1_PROBE_TIMEOUT="$PROBE_TIMEOUT" \
    -e P1_PROBE_READY_TIMEOUT="$PROBE_READY_TIMEOUT" \
    -e P1_PROBE_PLAN_TIMEOUT="$PROBE_PLAN_TIMEOUT" \
    -e P1_PRECONDITION_WAIT="$PRECONDITION_WAIT" \
    -e P1_CONTROLLER_WAIT="$CONTROLLER_WAIT" \
    -e P1_FEEDER_DURATION="$FEEDER_DURATION" \
    -v "$WS_DIR:/workspace/ros_ws" \
    -v "$ROOT/maintenance/archived_tools/tools/p1_fake_joint_states.py:/workspace/tools/p1_fake_joint_states.py:ro" \
    -v "$ROOT/maintenance/archived_tools/tools/p1_moveit_offline_probe.py:/workspace/tools/p1_moveit_offline_probe.py:ro" \
    -v "$OUT_DIR:/artifacts" \
    "$IMAGE" bash -s <<'INNER' >"$CONTAINER_LOG" 2>&1
set -euo pipefail
# The ament setup scripts reference optionally-unset trace variables; source
# them with -u relaxed (same fix as docker/ros-entrypoint.sh).
set +u
source /opt/ros/humble/setup.bash

# (Re)build the two packages ONLY if the ws install dir is missing.
if [ ! -f /workspace/ros_ws/install/setup.bash ]; then
  echo "[p1-inner] install dir missing; building the two packages"
  cd /workspace/ros_ws
  timeout "${P1_BUILD_TIMEOUT:-300}" \
    colcon build --packages-select agx_arm_description rocobrick_piper_l_moveit_config \
    --symlink-install
  cd /
fi
source /workspace/ros_ws/install/setup.bash
set -u
mkdir -p /artifacts

# 1. Feeder in the background: replaces ONLY the host Isaac bridge.
nohup python3 /workspace/tools/p1_fake_joint_states.py \
  --duration "${P1_FEEDER_DURATION:-420}" >/tmp/p1-feeder.log 2>&1 &
FEEDER_PID=$!

# 2. move_group stack in the background; offline => no sim time (/clock).
nohup ros2 launch rocobrick_piper_l_moveit_config move_group.launch.py \
  use_sim_time:=false >/tmp/p1-move_group.log 2>&1 &
LAUNCH_PID=$!

# 3. Bounded wait for the probe preconditions (planning service visible).
ready=0
deadline=$(( $(date +%s) + ${P1_PRECONDITION_WAIT:-120} ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ros2 service list 2>/dev/null | grep -q plan_kinematic_path; then
    ready=1
    break
  fi
  kill -0 "$LAUNCH_PID" 2>/dev/null || break
  sleep 2
done
echo "[p1-inner] precondition ready=$ready"

# 4. Run the bounded probe (it also waits for /joint_states + the service).
probe_status=0
timeout --signal=INT --kill-after=10 "${P1_PROBE_TIMEOUT:-300}" \
  python3 /workspace/tools/p1_moveit_offline_probe.py \
    --timeout "${P1_PROBE_READY_TIMEOUT:-90}" \
    --plan-timeout "${P1_PROBE_PLAN_TIMEOUT:-60}" \
    >/tmp/p1-probe.jsonl 2>/tmp/p1-probe.log || probe_status=$?
cp /tmp/p1-probe.jsonl /artifacts/probe.jsonl 2>/dev/null || true
cp /tmp/p1-probe.log /artifacts/probe.log 2>/dev/null || true
echo "[p1-inner] probe_status=$probe_status"

# 5. Controller listing: DIAGNOSTIC ONLY — controllers failing to activate
#    must NOT fail this smoke, but must be recorded in the evidence.
timeout "${P1_CONTROLLER_WAIT:-30}" ros2 control list_controllers \
  >/artifacts/controllers.txt 2>&1 || true

# 6. Teardown: kill the background jobs, then bounded log tails out.
kill -INT "$LAUNCH_PID" 2>/dev/null || true
sleep 3
kill -KILL "$LAUNCH_PID" 2>/dev/null || true
kill -KILL "$FEEDER_PID" 2>/dev/null || true
wait 2>/dev/null || true
tail -c 20000 /tmp/p1-feeder.log >/artifacts/feeder.log.tail 2>/dev/null || true
tail -c 20000 /tmp/p1-move_group.log >/artifacts/move_group.log.tail 2>/dev/null || true
exit "$probe_status"
INNER
container_status=$?
set -e

# Aggregate artifact; exit 0 only when the probe passed.
"$ROOT/.venv/bin/python" - \
  "$AGGREGATE" "$OUT_DIR" "$container_status" "$IMAGE" "$CONTAINER_NAME" "$TAIL_BYTES" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

aggregate_path, out_dir, container_status, image, container_name, tail_bytes = sys.argv[1:]
out = Path(out_dir)
tail_bytes = int(tail_bytes)


def read_text(name: str) -> str:
    try:
        return (out / name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def bounded_tail(name: str) -> str:
    return read_text(name)[-tail_bytes:]


probe = None
for line in read_text("probe.jsonl").splitlines():
    line = line.strip()
    if line.startswith("{") and "rocobrick-p1-moveit-offline/v1" in line:
        try:
            probe = json.loads(line)
        except ValueError:
            probe = None

report = {
    "schema": "rocobrick-p1-moveit-offline/v1",
    "phase": "P1",
    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    "container": {
        "image": image,
        "name": container_name,
        "exit_status": container_status,
    },
    "probe": probe,
    "controllers_listing": bounded_tail("controllers.txt"),
    "probe_log_tail": bounded_tail("probe.log"),
    "feeder_log_tail": bounded_tail("feeder.log.tail"),
    "move_group_log_tail": bounded_tail("move_group.log.tail"),
}
checks = {
    "container_exit_zero": container_status == "0",
    "probe_passed": bool(probe and probe.get("passed")),
    "valid_goal_planned": bool(
        probe and probe.get("valid_goal", {}).get("passed")
    ),
    "wide_goal_never_exceeds_limits": bool(
        probe and probe.get("wide_goal", {}).get("never_exceeds_limits")
    ),
}
report["checks"] = checks
report["passed"] = all(checks.values())
Path(aggregate_path).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({"schema": report["schema"], "passed": report["passed"],
                  "checks": checks, "aggregate": aggregate_path}, sort_keys=True))
if not report["passed"]:
    failed = sorted(name for name, ok in checks.items() if not ok)
    raise SystemExit(f"p1 moveit offline smoke FAILED: checks {failed}; log {out_dir}")
PY
