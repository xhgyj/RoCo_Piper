#!/usr/bin/env bash
# P1.A bounded live gate: container multi-participant Fast DDS across the
# rootless boundary (CPU-only; never starts Isaac Sim, never touches GPU).
#
# Topology under test (client-only):
#   - exactly ONE listening endpoint across the boundary: the host listener —
#     a single host participant (bundled rclpy via tools/isaac-ros-env.sh)
#     owning fixed TCP 17650;
#   - FOUR independent container OS processes (one `docker exec` each, each a
#     separate rclpy context, each with its own client-only
#     fastdds-container-p<N>.xml pinning a distinct participant ID); there is
#     NO container listening port anywhere.
#
# Protocol: one gate topic; payloads are addressed strings
#   host-><id>|<nonce>  (host -> participant)  and
#   <id>->host|<nonce>  (participant -> host);
# verification is per participant, per round, in both directions.
#
# Round 1: p1..p4 all pass the gate. Round 2: p2 and p3 are killed (SIGINT
# pkill on their unique gate tags), recreated with NEW nonces, and the whole
# gate repeats while p1/p4 continue across rounds and must also receive their
# round-2 host payloads.
#
# Every process and wait is bounded; the container name is unique per run; a
# cleanup trap (and an explicit final stop) remove all background clients, the
# container and the work dir; every log is scanned for zero "Address already
# in use" / "Error binding" / SHM / UDP / multicast fallback findings.
# Artifact: artifacts/phase1/dds-multiprocess-gate.json
# (schema rocobrick-p1-dds-multiprocess-gate/v1, passed = all(checks)).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/artifacts/phase1/dds-multiprocess-gate.json}"
DOMAIN="${ROS_DOMAIN_ID:-73}"
HOST_PORT="${DDS_HOST_PORT:-17650}"
IMAGE="${ROS_IMAGE:-roco-piper-ros:humble}"
RUNTIME="$ROOT/.runtime/dds"
PARTICIPANTS=(p1 p2 p3 p4)
RESTARTED=(p2 p3)

WORK="$(mktemp -d /tmp/rocobrick-p1-gate-XXXXXX)"
LOG_DIR="$WORK/logs"
CODE_DIR="$WORK/codes"
CONTAINER_NAME="rocobrick-p1-gate-$$"
mkdir -p "$LOG_DIR" "$CODE_DIR" "$RUNTIME" "$(dirname "$OUT")"

# Bounds (seconds): probe-internal timeouts and hard `timeout` wrappers.
HOST_PROBE_TIMEOUT=75    ; HOST_EXEC_TIMEOUT=110
P_SPAN_PROBE_TIMEOUT=200 ; P_SPAN_EXEC_TIMEOUT=240   # p1/p4 span both rounds
P_HOLD_PROBE_TIMEOUT=200 ; P_HOLD_EXEC_TIMEOUT=200   # p2/p3 round-1 (killed)
R2_PROBE_TIMEOUT=75      ; R2_EXEC_TIMEOUT=110
KILL_WAIT=15
BG_PIDS=()

rand_hex() { head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n'; }

declare -A NONCE_R1 HOST_NONCE_R1 HOST_NONCE_R2 NONCE_R2 KILLED_OK
for p in "${PARTICIPANTS[@]}"; do
  NONCE_R1[$p]="r1-${p}-$(rand_hex)"
  HOST_NONCE_R1[$p]="r1-h-${p}-$(rand_hex)"
  HOST_NONCE_R2[$p]="r2-h-${p}-$(rand_hex)"
done
for p in "${RESTARTED[@]}"; do
  NONCE_R2[$p]="r2-${p}-$(rand_hex)"
  KILLED_OK[$p]=0
done

stop_background() {
  local pid
  for pid in ${BG_PIDS[@]+"${BG_PIDS[@]}"}; do
    kill -INT "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in ${BG_PIDS[@]+"${BG_PIDS[@]}"}; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}

cleanup() {
  set +e
  stop_background
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  rm -rf "$WORK"
  set -e
}
trap cleanup EXIT

run_host_probe() { # name probe_timeout exec_timeout --target/--expect args...
  local name="$1" probe_timeout="$2" exec_timeout="$3"
  shift 3
  ROS_DOMAIN_ID="$DOMAIN" FASTRTPS_DEFAULT_PROFILES_FILE="$RUNTIME/fastdds-host.xml" \
    timeout --signal=INT --kill-after=10 "$exec_timeout" \
    bash "$ROOT/tools/isaac-ros-env.sh" "$ROOT/.venv/bin/python" \
      "$ROOT/tools/p1_host_dds_probe.py" --timeout "$probe_timeout" "$@" \
      >"$LOG_DIR/$name.log" 2>&1
}

run_container_probe() { # name id participant_id nonce expect_host probe_timeout exec_timeout extra_flags...
  local name="$1" id="$2" participant_id="$3" nonce="$4" expect_host="$5"
  local probe_timeout="$6" exec_timeout="$7"
  shift 7
  timeout --signal=INT --kill-after=10 "$exec_timeout" \
    docker exec -e ROS_DOMAIN_ID="$DOMAIN" -e ROS_LOCALHOST_ONLY=0 \
      -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
      -e FASTRTPS_DEFAULT_PROFILES_FILE="/dds/fastdds-container-${id}.xml" \
      "$CONTAINER_NAME" \
      bash -c "source /opt/ros/humble/setup.bash && exec python3 /probe.py --id ${id} --participant-id ${participant_id} --nonce ${nonce} --expect-host ${expect_host} --timeout ${probe_timeout} $*" \
      >"$LOG_DIR/$name.log" 2>&1
}

record_code() { # name exit_code
  printf '%s\n' "$2" >"$CODE_DIR/$1"
}

reap() { # name pid
  local rc=0
  set +e
  wait "$2"
  rc=$?
  set -e
  record_code "$1" "$rc"
}

# 0. Client-only profiles (host listener + container clients), plus facts.
"$ROOT/.venv/bin/python" "$ROOT/tools/generate_dds_profiles.py" \
  --output-dir "$RUNTIME" --host-port "$HOST_PORT" --container-count 4 \
  >"$WORK/profiles.json"
docker image inspect "$IMAGE" --format '{{.Id}}' >"$WORK/image_id"
{
  printf '{\n'
  printf '  "docker_context": "%s",\n' "$(docker context show 2>/dev/null || echo unknown)"
  printf '  "server_version": "%s",\n' "$(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo unknown)"
  printf '  "storage_driver": "%s",\n' "$(docker info --format '{{.Driver}}' 2>/dev/null || echo unknown)"
  printf '  "cgroup_version": "%s",\n' "$(docker info --format '{{.CgroupVersion}}' 2>/dev/null || echo unknown)"
  printf '  "security_options": %s,\n' "$(docker info --format '{{json .SecurityOptions}}' 2>/dev/null || echo '[]')"
  printf '  "rootless_context": true\n'
  printf '}\n'
} >"$WORK/facts.json"

CONTAINER_NAME="$CONTAINER_NAME" \
NONCE_R1_P1="${NONCE_R1[p1]}" NONCE_R1_P2="${NONCE_R1[p2]}" \
NONCE_R1_P3="${NONCE_R1[p3]}" NONCE_R1_P4="${NONCE_R1[p4]}" \
HOST_NONCE_R1_P1="${HOST_NONCE_R1[p1]}" HOST_NONCE_R1_P2="${HOST_NONCE_R1[p2]}" \
HOST_NONCE_R1_P3="${HOST_NONCE_R1[p3]}" HOST_NONCE_R1_P4="${HOST_NONCE_R1[p4]}" \
HOST_NONCE_R2_P1="${HOST_NONCE_R2[p1]}" HOST_NONCE_R2_P2="${HOST_NONCE_R2[p2]}" \
HOST_NONCE_R2_P3="${HOST_NONCE_R2[p3]}" HOST_NONCE_R2_P4="${HOST_NONCE_R2[p4]}" \
NONCE_R2_P2="${NONCE_R2[p2]}" NONCE_R2_P3="${NONCE_R2[p3]}" \
HOST_PROBE_TIMEOUT="$HOST_PROBE_TIMEOUT" HOST_EXEC_TIMEOUT="$HOST_EXEC_TIMEOUT" \
P_SPAN_PROBE_TIMEOUT="$P_SPAN_PROBE_TIMEOUT" P_SPAN_EXEC_TIMEOUT="$P_SPAN_EXEC_TIMEOUT" \
P_HOLD_PROBE_TIMEOUT="$P_HOLD_PROBE_TIMEOUT" P_HOLD_EXEC_TIMEOUT="$P_HOLD_EXEC_TIMEOUT" \
R2_PROBE_TIMEOUT="$R2_PROBE_TIMEOUT" R2_EXEC_TIMEOUT="$R2_EXEC_TIMEOUT" \
KILL_WAIT="$KILL_WAIT" \
  "$ROOT/.venv/bin/python" - "$WORK/plan.json" <<'PY'
import json
import os
import sys

participants = ["p1", "p2", "p3", "p4"]
env = os.environ
plan = {
    "participants": participants,
    "restarted": ["p2", "p3"],
    "participant_ids": {p: i + 1 for i, p in enumerate(participants)},
    "container_nonce_r1": {p: env[f"NONCE_R1_{p.upper()}"] for p in participants},
    "container_nonce_r2": {p: env[f"NONCE_R2_{p.upper()}"] for p in ("p2", "p3")},
    "host_nonce_r1": {p: env[f"HOST_NONCE_R1_{p.upper()}"] for p in participants},
    "host_nonce_r2": {p: env[f"HOST_NONCE_R2_{p.upper()}"] for p in participants},
    "container_name": env["CONTAINER_NAME"],
    "topic": "/rocobrick/p1/dds/gate",
    "bounds_seconds": {
        "host_probe": float(env["HOST_PROBE_TIMEOUT"]),
        "host_exec": float(env["HOST_EXEC_TIMEOUT"]),
        "p_span_probe": float(env["P_SPAN_PROBE_TIMEOUT"]),
        "p_span_exec": float(env["P_SPAN_EXEC_TIMEOUT"]),
        "p_hold_probe": float(env["P_HOLD_PROBE_TIMEOUT"]),
        "p_hold_exec": float(env["P_HOLD_EXEC_TIMEOUT"]),
        "round2_probe": float(env["R2_PROBE_TIMEOUT"]),
        "round2_exec": float(env["R2_EXEC_TIMEOUT"]),
        "kill_wait": float(env["KILL_WAIT"]),
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(plan, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

# 1. Bounded runtime self-check, then the detached gate container.
timeout 60 docker run -d --name "$CONTAINER_NAME" \
  -e ROS_DOMAIN_ID="$DOMAIN" -e ROS_LOCALHOST_ONLY=0 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/dds/fastdds-container.xml \
  -v "$RUNTIME:/dds:ro" \
  -v "$ROOT/tools/p1_container_dds_probe.py:/probe.py:ro" \
  "$IMAGE" sleep 900 >/dev/null
timeout 60 docker exec -e ROS_DOMAIN_ID="$DOMAIN" "$CONTAINER_NAME" \
  bash -c 'source /opt/ros/humble/setup.bash && python3 -c "import rclpy, std_msgs"'

# 2. ROUND 1: host probe plus four independent container OS processes.
target_args=(); expect_args=()
for p in "${PARTICIPANTS[@]}"; do
  target_args+=(--target "$p=${HOST_NONCE_R1[$p]}")
  expect_args+=(--expect "$p=${NONCE_R1[$p]}")
done
run_host_probe host-r1 "$HOST_PROBE_TIMEOUT" "$HOST_EXEC_TIMEOUT" \
  "${target_args[@]}" "${expect_args[@]}" & HOST_R1_PID=$!
BG_PIDS+=("$HOST_R1_PID")

run_container_probe p1-r1 p1 1 "${NONCE_R1[p1]}" \
  "${HOST_NONCE_R1[p1]},${HOST_NONCE_R2[p1]}" \
  "$P_SPAN_PROBE_TIMEOUT" "$P_SPAN_EXEC_TIMEOUT" & P1_R1_PID=$!
BG_PIDS+=("$P1_R1_PID")
run_container_probe p2-r1 p2 2 "${NONCE_R1[p2]}" \
  "${HOST_NONCE_R1[p2]}" \
  "$P_HOLD_PROBE_TIMEOUT" "$P_HOLD_EXEC_TIMEOUT" --hold --gate-tag p2-r1 & P2_R1_PID=$!
BG_PIDS+=("$P2_R1_PID")
run_container_probe p3-r1 p3 3 "${NONCE_R1[p3]}" \
  "${HOST_NONCE_R1[p3]}" \
  "$P_HOLD_PROBE_TIMEOUT" "$P_HOLD_EXEC_TIMEOUT" --hold --gate-tag p3-r1 & P3_R1_PID=$!
BG_PIDS+=("$P3_R1_PID")
run_container_probe p4-r1 p4 4 "${NONCE_R1[p4]}" \
  "${HOST_NONCE_R1[p4]},${HOST_NONCE_R2[p4]}" \
  "$P_SPAN_PROBE_TIMEOUT" "$P_SPAN_EXEC_TIMEOUT" & P4_R1_PID=$!
BG_PIDS+=("$P4_R1_PID")

reap host-r1 "$HOST_R1_PID"

# 3. ROUND 2: kill p2/p3 (unique gate tags), confirm death, recreate with NEW
#    nonces; p1/p4 keep running and must accept their round-2 host payloads.
for p in "${RESTARTED[@]}"; do
  # The [g] bracket keeps pkill/pgrep from matching their own bash wrapper.
  timeout 30 docker exec "$CONTAINER_NAME" \
    bash -c "pkill -INT -f 'gate-ta[g] ${p}-r1'" >/dev/null 2>&1 || true
  killed=0
  for _ in $(seq 1 "$KILL_WAIT"); do
    if timeout 30 docker exec "$CONTAINER_NAME" \
        bash -c "pgrep -f 'gate-ta[g] ${p}-r1' >/dev/null" >/dev/null 2>&1; then
      sleep 1
    else
      killed=1
      break
    fi
  done
  KILLED_OK[$p]=$killed
  record_code "kill-$p-r1" "$killed"
done
# Reap the killed round-1 p2/p3 exec clients (exit after the in-container kill).
reap p2-r1 "$P2_R1_PID"
reap p3-r1 "$P3_R1_PID"

for p in "${PARTICIPANTS[@]}"; do
  target_args_r2+=(--target "$p=${HOST_NONCE_R2[$p]}")
done
expect_args_r2=(--expect "p1=${NONCE_R1[p1]}" --expect "p2=${NONCE_R2[p2]}" \
  --expect "p3=${NONCE_R2[p3]}" --expect "p4=${NONCE_R1[p4]}")
run_host_probe host-r2 "$HOST_PROBE_TIMEOUT" "$HOST_EXEC_TIMEOUT" \
  "${target_args_r2[@]}" "${expect_args_r2[@]}" & HOST_R2_PID=$!
BG_PIDS+=("$HOST_R2_PID")

run_container_probe p2-r2 p2 2 "${NONCE_R2[p2]}" \
  "${HOST_NONCE_R2[p2]}" "$R2_PROBE_TIMEOUT" "$R2_EXEC_TIMEOUT" \
  --gate-tag p2-r2 & P2_R2_PID=$!
BG_PIDS+=("$P2_R2_PID")
run_container_probe p3-r2 p3 3 "${NONCE_R2[p3]}" \
  "${HOST_NONCE_R2[p3]}" "$R2_PROBE_TIMEOUT" "$R2_EXEC_TIMEOUT" \
  --gate-tag p3-r2 & P3_R2_PID=$!
BG_PIDS+=("$P3_R2_PID")

reap host-r2 "$HOST_R2_PID"
reap p2-r2 "$P2_R2_PID"
reap p3-r2 "$P3_R2_PID"
# p1/p4 exit only after receiving their round-2 host payloads.
reap p1-r1 "$P1_R1_PID"
reap p4-r1 "$P4_R1_PID"

# 4. Explicit bounded stop (the EXIT trap is the safety net), leftovers scan.
stop_background
docker logs "$CONTAINER_NAME" >"$LOG_DIR/docker-container.log" 2>&1 || true
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
sleep 1
leftover_container="$(docker ps -a --filter "name=rocobrick-p1-gate-" --format '{{.Names}}' | head -1 || true)"
leftover_probes="$(pgrep -f 'p1_host_dds_probe|p1_container_dds_probe' | head -1 || true)"

# 5. Artifact: per-participant, per-round, both-direction payload matrix.
"$ROOT/.venv/bin/python" - "$OUT" "$WORK" "$RUNTIME" "$WORK/profiles.json" \
  "$WORK/facts.json" "$WORK/image_id" "$DOMAIN" "$leftover_container" \
  "$leftover_probes" <<'PY'
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

out, work, dds_dir, profiles_path, facts_path, image_id_path, domain, \
    leftover_container, leftover_probes = sys.argv[1:]
work, dds_dir = Path(work), Path(dds_dir)
logs_dir, codes_dir = work / "logs", work / "codes"

plan = json.loads((work / "plan.json").read_text(encoding="utf-8"))
facts = json.loads(Path(facts_path).read_text(encoding="utf-8"))
profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
image_id = Path(image_id_path).read_text(encoding="utf-8").strip()

PROCESS_LOGS = ["p1-r1", "p2-r1", "p3-r1", "p4-r1", "p2-r2", "p3-r2"]
HOST_LOGS = {1: "host-r1", 2: "host-r2"}


def code(name):
    path = codes_dir / name
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def json_lines(path):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and '"schema": "rocobrick-' in line:
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
    return records


host_evidence = {}
for rnd, name in HOST_LOGS.items():
    records = [r for r in json_lines(logs_dir / f"{name}.log")
               if r.get("schema") == "rocobrick-p1-host-gate-probe/v1"]
    host_evidence[rnd] = records[-1] if records else None


def container_evidence(name):
    events = [e for e in json_lines(logs_dir / f"{name}.log")
              if e.get("schema") == "rocobrick-p1-container-gate-probe/v1"]
    start = next((e for e in events if e.get("event") == "start"), None)
    final = next((e for e in reversed(events) if e.get("event") == "final"), None)
    milestones = [e["payload"] for e in events if e.get("event") == "received"]
    return {"start": start, "final": final, "milestones": milestones}


rounds = []
for rnd in (1, 2):
    participants = {}
    for pid_label in plan["participants"]:
        if rnd == 1:
            process = f"{pid_label}-r1"
            nonce = plan["container_nonce_r1"][pid_label]
            host_nonce = plan["host_nonce_r1"][pid_label]
            role = "initial"
        elif pid_label in plan["restarted"]:
            process = f"{pid_label}-r2"
            nonce = plan["container_nonce_r2"][pid_label]
            host_nonce = plan["host_nonce_r2"][pid_label]
            role = "recreated_new_nonce"
        else:
            process = f"{pid_label}-r1"
            nonce = plan["container_nonce_r1"][pid_label]
            host_nonce = plan["host_nonce_r2"][pid_label]
            role = "continuing_across_rounds"
        sent_payload = f"{pid_label}->host|{nonce}"
        want_payload = f"host->{pid_label}|{host_nonce}"
        host = host_evidence[rnd]
        host_record = (host or {}).get("received", {}).get(pid_label, {})
        evidence = container_evidence(process)
        host_got = (
            host is not None
            and host_record.get("payload") == sent_payload
            and host_record.get("count", 0) > 0
        )
        host_sent = (host or {}).get("published", {}).get(pid_label) == want_payload
        container_got = (
            want_payload in evidence["milestones"]
            or (evidence["final"] or {}).get("received", {}).get(want_payload, 0) > 0
        )
        container_sent = (
            (evidence["start"] or {}).get("published") == sent_payload
            or (evidence["final"] or {}).get("published") == sent_payload
        )
        participants[pid_label] = {
            "process": process,
            "process_exit_code": code(process),
            "role_in_round": role,
            "participant_id": plan["participant_ids"][pid_label],
            "nonce": nonce,
            "host_nonce": host_nonce,
            "sent_payload": sent_payload,
            "send_evidence": {
                "published_by_process": container_sent,
                "received_by_host": host_got,
                "host_receive_count": host_record.get("count", 0) if host_got else 0,
            },
            "expected_host_payload": want_payload,
            "receive_evidence": {
                "published_by_host": host_sent,
                "received_by_process": container_got,
                "process_milestones": evidence["milestones"],
            },
            "passed": container_sent and host_got and host_sent and container_got,
        }
    entry = {
        "round": rnd,
        "participants": participants,
        "passed": all(v["passed"] for v in participants.values()),
    }
    if rnd == 2:
        entry["restarted"] = plan["restarted"]
    rounds.append(entry)

# Profile contract: report fields and the generated XML text itself.
def xml_text(name):
    text = (dds_dir / f"fastdds-{name}.xml").read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


container_names = [k for k, v in profiles["profiles"].items()
                   if v["role"] == "container-client"]
participant_ids = [profiles["profiles"][k]["participant_id"] for k in container_names]
host_xml = xml_text("host")
profiles_xml_ok = (
    host_xml.count("<listening_ports>") == 1
    and f"<port>{profiles['host_port']}</port>" in host_xml
    and "wan_addr" in host_xml
)
for name in container_names:
    xml = xml_text(name)
    profiles_xml_ok = profiles_xml_ok and "<listening_ports>" not in xml \
        and "<initialPeersList>" in xml
profiles_contract_ok = (
    profiles["schema"] == "rocobrick-dds-profiles/v2"
    and profiles["topology"] == "single-host-listener-17650-client-only-containers"
    and profiles["profiles"]["host"]["role"] == "host-listener"
    and profiles["profiles"]["host"]["listening_ports"] == [profiles["host_port"]]
    and all(profiles["profiles"][k]["listening_ports"] == [] for k in container_names)
    and all(profiles["profiles"][k]["initial_peers"]
            == [[profiles["host_address"], profiles["host_port"]]]
            for k in container_names)
    and len(set(participant_ids)) == len(participant_ids)
    and profiles_xml_ok
    and "container_port" not in profiles
)

restart_ok = all(
    plan["container_nonce_r2"][p] != plan["container_nonce_r1"][p]
    and code(f"kill-{p}-r1") == 1
    and code(f"{p}-r1") not in (0, None)  # old process ended by the kill
    and code(f"{p}-r2") == 0
    for p in plan["restarted"]
)

all_nonces = (
    list(plan["container_nonce_r1"].values())
    + list(plan["container_nonce_r2"].values())
    + list(plan["host_nonce_r1"].values())
    + list(plan["host_nonce_r2"].values())
)

expected_clean_processes = ["host-r1", "host-r2", "p1-r1", "p4-r1", "p2-r2", "p3-r2"]
exit_ok = all(code(name) == 0 for name in expected_clean_processes)

SCAN_PATTERNS = ["address already in use", "error binding", "binding at port",
                 "shared memory", "shm", "udpv4", "udpv6", "multicast"]
findings = []
scanned = []
for log_file in sorted(logs_dir.glob("*.log")):
    scanned.append(log_file.name)
    text = log_file.read_text(encoding="utf-8", errors="replace")
    for pattern in SCAN_PATTERNS:
        for line in text.splitlines():
            if pattern in line.lower():
                findings.append({"file": log_file.name, "pattern": pattern,
                                 "line": line.strip()[:300]})
scan_ok = not findings

cleanup_ok = (
    leftover_container == ""
    and leftover_probes == ""
    and bool(plan["bounds_seconds"])
)

checks = {
    "profiles_contract": profiles_contract_ok,
    "round1_all_participants_bidirectional": rounds[0]["passed"],
    "round2_all_participants_bidirectional": rounds[1]["passed"],
    "round2_restart_new_nonce": restart_ok,
    "per_participant_payloads_unique": len(set(all_nonces)) == len(all_nonces),
    "process_exit_codes_clean": exit_ok,
    "log_scan_clean": scan_ok,
    "bounded_and_no_leftovers": cleanup_ok,
}
report = {
    "schema": "rocobrick-p1-dds-multiprocess-gate/v1",
    "phase": "P1.A",
    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    "transport": "TCPv4",
    "topology": profiles["topology"],
    "topology_description": (
        "exactly one boundary listening endpoint: host listener on fixed TCP "
        f"{profiles['host_port']}; every container participant is client-only "
        "TCPv4 (no listening_ports) with an explicit initialPeersList locator"
    ),
    "ros_domain_id": int(domain),
    "host_address": profiles["host_address"],
    "host_port": profiles["host_port"],
    "image_id": image_id,
    "container_name": plan["container_name"],
    "process_model": ("four independent container OS processes (docker exec), "
                      "each a separate rclpy participant with its own "
                      "pinned-ID client-only profile"),
    "rootless": facts,
    "profiles": profiles,
    "plan": plan,
    "rounds": rounds,
    "processes": {name: {"exit_code": code(name),
                         "killed_for_restart": name.endswith("-r1")
                         and name[:2] in plan["restarted"],
                         "log": f"{name}.log"}
                  for name in PROCESS_LOGS + list(HOST_LOGS.values())},
    "log_scan": {"patterns": SCAN_PATTERNS, "files": scanned,
                 "findings": findings, "passed": scan_ok},
    "checks": checks,
}
report["passed"] = all(checks.values())
Path(out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
print(json.dumps({"schema": report["schema"], "passed": report["passed"],
                  "checks": checks, "artifact": out}, sort_keys=True))
if not report["passed"]:
    failed = sorted(name for name, ok in checks.items() if not ok)
    raise SystemExit(f"p1 dds multiprocess gate FAILED: {failed}")
PY
