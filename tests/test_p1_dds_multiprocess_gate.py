"""P1.A offline contract: container multi-participant Fast DDS gate.

Covers the client-only rootless topology (exactly one boundary listening
endpoint — the host listener on TCP 17650 — and container participants with no
listening port at all), the P1 multi-process smoke structure, and the live
gate artifact produced by tools/smoke_p1_dds_multiprocess.sh.

Historical P0/P0.5 evidence is preserved untouched: the 17651 scan is scoped
to the ACTIVE compose/generator/template/smoke files only.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
GENERATOR_PATH = ROOT / "tools/generate_dds_profiles.py"
TEMPLATE_PATH = ROOT / "config/dds/fastdds-rootless-tcp.xml.in"
COMPOSE_PATH = ROOT / "compose.yaml"
SMOKE_P1_PATH = ROOT / "tools/smoke_p1_dds_multiprocess.sh"
SMOKE_ROOTLESS_PATH = ROOT / "tools/smoke_rootless_dds.sh"
HOST_PROBE_PATH = ROOT / "tools/p1_host_dds_probe.py"
CONTAINER_PROBE_PATH = ROOT / "tools/p1_container_dds_probe.py"
ARTIFACT_PATH = ROOT / "artifacts/phase1/dds-multiprocess-gate.json"

ACTIVE_FILES = (
    COMPOSE_PATH,
    GENERATOR_PATH,
    TEMPLATE_PATH,
    SMOKE_P1_PATH,
    SMOKE_ROOTLESS_PATH,
)
PARTICIPANTS = ("p1", "p2", "p3", "p4")
RESTARTED = ("p2", "p3")
TOPOLOGY = "single-host-listener-17650-client-only-containers"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _without_xml_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _generate(tmp_path: Path, *extra: str) -> tuple[Path, dict]:
    output = tmp_path / "dds"
    result = subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--output-dir", str(output), *extra],
        capture_output=True,
        text=True,
        check=True,
    )
    return output, json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Generator: two distinct roles, topology and per-profile role stated.
# ---------------------------------------------------------------------------


def test_generator_report_states_topology_and_roles(tmp_path: Path) -> None:
    _, report = _generate(tmp_path)
    assert report["schema"] == "rocobrick-dds-profiles/v2"
    assert report["topology"] == TOPOLOGY
    assert report["transport"] == "TCPv4"
    assert report["profiles"]["host"]["role"] == "host-listener"
    for name in ("container", "container-p1", "container-p2", "container-p3",
                 "container-p4"):
        assert report["profiles"][name]["role"] == "container-client"
    assert set(report["roles"]) == {"host-listener", "container-client"}
    # No stale fields: the obsolete container port is gone from the report.
    assert "container_port" not in report


def test_host_profile_is_the_only_boundary_listener(tmp_path: Path) -> None:
    output, report = _generate(tmp_path)
    xml = _read(output / "fastdds-host.xml")
    assert xml.count("<listening_ports>") == 1
    assert f"<port>{report['host_port']}</port>" in xml
    assert f"<wan_addr>{report['host_address']}</wan_addr>" in xml
    assert "<initialPeersList>" not in xml  # the listener only accepts clients
    assert report["profiles"]["host"]["listening_ports"] == [report["host_port"]]
    assert report["profiles"]["host"]["initial_peers"] == []
    assert report["profiles"]["host"]["participant_id"] == 0


def test_container_profiles_are_client_only_with_explicit_peer(tmp_path: Path) -> None:
    output, report = _generate(tmp_path)
    for name in ("container", "container-p1", "container-p2", "container-p3",
                 "container-p4"):
        xml = _without_xml_comments(_read(output / f"fastdds-{name}.xml"))
        assert "<listening_ports>" not in xml  # client-only: binds nothing
        assert "wan_addr" not in xml
        assert "<initialPeersList>" in xml
        assert f"<address>{report['host_address']}</address>" in xml
        assert f"<physical_port>{report['host_port']}</physical_port>" in xml
        assert report["profiles"][name]["listening_ports"] == []
        assert report["profiles"][name]["initial_peers"] == [
            [report["host_address"], report["host_port"]]
        ]


def test_container_participant_ids_are_pinned_and_distinct(tmp_path: Path) -> None:
    output, report = _generate(tmp_path)
    base = _read(output / "fastdds-container.xml")
    assert "<participantID>" not in base  # base runtime profile: automatic ID
    ids = []
    for index, name in enumerate(("container-p1", "container-p2", "container-p3",
                                  "container-p4"), start=1):
        xml = _read(output / f"fastdds-{name}.xml")
        assert f"<participantID>{index}</participantID>" in xml
        assert report["profiles"][name]["participant_id"] == index
        ids.append(report["profiles"][name]["participant_id"])
    assert len(set(ids)) == len(ids) == 4


def test_generated_profiles_keep_no_shm_no_udp_no_multicast(tmp_path: Path) -> None:
    output, _ = _generate(tmp_path)
    for xml_path in sorted(output.glob("*.xml")):
        xml = _without_xml_comments(_read(xml_path))
        assert "<useBuiltinTransports>false</useBuiltinTransports>" in xml
        assert "<avoid_builtin_multicast>true</avoid_builtin_multicast>" in xml
        assert "<userTransports>" in xml
        assert "UDPv4" not in xml and "UDPv6" not in xml
        assert "SHM" not in xml and "builtin_transports" not in xml


def test_generator_keeps_p05_call_compatible_and_ignores_old_port(
    tmp_path: Path,
) -> None:
    """The P0.5 smoke still calls --container-port; it must be accepted (and
    ignored: the container role is client-only now)."""
    _, plain = _generate(tmp_path)
    output, legacy_call = _generate(tmp_path, "--container-port", "17651",
                                    "--host-port", "17650")
    assert "container_port" not in legacy_call
    assert legacy_call["profiles"]["container"] == plain["profiles"]["container"]
    assert "17651" not in _read(output / "fastdds-container.xml")


# ---------------------------------------------------------------------------
# Active files: no obsolete 17651 anywhere; compose stops publishing it.
# ---------------------------------------------------------------------------


def test_no_obsolete_container_port_in_active_files() -> None:
    for path in ACTIVE_FILES:
        text = _read(path)
        assert "17651" not in text, f"stale port in active file: {path}"
        assert "DDS_CONTAINER_PORT" not in text, f"stale variable in: {path}"


def test_compose_stops_publishing_a_container_port() -> None:
    compose = _read(COMPOSE_PATH)
    assert "ports:" not in compose  # nothing is published across the boundary
    assert 'FASTRTPS_DEFAULT_PROFILES_FILE: /etc/rocobrick/dds/fastdds-container.xml' in compose
    assert "./.runtime/dds/fastdds-container.xml:/etc/rocobrick/dds/fastdds-container.xml:ro" in compose
    assert "network_mode: host" not in compose
    assert "driver: bridge" in compose  # rootless bridge network preserved
    assert 'ROS_LOCALHOST_ONLY: "0"' in compose
    assert "RMW_IMPLEMENTATION: rmw_fastrtps_cpp" in compose
    assert "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-73}" in compose
    assert 'test: ["CMD", "self-check"]' in compose  # runtime self-check kept


def test_smoke_rootless_dds_uses_client_only_profiles() -> None:
    smoke = _read(SMOKE_ROOTLESS_PATH)
    assert "generate_dds_profiles.py" in smoke
    # No container port is published: the old `-p "$PORT:$PORT/tcp"` flag and
    # its variable are gone; only the client-only profile mount remains.
    assert '"$CONTAINER_PORT:$CONTAINER_PORT/tcp"' not in smoke
    assert "CONTAINER_PORT" not in smoke
    assert not re.search(r"docker (run|create)[^\n]*\s-p\b", smoke)
    assert "fastdds-container.xml" in smoke


# ---------------------------------------------------------------------------
# P1 multi-process smoke structure.
# ---------------------------------------------------------------------------


def test_smoke_gate_runs_four_independent_processes_and_two_rounds() -> None:
    smoke = _read(SMOKE_P1_PATH)
    assert 'PARTICIPANTS=(p1 p2 p3 p4)' in smoke
    # Four independent container OS processes per round (docker exec each).
    for label in ("p1-r1", "p2-r1", "p3-r1", "p4-r1", "p2-r2", "p3-r2"):
        assert f"run_container_probe {label}" in smoke, label
    # Two host probe rounds.
    assert "run_host_probe host-r1" in smoke
    assert "run_host_probe host-r2" in smoke
    # One host participant, bundled rclpy, no Isaac Sim/GPU in the gate.
    assert "tools/isaac-ros-env.sh" in smoke
    assert "tools/p1_host_dds_probe.py" in smoke
    assert "SimulationApp" not in smoke


def test_smoke_gate_kills_and_recreates_with_new_nonce() -> None:
    smoke = _read(SMOKE_P1_PATH)
    assert 'RESTARTED=(p2 p3)' in smoke
    assert "--hold --gate-tag p2-r1" in smoke
    assert "--hold --gate-tag p3-r1" in smoke
    # The [g] bracket keeps pkill/pgrep from matching their own wrapper.
    assert "pkill -INT -f 'gate-ta[g] ${p}-r1'" in smoke
    assert "pgrep -f 'gate-ta[g] ${p}-r1'" in smoke  # death is confirmed
    assert 'NONCE_R2[$p]="r2-${p}-$(rand_hex)"' in smoke  # NEW round-2 nonces
    # Round-2 host expectations mix continuing and recreated participants.
    assert '--expect "p1=${NONCE_R1[p1]}"' in smoke
    assert '--expect "p4=${NONCE_R1[p4]}"' in smoke
    assert '--expect "p2=${NONCE_R2[p2]}"' in smoke
    assert '--expect "p3=${NONCE_R2[p3]}"' in smoke
    # Round-1 spanning processes must expect both their host nonces.
    assert '"${HOST_NONCE_R1[p1]},${HOST_NONCE_R2[p1]}"' in smoke
    assert '"${HOST_NONCE_R1[p4]},${HOST_NONCE_R2[p4]}"' in smoke


def test_smoke_gate_is_bounded_and_cleans_up() -> None:
    smoke = _read(SMOKE_P1_PATH)
    assert 'CONTAINER_NAME="rocobrick-p1-gate-$$"' in smoke  # unique name
    assert 'WORK="$(mktemp -d' in smoke
    assert "trap cleanup EXIT" in smoke
    assert "docker rm -f \"$CONTAINER_NAME\"" in smoke
    assert "rm -rf \"$WORK\"" in smoke
    # Every probe is wrapped by a hard timeout with escalation.
    assert smoke.count("timeout --signal=INT --kill-after=10") >= 2
    assert "$HOST_EXEC_TIMEOUT" in smoke and "$R2_EXEC_TIMEOUT" in smoke
    assert "$P_SPAN_EXEC_TIMEOUT" in smoke and "$P_HOLD_EXEC_TIMEOUT" in smoke
    # Bounded kill-confirmation wait.
    assert "KILL_WAIT" in smoke and "seq 1 \"$KILL_WAIT\"" in smoke
    # Leftover scans feed the artifact checks.
    assert 'docker ps -a --filter "name=rocobrick-p1-gate-"' in smoke
    assert "pgrep -f 'p1_host_dds_probe|p1_container_dds_probe'" in smoke


def test_smoke_gate_scans_all_logs_for_transport_failures() -> None:
    smoke = _read(SMOKE_P1_PATH)
    assert '"address already in use"' in smoke
    assert '"error binding"' in smoke
    assert '"binding at port"' in smoke
    assert '"shared memory"' in smoke and '"shm"' in smoke
    assert '"udpv4"' in smoke and '"udpv6"' in smoke
    assert '"multicast"' in smoke
    assert '"log_scan_clean"' in smoke
    assert "docker logs" in smoke  # container-level logs are scanned too


def test_smoke_gate_verifies_per_participant_payloads_and_writes_artifact() -> None:
    smoke = _read(SMOKE_P1_PATH)
    assert "rocobrick-p1-dds-multiprocess-gate/v1" in smoke
    assert 'report["passed"] = all(checks.values())' in smoke
    # Per-participant, per-round, both-direction checks in the builder.
    for token in ("received_by_host", "received_by_process",
                  "sent_payload", "expected_host_payload",
                  "round1_all_participants_bidirectional",
                  "round2_all_participants_bidirectional",
                  "round2_restart_new_nonce", "per_participant_payloads_unique",
                  "profiles_contract", "process_exit_codes_clean",
                  "bounded_and_no_leftovers"):
        assert token in smoke, token


def test_probe_tools_are_single_participant_single_context() -> None:
    host_src = _read(HOST_PROBE_PATH)
    container_src = _read(CONTAINER_PROBE_PATH)
    assert host_src.count("rclpy.create_node") == 1
    assert container_src.count("rclpy.create_node") == 1
    # Both probes are bounded by an internal deadline.
    assert "deadline = time.monotonic() + args.timeout" in host_src
    assert "deadline = time.monotonic() + args.timeout" in container_src
    # Payloads are addressed per participant in both directions.
    assert "host->" in host_src and "->host|" in host_src
    assert "->host|" in container_src and "host->" in container_src
    # Container probe records milestones so kills cannot erase round evidence.
    assert 'emit("received"' in container_src
    assert "--hold" in container_src and "--gate-tag" in container_src


# ---------------------------------------------------------------------------
# Live gate artifact.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate() -> dict:
    return json.loads(_read(ARTIFACT_PATH))


def test_gate_artifact_schema_and_pass(gate: dict) -> None:
    assert gate["schema"] == "rocobrick-p1-dds-multiprocess-gate/v1"
    assert gate["phase"] == "P1.A"
    assert gate["topology"] == TOPOLOGY
    assert gate["transport"] == "TCPv4"
    assert gate["passed"] is True
    assert all(gate["checks"].values())
    assert set(gate["checks"]) == {
        "profiles_contract",
        "round1_all_participants_bidirectional",
        "round2_all_participants_bidirectional",
        "round2_restart_new_nonce",
        "per_participant_payloads_unique",
        "process_exit_codes_clean",
        "log_scan_clean",
        "bounded_and_no_leftovers",
    }


def test_gate_artifact_profiles_embed_hashes_and_roles(gate: dict) -> None:
    profiles = gate["profiles"]
    assert profiles["schema"] == "rocobrick-dds-profiles/v2"
    assert profiles["profiles"]["host"]["role"] == "host-listener"
    assert profiles["profiles"]["host"]["listening_ports"] == [gate["host_port"]]
    for name, entry in profiles["profiles"].items():
        if name == "host":
            continue
        assert entry["role"] == "container-client"
        assert entry["listening_ports"] == []
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    pinned = [profiles["profiles"][f"container-p{i}"]["participant_id"]
              for i in range(1, 5)]
    assert sorted(pinned) == [1, 2, 3, 4]


def test_gate_artifact_round_matrix_both_directions(gate: dict) -> None:
    assert len(gate["rounds"]) == 2
    for round_entry, round_number in zip(gate["rounds"], (1, 2)):
        assert round_entry["round"] == round_number
        assert round_entry["passed"] is True
        assert set(round_entry["participants"]) == set(PARTICIPANTS)
        for pid_label, entry in round_entry["participants"].items():
            assert entry["sent_payload"] == f"{pid_label}->host|{entry['nonce']}"
            assert entry["expected_host_payload"] == (
                f"host->{pid_label}|{entry['host_nonce']}"
            )
            assert entry["send_evidence"]["published_by_process"] is True
            assert entry["send_evidence"]["received_by_host"] is True
            assert entry["send_evidence"]["host_receive_count"] > 0
            assert entry["receive_evidence"]["published_by_host"] is True
            assert entry["receive_evidence"]["received_by_process"] is True
            assert entry["passed"] is True


def test_gate_artifact_restart_evidence(gate: dict) -> None:
    round1, round2 = gate["rounds"]
    assert round2["restarted"] == list(RESTARTED)
    for pid_label in RESTARTED:
        old = round1["participants"][pid_label]
        new = round2["participants"][pid_label]
        assert new["role_in_round"] == "recreated_new_nonce"
        assert new["nonce"] != old["nonce"]  # recreated with a NEW nonce
        assert gate["processes"][f"{pid_label}-r1"]["killed_for_restart"] is True
        assert gate["processes"][f"{pid_label}-r1"]["exit_code"] != 0
        assert new["process_exit_code"] == 0
    for pid_label in ("p1", "p4"):
        assert round2["participants"][pid_label]["role_in_round"] == \
            "continuing_across_rounds"
        assert round2["participants"][pid_label]["process_exit_code"] == 0


def test_gate_artifact_log_scan_and_bounds(gate: dict) -> None:
    scan = gate["log_scan"]
    assert scan["passed"] is True
    assert scan["findings"] == []
    assert "address already in use" in scan["patterns"]
    assert "error binding" in scan["patterns"]
    assert "shm" in scan["patterns"] and "multicast" in scan["patterns"]
    assert len(scan["files"]) >= 8  # 2 host + 6 container + docker logs
    bounds = gate["plan"]["bounds_seconds"]
    assert bounds and all(isinstance(v, (int, float)) and v > 0
                          for v in bounds.values())
    assert gate["container_name"].startswith("rocobrick-p1-gate-")
    assert gate["rootless"]["docker_context"] == "rootless"


def test_gate_artifact_payloads_are_unique_per_participant_per_round(
    gate: dict,
) -> None:
    round1, round2 = gate["rounds"]
    # Eight distinct host payloads: one per participant per round.
    host_nonces = [round1["participants"][p]["host_nonce"] for p in PARTICIPANTS]
    host_nonces += [round2["participants"][p]["host_nonce"] for p in PARTICIPANTS]
    assert len(set(host_nonces)) == 8
    # Six distinct participant nonce identities: four initial + two recreated.
    container_nonces = [round1["participants"][p]["nonce"] for p in PARTICIPANTS]
    container_nonces += [round2["participants"][p]["nonce"] for p in RESTARTED]
    assert len(set(container_nonces)) == 6
    # Continuing participants keep their identity (same nonce) across rounds.
    for pid_label in ("p1", "p4"):
        assert round2["participants"][pid_label]["nonce"] == \
            round1["participants"][pid_label]["nonce"]
