#!/usr/bin/env python3
"""Bounded bidirectional host<->container ROS 2 DDS smoke test (rootless Docker).

What it does, end to end:

1. Auto-detects the host IP and (re)generates the Fast DDS profiles with
   ``tools/dds/generate_dds_profiles.py`` (UDP fixed locators by default,
   ``--transport tcp`` as the documented fallback).
2. Starts the ROS Humble container on a named bridge network with the
   deterministic published ports and ``FASTRTPS_DEFAULT_PROFILES_FILE``
   pointed at the generated container profile (same contract as
   ``compose.yaml``; used directly because the compose CLI plugin is not
   installed on this host).
3. Runs bounded ROS message exchanges in BOTH directions using the Isaac Sim
   venv Python (internal rclpy + Fast DDS 2.6.10) on the host and the ROS
   Humble stack in the container. Payloads are prefixed and matched by
   content, so discovery alone is not enough.
4. Restarts the container and repeats both directions (restart test).
5. Writes a truthful JSON evidence report under ``runs/dds-smoke-<ts>/``.

Exit code is 0 only if every round delivered payloads in both directions.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "dds"))

import generate_dds_profiles as generator  # noqa: E402

RUNNER_IN_CONTAINER = "/workspace/tools/dds/node_runner.py"
CONTAINER_PROFILE_PATH = "/workspace/dds/container-fastdds-profile.xml"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--host-ip", default="auto")
    parser.add_argument("--transport", choices=("udp", "tcp"), default="udp")
    parser.add_argument("--container-image", default="roco-piper-ros:humble")
    parser.add_argument("--container-name", default="rocobrick-dds-smoke")
    parser.add_argument("--network", default="rocobrick-ros-net")
    parser.add_argument("--rounds", type=int, default=2, help=">=2 includes the restart test")
    parser.add_argument("--count", type=int, default=5, help="payload messages per direction")
    parser.add_argument("--timeout", type=float, default=45.0, help="per-direction deadline (s)")
    parser.add_argument("--keep", action="store_true", help="keep the smoke container running")
    parser.add_argument(
        "--report-dir", type=Path, default=None, help="report directory (default runs/dds-smoke-<ts>)"
    )
    return parser.parse_args(argv)


def run_cmd(cmd: list[str], timeout: float, env: dict | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env
    )
    return proc.returncode, proc.stdout, proc.stderr


def docker(*args: str, timeout: float = 60.0) -> tuple[int, str, str]:
    return run_cmd(["docker", *args], timeout=timeout)


def free_udp_port(port: int) -> bool:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("0.0.0.0", port))
        probe.close()
        return True
    except OSError:
        return False


def start_container(args: argparse.Namespace, facts: dict) -> None:
    docker("rm", "-f", args.container_name, timeout=30)
    cmd = [
        "docker", "run", "-d", "--name", args.container_name,
        "--network", args.network,
        "--init",
        "-e", f"ROS_DOMAIN_ID={args.domain_id}",
        "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
        "-e", "ROS_LOCALHOST_ONLY=0",
        "-e", f"FASTRTPS_DEFAULT_PROFILES_FILE={CONTAINER_PROFILE_PATH}",
        "-v", f"{generator.DEFAULT_OUTPUT_DIR / 'container-fastdds-profile.xml'}:{CONTAINER_PROFILE_PATH}:ro",
        "-v", f"{REPO_ROOT / 'tools' / 'dds'}:{RUNNER_IN_CONTAINER.rsplit('/', 1)[0]}:ro",
    ]
    for publish in facts.get("compose_publish_udp", []) + facts.get("compose_publish_tcp", []):
        cmd += ["-p", publish]
    cmd += [args.container_image, "sleep", "3600"]
    code, out, err = docker(*cmd[1:], timeout=60)
    if code != 0:
        raise SystemExit(f"error: docker run failed: {err.strip() or out.strip()}")
    # Wait for the entrypoint/self-check environment to be ready.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        code, _, _ = docker(
            "exec", args.container_name, "bash", "-c",
            "source /opt/ros/humble/setup.bash >/dev/null 2>&1 && python3 -c 'import rclpy'",
            timeout=30,
        )
        if code == 0:
            return
        time.sleep(2)
    raise SystemExit("error: container rclpy never became ready")


def host_env(args: argparse.Namespace, facts: dict) -> dict:
    env = dict(os.environ)
    bridge_dir = os.environ.get(
        "ISAAC_BRIDGE_DIR",
        str(REPO_ROOT / ".venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge"),
    )
    lib_dir = f"{bridge_dir}/{facts['ros_distro']}/lib"
    rclpy_dir = f"{bridge_dir}/{facts['ros_distro']}/rclpy"
    env["ROS_DISTRO"] = facts["ros_distro"]
    env["RMW_IMPLEMENTATION"] = facts["rmw_implementation"]
    env["ROS_DOMAIN_ID"] = str(args.domain_id)
    env["ROS_LOCALHOST_ONLY"] = "0"
    env["LD_LIBRARY_PATH"] = lib_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["PYTHONPATH"] = rclpy_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(
        generator.DEFAULT_OUTPUT_DIR / "host-fastdds-profile.xml"
    )
    return env


def run_direction(
    args: argparse.Namespace,
    env: dict,
    round_no: int,
    direction: str,
) -> dict:
    """direction: host_to_container (host talks) or container_to_host."""
    host_prefix = f"host-round{round_no}"
    container_prefix = f"container-round{round_no}"
    report_name = f"{direction}-round{round_no}.json"
    if direction == "host_to_container":
        host_cmd = [
            str(REPO_ROOT / ".venv/bin/python"), str(REPO_ROOT / "tools/dds/node_runner.py"),
            "--mode", "talker", "--publish-prefix", host_prefix,
            "--expect-prefix", container_prefix, "--count", str(args.count),
            "--timeout", str(args.timeout),
            "--report", str(REPO_ROOT / "runs" / "_smoke_tmp_host" / report_name),
        ]
        container_cmd = [
            "docker", "exec", args.container_name, "bash", "-c",
            "source /opt/ros/humble/setup.bash && python3 "
            f"{RUNNER_IN_CONTAINER} --mode listener --publish-prefix {container_prefix} "
            f"--expect-prefix {host_prefix} --count {args.count} --timeout {args.timeout} "
            f"--report /tmp/{report_name}",
        ]
    else:
        host_cmd = [
            str(REPO_ROOT / ".venv/bin/python"), str(REPO_ROOT / "tools/dds/node_runner.py"),
            "--mode", "listener", "--publish-prefix", host_prefix,
            "--expect-prefix", container_prefix, "--count", str(args.count),
            "--timeout", str(args.timeout),
            "--report", str(REPO_ROOT / "runs" / "_smoke_tmp_host" / report_name),
        ]
        container_cmd = [
            "docker", "exec", args.container_name, "bash", "-c",
            "source /opt/ros/humble/setup.bash && python3 "
            f"{RUNNER_IN_CONTAINER} --mode talker --publish-prefix {container_prefix} "
            f"--expect-prefix {host_prefix} --count {args.count} --timeout {args.timeout} "
            f"--report /tmp/{report_name}",
        ]

    host_proc = subprocess.Popen(
        host_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    container_proc = subprocess.Popen(
        container_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        host_out, host_err = host_proc.communicate(timeout=args.timeout + 60)
        container_out, container_err = container_proc.communicate(timeout=args.timeout + 60)
    except subprocess.TimeoutExpired:
        host_proc.kill()
        container_proc.kill()
        host_out, host_err = host_proc.communicate()
        container_out, container_err = container_proc.communicate()
    result = {
        "direction": direction,
        "round": round_no,
        "host_returncode": host_proc.returncode,
        "container_returncode": container_proc.returncode,
        "host_result": _safe_json(host_out),
        "container_result": _safe_json(container_out),
    }
    if host_err.strip():
        result["host_stderr_tail"] = host_err.strip().splitlines()[-5:]
    if container_err.strip():
        result["container_stderr_tail"] = container_err.strip().splitlines()[-5:]
    host_ok = bool(result["host_result"].get("ok")) if result["host_result"] else False
    container_ok = bool(result["container_result"].get("ok")) if result["container_result"] else False
    result["passed"] = host_ok and container_ok
    return result


def _safe_json(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)

    # 1. Profiles (never hardcodes the IP; auto-detects unless --host-ip).
    host_ip = generator.detect_host_ip() if args.host_ip == "auto" else args.host_ip
    code = generator.main(["--host-ip", host_ip, "--domain-id", str(args.domain_id),
                           "--transport", args.transport])
    if code != 0:
        raise SystemExit("error: profile generation failed")
    facts = generator.profile_facts(host_ip, args.domain_id, args.transport)
    facts["ros_distro"] = "humble"
    facts["rmw_implementation"] = "rmw_fastrtps_cpp"

    blocked = [p for p in facts["ports"].values() if not free_udp_port(int(p))] if args.transport == "udp" else []
    # TCP port shares the UDP space check only loosely; UDP-only check is fine here.
    report_dir = args.report_dir or (REPO_ROOT / "runs" / f"dds-smoke-{started.strftime('%Y%m%dT%H%M%SZ')}")

    # 2. Network + container.
    code, _, _ = docker("network", "inspect", args.network, timeout=15)
    if code != 0:
        code, out, err = docker("network", "create", args.network, timeout=30)
        if code != 0:
            raise SystemExit(f"error: docker network create failed: {err.strip() or out.strip()}")
    start_container(args, facts)

    env = host_env(args, facts)
    rounds = []
    for round_no in range(1, args.rounds + 1):
        if round_no > 1:
            # Restart test: full container recreate with identical settings.
            start_container(args, facts)
        round_result = {"round": round_no}
        round_result["host_to_container"] = run_direction(args, env, round_no, "host_to_container")
        round_result["container_to_host"] = run_direction(args, env, round_no, "container_to_host")
        round_result["passed"] = (
            round_result["host_to_container"]["passed"] and round_result["container_to_host"]["passed"]
        )
        rounds.append(round_result)
        print(f"round {round_no}: passed={round_result['passed']}", file=sys.stderr)

    if not args.keep:
        docker("rm", "-f", args.container_name, timeout=30)

    passed = bool(rounds) and all(r["passed"] for r in rounds)
    report = {
        "schema": "rocobrick-dds-smoke/v1",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "host_ip": host_ip,
        "domain_id": args.domain_id,
        "transport": args.transport,
        "container_image": args.container_image,
        "published_ports": facts.get("compose_publish_udp", []) + facts.get("compose_publish_tcp", []),
        "blocked_dds_ports_at_start": sorted(set(blocked)),
        "restart_test": args.rounds > 1,
        "rounds": rounds,
        "profiles": {
            "host": "config/dds/generated/host-fastdds-profile.xml",
            "container": "config/dds/generated/container-fastdds-profile.xml",
            "facts": "config/dds/generated/profile-facts.json",
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
