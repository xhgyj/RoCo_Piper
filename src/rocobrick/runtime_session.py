"""Formal Isaac host-bridge + rootless MoveIt runtime session (R4.2).

Owns the proven live stack as a formal library (it never calls the regression
smoke scripts): the host Isaac bridge (``isaac/p05_host_bridge.py`` through
``tools/isaac-ros-env.sh``) plus ONE unique rootless Docker container running
``move_group`` with ``use_sim_time:=true``, readiness through the container
child entry (:mod:`rocobrick.ros_session_entry`), bounded operations, and an
idempotent orderly stop with lifecycle evidence.

Topology (identical to the live-verified P1.A/B/C + R2 evidence): the host
bridge is the single bundled-rclpy participant owning TCP 17650 via the
host-listener profile; every container participant is a client-only TCPv4
client of that listener through the P1.B additive container-mesh profile, so
no TCP listening port ever exists inside the container namespace.

Every command and process crosses an injected :class:`ProcessRunner`
boundary, so unit tests can fake the whole stack offline. The model audit,
profile generation and container entry are subprocesses of the repository's
own pinned tools (never imported). All waits are bounded by the injected
clock/sleep hooks and the :class:`RuntimeSessionConfig` timeouts; no foreign
process is ever killed and no motion is ever sent during start (the readiness
operation is read-only). No ROS/Isaac module is imported here: the parent
process never touches rclpy.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess  # noqa: S404 - the default runner boundary only
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from rocobrick.compat import StrEnum
from rocobrick.isaac_backend import (
    OperationKind,
    RuntimeOperation,
    RuntimeOperationResult,
    readiness_operation,
)

SESSION_SCHEMA = "rocobrick-runtime-session/v1"

# Pinned proven-stack locations, relative to the repository root.
P1A_ARTIFACT_RELPATH = Path("artifacts/phase1/dds-multiprocess-gate.json")
P1A_SCHEMA = "rocobrick-p1-dds-multiprocess-gate/v1"
HOST_BRIDGE_RELPATH = Path("isaac/p05_host_bridge.py")
HOST_ENV_RELPATH = Path("tools/isaac-ros-env.sh")
MODEL_AUDIT_RELPATH = Path("tools/piper_l_model_audit.py")
PROFILE_TOOL_RELPATH = Path("tools/generate_dds_profiles.py")

# Pinned rootless DDS topology constants (the P1.A frozen contract).
READY_MARKER = "p0.5_host_bridge_ready"
HOST_DDS_PORT = 17650
CONTAINER_MESH_PROFILE = "fastdds-container-mesh.xml"
HOST_PROFILE = "fastdds-host.xml"
CONTAINER_NAME_PREFIX = "rocobrick-runtime-session-"
CONTAINER_SESSION_MOUNT = "/session"
CONTAINER_DDS_MOUNT = "/dds"
ROS_WS_RELPATH = Path("ros_ws")
SRC_RELPATH = Path("src")

# Container-side move_group entry (the container's PID-1 process). The two
# ament packages are built only when the mounted install tree is missing.
CONTAINER_LAUNCH_SCRIPT = """set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
if [ ! -f /workspace/ros_ws/install/setup.bash ]; then
  cd /workspace/ros_ws
  timeout 300 colcon build \\
    --packages-select agx_arm_description rocobrick_piper_l_moveit_config \\
    --symlink-install
  cd /
fi
source /workspace/ros_ws/install/setup.bash
export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"
set -u
exec ros2 launch rocobrick_piper_l_moveit_config \\
  move_group.launch.py use_sim_time:=true
"""
CONTAINER_PYTHONPATH = "/workspace/src"
# Host hold step bound: the bridge exits its hold at max(steps, seconds), so
# the step bound must be reachable WITHIN the hold seconds (the proven P1/R2
# smoke ratio 220000 steps / 420 s pins this machine at ~458-524 steps/s;
# 110000 steps therefore always finishes inside the hold seconds and the
# bridge self-exits orderly with its own evidence).
HOST_HOLD_STEPS = 110000
AUDIT_TIMEOUT_S = 120.0
PROFILE_TIMEOUT_S = 60.0
MARKER_POLL_INTERVAL_S = 0.5
STOP_POLL_INTERVAL_S = 1.0
HOST_INT_GRACE_S = 90.0
DOCKER_STOP_GRACE_S = 30.0
CHILD_EXEC_MARGIN_S = 30.0
LEFTOVER_SCAN_ATTEMPTS = 4
LEFTOVER_SCAN_PAUSE_S = 3.0
REMOVAL_VERIFY_ATTEMPTS = 5
REMOVAL_VERIFY_PAUSE_S = 2.0
REAP_TIMEOUT_S = 5.0
STOP_REAP_GRACE_S = 15.0


class RuntimeSessionStatus(StrEnum):
    """Exact lifecycle status vocabulary of a runtime session."""

    NOT_STARTED = "NOT_STARTED"
    STARTING = "STARTING"
    READY = "READY"
    FAILED = "FAILED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class RuntimeSessionError(RuntimeError):
    """Raised when the runtime session cannot reach or keep readiness."""


def _is_finite_positive(value: object) -> bool:
    """True for a finite strictly positive real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0.0


@dataclass(frozen=True)
class RuntimeSessionConfig:
    """Frozen bounds and identities of one runtime session."""

    repo_root: Path
    ros_image: str = "roco-piper-ros:humble"
    scene_manifest: Path | None = None
    scene_snapshot_out: Path | None = None
    host_instance_id: str | None = None
    ros_domain_id: int = 73
    startup_timeout_s: float = 120.0
    operation_timeout_s: float = 180.0
    shutdown_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        """Validate the repository path and every positive bound."""
        if not isinstance(self.repo_root, Path) or not self.repo_root.is_dir():
            raise ValueError(
                f"repo_root must be an existing directory, got {self.repo_root!r}"
            )
        if not isinstance(self.ros_image, str) or not self.ros_image:
            raise ValueError("ros_image must be a non-empty string")
        scene_fields = (self.scene_manifest, self.scene_snapshot_out, self.host_instance_id)
        if any(value is not None for value in scene_fields) and not all(value is not None for value in scene_fields):
            raise ValueError("scene_manifest, scene_snapshot_out and host_instance_id must be provided together")
        if self.scene_manifest is not None and not isinstance(self.scene_manifest, Path):
            raise ValueError("scene_manifest must be a Path")
        if self.scene_snapshot_out is not None and not isinstance(self.scene_snapshot_out, Path):
            raise ValueError("scene_snapshot_out must be a Path")
        if (
            isinstance(self.ros_domain_id, bool)
            or not isinstance(self.ros_domain_id, int)
            or self.ros_domain_id < 0
        ):
            raise ValueError(
                "ros_domain_id must be a non-negative integer, "
                f"got {self.ros_domain_id!r}"
            )
        for name in (
            "startup_timeout_s",
            "operation_timeout_s",
            "shutdown_timeout_s",
        ):
            if not _is_finite_positive(getattr(self, name)):
                raise ValueError(
                    f"{name} must be finite and positive, got {getattr(self, name)!r}"
                )


@dataclass(frozen=True)
class RuntimeReadinessEvidence:
    """Gate-by-gate readiness of the live stack; all fields must be true."""

    host_marker: bool
    container_started: bool
    moveit_ready: bool
    controllers_active: bool
    adapter_ready: bool

    @property
    def passed(self) -> bool:
        """True only when every readiness gate holds."""
        return all(
            (
                self.host_marker,
                self.container_started,
                self.moveit_ready,
                self.controllers_active,
                self.adapter_ready,
            )
        )


@dataclass(frozen=True)
class RuntimeSessionEvidence:
    """Serializable lifecycle evidence of one runtime session."""

    status: RuntimeSessionStatus
    readiness: RuntimeReadinessEvidence | None
    host_pid: int | None
    container_name: str | None
    profile_report: Mapping[str, object] = field(default_factory=dict)
    host_evidence_path: str | None = None
    log_paths: tuple[str, ...] = ()
    cleanup: Mapping[str, object] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    schema: str = SESSION_SCHEMA

    def serializable(self) -> dict[str, object]:
        """Return the JSON-safe evidence document."""
        return {
            "schema": self.schema,
            "status": str(self.status.value),
            "readiness": asdict(self.readiness) if self.readiness is not None else None,
            "host_pid": self.host_pid,
            "container_name": self.container_name,
            "profile_report": dict(self.profile_report),
            "host_evidence_path": self.host_evidence_path,
            "log_paths": list(self.log_paths),
            "cleanup": dict(self.cleanup),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class RunOutcome:
    """Bounded result of one completed runner command."""

    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


class ProcessHandle(Protocol):
    """A launched long-running process (the host bridge)."""

    pid: int

    def poll(self) -> int | None:
        """Return the exit code once finished, else None."""
        ...

    def wait(self, timeout_s: float | None = None) -> int:
        """Reap the process; return its exit code."""
        ...

    def send_signal(self, signum: int) -> None:
        """Deliver an OS signal to the process."""
        ...


class ProcessRunner(Protocol):
    """Every external command/process of the session crosses this boundary."""

    def launch(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> ProcessHandle:
        """Start a long-running process with redirected output streams."""
        ...

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RunOutcome:
        """Run one bounded command to completion and capture its streams."""
        ...

    def process_alive(self, pid: int) -> bool:
        """True while the process runs (a reaped zombie is dead)."""
        ...

    def processes_matching(self, pattern: str) -> tuple[str, ...]:
        """PIDs (as strings) of live processes whose command line matches."""
        ...


class _PopenHandle:
    """:class:`ProcessHandle` adapter over :class:`subprocess.Popen`.

    ``Popen.wait`` takes ``timeout=`` while the protocol fixes the keyword
    name to ``timeout_s``; the delegation boundary lives here so every
    caller can reap processes through one exact contract.
    """

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self.pid = process.pid

    def poll(self) -> int | None:
        """Return the exit code once finished, else None."""
        return self._process.poll()

    def wait(self, timeout_s: float | None = None) -> int:
        """Reap the process; return its exit code."""
        return self._process.wait(timeout=timeout_s)

    def send_signal(self, signum: int) -> None:
        """Deliver an OS signal to the process."""
        self._process.send_signal(signum)


class SubprocessRunner:
    """Production :class:`ProcessRunner` over :mod:`subprocess`."""

    def launch(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> ProcessHandle:
        """Start the process with its streams bound to files."""
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("ab") as out, stderr_path.open("ab") as err:
            return _PopenHandle(
                subprocess.Popen(  # noqa: S603 - argv list, no shell
                    list(argv),
                    env=dict(env),
                    stdout=out,
                    stderr=err,
                )
            )

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RunOutcome:
        """Run one bounded command; a timeout is recorded, never raised."""
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, no shell
                list(argv),
                env=dict(env) if env is not None else None,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return RunOutcome(
                argv=tuple(argv),
                returncode=None,
                stdout=error.stdout if isinstance(error.stdout, str) else "",
                stderr=error.stderr if isinstance(error.stderr, str) else "",
                timed_out=True,
            )
        return RunOutcome(
            argv=tuple(argv),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def process_alive(self, pid: int) -> bool:
        """True while the PID runs and is not a zombie."""
        try:
            state = subprocess.run(  # noqa: S603 - argv list, no shell
                ["ps", "-o", "stat=", "-p", str(pid)],  # noqa: S607 - pinned tool
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return False
        return bool(state) and not state.startswith("Z")

    def processes_matching(self, pattern: str) -> tuple[str, ...]:
        """Bounded pgrep over full command lines."""
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, no shell
                ["pgrep", "-f", pattern],  # noqa: S607 - pinned tool
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ()
        return tuple(line for line in completed.stdout.splitlines() if line.strip())


def _host_time_limit_s(config: RuntimeSessionConfig) -> float:
    """Host bridge ``--time-limit``: the whole session plus an orderly margin."""
    return config.startup_timeout_s + config.operation_timeout_s + 120.0


def _host_hold_seconds(config: RuntimeSessionConfig) -> float:
    """Host hold: ends inside the stop grace so the bridge self-exits.

    The bridge writes its evidence only when it finishes its hold and exits
    on its own (Kit intercepts an external SIGINT at the C level), so the
    hold must end BEFORE ``stop()`` stops waiting
    (``shutdown_timeout_s``), with a further margin for the slow Kit
    teardown.
    """
    return max(
        min(
            config.startup_timeout_s + config.operation_timeout_s + 60.0,
            config.shutdown_timeout_s - 60.0,
        ),
        30.0,
    )


class IsaacMoveItSession:
    """Orchestrates the proven host bridge + rootless MoveIt stack.

    The session is a context manager: ``__enter__`` starts the stack and
    requires full readiness, ``__exit__`` always stops it (idempotently) and
    records cleanup evidence even after a partial start. Commands and
    processes are issued only through the injected :class:`ProcessRunner`;
    waits are paced by the injected clock/sleep hooks.
    """

    def __init__(
        self,
        config: RuntimeSessionConfig,
        *,
        runner: ProcessRunner,
        session_dir: Path | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        base_env: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.session_dir = (
            Path(session_dir)
            if session_dir is not None
            else config.repo_root
            / "runs"
            / (
                f"runtime-session-"
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
            )
        )
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        self._base_env: dict[str, str] = (
            dict(base_env) if base_env is not None else dict(os.environ)
        )
        self.status = RuntimeSessionStatus.NOT_STARTED
        self.errors: list[str] = []
        self.readiness: RuntimeReadinessEvidence | None = None
        self.host_handle: ProcessHandle | None = None
        self.host_pid: int | None = None
        self.container_name: str | None = None
        self.profile_report: dict[str, object] = {}
        self.last_result: RuntimeOperationResult | None = None
        self._stop_evidence: RuntimeSessionEvidence | None = None
        self._dds_dir = self.session_dir / "dds"
        self._host_log = self.session_dir / "host-bridge.log"

    # ------------------------------------------------------------ evidence

    def _log_paths(self) -> tuple[str, ...]:
        return (
            str(self._host_log),
            str(self._dds_dir / CONTAINER_MESH_PROFILE),
            str(self.session_dir / "readiness-result.json"),
            str(self.session_dir / "runtime-session.json"),
        )

    def _host_evidence_path(self) -> str | None:
        path = self.session_dir / "host-bridge-evidence.json"
        return str(path) if path.is_file() else None

    def _snapshot_evidence(self) -> RuntimeSessionEvidence:
        return RuntimeSessionEvidence(
            status=self.status,
            readiness=self.readiness,
            host_pid=self.host_pid,
            container_name=self.container_name,
            profile_report=dict(self.profile_report),
            host_evidence_path=self._host_evidence_path(),
            log_paths=self._log_paths(),
            cleanup={},
            errors=tuple(self.errors),
        )

    def _write_evidence(self) -> None:
        """Write the lifecycle evidence document atomically."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        document = (
            self._stop_evidence.serializable()
            if self._stop_evidence is not None
            else self._snapshot_evidence().serializable()
        )
        target = self.session_dir / "runtime-session.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(target)

    def _record_error(self, message: str) -> None:
        self.errors.append(message)

    def _fail(self, phase: str, detail: str) -> RuntimeSessionError:
        message = f"{phase} failed: {detail}"
        self._record_error(message)
        return RuntimeSessionError(message)

    @staticmethod
    def _write_json(path: Path, document: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    # --------------------------------------------------------------- start

    def start(self) -> RuntimeReadinessEvidence:
        """Start the full stack and require every readiness gate.

        Raises:
            RuntimeSessionError: When any start phase fails; the accumulated
                evidence keeps the failure details and ``stop()`` still
                performs and records cleanup.
        """
        if self.status is not RuntimeSessionStatus.NOT_STARTED:
            raise RuntimeSessionError(
                f"start is only valid from NOT_STARTED, not {self.status}"
            )
        self.status = RuntimeSessionStatus.STARTING
        self.session_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._verify_p1a_prerequisite()
            self._run_model_audit()
            self._generate_profiles()
            self._launch_host()
            self._wait_for_marker()
            self._launch_container()
            evidence = self._run_readiness_operation()
        except RuntimeSessionError as error:
            self.status = RuntimeSessionStatus.FAILED
            self._record_error(str(error))
            self._write_evidence()
            raise
        self.readiness = evidence
        self.status = RuntimeSessionStatus.READY
        self._write_evidence()
        return evidence

    def _verify_p1a_prerequisite(self) -> None:
        """Require the P1.A multiprocess DDS gate artifact to have passed."""
        path = self.config.repo_root / P1A_ARTIFACT_RELPATH
        if not path.is_file():
            raise self._fail("p1a_prerequisite", f"artifact missing: {path}")
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise self._fail("p1a_prerequisite", f"artifact unreadable: {error}")
        checks = artifact.get("checks") if isinstance(artifact, dict) else None
        if (
            not isinstance(artifact, dict)
            or artifact.get("schema") != P1A_SCHEMA
            or artifact.get("passed") is not True
            or not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
        ):
            raise self._fail(
                "p1a_prerequisite",
                f"artifact did not pass every check: {path}",
            )

    def _run_model_audit(self) -> None:
        """Offline model audit before any GPU/Docker time."""
        outcome = self.runner.run(
            [
                sys.executable,
                str(self.config.repo_root / MODEL_AUDIT_RELPATH),
                "--output",
                str(self.session_dir / "preflight-model-audit.json"),
            ],
            timeout_s=min(AUDIT_TIMEOUT_S, self.config.startup_timeout_s),
        )
        self._write_json(
            self.session_dir / "preflight-model-audit.log",
            {"stdout": outcome.stdout, "stderr": outcome.stderr},
        )
        if outcome.timed_out:
            raise self._fail(
                "model_audit", f"timed out: {shlex.join(outcome.argv)}"
            )
        if outcome.returncode != 0:
            raise self._fail(
                "model_audit",
                f"exit {outcome.returncode}: {outcome.stderr.strip()[-400:]}",
            )

    def _generate_profiles(self) -> None:
        """Generate the host-listener + container-mesh DDS profiles."""
        self._dds_dir.mkdir(parents=True, exist_ok=True)
        outcome = self.runner.run(
            [
                sys.executable,
                str(self.config.repo_root / PROFILE_TOOL_RELPATH),
                "--output-dir",
                str(self._dds_dir),
                "--host-port",
                str(HOST_DDS_PORT),
                "--container-mesh",
                "--ros-domain-id",
                str(self.config.ros_domain_id),
            ],
            timeout_s=PROFILE_TIMEOUT_S,
        )
        if outcome.timed_out or outcome.returncode != 0:
            raise self._fail(
                "dds_profiles",
                f"exit={outcome.returncode} timed_out={outcome.timed_out}: "
                f"{outcome.stderr.strip()[-400:]}",
            )
        try:
            report = json.loads(outcome.stdout)
        except ValueError as error:
            raise self._fail("dds_profiles", f"profile report unparsable: {error}")
        if not isinstance(report, dict):
            raise self._fail("dds_profiles", "profile report is not an object")
        self.profile_report = report
        if not (self._dds_dir / CONTAINER_MESH_PROFILE).is_file():
            raise self._fail(
                "dds_profiles",
                f"the mesh profile {CONTAINER_MESH_PROFILE} was not generated",
            )

    def _host_argv(self) -> list[str]:
        repo = self.config.repo_root
        argv = [
            "bash",
            str(repo / HOST_ENV_RELPATH),
            sys.executable,
            str(repo / HOST_BRIDGE_RELPATH),
            "--max-steps",
            "2000000",
            "--time-limit",
            str(_host_time_limit_s(self.config)),
            "--ready-timeout",
            str(min(120.0, self.config.startup_timeout_s)),
            "--hold-steps",
            str(HOST_HOLD_STEPS),
            "--hold-seconds",
            str(_host_hold_seconds(self.config)),
            "--evidence-out",
            str(self.session_dir / "host-bridge-evidence.json"),
        ]
        if self.config.scene_manifest is not None:
            argv.extend((
                "--scene-manifest", str(self.config.scene_manifest),
                "--scene-snapshot-out", str(self.config.scene_snapshot_out),
                "--host-instance-id", self.config.host_instance_id,
            ))
        return argv

    def _host_env(self) -> dict[str, str]:
        env = dict(self._base_env)
        env["ROS_DOMAIN_ID"] = str(self.config.ros_domain_id)
        env["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(self._dds_dir / HOST_PROFILE)
        return env

    def _launch_host(self) -> None:
        """Launch the host Isaac bridge through isaac-ros-env.sh (bounded)."""
        bridge = self.config.repo_root / HOST_BRIDGE_RELPATH
        if not bridge.is_file():
            raise self._fail("host_launch", f"host bridge missing: {bridge}")
        handle = self.runner.launch(
            self._host_argv(),
            env=self._host_env(),
            stdout_path=self._host_log,
            stderr_path=self._host_log,
        )
        self.host_handle = handle
        self.host_pid = handle.pid

    def _read_host_log(self) -> str:
        try:
            return self._host_log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _wait_for_marker(self) -> None:
        """Bounded wait for the exact host readiness marker line."""
        needle = f'"marker": "{READY_MARKER}"'
        deadline = self._clock() + self.config.startup_timeout_s
        while True:
            if needle in self._read_host_log():
                return
            handle = self.host_handle
            if handle is None or not self.runner.process_alive(handle.pid):
                raise self._fail(
                    "host_marker", f"the host bridge exited before {READY_MARKER}"
                )
            if self._clock() >= deadline:
                raise self._fail(
                    "host_marker",
                    f"the {READY_MARKER} marker was not seen within "
                    f"{self.config.startup_timeout_s}s",
                )
            self._sleep(MARKER_POLL_INTERVAL_S)

    def _container_argv(self, name: str) -> list[str]:
        repo = self.config.repo_root
        return [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            name,
            "-e",
            f"ROS_DOMAIN_ID={self.config.ros_domain_id}",
            "-e",
            "ROS_LOCALHOST_ONLY=0",
            "-e",
            "RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
            "-e",
            (
                f"FASTRTPS_DEFAULT_PROFILES_FILE="
                f"{CONTAINER_DDS_MOUNT}/{CONTAINER_MESH_PROFILE}"
            ),
            "-v",
            f"{repo / ROS_WS_RELPATH}:/workspace/ros_ws",
            "-v",
            f"{repo / SRC_RELPATH}:/workspace/src:ro",
            "-v",
            f"{self._dds_dir}:{CONTAINER_DDS_MOUNT}:ro",
            "-v",
            f"{self.session_dir}:{CONTAINER_SESSION_MOUNT}",
            self.config.ros_image,
            "bash",
            "-c",
            CONTAINER_LAUNCH_SCRIPT,
        ]

    def _launch_container(self) -> None:
        """Start ONE unique rootless container whose PID-1 is move_group."""
        name = f"{CONTAINER_NAME_PREFIX}{uuid4().hex[:12]}"
        outcome = self.runner.run(
            self._container_argv(name),
            timeout_s=self.config.startup_timeout_s,
        )
        if outcome.timed_out or outcome.returncode != 0:
            raise self._fail(
                "container_start",
                f"docker run exit={outcome.returncode} "
                f"timed_out={outcome.timed_out}: {outcome.stderr.strip()[-400:]}",
            )
        self.container_name = name

    def _container_path(self, path: Path) -> str:
        """Container-side path for a file inside the session directory."""
        try:
            relative = path.resolve().relative_to(self.session_dir.resolve())
        except ValueError as error:
            raise self._fail(
                "operation_paths",
                f"{path} must live inside the session directory {self.session_dir}",
            ) from error
        return f"{CONTAINER_SESSION_MOUNT}/{relative.as_posix()}"

    def _child_inner_script(self, request_path: Path, result_path: Path) -> list[str]:
        """Docker exec argv for the child entry.

        ``docker exec`` bypasses the container entrypoint, so the ROS and
        workspace environments are sourced here; the two paths are passed
        positionally to avoid any quoting ambiguity.
        """
        inner = (
            "set +u; source /opt/ros/humble/setup.bash; "
            "if [ -f /workspace/ros_ws/install/setup.bash ]; then "
            "source /workspace/ros_ws/install/setup.bash; fi; "
            f'set -u; export PYTHONPATH="{CONTAINER_PYTHONPATH}:${{PYTHONPATH:-}}"; '
            'exec python3 -m rocobrick.ros_session_entry '
            '--request "$1" --result "$2"'
        )
        return [
            "docker",
            "exec",
            "-e",
            f"ROS_DOMAIN_ID={self.config.ros_domain_id}",
            "-e",
            "ROS_LOCALHOST_ONLY=0",
            "-e",
            "RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
            "-e",
            (
                f"FASTRTPS_DEFAULT_PROFILES_FILE="
                f"{CONTAINER_DDS_MOUNT}/{CONTAINER_MESH_PROFILE}"
            ),
            self.container_name,
            "bash",
            "-c",
            inner,
            "bash",
            self._container_path(request_path),
            self._container_path(result_path),
        ]

    def _run_child(
        self, request_path: Path, result_path: Path, *, timeout_s: float
    ) -> RuntimeOperationResult:
        """Run the container child entry for one operation request (bounded)."""
        if self.container_name is None:
            raise self._fail("child_exec", "no container is running")
        exec_argv = self._child_inner_script(request_path, result_path)
        outcome = self.runner.run(exec_argv, timeout_s=timeout_s)
        if outcome.timed_out:
            raise self._fail(
                "child_exec", f"the child entry exceeded its {timeout_s}s bound"
            )
        if not result_path.is_file():
            raise self._fail(
                "child_exec",
                f"the child entry wrote no result (exit={outcome.returncode}): "
                f"{outcome.stderr.strip()[-400:]}",
            )
        try:
            document = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise self._fail("child_exec", f"result unreadable: {error}")
        try:
            return RuntimeOperationResult.parse(document)
        except ValueError as error:
            raise self._fail("child_exec", f"result schema invalid: {error}")

    def _run_readiness_operation(self) -> RuntimeReadinessEvidence:
        """Zero-motion readiness through the child entry; every gate required."""
        request = readiness_operation(
            timeout_s=self.config.operation_timeout_s,
            request_id=f"readiness-{uuid4().hex[:12]}",
        )
        request_path = self.session_dir / "readiness-request.json"
        result_path = self.session_dir / "readiness-result.json"
        self._write_request(request, request_path)
        result = self._run_child(
            request_path,
            result_path,
            timeout_s=self.config.operation_timeout_s + CHILD_EXEC_MARGIN_S,
        )
        self.last_result = result
        if result.request_id != request.request_id or result.kind is not request.kind:
            raise self._fail(
                "readiness", "the child result does not answer the readiness request"
            )
        snapshot = result.evidence.get("snapshot")
        if not isinstance(snapshot, Mapping):
            snapshot = {}
        if result.evidence.get("motion_sent") is not False:
            raise self._fail(
                "readiness",
                "the readiness operation reported motion; start must not move",
            )
        evidence = RuntimeReadinessEvidence(
            host_marker=True,
            container_started=self.container_name is not None,
            moveit_ready=bool(
                snapshot.get("arm_action_ready")
                and snapshot.get("move_group_action_ready")
                and snapshot.get("compute_fk_ready")
            ),
            controllers_active=bool(snapshot.get("controllers_active")),
            adapter_ready=bool(
                result.evidence.get("waited_ok") and result.evidence.get("passed")
            ),
        )
        if not evidence.passed:
            raise self._fail(
                "readiness", f"not every readiness gate passed: {asdict(evidence)}"
            )
        return evidence

    # -------------------------------------------------------- run_operation

    def run_operation(
        self, request_path: Path, result_path: Path
    ) -> Mapping[str, object]:
        """Execute one bounded operation through the running container.

        The request is validated strictly, (re-)written into the session
        directory, dispatched to the child entry via ``docker exec``, and the
        result document is validated against the request (schema, request id,
        kind) before it is returned. A failing operation (``passed=False``)
        is a returned result, not an exception; transport/schema failures
        raise and mark the session FAILED.

        Raises:
            RuntimeSessionError: When the session is not READY, the request
                violates its contract, or the child transport fails.
        """
        if self.status is not RuntimeSessionStatus.READY:
            raise RuntimeSessionError(
                f"run_operation requires READY, not {self.status}"
            )
        try:
            document = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise self._fail("operation", f"request unreadable: {error}")
        try:
            request = RuntimeOperation.parse(document)
        except ValueError as error:
            raise self._fail("operation", f"request schema invalid: {error}")
        if request.kind is OperationKind.READINESS:
            raise self._fail(
                "operation", "READINESS is an internal start-phase operation"
            )
        if not result_path.resolve().is_relative_to(self.session_dir.resolve()):
            raise self._fail(
                "operation_paths",
                f"result path {result_path} must live inside the session "
                f"directory {self.session_dir}",
            )
        if request_path.resolve().is_relative_to(self.session_dir.resolve()):
            effective_request = request_path
        else:
            effective_request = self.session_dir / (
                f"operation-{request.request_id}-request.json"
            )
        self._write_request(request, effective_request)
        try:
            result = self._run_child(
                effective_request,
                result_path,
                timeout_s=request.timeout_s + CHILD_EXEC_MARGIN_S,
            )
        except RuntimeSessionError:
            self.status = RuntimeSessionStatus.FAILED
            self._write_evidence()
            raise
        self.last_result = result
        if result.request_id != request.request_id or result.kind is not request.kind:
            self.status = RuntimeSessionStatus.FAILED
            self._write_evidence()
            raise self._fail(
                "operation",
                "the child result does not answer the request "
                f"(request_id={request.request_id}, kind={request.kind})",
            )
        if not result.passed:
            self.status = RuntimeSessionStatus.FAILED
        self._write_evidence()
        return result.serialize()

    @staticmethod
    def _write_request(request: RuntimeOperation, path: Path) -> None:
        IsaacMoveItSession._write_json(path, request.serialize())

    # ---------------------------------------------------------------- stop

    def stop(self) -> RuntimeSessionEvidence:
        """Orderly, idempotent teardown with cleanup evidence.

        Asks the container launch to shut down (bounded), stops the host
        bridge (wait for its bounded self-exit, then INT, then KILL), gathers
        the host evidence, and scans for leftovers. The cleanup verdict
        requires the reaped host exit, the bridge's own evidence (``passed``
        plus its orderly cleanup flags), a verified container removal, and no
        leftovers; empty leftovers alone never count as a passing stop. Safe
        to call from any state, including a partial start; a second call
        returns the same evidence.
        """
        if self._stop_evidence is not None:
            return self._stop_evidence
        previous = self.status
        self.status = RuntimeSessionStatus.STOPPING
        cleanup: dict[str, object] = {
            "container_stop": None,
            "container_removed": False,
            "host_stopped": False,
            "host_exit_code": None,
            "host_orderly_self_exit": False,
            "host_process": (
                f"pid {self.host_pid} (isaac-ros-env.sh exec's the bridge, so "
                "the wrapper PID is the actual p05_host_bridge process)"
                if self.host_pid is not None
                else None
            ),
            "host_evidence": {},
            "leftovers": {},
            "passed": False,
        }
        cleanup["container_stop"] = self._stop_container(cleanup)
        self._stop_host(cleanup)
        host_evidence = self._host_evidence_summary()
        cleanup["host_evidence"] = host_evidence
        cleanup["leftovers"] = self._scan_leftovers()
        leftovers = cleanup["leftovers"]
        host_evidence_orderly = bool(
            self.host_pid is None
            or (
                host_evidence.get("passed")
                and host_evidence.get("rclpy_shutdown")
                and host_evidence.get("world_stopped")
                and host_evidence.get("timeline_stopped")
                and host_evidence.get("app_close_initiated")
            )
        )
        cleanup["host_evidence_orderly"] = host_evidence_orderly
        if self.host_pid is not None and not host_evidence_orderly:
            self._record_error(
                "the host bridge evidence does not record an orderly stop"
            )
        cleanup["passed"] = bool(
            cleanup["container_removed"]
            and cleanup["host_stopped"]
            and host_evidence_orderly
            and not leftovers.get("containers")
            and not leftovers.get("host_processes")
        )
        self.status = (
            RuntimeSessionStatus.STOPPED
            if previous is RuntimeSessionStatus.READY
            else RuntimeSessionStatus.FAILED
        )
        self._stop_evidence = RuntimeSessionEvidence(
            status=self.status,
            readiness=self.readiness,
            host_pid=self.host_pid,
            container_name=self.container_name,
            profile_report=dict(self.profile_report),
            host_evidence_path=self._host_evidence_path(),
            log_paths=self._log_paths(),
            cleanup=cleanup,
            errors=tuple(self.errors),
        )
        self._write_evidence()
        return self._stop_evidence

    def _stop_container(self, cleanup: dict[str, object]) -> str:
        """Bounded container shutdown; a removal is verified, never assumed."""
        name = self.container_name
        if name is None:
            cleanup["container_removed"] = True
            return "not_started"
        grace = max(1, int(min(DOCKER_STOP_GRACE_S, self.config.shutdown_timeout_s)))
        outcome = self.runner.run(
            ["docker", "stop", "-t", str(grace), name],
            timeout_s=self.config.shutdown_timeout_s,
        )
        if outcome.timed_out or outcome.returncode != 0:
            self.runner.run(
                ["docker", "rm", "-f", name],
                timeout_s=self.config.shutdown_timeout_s,
            )
        listed = None
        removed = False
        for _ in range(REMOVAL_VERIFY_ATTEMPTS):
            # A freshly stopped --rm container can still be listed for a few
            # seconds ("Removal In Progress"); only a persistent entry fails.
            listed = self.runner.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"name={name}",
                    "--format",
                    "{{.Names}}",
                ],
                timeout_s=30.0,
            )
            removed = not listed.stdout.strip() and not listed.timed_out
            if removed:
                break
            self._sleep(REMOVAL_VERIFY_PAUSE_S)
        cleanup["container_removed"] = removed
        if not removed:
            self._record_error(f"container {name} could not be removed")
        return f"exit={outcome.returncode} timed_out={outcome.timed_out}"

    def _stop_host(self, cleanup: dict[str, object]) -> None:
        """Wait for the host's bounded self-exit; escalate INT then KILL.

        Identity: ``tools/isaac-ros-env.sh`` ``exec``s into the bridge python,
        so ``host_pid`` is the actual ``p05_host_bridge`` process, not a
        wrapper. Every verdict is driven by the reaped exit code of that PID
        plus the bridge's own evidence file; ``ps`` liveness answers only
        pace the wait (they go stale under the heavy Kit native teardown),
        and empty leftovers never substitute for a real stop verdict.
        """
        handle = self.host_handle
        if handle is None:
            cleanup["host_stopped"] = True
            return
        deadline = self._clock() + self.config.shutdown_timeout_s
        while self.runner.process_alive(handle.pid):
            if self._clock() >= deadline:
                break
            self._sleep(STOP_POLL_INTERVAL_S)
        exit_code = self._await_exit(handle, STOP_REAP_GRACE_S)
        if exit_code is None and self._host_finished_orderly():
            # The bridge completed its hold and wrote its passed evidence; it
            # is only inside the slow native Kit teardown. Give it the whole
            # stop budget instead of killing it mid-teardown (LIVE evidence:
            # the teardown can take over a minute after "App Shutting Down").
            exit_code = self._await_exit(handle, self.config.shutdown_timeout_s)
        if exit_code is not None:
            cleanup["host_stopped"] = True
            cleanup["host_orderly_self_exit"] = True
            cleanup["host_exit_code"] = exit_code
            return
        handle.send_signal(2)  # SIGINT
        exit_code = self._await_exit(handle, HOST_INT_GRACE_S)
        if exit_code is None:
            handle.send_signal(9)  # SIGKILL
            exit_code = self._await_exit(handle, HOST_INT_GRACE_S)
        cleanup["host_stopped"] = exit_code is not None
        cleanup["host_exit_code"] = exit_code
        if exit_code is None:
            self._record_error(
                f"host bridge pid {handle.pid} did not stop before its bound"
            )

    def _await_exit(self, handle: ProcessHandle, grace_s: float) -> int | None:
        """Bounded reaped-exit wait; None when the process would not exit."""
        deadline = self._clock() + max(grace_s, STOP_POLL_INTERVAL_S)
        while True:
            try:
                return handle.wait(timeout_s=REAP_TIMEOUT_S)
            except Exception:
                pass
            if self._clock() >= deadline:
                return None
            self._sleep(STOP_POLL_INTERVAL_S)

    def _host_finished_orderly(self) -> bool:
        """True when the bridge wrote evidence of a completed, error-free hold.

        A finished bridge is only doing its slow native Kit teardown at that
        point; it no longer needs any signal, just time to exit.
        """
        summary = self._host_evidence_summary()
        return bool(summary.get("hold_completed")) and (
            summary.get("error") is None
        )

    def _host_evidence_summary(self) -> dict[str, object]:
        """Read the bridge's own evidence document into a bounded summary.

        The summary records the bridge's ``passed`` verdict and its orderly
        cleanup flags (``rclpy_shutdown``, ``world_stopped``,
        ``timeline_stopped``, ``app_close_initiated``/``app_closed``); an
        unreadable or missing file yields ``{}`` so the caller can fail
        honestly instead of assuming an orderly stop.
        """
        path = self.session_dir / "host-bridge-evidence.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(document, dict):
            return {}
        host_cleanup = document.get("cleanup")
        if not isinstance(host_cleanup, dict):
            host_cleanup = {}
        return {
            "path": str(path),
            "passed": document.get("passed") is True,
            "hold_completed": document.get("hold_completed") is True,
            "error": document.get("error"),
            "rclpy_shutdown": host_cleanup.get("rclpy_shutdown") is True,
            "world_stopped": host_cleanup.get("world_stopped") is True,
            "timeline_stopped": host_cleanup.get("timeline_stopped") is True,
            "app_close_initiated": bool(
                host_cleanup.get("app_close_initiated")
                or host_cleanup.get("app_closed")
            ),
        }

    def _scan_leftovers(self) -> dict[str, object]:
        """Bounded leftover scan with a small retry for in-flight teardown."""
        containers: tuple[str, ...] = ()
        hosts: tuple[str, ...] = ()
        for _ in range(LEFTOVER_SCAN_ATTEMPTS):
            listed = self.runner.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"name={CONTAINER_NAME_PREFIX}",
                    "--format",
                    "{{.Names}}",
                ],
                timeout_s=30.0,
            )
            containers = tuple(
                line.strip() for line in listed.stdout.splitlines() if line.strip()
            )
            hosts = self.runner.processes_matching("p05_host_bridge")
            if not containers and not hosts:
                break
            self._sleep(LEFTOVER_SCAN_PAUSE_S)
        return {"containers": list(containers), "host_processes": list(hosts)}

    # ------------------------------------------------------ context manager

    def __enter__(self) -> "IsaacMoveItSession":
        """Start the stack and require full readiness."""
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Always stop (idempotently); never suppress the caller's error."""
        self.stop()
        return False
