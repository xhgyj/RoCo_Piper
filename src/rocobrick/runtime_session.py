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

Multi-robot (A6): :class:`MultiRobotIsaacMoveItSession` keeps the very same
topology - ONE host bridge, ONE rootless container, no per-robot process or
ROS domain - but drives the A3 dual launch inside that container (one
namespaced stack per plan robot) and the A5 multi-robot session envelope
(``python3 -m rocobrick.ros_session_entry --robots ...``). Readiness is proved
for EVERY plan robot, including a robot with no operation, and the only
operations it dispatches are robot-targeted read-only scene observations;
motion and planning-scene changes are refused before any command is issued.
Its cleanup stops only the container and the host process that session
started.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shlex
import signal
import subprocess  # noqa: S404 - the default runner boundary only
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from rocobrick.compat import StrEnum
from rocobrick.isaac_backend import (
    OperationContractError,
    OperationKind,
    RuntimeOperation,
    RuntimeOperationResult,
    readiness_operation,
    scene_query_operation,
)
from rocobrick.moveit_contract import GRIPPER_JOINT_NAMES
from rocobrick.ros_session_entry import (
    SCENE_STATE_SCHEMA,
    RobotOperationRequest,
    RosSessionRequest,
    RosSessionResult,
)
from rocobrick.stap_contract import parse_stap_plan

SESSION_SCHEMA = "rocobrick-runtime-session/v1"


def _with_headed_display(env: Mapping[str, str]) -> dict[str, str]:
    """Bind headed runs to the active local X11 session when SSH omits it."""
    result = dict(env)
    if result.get("ROCOBRICK_HEADED") != "1" or result.get("DISPLAY"):
        return result
    sockets = sorted(Path("/tmp/.X11-unix").glob("X*"))
    if not sockets:
        return result
    socket = sockets[-1]
    result["DISPLAY"] = f":{socket.name[1:]}"
    if not result.get("XAUTHORITY"):
        candidates = (
            Path(f"/run/user/{os.getuid()}/gdm/Xauthority"),
            Path(f"/run/user/{os.getuid()}/Xauthority"),
        )
        for candidate in candidates:
            if candidate.is_file():
                result["XAUTHORITY"] = str(candidate)
                break
    return result

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
CANCEL_TIMEOUT_S = 10.0
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
    plan_path: Path | None = None
    config_path: Path | None = None
    collision_filter_robot_id: str | None = None
    collision_filter_instance_ids: tuple[str, ...] = ()
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
        scene_fields = (
            self.scene_manifest,
            self.scene_snapshot_out,
            self.host_instance_id,
            self.plan_path,
            self.config_path,
        )
        if any(value is not None for value in scene_fields) and not all(value is not None for value in scene_fields):
            raise ValueError(
                "scene_manifest, scene_snapshot_out, host_instance_id, plan_path "
                "and config_path must be provided together"
            )
        if self.scene_manifest is not None and not isinstance(self.scene_manifest, Path):
            raise ValueError("scene_manifest must be a Path")
        if self.scene_snapshot_out is not None and not isinstance(self.scene_snapshot_out, Path):
            raise ValueError("scene_snapshot_out must be a Path")
        if self.plan_path is not None and not isinstance(self.plan_path, Path):
            raise ValueError("plan_path must be a Path")
        if self.config_path is not None and not isinstance(self.config_path, Path):
            raise ValueError("config_path must be a Path")
        if self.collision_filter_robot_id is not None and (
            type(self.collision_filter_robot_id) is not str
            or not self.collision_filter_robot_id
        ):
            raise ValueError("collision_filter_robot_id must be a non-empty string")
        if len(set(self.collision_filter_instance_ids)) != len(self.collision_filter_instance_ids):
            raise ValueError("collision_filter_instance_ids must be unique")
        if any(type(item) is not str or not item for item in self.collision_filter_instance_ids):
            raise ValueError("collision_filter_instance_ids must be non-empty strings")
        if self.collision_filter_instance_ids and self.collision_filter_robot_id is None:
            raise ValueError("collision_filter_instance_ids require collision_filter_robot_id")
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
            _with_headed_display(
                dict(base_env) if base_env is not None else dict(os.environ)
            )
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
                "--plan", str(self.config.plan_path),
                "--config", str(self.config.config_path),
            ))
        if self.config.collision_filter_robot_id is not None:
            argv.extend((
                "--physics-filter-robot-id",
                self.config.collision_filter_robot_id,
            ))
            for instance_id in self.config.collision_filter_instance_ids:
                argv.extend(("--physics-filter-instance-id", instance_id))
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


# ===========================================================================
# Multi-robot live session (A6): ONE host bridge + ONE dual-stack container
# ===========================================================================
#
# Same proven topology as the single-robot session above (host-bridge listener
# on TCP 17650, ONE rootless container whose participants are client-only
# members of the container mesh), with the A3 dual launch inside the one
# container: one namespaced ``robot_state_publisher`` + ``controller_manager``
# + ``move_group`` stack per plan robot on the one ROS domain, sharing the one
# ``/clock`` and the one global ``/p05_host_bridge/scene_state`` service of the
# A4 host bridge. The session NEVER starts a per-robot process or ROS domain:
# every robot lives in the single container session.
#
# Readiness is the A5 session envelope (:mod:`rocobrick.ros_session_entry`
# ``--robots``): the strict ``rocobrick-ros-session-request/v1`` document with
# an EMPTY ``operations`` array answers ``readiness_by_robot`` for EVERY plan
# robot - including robots that carry no operation at all - and the session
# requires every one of them to pass. Observations reuse the same envelope
# with one robot-targeted read-only SCENE_QUERY operation: the A5 child slices
# the global service answer to the operation's own robot, and the session
# returns the A4 document shape (shared ``sim_time_s``/``bricks``/
# ``connections`` plus the ``robots[<robot_id>]`` block) with exactly the
# values that robot's own answer carried. There is no fallback to another
# robot's block and no designated-robot single snapshot.
#
# This stage dispatches NO motion: ``run_read_only_operation`` accepts only
# SCENE_QUERY/READINESS requests and refuses every motion kind before any
# command is issued (zero processes, zero commands). Nothing is faked: a
# failed gate fails ``start()``, and cleanup stops only the container and the
# host process THIS session started.

MULTI_ROBOT_SESSION_SCHEMA = "rocobrick-multi-robot-live-session/v1"
DUAL_LAUNCH_PACKAGE = "rocobrick_piper_l_moveit_config"
DUAL_LAUNCH_FILE = "dual_move_group.launch.py"
DUAL_LAUNCH_RELPATH = Path(
    f"ros_ws/src/{DUAL_LAUNCH_PACKAGE}/launch/{DUAL_LAUNCH_FILE}"
)
DUAL_CONTAINER_LAUNCH_SCRIPT = """set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
if [ ! -f /workspace/ros_ws/install/setup.bash ]; then
  cd /workspace/ros_ws
  timeout 300 colcon build \\
    --packages-select agx_arm_description %(package)s \\
    --symlink-install
  cd /
fi
source /workspace/ros_ws/install/setup.bash
set -u
exec ros2 launch %(package)s %(launch_file)s \\
  robot_ids:=%(robots)s use_sim_time:=true \\
  config_output_dir:=%(config_dir)s
"""
GENERATED_CONFIG_FILES = (
    "model.urdf",
    "model.srdf",
    "kinematics.yaml",
    "joint_limits.yaml",
    "moveit_controllers.yaml",
    "ros2_controllers.yaml",
)
# The frozen scene-state manifest schema (A4): the observation instance ids of
# a session default to the manifest's own physical instance ids.
MANIFEST_SCHEMA = "rocobrick-scene-snapshot-manifest/v1"
MULTI_ROBOT_CONTAINER_PREFIX = "rocobrick-multi-robot-session-"
# Read-only operations the multi-robot session may dispatch: no target, no
# motion, no planning-scene mutation.
READ_ONLY_OPERATION_KINDS = frozenset(
    {OperationKind.READINESS, OperationKind.SCENE_QUERY}
)
MULTI_ROBOT_MAX_STEPS = 2000000
# Pinned low end of the measured host step rate (~458-524 steps/s on this
# machine, P1/R2 evidence). ``--hold-steps`` is derived from it so the bridge
# exits on its ``--hold-seconds`` bound instead of the step bound.
HOST_STEPS_PER_SECOND = 458.0
# Bounded host teardown budget of the multi-robot session (see
# ``_stop_host``): a self-exit is preferred, the bridge's hold is never waited
# out, and the bridge catches KeyboardInterrupt itself, so an interrupted stop
# still produces the bridge's own ``stopped``/cleanup evidence.
MULTI_ROBOT_HOST_SELF_EXIT_GRACE_S = 90.0
MULTI_ROBOT_HOST_INT_GRACE_S = 60.0
MULTI_ROBOT_HOST_KILL_GRACE_S = 15.0
CONTAINER_ALIVE_ATTEMPTS = 3
CONTAINER_ALIVE_PAUSE_S = 1.0
CONTAINER_LOG_TAIL = 200
CONTAINER_LOG_TIMEOUT_S = 30.0
ROBOT_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CONFIG_SUBDIR_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class MultiRobotSessionError(RuntimeSessionError):
    """Raised when the multi-robot session cannot reach or keep readiness."""


def _finite_number(value: object) -> float | None:
    """The float value of a finite JSON number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _vector3(value: object, label: str) -> list[float]:
    """Validate one three-component finite vector."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must hold three components")
    components: list[float] = []
    for index, component in enumerate(value):
        number = _finite_number(component)
        if number is None:
            raise ValueError(f"{label}[{index}] must be a finite number")
        components.append(number)
    return components


def _applied_command_messages(document: Mapping[str, object]) -> int | None:
    """The bridge's own applied-command total, when its evidence carries one.

    A4's evidence aggregates ``commands.applied`` (and ``commands.by_robot``
    in scene mode); when neither is present the total is unknown (``None``)
    and the caller trusts the bridge's own attestation instead of inventing a
    number. The zero-motion cross-check of an accepted early stop uses it.
    """
    commands = document.get("commands")
    if not isinstance(commands, Mapping):
        return None
    total = commands.get("applied")
    if isinstance(total, int) and not isinstance(total, bool):
        return total
    by_robot = commands.get("by_robot")
    if isinstance(by_robot, Mapping) and by_robot:
        applied = [
            entry.get("applied")
            for entry in by_robot.values()
            if isinstance(entry, Mapping)
        ]
        if applied and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in applied
        ):
            return sum(applied)
    return None


def _write_json_atomic(path: Path, document: Mapping[str, object]) -> None:
    """Write one JSON document atomically (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _manifest_physical_instance_ids(manifest_path: Path) -> tuple[str, ...]:
    """Every physical instance id of a frozen scene-snapshot manifest.

    Pure JSON read (no Isaac import): the manifest is exactly the set of
    physical instances the A4 host bridge loaded, so its ids are the scene
    selection an observation can ask for. Any deviation fails closed.
    """
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MultiRobotSessionError(
            f"scene manifest unreadable: {manifest_path}: {error}"
        )
    if not isinstance(document, Mapping) or document.get("schema") != MANIFEST_SCHEMA:
        raise MultiRobotSessionError(
            f"scene manifest {manifest_path} schema must be exactly {MANIFEST_SCHEMA!r}"
        )
    objects = document.get("objects")
    if not isinstance(objects, Sequence) or isinstance(
        objects, (str, bytes, bytearray)
    ):
        raise MultiRobotSessionError(
            f"scene manifest {manifest_path} carries no objects array"
        )
    if not objects:
        raise MultiRobotSessionError(
            f"scene manifest {manifest_path} carries no objects"
        )
    instance_ids: list[str] = []
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise MultiRobotSessionError(
                f"scene manifest {manifest_path} objects[{index}] is not an object"
            )
        instance_id = item.get("physical_instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise MultiRobotSessionError(
                f"scene manifest {manifest_path} objects[{index}] carries no "
                "physical_instance_id"
            )
        instance_ids.append(instance_id)
    if len(set(instance_ids)) != len(instance_ids):
        raise MultiRobotSessionError(
            f"scene manifest {manifest_path} repeats a physical_instance_id"
        )
    return tuple(sorted(instance_ids))


def build_global_scene_document(
    robot_id: str, view: Mapping[str, object]
) -> dict[str, object]:
    """Rebuild the A4 scene-state document around one robot-scoped answer.

    The A5 child answers a robot-targeted SCENE_QUERY with the robot-scoped
    slice of the global ``/p05_host_bridge/scene_state`` response
    (``rocobrick.ros_moveit_runtime.robot_scene_view`` projected by
    ``rocobrick.ros_session_entry.filter_scene_state``): the shared
    ``sim_time_s``/``bricks``/``connections`` plus THIS robot's hoisted
    ``robot_id``/``gripper``/``opening_m``. That projection carries no schema
    key of its own (the child validated the A4 document schema before slicing
    it) and is accepted as-is; a ``schema`` key, when the child sends one,
    must be the A4 schema. A7 consumes the A4 document shape, so this pure
    function re-attaches the ``robots[<robot_id>]`` wrapper with exactly
    those observed values - never a value from another robot, never a
    default, never a synthesized reading. Every field is validated, the
    mandatory ``robot_id`` must be THIS robot (ownership), and any deviation
    raises ``ValueError`` so the caller fails closed.

    The A4 host writes the one shared ``sim_time_s`` into every robot block,
    so the robot-scoped answer carries the robot's own sim time, used for the
    shared key as well.
    """
    if not isinstance(robot_id, str) or not ROBOT_ID_PATTERN.fullmatch(robot_id):
        raise ValueError(f"robot_id must be a ROS-safe id, got {robot_id!r}")
    if not isinstance(view, Mapping):
        raise ValueError("the robot-scoped scene answer is not an object")
    schema = view.get("schema")
    if schema is not None and schema != SCENE_STATE_SCHEMA:
        raise ValueError(
            f"the robot-scoped scene answer schema must be exactly "
            f"{SCENE_STATE_SCHEMA!r}, got {schema!r}"
        )
    if view.get("robot_id") != robot_id:
        raise ValueError(
            f"the robot-scoped scene answer is owned by {view.get('robot_id')!r}, "
            f"not {robot_id!r}"
        )
    sim_time = _finite_number(view.get("sim_time_s"))
    if sim_time is None:
        raise ValueError(
            f"the robot-scoped scene answer of {robot_id!r} carries no finite "
            "sim_time_s"
        )
    bricks_document = view.get("bricks")
    if not isinstance(bricks_document, Mapping) or not bricks_document:
        raise ValueError(
            f"the robot-scoped scene answer of {robot_id!r} carries no bricks"
        )
    bricks: dict[str, dict[str, list[float]]] = {}
    for brick_id, entry in bricks_document.items():
        if not isinstance(brick_id, str) or not brick_id:
            raise ValueError("a brick id of the scene answer is not a string")
        if not isinstance(entry, Mapping):
            raise ValueError(f"scene answer brick {brick_id!r} is not an object")
        minimum = _vector3(entry.get("aabb_min"), f"bricks[{brick_id!r}].aabb_min")
        maximum = _vector3(entry.get("aabb_max"), f"bricks[{brick_id!r}].aabb_max")
        if any(low > high for low, high in zip(minimum, maximum)):
            raise ValueError(f"scene answer brick {brick_id!r} min exceeds max")
        bricks[brick_id] = {"aabb_min": minimum, "aabb_max": maximum}
        if "pose" in entry:
            bricks[brick_id]["pose"] = entry["pose"]
    connections_document = view.get("connections")
    if not isinstance(connections_document, Mapping):
        raise ValueError(
            f"the robot-scoped scene answer of {robot_id!r} carries no "
            "connections mapping"
        )
    connections: dict[str, bool] = {}
    for key, value in connections_document.items():
        if not isinstance(key, str) or not key or not isinstance(value, bool):
            raise ValueError(
                "scene answer connections must map non-empty strings to booleans"
            )
        connections[key] = value
    gripper_document = view.get("gripper")
    if (
        not isinstance(gripper_document, Mapping)
        or set(gripper_document) != set(GRIPPER_JOINT_NAMES)
    ):
        raise ValueError(
            f"the robot-scoped scene answer of {robot_id!r} must carry exactly "
            f"the gripper reading {sorted(GRIPPER_JOINT_NAMES)}"
        )
    gripper: dict[str, float] = {}
    for name in GRIPPER_JOINT_NAMES:
        number = _finite_number(gripper_document[name])
        if number is None:
            raise ValueError(
                f"scene answer gripper joint {name!r} of {robot_id!r} is not a "
                "finite number"
            )
        gripper[name] = number
    opening = view.get("opening_m")
    if opening is not None:
        number = _finite_number(opening)
        if number is None:
            raise ValueError(
                f"scene answer opening_m of {robot_id!r} is not a finite number"
            )
        opening = number
    return {
        "schema": SCENE_STATE_SCHEMA,
        "sim_time_s": sim_time,
        "bricks": bricks,
        "connections": connections,
        "robots": {
            robot_id: {
                "robot_id": robot_id,
                "sim_time_s": sim_time,
                "gripper": gripper,
                "opening_m": opening,
                **({"tcp_pose": view["tcp_pose"]} if "tcp_pose" in view else {}),
                **({"sequence": view["robot_sequence"]}
                   if "robot_sequence" in view else {}),
            }
        },
        **({"session_id": view["session_id"]} if "session_id" in view else {}),
        **({"manifest_sha256": view["manifest_sha256"]}
           if "manifest_sha256" in view else {}),
        **({"sequence": view["sequence"]} if "sequence" in view else {}),
    }


@dataclass(frozen=True)
class MultiRobotSessionConfig:
    """Frozen bounds and identities of one multi-robot live session.

    ``robot_ids`` are the validated plan's robots (A8 passes them): ONE host
    bridge and ONE container are started for all of them, and the session
    envelope verifies readiness of every id - including a robot with no
    operation at all.

    The A4 scene-mode group is REQUIRED: only a scene-mode host bridge serves
    the global ``/p05_host_bridge/scene_state`` document with a
    ``robots[<robot_id>]`` block per loaded robot, and the A5 per-robot
    capability readiness (scene component) and the observation transport both
    need it. Observation instance ids default to the scene manifest's own
    physical instance ids when none are given.

    ``host_hold_budget_s`` is the post-marker observation budget: the host
    bridge holds for ``readiness_timeout_s + host_hold_budget_s + 120`` seconds
    capped by ``shutdown_timeout_s - 60`` (the hold must end before ``stop()``
    stops waiting, because the bridge writes its evidence only when its own
    hold completes). ``host_teardown_grace_s`` is the extra bounded grace the
    stop gives a bridge whose own evidence PROVES its hold already completed
    (it is then only inside its slow native Kit teardown); an early stop never
    waits out the remaining hold - see :meth:`stop`.

    An intentional early stop is accepted ONLY from the bridge's own
    attestation: its completed hold, or A4's additive ``early_stop_accepted``
    key (ready bridge + bound SIGINT + orderly shutdown). Controller hold
    commands are recorded separately from skill-operation motion. There is no
    session-side bypass flag.

    ``structure_goal_path``/``structure_start_path`` are the optional explicit
    structure files forwarded to the scene-mode host bridge as
    ``--structure-goal``/``--structure-start`` (scene loading only; the
    bridge binds the internal topology carrier from them and dispatches no
    operation). The start is optional and ``None`` means EMPTY initial
    structure - no directory default is ever inferred. They cannot be
    combined with the legacy ``structure_dir``.
    """

    repo_root: Path
    robot_ids: tuple[str, ...]
    scene_manifest: Path
    scene_snapshot_out: Path
    host_instance_id: str
    plan_path: Path
    config_path: Path
    structure_dir: Path | None = None
    structure_goal_path: Path | None = None
    structure_start_path: Path | None = None
    ros_image: str = "roco-piper-ros:humble"
    collision_filter_robot_id: str | None = None
    collision_filter_instance_ids: tuple[str, ...] = ()
    observation_physical_instance_ids: tuple[str, ...] = ()
    observation_connection_pairs: tuple[tuple[str, str], ...] = ()
    config_output_subdir: str = "dual-robot-config"
    verify_plan_robots: bool = True
    ros_domain_id: int = 73
    startup_timeout_s: float = 300.0
    readiness_timeout_s: float = 240.0
    operation_timeout_s: float = 180.0
    host_hold_budget_s: float = 300.0
    host_teardown_grace_s: float = 120.0
    shutdown_timeout_s: float = 900.0

    def __post_init__(self) -> None:
        """Validate the paths, robot ids, observation selection and bounds."""
        if not isinstance(self.repo_root, Path) or not self.repo_root.is_dir():
            raise ValueError(
                f"repo_root must be an existing directory, got {self.repo_root!r}"
            )
        if (
            not isinstance(self.robot_ids, tuple)
            or not self.robot_ids
            or any(not isinstance(item, str) for item in self.robot_ids)
        ):
            raise ValueError("robot_ids must be a non-empty tuple of strings")
        if len(set(self.robot_ids)) != len(self.robot_ids):
            raise ValueError("robot_ids must be unique")
        for robot_id in self.robot_ids:
            if not ROBOT_ID_PATTERN.fullmatch(robot_id):
                raise ValueError(
                    f"robot id {robot_id!r} is not a ROS-safe identifier"
                )
        for name in (
            "scene_manifest",
            "scene_snapshot_out",
            "plan_path",
            "config_path",
        ):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a Path")
        for name in ("structure_goal_path", "structure_start_path"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise ValueError(f"{name} must be a Path or None")
        if (
            self.structure_start_path is not None
            and self.structure_goal_path is None
        ):
            raise ValueError(
                "structure_start_path requires structure_goal_path"
            )
        if (
            self.structure_goal_path is not None
            and self.structure_dir is not None
        ):
            raise ValueError(
                "structure_goal_path cannot be combined with structure_dir"
            )
        if not isinstance(self.host_instance_id, str) or not self.host_instance_id:
            raise ValueError("host_instance_id must be a non-empty string")
        if not isinstance(self.ros_image, str) or not self.ros_image:
            raise ValueError("ros_image must be a non-empty string")
        if self.collision_filter_robot_id is not None and (
            type(self.collision_filter_robot_id) is not str
            or not self.collision_filter_robot_id
        ):
            raise ValueError("collision_filter_robot_id must be a non-empty string")
        if len(set(self.collision_filter_instance_ids)) != len(
            self.collision_filter_instance_ids
        ):
            raise ValueError("collision_filter_instance_ids must be unique")
        if any(
            type(item) is not str or not item
            for item in self.collision_filter_instance_ids
        ):
            raise ValueError("collision_filter_instance_ids must be non-empty strings")
        if (
            self.collision_filter_instance_ids
            and self.collision_filter_robot_id is None
        ):
            raise ValueError(
                "collision_filter_instance_ids require collision_filter_robot_id"
            )
        if len(set(self.observation_physical_instance_ids)) != len(
            self.observation_physical_instance_ids
        ):
            raise ValueError("observation_physical_instance_ids must be unique")
        if any(
            type(item) is not str or not item
            for item in self.observation_physical_instance_ids
        ):
            raise ValueError(
                "observation_physical_instance_ids must be non-empty strings"
            )
        for index, pair in enumerate(self.observation_connection_pairs):
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(type(item) is not str or not item for item in pair)
            ):
                raise ValueError(
                    f"observation_connection_pairs[{index}] must be exactly two "
                    "non-empty strings"
                )
        if not isinstance(self.config_output_subdir, str) or not (
            CONFIG_SUBDIR_PATTERN.fullmatch(self.config_output_subdir)
        ):
            raise ValueError(
                "config_output_subdir must be a plain directory name, got "
                f"{self.config_output_subdir!r}"
            )
        if not isinstance(self.verify_plan_robots, bool):
            raise ValueError("verify_plan_robots must be a boolean")
        if (
            isinstance(self.ros_domain_id, bool)
            or not isinstance(self.ros_domain_id, int)
            or self.ros_domain_id < 0
        ):
            raise ValueError(
                f"ros_domain_id must be a non-negative integer, got "
                f"{self.ros_domain_id!r}"
            )
        for name in (
            "startup_timeout_s",
            "readiness_timeout_s",
            "operation_timeout_s",
            "host_hold_budget_s",
            "host_teardown_grace_s",
            "shutdown_timeout_s",
        ):
            if not _is_finite_positive(getattr(self, name)):
                raise ValueError(
                    f"{name} must be finite and positive, got {getattr(self, name)!r}"
                )

    def resolved_observation_instance_ids(self) -> tuple[str, ...]:
        """The physical instances an observation asks for, or fail closed.

        Explicit ``observation_physical_instance_ids`` win; otherwise every
        physical instance id of the configured scene manifest is used, which
        is exactly the scene the A4 host bridge loaded.
        """
        if self.observation_physical_instance_ids:
            return self.observation_physical_instance_ids
        return _manifest_physical_instance_ids(self.scene_manifest)


@dataclass(frozen=True)
class MultiRobotReadinessEvidence:
    """Per-robot readiness of the one host + one dual-stack container."""

    robots: tuple[str, ...]
    readiness_by_robot: Mapping[str, Mapping[str, object]]
    all_ready: bool
    request_path: str | None = None
    result_path: str | None = None
    error: str | None = None

    @property
    def not_ready_robots(self) -> tuple[str, ...]:
        """Configured robots whose readiness report did not pass."""
        return tuple(
            robot_id
            for robot_id in self.robots
            if not bool(self.readiness_by_robot.get(robot_id, {}).get("passed"))
        )

    @property
    def passed(self) -> bool:
        """True only when THE SESSION envelope proved every robot ready.

        The verdict requires the child's ``all_ready``, one report per
        configured robot, and every report ``passed``: a missing robot report
        fails closed exactly like an unready robot.
        """
        return bool(
            self.all_ready
            and self.robots
            and set(self.readiness_by_robot) == set(self.robots)
            and not self.not_ready_robots
        )

    def serialize(self) -> dict[str, object]:
        """Return the JSON-safe readiness document."""
        return {
            "robots": list(self.robots),
            "readiness_by_robot": {
                str(robot_id): dict(report)
                for robot_id, report in self.readiness_by_robot.items()
            },
            "all_ready": bool(self.all_ready),
            "passed": self.passed,
            "not_ready_robots": list(self.not_ready_robots),
            "request_path": self.request_path,
            "result_path": self.result_path,
            "error": self.error,
        }


@dataclass(frozen=True)
class MultiRobotObservationEvidence:
    """One robot-targeted read-only observation record."""

    robot_id: str
    operation_id: str
    kind: str
    passed: bool
    sim_time_s: float | None
    request_path: str
    result_path: str
    error: str | None = None

    def serialize(self) -> dict[str, object]:
        """Return the JSON-safe observation record."""
        return {
            "robot_id": self.robot_id,
            "operation_id": self.operation_id,
            "kind": self.kind,
            "passed": bool(self.passed),
            "motion_sent": False,
            "sim_time_s": self.sim_time_s,
            "request_path": self.request_path,
            "result_path": self.result_path,
            "error": self.error,
        }


@dataclass(frozen=True)
class MultiRobotSessionEvidence:
    """Serializable lifecycle evidence of one multi-robot session."""

    status: RuntimeSessionStatus
    robots: tuple[str, ...]
    readiness: MultiRobotReadinessEvidence | None
    host_pid: int | None
    container_name: str | None
    container_id: str | None = None
    launch_command: tuple[str, ...] = ()
    config_output_dir: str | None = None
    generated_configs: Mapping[str, object] = field(default_factory=dict)
    plan_check: Mapping[str, object] = field(default_factory=dict)
    observation_instance_ids: tuple[str, ...] = ()
    observations: tuple[Mapping[str, object], ...] = ()
    profile_report: Mapping[str, object] = field(default_factory=dict)
    host_evidence_path: str | None = None
    log_paths: tuple[str, ...] = ()
    cleanup: Mapping[str, object] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    schema: str = MULTI_ROBOT_SESSION_SCHEMA

    def serializable(self) -> dict[str, object]:
        """Return the JSON-safe evidence document."""
        return {
            "schema": self.schema,
            "status": str(self.status.value),
            "robots": list(self.robots),
            "readiness": (
                self.readiness.serialize() if self.readiness is not None else None
            ),
            "host_pid": self.host_pid,
            "container_name": self.container_name,
            "container_id": self.container_id,
            "launch_command": list(self.launch_command),
            "config_output_dir": self.config_output_dir,
            "generated_configs": dict(self.generated_configs),
            "plan_check": dict(self.plan_check),
            "observation_instance_ids": list(self.observation_instance_ids),
            "observations": [dict(record) for record in self.observations],
            "profile_report": dict(self.profile_report),
            "host_evidence_path": self.host_evidence_path,
            "log_paths": list(self.log_paths),
            "cleanup": dict(self.cleanup),
            "errors": list(self.errors),
        }


class MultiRobotIsaacMoveItSession:
    """Orchestrates the one host bridge + one dual-stack ROS/MoveIt container.

    Lifecycle: ``start()`` runs the offline preflight, generates the DDS
    profiles, launches the A4 host bridge (scene mode, every plan robot), waits
    for its readiness marker, starts ONE container running the A3 dual launch,
    and then requires the A5 session envelope to prove EVERY plan robot ready.
    ``stop()`` is idempotent, bounded, and touches only the container and the
    host process this session started. Every command crosses the injected
    :class:`ProcessRunner`; waits are paced by the injected clock/sleep hooks,
    so the whole session is exercised offline with fakes.

    This stage sends no motion: the only operations it ever dispatches are the
    readiness envelope (``operations`` empty) and robot-targeted read-only
    SCENE_QUERY observations.
    """

    def __init__(
        self,
        config: MultiRobotSessionConfig,
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
                "multi-robot-session-"
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid4().hex[:8]}"
            )
        )
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        self._base_env: dict[str, str] = (
            _with_headed_display(
                dict(base_env) if base_env is not None else dict(os.environ)
            )
        )
        self._lock = threading.Lock()
        self.status = RuntimeSessionStatus.NOT_STARTED
        self.errors: list[str] = []
        self.readiness: MultiRobotReadinessEvidence | None = None
        self.host_handle: ProcessHandle | None = None
        self.host_pid: int | None = None
        self.container_name: str | None = None
        self.container_id: str | None = None
        self.profile_report: dict[str, object] = {}
        self.observations: list[MultiRobotObservationEvidence] = []
        self.plan_check: dict[str, object] = {}
        self.generated_configs: dict[str, object] = {}
        self._observation_instance_ids: tuple[str, ...] = ()
        self.last_result: RuntimeOperationResult | None = None
        self._stop_evidence: MultiRobotSessionEvidence | None = None
        self._dds_dir = self.session_dir / "dds"
        self._host_log = self.session_dir / "host-bridge.log"
        self._container_log = self.session_dir / "dual-launch.log"
        self._active_motion_children: dict[
            str, tuple[ProcessHandle, Path, str, OperationKind, str]
        ] = {}
        self._active_motion_lock = threading.Lock()
        self._config_output_dir = (
            self.session_dir / config.config_output_subdir
        )

    # ------------------------------------------------------------- evidence

    def _log_paths(self) -> tuple[str, ...]:
        return (
            str(self._host_log),
            str(self._dds_dir / CONTAINER_MESH_PROFILE),
            str(self.session_dir / "readiness-request.json"),
            str(self.session_dir / "readiness-result.json"),
            str(self.session_dir / "multi-robot-session.json"),
            str(self._container_log),
        )

    def _host_evidence_path(self) -> str | None:
        path = self.session_dir / "host-bridge-evidence.json"
        return str(path) if path.is_file() else None

    def _evidence(
        self, *, cleanup: Mapping[str, object] | None = None
    ) -> MultiRobotSessionEvidence:
        """Snapshot the current session evidence (cleanup optional)."""
        return MultiRobotSessionEvidence(
            status=self.status,
            robots=self.config.robot_ids,
            readiness=self.readiness,
            host_pid=self.host_pid,
            container_name=self.container_name,
            container_id=self.container_id,
            launch_command=tuple(self._container_launch_command()),
            config_output_dir=str(self._config_output_dir),
            generated_configs=dict(self.generated_configs),
            plan_check=dict(self.plan_check),
            observation_instance_ids=self._observation_instance_ids,
            observations=tuple(
                record.serialize() for record in self.observations
            ),
            profile_report=dict(self.profile_report),
            host_evidence_path=self._host_evidence_path(),
            log_paths=self._log_paths(),
            cleanup=dict(cleanup) if cleanup is not None else {},
            errors=tuple(self.errors),
        )

    def _write_evidence(self) -> None:
        """Write the lifecycle evidence document atomically."""
        with self._lock:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            document = (
                self._stop_evidence.serializable()
                if self._stop_evidence is not None
                else self._evidence().serializable()
            )
            target = self.session_dir / "multi-robot-session.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)

    def _record_error(self, message: str) -> None:
        self.errors.append(message)

    def _fail(self, phase: str, detail: str) -> MultiRobotSessionError:
        message = f"{phase} failed: {detail}"
        self._record_error(message)
        return MultiRobotSessionError(message)

    # ---------------------------------------------------------------- start

    def start(self) -> MultiRobotReadinessEvidence:
        """Start the one host + one dual-stack container and require readiness.

        Raises:
            MultiRobotSessionError: When any phase fails; the accumulated
                evidence keeps the failure details and ``stop()`` still
                performs and records cleanup.
        """
        if self.status is not RuntimeSessionStatus.NOT_STARTED:
            raise MultiRobotSessionError(
                f"start is only valid from NOT_STARTED, not {self.status}"
            )
        self.status = RuntimeSessionStatus.STARTING
        self.session_dir.mkdir(parents=True, exist_ok=True)
        try:
            if (
                self.config.scene_manifest.is_file()
                or self.config.observation_physical_instance_ids
            ):
                self._observation_instance_ids = (
                    self.config.resolved_observation_instance_ids()
                )
            self._verify_plan_robots()
            self._verify_p1a_prerequisite()
            self._verify_session_assets()
            self._run_model_audit()
            self._generate_profiles()
            self._launch_host()
            self._wait_for_marker()
            if not self._observation_instance_ids:
                self._observation_instance_ids = (
                    self.config.resolved_observation_instance_ids()
                )
            self._launch_container()
            self.generated_configs = self._collect_generated_configs()
            evidence = self._run_readiness_session()
        except (MultiRobotSessionError, RuntimeSessionError) as error:
            self.status = RuntimeSessionStatus.FAILED
            if str(error) not in self.errors:
                self._record_error(str(error))
            self._write_evidence()
            raise
        self.readiness = evidence
        self.status = RuntimeSessionStatus.READY
        self._write_evidence()
        return evidence

    def _verify_plan_robots(self) -> None:
        """Require the session robots to be the validated plan's robots.

        The A4 host bridge binds one endpoint per validated plan robot, so a
        session whose robot list disagreed with the plan could collect
        readiness from a robot the plan does not contain (or miss one of its
        robots). An ``assembly-plan/v2`` document is validated by the frozen
        STAP contract; a ``robot-operation-plan/v1`` document is validated by
        the strict structure-only robot-operation contract with
        ``source_plan=None`` (the external plan is immutable input that is
        never bound, reordered, or repaired here). Any other schema falls to
        the STAP parser and fails closed. The check happens BEFORE any
        process starts, and the plan contract module is imported lazily so
        the runtime session import graph stays free of it until verification
        actually runs.
        """
        if not self.config.verify_plan_robots:
            self.plan_check = {"verified": False, "reason": "disabled by config"}
            return
        try:
            document = json.loads(
                self.config.plan_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise self._fail("plan_robots", f"plan unreadable: {error}")
        from rocobrick.robot_operation_plan import (
            ROBOT_OPERATION_PLAN_SCHEMA,
            parse_robot_operation_plan,
        )

        if (
            isinstance(document, dict)
            and document.get("schema") == ROBOT_OPERATION_PLAN_SCHEMA
        ):
            try:
                operation_plan = parse_robot_operation_plan(
                    document, source_plan=None
                )
            except ValueError as error:
                raise self._fail(
                    "plan_robots", f"{type(error).__name__}: {error}"
                )
            if set(operation_plan.robots) != set(self.config.robot_ids):
                raise self._fail(
                    "plan_robots",
                    f"the plan robots {sorted(operation_plan.robots)} are not "
                    f"the session robots {sorted(self.config.robot_ids)}",
                )
            self.plan_check = {
                "verified": True,
                "schema": operation_plan.schema,
                "plan_id": operation_plan.plan_id,
                "source_plan_id": operation_plan.source_plan_id,
                "plan_robots": list(operation_plan.robots),
                "session_robots": list(self.config.robot_ids),
            }
            return
        try:
            plan = parse_stap_plan(document)
        except ValueError as error:
            raise self._fail("plan_robots", f"{type(error).__name__}: {error}")
        if set(plan.robots) != set(self.config.robot_ids):
            raise self._fail(
                "plan_robots",
                f"the plan robots {sorted(plan.robots)} are not the session "
                f"robots {sorted(self.config.robot_ids)}",
            )
        self.plan_check = {
            "verified": True,
            "plan_id": plan.plan_id,
            "plan_robots": list(plan.robots),
            "session_robots": list(self.config.robot_ids),
        }

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
                "p1a_prerequisite", f"artifact did not pass every check: {path}"
            )

    def _verify_session_assets(self) -> None:
        """Every pinned file this session launches must exist before launch."""
        repo = self.config.repo_root
        for label, path in (
            ("host bridge", repo / HOST_BRIDGE_RELPATH),
            ("host environment wrapper", repo / HOST_ENV_RELPATH),
            ("model audit tool", repo / MODEL_AUDIT_RELPATH),
            ("DDS profile tool", repo / PROFILE_TOOL_RELPATH),
            ("dual-robot launch", repo / DUAL_LAUNCH_RELPATH),
        ):
            if not path.is_file():
                raise self._fail("session_assets", f"{label} missing: {path}")

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
        _write_json_atomic(
            self.session_dir / "preflight-model-audit.log",
            {"stdout": outcome.stdout, "stderr": outcome.stderr},
        )
        if outcome.timed_out:
            raise self._fail("model_audit", f"timed out: {shlex.join(outcome.argv)}")
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

    def _host_time_limit_s(self) -> float:
        """Host bridge ``--time-limit``: the whole bridge life plus a margin."""
        return (
            self.config.startup_timeout_s
            + self._host_hold_seconds()
            + 120.0
        )

    def _host_hold_seconds(self) -> float:
        """Host hold: the post-marker session budget inside the stop grace.

        The hold starts at the readiness marker, so it must cover the container
        launch, the readiness envelope and the observation budget in one bound;
        it also has to end BEFORE ``stop()`` gives up waiting
        (``shutdown_timeout_s``), because the bridge writes its evidence only
        when its hold completes and Kit intercepts an external SIGINT at the C
        level - so the bridge must be left enough time to self-exit orderly.

        The budget is ``host_hold_budget_s`` (default 300 s): the session now
        executes full robot operations (approach/close/lift plus the follow
        observation windows), not just readiness checks - a 45 s hold stopped
        the clock mid-operation and failed every later readiness gate
        (``clock_increasing``).  The bound still ends inside
        ``shutdown_timeout_s`` (900 s) with the teardown grace.
        """
        return float(self.config.host_hold_budget_s)

    def _host_hold_steps(self) -> int:
        """Step bound matching the hold seconds at the measured step rate.

        The bridge holds for ``max(steps, seconds)``; the low end of the
        measured rate keeps the step bound from extending the hold beyond
        ``--hold-seconds`` on this machine.
        """
        return max(
            1,
            int(round(self._host_hold_seconds() * HOST_STEPS_PER_SECOND)),
        )

    def _host_argv(self) -> list[str]:
        """Scene-mode host bridge argv: every plan robot from the plan/config.

        The A4 bridge derives its robot set (one namespaced endpoint pair per
        robot) from the validated STAP plan and the user config; the session
        never passes a robot list of its own, so it cannot disagree with the
        plan it verified in :meth:`_verify_plan_robots`. Explicit structure
        files are forwarded verbatim as ``--structure-goal`` (whenever set)
        and ``--structure-start`` (only when not ``None``); the start flag is
        never inferred.
        """
        repo = self.config.repo_root
        argv = [
            "bash",
            str(repo / HOST_ENV_RELPATH),
            sys.executable,
            str(repo / HOST_BRIDGE_RELPATH),
            "--max-steps",
            str(MULTI_ROBOT_MAX_STEPS),
            "--time-limit",
            str(self._host_time_limit_s()),
            "--ready-timeout",
            str(min(120.0, self.config.startup_timeout_s)),
            "--hold-steps",
            str(self._host_hold_steps()),
            "--hold-seconds",
            str(self._host_hold_seconds()),
            "--evidence-out",
            str(self.session_dir / "host-bridge-evidence.json"),
            "--scene-manifest",
            str(self.config.scene_manifest),
            "--scene-snapshot-out",
            str(self.config.scene_snapshot_out),
            "--host-instance-id",
            self.config.host_instance_id,
            "--plan",
            str(self.config.plan_path),
            "--config",
            str(self.config.config_path),
        ]
        if self.config.structure_dir is not None:
            argv.extend(("--structure-dir", str(self.config.structure_dir)))
        if self.config.structure_goal_path is not None:
            argv.extend(
                ("--structure-goal", str(self.config.structure_goal_path))
            )
        if self.config.structure_start_path is not None:
            argv.extend(
                ("--structure-start", str(self.config.structure_start_path))
            )
        if self.config.collision_filter_robot_id is not None:
            argv.extend(
                ("--physics-filter-robot-id", self.config.collision_filter_robot_id)
            )
            for instance_id in self.config.collision_filter_instance_ids:
                argv.extend(("--physics-filter-instance-id", instance_id))
        return argv

    def _host_env(self) -> dict[str, str]:
        env = dict(self._base_env)
        env["ROS_DOMAIN_ID"] = str(self.config.ros_domain_id)
        env["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(self._dds_dir / HOST_PROFILE)
        return env

    def _launch_host(self) -> None:
        """Launch the host Isaac bridge through isaac-ros-env.sh (bounded)."""
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

    def _dual_launch_script(self) -> str:
        """Container-side script: build if needed, then the A3 dual launch."""
        return DUAL_CONTAINER_LAUNCH_SCRIPT % {
            "package": DUAL_LAUNCH_PACKAGE,
            "launch_file": DUAL_LAUNCH_FILE,
            "robots": ",".join(self.config.robot_ids),
            "config_dir": (
                f"{CONTAINER_SESSION_MOUNT}/{self.config.config_output_subdir}"
            ),
        }

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
            self._dual_launch_script(),
        ]

    def _container_launch_command(self) -> list[str]:
        """The exact A3 dual launch command inside the container."""
        return [
            "ros2",
            "launch",
            DUAL_LAUNCH_PACKAGE,
            DUAL_LAUNCH_FILE,
            f"robot_ids:={','.join(self.config.robot_ids)}",
            "use_sim_time:=true",
            "config_output_dir:="
            f"{CONTAINER_SESSION_MOUNT}/{self.config.config_output_subdir}",
        ]

    def _launch_container(self) -> None:
        """Start ONE unique rootless container whose PID-1 is the dual launch."""
        name = f"{MULTI_ROBOT_CONTAINER_PREFIX}{uuid4().hex[:12]}"
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
        self.container_id = outcome.stdout.strip() or None
        for attempt in range(CONTAINER_ALIVE_ATTEMPTS):
            if self._container_running():
                return
            if attempt + 1 < CONTAINER_ALIVE_ATTEMPTS:
                self._sleep(CONTAINER_ALIVE_PAUSE_S)
        log_path = self._capture_container_log()
        raise self._fail(
            "container_start",
            f"container {name} is not running after docker run; its log tail is "
            f"at {log_path}",
        )

    def _container_running(self) -> bool:
        """True while the session's own container is listed as running."""
        if self.container_name is None:
            return False
        outcome = self.runner.run(
            [
                "docker",
                "ps",
                "--filter",
                f"name={self.container_name}",
                "--format",
                "{{.Names}}",
            ],
            timeout_s=CONTAINER_LOG_TIMEOUT_S,
        )
        return not outcome.timed_out and any(
            line.strip() == self.container_name
            for line in outcome.stdout.splitlines()
        )

    def _capture_container_log(self) -> str | None:
        """Persist the container's own log tail as durable session evidence."""
        name = self.container_name
        if name is None:
            return None
        outcome = self.runner.run(
            ["docker", "logs", "--tail", str(CONTAINER_LOG_TAIL), name],
            timeout_s=CONTAINER_LOG_TIMEOUT_S,
        )
        text = ""
        if not outcome.timed_out:
            text = outcome.stdout
            if outcome.stderr:
                text = f"{text}\n{outcome.stderr}"
        try:
            self._container_log.write_text(text, encoding="utf-8")
        except OSError:
            return None
        return str(self._container_log)

    def _collect_generated_configs(self) -> dict[str, object]:
        """Record the A3 per-robot generated config files (launch evidence)."""
        report: dict[str, object] = {}
        for robot_id in self.config.robot_ids:
            directory = self._config_output_dir / robot_id
            files = {
                name: (directory / name).is_file()
                for name in GENERATED_CONFIG_FILES
            }
            report[robot_id] = {
                "directory": str(directory),
                "files": files,
                "complete": all(files.values()),
            }
        return report

    # ------------------------------------------------------------ dispatch

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

    def _session_child_argv(self, request_path: Path, result_path: Path) -> list[str]:
        """Docker exec argv for the A5 multi-robot session entry.

        The child entry is invoked with ``--robots`` (the same ids the A3 dual
        launch was given), so it builds one isolated runtime per robot on ONE
        node and answers the strict session documents.
        """
        inner = (
            "set +u; source /opt/ros/humble/setup.bash; "
            "if [ -f /workspace/ros_ws/install/setup.bash ]; then "
            "source /workspace/ros_ws/install/setup.bash; fi; "
            f'set -u; export PYTHONPATH="{CONTAINER_PYTHONPATH}:${{PYTHONPATH:-}}"; '
            'exec python3 -m rocobrick.ros_session_entry '
            '--request "$1" --result "$2" --robots "$3"'
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
            ",".join(self.config.robot_ids),
        ]

    def _dispatch_session(
        self,
        request: RosSessionRequest,
        request_path: Path,
        result_path: Path,
        *,
        timeout_s: float,
        cancellable: bool = False,
        cancellable_operation_id: str | None = None,
    ) -> RosSessionResult:
        """Run one session envelope through the container and validate it.

        The request is rewritten into the session directory (strictly
        serialized by the A5 contract), dispatched to the child entry, and the
        answer is parsed with the strict session result schema and checked to
        answer exactly this request and this robot set. A transport, schema or
        identity failure raises and marks the session FAILED.
        """
        if self.container_name is None:
            raise self._fail("session_dispatch", "no container is running")
        if cancellable and (
            not isinstance(cancellable_operation_id, str)
            or not cancellable_operation_id
            or len(request.operations) != 1
            or request.operations[0].operation.request_id != cancellable_operation_id
        ):
            raise self._fail(
                "session_dispatch", "cancellable operation identity mismatch"
            )
        _write_json_atomic(request_path, request.serialize())
        if not cancellable:
            outcome = self.runner.run(
                self._session_child_argv(request_path, result_path),
                timeout_s=timeout_s,
            )
            if outcome.timed_out:
                raise self._fail(
                    "session_dispatch",
                    f"the child session entry exceeded its {timeout_s}s bound",
                )
            if not result_path.is_file():
                raise self._fail(
                    "session_dispatch",
                    f"the child session entry wrote no result "
                    f"(exit={outcome.returncode}): {outcome.stderr.strip()[-400:]}",
                )
        else:
            output_path = result_path.with_suffix(result_path.suffix + ".stdout.log")
            error_path = result_path.with_suffix(result_path.suffix + ".stderr.log")
            with self._active_motion_lock:
                if cancellable_operation_id in self._active_motion_children:
                    raise self._fail(
                        "session_dispatch", "motion operation is already active"
                    )
                handle = self.runner.launch(
                    self._session_child_argv(request_path, result_path),
                    env=dict(self._base_env),
                    stdout_path=output_path,
                    stderr_path=error_path,
                )
                self._active_motion_children[cancellable_operation_id] = (
                    handle,
                    result_path,
                    request.operations[0].robot_id,
                    request.operations[0].operation.kind,
                    request.request_id,
                )
            deadline = self._clock() + timeout_s
            try:
                while handle.poll() is None:
                    if self._clock() >= deadline:
                        try:
                            handle.send_signal(signal.SIGINT)
                        except Exception:
                            pass
                        raise self._fail(
                            "session_dispatch",
                            f"the child session entry exceeded its {timeout_s}s bound",
                        )
                    self._sleep(0.05)
                handle.wait(timeout_s=0.0)
            finally:
                with self._active_motion_lock:
                    self._active_motion_children.pop(cancellable_operation_id, None)
            if not result_path.is_file():
                error_text = error_path.read_text(encoding="utf-8")[-400:]
                raise self._fail(
                    "session_dispatch",
                    f"the child session entry wrote no result "
                    f"(exit={handle.poll()}): {error_text}",
                )
        try:
            document = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise self._fail("session_dispatch", f"session result unreadable: {error}")
        try:
            result = RosSessionResult.parse(document)
        except ValueError as error:
            raise self._fail(
                "session_dispatch", f"session result schema invalid: {error}"
            )
        if result.request_id != request.request_id:
            raise self._fail(
                "session_dispatch",
                f"the child answered request {result.request_id!r}, not "
                f"{request.request_id!r}",
            )
        if result.robots != request.robots:
            raise self._fail(
                "session_dispatch",
                f"the child answered robots {list(result.robots)}, not "
                f"{list(request.robots)}",
            )
        return result

    def _run_readiness_session(self) -> MultiRobotReadinessEvidence:
        """Zero-motion readiness envelope for ALL plan robots (A5).

        The request carries an EMPTY ``operations`` array: the child answers
        ``readiness_by_robot`` for every configured robot - including a robot
        with no operation at all - and never dispatches anything.
        """
        request = RosSessionRequest(
            request_id=f"readiness-{uuid4().hex[:12]}",
            robots=self.config.robot_ids,
            timeout_s=self.config.readiness_timeout_s,
            operations=(),
        )
        request_path = self.session_dir / "readiness-request.json"
        result_path = self.session_dir / "readiness-result.json"
        try:
            result = self._dispatch_session(
                request,
                request_path,
                result_path,
                timeout_s=self.config.readiness_timeout_s + CHILD_EXEC_MARGIN_S,
            )
        except MultiRobotSessionError:
            self.readiness = MultiRobotReadinessEvidence(
                robots=self.config.robot_ids,
                readiness_by_robot={},
                all_ready=False,
                request_path=str(request_path),
                result_path=str(result_path),
                error="the readiness session envelope could not be dispatched",
            )
            raise
        evidence = MultiRobotReadinessEvidence(
            robots=self.config.robot_ids,
            readiness_by_robot={
                str(robot_id): dict(report)
                for robot_id, report in result.readiness_by_robot.items()
            },
            all_ready=bool(result.all_ready),
            request_path=str(request_path),
            result_path=str(result_path),
            error=result.error,
        )
        self.readiness = evidence
        if not evidence.passed:
            raise self._fail(
                "readiness",
                f"not every plan robot is ready: {list(evidence.not_ready_robots)} "
                f"of {list(self.config.robot_ids)}; {result.error}",
            )
        return evidence

    # ------------------------------------------------------ motion dispatch

    def run_motion_operation(
        self, robot_id: str, operation: RuntimeOperation
    ) -> Mapping[str, object]:
        """Dispatch one validated motion operation to its robot runtime.

        The child session performs the final target admission and action
        execution. This wrapper only enforces session readiness, robot
        ownership, request/result identity and fail-closed session state.
        Cancellation remains owned by the child action runtime; a returned
        operation record is not treated as safe unless its execution evidence
        says so.
        """
        target = self._require_robot(robot_id)
        if not isinstance(operation, RuntimeOperation):
            raise MultiRobotSessionError("operation must be a RuntimeOperation")
        allowed = frozenset(
            {
                OperationKind.JOINT_TARGET,
                OperationKind.LINK6_POSE,
                OperationKind.TCP_POSE,
                OperationKind.GRIPPER_TARGET,
                OperationKind.APPLY_PLANNING_SCENE,
                OperationKind.ATTACH_PLANNING_SCENE,
            }
        )
        if operation.kind not in allowed:
            raise MultiRobotSessionError(
                f"{operation.kind.value} is not a motion operation accepted by "
                f"this session (allowed: {sorted(kind.value for kind in allowed)})"
            )
        self._require_ready("run_motion_operation")
        request = RosSessionRequest(
            request_id=f"session-{uuid4().hex[:12]}",
            robots=self.config.robot_ids,
            timeout_s=self.config.operation_timeout_s,
            operations=(RobotOperationRequest(robot_id=target, operation=operation),),
        )
        directory = self.session_dir / "operations"
        request_path = directory / f"{target}-{operation.request_id}-request.json"
        result_path = directory / f"{target}-{operation.request_id}-result.json"
        try:
            result = self._dispatch_session(
                request,
                request_path,
                result_path,
                timeout_s=self.config.operation_timeout_s + CHILD_EXEC_MARGIN_S,
                cancellable=True,
                cancellable_operation_id=operation.request_id,
            )
            matches = [
                record
                for record in result.operations
                if str(record.get("robot_id")) == target
                and str(record.get("operation_id")) == operation.request_id
            ]
            if len(matches) != 1:
                raise self._fail(
                    "motion_operation",
                    f"the session answer carries {len(matches)} records for "
                    f"{target}/{operation.request_id}; exactly one is required",
                )
            record = matches[0]
            if str(record.get("kind")) != operation.kind.value:
                raise self._fail(
                    "motion_operation",
                    f"the record kind {record.get('kind')!r} does not answer "
                    f"{operation.kind.value!r}",
                )
            inner = RuntimeOperationResult.parse(record.get("result"))
            if inner.request_id != operation.request_id or inner.kind is not operation.kind:
                raise self._fail(
                    "motion_operation",
                    "the operation result does not answer the request",
                )
            self.last_result = inner
            if not inner.passed:
                self.status = RuntimeSessionStatus.FAILED
            return record
        except MultiRobotSessionError:
            self.status = RuntimeSessionStatus.FAILED
            self._write_evidence()
            raise

    # ------------------------------------------------------ motion cancellation

    def cancel_motion_operation(self, operation_id: str) -> Mapping[str, object]:
        """Signal one running motion child and wait for its strict result."""
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a non-empty string")
        with self._active_motion_lock:
            entry = self._active_motion_children.get(operation_id)
        if entry is None:
            return {"attempted": False, "terminal": False, "safe": False,
                    "error": "motion child is not active"}
        handle, result_path, robot_id, kind, session_request_id = entry
        try:
            handle.send_signal(signal.SIGINT)
        except Exception as exc:
            return {"attempted": True, "terminal": False, "safe": False,
                    "error": f"failed to signal motion child: {type(exc).__name__}: {exc}"}
        deadline = self._clock() + CANCEL_TIMEOUT_S
        while handle.poll() is None and self._clock() < deadline:
            self._sleep(0.05)
        if handle.poll() is None:
            return {"attempted": True, "terminal": False, "safe": False,
                    "error": "motion child did not exit after SIGINT"}
        terminal = False
        safe = False
        try:
            document = json.loads(result_path.read_text(encoding="utf-8"))
            records = document.get("operations", [])
            if isinstance(records, list) and len(records) == 1:
                record = records[0]
                result = record.get("result") if isinstance(record, dict) else None
                execution = result.get("execution") if isinstance(result, dict) else None
                if (
                    isinstance(record, dict)
                    and document.get("request_id") == session_request_id
                    and record.get("operation_id") == operation_id
                    and record.get("robot_id") == robot_id
                    and record.get("kind") == kind.value
                    and isinstance(result, dict)
                    and result.get("request_id") == operation_id
                    and result.get("kind") == kind.value
                    and isinstance(execution, dict)
                ):
                    terminal = execution.get("terminal") is True
                    safe = terminal and execution.get("safe_to_release") is True
        except (OSError, ValueError, AttributeError):
            pass
        return {"attempted": True, "terminal": terminal, "safe": safe,
                "process_exit": handle.poll(), "result_path": str(result_path)}

    # ---------------------------------------------------- read-only dispatch

    def _require_ready(self, phase: str) -> None:
        if self.status is not RuntimeSessionStatus.READY:
            raise MultiRobotSessionError(f"{phase} requires READY, not {self.status}")

    def _require_robot(self, robot_id: object) -> str:
        if not isinstance(robot_id, str) or robot_id not in self.config.robot_ids:
            raise MultiRobotSessionError(
                f"unknown session robot {robot_id!r}; this session covers "
                f"{list(self.config.robot_ids)}"
            )
        return robot_id

    def run_read_only_operation(
        self, robot_id: str, operation: RuntimeOperation
    ) -> Mapping[str, object]:
        """Dispatch ONE robot-targeted read-only operation; never motion.

        Only :data:`READ_ONLY_OPERATION_KINDS` (SCENE_QUERY, READINESS) are
        accepted: a motion or planning-scene kind is refused here, before any
        command or process is created, so this stage can never send motion.
        The A5 envelope still verifies every plan robot's readiness first: when
        any robot is not ready, the operation is refused and the session fails
        closed instead of answering from another robot.

        Returns:
            The strict A5 session operation record of this robot's operation
            (``robot_id``, ``operation_id``, ``kind``, ``dispatched``,
            ``motion_sent``, ``result``).

        Raises:
            MultiRobotSessionError: On an unknown robot, a non-read-only kind,
                a transport/schema failure, or when any robot is not ready.
        """
        target = self._require_robot(robot_id)
        if not isinstance(operation, RuntimeOperation):
            raise MultiRobotSessionError("operation must be a RuntimeOperation")
        if operation.kind not in READ_ONLY_OPERATION_KINDS:
            raise MultiRobotSessionError(
                f"{operation.kind.value} is not a read-only operation; this "
                "session dispatches no motion and no planning-scene change "
                f"(allowed: {sorted(kind.value for kind in READ_ONLY_OPERATION_KINDS)})"
            )
        self._require_ready("run_read_only_operation")
        request = RosSessionRequest(
            request_id=f"session-{uuid4().hex[:12]}",
            robots=self.config.robot_ids,
            timeout_s=self.config.operation_timeout_s,
            operations=(
                RobotOperationRequest(robot_id=target, operation=operation),
            ),
        )
        directory = self.session_dir / "operations"
        request_path = directory / f"{target}-{operation.request_id}-request.json"
        result_path = directory / f"{target}-{operation.request_id}-result.json"
        try:
            result = self._dispatch_session(
                request,
                request_path,
                result_path,
                timeout_s=self.config.operation_timeout_s + CHILD_EXEC_MARGIN_S,
            )
            record = self._validate_operation_record(
                result, target, operation, request_path, result_path
            )
        except MultiRobotSessionError:
            self.status = RuntimeSessionStatus.FAILED
            self._write_evidence()
            raise
        return record

    def _validate_operation_record(
        self,
        result: RosSessionResult,
        robot_id: str,
        operation: RuntimeOperation,
        request_path: Path,
        result_path: Path,
    ) -> Mapping[str, object]:
        """Strict per-robot checks of one session envelope answer."""
        if not result.all_ready:
            self._record_error(
                f"robot readiness refused the {operation.kind.value} operation "
                f"of {robot_id}: {result.error}"
            )
            raise self._fail(
                "operation",
                f"not every plan robot is ready; the {operation.kind.value} "
                f"operation of {robot_id} was not dispatched ({result.error})",
            )
        matches = [
            record
            for record in result.operations
            if str(record.get("robot_id")) == robot_id
            and str(record.get("operation_id")) == operation.request_id
        ]
        if len(matches) != 1:
            raise self._fail(
                "operation",
                f"the session answer carries {len(matches)} records for "
                f"{robot_id}/{operation.request_id}; exactly one is required",
            )
        record = matches[0]
        if str(record.get("kind")) != str(operation.kind.value):
            raise self._fail(
                "operation",
                f"the record kind {record.get('kind')!r} does not answer "
                f"{operation.kind.value!r}",
            )
        if record.get("dispatched") is not True:
            raise self._fail(
                "operation",
                f"the {operation.kind.value} operation of {robot_id} was not "
                "dispatched",
            )
        if record.get("motion_sent") is not False:
            raise self._fail(
                "operation",
                f"the {operation.kind.value} operation of {robot_id} reported "
                "motion; this stage must never move",
            )
        try:
            inner = RuntimeOperationResult.parse(record.get("result"))
        except ValueError as error:
            raise self._fail("operation", f"operation result schema invalid: {error}")
        if inner.request_id != operation.request_id or inner.kind is not operation.kind:
            raise self._fail(
                "operation", "the operation result does not answer the request"
            )
        if not inner.passed:
            raise self._fail(
                "operation",
                f"the {operation.kind.value} operation of {robot_id} failed: "
                f"{inner.error}",
            )
        self.last_result = inner
        return record

    def observe_sync(self, robot_id: str) -> Mapping[str, object]:
        """Read one robot's own scene observation (blocking form).

        Dispatches a robot-targeted read-only SCENE_QUERY through the A5
        envelope and returns the A4 scene-state document shape: the shared
        ``sim_time_s``/``bricks``/``connections`` plus a
        ``robots[<robot_id>]`` block carrying THIS robot's ``robot_id``,
        ``sim_time_s``, ``gripper`` and ``opening_m``. There is no fallback:
        a missing or foreign robot block fails closed (the A5 child already
        refuses to slice another robot's or a designated-robot snapshot).

        Raises:
            MultiRobotSessionError: On an unknown robot, an unreadiness, a
                transport/schema failure, or an unverifiable scene answer.
        """
        target = self._require_robot(robot_id)
        self._require_ready("observe")
        if not self._observation_instance_ids:
            raise MultiRobotSessionError(
                "this session resolved no observation instance ids; configure "
                "observation_physical_instance_ids or a scene manifest"
            )
        try:
            operation = scene_query_operation(
                list(self._observation_instance_ids),
                [
                    list(pair)
                    for pair in self.config.observation_connection_pairs
                ],
                include_gripper=True,
                timeout_s=self.config.operation_timeout_s,
                request_id=f"observe-{target}-{uuid4().hex[:12]}",
            )
        except OperationContractError as error:
            raise MultiRobotSessionError(
                f"the observation query of {target} is not admissible: {error}"
            )
        try:
            record = self.run_read_only_operation(target, operation)
            document = self._observation_document(target, operation, record)
        except MultiRobotSessionError:
            self.status = RuntimeSessionStatus.FAILED
            self._write_evidence()
            raise
        with self._lock:
            self.observations.append(
                MultiRobotObservationEvidence(
                    robot_id=target,
                    operation_id=operation.request_id,
                    kind=operation.kind.value,
                    passed=True,
                    sim_time_s=float(document["sim_time_s"]),
                    request_path=str(
                        self.session_dir
                        / "operations"
                        / f"{target}-{operation.request_id}-request.json"
                    ),
                    result_path=str(
                        self.session_dir
                        / "operations"
                        / f"{target}-{operation.request_id}-result.json"
                    ),
                )
            )
        self._write_evidence()
        return document

    def _observation_document(
        self,
        robot_id: str,
        operation: RuntimeOperation,
        record: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Extract and validate the robot-scoped scene answer of one record."""
        try:
            inner = RuntimeOperationResult.parse(record.get("result"))
        except ValueError as error:
            raise self._fail("observation", f"observation result invalid: {error}")
        if inner.kind is not OperationKind.SCENE_QUERY:
            raise self._fail(
                "observation", f"the observation answered {inner.kind.value}"
            )
        scene_state = inner.execution.get("scene_state")
        if not isinstance(scene_state, Mapping):
            raise self._fail(
                "observation",
                f"the scene query of {robot_id} carried no scene_state document",
            )
        try:
            return build_global_scene_document(robot_id, scene_state)
        except ValueError as error:
            raise self._fail(
                "observation",
                f"the scene query of {robot_id} did not answer with its own "
                f"validated block: {error}",
            )

    async def observe(self, robot_id: str) -> Mapping[str, object]:
        """Async (A7 observation transport) form of :meth:`observe_sync`.

        The blocking dispatch runs in a worker thread so the caller's event
        loop stays free; every validation and fail-closed rule of
        :meth:`observe_sync` applies unchanged.
        """
        return await asyncio.to_thread(self.observe_sync, robot_id)

    # ---------------------------------------------------------------- stop

    def stop(self) -> MultiRobotSessionEvidence:
        """Orderly, bounded, idempotent teardown of THIS session's resources.

        Order and budget: the container is stopped and its removal verified
        first, then the host is stopped bounded - a short self-exit reap, the
        ``host_teardown_grace_s`` grace only when the bridge's own evidence
        proves its hold completed, otherwise SIGINT (bounded) and finally
        SIGKILL (bounded). An early zero-motion stop therefore never waits out
        the bridge's remaining hold, and the bridge turns the SIGINT into its
        own ``sigint_orderly_stop`` evidence.

        Only the container this session started (by its exact name) and the
        host process this session launched (by its reaped PID) are stopped:
        foreign containers, other sessions and pre-existing leftovers are
        neither killed nor claimed. ``cleanup["passed"]`` requires the
        verified container removal, the reaped host exit, the bridge's own
        orderly shutdown evidence (a completed hold or an evidenced
        intentional stop) and an empty session-scoped leftover scan;
        ``host_exit_kind`` always records how the host process ended.
        Safe to call from any state, including a partial start; a second call
        returns the same evidence.
        """
        if self._stop_evidence is not None:
            return self._stop_evidence
        previous = self.status
        self.status = RuntimeSessionStatus.STOPPING
        cleanup: dict[str, object] = {
            "scope": {
                "container": self.container_name,
                "host_pid": self.host_pid,
                "note": (
                    "cleanup is bounded to the container and the host process "
                    "started by THIS session; foreign sessions and pre-existing "
                    "leftovers are never touched"
                ),
            },
            "container_stop": None,
            "container_removed": False,
            "container_log": None,
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
        cleanup["container_log"] = self._capture_container_log()
        cleanup["container_stop"] = self._stop_container(cleanup)
        self._stop_host(cleanup)
        host_evidence = self._host_evidence_summary()
        cleanup["host_evidence"] = host_evidence
        cleanup["leftovers"] = self._scan_leftovers()
        leftovers = cleanup["leftovers"]
        host_evidence_orderly, early_stop_source = self._orderly_host_verdict(
            host_evidence, host_exit_kind=cleanup.get("host_exit_kind")
        )
        cleanup["host_evidence_orderly"] = host_evidence_orderly
        cleanup["host_early_stop_source"] = early_stop_source
        cleanup["host_orderly_self_exit"] = bool(
            cleanup.get("host_exit_kind") == "self_exit"
            and host_evidence_orderly
        )
        if self.host_pid is not None and not host_evidence_orderly:
            self._record_error(
                "the host bridge evidence records neither a completed hold nor "
                "an accepted intentional stop with an orderly shutdown"
            )
        cleanup["passed"] = bool(
            cleanup["container_removed"]
            and cleanup["host_stopped"]
            and host_evidence_orderly
            and not leftovers.get("session_containers")
            and not leftovers.get("session_processes")
        )
        self.status = (
            RuntimeSessionStatus.STOPPED
            if previous is RuntimeSessionStatus.READY
            else RuntimeSessionStatus.FAILED
        )
        self._stop_evidence = self._evidence(cleanup=cleanup)
        self._write_evidence()
        return self._stop_evidence

    def _orderly_host_verdict(
        self,
        host_evidence: Mapping[str, object],
        *,
        host_exit_kind: object,
    ) -> tuple[bool, str | None]:
        """Judge the host teardown from the BRIDGE'S OWN evidence only.

        Accepted terminal states, in order, each additionally requiring the
        bridge's four orderly shutdown flags (rclpy + world + timeline + app):

        * ``hold_completed``: the bridge finished its own bounded hold (the
          normal end of its lifetime);
        * ``bridge_early_stop_accepted``: A4's additive ``early_stop_accepted``
          attestation of an accepted early stop, cross-checked against this
          session's own teardown record - the process must have ended on THIS
          session's SIGINT and the bridge must record the same SIGINT stop.
          Applied controller commands remain audit evidence, not a cleanup
          failure or proof of skill-operation dispatch.

        There is NO session-side bypass and no marker-string shortcut: a
        SIGINT-ed bridge without the attestation, an attestation that does not
        match the recorded SIGINT stop, a generic Kit cancellation and a
        forced SIGKILL all leave the cleanup failing.
        """
        if self.host_pid is None:
            return True, "not_started"
        if not host_evidence.get("shutdown_orderly"):
            return False, None
        if host_evidence.get("hold_completed"):
            return True, "hold_completed"
        if not host_evidence.get("early_stop_accepted"):
            return False, None
        # A4's attestation must also agree with what actually happened: THIS
        # session ended the process with its own bound SIGINT, and the bridge
        # recorded that same SIGINT stop. Generic Kit cancellation never counts.
        if host_evidence.get("stopped") != "sigint_orderly_stop":
            self._record_error(
                "the host bridge attested an accepted early stop but its "
                "evidence does not record the SIGINT stop "
                f"(stopped={host_evidence.get('stopped')!r}); refusing the "
                "attestation"
            )
            return False, None
        if host_exit_kind != "sigint":
            self._record_error(
                "the host bridge attested an accepted early stop but this "
                f"session did not interrupt it with SIGINT "
                f"(host_exit_kind={host_exit_kind!r}); refusing the attestation"
            )
            return False, None
        return True, "bridge_early_stop_accepted"

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
                timeout_s=CONTAINER_LOG_TIMEOUT_S,
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
        """Bounded host teardown that never waits out the bridge's hold.

        The bridge writes its evidence only when ITS OWN hold completes (or
        when it is interrupted: it catches ``KeyboardInterrupt`` itself and
        still writes its ``stopped``/cleanup evidence), so an early stop must
        not wait for the rest of the hold. Policy, each step bounded:

        1. a short reap wait for a host that is already exiting;
        2. if the bridge's own evidence proves its hold completed, it is only
           inside its slow native Kit teardown: give it
           ``host_teardown_grace_s`` and keep its orderly evidence;
        3. otherwise SIGINT (the bridge turns it into an evidenced
           ``sigint_orderly_stop``) and then SIGKILL, each bounded.

        ``cleanup["host_exit_kind"]`` records exactly how the process ended
        (``self_exit``/``sigint``/``sigkill``/``undying``/``not_started``) so
        the verdict never pretends an interrupted stop completed the hold.
        """
        handle = self.host_handle
        if handle is None:
            cleanup["host_stopped"] = True
            cleanup["host_exit_kind"] = "not_started"
            return
        if self._host_finished_orderly():
            exit_code = self._await_exit(handle, self.config.host_teardown_grace_s)
        else:
            exit_code = self._await_exit(handle, MULTI_ROBOT_HOST_SELF_EXIT_GRACE_S)
        if exit_code is not None:
            cleanup["host_stopped"] = True
            cleanup["host_exit_kind"] = "self_exit"
            cleanup["host_exit_code"] = exit_code
            return
        handle.send_signal(2)  # SIGINT: bounded intentional stop
        exit_code = self._await_exit(handle, MULTI_ROBOT_HOST_INT_GRACE_S)
        if exit_code is not None:
            cleanup["host_stopped"] = True
            cleanup["host_exit_kind"] = "sigint"
            cleanup["host_exit_code"] = exit_code
            return
        handle.send_signal(9)  # SIGKILL
        exit_code = self._await_exit(handle, MULTI_ROBOT_HOST_KILL_GRACE_S)
        cleanup["host_stopped"] = exit_code is not None
        cleanup["host_exit_kind"] = "sigkill" if exit_code is not None else "undying"
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
        """True when the bridge wrote evidence of a completed, error-free hold."""
        summary = self._host_evidence_summary()
        return bool(summary.get("hold_completed")) and summary.get("error") is None

    def _host_evidence_summary(self) -> dict[str, object]:
        """Read the bridge's own evidence document into a bounded summary.

        ``shutdown_orderly`` is the four orderly shutdown flags the bridge
        writes from its own teardown path; ``intentional_stop`` is the
        bridge's own marker for an interrupted (SIGINT) stop, which it
        records without completing its hold - an intentional stop is
        evidenced, a missing one is not.
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
        robots = document.get("robots")
        stopped = document.get("stopped")
        commands_applied = _applied_command_messages(document)
        shutdown_orderly = bool(
            host_cleanup.get("rclpy_shutdown") is True
            and host_cleanup.get("world_stopped") is True
            and host_cleanup.get("timeline_stopped") is True
            and (
                host_cleanup.get("app_close_initiated")
                or host_cleanup.get("app_closed")
            )
        )
        return {
            "path": str(path),
            "passed": document.get("passed") is True,
            "ready": document.get("ready") is True,
            "hold_completed": document.get("hold_completed") is True,
            "error": document.get("error"),
            "stopped": stopped,
            # The bridge's own SIGINT marker only: a generic Kit cancellation
            # is NOT an intentional stop.
            "intentional_stop": stopped == "sigint_orderly_stop",
            "commands_applied": commands_applied,
            # A4's additive attestation of an accepted early stop (ready
            # bridge + bound SIGINT + orderly cleanup), honored when present.
            "early_stop_accepted": document.get("early_stop_accepted") is True,
            "shutdown_orderly": shutdown_orderly,
            "robots": sorted(str(robot_id) for robot_id in robots)
            if isinstance(robots, dict)
            else [],
            "rclpy_shutdown": host_cleanup.get("rclpy_shutdown") is True,
            "world_stopped": host_cleanup.get("world_stopped") is True,
            "timeline_stopped": host_cleanup.get("timeline_stopped") is True,
            "app_close_initiated": bool(
                host_cleanup.get("app_close_initiated")
                or host_cleanup.get("app_closed")
            ),
        }

    def _scan_leftovers(self) -> dict[str, object]:
        """Bounded leftover scan scoped to THIS session's own resources.

        The container is looked up by the exact name this session generated
        and the host by the unique session directory on its command line, so a
        foreign container, another live session or a pre-existing leftover is
        never reported as (or treated as) this session's leftover.
        """
        containers: tuple[str, ...] = ()
        hosts: tuple[str, ...] = ()
        for _ in range(LEFTOVER_SCAN_ATTEMPTS):
            if self.container_name is None:
                containers = ()
            else:
                listed = self.runner.run(
                    [
                        "docker",
                        "ps",
                        "-a",
                        "--filter",
                        f"name={self.container_name}",
                        "--format",
                        "{{.Names}}",
                    ],
                    timeout_s=CONTAINER_LOG_TIMEOUT_S,
                )
                containers = tuple(
                    line.strip()
                    for line in listed.stdout.splitlines()
                    if line.strip()
                )
            hosts = self.runner.processes_matching(str(self.session_dir))
            if not containers and not hosts:
                break
            self._sleep(LEFTOVER_SCAN_PAUSE_S)
        return {
            "scope": str(self.session_dir),
            "session_containers": list(containers),
            "session_processes": list(hosts),
        }

    # ------------------------------------------------------ context manager

    def __enter__(self) -> "MultiRobotIsaacMoveItSession":
        """Start the stack and require every plan robot ready."""
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Always stop (idempotently); never suppress the caller's error."""
        self.stop()
        return False
