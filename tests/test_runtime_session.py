"""Offline tests for the R4.2 runtime session and operation contracts.

Everything runs against a Fake ``ProcessRunner``/clock: no Docker, no Isaac,
no GPU, no rclpy, and no motion is ever started. The fake runner replays the
proven command topology (model audit, profile generation with
``--container-mesh``, host bridge launch with the exact marker, one unique
rootless container, ``docker exec`` readiness child) and the tests assert the
exact command sequences, bounds, marker discipline, readiness gates, zero
motion during start, partial-failure cleanup, idempotent stop, and the
strict operation/result schemas. AST scans prove the production modules
import no tools/smoke/STAP modules and keep ROS/Isaac imports lazy.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from rocobrick import ros_session_entry, runtime_session
from rocobrick.isaac_backend import (
    OPERATION_KEYS,
    OPERATION_RESULT_KEYS,
    OPERATION_RESULT_SCHEMA,
    OPERATION_SCHEMA,
    OperationContractError,
    OperationKind,
    OperationRejected,
    RuntimeOperation,
    RuntimeOperationResult,
    joint_operation,
    joint_target_from_payload,
    link6_pose_operation,
    link6_pose_target_from_payload,
    readiness_operation,
    target_for_operation,
)
from rocobrick.moveit_contract import JointTarget, Link6PoseTarget
from rocobrick.moveit_execution import ExecutionResult, ExecutionStatus
from rocobrick.ros_moveit_runtime import ReadinessSnapshot
from rocobrick.runtime_session import (
    CONTAINER_MESH_PROFILE,
    READY_MARKER,
    IsaacMoveItSession,
    RuntimeReadinessEvidence,
    RuntimeSessionConfig,
    RuntimeSessionError,
    RuntimeSessionEvidence,
    RuntimeSessionStatus,
    SubprocessRunner,
)

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src" / "rocobrick"

ADMITTED_POSITIONS = {
    "joint1": 0.0,
    "joint2": 0.5,
    "joint3": -0.5,
    "joint4": 0.3,
    "joint5": 0.0,
    "joint6": 0.0,
}

READY_SNAPSHOT = {
    "clock_increasing": True,
    "isaac_state_complete": True,
    "joint_state_complete": True,
    "broadcaster_match": True,
    "controllers_active": True,
    "arm_action_ready": True,
    "move_group_action_ready": True,
    "compute_fk_ready": True,
}


def readiness_result_document(request: dict) -> dict:
    """A passing zero-motion READINESS result answering ``request``."""
    return {
        "schema": OPERATION_RESULT_SCHEMA,
        "request_id": request["request_id"],
        "kind": request["kind"],
        "passed": True,
        "execution": {
            "motion_sent": False,
            "goals_sent": 0,
            "command_messages": 0,
        },
        "evidence": {
            "waited_ok": True,
            "passed": True,
            "motion_sent": False,
            "snapshot": dict(READY_SNAPSHOT),
        },
        "error": None,
    }


def echo_result_document(request: dict, *, passed: bool = True) -> dict:
    """A passing motion result answering ``request`` (no real motion)."""
    return {
        "schema": OPERATION_RESULT_SCHEMA,
        "request_id": request["request_id"],
        "kind": request["kind"],
        "passed": passed,
        "execution": {
            "status": "SUCCEEDED" if passed else "FAILED",
            "is_success": passed,
            "moveit_error_code": 1 if passed else 0,
            "planned_waypoints": 6 if passed else 0,
            "command_messages": 4 if passed else 0,
            "final_errors": {},
        },
        "evidence": {"waited_ok": True, "passed": True},
        "error": None if passed else "the goal did not pass the execution gates",
    }


class FakeClock:
    """Fake monotonic clock; sleeps advance time deterministically."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeHandle:
    """Fake long-running process handle for the host bridge."""

    def __init__(self, pid: int, runner: "FakeRunner") -> None:
        self.pid = pid
        self._runner = runner
        self.exit_code: int | None = None
        self.signals: list[int] = []

    def poll(self) -> int | None:
        self._runner._maybe_self_exit(self)
        return self.exit_code

    def wait(self, timeout_s: float | None = None) -> int:
        self._runner._maybe_self_exit(self)
        if self.exit_code is None:
            raise TimeoutError(f"process {self.pid} is still running")
        return self.exit_code


    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)
        if signum == 9:
            if not self._runner.host_ignores_kill:
                self.exit_code = -9
        elif signum == 2:
            if not self._runner.host_ignores_int:
                self.exit_code = 0


class FakeRunner:
    """Fake ProcessRunner replaying the proven command topology."""

    def __init__(
        self,
        *,
        session_dir: Path,
        marker: bool = True,
        host_dies: bool = False,
        host_self_exit_after: float | None = None,
        host_ignores_int: bool = False,
        host_ignores_kill: bool = False,
        docker_run_fails: bool = False,
        docker_stop_fails: bool = False,
        child_writes_no_result: bool = False,
        child_times_out: bool = False,
        operation_child_times_out: bool = False,
        removal_verify_flaky: int = 0,
        host_writes_evidence: bool = True,
        child_result=None,
        leftover_containers: tuple[str, ...] = (),
        leftover_processes: tuple[str, ...] = (),
    ) -> None:
        self.session_dir = session_dir
        self.marker = marker
        self.host_dies = host_dies
        self.host_self_exit_after = host_self_exit_after
        self.host_ignores_int = host_ignores_int
        self.host_ignores_kill = host_ignores_kill
        self.docker_run_fails = docker_run_fails
        self.docker_stop_fails = docker_stop_fails
        self.child_writes_no_result = child_writes_no_result
        self.child_times_out = child_times_out
        self.operation_child_times_out = operation_child_times_out
        self.removal_verify_flaky = removal_verify_flaky
        self.host_writes_evidence = host_writes_evidence
        self.child_result = child_result
        self.leftover_containers = leftover_containers
        self.leftover_processes = leftover_processes
        self.launches: list[tuple[list[str], dict[str, str], Path]] = []
        self.commands: list[tuple[list[str], float | None]] = []
        self.execs: list[list[str]] = []
        self.container_names: list[str] = []
        self.clock = FakeClock()
        self._next_pid = 4200
        self._handle_by_pid: dict[int, FakeHandle] = {}

    # ---------------------------------------------------------- processes

    def launch(self, argv, *, env, stdout_path, stderr_path):
        self.launches.append((list(argv), dict(env), Path(stdout_path)))
        self._next_pid += 1
        handle = FakeHandle(self._next_pid, self)
        self._handle_by_pid[handle.pid] = handle
        if self.host_dies:
            handle.exit_code = 1
        Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
        if self.host_writes_evidence:
            Path(stdout_path).with_name("host-bridge-evidence.json").write_text(
                json.dumps(
                    {
                        "schema": "rocobrick-host-bridge-smoke/v1",
                        "ready": True,
                        "passed": True,
                        "hold_completed": True,
                        "error": None,
                        "cleanup": {
                            "rclpy_shutdown": True,
                            "world_stopped": True,
                            "timeline_stopped": True,
                            "app_close_initiated": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
        if self.marker:
            Path(stdout_path).write_text(
                json.dumps({"marker": READY_MARKER}) + "\n", encoding="utf-8"
            )
        else:
            Path(stdout_path).write_text("booting\n", encoding="utf-8")
        return handle

    def _maybe_self_exit(self, handle: FakeHandle) -> None:
        if (
            self.host_self_exit_after is not None
            and handle.exit_code is None
            and self.clock.now >= self.host_self_exit_after
        ):
            handle.exit_code = 0

    def process_alive(self, pid: int) -> bool:
        handle = self._handle_by_pid.get(pid)
        return False if handle is None else handle.poll() is None

    def processes_matching(self, pattern: str) -> tuple[str, ...]:
        return self.leftover_processes

    # ---------------------------------------------------------- commands

    def run(self, argv, *, env=None, timeout_s=None):
        argv = [str(part) for part in argv]
        self.commands.append((list(argv), timeout_s))
        joined = " ".join(argv)
        if "generate_dds_profiles.py" in joined:
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / CONTAINER_MESH_PROFILE).write_text("<mesh/>\n")
            (output_dir / "fastdds-host.xml").write_text("<host/>\n")
            return self._outcome(
                argv,
                0,
                json.dumps({
                    "schema": "rocobrick-dds-profiles/v2",
                    "profiles": {"container-mesh": {"role": "container-client-mesh"}},
                }),
                "",
            )
        if "piper_l_model_audit.py" in joined:
            Path(argv[argv.index("--output") + 1]).write_text("{}\n")
            return self._outcome(argv, 0, "audit-ok\n", "")
        if argv[0] == "docker":
            return self._docker(argv, joined)
        return self._outcome(argv, 0, "", "")

    def _outcome(self, argv, returncode, stdout, stderr, timed_out=False):
        return runtime_session.RunOutcome(
            argv=tuple(argv),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )

    def _docker(self, argv: list[str], joined: str):
        if argv[1] == "run":
            if self.docker_run_fails:
                return self._outcome(argv, 125, "", "docker run failed")
            name = argv[argv.index("--name") + 1]
            self.container_names.append(name)
            return self._outcome(argv, 0, "fake-container-id\n", "")
        if argv[1] == "exec":
            self.execs.append(list(argv))
            if self.child_times_out or (
                self.operation_child_times_out and len(self.execs) > 1
            ):
                return self._outcome(argv, None, "", "", timed_out=True)
            return self._run_child(argv)
        if argv[1] == "stop":
            if self.docker_stop_fails:
                return self._outcome(argv, 1, "", "stop failed")
            return self._outcome(argv, 0, argv[-1] + "\n", "")
        if argv[1] == "rm":
            return self._outcome(argv, 0, "", "")
        if argv[1] == "ps":
            filter_value = argv[argv.index("--filter") + 1]
            exact = (
                self.container_names
                and filter_value == f"name={self.container_names[-1]}"
            )
            if exact:
                if self.removal_verify_flaky > 0:
                    self.removal_verify_flaky -= 1
                    return self._outcome(
                        argv, 0, self.container_names[-1] + "\n", ""
                    )
                return self._outcome(argv, 0, "", "")
            if filter_value.endswith("-"):
                return self._outcome(
                    argv,
                    0,
                    "".join(name + "\n" for name in self.leftover_containers),
                    "",
                )
            return self._outcome(argv, 0, "", "")
        return self._outcome(argv, 0, "", "")

    def _run_child(self, argv: list[str]):
        request_container = argv[-2]
        result_container = argv[-1]
        assert request_container.startswith("/session/")
        assert result_container.startswith("/session/")
        request_path = self.session_dir / request_container[len("/session/"):]
        result_path = self.session_dir / result_container[len("/session/"):]
        if self.child_writes_no_result:
            return self._outcome(argv, 1, "", "child exploded")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        RuntimeOperation.parse(request)
        document = (
            self.child_result(request)
            if self.child_result is not None
            else readiness_result_document(request)
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(document), encoding="utf-8")
        return self._outcome(argv, 0 if document["passed"] else 1, "summary\n", "")


def make_session(tmp_path, runner: FakeRunner, **config_overrides):
    """Build a session wired to the fake runner and fake clock."""
    config = RuntimeSessionConfig(
        repo_root=ROOT,
        startup_timeout_s=config_overrides.pop("startup_timeout_s", 6.0),
        operation_timeout_s=config_overrides.pop("operation_timeout_s", 10.0),
        shutdown_timeout_s=config_overrides.pop("shutdown_timeout_s", 5.0),
        **config_overrides,
    )
    session = IsaacMoveItSession(
        config,
        runner=runner,
        session_dir=tmp_path / "session",
        clock=runner.clock.clock,
        sleep=runner.clock.sleep,
        base_env={"PATH": "/usr/bin"},
    )
    runner._handle_by_pid = {}
    return session


# ---------------------------------------------------------------------------
# Contracts: status vocabulary, config validation, evidence documents
# ---------------------------------------------------------------------------


def test_status_vocabulary_is_exact():
    assert {member.value for member in RuntimeSessionStatus} == {
        "NOT_STARTED",
        "STARTING",
        "READY",
        "FAILED",
        "STOPPING",
        "STOPPED",
    }


def test_config_defaults_and_validation(tmp_path):
    config = RuntimeSessionConfig(repo_root=tmp_path)
    assert config.ros_image == "roco-piper-ros:humble"
    assert config.ros_domain_id == 73
    assert config.startup_timeout_s == 120.0
    assert config.operation_timeout_s == 180.0
    assert config.shutdown_timeout_s == 300.0
    with pytest.raises(ValueError):
        RuntimeSessionConfig(repo_root=tmp_path / "missing")
    with pytest.raises(ValueError):
        RuntimeSessionConfig(repo_root=tmp_path, ros_image="")
    with pytest.raises(ValueError):
        RuntimeSessionConfig(repo_root=tmp_path, ros_domain_id=-1)
    with pytest.raises(ValueError):
        RuntimeSessionConfig(repo_root=tmp_path, startup_timeout_s=0.0)
    with pytest.raises(ValueError):
        RuntimeSessionConfig(repo_root=tmp_path, operation_timeout_s=-1.0)
    with pytest.raises(ValueError):
        RuntimeSessionConfig(repo_root=tmp_path, shutdown_timeout_s=0.0)


def test_readiness_evidence_passed_requires_every_gate():
    full = RuntimeReadinessEvidence(
        host_marker=True,
        container_started=True,
        moveit_ready=True,
        controllers_active=True,
        adapter_ready=True,
    )
    assert full.passed is True
    for field in (
        "host_marker",
        "container_started",
        "moveit_ready",
        "controllers_active",
        "adapter_ready",
    ):
        kwargs = {name: True for name in (
            "host_marker",
            "container_started",
            "moveit_ready",
            "controllers_active",
            "adapter_ready",
        )}
        kwargs[field] = False
        assert RuntimeReadinessEvidence(**kwargs).passed is False


def test_session_evidence_serializable_shape():
    evidence = RuntimeSessionEvidence(
        status=RuntimeSessionStatus.STOPPED,
        readiness=RuntimeReadinessEvidence(True, True, True, True, True),
        host_pid=7,
        container_name="c",
        profile_report={"schema": "rocobrick-dds-profiles/v2"},
        host_evidence_path="/tmp/host.json",
        log_paths=("/tmp/a.log", "/tmp/b.log"),
        cleanup={"passed": True},
        errors=("boom",),
    )
    document = evidence.serializable()
    assert set(document) == {
        "schema",
        "status",
        "readiness",
        "host_pid",
        "container_name",
        "profile_report",
        "host_evidence_path",
        "log_paths",
        "cleanup",
        "errors",
    }
    assert document["schema"] == "rocobrick-runtime-session/v1"
    assert document["status"] == "STOPPED"
    assert json.loads(json.dumps(document)) == document


# ---------------------------------------------------------------------------
# isaac_backend: strict operation/result schemas and builders
# ---------------------------------------------------------------------------


def test_operation_serialize_has_exact_schema():
    operation = readiness_operation(timeout_s=30.0, request_id="r1")
    document = operation.serialize()
    assert set(document) == set(OPERATION_KEYS)
    assert document["schema"] == OPERATION_SCHEMA
    assert document["kind"] == "READINESS"
    assert document["payload"] == {}
    assert document["timeout_s"] == 30.0
    assert document["request_id"] == "r1"


def test_operation_parse_roundtrip_and_strictness():
    operation = joint_operation(
        JointTarget(positions=dict(ADMITTED_POSITIONS), goal_tolerance=0.01),
        45.0,
        "j1",
    )
    parsed = RuntimeOperation.parse(operation.serialize())
    assert parsed == operation
    for broken in (
        {"schema": "other/v1", "kind": "READINESS", "payload": {},
         "timeout_s": 1.0, "request_id": "x"},
        {"kind": "READINESS", "payload": {}, "timeout_s": 1.0, "request_id": "x"},
        {"schema": OPERATION_SCHEMA, "kind": "READINESS", "payload": {},
         "timeout_s": 1.0, "request_id": "x", "extra": True},
        {"schema": OPERATION_SCHEMA, "kind": "NOT_A_KIND", "payload": {},
         "timeout_s": 1.0, "request_id": "x"},
        {"schema": OPERATION_SCHEMA, "kind": "READINESS", "payload": {},
         "timeout_s": 0.0, "request_id": "x"},
        {"schema": OPERATION_SCHEMA, "kind": "READINESS", "payload": {},
         "timeout_s": 1.0, "request_id": ""},
    ):
        with pytest.raises(OperationContractError):
            RuntimeOperation.parse(broken)
    with pytest.raises(OperationContractError):
        RuntimeOperation.parse("not a mapping")


def test_result_parse_roundtrip_and_strictness():
    result = RuntimeOperationResult(
        request_id="r1",
        kind=OperationKind.READINESS,
        passed=False,
        execution={"motion_sent": False},
        evidence={"snapshot": {}},
        error="nope",
    )
    parsed = RuntimeOperationResult.parse(result.serialize())
    assert parsed == result
    document = result.serialize()
    assert set(document) == set(OPERATION_RESULT_KEYS)
    assert document["schema"] == OPERATION_RESULT_SCHEMA
    for broken in (
        {**document, "schema": "wrong/v9"},
        {key: value for key, value in document.items() if key != "error"},
        {**document, "surplus": 1},
        {**document, "passed": "yes"},
        {**document, "execution": []},
        {**document, "error": 5},
        {**document, "request_id": ""},
    ):
        with pytest.raises(OperationContractError):
            RuntimeOperationResult.parse(broken)


def test_joint_builder_admits_then_serializes_and_rejects_early():
    target = JointTarget(positions=dict(ADMITTED_POSITIONS), goal_tolerance=0.01)
    operation = joint_operation(target, 60.0, "j2")
    assert operation.kind is OperationKind.JOINT_TARGET
    assert operation.payload["positions"] == ADMITTED_POSITIONS
    rebuilt = joint_target_from_payload(operation.payload)
    assert rebuilt == target
    out_of_limits = JointTarget(
        positions={**ADMITTED_POSITIONS, "joint2": 4.0}, goal_tolerance=0.01
    )
    with pytest.raises(OperationRejected):
        joint_operation(out_of_limits, 60.0, "j3")


def test_link6_builder_admits_then_serializes_and_rejects_early():
    target = Link6PoseTarget(
        frame_id="base_link",
        position=(0.2, 0.0, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.015,
    )
    operation = link6_pose_operation(target, 60.0, "p1")
    assert operation.kind is OperationKind.LINK6_POSE
    rebuilt = link6_pose_target_from_payload(operation.payload)
    assert rebuilt == target
    bad_frame = Link6PoseTarget(
        frame_id="world",
        position=(0.2, 0.0, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(OperationRejected):
        link6_pose_operation(bad_frame, 60.0, "p2")


def test_target_for_operation_rechecks_admission_and_refuses_readiness():
    with pytest.raises(OperationContractError):
        target_for_operation(
            readiness_operation(timeout_s=10.0, request_id="r")
        )
    forged = RuntimeOperation(
        kind=OperationKind.JOINT_TARGET,
        payload={
            "positions": {**ADMITTED_POSITIONS, "joint3": 1.0},
            "goal_tolerance": 0.01,
        },
        timeout_s=10.0,
        request_id="forged",
    )
    with pytest.raises(OperationRejected):
        target_for_operation(forged)
    good = target_for_operation(
        joint_operation(
            JointTarget(positions=dict(ADMITTED_POSITIONS)), 10.0, "good"
        )
    )
    assert isinstance(good, JointTarget)


# ---------------------------------------------------------------------------
# Fake runner: exact start sequence, bounds, marker, readiness, no motion
# ---------------------------------------------------------------------------


def test_start_runs_exact_command_sequence_and_reaches_ready(tmp_path):
    runner = FakeRunner(session_dir=tmp_path / "session")
    session = make_session(tmp_path, runner)
    evidence = session.start()
    assert evidence.passed is True
    assert (
        evidence.host_marker
        and evidence.container_started
        and evidence.moveit_ready
        and evidence.controllers_active
        and evidence.adapter_ready
    )
    assert session.status is RuntimeSessionStatus.READY
    # Exactly one host launch, through isaac-ros-env.sh with bounded hold.
    assert len(runner.launches) == 1
    argv, env, log_path = runner.launches[0]
    joined = " ".join(argv)
    assert "tools/isaac-ros-env.sh" in joined
    assert "isaac/p05_host_bridge.py" in joined
    assert "--hold-seconds" in argv
    # The hold is capped inside the stop grace so the bridge always
    # self-exits orderly before stop() stops waiting (shutdown=5 -> floor 30),
    # and the step bound is sized to finish inside the hold seconds.
    assert argv[argv.index("--hold-seconds") + 1] == "30.0"
    assert argv[argv.index("--hold-steps") + 1] == "110000"
    assert argv[argv.index("--time-limit") + 1] == str(6.0 + 10.0 + 120.0)
    assert env["ROS_DOMAIN_ID"] == "73"
    assert env["FASTRTPS_DEFAULT_PROFILES_FILE"].endswith(
        "dds/fastdds-host.xml"
    )
    assert log_path == tmp_path / "session" / "host-bridge.log"
    # The bounded command sequence in order: audit -> profiles -> docker run
    # -> docker exec (readiness child); every bound present.
    kinds = []
    for command, bound in runner.commands:
        text = " ".join(command)
        if "piper_l_model_audit.py" in text:
            kinds.append("audit")
            assert bound == 6.0
        elif "generate_dds_profiles.py" in text:
            kinds.append("profiles")
            assert "--container-mesh" in command
            assert command[command.index("--ros-domain-id") + 1] == "73"
            assert command[command.index("--host-port") + 1] == "17650"
            assert bound is not None
        elif text.startswith("docker run"):
            kinds.append("container")
            assert "--rm" in command and "-d" in command
            name = command[command.index("--name") + 1]
            assert name.startswith("rocobrick-runtime-session-")
            assert len(name) > len("rocobrick-runtime-session-")
            assert "use_sim_time:=true" in command[-1]
            assert f"{ROOT / 'src'}:/workspace/src:ro" in command
            assert f"{tmp_path / 'session' / 'dds'}:/dds:ro" in command
            assert (
                f"{tmp_path / 'session'}:/session" in command
            )
            assert command[command.index("-e") + 1] == "ROS_DOMAIN_ID=73"
            assert bound == 6.0
        elif text.startswith("docker exec"):
            kinds.append("exec")
            assert "rocobrick.ros_session_entry" in text
            assert bound == 10.0 + 30.0
        else:
            kinds.append(text.split(" ")[1])
    assert kinds == ["audit", "profiles", "container", "exec"]
    # Zero motion during start: the only child request is a READINESS op.
    assert len(runner.execs) == 1
    request_container_path = runner.execs[0][-2]
    request = json.loads(
        (tmp_path / "session" / request_container_path[len("/session/"):])
        .read_text(encoding="utf-8")
    )
    assert request["kind"] == "READINESS"
    # Lifecycle evidence written.
    document = json.loads(
        (tmp_path / "session" / "runtime-session.json").read_text(encoding="utf-8")
    )
    assert document["schema"] == "rocobrick-runtime-session/v1"
    assert document["status"] == "READY"
    assert all(document["readiness"].values())
    assert document["host_pid"] is not None


def test_start_requires_the_p1a_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runtime_session,
        "P1A_ARTIFACT_RELPATH",
        Path("artifacts/phase1/does-not-exist.json"),
    )
    runner = FakeRunner(session_dir=tmp_path / "session")
    session = make_session(tmp_path, runner)
    with pytest.raises(RuntimeSessionError, match="p1a_prerequisite"):
        session.start()
    assert session.status is RuntimeSessionStatus.FAILED
    assert runner.launches == []
    assert runner.commands == []
    document = json.loads(
        (tmp_path / "session" / "runtime-session.json").read_text(encoding="utf-8")
    )
    assert document["status"] == "FAILED"
    assert document["errors"]


def test_marker_timeout_stops_the_host_and_records_failure(tmp_path):
    runner = FakeRunner(
        session_dir=tmp_path / "session", marker=False, host_self_exit_after=1e9
    )
    runner.host_ignores_int = True
    runner.host_ignores_kill = True
    session = make_session(tmp_path, runner, startup_timeout_s=2.0)
    with pytest.raises(RuntimeSessionError, match="host_marker"):
        session.start()
    evidence = session.stop()
    assert evidence.status is RuntimeSessionStatus.FAILED
    assert evidence.cleanup["host_stopped"] is False
    assert evidence.cleanup["passed"] is False
    assert any("host_marker" in error for error in evidence.errors)
    assert evidence.readiness is None


def test_host_death_before_marker_fails_fast(tmp_path):
    runner = FakeRunner(
        session_dir=tmp_path / "session", host_dies=True, marker=False
    )
    session = make_session(tmp_path, runner)
    with pytest.raises(RuntimeSessionError, match="host_marker"):
        session.start()
    assert session.stop().cleanup["host_orderly_self_exit"] is True


def test_docker_run_failure_cleans_up_the_host(tmp_path):
    runner = FakeRunner(session_dir=tmp_path / "session", docker_run_fails=True)
    session = make_session(tmp_path, runner)
    with pytest.raises(RuntimeSessionError, match="container_start"):
        session.start()
    evidence = session.stop()
    assert evidence.container_name is None
    assert evidence.cleanup["container_removed"] is True
    assert evidence.cleanup["host_stopped"] is True
    assert evidence.cleanup["passed"] is True


def test_readiness_failure_fails_start_and_cleanup_still_passes(tmp_path):
    def failing_child(request):
        document = readiness_result_document(request)
        document["evidence"]["snapshot"] = {
            **READY_SNAPSHOT,
            "controllers_active": False,
        }
        document["passed"] = False
        document["error"] = "readiness phase failed"
        return document

    runner = FakeRunner(
        session_dir=tmp_path / "session", child_result=failing_child
    )
    session = make_session(tmp_path, runner)
    with pytest.raises(RuntimeSessionError, match="readiness"):
        session.start()
    assert session.status is RuntimeSessionStatus.FAILED
    evidence = session.stop()
    assert evidence.status is RuntimeSessionStatus.FAILED
    assert evidence.cleanup["container_removed"] is True
    assert evidence.cleanup["host_stopped"] is True
    assert evidence.cleanup["passed"] is True
    assert evidence.readiness is None


def test_motion_during_start_is_refused(tmp_path):
    def motion_child(request):
        document = readiness_result_document(request)
        document["evidence"]["motion_sent"] = True
        return document

    runner = FakeRunner(session_dir=tmp_path / "session", child_result=motion_child)
    session = make_session(tmp_path, runner)
    with pytest.raises(RuntimeSessionError, match="motion"):
        session.start()


def test_child_timeout_fails_start(tmp_path):
    runner = FakeRunner(session_dir=tmp_path / "session", child_times_out=True)
    session = make_session(tmp_path, runner)
    with pytest.raises(RuntimeSessionError, match="child_exec"):
        session.start()


def test_child_without_result_fails_start(tmp_path):
    runner = FakeRunner(
        session_dir=tmp_path / "session", child_writes_no_result=True
    )
    session = make_session(tmp_path, runner)
    with pytest.raises(RuntimeSessionError, match="child_exec"):
        session.start()


# ---------------------------------------------------------------------------
# stop(): orderly teardown, escalation, leftovers, idempotence
# ---------------------------------------------------------------------------


def test_stop_is_orderly_idempotent_and_verified(tmp_path):
    runner = FakeRunner(
        session_dir=tmp_path / "session",
        host_self_exit_after=2.0,
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    assert evidence.status is RuntimeSessionStatus.STOPPED
    assert evidence.cleanup["container_removed"] is True
    assert evidence.cleanup["host_stopped"] is True
    assert evidence.cleanup["host_orderly_self_exit"] is True
    assert evidence.cleanup["passed"] is True
    assert evidence.cleanup["leftovers"] == {
        "containers": [],
        "host_processes": [],
    }
    docker_commands = [
        command for command, _ in runner.commands if command[0] == "docker"
    ]
    stops = [c for c in docker_commands if c[1] == "stop"]
    assert stops and stops[0][2] == "-t" and int(stops[0][3]) <= 30
    assert any(c[1] == "ps" for c in docker_commands)
    # Idempotent: a second stop returns the same evidence without new work.
    commands_after = len(runner.commands)
    assert session.stop() is evidence
    assert len(runner.commands) == commands_after
    assert session.status is RuntimeSessionStatus.STOPPED


def test_stop_escalates_int_then_kill(tmp_path):
    runner = FakeRunner(
        session_dir=tmp_path / "session",
        host_self_exit_after=None,
        host_ignores_int=True,
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    handle = session.host_handle
    assert 2 in handle.signals and 9 in handle.signals
    assert evidence.cleanup["host_stopped"] is True
    assert evidence.cleanup["host_orderly_self_exit"] is False
    assert evidence.cleanup["passed"] is True


def test_stop_reports_undying_host_and_fails_cleanup(tmp_path):
    runner = FakeRunner(
        session_dir=tmp_path / "session",
        host_self_exit_after=None,
        host_ignores_int=True,
        host_ignores_kill=True,
        leftover_processes=("999",),
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    assert evidence.cleanup["host_stopped"] is False
    assert evidence.cleanup["passed"] is False
    assert evidence.cleanup["leftovers"]["host_processes"] == ["999"]
    assert any("did not stop" in error for error in evidence.errors)


def test_stop_waits_out_the_inflight_container_removal(tmp_path):
    """LIVE regression: a stopped --rm container is still listed for a few
    seconds ("Removal In Progress"); the removal verdict must retry instead
    of failing the cleanup."""
    runner = FakeRunner(
        session_dir=tmp_path / "session",
        host_self_exit_after=1.0,
        removal_verify_flaky=2,
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    assert evidence.cleanup["container_removed"] is True
    assert evidence.cleanup["passed"] is True
    ps_calls = [
        command for command, _ in runner.commands if command[:2] == ["docker", "ps"]
    ]
    assert len(ps_calls) == 4  # 3 removal-verification polls + 1 leftover scan


def test_stop_fails_when_the_container_never_disappears(tmp_path):
    runner = FakeRunner(
        session_dir=tmp_path / "session",
        host_self_exit_after=1.0,
        removal_verify_flaky=99,
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    assert evidence.cleanup["container_removed"] is False
    assert evidence.cleanup["passed"] is False
    assert any("could not be removed" in error for error in evidence.errors)


def test_stop_records_the_reaped_host_exit_code_after_slow_teardown(tmp_path):
    """LIVE regression: the Kit teardown can outlast a single ps answer; the
    stopped verdict and the exit code must come from the reaped process."""
    runner = FakeRunner(
        session_dir=tmp_path / "session",
        host_self_exit_after=None,
        host_ignores_int=True,
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    handle = session.host_handle
    assert 2 in handle.signals and 9 in handle.signals
    assert evidence.cleanup["host_stopped"] is True
    assert evidence.cleanup["host_exit_code"] == -9
    assert evidence.cleanup["passed"] is True


def test_stop_requires_orderly_host_evidence(tmp_path):
    """A stopped host WITHOUT its own orderly evidence must not count as a
    passing cleanup: empty leftovers never substitute for the verdict."""
    runner = FakeRunner(
        session_dir=tmp_path / "session",
        host_self_exit_after=1.0,
        host_writes_evidence=False,
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    assert evidence.cleanup["host_stopped"] is True
    assert evidence.cleanup["host_orderly_self_exit"] is True
    assert evidence.cleanup["host_evidence_orderly"] is False
    assert evidence.cleanup["passed"] is False
    assert any("orderly stop" in error for error in evidence.errors)


def test_stop_gives_the_finished_bridge_time_to_exit(tmp_path):
    """LIVE regression: a bridge that wrote its passed evidence and is only
    inside the slow native Kit teardown must never be INT/KILLed; the stop
    simply reaps it once the native teardown finishes."""
    runner = FakeRunner(
        session_dir=tmp_path / "session",
        # Exits at fake-clock 16: after the initial 15s reap grace but inside
        # the extended stop budget (shutdown_timeout_s=5 here).
        host_self_exit_after=16.0,
        host_writes_evidence=True,
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    handle = session.host_handle
    assert handle.signals == []  # a finishing bridge is never signalled
    assert evidence.cleanup["host_stopped"] is True
    assert evidence.cleanup["host_orderly_self_exit"] is True
    assert evidence.cleanup["host_exit_code"] == 0
    assert evidence.cleanup["passed"] is True
    assert evidence.host_evidence_path is not None


def test_context_manager_starts_and_always_stops(tmp_path):
    runner = FakeRunner(session_dir=tmp_path / "session")
    with IsaacMoveItSession(
        RuntimeSessionConfig(
            repo_root=ROOT,
            startup_timeout_s=6.0,
            operation_timeout_s=10.0,
            shutdown_timeout_s=5.0,
        ),
        runner=runner,
        session_dir=tmp_path / "session",
        clock=runner.clock.clock,
        sleep=runner.clock.sleep,
        base_env={"PATH": "/usr/bin"},
    ) as session:
        assert session.status is RuntimeSessionStatus.READY
    assert session.status is RuntimeSessionStatus.STOPPED
    assert session._stop_evidence is not None


def test_context_manager_stops_after_operation_error(tmp_path):
    runner = FakeRunner(
        session_dir=tmp_path / "session", operation_child_times_out=True
    )
    session = make_session(tmp_path, runner)
    with pytest.raises(RuntimeSessionError):
        with session:
            request = tmp_path / "request.json"
            request.write_text(json.dumps(joint_request_document()))
            session.run_operation(
                request, tmp_path / "session" / "operation-result.json"
            )
    assert session.status is RuntimeSessionStatus.FAILED
    assert session._stop_evidence.cleanup["passed"] is True


# ---------------------------------------------------------------------------
# run_operation: request/result validation through the fake child
# ---------------------------------------------------------------------------


def joint_request_document(request_id="op-1"):
    return {
        "schema": OPERATION_SCHEMA,
        "kind": "JOINT_TARGET",
        "payload": {
            "positions": dict(ADMITTED_POSITIONS),
            "goal_tolerance": 0.01,
        },
        "timeout_s": 10.0,
        "request_id": request_id,
    }


def test_run_operation_requires_ready(tmp_path):
    runner = FakeRunner(session_dir=tmp_path / "session")
    session = make_session(tmp_path, runner)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(joint_request_document()))
    with pytest.raises(RuntimeSessionError, match="READY"):
        session.run_operation(request, tmp_path / "session" / "result.json")


def test_run_operation_validates_request_and_delegates(tmp_path):
    runner = FakeRunner(session_dir=tmp_path / "session")
    session = make_session(tmp_path, runner)
    session.start()
    request = tmp_path / "outside" / "request.json"
    request.parent.mkdir(parents=True)
    request.write_text(json.dumps(joint_request_document()))
    result = session.run_operation(
        request, tmp_path / "session" / "operation-result.json"
    )
    assert result["passed"] is True
    assert result["request_id"] == "op-1"
    assert result["kind"] == "JOINT_TARGET"
    # The request was copied into the mounted session directory and the child
    # received container paths under /session.
    copied = tmp_path / "session" / "operation-op-1-request.json"
    assert copied.is_file()
    assert len(runner.execs) == 2
    assert runner.execs[1][-2] == "/session/operation-op-1-request.json"
    assert runner.execs[1][-1] == "/session/operation-result.json"
    assert runner.execs[1][1] == "exec"
    assert session.last_result.passed is True
    assert session.status is RuntimeSessionStatus.READY


def test_run_operation_rejects_malformed_request(tmp_path):
    runner = FakeRunner(session_dir=tmp_path / "session")
    session = make_session(tmp_path, runner)
    session.start()
    request = tmp_path / "bad.json"
    request.write_text('{"schema": "wrong"}')
    with pytest.raises(RuntimeSessionError, match="request schema invalid"):
        session.run_operation(
            request, tmp_path / "session" / "operation-result.json"
        )
    assert session.status is RuntimeSessionStatus.READY


def test_run_operation_refuses_readiness_kind(tmp_path):
    runner = FakeRunner(session_dir=tmp_path / "session")
    session = make_session(tmp_path, runner)
    session.start()
    request = tmp_path / "readiness.json"
    request.write_text(
        json.dumps(readiness_operation(timeout_s=5.0, request_id="r").serialize())
    )
    with pytest.raises(RuntimeSessionError, match="READINESS"):
        session.run_operation(
            request, tmp_path / "session" / "operation-result.json"
        )


def test_run_operation_rejects_result_schema_mismatch(tmp_path):
    def broken_child(request):
        if request["kind"] == "READINESS":
            return readiness_result_document(request)
        document = echo_result_document(request)
        document["schema"] = "wrong-result/v1"
        return document

    runner = FakeRunner(session_dir=tmp_path / "session", child_result=broken_child)
    session = make_session(tmp_path, runner)
    session.start()
    request = tmp_path / "request.json"
    request.write_text(json.dumps(joint_request_document()))
    with pytest.raises(RuntimeSessionError, match="result schema invalid"):
        session.run_operation(
            request, tmp_path / "session" / "operation-result.json"
        )
    assert session.status is RuntimeSessionStatus.FAILED


def test_run_operation_rejects_mismatched_answer(tmp_path):
    def liar_child(request):
        if request["kind"] == "READINESS":
            return readiness_result_document(request)
        document = echo_result_document(request)
        document["request_id"] = "someone-else"
        return document

    runner = FakeRunner(session_dir=tmp_path / "session", child_result=liar_child)
    session = make_session(tmp_path, runner)
    session.start()
    request = tmp_path / "request.json"
    request.write_text(json.dumps(joint_request_document()))
    with pytest.raises(RuntimeSessionError, match="does not answer"):
        session.run_operation(
            request, tmp_path / "session" / "operation-result.json"
        )


def test_failed_operation_result_is_returned_not_raised(tmp_path):
    def failing_child(request):
        if request["kind"] == "READINESS":
            return readiness_result_document(request)
        return echo_result_document(request, passed=False)

    runner = FakeRunner(session_dir=tmp_path / "session", child_result=failing_child)
    session = make_session(tmp_path, runner)
    session.start()
    request = tmp_path / "request.json"
    request.write_text(json.dumps(joint_request_document()))
    result = session.run_operation(
        request, tmp_path / "session" / "operation-result.json"
    )
    assert result["passed"] is False
    assert result["execution"]["status"] == "FAILED"
    assert session.status is RuntimeSessionStatus.FAILED


# ---------------------------------------------------------------------------
# SubprocessRunner protocol surface (offline: process_alive on a dead PID)
# ---------------------------------------------------------------------------


def test_subprocess_runner_dead_pid_is_not_alive():
    runner = SubprocessRunner()
    assert runner.process_alive(-1) is False
    assert runner.processes_matching("definitely-not-a-process-name-xyz") == ()


def test_subprocess_runner_handle_reaps_through_the_protocol_keyword(tmp_path):
    """LIVE regression: the raw Popen takes timeout=, the protocol fixes
    timeout_s=; the adapter must reap so the host exit code is recorded."""
    import sys
    import time

    runner = SubprocessRunner()
    handle = runner.launch(
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        env={"PATH": "/usr/bin:/bin"},
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
    )
    assert handle.poll() is None
    deadline = time.monotonic() + 10.0
    while handle.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert handle.wait(timeout_s=5.0) == 0
    assert handle.poll() == 0
    assert runner.process_alive(handle.pid) is False


# ---------------------------------------------------------------------------
# ros_session_entry: child-side schemas and execution flow (fake runtime)
# ---------------------------------------------------------------------------


class FakeChildRuntime:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.registered = 0
        self.shutdown_called = False
        self.ready_timeout: float | None = None

    def register_subscriptions(self) -> None:
        self.registered += 1

    def register_clients(self) -> None:
        self.registered += 1

    def wait_until_ready(self, timeout_s: float) -> bool:
        self.ready_timeout = timeout_s
        return self.ready

    def readiness_snapshot(self) -> ReadinessSnapshot:
        if self.ready:
            return ReadinessSnapshot(**READY_SNAPSHOT)
        return ReadinessSnapshot()

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeChildExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, float]] = []

    def execute_joint(self, target, *, timeout_s):
        self.calls.append(("joint", target, timeout_s))
        return ExecutionResult(
            status=ExecutionStatus.SUCCEEDED,
            moveit_error_code=1,
            planned_waypoints=5,
            command_messages=4,
        )

    def execute_link6_pose(self, target, *, timeout_s):
        self.calls.append(("pose", target, timeout_s))
        return ExecutionResult(
            status=ExecutionStatus.SUCCEEDED,
            moveit_error_code=1,
            planned_waypoints=7,
            command_messages=6,
        )


def test_child_readiness_operation_never_sends_motion(tmp_path):
    runtime = FakeChildRuntime()
    executor = FakeChildExecutor()
    operation = readiness_operation(timeout_s=12.0, request_id="ready-1")
    result = ros_session_entry.perform_operation(
        operation, runtime=runtime, executor=executor
    )
    assert result.passed is True
    assert result.error is None
    assert result.execution["motion_sent"] is False
    assert result.execution["goals_sent"] == 0
    assert result.evidence["waited_ok"] is True
    assert result.evidence["snapshot"]["controllers_active"] is True
    assert executor.calls == []
    assert runtime.ready_timeout == 12.0
    parsed = RuntimeOperationResult.parse(result.serialize())
    assert parsed == result


def test_child_readiness_failure_reports_and_never_moves():
    runtime = FakeChildRuntime(ready=False)
    executor = FakeChildExecutor()
    result = ros_session_entry.perform_operation(
        readiness_operation(timeout_s=5.0, request_id="ready-2"),
        runtime=runtime,
        executor=executor,
    )
    assert result.passed is False
    assert result.error is not None
    assert result.execution["motion_sent"] is False
    assert executor.calls == []


def test_child_joint_operation_delegates_once_with_admitted_target():
    runtime = FakeChildRuntime()
    executor = FakeChildExecutor()
    operation = joint_operation(
        JointTarget(positions=dict(ADMITTED_POSITIONS), goal_tolerance=0.01),
        33.0,
        "joint-child-1",
    )
    result = ros_session_entry.perform_operation(
        operation, runtime=runtime, executor=executor
    )
    assert result.passed is True
    assert result.execution["status"] == "SUCCEEDED"
    assert result.execution["is_success"] is True
    assert result.execution["command_messages"] == 4
    assert result.evidence["motion_sent"] is True
    assert len(executor.calls) == 1
    kind, target, timeout_s = executor.calls[0]
    assert kind == "joint"
    assert target.positions == ADMITTED_POSITIONS
    assert timeout_s == 33.0


def test_child_pose_operation_delegates_once():
    runtime = FakeChildRuntime()
    executor = FakeChildExecutor()
    operation = link6_pose_operation(
        Link6PoseTarget(
            frame_id="base_link",
            position=(0.2, 0.0, 0.3),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            position_tolerance_m=0.005,
            orientation_tolerance_rad=0.015,
        ),
        21.0,
        "pose-child-1",
    )
    result = ros_session_entry.perform_operation(
        operation, runtime=runtime, executor=executor
    )
    assert result.passed is True
    assert executor.calls[0][0] == "pose"
    assert executor.calls[0][2] == 21.0


def test_child_operation_refuses_inadmissible_target_without_motion():
    runtime = FakeChildRuntime()
    executor = FakeChildExecutor()
    forged = RuntimeOperation(
        kind=OperationKind.JOINT_TARGET,
        payload={
            "positions": {**ADMITTED_POSITIONS, "joint5": 99.0},
            "goal_tolerance": 0.01,
        },
        timeout_s=10.0,
        request_id="forged-child",
    )
    result = ros_session_entry.perform_operation(
        forged, runtime=runtime, executor=executor
    )
    assert result.passed is False
    assert result.execution["status"] == "REJECTED"
    assert result.execution["sent"] is False
    assert executor.calls == []
    assert result.evidence["motion_sent"] is False


def test_child_operation_failure_of_executor_gates_reports_failed():
    class FailingExecutor(FakeChildExecutor):
        def execute_joint(self, target, *, timeout_s):
            self.calls.append(("joint", target, timeout_s))
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                moveit_error_code=0,
                planned_waypoints=0,
                command_messages=0,
            )

    runtime = FakeChildRuntime()
    executor = FailingExecutor()
    result = ros_session_entry.perform_operation(
        joint_operation(
            JointTarget(positions=dict(ADMITTED_POSITIONS)), 10.0, "fails"
        ),
        runtime=runtime,
        executor=executor,
    )
    assert result.passed is False
    assert result.execution["status"] == "FAILED"
    assert result.error is not None


def test_child_write_result_is_atomic_and_strict(tmp_path):
    result = RuntimeOperationResult(
        request_id="w1",
        kind=OperationKind.JOINT_TARGET,
        passed=True,
        execution={"status": "SUCCEEDED"},
        evidence={"waited_ok": True},
        error=None,
    )
    destination = tmp_path / "nested" / "result.json"
    ros_session_entry.write_result(destination, result)
    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["schema"] == OPERATION_RESULT_SCHEMA
    assert RuntimeOperationResult.parse(document) == result
    leftovers = [path.name for path in destination.parent.iterdir()]
    assert leftovers == ["result.json"]


def test_child_main_writes_fallback_result_for_missing_request(
    tmp_path, capsys, monkeypatch
):
    def explode(*args, **kwargs):
        raise AssertionError("no runtime may be created for a bad request")

    monkeypatch.setattr(ros_session_entry, "_spin_up", explode)
    request = tmp_path / "missing.json"
    destination = tmp_path / "result.json"
    code = ros_session_entry.main([
        "--request", str(request),
        "--result", str(destination),
    ])
    assert code == 2
    lines = [
        line for line in capsys.readouterr().out.splitlines() if line.strip()
    ]
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert summary["schema"] == "rocobrick-runtime-operation-summary/v1"
    assert summary["passed"] is False
    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["schema"] == OPERATION_RESULT_SCHEMA
    assert document["passed"] is False


def test_child_main_reports_runtime_failure_with_result_file(
    tmp_path, capsys, monkeypatch
):
    def failing_spin_up(node_name, fk_timeout_s):
        raise RuntimeError("no ROS in unit tests")

    monkeypatch.setattr(ros_session_entry, "_spin_up", failing_spin_up)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(joint_request_document("in-container-1")))
    destination = tmp_path / "result.json"
    code = ros_session_entry.main([
        "--request", str(request),
        "--result", str(destination),
    ])
    assert code == 1
    summary = json.loads(
        [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.strip()
        ][0]
    )
    assert summary["request_id"] == "in-container-1"
    assert summary["kind"] == "JOINT_TARGET"
    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["request_id"] == "in-container-1"
    assert document["passed"] is False


# ---------------------------------------------------------------------------
# AST hygiene: no tools/smoke imports, lazy ROS, no STAP in the child,
# no example1 pose/AABB/grasp/TCP derivation anywhere in R4.2 production code
# ---------------------------------------------------------------------------


PRODUCTION_MODULES = (
    "isaac_backend.py",
    "runtime_session.py",
    "ros_session_entry.py",
    "piper_run.py",
)

BANNED_IMPORT_ROOTS = (
    "tools",
    "smoke",
    "isaacsim",
    "omni",
    "pxr",
)

CHILD_BANNED_MODULES = (
    "rocobrick.stap_contract",
    "rocobrick.structural_binding",
    "rocobrick.tcp_contract",
    "rocobrick.planner_contract",
    "rocobrick.contracts",
    "rocobrick.adapters",
    "rocobrick.skills",
    "rocobrick.execution",
)


def module_imports(name: str) -> set[str]:
    tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def top_level_imports(name: str) -> set[str]:
    tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


@pytest.mark.parametrize("module", PRODUCTION_MODULES)
@pytest.mark.parametrize("root", BANNED_IMPORT_ROOTS)
def test_production_imports_no_tools_or_smoke_modules(module, root):
    for name in module_imports(module):
        assert name.split(".")[0] != root, f"{module} imports {name}"


def test_runtime_session_never_imports_ros_or_isaac():
    imported = module_imports("runtime_session.py")
    for banned in (
        "rclpy",
        "rocobrick.ros_moveit_runtime",
        "rocobrick.ros_moveit_executor",
        "isaacsim",
    ):
        assert banned not in imported


def test_child_entry_ros_imports_are_lazy():
    top = top_level_imports("ros_session_entry.py")
    for banned in (
        "rclpy",
        "rocobrick.ros_moveit_runtime",
        "rocobrick.ros_moveit_executor",
    ):
        assert banned not in top
    whole = module_imports("ros_session_entry.py")
    for lazy in (
        "rclpy",
        "rocobrick.ros_moveit_runtime",
        "rocobrick.ros_moveit_executor",
    ):
        assert lazy in whole, f"{lazy} must be imported lazily in the child"


@pytest.mark.parametrize("module", ("isaac_backend.py", "ros_session_entry.py"))
@pytest.mark.parametrize("banned", CHILD_BANNED_MODULES)
def test_child_path_imports_no_stap_or_metric_modules(module, banned):
    assert banned not in module_imports(module)


@pytest.mark.parametrize(
    "module",
    ("isaac_backend.py", "runtime_session.py", "ros_session_entry.py"),
)
def test_no_example1_pose_aabb_grasp_or_tcp_derivation(module):
    source = (SRC / module).read_text(encoding="utf-8")
    for banned in (
        "example1",
        "assess_piper_l_tcp",
        "TcpContract",
        "require_resolved_tcp",
    ):
        assert banned not in source, f"{module} mentions {banned}"
