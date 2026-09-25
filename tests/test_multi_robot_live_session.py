"""Offline tests for the A6 multi-robot live session.

Everything runs against a fake ``ProcessRunner``/clock and a fake container
child that speaks the A5 session-envelope wire contract
(``rocobrick-ros-session-request/v1`` in, ``rocobrick-ros-session-result/v1``
out) and builds its answers with the real A3/A5 slicing helpers - no Docker,
no Isaac, no ROS, no rclpy and no motion is ever involved.

The tests pin: the ONE-host + ONE-container dual-stack topology (no per-robot
process or ROS domain), the A3 dual launch arguments, the all-plan-robot
readiness envelope (including a robot with no operation), the robot-scoped
read-only observation transport, the refusal of every motion kind with zero
commands, the fail-closed gates, and the session-scoped cleanup that never
touches another session's or a pre-existing leftover's resources.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from pathlib import Path

import pytest
from test_stap_contract import payload as stap_plan_payload

from rocobrick.isaac_backend import (
    OPERATION_RESULT_SCHEMA,
    OperationKind,
    RuntimeOperation,
    scene_query_from_payload,
)
from rocobrick.moveit_contract import JOINT_NAMES
from rocobrick.ros_moveit_runtime import RobotCapabilityReadiness, robot_scene_view
from rocobrick.ros_session_entry import (
    SCENE_STATE_SCHEMA,
    SESSION_REQUEST_KEYS,
    SESSION_REQUEST_SCHEMA,
    SESSION_RESULT_SCHEMA,
    RosSessionRequest,
    filter_scene_state,
)
from rocobrick.runtime_session import (
    DUAL_LAUNCH_FILE,
    DUAL_LAUNCH_PACKAGE,
    MULTI_ROBOT_SESSION_SCHEMA,
    READ_ONLY_OPERATION_KINDS,
    MultiRobotIsaacMoveItSession,
    MultiRobotSessionConfig,
    MultiRobotSessionError,
    RuntimeSessionStatus,
    build_global_scene_document,
)

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src" / "rocobrick"
ROBOTS = ("piper_0", "piper_1")
DEVICE_JOINTS = {
    robot_id: tuple(f"{robot_id}_{name}" for name in JOINT_NAMES)
    for robot_id in ROBOTS
}
SCENE_MANIFEST_SCHEMA = "rocobrick-scene-snapshot-manifest/v1"
# Distinct per-robot gripper readings: any leak between robots is visible.
GRIPPER_READINGS = {
    "piper_0": {"gripper_joint1": 0.031, "gripper_joint2": -0.031},
    "piper_1": {"gripper_joint1": 0.005, "gripper_joint2": -0.005},
}
SIM_TIME_S = 12.5
SCENE_MANIFEST_OBJECTS = (
    {
        "physical_instance_id": "brick-alpha",
        "brick_id": 2,
        "prim_path": "/World/managed/Parts/Part_0",
        "asset_revision_sha256": "a" * 64,
        "provenance": {"source": "test-manifest"},
    },
    {
        "physical_instance_id": "brick-beta",
        "brick_id": 5,
        "prim_path": "/World/managed/Parts/Part_1",
        "asset_revision_sha256": "b" * 64,
        "provenance": {"source": "test-manifest"},
    },
)


# ---------------------------------------------------------------------------
# Fakes: clock, host handle, runner with an A5-speaking container child
# ---------------------------------------------------------------------------


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
    """Fake long-running handle for the host bridge process."""

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
        self._runner.events.append(("signal", signum))
        if signum == 9:
            if not self._runner.host_ignores_kill:
                self.exit_code = -9
        elif signum == 2:
            if not self._runner.host_ignores_int:
                self.exit_code = 0
                # The real bridge catches KeyboardInterrupt and still writes
                # its own intentional-stop evidence (without hold_completed).
                self._runner.write_host_stop_evidence()


class FakeRunner:
    """Fake ProcessRunner with a fake A5 session child inside the container.

    The child replays the real contract: it strictly parses the session
    request, slices the fake global scene document with the real A5 helpers
    (``robot_scene_view`` + ``filter_scene_state``) and writes a strict
    ``rocobrick-ros-session-result/v1`` document, exactly like
    ``python3 -m rocobrick.ros_session_entry --robots ...`` does in the
    container.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        robot_ids: tuple[str, ...] = ROBOTS,
        marker: bool = True,
        host_dies: bool = False,
        host_self_exit_after: float | None = 2.0,
        host_ignores_int: bool = False,
        host_ignores_kill: bool = False,
        host_writes_evidence: bool = True,
        host_writes_stop_evidence: bool = True,
        host_early_stop_accepted: bool = False,
        host_commands_applied: int | None = None,
        host_prewritten_stop_evidence: bool = False,
        docker_run_fails: bool = False,
        container_dies: bool = False,
        removal_verify_flaky: int = 0,
        child_times_out: bool = False,
        child_writes_no_result: bool = False,
        child_robots_mismatch: bool = False,
        readiness_by_robot: dict[str, bool] | None = None,
        drop_readiness_report: str | None = None,
        force_all_ready: bool = False,
        scene: dict | None = None,
        scene_query_passed: bool = True,
        tamper_scene_state=None,
        writes_generated_configs: bool = True,
        leftover_containers: tuple[str, ...] = (),
        foreign_processes: tuple[str, ...] = (),
    ) -> None:
        self.session_dir = session_dir
        self.robot_ids = robot_ids
        self.marker = marker
        self.host_dies = host_dies
        self.host_self_exit_after = host_self_exit_after
        self.host_ignores_int = host_ignores_int
        self.host_ignores_kill = host_ignores_kill
        self.host_writes_evidence = host_writes_evidence
        self.host_writes_stop_evidence = host_writes_stop_evidence
        self.host_early_stop_accepted = host_early_stop_accepted
        self.host_commands_applied = host_commands_applied
        self.host_prewritten_stop_evidence = host_prewritten_stop_evidence
        self.docker_run_fails = docker_run_fails
        self.container_dies = container_dies
        self.removal_verify_flaky = removal_verify_flaky
        self.child_times_out = child_times_out
        self.child_writes_no_result = child_writes_no_result
        self.child_robots_mismatch = child_robots_mismatch
        self.readiness_by_robot = (
            dict(readiness_by_robot)
            if readiness_by_robot is not None
            else {robot_id: True for robot_id in robot_ids}
        )
        self.drop_readiness_report = drop_readiness_report
        self.force_all_ready = force_all_ready
        self.scene = scene if scene is not None else global_scene_document()
        self.scene_query_passed = scene_query_passed
        self.tamper_scene_state = tamper_scene_state
        self.writes_generated_configs = writes_generated_configs
        self.leftover_containers = leftover_containers
        self.foreign_processes = foreign_processes
        self.launches: list[tuple[list[str], dict[str, str], Path]] = []
        self.commands: list[tuple[list[str], float | None]] = []
        self.execs: list[list[str]] = []
        self.container_names: list[str] = []
        self.containers: dict[str, bool] = {}
        self.auto_remove: dict[str, bool] = {}
        self.ps_filters: list[str] = []
        self.process_patterns: list[str] = []
        self.events: list[tuple[str, object]] = []
        self.clock = FakeClock()
        self._next_pid = 4200
        self._handle_by_pid: dict[int, FakeHandle] = {}
        self._process_commands: dict[int, str] = {}

    # ------------------------------------------------------------ processes

    def launch(self, argv, *, env, stdout_path, stderr_path):
        argv = [str(part) for part in argv]
        self.launches.append((list(argv), dict(env), Path(stdout_path)))
        self._next_pid += 1
        handle = FakeHandle(self._next_pid, self)
        self._handle_by_pid[handle.pid] = handle
        self._process_commands[handle.pid] = " ".join(argv)
        if self.host_dies:
            handle.exit_code = 1
        Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
        if self.host_prewritten_stop_evidence:
            # The bridge already wrote a stop attestation before it exited on
            # its own: no SIGINT ever reached it.
            self.write_host_stop_evidence()
        if self.host_writes_evidence:
            Path(stdout_path).with_name("host-bridge-evidence.json").write_text(
                json.dumps(
                    {
                        "schema": "rocobrick-host-bridge-smoke/v1",
                        "ready": True,
                        "passed": True,
                        "hold_completed": True,
                        "error": None,
                        "robots": {
                            robot_id: {"prim_path": f"/World/{robot_id}"}
                            for robot_id in self.robot_ids
                        },
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
                json.dumps({"marker": "p0.5_host_bridge_ready"}) + "\n",
                encoding="utf-8",
            )
        else:
            Path(stdout_path).write_text("booting\n", encoding="utf-8")
        return handle

    def write_host_stop_evidence(self) -> None:
        """Rewrite the bridge evidence as the real SIGINT stop path does.

        The bridge catches ``KeyboardInterrupt``, writes ``stopped:
        sigint_orderly_stop`` plus its orderly rclpy/world/timeline/app
        shutdown flags, and never sets ``hold_completed`` for that stop.
        """
        if not self.host_writes_stop_evidence:
            return
        path = self.session_dir / "host-bridge-evidence.json"
        document = {}
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            document = {}
        if not isinstance(document, dict):
            document = {}
        document.update(
            {
                "schema": "rocobrick-host-bridge-smoke/v1",
                "hold_completed": False,
                "stopped": "sigint_orderly_stop",
                "error": None,
                "cleanup": {
                    "rclpy_shutdown": True,
                    "world_stopped": True,
                    "timeline_stopped": True,
                    "app_close_initiated": True,
                },
            }
        )
        document["passed"] = False
        document["ready"] = True
        applied = self.host_commands_applied
        if self.host_early_stop_accepted:
            # A4 attests the bound SIGINT and orderly bridge shutdown.
            document["early_stop_accepted"] = True
            if applied is None:
                applied = 0
        if applied is not None:
            document["commands"] = {"received": applied, "applied": applied}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")

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
        """PIDs of live processes whose command line matches ``pattern``."""
        self.process_patterns.append(pattern)
        matches: list[str] = []
        matcher = re.compile(pattern)
        for pid, command in self._process_commands.items():
            if not matcher.search(command):
                continue
            handle = self._handle_by_pid.get(pid)
            if handle is not None and handle.poll() is None:
                matches.append(str(pid))
        for index, command in enumerate(self.foreign_processes):
            if matcher.search(command):
                matches.append(str(900000 + index))
        return tuple(matches)

    # ------------------------------------------------------------- commands

    def run(self, argv, *, env=None, timeout_s=None):
        argv = [str(part) for part in argv]
        self.commands.append((list(argv), timeout_s))
        joined = " ".join(argv)
        if "generate_dds_profiles.py" in joined:
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "fastdds-container-mesh.xml").write_text("<mesh/>\n")
            (output_dir / "fastdds-host.xml").write_text("<host/>\n")
            return self._outcome(
                argv,
                0,
                json.dumps(
                    {
                        "schema": "rocobrick-dds-profiles/v2",
                        "profiles": {
                            "container-mesh": {"role": "container-client-mesh"}
                        },
                    }
                ),
                "",
            )
        if "piper_l_model_audit.py" in joined:
            Path(argv[argv.index("--output") + 1]).write_text("{}\n")
            return self._outcome(argv, 0, "audit-ok\n", "")
        if argv[0] == "docker":
            return self._docker(argv)
        return self._outcome(argv, 0, "", "")

    def _outcome(self, argv, returncode, stdout, stderr, timed_out=False):
        from rocobrick.runtime_session import RunOutcome

        return RunOutcome(
            argv=tuple(argv),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )

    def _docker(self, argv: list[str]):
        self.events.append(("docker", argv[1]))
        if argv[1] == "run":
            if self.docker_run_fails:
                return self._outcome(argv, 125, "", "docker run failed")
            name = argv[argv.index("--name") + 1]
            self.container_names.append(name)
            self.containers[name] = not self.container_dies
            self.auto_remove[name] = "--rm" in argv
            if self.writes_generated_configs:
                for robot_id in self.robot_ids:
                    directory = (
                        self.session_dir / "dual-robot-config" / robot_id
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    for filename in (
                        "model.urdf",
                        "model.srdf",
                        "kinematics.yaml",
                        "joint_limits.yaml",
                        "moveit_controllers.yaml",
                        "ros2_controllers.yaml",
                    ):
                        (directory / filename).write_text("<generated/>\n")
            return self._outcome(argv, 0, "fake-container-id\n", "")
        if argv[1] == "exec":
            self.execs.append(list(argv))
            if self.child_times_out:
                return self._outcome(argv, None, "", "", timed_out=True)
            if self.child_writes_no_result:
                return self._outcome(argv, 1, "", "child exploded")
            return self._run_child(argv)
        if argv[1] == "logs":
            return self._outcome(argv, 0, "[dual-robot] launching\n", "")
        if argv[1] == "stop":
            name = argv[-1]
            if self.auto_remove.get(name):
                # ``docker run --rm`` removes the container once it stops.
                self.containers.pop(name, None)
            else:
                self.containers[name] = False
            return self._outcome(argv, 0, name + "\n", "")
        if argv[1] == "rm":
            name = argv[-1]
            self.containers.pop(name, None)
            return self._outcome(argv, 0, "", "")
        if argv[1] == "ps":
            return self._docker_ps(argv)
        return self._outcome(argv, 0, "", "")

    def _docker_ps(self, argv: list[str]):
        filter_value = argv[argv.index("--filter") + 1]
        assert filter_value.startswith("name=")
        self.ps_filters.append(filter_value)
        pattern = filter_value[len("name=") :]
        list_all = "-a" in argv
        names = [
            name
            for name, running in self.containers.items()
            if (running or list_all) and pattern in name
        ]
        if list_all:
            names.extend(
                name for name in self.leftover_containers if pattern in name
            )
        if (
            self.removal_verify_flaky > 0
            and self.container_names
            and pattern in self.container_names[-1]
        ):
            self.removal_verify_flaky -= 1
            names.append(self.container_names[-1])
        return self._outcome(
            argv, 0, "".join(name + "\n" for name in names), ""
        )

    # ---------------------------------------------------------------- child

    def _run_child(self, argv: list[str]):
        request_container = argv[-3]
        result_container = argv[-2]
        robots_csv = argv[-1]
        assert request_container.startswith("/session/")
        assert result_container.startswith("/session/")
        request_path = self.session_dir / request_container[len("/session/") :]
        result_path = self.session_dir / result_container[len("/session/") :]
        request = json.loads(request_path.read_text(encoding="utf-8"))
        RosSessionRequest.parse(request)
        robots = tuple(robots_csv.split(","))
        if self.child_robots_mismatch:
            robots = tuple(reversed(robots))
        document = self._answer(request, robots)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(document), encoding="utf-8")
        return self._outcome(
            argv, 0 if document["passed"] else 1, "summary\n", ""
        )

    def _answer(self, request: dict, robots: tuple[str, ...]) -> dict:
        if tuple(request["robots"]) != robots:
            # Same fail-closed answer as the A5 child: no request id, no
            # readiness, nothing dispatched.
            return {
                "schema": SESSION_RESULT_SCHEMA,
                "request_id": "",
                "robots": list(robots),
                "readiness_by_robot": {},
                "all_ready": False,
                "operations": [],
                "passed": False,
                "error": (
                    f"session request robots {list(request['robots'])} do not "
                    f"match --robots {list(robots)}"
                ),
            }
        readiness = {
            robot_id: self._readiness_report(robot_id) for robot_id in robots
        }
        if self.drop_readiness_report is not None:
            readiness.pop(self.drop_readiness_report, None)
        all_ready = bool(
            readiness
            and set(readiness) == set(robots)
            and all(report["passed"] for report in readiness.values())
        )
        if self.force_all_ready:
            all_ready = True
        operations = [
            self._operation_record(
                item["robot_id"], item["operation"], all_ready=all_ready
            )
            for item in request["operations"]
        ]
        failed = [
            record["operation_id"]
            for record in operations
            if not record["result"]["passed"]
        ]
        passed = bool(all_ready and not failed)
        error = None
        if not all_ready:
            error = "robot readiness failed; no operation was dispatched"
        elif failed:
            error = f"session operations failed: {failed}"
        return self._session_document(
            request,
            robots=robots,
            readiness=readiness,
            all_ready=all_ready,
            operations=operations,
            passed=passed,
            error=error,
        )

    def _session_document(
        self,
        request: dict,
        *,
        robots: tuple[str, ...],
        readiness: dict,
        all_ready: bool,
        operations: list,
        passed: bool,
        error: str | None,
    ) -> dict:
        return {
            "schema": SESSION_RESULT_SCHEMA,
            "request_id": request["request_id"],
            "robots": list(robots),
            "readiness_by_robot": readiness,
            "all_ready": bool(all_ready),
            "operations": operations,
            "passed": bool(passed),
            "error": error,
        }

    def _readiness_report(self, robot_id: str) -> dict:
        components = {
            name: True for name in RobotCapabilityReadiness.component_names()
        }
        report = RobotCapabilityReadiness(
            robot_id=robot_id,
            namespace=f"/{robot_id}",
            endpoints=(("move_group_action", f"/{robot_id}/move_action"),),
            **components,
        )
        document = report.serialize()
        document["passed"] = bool(self.readiness_by_robot.get(robot_id, True))
        return document

    def _operation_record(
        self, robot_id: str, operation: dict, *, all_ready: bool
    ) -> dict:
        kind = str(operation["kind"])
        passed = False
        execution: dict = {"motion_sent": False}
        error: str | None = "the runtime readiness failed before dispatch"
        dispatched = bool(all_ready)
        if kind == "READINESS":
            passed = bool(all_ready and self.readiness_by_robot.get(robot_id, True))
            execution = {"motion_sent": False, "goals_sent": 0, "command_messages": 0}
            error = None if passed else "readiness failed"
        elif kind == "SCENE_QUERY" and all_ready:
            try:
                query = scene_query_from_payload(operation["payload"])
                view = robot_scene_view(
                    self.scene, robot_id, joint_names=DEVICE_JOINTS[robot_id]
                )
                scene_state = filter_scene_state(view, query)
                if self.tamper_scene_state is not None:
                    scene_state = self.tamper_scene_state(scene_state)
            except (RuntimeError, ValueError) as exc:
                passed = False
                execution = {"motion_sent": False, "scene_state": None}
                error = f"{type(exc).__name__}: {exc}"
            else:
                passed = bool(self.scene_query_passed)
                execution = {
                    "motion_sent": False,
                    "scene_state": scene_state if passed else None,
                }
                error = None if passed else "the scene-state query failed"
        return {
            "robot_id": robot_id,
            "operation_id": str(operation["request_id"]),
            "kind": kind,
            "dispatched": dispatched,
            "motion_sent": False,
            "result": {
                "schema": OPERATION_RESULT_SCHEMA,
                "request_id": str(operation["request_id"]),
                "kind": kind,
                "passed": bool(passed),
                "execution": execution,
                "evidence": {"robot_id": robot_id},
                "error": error,
            },
        }


# ---------------------------------------------------------------------------
# Fixtures and payload builders
# ---------------------------------------------------------------------------


def global_scene_document() -> dict:
    """One A4 dual scene-state document with a block per robot."""
    return {
        "schema": SCENE_STATE_SCHEMA,
        "sim_time_s": SIM_TIME_S,
        "bricks": {
            "brick-alpha": {
                "aabb_min": [0.0, 0.0, 0.0],
                "aabb_max": [0.1, 0.05, 0.02],
            },
            "brick-beta": {
                "aabb_min": [0.2, 0.0, 0.0],
                "aabb_max": [0.3, 0.05, 0.02],
            },
        },
        "connections": {"brick-alpha|brick-beta": True},
        "robots": {
            robot_id: {
                "robot_id": robot_id,
                "sim_time_s": SIM_TIME_S,
                "gripper": dict(GRIPPER_READINGS[robot_id]),
                "opening_m": abs(GRIPPER_READINGS[robot_id]["gripper_joint1"]) * 2.0,
            }
            for robot_id in ROBOTS
        },
    }


def write_scene_manifest(path: Path) -> Path:
    """Write one frozen scene-snapshot manifest (strictly ordered ids)."""
    path.write_text(
        json.dumps(
            {
                "schema": SCENE_MANIFEST_SCHEMA,
                "scene_id": "scene-test",
                "frame_id": "world",
                "objects": [dict(item) for item in SCENE_MANIFEST_OBJECTS],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_stap_plan(path: Path) -> Path:
    """Write one valid frozen STAP plan (real contract, two robots)."""
    path.write_text(json.dumps(stap_plan_payload()), encoding="utf-8")
    return path


def config_kwargs(tmp_path: Path) -> dict:
    """The shared valid configuration arguments of the session under test."""
    return {
        "repo_root": ROOT,
        "robot_ids": ROBOTS,
        "scene_manifest": write_scene_manifest(tmp_path / "scene-manifest.json"),
        "scene_snapshot_out": tmp_path / "scene-snapshot.json",
        "host_instance_id": "host-session-test",
        "plan_path": write_stap_plan(tmp_path / "plan.json"),
        "config_path": tmp_path / "user-config.json",
        "startup_timeout_s": 6.0,
        "readiness_timeout_s": 10.0,
        "operation_timeout_s": 10.0,
        "host_hold_budget_s": 30.0,
        "shutdown_timeout_s": 200.0,
    }


def make_session(tmp_path: Path, runner: FakeRunner, **config_overrides):
    """Build a multi-robot session wired to the fake runner and fake clock."""
    kwargs = config_kwargs(tmp_path)
    kwargs.update(config_overrides)
    config = MultiRobotSessionConfig(**kwargs)
    return MultiRobotIsaacMoveItSession(
        config,
        runner=runner,
        session_dir=tmp_path / "session",
        clock=runner.clock.clock,
        sleep=runner.clock.sleep,
        base_env={"PATH": "/usr/bin"},
    )


def make_runner(tmp_path: Path, **overrides) -> FakeRunner:
    overrides.setdefault("session_dir", tmp_path / "session")
    return FakeRunner(**overrides)


def fake_repo_root(
    tmp_path: Path, *, with_p1a: bool = True, with_dual_launch: bool = True
) -> Path:
    """A minimal repository tree for the preflight asset gates."""
    root = tmp_path / "fake-repo"
    (root / "isaac").mkdir(parents=True, exist_ok=True)
    (root / "tools").mkdir(parents=True, exist_ok=True)
    (root / "isaac" / "p05_host_bridge.py").write_text("# bridge\n")
    (root / "tools" / "isaac-ros-env.sh").write_text("# env\n")
    (root / "tools" / "piper_l_model_audit.py").write_text("# audit\n")
    (root / "tools" / "generate_dds_profiles.py").write_text("# profiles\n")
    if with_p1a:
        (root / "artifacts" / "phase1").mkdir(parents=True, exist_ok=True)
        (
            root / "artifacts" / "phase1" / "dds-multiprocess-gate.json"
        ).write_text(
            json.dumps(
                {
                    "schema": "rocobrick-p1-dds-multiprocess-gate/v1",
                    "passed": True,
                    "checks": {"two_processes": True},
                }
            ),
            encoding="utf-8",
        )
    if with_dual_launch:
        launch_dir = (
            root / "ros_ws" / "src" / DUAL_LAUNCH_PACKAGE / "launch"
        )
        launch_dir.mkdir(parents=True, exist_ok=True)
        (launch_dir / DUAL_LAUNCH_FILE).write_text("# launch\n")
    return root


def read_request(runner: FakeRunner, index: int = 0) -> dict:
    """The session request document written for the ``index``-th child exec."""
    argv = runner.execs[index]
    container_path = argv[-3]
    assert container_path.startswith("/session/")
    path = runner.session_dir / container_path[len("/session/") :]
    return json.loads(path.read_text(encoding="utf-8"))


def read_result(runner: FakeRunner, index: int = 0) -> dict:
    argv = runner.execs[index]
    container_path = argv[-2]
    path = runner.session_dir / container_path[len("/session/") :]
    return json.loads(path.read_text(encoding="utf-8"))


def docker_commands(runner: FakeRunner) -> list[list[str]]:
    return [command for command, _ in runner.commands if command[0] == "docker"]


def motion_operations() -> list[RuntimeOperation]:
    """One operation of every kind this stage must refuse to dispatch."""
    return [
        RuntimeOperation(
            kind=kind,
            payload={},
            timeout_s=1.0,
            request_id=f"refused-{kind.value.lower()}",
        )
        for kind in (
            OperationKind.JOINT_TARGET,
            OperationKind.LINK6_POSE,
            OperationKind.TCP_POSE,
            OperationKind.GRIPPER_TARGET,
            OperationKind.APPLY_PLANNING_SCENE,
        )
    ]


# ---------------------------------------------------------------------------
# Config contract
# ---------------------------------------------------------------------------


def test_config_rejects_unusable_robot_ids_paths_and_bounds(tmp_path):
    kwargs = config_kwargs(tmp_path)
    with pytest.raises(ValueError, match="robot_ids must be a non-empty"):
        MultiRobotSessionConfig(**{**kwargs, "robot_ids": ()})
    with pytest.raises(ValueError, match="robot_ids must be unique"):
        MultiRobotSessionConfig(**{**kwargs, "robot_ids": ("piper_0", "piper_0")})
    with pytest.raises(ValueError, match="ROS-safe"):
        MultiRobotSessionConfig(**{**kwargs, "robot_ids": ("piper/0",)})
    with pytest.raises(ValueError, match="scene_manifest must be a Path"):
        MultiRobotSessionConfig(**{**kwargs, "scene_manifest": "not-a-path"})
    with pytest.raises(ValueError, match="config_output_subdir"):
        MultiRobotSessionConfig(
            **{**kwargs, "config_output_subdir": "nested/dir"}
        )
    with pytest.raises(ValueError, match="startup_timeout_s"):
        MultiRobotSessionConfig(**{**kwargs, "startup_timeout_s": 0.0})
    with pytest.raises(ValueError, match="ros_domain_id"):
        MultiRobotSessionConfig(**{**kwargs, "ros_domain_id": -1})
    with pytest.raises(ValueError, match="observation_physical_instance_ids"):
        MultiRobotSessionConfig(
            **{
                **kwargs,
                "observation_physical_instance_ids": ("brick-a", "brick-a"),
            }
        )
    with pytest.raises(ValueError, match="observation_connection_pairs"):
        MultiRobotSessionConfig(
            **{**kwargs, "observation_connection_pairs": (("brick-a",),)}
        )
    with pytest.raises(ValueError, match="collision_filter_instance_ids require"):
        MultiRobotSessionConfig(
            **{**kwargs, "collision_filter_instance_ids": ("brick-a",)}
        )
    with pytest.raises(ValueError, match="must be an existing directory"):
        MultiRobotSessionConfig(**{**kwargs, "repo_root": tmp_path / "missing"})


def test_observation_instance_ids_default_to_the_manifest(tmp_path):
    config = MultiRobotSessionConfig(**config_kwargs(tmp_path))
    assert config.resolved_observation_instance_ids() == (
        "brick-alpha",
        "brick-beta",
    )
    explicit = MultiRobotSessionConfig(
        **{
            **config_kwargs(tmp_path),
            "observation_physical_instance_ids": ("brick-beta",),
        }
    )
    assert explicit.resolved_observation_instance_ids() == ("brick-beta",)


def test_a_non_manifest_scene_file_fails_closed(tmp_path):
    broken = tmp_path / "broken-manifest.json"
    broken.write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
    config = MultiRobotSessionConfig(
        **{**config_kwargs(tmp_path), "scene_manifest": broken}
    )
    with pytest.raises(MultiRobotSessionError, match="schema must be exactly"):
        config.resolved_observation_instance_ids()


# ---------------------------------------------------------------------------
# start: one host, one container, all-robot readiness
# ---------------------------------------------------------------------------


def test_start_proves_every_robot_ready_in_one_host_and_one_container(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    evidence = session.start()

    assert evidence.passed is True
    assert evidence.all_ready is True
    assert set(evidence.readiness_by_robot) == set(ROBOTS)
    assert all(
        report["passed"] for report in evidence.readiness_by_robot.values()
    )
    # Each report is the robot's own A5 capability report (own namespace).
    assert evidence.readiness_by_robot["piper_1"]["namespace"] == "/piper_1"
    assert session.status is RuntimeSessionStatus.READY
    assert session.readiness is evidence

    # ONE host launch: the A4 scene-mode bridge for the validated plan robots.
    assert len(runner.launches) == 1
    argv, env, log_path = runner.launches[0]
    joined = " ".join(argv)
    assert "tools/isaac-ros-env.sh" in joined
    assert "isaac/p05_host_bridge.py" in joined
    assert argv[argv.index("--scene-manifest") + 1] == str(
        session.config.scene_manifest
    )
    assert argv[argv.index("--plan") + 1] == str(session.config.plan_path)
    assert argv[argv.index("--config") + 1] == str(session.config.config_path)
    assert argv[argv.index("--host-instance-id") + 1] == "host-session-test"
    # Bounds: the short natural-exit hold is 45 s.
    assert argv[argv.index("--hold-seconds") + 1] == "45.0"
    assert argv[argv.index("--hold-steps") + 1] == str(round(45.0 * 458.0))
    assert argv[argv.index("--time-limit") + 1] == str(6.0 + 45.0 + 120.0)
    assert env["ROS_DOMAIN_ID"] == "73"
    assert env["FASTRTPS_DEFAULT_PROFILES_FILE"].endswith("dds/fastdds-host.xml")
    assert log_path == tmp_path / "session" / "host-bridge.log"

    # The bounded command sequence: audit -> profiles -> container -> exec.
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
        elif text.startswith("docker run"):
            kinds.append("container")
            assert "--rm" in command and "-d" in command
            name = command[command.index("--name") + 1]
            assert name.startswith("rocobrick-multi-robot-session-")
            assert len(name) > len("rocobrick-multi-robot-session-")
            assert f"{ROOT / 'src'}:/workspace/src:ro" in command
            assert f"{tmp_path / 'session' / 'dds'}:/dds:ro" in command
            assert f"{tmp_path / 'session'}:/session" in command
            assert command[command.index("-e") + 1] == "ROS_DOMAIN_ID=73"
            assert bound == 6.0
        elif text.startswith("docker exec"):
            kinds.append("exec")
            assert "rocobrick.ros_session_entry" in text
            assert bound == 10.0 + 30.0
        elif text.startswith("docker ps"):
            kinds.append("ps")
        else:  # pragma: no cover - no other command may run
            raise AssertionError(f"unexpected command: {text}")
    assert [kind for kind in kinds if kind != "ps"] == [
        "audit",
        "profiles",
        "container",
        "exec",
    ]

    # ONE container running the A3 dual launch for BOTH robots.
    assert len(runner.container_names) == 1
    container_script = runner.commands[2][0][-1]
    assert "robot_ids:=piper_0,piper_1" in container_script
    assert "use_sim_time:=true" in container_script
    assert "config_output_dir:=/session/dual-robot-config" in container_script
    assert DUAL_LAUNCH_PACKAGE in container_script
    assert DUAL_LAUNCH_FILE in container_script
    assert "--symlink-install" in container_script

    # The readiness envelope: strict request, EMPTY operations, both robots.
    request = read_request(runner)
    assert set(request) == set(SESSION_REQUEST_KEYS)
    assert request["schema"] == SESSION_REQUEST_SCHEMA
    assert request["robots"] == list(ROBOTS)
    assert request["operations"] == []
    assert request["timeout_s"] == 10.0
    assert runner.execs[0][-1] == "piper_0,piper_1"
    result = read_result(runner)
    assert result["schema"] == SESSION_RESULT_SCHEMA
    assert result["all_ready"] is True

    # Per-robot evidence plus the generated A3 configs of this launch.
    document = json.loads(
        (tmp_path / "session" / "multi-robot-session.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["schema"] == MULTI_ROBOT_SESSION_SCHEMA
    assert document["status"] == "READY"
    assert document["robots"] == list(ROBOTS)
    assert set(document["readiness"]["readiness_by_robot"]) == set(ROBOTS)
    assert document["readiness"]["passed"] is True
    assert document["plan_check"]["verified"] is True
    assert document["plan_check"]["plan_robots"] == list(ROBOTS)
    assert document["observation_instance_ids"] == ["brick-alpha", "brick-beta"]
    assert document["config_output_dir"] == str(
        tmp_path / "session" / "dual-robot-config"
    )
    assert all(
        entry["complete"] for entry in document["generated_configs"].values()
    )
    assert document["container_name"] == runner.container_names[0]
    assert document["container_id"] == "fake-container-id"

    # No motion command of any kind was ever issued during start.
    assert_motion_free(runner.commands)


def test_start_forwards_the_collision_filter_to_the_host_bridge(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(
        tmp_path,
        runner,
        collision_filter_robot_id="piper_1",
        collision_filter_instance_ids=("brick-alpha",),
    )
    session.start()
    argv = runner.launches[0][0]
    assert argv[argv.index("--physics-filter-robot-id") + 1] == "piper_1"
    assert argv[argv.index("--physics-filter-instance-id") + 1] == "brick-alpha"


def test_one_unready_robot_fails_start_with_no_operation_dispatched(tmp_path):
    runner = make_runner(tmp_path, readiness_by_robot={"piper_1": False})
    session = make_session(tmp_path, runner)
    with pytest.raises(MultiRobotSessionError, match="piper_1"):
        session.start()
    assert session.status is RuntimeSessionStatus.FAILED
    assert session.readiness is not None
    assert session.readiness.not_ready_robots == ("piper_1",)
    assert session.readiness.passed is False
    # The readiness envelope carried no operation at all, and no second exec
    # (a scene query) was ever sent.
    assert read_request(runner)["operations"] == []
    assert len(runner.execs) == 1
    document = json.loads(
        (tmp_path / "session" / "multi-robot-session.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["status"] == "FAILED"
    assert document["readiness"]["not_ready_robots"] == ["piper_1"]
    assert len(document["errors"]) == 1


def test_a_missing_robot_report_fails_closed_even_with_all_ready(tmp_path):
    runner = make_runner(
        tmp_path,
        force_all_ready=True,
        drop_readiness_report="piper_1",
    )
    session = make_session(tmp_path, runner)
    with pytest.raises(MultiRobotSessionError, match="not every plan robot"):
        session.start()
    assert session.status is RuntimeSessionStatus.FAILED
    assert session.readiness is not None
    assert session.readiness.all_ready is True
    assert session.readiness.passed is False
    assert session.readiness.not_ready_robots == ("piper_1",)


def test_a_plan_whose_robots_differ_refuses_before_any_process(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(
        tmp_path, runner, robot_ids=("piper_0",)
    )
    with pytest.raises(MultiRobotSessionError, match="plan_robots"):
        session.start()
    assert runner.commands == []
    assert runner.launches == []
    assert runner.execs == []
    assert session.status is RuntimeSessionStatus.FAILED


def test_a_non_stap_plan_refuses_before_any_process(tmp_path):
    plan = tmp_path / "metric-plan.json"
    plan.write_text(json.dumps({"schema": "assembly-plan/v2"}), encoding="utf-8")
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner, plan_path=plan)
    with pytest.raises(MultiRobotSessionError, match="plan_robots"):
        session.start()
    assert runner.commands == []
    assert runner.launches == []


def test_plan_verification_can_be_disabled_explicitly(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(
        tmp_path,
        runner,
        plan_path=tmp_path / "not-a-stap-plan.json",
        verify_plan_robots=False,
    )
    (tmp_path / "not-a-stap-plan.json").write_text("{}", encoding="utf-8")
    session.start()
    document = json.loads(
        (tmp_path / "session" / "multi-robot-session.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["plan_check"] == {
        "verified": False,
        "reason": "disabled by config",
    }


def test_a_missing_p1a_artifact_refuses_before_any_process(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    fake_root = fake_repo_root(tmp_path, with_p1a=False)
    session.config = MultiRobotSessionConfig(
        **{**config_kwargs(tmp_path), "repo_root": fake_root}
    )
    with pytest.raises(MultiRobotSessionError, match="p1a_prerequisite"):
        session.start()
    assert runner.commands == []


def test_a_missing_dual_launch_asset_refuses_before_any_process(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    fake_root = fake_repo_root(tmp_path, with_dual_launch=False)
    session.config = MultiRobotSessionConfig(
        **{**config_kwargs(tmp_path), "repo_root": fake_root}
    )
    with pytest.raises(MultiRobotSessionError, match="dual-robot launch missing"):
        session.start()
    assert runner.commands == []
    assert runner.launches == []


def test_a_dead_container_is_reported_with_its_log(tmp_path):
    runner = make_runner(tmp_path, container_dies=True)
    session = make_session(tmp_path, runner)
    with pytest.raises(MultiRobotSessionError, match="is not running"):
        session.start()
    assert session.status is RuntimeSessionStatus.FAILED
    assert (tmp_path / "session" / "dual-launch.log").is_file()


def test_a_docker_run_failure_cleans_up_the_host(tmp_path):
    runner = make_runner(tmp_path, docker_run_fails=True)
    session = make_session(tmp_path, runner)
    with pytest.raises(MultiRobotSessionError, match="container_start"):
        session.start()
    evidence = session.stop()
    assert evidence.cleanup["container_removed"] is True
    assert evidence.cleanup["host_stopped"] is True
    assert evidence.status is RuntimeSessionStatus.FAILED


def test_a_host_death_before_the_marker_fails_fast(tmp_path):
    runner = make_runner(tmp_path, host_dies=True, marker=False)
    session = make_session(tmp_path, runner)
    with pytest.raises(MultiRobotSessionError, match="exited before"):
        session.start()
    assert runner.execs == []


# ---------------------------------------------------------------------------
# Read-only dispatch: no motion, robot-scoped observation
# ---------------------------------------------------------------------------


def test_every_motion_and_planning_scene_kind_is_refused_with_zero_commands(
    tmp_path,
):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    session.start()
    baseline = len(runner.commands)
    for operation in motion_operations():
        with pytest.raises(MultiRobotSessionError, match="not a read-only"):
            session.run_read_only_operation("piper_0", operation)
    assert len(runner.commands) == baseline
    assert len(runner.execs) == 1
    assert session.status is RuntimeSessionStatus.READY
    assert session.observations == []


def test_an_unknown_robot_is_refused_with_zero_commands(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    session.start()
    baseline = len(runner.commands)
    with pytest.raises(MultiRobotSessionError, match="unknown session robot"):
        session.run_read_only_operation("piper_9", motion_operations()[0])
    with pytest.raises(MultiRobotSessionError, match="unknown session robot"):
        session.observe_sync("piper_9")
    with pytest.raises(MultiRobotSessionError, match="unknown session robot"):
        asyncio.run(session.observe("piper_9"))
    assert len(runner.commands) == baseline
    assert session.status is RuntimeSessionStatus.READY


def test_read_only_dispatch_requires_ready(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    with pytest.raises(MultiRobotSessionError, match="requires READY"):
        session.observe_sync("piper_0")
    assert runner.commands == []


def test_observe_returns_the_global_document_of_that_robot_only(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(
        tmp_path,
        runner,
        observation_connection_pairs=(
            ("brick-alpha", "brick-beta"),
            ("brick-beta", "brick-alpha"),
        ),
    )
    session.start()
    document = session.observe_sync("piper_1")

    assert set(document) == {
        "schema",
        "sim_time_s",
        "bricks",
        "connections",
        "robots",
    }
    assert document["schema"] == SCENE_STATE_SCHEMA
    assert document["sim_time_s"] == SIM_TIME_S
    assert set(document["bricks"]) == {"brick-alpha", "brick-beta"}
    assert document["bricks"]["brick-alpha"]["aabb_max"] == [0.1, 0.05, 0.02]
    # Both requested pair orders normalize onto the sorted key.
    assert document["connections"] == {"brick-alpha|brick-beta": True}
    assert set(document["robots"]) == {"piper_1"}
    block = document["robots"]["piper_1"]
    assert block["robot_id"] == "piper_1"
    assert block["sim_time_s"] == SIM_TIME_S
    assert block["gripper"] == GRIPPER_READINGS["piper_1"]
    assert block["opening_m"] == pytest.approx(0.01)
    # piper_0's reading never appears anywhere in piper_1's observation.
    assert "0.031" not in json.dumps(document)
    assert session.status is RuntimeSessionStatus.READY

    # The observation was one robot-targeted, gripper-including SCENE_QUERY.
    assert len(runner.execs) == 2
    request = read_request(runner, 1)
    assert request["robots"] == list(ROBOTS)
    assert len(request["operations"]) == 1
    item = request["operations"][0]
    assert item["robot_id"] == "piper_1"
    assert item["operation"]["kind"] == "SCENE_QUERY"
    payload = item["operation"]["payload"]
    assert payload["brick_physical_instance_ids"] == ["brick-alpha", "brick-beta"]
    assert payload["connection_pairs"] == [
        ["brick-alpha", "brick-beta"],
        ["brick-beta", "brick-alpha"],
    ]
    assert payload["include_gripper"] is True
    assert runner.execs[1][-1] == "piper_0,piper_1"

    # Per-robot evidence, request/result files and no motion anywhere.
    document_file = json.loads(
        (tmp_path / "session" / "multi-robot-session.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(document_file["observations"]) == 1
    observation = document_file["observations"][0]
    assert observation["robot_id"] == "piper_1"
    assert observation["kind"] == "SCENE_QUERY"
    assert observation["passed"] is True
    assert observation["motion_sent"] is False
    assert observation["sim_time_s"] == SIM_TIME_S
    operation_id = observation["operation_id"]
    for suffix in ("request.json", "result.json"):
        path = (
            tmp_path
            / "session"
            / "operations"
            / f"piper_1-{operation_id}-{suffix}"
        )
        assert path.is_file(), path
    assert_motion_free(runner.commands)


def test_observe_of_the_other_robot_keeps_the_readings_apart(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    session.start()
    first = session.observe_sync("piper_0")
    second = session.observe_sync("piper_1")
    assert first["robots"]["piper_0"]["gripper"] == GRIPPER_READINGS["piper_0"]
    assert second["robots"]["piper_1"]["gripper"] == GRIPPER_READINGS["piper_1"]
    assert "0.005" not in json.dumps(first)
    assert "0.031" not in json.dumps(second)
    assert len(appended_observations(tmp_path)) == 2


def test_async_observe_returns_the_same_document(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    session.start()
    document = asyncio.run(session.observe("piper_1"))
    assert document["robots"]["piper_1"]["opening_m"] == pytest.approx(0.01)
    assert len(appended_observations(tmp_path)) == 1


def test_observe_fails_closed_when_the_robot_block_is_missing(tmp_path):
    scene = global_scene_document()
    del scene["robots"]
    runner = make_runner(tmp_path, scene=scene)
    session = make_session(tmp_path, runner)
    session.start()
    with pytest.raises(MultiRobotSessionError, match="SCENE_QUERY operation"):
        session.observe_sync("piper_1")
    assert session.status is RuntimeSessionStatus.FAILED
    assert appended_observations(tmp_path) == []


def test_observe_fails_closed_on_a_foreign_robot_binding(tmp_path):
    scene = global_scene_document()
    scene["robots"]["piper_1"] = dict(
        scene["robots"]["piper_1"],
        joint_names=list(DEVICE_JOINTS["piper_0"]),
    )
    runner = make_runner(tmp_path, scene=scene)
    session = make_session(tmp_path, runner)
    session.start()
    with pytest.raises(MultiRobotSessionError, match="SCENE_QUERY operation"):
        session.observe_sync("piper_1")
    assert session.status is RuntimeSessionStatus.FAILED


def test_observe_of_a_sibling_robot_block_cannot_be_substituted(tmp_path):
    scene = global_scene_document()
    scene["robots"]["piper_1"] = dict(
        scene["robots"]["piper_1"], robot_id="piper_0"
    )
    runner = make_runner(tmp_path, scene=scene)
    session = make_session(tmp_path, runner)
    session.start()
    with pytest.raises(MultiRobotSessionError, match="SCENE_QUERY operation"):
        session.observe_sync("piper_1")


def test_a_child_claimed_pass_with_a_tampered_block_fails_closed(tmp_path):
    """The session validates the answer itself, never only the child verdict."""
    runner = make_runner(
        tmp_path,
        tamper_scene_state=lambda state: {
            key: value for key, value in state.items() if key != "gripper"
        },
    )
    session = make_session(tmp_path, runner)
    session.start()
    with pytest.raises(MultiRobotSessionError, match="validated block"):
        session.observe_sync("piper_1")
    assert session.status is RuntimeSessionStatus.FAILED
    assert appended_observations(tmp_path) == []


def test_a_failed_scene_query_is_not_an_observation(tmp_path):
    runner = make_runner(tmp_path, scene_query_passed=False)
    session = make_session(tmp_path, runner)
    session.start()
    with pytest.raises(MultiRobotSessionError, match="SCENE_QUERY operation"):
        session.observe_sync("piper_0")
    assert session.status is RuntimeSessionStatus.FAILED
    assert appended_observations(tmp_path) == []


def test_observe_fails_closed_when_a_sibling_robot_is_not_ready(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    session.start()
    # The sibling degrades after readiness: the envelope refuses to dispatch
    # and the session fails closed instead of answering anyway.
    runner.readiness_by_robot["piper_0"] = False
    with pytest.raises(MultiRobotSessionError, match="not every plan robot"):
        session.observe_sync("piper_1")
    assert session.status is RuntimeSessionStatus.FAILED
    assert appended_observations(tmp_path) == []


def test_the_envelope_identity_mismatch_fails_closed(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    session.start()
    runner.child_robots_mismatch = True
    with pytest.raises(MultiRobotSessionError, match="session_dispatch"):
        session.observe_sync("piper_1")
    assert session.status is RuntimeSessionStatus.FAILED


def test_a_child_timeout_fails_closed(tmp_path):
    runner = make_runner(tmp_path, child_times_out=True)
    session = make_session(tmp_path, runner)
    with pytest.raises(MultiRobotSessionError, match="child session entry exceeded"):
        session.start()
    assert session.status is RuntimeSessionStatus.FAILED
    assert runner.execs != []


def test_a_child_without_a_result_fails_closed(tmp_path):
    runner = make_runner(tmp_path, child_writes_no_result=True)
    session = make_session(tmp_path, runner)
    with pytest.raises(MultiRobotSessionError, match="wrote no result"):
        session.start()
    assert session.status is RuntimeSessionStatus.FAILED


# ---------------------------------------------------------------------------
# The A4 document re-wrap (pure function)
# ---------------------------------------------------------------------------


def test_build_global_scene_document_round_trips_one_robot_view():
    document = global_scene_document()
    view = robot_scene_view(
        document, "piper_1", joint_names=DEVICE_JOINTS["piper_1"]
    )
    rebuilt = build_global_scene_document("piper_1", view)
    assert set(rebuilt) == {"schema", "sim_time_s", "bricks", "connections", "robots"}
    assert set(rebuilt["robots"]) == {"piper_1"}
    assert rebuilt["robots"]["piper_1"]["gripper"] == GRIPPER_READINGS["piper_1"]
    assert rebuilt["bricks"] == document["bricks"]
    assert rebuilt["connections"] == document["connections"]


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda view: view.update(schema="other/v1"), "schema"),
        (lambda view: view.update(robot_id="piper_0"), "owned by"),
        (lambda view: view.pop("sim_time_s"), "sim_time_s"),
        (lambda view: view.pop("bricks"), "no bricks"),
        (lambda view: view.pop("gripper"), "gripper"),
        (lambda view: view.update(gripper={"gripper_joint1": 0.0}), "gripper"),
        (lambda view: view.update(opening_m="wide"), "opening_m"),
        (lambda view: view.update(connections={"brick-alpha|brick-beta": 1}),
         "connections"),
    ],
)
def test_build_global_scene_document_rejects_any_deviation(change, message):
    document = global_scene_document()
    view = robot_scene_view(
        document, "piper_1", joint_names=DEVICE_JOINTS["piper_1"]
    )
    change(view)
    with pytest.raises(ValueError, match=message):
        build_global_scene_document("piper_1", view)


def test_build_global_scene_document_rejects_a_sibling_robot_id():
    document = global_scene_document()
    view = robot_scene_view(
        document, "piper_1", joint_names=DEVICE_JOINTS["piper_1"]
    )
    with pytest.raises(ValueError, match="ROS-safe"):
        build_global_scene_document("piper/1", view)


# ---------------------------------------------------------------------------
# stop: bounded, session-scoped, idempotent cleanup
# ---------------------------------------------------------------------------


def test_stop_is_orderly_scoped_and_idempotent(tmp_path):
    foreign_container = "rocobrick-multi-robot-session-foreign"
    foreign_process = "python3 /somewhere/else/p05_host_bridge.py"
    runner = make_runner(
        tmp_path,
        leftover_containers=(foreign_container,),
        foreign_processes=(foreign_process,),
    )
    session = make_session(tmp_path, runner)
    session.start()
    session.observe_sync("piper_1")
    evidence = session.stop()

    assert evidence.status is RuntimeSessionStatus.STOPPED
    assert evidence.cleanup["passed"] is True
    assert evidence.cleanup["container_removed"] is True
    assert evidence.cleanup["container_stop"] == "exit=0 timed_out=False"
    assert evidence.cleanup["host_stopped"] is True
    assert evidence.cleanup["host_orderly_self_exit"] is True
    assert evidence.cleanup["host_evidence_orderly"] is True
    assert evidence.cleanup["leftovers"] == {
        "scope": str(tmp_path / "session"),
        "session_containers": [],
        "session_processes": [],
    }
    assert evidence.cleanup["container_log"] == str(
        tmp_path / "session" / "dual-launch.log"
    )
    # The foreign container and the foreign process are never claimed as this
    # session's leftovers, and no command ever touches them.
    serialized = json.dumps(evidence.cleanup)
    assert foreign_container not in serialized
    assert "somewhere/else" not in serialized
    commands = docker_commands(runner)
    for command in commands:
        if command[1] in ("stop", "rm"):
            assert command[-1] == runner.container_names[0]
    # Every container listing of this session is scoped to its own name.
    assert runner.ps_filters
    assert all(
        filter_value == f"name={runner.container_names[0]}"
        for filter_value in runner.ps_filters
    )
    # Every leftover scan is scoped to the unique session directory.
    assert all(
        pattern == str(tmp_path / "session")
        for pattern in runner.process_patterns
    )

    # Idempotent: a second stop returns the same evidence and runs nothing.
    commands_after = len(runner.commands)
    assert session.stop() is evidence
    assert len(runner.commands) == commands_after


def test_stop_escalates_int_then_kill_and_fails_cleanup(tmp_path):
    runner = make_runner(
        tmp_path,
        host_self_exit_after=None,
        host_ignores_int=True,
        host_ignores_kill=True,
    )
    # The fake bridge claims a completed hold, so the bounded teardown grace
    # (shrunk here) runs before the escalation; the process then ignores both
    # signals and the cleanup fails honestly.
    session = make_session(tmp_path, runner, host_teardown_grace_s=5.0)
    session.start()
    evidence = session.stop()
    handle = next(iter(runner._handle_by_pid.values()))
    assert handle.signals[:2] == [2, 9]
    assert evidence.cleanup["host_stopped"] is False
    assert evidence.cleanup["host_exit_kind"] == "undying"
    assert evidence.cleanup["passed"] is False
    assert any("did not stop" in error for error in evidence.errors)
    # Same lifecycle vocabulary as the single-robot session: the status says
    # the stack WAS ready; the honest teardown verdict is cleanup["passed"].
    assert evidence.status is RuntimeSessionStatus.STOPPED
    # Bounded budget: 5 s reap + 5 s teardown grace + 60 s SIGINT + 15 s
    # SIGKILL + the session-scoped leftover scan. The natural short hold does
    # not affect this undying-host path.
    assert runner.clock.now == pytest.approx(92.0)


def test_early_stop_interrupts_the_host_promptly_with_evidenced_cleanup(tmp_path):
    """The A9 scenario: an early zero-motion stop must not outwait the hold.

    The bridge has NOT completed its hold (no hold evidence, and it would hold
    far longer than the session), so the stop interrupts it right after the
    container is down. Without the bridge's own accepted-early-stop
    attestation the session reports the teardown as complete but NOT passing -
    it never claims the hold finished, and there is no session-side flag that
    could turn the SIGINT facts alone into a pass.
    """
    runner = make_runner(
        tmp_path,
        host_self_exit_after=None,
        host_writes_evidence=False,
    )
    session = make_session(tmp_path, runner)
    session.start()
    session.observe_sync("piper_1")
    evidence = session.stop()

    cleanup = evidence.cleanup
    assert cleanup["container_removed"] is True
    assert cleanup["host_stopped"] is True
    assert cleanup["host_exit_kind"] == "sigint"
    assert cleanup["host_orderly_self_exit"] is False
    # The bridge's own SIGINT evidence is recorded, but it is not an accepted
    # terminal state without its attestation: the cleanup does not pass.
    assert cleanup["host_evidence_orderly"] is False
    assert cleanup["host_early_stop_source"] is None
    assert cleanup["passed"] is False
    host_evidence = cleanup["host_evidence"]
    assert host_evidence["hold_completed"] is False
    assert host_evidence["intentional_stop"] is True
    assert host_evidence["ready"] is True
    assert host_evidence["stopped"] == "sigint_orderly_stop"
    assert host_evidence["shutdown_orderly"] is True
    assert host_evidence["early_stop_accepted"] is False
    assert host_evidence["rclpy_shutdown"] is True
    assert host_evidence["world_stopped"] is True
    assert host_evidence["timeline_stopped"] is True
    assert host_evidence["app_close_initiated"] is True
    assert any(
        "accepted intentional stop" in error for error in evidence.errors
    )
    # If a host ignores the cooperative short-hold exit, stop escalates only
    # after the bounded natural-exit grace.
    stop_index = runner.events.index(("docker", "stop"))
    signal_index = runner.events.index(("signal", 2))
    assert stop_index < signal_index
    assert evidence.cleanup["container_stop"] == "exit=0 timed_out=False"
    assert runner.clock.now == pytest.approx(90.0)


def test_an_accepted_early_stop_honors_the_bridge_attestation(tmp_path):
    """A4's ``early_stop_accepted`` key makes the prompt stop pass as-is."""
    runner = make_runner(
        tmp_path,
        host_self_exit_after=None,
        host_writes_evidence=False,
        host_early_stop_accepted=True,
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    cleanup = evidence.cleanup
    assert cleanup["passed"] is True
    assert cleanup["host_exit_kind"] == "sigint"
    assert cleanup["host_evidence_orderly"] is True
    assert cleanup["host_early_stop_source"] == "bridge_early_stop_accepted"
    assert cleanup["host_evidence"]["hold_completed"] is False
    assert cleanup["host_evidence"]["early_stop_accepted"] is True
    # The bridge's zero-motion counter is cross-checked, never assumed.
    assert cleanup["host_evidence"]["commands_applied"] == 0
    assert cleanup["host_orderly_self_exit"] is False
    assert evidence.errors == ()
    assert runner.clock.now == pytest.approx(90.0)


def test_controller_hold_commands_do_not_invalidate_orderly_stop(tmp_path):
    """Applied startup commands stay audited while cleanup can pass."""
    runner = make_runner(
        tmp_path,
        host_self_exit_after=None,
        host_writes_evidence=False,
        host_early_stop_accepted=True,
        host_commands_applied=1,
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    assert evidence.cleanup["host_evidence"]["commands_applied"] == 1
    assert evidence.cleanup["host_evidence_orderly"] is True
    assert evidence.cleanup["host_early_stop_source"] == "bridge_early_stop_accepted"
    assert evidence.cleanup["passed"] is True
    assert evidence.errors == ()


def test_there_is_no_session_side_early_stop_bypass():
    """Only the bridge's own attestation can accept an early stop.

    The contract requires A4's ``early_stop_accepted`` (ready bridge,
    bound SIGINT and orderly shutdown); the session must
    offer no flag, config field or code path that turns raw SIGINT facts into
    an accepted stop, and the CLI must never set one.
    """
    source = (SRC / "runtime_session.py").read_text(encoding="utf-8")
    assert "accept_sigint_stop_facts" not in source
    fields = {field.name for field in dataclasses.fields(MultiRobotSessionConfig)}
    assert not any(name.startswith("accept_") for name in fields), sorted(fields)
    for name in ("piper_run.py", "multi_robot_runtime.py"):
        path = SRC / name
        if path.is_file():
            assert "accept_sigint_stop_facts" not in path.read_text(
                encoding="utf-8"
            )


def test_an_attestation_without_this_sessions_sigint_is_refused(tmp_path):
    """The attestation must match this session's own SIGINT teardown."""
    runner = make_runner(
        tmp_path,
        host_self_exit_after=2.0,
        host_writes_evidence=False,
        host_prewritten_stop_evidence=True,
        host_early_stop_accepted=True,
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    # The bridge exited on its own and was never interrupted by this session.
    assert runner.events  # signals are recorded as events too
    assert ("signal", 2) not in runner.events
    assert evidence.cleanup["host_exit_kind"] == "self_exit"
    assert evidence.cleanup["host_evidence"]["early_stop_accepted"] is True
    assert evidence.cleanup["host_evidence"]["stopped"] == "sigint_orderly_stop"
    assert evidence.cleanup["host_evidence_orderly"] is False
    assert evidence.cleanup["passed"] is False
    assert any(
        "did not interrupt it with SIGINT" in error for error in evidence.errors
    )


def test_a_cancellation_marker_alone_never_accepts_an_early_stop(tmp_path):
    """No marker string is an acceptance path: only the bridge's key attests.

    A6 does not pattern-match the bridge's internal stop markers as an
    acceptance basis - the attestation key and orderly shutdown are what
    count, so a generic Kit cancellation leaves the
    cleanup failing exactly like an unattested SIGINT.
    """
    runner = make_runner(
        tmp_path,
        host_self_exit_after=None,
        host_writes_evidence=False,
        host_early_stop_accepted=True,
        host_commands_applied=0,
    )
    session = make_session(tmp_path, runner)
    session.start()
    # The bridge attests an accepted stop but reports a generic Kit
    # cancellation instead of its SIGINT marker: the attestation is refused.
    original = runner.write_host_stop_evidence

    def cancelled() -> None:
        original()
        path = tmp_path / "session" / "host-bridge-evidence.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["stopped"] = "kit_child_cancelled_orderly_stop"
        path.write_text(json.dumps(document), encoding="utf-8")

    runner.write_host_stop_evidence = cancelled
    evidence = session.stop()
    assert evidence.cleanup["host_evidence"]["intentional_stop"] is False
    assert evidence.cleanup["host_evidence"]["early_stop_accepted"] is True
    assert evidence.cleanup["host_evidence_orderly"] is False
    assert evidence.cleanup["host_early_stop_source"] is None
    assert evidence.cleanup["passed"] is False
    assert any(
        "does not record the SIGINT stop" in error for error in evidence.errors
    )


def test_a_finished_bridge_gets_bounded_teardown_time_without_a_signal(tmp_path):
    """A bridge whose own evidence proves its hold completed is not killed."""
    runner = make_runner(tmp_path, host_self_exit_after=30.0)
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    handle = next(iter(runner._handle_by_pid.values()))
    assert handle.signals == []
    assert evidence.cleanup["host_exit_kind"] == "self_exit"
    assert evidence.cleanup["host_evidence"]["hold_completed"] is True
    assert evidence.cleanup["host_evidence_orderly"] is True
    assert evidence.cleanup["host_orderly_self_exit"] is True
    assert evidence.cleanup["passed"] is True
    # The graces cost nothing here: the bridge exits on its own at 30 s,
    # inside the bounded teardown grace, and no leftover scan retries.
    assert runner.clock.now == pytest.approx(30.0)


def test_stop_fails_when_the_container_never_disappears(tmp_path):
    runner = make_runner(tmp_path, removal_verify_flaky=10_000)
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    assert evidence.cleanup["container_removed"] is False
    assert evidence.cleanup["passed"] is False
    assert any("could not be removed" in error for error in evidence.errors)


def test_stop_without_orderly_host_evidence_is_recorded_and_not_passing(tmp_path):
    runner = make_runner(
        tmp_path, host_writes_evidence=False, host_self_exit_after=2.0
    )
    session = make_session(tmp_path, runner)
    session.start()
    evidence = session.stop()
    # The host exited on its own but wrote nothing about an orderly shutdown.
    assert evidence.cleanup["host_exit_kind"] == "self_exit"
    assert evidence.cleanup["host_evidence"] == {}
    assert evidence.cleanup["host_evidence_orderly"] is False
    assert evidence.cleanup["host_orderly_self_exit"] is False
    assert evidence.cleanup["passed"] is False
    assert any(
        "intentional stop with an orderly shutdown" in error
        for error in evidence.errors
    )


def test_stop_without_start_is_safe_and_touches_nothing(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    evidence = session.stop()
    assert evidence.status is RuntimeSessionStatus.FAILED
    assert evidence.cleanup["container_stop"] == "not_started"
    assert evidence.cleanup["container_removed"] is True
    assert evidence.cleanup["host_exit_kind"] == "not_started"
    assert evidence.cleanup["passed"] is True
    assert docker_commands(runner) == []


def test_stop_after_a_failed_start_cleans_up_only_what_it_started(tmp_path):
    runner = make_runner(tmp_path, readiness_by_robot={"piper_1": False})
    session = make_session(tmp_path, runner)
    with pytest.raises(MultiRobotSessionError):
        session.start()
    evidence = session.stop()
    assert evidence.status is RuntimeSessionStatus.FAILED
    assert evidence.cleanup["container_removed"] is True
    assert evidence.cleanup["host_stopped"] is True
    assert evidence.cleanup["passed"] is True
    stops = [c for c in docker_commands(runner) if c[1] == "stop"]
    assert stops and stops[0][-1] == runner.container_names[0]


def test_the_context_manager_starts_and_always_stops(tmp_path):
    runner = make_runner(tmp_path)
    with make_session(tmp_path, runner) as session:
        assert session.status is RuntimeSessionStatus.READY
        assert session.observe_sync("piper_0")["robots"]["piper_0"]["opening_m"] == (
            pytest.approx(0.062)
        )
    assert session.status is RuntimeSessionStatus.STOPPED
    assert session.stop().cleanup["passed"] is True


# ---------------------------------------------------------------------------
# Topology and scope guards
# ---------------------------------------------------------------------------


def test_only_read_only_kinds_are_whitelisted():
    assert {
        kind.value for kind in READ_ONLY_OPERATION_KINDS
    } == {"READINESS", "SCENE_QUERY"}


def test_the_read_only_whitelist_covers_every_operation_kind():
    covered = {kind.value for kind in OperationKind}
    assert covered - {kind.value for kind in READ_ONLY_OPERATION_KINDS} == {
        "JOINT_TARGET",
        "LINK6_POSE",
        "TCP_POSE",
        "GRIPPER_TARGET",
        "APPLY_PLANNING_SCENE",
    }


def test_no_production_module_uses_the_historical_dry_run_session():
    for name in (
        "runtime_session.py",
        "piper_run.py",
        "multi_robot_scheduler.py",
        "robot_operation_runner.py",
        "robot_operation_executor.py",
    ):
        path = SRC / name
        if not path.is_file():  # pragma: no cover - module set may grow
            continue
        assert "DryRunMultiRobotSession" not in path.read_text(encoding="utf-8"), (
            f"{name} must not use the historical dry-run session"
        )


def test_multi_robot_runtime_reexports_the_live_session_seam():
    from rocobrick import multi_robot_runtime, runtime_session

    for name in (
        "MultiRobotIsaacMoveItSession",
        "MultiRobotSessionConfig",
        "MultiRobotSessionError",
        "MultiRobotReadinessEvidence",
        "MultiRobotSessionEvidence",
        "MultiRobotObservationEvidence",
        "build_global_scene_document",
        # The runner seam the plan entry point constructs the session with.
        "SubprocessRunner",
        "ProcessRunner",
        "ProcessHandle",
        "RunOutcome",
        "RuntimeSessionError",
        "RuntimeSessionStatus",
    ):
        assert getattr(multi_robot_runtime, name) is getattr(
            runtime_session, name
        ), name
    # The historical dry-run helper stays importable for its own legacy test,
    # but is not part of the production seam.
    assert "historical" in (multi_robot_runtime.__doc__ or "").lower()


def test_no_motion_or_controller_command_is_ever_issued(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    session.start()
    session.observe_sync("piper_0")
    session.observe_sync("piper_1")
    session.stop()
    assert_motion_free(runner.commands)


def test_the_session_never_starts_a_second_ros_domain_or_robot_process(tmp_path):
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner)
    session.start()
    session.observe_sync("piper_1")
    assert len(runner.launches) == 1
    assert len(runner.container_names) == 1
    runs = [c for c in docker_commands(runner) if c[1] == "run"]
    assert len(runs) == 1
    # One domain, one container: every recorded session process/env agrees.
    assert runner.launches[0][1]["ROS_DOMAIN_ID"] == "73"
    assert runner.commands[2][0][runner.commands[2][0].index("-e") + 1] == (
        "ROS_DOMAIN_ID=73"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOTION_MARKERS = (
    "follow_joint_trajectory",
    "move_action",
    "isaac_joint_command",
    "ros2 action",
    "ros2 topic pub",
    "compute_ik",
)


def assert_motion_free(commands) -> None:
    """No recorded command line carries a motion or controller endpoint."""
    for command, _ in commands:
        text = " ".join(command)
        for marker in MOTION_MARKERS:
            assert marker not in text, f"unexpected motion command: {text}"


def appended_observations(tmp_path: Path) -> list:
    path = tmp_path / "session" / "multi-robot-session.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    return list(document["observations"])
