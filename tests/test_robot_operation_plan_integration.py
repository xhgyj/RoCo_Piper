"""Integration acceptance tests for the full example1 operation plan.

The tests drive the ONLY production dispatch entry
(``piper-run --robot-operation-plan``) against the frozen example1 artifacts
(the STAP plan fixture, the external ``robot-operation-plan/v1`` document, the
scene manifest and the task directory) while the A6 multi-robot session is
replaced by an injected test double at its documented construction seam
(``piper_run._create_session``). The double supplies readiness evidence and
the read-only per-robot scene observation transport; nothing here starts
Isaac, ROS, Docker, MoveIt, a GPU job, or any motion, and no repository input
byte is ever modified.

Covered acceptance properties:

* the full example1 plan schedules its four layers, decides layer 0 by
  evidence and blocks the dependent layers, always without motion;
* a WAIT is confirmed only from the executor robot's own injected
  observation with a real sim-time advance and a stable measured AABB;
* a pending operation is settled by a layer deadline or a peer failure with
  explicit cancel evidence and no unsafe terminal;
* every motion skill fails closed with ``CAPABILITY_UNAVAILABLE`` before any
  command and without any observation;
* original inputs (STAP plan, external operation plan, task files, config)
  are byte-identical before and after every run and the captured copy of the
  external plan is verbatim;
* a session that reports any robot not ready dispatches zero operations;
* the production entry constructs the real executor and the real live
  session class: there is no production fake/dry-run fallback and the
  acceptance evidence carries no dry-run marker.
"""

from __future__ import annotations

import ast
import contextlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from pathlib import Path

import pytest

from rocobrick import piper_run
from rocobrick.piper_run import main
from rocobrick.robot_operation_executor import (
    REASON_CAPABILITY_UNAVAILABLE,
    SkillOperationExecutor,
)

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src/rocobrick"
PIPER_RUN_SOURCE = SRC / "piper_run.py"
TASK_DIR = ROOT / "tasks/type1/example1"
STAP_PLAN = ROOT / "runs/scene-smoke-20260914T093942Z/input/plan.json"
OPERATION_PLAN = ROOT / "artifacts/scene/example1-robot-operation-plan.json"
SCENE_MANIFEST = ROOT / "artifacts/scene/example1-scene-manifest.json"
USER_CONFIG = ROOT / "config/user_config.json"

PLAN_ROBOTS = ("piper_0", "piper_1")
SCENE_STATE_SCHEMA = "rocobrick-scene-state/v1"
PART_BRICK = "brick-768de5a17827920c48d4"
SUPPORT_BRICK = "brick-4bbba49bec24496bb21e"
TRANSIT_TO_PICK = "example1-piper0-transit-to-pick"
PIPER0_PICK = "example1-piper0-pick"
PIPER0_TRANSIT_TO_PLACE = "example1-piper0-transit-to-place"
PIPER0_PLACE_DOWN = "example1-piper0-place-down"
PIPER1_WAIT = "example1-piper1-wait-stable"

PART_BRICK_MIN = (0.30, -0.26, 0.0)
PART_BRICK_MAX = (0.332, -0.244, 0.016)
SUPPORT_BRICK_MIN = (0.34, -0.26, 0.0)
SUPPORT_BRICK_MAX = (0.372, -0.244, 0.016)

# Every repository file an acceptance run must leave byte-identical.
ANCHORED_INPUTS = (
    STAP_PLAN,
    OPERATION_PLAN,
    USER_CONFIG,
    SCENE_MANIFEST,
    TASK_DIR / "structure_start.json",
    TASK_DIR / "structure_goal.json",
)


def _read_json(path: Path) -> dict:
    """Load one JSON document from disk.

    Args:
        path: File to read.

    Returns:
        The decoded document.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _digests(paths: Sequence[Path]) -> dict[str, str]:
    """Hash every given file from disk.

    Args:
        paths: Files to hash.

    Returns:
        Mapping of absolute path string to SHA-256 hex digest.
    """
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}


def _invoke(argv: Sequence[str]) -> tuple[int, dict]:
    """Run ``piper_run.main`` with captured stdout and stderr.

    Args:
        argv: Command-line arguments for ``main``.

    Returns:
        The exit code and the single parsed stdout JSON line.

    Raises:
        AssertionError: If stdout does not carry exactly one JSON line.
    """
    stdout, stderr = StringIO(), StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(list(argv))
    lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1, stdout.getvalue()
    return code, json.loads(lines[0])


# ---------------------------------------------------------------------------
# Injected session observation (test double at the A6 session seam)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _InjectedReadiness:
    """Readiness evidence of the injected test session.

    The shape mirrors the real ``MultiRobotReadinessEvidence`` fields the plan
    entry point validates: ``robots``, ``readiness_by_robot``, ``all_ready``
    and ``passed``.
    """

    robots: tuple[str, ...]
    readiness_by_robot: dict[str, dict[str, object]]
    all_ready: bool
    passed: bool

    def serialize(self) -> dict[str, object]:
        """Return the JSON-safe readiness document.

        Returns:
            The readiness mapping handed to the plan entry point.
        """
        return {
            "mode": "injected-test-session",
            "robots": list(self.robots),
            "readiness_by_robot": {
                robot: dict(report)
                for robot, report in self.readiness_by_robot.items()
            },
            "all_ready": self.all_ready,
            "passed": self.passed,
        }


def _ready_readiness(
    robots: Sequence[str] = PLAN_ROBOTS,
    *,
    not_ready: Sequence[str] = (),
) -> _InjectedReadiness:
    """Build readiness evidence for the given robots.

    Args:
        robots: Robot ids the session reports, in plan order.
        not_ready: Robot ids whose own report must say ``passed`` is false.

    Returns:
        The readiness evidence document of the injected session.
    """
    unusable = set(not_ready)
    reports = {
        robot: {
            "robot_id": robot,
            "passed": robot not in unusable,
            "components": {
                name: robot not in unusable
                for name in (
                    "joint_states",
                    "controllers",
                    "move_group",
                    "fk",
                    "scene_state",
                )
            },
        }
        for robot in robots
    }
    ready = all(report["passed"] is True for report in reports.values())
    return _InjectedReadiness(
        robots=tuple(robots),
        readiness_by_robot=reports,
        all_ready=ready,
        passed=ready,
    )


def _cleanup_document(passed: bool, *, container: str) -> dict:
    """Build A6-shaped session cleanup evidence.

    Args:
        passed: Whether every owned termination flag is proven passing.
        container: Container identity this cleanup claims to have removed.

    Returns:
        The cleanup mapping of one stopped session envelope.
    """
    return {
        "scope": {"container": container},
        "passed": passed,
        "container_removed": passed,
        "host_stopped": passed,
        "host_evidence_orderly": passed,
        "leftovers": {"session_containers": [], "session_processes": []},
    }


@dataclass(frozen=True)
class _InjectedSessionEvidence:
    """Session stop evidence of the injected test session."""

    status: str
    robots: tuple[str, ...]
    cleanup: Mapping[str, object]
    errors: tuple[str, ...] = ()

    def serialize(self) -> dict[str, object]:
        """Return the JSON-safe session evidence document.

        Returns:
            The session evidence mapping handed to the plan entry point.
        """
        return {
            "status": self.status,
            "robots": list(self.robots),
            "cleanup": dict(self.cleanup),
            "errors": list(self.errors),
        }


class _InjectedSession:
    """Offline stand-in for the ONE real A6 multi-robot session.

    It starts no process and observes nothing by itself: readiness and every
    per-robot read-only observation come from the injected reader, and every
    lifecycle/observation call is appended to the shared call log so the tests
    can prove which robot was observed, that no motion was dispatched, and
    that the session was always stopped.
    """

    def __init__(
        self,
        session_dir: Path,
        *,
        robot_ids: tuple[str, ...],
        calls: list[tuple[object, ...]],
        readiness: _InjectedReadiness | None = None,
        reader: Callable[[str], Mapping[str, object]] | None = None,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
        cleanup_passed: bool = True,
        mutate_path: Path | None = None,
    ) -> None:
        """Store the scripted behavior and the shared call log."""
        self.session_dir = Path(session_dir)
        self.robot_ids = tuple(robot_ids)
        self._calls = calls
        self._readiness = readiness or _ready_readiness(robot_ids)
        self._reader = reader or _advancing_reader()
        self._start_error = start_error
        self._stop_error = stop_error
        self._cleanup_passed = cleanup_passed
        self._mutate_path = mutate_path

    def start(self) -> _InjectedReadiness:
        """Record the start, optionally mutate an input, answer readiness.

        Returns:
            The scripted readiness evidence.

        Raises:
            Exception: The scripted start error, when configured.
        """
        self._calls.append(("start",))
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self._mutate_path is not None:
            self._mutate_path.write_text(
                json.dumps({"mutated": True}), encoding="utf-8"
            )
        if self._start_error is not None:
            raise self._start_error
        return self._readiness

    async def observe(self, robot_id: str) -> Mapping[str, object]:
        """Record and answer one read-only whole-scene observation.

        Args:
            robot_id: Robot block the observation must carry.

        Returns:
            The injected scene-state document of that robot.
        """
        self._calls.append(("observe", robot_id))
        return self._reader(robot_id)

    def stop(self) -> _InjectedSessionEvidence:
        """Record the stop and answer the scripted session evidence.

        Returns:
            The session evidence carrying the cleanup verdict.

        Raises:
            Exception: The scripted stop error, when configured.
        """
        self._calls.append(("stop",))
        if self._stop_error is not None:
            raise self._stop_error
        return _InjectedSessionEvidence(
            status="STOPPED",
            robots=tuple(self._readiness.robots),
            cleanup=_cleanup_document(
                self._cleanup_passed, container="injected-test-session"
            ),
        )


def _install_injected_session(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[object, ...]],
    **kwargs: object,
) -> list[_InjectedSession]:
    """Patch ``piper_run._create_session`` with the injected session factory.

    The factory keeps the production signature (session directory, parsed
    arguments, validated plan robots), so the CLI is exercised unmodified
    apart from the documented session construction seam.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        calls: Mutable list recording lifecycle and observation calls.
        **kwargs: Constructor overrides for the injected session.

    Returns:
        The list of created injected sessions.
    """
    created: list[_InjectedSession] = []

    def factory(session_dir: Path, args: object, robot_ids: Sequence[str]):
        session = _InjectedSession(
            Path(session_dir),
            robot_ids=tuple(robot_ids),
            calls=calls,
            **kwargs,
        )
        session.plan_robots = tuple(robot_ids)
        session.arguments = args
        created.append(session)
        return session

    monkeypatch.setattr(piper_run, "_create_session", factory)
    return created


# ---------------------------------------------------------------------------
# Injected observations
# ---------------------------------------------------------------------------


def _scene_document(robot_id: str, sim_time_s: float) -> dict:
    """Build one whole-scene ``rocobrick-scene-state/v1`` observation.

    Both example1 physical instances are reported so the WAIT of the frozen
    example1 plan (which observes the support brick) stays observable, and the
    robot block belongs to the requested robot only.

    Args:
        robot_id: Robot whose block the observation carries.
        sim_time_s: Observed robot and shared simulation time.

    Returns:
        The scene-state document returned by the injected transport.
    """
    return {
        "schema": SCENE_STATE_SCHEMA,
        "sim_time_s": sim_time_s,
        "robots": {
            robot_id: {
                "robot_id": robot_id,
                "sim_time_s": sim_time_s,
                "gripper": {"gripper_joint1": 0.0, "gripper_joint2": 0.0},
                "opening_m": 0.02,
            }
        },
        "bricks": {
            PART_BRICK: {
                "aabb_min": list(PART_BRICK_MIN),
                "aabb_max": list(PART_BRICK_MAX),
            },
            SUPPORT_BRICK: {
                "aabb_min": list(SUPPORT_BRICK_MIN),
                "aabb_max": list(SUPPORT_BRICK_MAX),
            },
        },
        "connections": {},
    }


def _advancing_reader() -> Callable[[str], Mapping[str, object]]:
    """Build a reader whose observed robot sim time advances per call.

    Returns:
        A reader answering the first call of a robot with 0.0 s and every
        later call with 5.0 s.
    """
    counters: dict[str, int] = {}

    def reader(robot_id: str) -> Mapping[str, object]:
        index = counters.get(robot_id, 0)
        counters[robot_id] = index + 1
        return _scene_document(robot_id, 0.0 if index == 0 else 5.0)

    return reader


def _stagnant_reader() -> Callable[[str], Mapping[str, object]]:
    """Build a reader whose observed robot sim time never advances.

    Returns:
        A reader that always answers the same simulation time, so a WAIT can
        never be confirmed and must be settled by a cancel or a deadline.
    """
    return lambda robot_id: _scene_document(robot_id, 4.0)


# ---------------------------------------------------------------------------
# CLI invocation helpers
# ---------------------------------------------------------------------------


def _dispatch_argv(
    tmp_path: Path,
    *,
    operation_plan: Path,
    run_dir: Path,
    stap_plan: Path = STAP_PLAN,
) -> list[str]:
    """Build a ``--robot-operation-plan`` invocation with its scene group.

    Args:
        tmp_path: Temporary directory root for the scene group outputs.
        operation_plan: External operation plan file.
        run_dir: Fresh evidence directory to create.
        stap_plan: STAP plan file the operation plan binds to.

    Returns:
        The argv list for ``piper_run.main``.
    """
    return [
        "--task", str(TASK_DIR),
        "--plan", str(stap_plan),
        "--backend", "isaac",
        "--run-dir", str(run_dir),
        "--robot-operation-plan", str(operation_plan),
        "--scene-manifest", str(SCENE_MANIFEST),
        "--scene-snapshot-out", str(tmp_path / "scene-snapshot.json"),
        "--host-instance-id", "injected-integration-host",
        "--config", str(USER_CONFIG),
    ]


def _read_run(line: dict) -> tuple[Path, dict, dict, dict, list[dict]]:
    """Load every evidence document of one dispatch run.

    Args:
        line: The parsed stdout JSON line of the run.

    Returns:
        The evidence directory, the preflight document, the
        ``robot-operation-plan-run.json`` document, the ``result.json``
        document and the decoded phase events in order.
    """
    evidence_dir = Path(line["result_path"]).parent
    events = [
        json.loads(raw)
        for raw in (evidence_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return (
        evidence_dir,
        _read_json(evidence_dir / "preflight.json"),
        _read_json(evidence_dir / "robot-operation-plan-run.json"),
        _read_json(evidence_dir / "result.json"),
        events,
    )


def _single_operation_plan(operation_id: str, *, plan_id: str) -> dict:
    """Build a strict single-operation plan from the frozen example1 plan.

    The operation document is copied verbatim from the committed
    ``robot-operation-plan/v1`` artifact, so every target/resource/evidence
    shape stays exactly what the frozen example1 plan declares.

    Args:
        operation_id: Operation id to isolate.
        plan_id: Plan id of the derived document.

    Returns:
        A ``robot-operation-plan/v1`` document with one layer and one
        operation.
    """
    document = _read_json(OPERATION_PLAN)
    operation = next(
        item
        for item in document["operations"]
        if item["operation_id"] == operation_id
    )
    return {
        "schema": "robot-operation-plan/v1",
        "plan_id": plan_id,
        "source_plan_id": document["source_plan_id"],
        "robots": list(document["robots"]),
        "operations": [operation],
        "dependencies": [],
        "execution_layers": [[operation_id]],
        "planned_schedule": [
            [
                {
                    "operation_id": operation_id,
                    "robot_id": operation["executor_robot_id"],
                }
            ]
        ],
        "evidence_requirements": [
            {
                "operation_id": operation_id,
                "evidence": operation["evidence_requirements"][0],
            }
        ],
    }


def _write_plan(
    tmp_path: Path, document: Mapping[str, object], name: str
) -> Path:
    """Write one derived operation plan document into the temp root.

    Args:
        tmp_path: Temporary directory root.
        document: Plan document to serialize.
        name: File name to use.

    Returns:
        The written plan path.
    """
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _run_full_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: list[tuple[object, ...]],
    reader: Callable[[str], Mapping[str, object]] | None = None,
    **session_kwargs: object,
) -> tuple[int, dict, Path, dict, dict, dict, list[dict]]:
    """Dispatch the frozen example1 plan with an injected session.

    Args:
        tmp_path: Temporary directory root of the run.
        monkeypatch: Pytest monkeypatch fixture.
        calls: Mutable list recording lifecycle and observation calls.
        reader: Injected scene reader override.
        **session_kwargs: Extra injected-session constructor overrides.

    Returns:
        The exit code, stdout line and the decoded evidence documents.
    """
    if reader is not None:
        session_kwargs["reader"] = reader
    _install_injected_session(monkeypatch, calls, **session_kwargs)
    code, line = _invoke(
        _dispatch_argv(
            tmp_path,
            operation_plan=OPERATION_PLAN,
            run_dir=tmp_path / "run",
        )
    )
    evidence_dir, preflight, document, result, events = _read_run(line)
    return code, line, evidence_dir, preflight, document, result, events


# ---------------------------------------------------------------------------
# Full example1 plan
# ---------------------------------------------------------------------------


def test_full_example1_plan_runs_zero_motion_with_injected_observation(
    tmp_path, monkeypatch
):
    """The frozen example1 plan runs its layers and never sends motion."""
    calls: list[tuple[object, ...]] = []
    created = _install_injected_session(monkeypatch, calls)
    instances: list[SkillOperationExecutor] = []
    real_init = SkillOperationExecutor.__init__

    def record_init(self, robot_ids, observation_transport, **kwargs):
        real_init(self, robot_ids, observation_transport, **kwargs)
        instances.append(self)

    monkeypatch.setattr(SkillOperationExecutor, "__init__", record_init)

    run_dir = tmp_path / "run"
    code, line = _invoke(
        _dispatch_argv(
            tmp_path, operation_plan=OPERATION_PLAN, run_dir=run_dir
        )
    )

    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    assert line["passed"] is False
    evidence_dir, preflight, document, result, events = _read_run(line)
    assert evidence_dir.is_relative_to(run_dir)

    # The real A7 executor was constructed exactly once for the plan robots
    # and received the injected session, so no fake executor ever ran.
    assert len(instances) == 1
    assert instances[0].robot_ids == PLAN_ROBOTS
    assert len(created) == 1
    assert created[0].plan_robots == PLAN_ROBOTS

    # Readiness of BOTH robots is proven before anything is dispatched.
    readiness = document["readiness"]
    assert readiness["robots"] == list(PLAN_ROBOTS)
    assert readiness["all_ready"] is True
    assert readiness["passed"] is True
    assert set(readiness["readiness_by_robot"]) == set(PLAN_ROBOTS)
    assert [
        entry
        for entry in readiness["readiness_by_robot"].values()
        if entry["passed"] is not True
    ] == []
    assert preflight["checks"]["robot_operation_plan_readiness_passed"] is True

    # Layer 0 (concurrent TRANSIT + WAIT) is the only layer that dispatches.
    assert document["schema"] == "rocobrick-operation-plan-run/v1"
    assert document["scheduler_ran"] is True
    assert document["scheduler_passed"] is False
    assert document["passed"] is False
    operations = document["operations"]
    assert set(operations) == {
        TRANSIT_TO_PICK,
        PIPER1_WAIT,
        PIPER0_PICK,
        PIPER0_TRANSIT_TO_PLACE,
        PIPER0_PLACE_DOWN,
    }

    # The motion skills fail closed before any motion command exists.
    transit = operations[TRANSIT_TO_PICK]
    assert transit["robot_id"] == "piper_0"
    assert transit["status"] == "FAILED"
    assert transit["passed"] is False
    assert transit["terminal"] is True
    assert transit["motion_sent"] is False
    assert transit["evidence"] == {}
    assert REASON_CAPABILITY_UNAVAILABLE in transit["reason"]

    # The dependent layers are blocked, not silently skipped or confirmed.
    for operation_id in (
        PIPER0_PICK,
        PIPER0_TRANSIT_TO_PLACE,
        PIPER0_PLACE_DOWN,
    ):
        blocked = operations[operation_id]
        assert blocked["status"] == "BLOCKED", operation_id
        assert blocked["passed"] is False, operation_id
        assert blocked["motion_sent"] is None, operation_id
        assert blocked["reason"] == "predecessor layer failed", operation_id

    # The piper_1 WAIT is decided by real observation or by a safe cancel.
    wait = operations[PIPER1_WAIT]
    assert wait["robot_id"] == "piper_1"
    assert wait["motion_sent"] is False
    assert wait["terminal"] is True
    if wait["status"] == "CONFIRMED":
        assert wait["passed"] is True
        assert set(wait["evidence"]) == {
            "real_scene_observation",
            "wait_duration_confirmed",
        }
        observed = wait["evidence"]["real_scene_observation"]
        assert observed["robot_id"] == "piper_1"
        assert observed["polls"] >= 2
        duration = wait["evidence"]["wait_duration_confirmed"]
        assert duration["observed_advance_s"] >= duration["requested_duration_s"]
    else:
        assert wait["status"] == "CANCELLED"
        assert wait["cancel"]["acknowledged"] is True
        assert wait["cancel"]["terminal"] is True
        assert wait["cancel"]["safe"] is True
        assert wait["cancel"]["evidence"]["motion_sent"] is False

    # Only validated plan robots are ever observed.
    observed_robots = [call[1] for call in calls if call[0] == "observe"]
    assert set(observed_robots) <= set(PLAN_ROBOTS)
    if wait["status"] == "CONFIRMED":
        assert set(observed_robots) == {"piper_1"}

    # Layer barriers and evidence follow the plan's execution layers.
    assert [
        (layer["layer_index"], layer["status"]) for layer in document["layers"]
    ] == [(0, "FAILED"), (1, "BLOCKED"), (2, "BLOCKED"), (3, "BLOCKED")]
    assert document["layers"][0]["operation_ids"] == [
        TRANSIT_TO_PICK,
        PIPER1_WAIT,
    ]
    for layer_index, operation_id in enumerate(
        (PIPER0_PICK, PIPER0_TRANSIT_TO_PLACE, PIPER0_PLACE_DOWN), start=1
    ):
        assert document["layers"][layer_index]["operation_ids"] == [
            operation_id
        ]

    # No terminal is unsafe: every failure is a terminal no-motion proof.
    assert document["unsafe_terminals"] == []

    # Resource leases exist for the dispatched layer only, and a confirmed
    # operation never keeps its lease.
    resources = document["resources"]
    assert resources
    assert {item["operation_id"] for item in resources} == {
        TRANSIT_TO_PICK,
        PIPER1_WAIT,
    }
    for item in resources:
        assert item["acquired"] is True
        assert isinstance(item["released"], bool)
        if item["released"] is False:
            assert operations[item["operation_id"]]["status"] != "CONFIRMED"

    # No motion, no applied command, clean scoped cleanup, immutable inputs.
    assert document["motion_sent"] is False
    assert preflight["checks"]["no_motion_dispatched"] is True
    assert document["cleanup_passed"] is True
    cleanup = document["cleanup"]["cleanup"]
    assert cleanup["passed"] is True
    assert cleanup["container_removed"] is True
    assert cleanup["host_stopped"] is True
    assert cleanup["leftovers"] == {
        "session_containers": [],
        "session_processes": [],
    }
    assert [call[0] for call in calls][0] == "start"
    assert calls[-1] == ("stop",)
    assert result["status"] == "EXECUTION_FAILED"
    assert result["robot_operation_plan_run"]["motion_sent"] is False

    # The acceptance evidence carries no dry-run/fake marker anywhere.
    serialized = json.dumps(document, sort_keys=True).lower()
    assert "dry_run" not in serialized
    assert "dry-run" not in serialized
    assert "fake" not in serialized


def test_full_example1_plan_leaves_every_original_input_untouched(
    tmp_path, monkeypatch
):
    """Every original input byte survives the run and the copy is verbatim."""
    before = _digests(ANCHORED_INPUTS)
    calls: list[tuple[object, ...]] = []
    code, _line, evidence_dir, _preflight, document, _result, _events = (
        _run_full_plan(tmp_path, monkeypatch, calls=calls)
    )
    assert code == 10
    assert _digests(ANCHORED_INPUTS) == before

    # The captured external plan is byte-identical to its source.
    captured = evidence_dir / "robot-operation-plan.json"
    assert captured.read_bytes() == OPERATION_PLAN.read_bytes()
    digest = sha256(OPERATION_PLAN.read_bytes()).hexdigest()
    metadata = _read_json(
        evidence_dir / "robot-operation-plan.json.metadata.json"
    )
    assert metadata == {
        "sha256": digest,
        "bytes": OPERATION_PLAN.stat().st_size,
    }
    assert document["inputs"]["sha256_before"]["robot_operation_plan"] == digest
    assert document["inputs"]["sha256_after_all_phases"] == document[
        "inputs"
    ]["sha256_before"]
    assert document["inputs"]["immutable_after_all_phases"] is True


def test_mutating_an_original_input_during_the_run_fails_the_dispatch(
    tmp_path, monkeypatch
):
    """A run that mutates its own external plan exits as INPUT_MUTATED."""
    source = tmp_path / "external-operation-plan.json"
    source.write_bytes(OPERATION_PLAN.read_bytes())
    calls: list[tuple[object, ...]] = []
    _install_injected_session(monkeypatch, calls, mutate_path=source)
    code, line = _invoke(
        _dispatch_argv(tmp_path, operation_plan=source, run_dir=tmp_path / "run")
    )
    assert code == 6
    assert line["status"] == "INPUT_MUTATED"
    assert line["passed"] is False
    _evidence_dir, _preflight, document, _result, _events = _read_run(line)
    assert document["inputs"]["immutable_after_all_phases"] is False
    assert document["scheduler_passed"] is False
    assert document["cleanup_passed"] is True
    assert calls[0] == ("start",)
    assert calls[-1] == ("stop",)


def test_waits_are_confirmed_only_from_the_executor_robots_observation(
    tmp_path, monkeypatch
):
    """A WAIT is confirmed by its own robot's advancing real observation."""
    plan = _write_plan(
        tmp_path,
        _single_operation_plan(PIPER1_WAIT, plan_id="wait-integration-v1"),
        "wait-plan.json",
    )
    calls: list[tuple[object, ...]] = []
    _install_injected_session(monkeypatch, calls, reader=_advancing_reader())
    code, line = _invoke(
        _dispatch_argv(tmp_path, operation_plan=plan, run_dir=tmp_path / "run")
    )
    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    _evidence_dir, _preflight, document, _result, _events = _read_run(line)
    wait = document["operations"][PIPER1_WAIT]
    assert wait["status"] == "CONFIRMED"
    assert wait["passed"] is True
    assert wait["motion_sent"] is False
    assert wait["terminal"] is True
    assert set(wait["evidence"]) == {
        "real_scene_observation",
        "wait_duration_confirmed",
    }
    observed = wait["evidence"]["real_scene_observation"]
    assert observed["robot_id"] == "piper_1"
    assert observed["physical_instance_id"] == SUPPORT_BRICK
    assert observed["polls"] >= 2
    assert observed["aabb_min"] == list(SUPPORT_BRICK_MIN)
    assert observed["aabb_max"] == list(SUPPORT_BRICK_MAX)
    duration = wait["evidence"]["wait_duration_confirmed"]
    assert duration["requested_duration_s"] == 1
    assert duration["observed_advance_s"] >= 1
    # The confirming observation came from the WAIT robot only.
    assert [call[1] for call in calls if call[0] == "observe"] == [
        "piper_1"
    ] * len([call for call in calls if call[0] == "observe"])
    assert not document["unsafe_terminals"]
    assert document["motion_sent"] is False


def test_a_stagnant_observation_never_confirms_a_wait(tmp_path, monkeypatch):
    """A non-advancing observation cannot confirm a WAIT by sleeping alone."""
    plan = _write_plan(
        tmp_path,
        _single_operation_plan(PIPER1_WAIT, plan_id="stagnant-wait-v1"),
        "stagnant-plan.json",
    )
    calls: list[tuple[object, ...]] = []
    _install_injected_session(monkeypatch, calls, reader=_stagnant_reader())
    monkeypatch.setattr(piper_run, "OPERATION_TIMEOUT_S", 0.6)
    code, line = _invoke(
        _dispatch_argv(tmp_path, operation_plan=plan, run_dir=tmp_path / "run")
    )
    assert code == 10
    _evidence_dir, _preflight, document, _result, _events = _read_run(line)
    wait = document["operations"][PIPER1_WAIT]
    assert wait["status"] == "CANCELLED"
    assert wait["passed"] is False
    assert wait["motion_sent"] is False
    assert wait["deadline_exceeded"] is True
    assert document["scheduler_passed"] is False
    assert document["unsafe_terminals"] == []
    assert document["motion_sent"] is False
    cancellations = document["cancellations"]
    assert [item["operation_id"] for item in cancellations] == [PIPER1_WAIT]
    assert cancellations[0]["acknowledged"] is True
    assert cancellations[0]["terminal"] is True
    assert cancellations[0]["safe"] is True
    assert cancellations[0]["evidence"]["motion_sent"] is False


def test_a_peer_failure_cancels_a_pending_wait_without_unsafe_terminal(
    tmp_path, monkeypatch
):
    """A same-layer peer failure cancels the WAIT safely, not silently."""
    calls: list[tuple[object, ...]] = []
    code, _line, _evidence_dir, _preflight, document, _result, _events = (
        _run_full_plan(tmp_path, monkeypatch, calls=calls, reader=_stagnant_reader())
    )
    assert code == 10
    wait = document["operations"][PIPER1_WAIT]
    assert wait["status"] == "CANCELLED"
    assert wait["passed"] is False
    assert wait["motion_sent"] is False
    assert wait["terminal"] is True
    assert wait["cancel"]["acknowledged"] is True
    assert wait["cancel"]["terminal"] is True
    assert wait["cancel"]["safe"] is True
    assert wait["cancel"]["evidence"]["motion_sent"] is False
    assert document["unsafe_terminals"] == []
    assert document["motion_sent"] is False
    assert document["cleanup_passed"] is True


@pytest.mark.parametrize(
    ("operation_id", "robot_id"),
    [
        (TRANSIT_TO_PICK, "piper_0"),
        (PIPER0_PICK, "piper_0"),
        (PIPER0_PLACE_DOWN, "piper_0"),
    ],
)
def test_motion_skills_fail_without_motion_or_observation(
    tmp_path, monkeypatch, operation_id, robot_id
):
    """Each motion skill fails closed before any motion or observation."""
    plan = _write_plan(
        tmp_path,
        _single_operation_plan(operation_id, plan_id=f"{operation_id}-v1"),
        "motion-plan.json",
    )
    calls: list[tuple[object, ...]] = []
    reader_calls: list[str] = []

    def reader(robot_id: str) -> Mapping[str, object]:
        reader_calls.append(robot_id)
        return _scene_document(robot_id, 0.0)

    _install_injected_session(monkeypatch, calls, reader=reader)
    code, line = _invoke(
        _dispatch_argv(tmp_path, operation_plan=plan, run_dir=tmp_path / "run")
    )
    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    _evidence_dir, _preflight, document, _result, _events = _read_run(line)
    operation = document["operations"][operation_id]
    assert operation["robot_id"] == robot_id
    assert operation["status"] == "FAILED"
    assert operation["passed"] is False
    assert operation["terminal"] is True
    assert operation["motion_sent"] is False
    assert operation["evidence"] == {}
    assert operation["reason"] == REASON_CAPABILITY_UNAVAILABLE
    assert operation["resources_released"] is False
    assert document["unsafe_terminals"] == []
    assert document["motion_sent"] is False
    # A refused motion skill never observes the scene either.
    assert reader_calls == []
    assert [call for call in calls if call[0] == "observe"] == []
    assert document["cleanup_passed"] is True


def test_a_not_ready_robot_dispatches_zero_operations(tmp_path, monkeypatch):
    """Any robot not ready aborts the dispatch before the scheduler runs."""
    calls: list[tuple[object, ...]] = []
    _install_injected_session(
        monkeypatch,
        calls,
        readiness=_ready_readiness(not_ready=("piper_1",)),
    )
    code, line = _invoke(
        _dispatch_argv(
            tmp_path, operation_plan=OPERATION_PLAN, run_dir=tmp_path / "run"
        )
    )
    assert code == 8
    assert line["status"] == "RUNTIME_START_FAILED"
    assert line["passed"] is False
    _evidence_dir, preflight, document, result, events = _read_run(line)
    assert document["scheduler_ran"] is False
    assert document["scheduler_passed"] is False
    assert document["operations"] == {}
    assert document["motion_sent"] is False
    assert "readiness_rejection" in document
    assert "piper_1" in document["readiness_rejection"]
    assert preflight["checks"].get(
        "robot_operation_plan_readiness_passed"
    ) is None
    assert document["cleanup_passed"] is True
    assert calls[0] == ("start",)
    assert calls[-1] == ("stop",)
    assert [call for call in calls if call[0] == "observe"] == []
    assert result["status"] == "RUNTIME_START_FAILED"
    assert "runtime_start" in [
        event["event"]["phase"] for event in events
    ]


def test_the_production_entry_has_no_fake_runtime_fallback():
    """The dispatch entry constructs the real session and the real executor."""
    source = PIPER_RUN_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    factory = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_create_session"
    )
    used = {
        node.id
        for node in ast.walk(factory)
        if isinstance(node, ast.Name)
    } | {
        node.attr
        for node in ast.walk(factory)
        if isinstance(node, ast.Attribute)
    }
    assert "MultiRobotIsaacMoveItSession" in used
    assert "MultiRobotSessionConfig" in used
    assert "SubprocessRunner" in used
    assert not {name for name in used if "fake" in name.lower()}
    assert not {name for name in used if "dry" in name.lower()}
    # The historical dry-run helper is not reachable from the dispatch entry.
    assert "DryRunMultiRobotSession" not in source
    # A disabled session check must still be a real session, never a fake:
    # the CLI rejects a missing scene group before any run directory exists.
    assert "requires --scene-manifest" in source
    assert "there is no fake runtime fallback" in source
