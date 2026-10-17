"""Offline tests for the formal ``piper-run`` preflight CLI.

Everything here is pure: the real example1 plan/task bytes, AST scans of the
new module, unit-injected TCP artifacts, and run-evidence documents produced
inside temporary run directories. No ROS, Isaac, GPU, Docker, MoveIt action,
or motion is ever started, and no repository input byte is ever modified.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import subprocess
import sys
from hashlib import sha256
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from rocobrick import piper_run
from rocobrick.piper_run import (
    EXIT_CODES,
    PREFLIGHT_SCHEMA,
    RESULT_SCHEMA,
    TCP_ARTIFACT_SCHEMA,
    RunStatus,
    main,
)

ROOT = Path(__file__).parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PIPER_RUN_SOURCE = ROOT / "src/rocobrick/piper_run.py"
PLAN = ROOT / "tests/fixtures/assembly_plans/example1-compound-pick.json"
TASK_DIR = ROOT / "tasks/type1/example1"
TASK_START = TASK_DIR / "structure_start.json"
TASK_GOAL = TASK_DIR / "structure_goal.json"

PLAN_ID = "assembly-plan-25a5d5cc586a1e5c65f2141d"
TASK_ID = "assemble-768de5a17827920c48d4"
ROBOT = "piper_0"
OPERATION_PLAN = (
    ROOT / "tests/fixtures/robot_operation_plans/example1-robot-operation-plan.json"
)

FAKE_PINNED_SOURCE = "audit/fake/pinned_tool.urdf"
FAKE_PINNED_HASH = "a" * 64

BANNED_IMPORT_ROOTS = (
    "rclpy",
    "rospy",
    "moveit_msgs",
    "sensor_msgs",
    "geometry_msgs",
    "moveit",
    "tf2_ros",
    "isaacsim",
    "omni",
    "pxr",
    "tools",
    "numpy",
    "torch",
)
ALLOWED_ROCOBRICK = {
    "rocobrick.compat",
    "rocobrick.evidence",
    "rocobrick.tcp_capability",
    "rocobrick.tcp_contract",
    "rocobrick.robot_operation_plan",
    # The ONLY production dispatch entry (--robot-operation-plan): the
    # multi-robot session, the strict plan runner and the zero-motion skill
    # executor are imported lazily, only after every immutable gate and the
    # TCP authorization have passed, and only for explicit dispatch.
    "rocobrick.multi_robot_runtime",
    "rocobrick.skills.target_context",
    "rocobrick.adapters.moveit.handover_live",
    "rocobrick.execution.operation_state_ledger",
    "rocobrick.isaac_snapshot_host_adapter",
    "rocobrick.isaac_backend",
    "rocobrick.moveit_contract",
    "rocobrick.skills.contracts",
    "rocobrick.skills.handover",
    "rocobrick.skills.direct_handover",
    "rocobrick.skills.handover_profile",
    "rocobrick.skills.observation",
    "rocobrick.skills.targets",
    "rocobrick.skills.pick",
    "rocobrick.skills.top_offset_grasp",
    "rocobrick.skills.place_down",
    "rocobrick.skills.place_up",
    "rocobrick.skills.support_top",
    "rocobrick.plan.structure_scene",
    "rocobrick.execution.robot_operation_executor",
    "rocobrick.robot_operation_runner",
    "rocobrick.unified_runtime_facade",
}
BANNED_RUNTIME_PREFIXES = (
    "rclpy",
    "rospy",
    "moveit",
    "sensor_msgs",
    "geometry_msgs",
    "isaacsim",
    "omni",
    "pxr",
    "tools",
    "rocobrick.adapters",
    "rocobrick.execution",
    "rocobrick.moveit",
    "rocobrick.ros",
    "rocobrick.isaac",
    "rocobrick.bricksim",
    "rocobrick.skills",
    "rocobrick.scheduler",
    "rocobrick.planner_client",
    "rocobrick.contracts",
)
EXPECTED_EVIDENCE_FILES = {
    "robot-operation-plan.json",
    "robot-operation-plan.json.metadata.json",
    "robot-operation-plan-run.json",
    "structure_start.json",
    "structure_start.json.metadata.json",
    "structure_goal.json",
    "structure_goal.json.metadata.json",
    "provenance.json",
    "events.jsonl",
    "preflight.json",
    "result.json",
}


def input_hashes() -> dict[str, str]:
    """Hash the three repository inputs of the direct plan entry.

    Returns:
        The raw SHA-256 digests of the plan, start, and goal inputs.
    """
    return {
        "robot_operation_plan": sha256(OPERATION_PLAN.read_bytes()).hexdigest(),
        "structure_start": sha256(TASK_START.read_bytes()).hexdigest(),
        "structure_goal": sha256(TASK_GOAL.read_bytes()).hexdigest(),
    }


def invoke(argv: list[str]) -> tuple[int, dict, str, str]:
    """Run ``piper_run.main`` with captured stdout and stderr.

    Args:
        argv: The command-line arguments for ``main``.

    Returns:
        A tuple of the exit code, the single parsed stdout JSON line, the raw
        stdout text, and the raw stderr text.

    Raises:
        AssertionError: If stdout does not carry exactly one JSON line.
    """
    stdout, stderr = StringIO(), StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(argv)
    lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1, stdout.getvalue()
    return code, json.loads(lines[0]), stdout.getvalue(), stderr.getvalue()


def run_example1(run_dir: Path, *extra: str) -> tuple[int, dict]:
    """Run the direct preflight CLI against example1 with an explicit run dir.

    Every example1 run of this module stays offline: the invocation is always a
    preflight, so no runtime session and no motion is requested.

    Args:
        run_dir: Fresh evidence directory to create.
        *extra: Additional command-line flags.

    Returns:
        The exit code and the parsed stdout JSON line.
    """
    code, line, _, _ = invoke([
        "--robot-operation-plan", str(OPERATION_PLAN),
        "--structure-goal", str(TASK_GOAL),
        "--structure-start", str(TASK_START),
        "--backend", "isaac",
        "--run-dir", str(run_dir),
        "--preflight-only",
        *extra,
    ])
    return code, line


def read_evidence(line: dict) -> tuple[Path, Path, list[dict]]:
    """Load the evidence directory and event list for a run.

    Args:
        line: The parsed stdout JSON line of the run.

    Returns:
        The evidence directory, its preflight document path, and the decoded
        phase events in order.
    """
    evidence_dir = Path(line["result_path"]).parent
    events_path = evidence_dir / "events.jsonl"
    events = [
        json.loads(raw)
        for raw in events_path.read_text(encoding="utf-8").splitlines()
    ]
    return evidence_dir, evidence_dir / "preflight.json", events


def copy_task_dir(destination: Path) -> Path:
    """Copy the example1 task directory into a temporary location.

    Args:
        destination: Directory to populate with the task files.

    Returns:
        The copied task directory path.
    """
    destination.mkdir(parents=True)
    for source in (TASK_START, TASK_GOAL):
        (destination / source.name).write_bytes(source.read_bytes())
    return destination


def fake_artifact(overrides: dict | None = None) -> dict:
    """Build a self-consistent TCP audit artifact whose contract resolves.

    Args:
        overrides: Top-level keys to override in the artifact.

    Returns:
        An artifact mapping that ``assess_piper_l_tcp`` classifies as
        ``RESOLVED``.
    """
    item = {
        "id": "official-tool0",
        "kind": "pinned_explicit_transform",
        "source": FAKE_PINNED_SOURCE,
        "source_sha256": FAKE_PINNED_HASH,
        "declared_tool_frame": True,
        "parent_frame": "link6",
        "child_frame": "tool0",
        "translation_m": [0.0, 0.0, 0.1],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    artifact = {
        "schema": TCP_ARTIFACT_SCHEMA,
        "model": "agilex_piper_l",
        "passed": True,
        "inputs": {
            "pinned_files": [
                {"path": FAKE_PINNED_SOURCE, "sha256": FAKE_PINNED_HASH}
            ]
        },
        "evidence": [item],
        "tcp_contract": {"status": "RESOLVED"},
        "tcp_execution_allowed": True,
        "checks": {},
    }
    artifact.update(overrides or {})
    return artifact


def patch_artifact(
    monkeypatch: pytest.MonkeyPatch, artifact: dict | None, tmp_path: Path
) -> None:
    """Point the module's TCP artifact constant at a temporary artifact.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        artifact: The artifact document to write, or ``None`` to point the
            constant at a path that does not exist.
        tmp_path: Temporary directory root for the artifact file.
    """
    if artifact is None:
        target = tmp_path / "missing" / "artifact.json"
    else:
        target = tmp_path / "artifact.json"
        target.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(piper_run, "TCP_ARTIFACT", target)


# ---------------------------------------------------------------------------
# Status vocabulary, exit codes, and script registration
# ---------------------------------------------------------------------------


def test_status_vocabulary_and_exit_codes_are_exact():
    assert {member.value for member in RunStatus} == {
        "PREFLIGHT_ONLY",
        "INVALID_INPUT",
        "BINDING_REJECTED",
        "UNRESOLVED_TCP",
        "TCP_CONTRACT_INVALID",
        "INPUT_MUTATED",
        "EXECUTION_NOT_IMPLEMENTED",
        "RUNTIME_START_FAILED",
        "OPERATION_REJECTED",
        "EXECUTION_FAILED",
        "EXECUTED",
        "INTERNAL_ERROR",
    }
    assert EXIT_CODES == {
        RunStatus.PREFLIGHT_ONLY: 0,
        RunStatus.EXECUTED: 0,
        RunStatus.INTERNAL_ERROR: 1,
        RunStatus.INVALID_INPUT: 2,
        RunStatus.BINDING_REJECTED: 3,
        RunStatus.UNRESOLVED_TCP: 4,
        RunStatus.TCP_CONTRACT_INVALID: 5,
        RunStatus.INPUT_MUTATED: 6,
        RunStatus.EXECUTION_NOT_IMPLEMENTED: 7,
        RunStatus.RUNTIME_START_FAILED: 8,
        RunStatus.OPERATION_REJECTED: 9,
        RunStatus.EXECUTION_FAILED: 10,
    }


def test_project_script_registered_exactly():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "[project.scripts]" in text
    section = text.split("[project.scripts]", 1)[1].split("[", 1)[0]
    assert 'piper-run = "rocobrick.piper_run:main"' in section
    assert 'rocobrick-validate = "rocobrick.cli:main"' not in section


# ---------------------------------------------------------------------------
# example1 end to end: UNRESOLVED_TCP with full evidence
# ---------------------------------------------------------------------------


def test_direct_preflight_passes_tcp_gate_and_never_dispatches(tmp_path):
    """The direct preflight evaluates every gate and never dispatches."""
    run_dir = tmp_path / "run"
    code, line = run_example1(run_dir)
    assert code == 0
    assert line["schema"] == RESULT_SCHEMA
    assert line["status"] == "PREFLIGHT_ONLY"
    assert line["passed"] is False
    assert line["run_dir"] == str(run_dir)
    result_path = Path(line["result_path"])
    assert result_path.is_file()
    assert result_path.name == "result.json"

    evidence_dir, preflight_path, events = read_evidence(line)
    assert evidence_dir.is_relative_to(run_dir)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["schema"] == PREFLIGHT_SCHEMA
    assert preflight["status"] == "PREFLIGHT_ONLY"
    assert preflight["passed"] is False
    assert preflight["exit_code"] == 0
    assert preflight["backend"] == "isaac"
    assert preflight["robot_id"] == ROBOT
    assert preflight["preflight_only"] is True
    assert preflight["robot_operation_plan"]["validated"] is True
    assert preflight["robot_operation_plan"]["plan_id"] == (
        "robot-operation-plan-example1-v2"
    )
    assert preflight["robot_operation_plan"]["robots"] == list(PLAN_ROBOTS)
    assert preflight["robot_operation_plan"]["source_plan_bound"] is False
    assert preflight["robot_operation_plan"]["stap_authorization"] is None
    # The direct path carries no metric pose: the plan is bound to the
    # structure carrier by physical instance id only.
    assert preflight["motion_geometry"] is None
    assert preflight["binding"]["accepted"] is True
    assert preflight["binding"]["kind"] == "structure-scene"
    assert preflight["binding"]["authoritative_stap"] is None
    assert preflight["inputs"]["records"]["robot_operation_plan"]["sha256"] == (
        input_hashes()["robot_operation_plan"]
    )
    assert preflight["inputs"]["records"]["structure_goal"]["sha256"] == (
        input_hashes()["structure_goal"]
    )
    assert preflight["inputs"]["records"]["structure_start"]["sha256"] == (
        input_hashes()["structure_start"]
    )
    assert preflight["inputs"]["immutable_after_all_phases"] is True
    assert preflight["tcp"]["serialized_status"] == "RESOLVED"
    assert preflight["tcp"]["derived_status"] == "RESOLVED"
    assert preflight["tcp"]["status_matches_serialized"] is True
    assert preflight["tcp"]["binding_requires_tcp"] is True
    assert preflight["tcp"]["gate"] == "PASSED"
    assert preflight["tcp"]["require_resolved_tcp"] == "satisfied"
    assert (
        preflight["tcp"]["derived_transform"]["child_frame"]
        == "rocobrick_tcp"
    )
    assert preflight["tcp"]["derived_blockers"] == []
    assert preflight["dispatch"]["implemented"] is False
    assert preflight["dispatch"]["dispatch_requested"] is False
    assert preflight["dispatch"]["blocked"] is True
    assert preflight["dispatch"]["runtime_created"] is False
    assert preflight["dispatch"]["stap_authorization"] is None
    assert preflight["dispatch"]["status"] == "PREFLIGHT_ONLY"
    assert preflight["dispatch"]["reason"] == preflight["reason"]
    assert preflight["checks"]["no_motion_dispatched"] is True
    assert preflight["checks"]["tcp_resolved"] is True
    assert preflight["checks"]["runtime_created"] is False
    assert preflight["checks"]["inputs_immutable_after_all_phases"] is True

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema"] == RESULT_SCHEMA
    assert result["status"] == "PREFLIGHT_ONLY"
    assert result["passed"] is False
    assert result["exit_code"] == 0
    assert result["input_valid"] is True
    assert result["mode"] == "structure-only-preflight"
    assert result["motion_sent"] is False
    assert result["evidence_complete"] is True

    phases = [event["event"]["phase"] for event in events]
    assert phases == [
        "immutable_input",
        "input_aliasing",
        "robot_operation_plan",
        "structure_scene",
        "immutability",
        "tcp_gate",
        "dispatch_blocked",
        "result",
    ]
    by_phase = {event["event"]["phase"]: event["event"] for event in events}
    assert by_phase["robot_operation_plan"]["status"] == "VALIDATED"
    assert by_phase["structure_scene"]["status"] == "VALIDATED"
    assert by_phase["tcp_gate"]["status"] == "PASSED"
    assert by_phase["tcp_gate"]["tcp"]["derived_status"] == "RESOLVED"
    assert by_phase["tcp_gate"]["tcp"]["binding_requires_tcp"] is True
    assert by_phase["dispatch_blocked"]["status"] == "PREFLIGHT_ONLY"


def test_example1_preflight_only_reports_preflight_only(tmp_path):
    code, line = run_example1(tmp_path / "run")
    assert code == 0
    assert line["status"] == "PREFLIGHT_ONLY"
    assert line["passed"] is False
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["preflight_only"] is True
    assert preflight["dispatch"]["dispatch_requested"] is False
    assert preflight["checks"]["runtime_created"] is False
    phases = [event["event"]["phase"] for event in events]
    assert phases == [
        "immutable_input",
        "input_aliasing",
        "robot_operation_plan",
        "structure_scene",
        "immutability",
        "tcp_gate",
        "dispatch_blocked",
        "result",
    ]


def test_example1_run_imports_no_runtime_modules(tmp_path):
    before = set(sys.modules)
    code, _line = run_example1(tmp_path / "run")
    assert code == 0
    new_modules = set(sys.modules) - before
    offenders = sorted(
        name for name in new_modules if name.startswith(BANNED_RUNTIME_PREFIXES)
    )
    assert offenders == []


def test_subprocess_cli_direct_preflight_dispatches_no_runtime(tmp_path):
    """The subprocess CLI entry runs the direct preflight and starts no runtime."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT / "src"),
        str(ROOT),
        env.get("PYTHONPATH", ""),
    ])
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from rocobrick.piper_run import main; "
            "sys.exit(main(sys.argv[1:]))",

            "--robot-operation-plan", str(OPERATION_PLAN),
            "--structure-goal", str(TASK_GOAL),
            "--structure-start", str(TASK_START),
            "--backend", "isaac",
            "--run-dir", str(tmp_path / "run"),
            "--preflight-only",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [raw for raw in proc.stdout.splitlines() if raw.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert set(payload) == {
        "schema",
        "status",
        "passed",
        "run_dir",
        "result_path",
    }
    assert payload["schema"] == RESULT_SCHEMA
    assert payload["status"] == "PREFLIGHT_ONLY"
    assert payload["passed"] is False
    assert Path(payload["result_path"]).is_file()


def test_python_dash_m_entry_runs_direct_preflight(tmp_path):
    """The ``python -m`` entry runs the direct preflight and starts no runtime."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT / "src"),
        str(ROOT),
        env.get("PYTHONPATH", ""),
    ])
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rocobrick.piper_run",

            "--robot-operation-plan", str(OPERATION_PLAN),
            "--structure-goal", str(TASK_GOAL),
            "--structure-start", str(TASK_START),
            "--backend", "isaac",
            "--run-dir", str(tmp_path / "run"),
            "--preflight-only",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [raw for raw in proc.stdout.splitlines() if raw.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert set(payload) == {
        "schema",
        "status",
        "passed",
        "run_dir",
        "result_path",
    }
    assert payload["schema"] == RESULT_SCHEMA
    assert payload["status"] == "PREFLIGHT_ONLY"
    assert payload["passed"] is False
    assert Path(payload["result_path"]).is_file()


# ---------------------------------------------------------------------------
# Immutability, run directories, and evidence hygiene
# ---------------------------------------------------------------------------


def test_inputs_unchanged_and_run_dir_holds_copies_only(tmp_path):
    before = input_hashes()
    run_dir = tmp_path / "run"
    code, line = run_example1(run_dir)
    assert code == 0
    assert input_hashes() == before
    evidence_dir = Path(line["result_path"]).parent
    assert (evidence_dir / "robot-operation-plan.json").read_bytes() == (
        OPERATION_PLAN.read_bytes()
    )
    assert (evidence_dir / "structure_start.json").read_bytes() == (
        TASK_START.read_bytes()
    )
    assert (evidence_dir / "structure_goal.json").read_bytes() == (
        TASK_GOAL.read_bytes()
    )
    assert {path.name for path in evidence_dir.iterdir()} == (
        EXPECTED_EVIDENCE_FILES
    )
    assert set(tmp_path.iterdir()) == {run_dir}


def test_repeated_invocations_create_distinct_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    argv = [
        "--robot-operation-plan", str(OPERATION_PLAN),
        "--structure-goal", str(TASK_GOAL),
        "--structure-start", str(TASK_START),
        "--backend", "isaac",
        "--preflight-only",
    ]
    first_code, first_line, _, _ = invoke(argv)
    second_code, second_line, _, _ = invoke(argv)
    assert first_code == second_code == 0
    first_dir = Path(first_line["run_dir"])
    second_dir = Path(second_line["run_dir"])
    assert first_dir != second_dir
    assert first_dir.is_dir() and second_dir.is_dir()
    assert first_dir.name != second_dir.name
    assert first_dir.resolve().parent == (tmp_path / "runs").resolve()
    pattern = re.compile(r"^piper-run-\d{8}T\d{6}Z-[0-9a-f]{8}$")
    assert pattern.match(first_dir.name)
    assert pattern.match(second_dir.name)


def test_existing_run_dir_rejected_without_overwrite(tmp_path):
    existing = tmp_path / "exists"
    existing.mkdir()
    marker = existing / "keepme.txt"
    marker.write_text("do not touch", encoding="utf-8")
    code, line, _, _ = invoke([
        "--robot-operation-plan", str(OPERATION_PLAN),
        "--structure-goal", str(TASK_GOAL),
        "--structure-start", str(TASK_START),
        "--backend", "isaac",
        "--run-dir", str(existing),
        "--preflight-only",
    ])
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    assert line["passed"] is False
    assert line["run_dir"] == str(existing)
    assert line["result_path"] is None
    assert marker.read_text(encoding="utf-8") == "do not touch"
    assert list(existing.iterdir()) == [marker]


# ---------------------------------------------------------------------------
# Binding and input rejections
# ---------------------------------------------------------------------------


def test_malformed_plan_is_invalid_input(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(b"{")
    code, line = run_with_operation_plan(tmp_path, plan_path=plan_path)
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    evidence_dir, _, events = read_evidence(line)
    phases = [event["event"]["phase"] for event in events]
    assert "immutable_input" in phases
    by_phase = {event["event"]["phase"]: event["event"] for event in events}
    assert by_phase["tcp_gate"]["status"] == "NOT_EVALUATED"
    assert "robot_operation_plan_run" not in by_phase
    preflight = json.loads(
        (evidence_dir / "preflight.json").read_text(encoding="utf-8")
    )
    assert "not valid UTF-8 JSON" in preflight["reason"]


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--robot-operation-plan", str(OPERATION_PLAN),
            "--structure-goal", str(TASK_GOAL),
            "--backend", "rosbag",
        ],
        ["--structure-goal", str(TASK_GOAL)],
        ["--robot-operation-plan", str(OPERATION_PLAN)],
    ],
    ids=["bad-backend", "missing-operation-plan", "missing-structure-goal"],
)
def test_invalid_usage_reports_invalid_input(argv):
    code, line, _, _ = invoke(argv)
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    assert line["run_dir"] is None
    assert line["result_path"] is None


def test_scene_options_with_config_parse_into_the_arguments(tmp_path):
    namespace = piper_run.build_parser().parse_args([
        "--robot-operation-plan", str(OPERATION_PLAN),
        "--structure-goal", str(TASK_GOAL),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
        "--scene-manifest", "scene-manifest.json",
        "--scene-snapshot-out", "scene-snapshot.json",
        "--host-instance-id", "host-example1",
        "--config", "user-config.json",
    ])
    assert namespace.robot_operation_plan == OPERATION_PLAN
    assert namespace.structure_goal == TASK_GOAL
    assert namespace.scene_manifest == Path("scene-manifest.json")
    assert namespace.config == Path("user-config.json")
    assert not hasattr(namespace, "task")
    assert not hasattr(namespace, "plan")
    assert not hasattr(namespace, "structure_dir")


# ---------------------------------------------------------------------------
# TCP gate: injected artifacts
# ---------------------------------------------------------------------------


def test_fake_resolved_tcp_preflight_only_passes(tmp_path, monkeypatch):
    patch_artifact(monkeypatch, fake_artifact(), tmp_path)
    code, line, _, _ = invoke(
        plan_argv(tmp_path, OPERATION_PLAN, "--preflight-only")
    )
    assert code == 0
    assert line["status"] == "PREFLIGHT_ONLY"
    _, preflight_path, _events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["status"] == "PREFLIGHT_ONLY"
    assert preflight["tcp"]["derived_status"] == "RESOLVED"
    assert preflight["tcp"]["serialized_status"] == "RESOLVED"
    assert preflight["tcp"]["gate"] == "PASSED"
    assert preflight["tcp"]["binding_requires_tcp"] is True
    assert preflight["tcp"]["derived_transform"]["child_frame"] == "tool0"
    assert preflight["dispatch"]["dispatch_requested"] is False
    result = json.loads(Path(line["result_path"]).read_text(encoding="utf-8"))
    assert result["status"] == "PREFLIGHT_ONLY"
    assert result["passed"] is False


def test_artifact_status_mismatch_is_tcp_contract_invalid(tmp_path, monkeypatch):
    artifact = json.loads(piper_run.TCP_ARTIFACT.read_text(encoding="utf-8"))
    artifact["tcp_contract"]["status"] = "UNRESOLVED"
    artifact["tcp_execution_allowed"] = False
    artifact["tcp_contract"]["transform"] = None
    patch_artifact(monkeypatch, artifact, tmp_path)
    code, line = run_example1(tmp_path / "run")
    assert code == 5
    assert line["status"] == "TCP_CONTRACT_INVALID"
    assert line["passed"] is False
    _, preflight_path, _events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["tcp"]["serialized_status"] == "UNRESOLVED"
    assert preflight["tcp"]["derived_status"] == "RESOLVED"


def test_fake_unresolved_tcp_stops_at_unresolved_gate(tmp_path, monkeypatch):
    """A self-consistent artifact whose TCP stays UNRESOLVED must stop the run
    at the TCP gate with UNRESOLVED_TCP (exit 4) and dispatch no motion."""
    item = {
        "id": "undeclared-frame",
        "kind": "pinned_explicit_transform",
        "source": FAKE_PINNED_SOURCE,
        "source_sha256": FAKE_PINNED_HASH,
        "declared_tool_frame": False,
        "parent_frame": "link6",
        "child_frame": "tool0",
        "translation_m": [0.0, 0.0, 0.1],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    artifact = {
        "schema": TCP_ARTIFACT_SCHEMA,
        "model": "agilex_piper_l",
        "passed": True,
        "inputs": {
            "pinned_files": [
                {"path": FAKE_PINNED_SOURCE, "sha256": FAKE_PINNED_HASH}
            ]
        },
        "evidence": [item],
        "tcp_contract": {"status": "UNRESOLVED"},
        "tcp_execution_allowed": False,
        "checks": {},
    }
    patch_artifact(monkeypatch, artifact, tmp_path)
    calls: list = []
    created = install_fake_multi_robot_session(monkeypatch, calls)
    code, line = run_with_operation_plan(tmp_path)
    assert code == 4
    assert line["status"] == "UNRESOLVED_TCP"
    assert line["passed"] is False
    assert created == []
    assert calls == []
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["tcp"]["serialized_status"] == "UNRESOLVED"
    assert preflight["tcp"]["derived_status"] == "UNRESOLVED"
    assert preflight["tcp"]["status_matches_serialized"] is True
    assert preflight["tcp"]["gate"] == "UNRESOLVED_TCP"
    assert preflight["tcp"]["derived_blockers"]
    assert preflight["dispatch"]["status"] == "UNRESOLVED_TCP"
    assert preflight["checks"]["no_motion_dispatched"] is True
    assert preflight["checks"]["runtime_created"] is False
    by_phase = {event["event"]["phase"]: event["event"] for event in events}
    assert by_phase["tcp_gate"]["status"] == "UNRESOLVED_TCP"


@pytest.mark.parametrize(
    "overrides",
    [
        {"passed": False},
        {"schema": "some-other-schema/v9"},
        {"tcp_execution_allowed": False},
    ],
    ids=["failed-audit", "wrong-schema", "allowed-flag-contradicts"],
)
def test_tampered_artifacts_are_tcp_contract_invalid(
    tmp_path, monkeypatch, overrides
):
    artifact = json.loads(piper_run.TCP_ARTIFACT.read_text(encoding="utf-8"))
    artifact.update(overrides)
    patch_artifact(monkeypatch, artifact, tmp_path)
    code, line = run_example1(tmp_path / "run")
    assert code == 5
    assert line["status"] == "TCP_CONTRACT_INVALID"


def test_missing_artifact_is_tcp_contract_invalid(tmp_path, monkeypatch):
    patch_artifact(monkeypatch, None, tmp_path)
    code, line = run_example1(tmp_path / "run")
    assert code == 5
    assert line["status"] == "TCP_CONTRACT_INVALID"


# ---------------------------------------------------------------------------
# Module hygiene: accepted libraries only, no metric parser, no rewrites
# ---------------------------------------------------------------------------


def piper_run_imports() -> set[str]:
    """Collect every import name in the piper_run module.

    Returns:
        The set of imported module names.
    """
    tree = ast.parse(PIPER_RUN_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


@pytest.mark.parametrize("root", BANNED_IMPORT_ROOTS)
def test_piper_run_imports_no_runtime_or_tools_modules(root):
    for name in piper_run_imports():
        assert name.split(".")[0] != root, f"piper_run imports {name}"


def test_piper_run_imports_only_accepted_libraries():
    imported = piper_run_imports()
    rocobrick = {name for name in imported if name.startswith("rocobrick")}
    assert rocobrick <= ALLOWED_ROCOBRICK
    assert "rocobrick.contracts" not in imported


def test_operation_phase_imports_its_owned_dependencies():
    """The deferred production imports must resolve without compatibility exports."""
    tree = ast.parse(PIPER_RUN_SOURCE.read_text(encoding="utf-8"))
    phase = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_robot_operation_plan_phase"
    )
    sources = {
        alias.name: node.module
        for node in ast.walk(phase)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    expected = {
        "OperationStateLedger": "rocobrick.execution.operation_state_ledger",
        "load_target_contexts": "rocobrick.skills.target_context",
        "SkillOperationExecutor": "rocobrick.execution.robot_operation_executor",
        "PickHandler": "rocobrick.skills.pick",
    }
    for name, module in expected.items():
        assert sources[name] == module
    assert not {
        "HandoverRequest", "HandoverHandler", "HandoverOperationBackend",
        "receiver_precontact_target", "pick_targets", "gripper_target_from_payload",
    } & sources.keys()
    assert not any(
        isinstance(node, ast.ClassDef) and node.name == "_LiveHandoverObservations"
        for node in ast.walk(phase)
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")
    ])
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rocobrick.skills.target_context import load_target_contexts; "
            "from rocobrick.execution.operation_state_ledger import "
            "OperationStateLedger; "
            "from rocobrick.execution.robot_operation_executor import "
            "SkillOperationExecutor; "
            "from rocobrick.skills.coordinated_handover import "
            "CoordinatedHandoverBackend; "
            "from rocobrick.adapters.moveit.handover_program import "
            "NativeCoordinatedHandoverFactory; "
            "from rocobrick.skills.handover_continuation import "
            "StationaryHandoverContinuation",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_piper_run_never_calls_the_metric_parser():
    source = PIPER_RUN_SOURCE.read_text(encoding="utf-8")
    assert "parse_plan(" not in source.replace("parse_stap_plan(", "")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                assert function.id != "parse_plan"
            if isinstance(function, ast.Attribute):
                assert function.attr != "parse_plan"


def test_piper_run_writes_only_through_the_evidence_recorder():
    tree = ast.parse(PIPER_RUN_SOURCE.read_text(encoding="utf-8"))
    writing_attrs = (
        "write_bytes",
        "write_text",
        "write",
        "dump",
        "open",
        "unlink",
        "rmdir",
        "replace",
    )
    writes = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in writing_attrs
    ]
    assert writes == []
    mkdirs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mkdir"
    ]
    assert len(mkdirs) == 2  # requested run dir + unique auto run dir


# ---------------------------------------------------------------------------
# --robot-operation-plan: the single production dispatch entry
# ---------------------------------------------------------------------------


from dataclasses import dataclass  # noqa: E402

import rocobrick.execution.robot_operation_executor as executor_module  # noqa: E402
from rocobrick.multi_robot_scheduler import (  # noqa: E402
    CancelEvidence,
    OperationExecution,
    OperationStatus,
)

PLAN_ROBOTS = ("piper_0", "piper_1")
SCENE_STATE_SCHEMA = "rocobrick-scene-state/v1"
PIPER0_PART = "brick-768de5a17827920c48d4"
SUPPORT_BRICK = "brick-4bbba49bec24496bb21e"
TASK_SOURCE_ID = "assemble-768de5a17827920c48d4"
PART_BRICK_MIN = (0.30, -0.26, 0.0)
PART_BRICK_MAX = (0.332, -0.244, 0.016)
SUPPORT_BRICK_MIN = (0.34, -0.26, 0.0)
SUPPORT_BRICK_MAX = (0.372, -0.244, 0.016)


def example1_operation_plan() -> dict:
    """Load the external example1 ``robot-operation-plan/v1`` document.

    Returns:
        A fresh mutable copy of the frozen example1 operation plan.
    """
    return json.loads(OPERATION_PLAN.read_text(encoding="utf-8"))


def wait_only_plan_document(*, duration_s: float = 0.2) -> dict:
    """Build a strict single-WAIT plan bound to the example1 STAP plan.

    Args:
        duration_s: Requested observed sim-time advance of the WAIT.

    Returns:
        A ``robot-operation-plan/v1`` document whose only operation is one
        piper_1 WAIT over the example1 task part.
    """
    return {
        "schema": "robot-operation-plan/v1",
        "plan_id": "robot-operation-plan-wait-test-v1",
        "source_plan_id": PLAN_ID,
        "robots": ["piper_0", "piper_1"],
        "operations": [
            {
                "operation_id": "wait-test",
                "source_task_id": TASK_SOURCE_ID,
                "skill": "WAIT",
                "executor_robot_id": "piper_1",
                "participants": ["piper_1"],
                "role_bindings": {"executor": "piper_1"},
                "target": None,
                "parameters": {
                    "observation": {
                        "kind": "structural_fact",
                        "fact": "stable",
                        "physical_instance_id": PIPER0_PART,
                    },
                    "duration_s": duration_s,
                },
                "preconditions": [],
                "effects": [],
                "resources": ["robot:piper_1"],
                "evidence_requirements": [
                    "real_scene_observation",
                    "wait_duration_confirmed",
                ],
            }
        ],
        "dependencies": [],
        "execution_layers": [["wait-test"]],
        "planned_schedule": [
            [{"operation_id": "wait-test", "robot_id": "piper_1"}]
        ],
        "evidence_requirements": [
            {
                "operation_id": "wait-test",
                "evidence": "wait_duration_confirmed",
            }
        ],
    }


def write_operation_plan(tmp_path: Path, document: dict) -> Path:
    """Write one external operation plan document into the temp root.

    Args:
        tmp_path: Temporary directory root.
        document: Plan document to serialize.

    Returns:
        The written plan path.
    """
    plan_path = tmp_path / "external-operation-plan.json"
    plan_path.write_text(json.dumps(document), encoding="utf-8")
    return plan_path


def scene_state_document(robot_id: str, sim_time_s: float) -> dict:
    """Build one whole-scene ``rocobrick-scene-state/v1`` observation.

    Both example1 physical instances are reported, so the WAIT of the frozen
    example1 plan (which observes the support brick) stays observable too.

    Args:
        robot_id: Robot whose block the observation carries.
        sim_time_s: Observed robot and shared simulation time.

    Returns:
        The scene-state document returned by the fake session transport.
    """
    return {
        "schema": SCENE_STATE_SCHEMA,
        "sim_time_s": sim_time_s,
        "robots": {
            robot_id: {
                "robot_id": robot_id,
                "sim_time_s": sim_time_s,
                "gripper": {
                    "gripper_joint1": 0.0,
                    "gripper_joint2": 0.0,
                },
                "opening_m": 0.02,
            }
        },
        "bricks": {
            PIPER0_PART: {
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


def static_scene_reader(robot_id: str) -> dict:
    """Return one non-advancing whole-scene observation (never confirms)."""
    return scene_state_document(robot_id, 0.0)


def advancing_scene_reader():
    """Return a reader whose observed sim time advances on each call."""
    state = {"calls": 0}

    def reader(robot_id: str) -> dict:
        index = state["calls"]
        state["calls"] += 1
        return scene_state_document(
            robot_id, 0.0 if index == 0 else 0.5
        )

    return reader


@dataclass(frozen=True)
class FakeReadinessEvidence:
    """Fake ``MultiRobotReadinessEvidence`` with the frozen field names."""

    robots: tuple
    readiness_by_robot: dict
    all_ready: bool
    passed: bool
    request_path: str = "runtime-session/readiness-request.json"
    result_path: str = "runtime-session/readiness-result.json"


@dataclass(frozen=True)
class FakeSessionEvidence:
    """Fake ``MultiRobotSessionEvidence`` with the frozen field names."""

    status: str
    robots: tuple
    cleanup: dict
    errors: tuple = ()


def ready_readiness_evidence(robots: tuple = PLAN_ROBOTS) -> FakeReadinessEvidence:
    """Return readiness evidence proving every given robot ready.

    Args:
        robots: Robot ids to report, one ready entry each.

    Returns:
        The ready evidence object.
    """
    return FakeReadinessEvidence(
        robots=robots,
        readiness_by_robot={
            robot: {"robot_id": robot, "passed": True}
            for robot in robots
        },
        all_ready=True,
        passed=True,
    )


def session_cleanup_document(passed: bool) -> dict:
    """Build an A6-shaped session cleanup evidence document.

    Args:
        passed: Whether every termination flag is proven passing.

    Returns:
        The cleanup mapping of one stopped session envelope.
    """
    return {
        "scope": {"container": "rocobrick-runtime-session-fake"},
        "passed": passed,
        "container_removed": passed,
        "host_stopped": passed,
        "host_evidence_orderly": passed,
        "leftovers": {"session_containers": [], "session_processes": []},
    }


class FakeMultiRobotSession:
    """Offline stand-in for the real A6 multi-robot session.

    It records every lifecycle/observation call, can fail or mutate inputs at
    chosen points, and never starts a process.
    """

    def __init__(
        self,
        session_dir,
        *,
        calls,
        readiness=None,
        start_error=None,
        stop_error=None,
        cleanup_passed=True,
        cleanup_document=None,
        scene_reader=None,
        observe_error=None,
        mutate_path=None,
    ):
        """Store the fake session's scripted behavior and call log."""
        self.session_dir = Path(session_dir)
        self._calls = calls
        self._readiness = readiness or ready_readiness_evidence()
        self._start_error = start_error
        self._stop_error = stop_error
        self._cleanup_passed = cleanup_passed
        self._cleanup_document = cleanup_document
        self._scene_reader = scene_reader or static_scene_reader
        self._observe_error = observe_error
        self._mutate_path = mutate_path

    def start(self):
        """Record the call, optionally mutate inputs, then answer readiness."""
        self._calls.append(("start",))
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self._mutate_path is not None:
            self._mutate_path.write_text(
                json.dumps({"mutated": True}), encoding="utf-8"
            )
        if self._start_error is not None:
            raise self._start_error
        return self._readiness

    async def observe(self, robot_id):
        """Record and answer one read-only whole-scene observation."""
        self._calls.append(("observe", robot_id))
        if self._observe_error is not None:
            raise self._observe_error
        return self._scene_reader(robot_id)

    def stop(self):
        """Record the stop and answer the scripted session evidence."""
        self._calls.append(("stop",))
        if self._stop_error is not None:
            raise self._stop_error
        return FakeSessionEvidence(
            status="STOPPED",
            robots=tuple(self._readiness.robots),
            cleanup=(
                self._cleanup_document
                if self._cleanup_document is not None
                else session_cleanup_document(self._cleanup_passed)
            ),
        )


def install_fake_multi_robot_session(monkeypatch, calls, **kwargs) -> list:
    """Patch ``piper_run._create_session`` with the fake session factory.

    The factory keeps the production signature (session dir, arguments,
    validated plan robots) and records every created session plus the exact
    scene group it was handed.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        calls: Mutable list recording lifecycle calls.
        **kwargs: Fake-session constructor overrides.

    Returns:
        The list of created fake sessions.
    """
    created = []

    def factory(session_dir, args, robot_ids):
        session = FakeMultiRobotSession(
            session_dir, calls=calls, **kwargs
        )
        session.robot_ids = tuple(robot_ids)
        session.args = args
        created.append(session)
        return session

    monkeypatch.setattr(piper_run, "_create_session", factory)
    return created


def plan_argv(
    tmp_path: Path,
    plan_path: Path,
    *extra: str,
    goal: Path = TASK_GOAL,
    start: Path | None = TASK_START,
) -> list[str]:
    """Build a direct ``--robot-operation-plan`` invocation.

    Args:
        tmp_path: Temporary directory root for the scene group files.
        plan_path: External operation plan path.
        *extra: Additional command-line flags.
        goal: Explicit structure goal file of the run.
        start: Explicit structure start file, or ``None`` to omit it.

    Returns:
        The argv list for ``piper_run.main``.
    """
    argv = [
        "--robot-operation-plan", str(plan_path),
        "--structure-goal", str(goal),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
        "--scene-manifest", str(tmp_path / "scene-manifest.json"),
        "--scene-snapshot-out", str(tmp_path / "scene-snapshot.json"),
        "--host-instance-id", "host-example1",
        "--config", str(tmp_path / "user-config.json"),
    ]
    if start is not None:
        argv += ["--structure-start", str(start)]
    return argv + list(extra)


def run_with_operation_plan(
    tmp_path: Path,
    *extra: str,
    plan_path: Path = OPERATION_PLAN,
    goal: Path = TASK_GOAL,
    start: Path | None = TASK_START,
) -> tuple[int, dict]:
    """Run the direct CLI against an operation plan with the scene group.

    Args:
        tmp_path: Temporary directory root for the scene group files.
        *extra: Additional command-line flags.
        plan_path: External operation plan path.
        goal: Explicit structure goal file of the run.
        start: Explicit structure start file, or ``None`` to omit it.

    Returns:
        The exit code and the parsed stdout JSON line.
    """
    code, line, _, _ = invoke(
        plan_argv(tmp_path, plan_path, *extra, goal=goal, start=start)
    )
    return code, line


def read_plan_run(line: dict) -> tuple[Path, dict, list[dict]]:
    """Load the run directory, its plan-run document and its events.

    Args:
        line: The parsed stdout JSON line of the run.

    Returns:
        The evidence directory, the decoded ``robot-operation-plan-run.json``
        document, and the decoded phase events in order.
    """
    evidence_dir, _preflight_path, events = read_evidence(line)
    document = json.loads(
        (evidence_dir / "robot-operation-plan-run.json").read_text(
            encoding="utf-8"
        )
    )
    return evidence_dir, document, events


@pytest.mark.parametrize("flag", ["--operation", "--skill-request"])
def test_removed_bridge_flags_are_rejected(tmp_path, flag):
    """The old internal bridge flags are gone from the only production CLI."""
    code, line, _, _stderr = invoke([
        "--robot-operation-plan", str(OPERATION_PLAN),
        "--structure-goal", str(TASK_GOAL),
        "--structure-start", str(TASK_START),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
        flag, str(tmp_path / "request.json"),
    ])
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    assert line["run_dir"] is None
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("case", "expected_code", "expected_status"),
    [
        ("malformed-external-plan", 2, "INVALID_INPUT"),
        ("unbindable-external-plan", 2, "INVALID_INPUT"),
        ("unknown-robot", 2, "INVALID_INPUT"),
        ("missing-tcp-artifact", 5, "TCP_CONTRACT_INVALID"),
    ],
)
def test_plan_gates_fail_before_any_session_is_created(
    tmp_path, monkeypatch, case, expected_code, expected_status
):
    """Every immutable gate rejects before the runtime session factory runs."""
    calls: list = []
    created = install_fake_multi_robot_session(monkeypatch, calls)
    plan_path = OPERATION_PLAN
    if case == "malformed-external-plan":
        plan_path = tmp_path / "malformed-plan.json"
        plan_path.write_bytes(b"{")
    elif case in ("unbindable-external-plan", "unknown-robot"):
        document = example1_operation_plan()
        if case == "unbindable-external-plan":
            document["robots"] = ["piper_0"]
        else:
            document["operations"][0]["executor_robot_id"] = "piper_9"
        plan_path = write_operation_plan(tmp_path, document)
    else:
        patch_artifact(monkeypatch, None, tmp_path)

    code, line = run_with_operation_plan(tmp_path, plan_path=plan_path)

    assert code == expected_code
    assert line["status"] == expected_status
    assert line["passed"] is False
    assert created == []
    assert calls == []
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    phases = [event["event"]["phase"] for event in events]
    assert "runtime_start" not in phases
    assert preflight["runtime"] is None
    assert preflight["checks"]["runtime_created"] is False


def test_tcp_authorization_denial_creates_zero_sessions(tmp_path, monkeypatch):
    def denied_authorize(*_args, **_kwargs):
        return SimpleNamespace(
            allowed=False,
            status="UNRESOLVED_TCP",
            reason="synthetic deny",
            checks={"synthetic_check": False},
        )

    monkeypatch.setattr(piper_run, "authorize_tcp", denied_authorize)
    calls: list = []
    created = install_fake_multi_robot_session(monkeypatch, calls)
    code, line = run_with_operation_plan(tmp_path)
    assert code == 4
    assert line["status"] == "UNRESOLVED_TCP"
    assert line["passed"] is False
    assert created == []
    assert calls == []
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    authorization = preflight["tcp"]["authorization"]
    assert authorization["decision"] == "DENY"
    assert authorization["allowed"] is False
    assert authorization["reason"] == "synthetic deny"
    assert preflight["robot_operation_plan"]["validated"] is True
    # The deny path finalizes the dispatch record without a session, so the
    # run document exists but carries the fail-closed terminal status.
    assert preflight["robot_operation_plan_run"]["status"] == "UNRESOLVED_TCP"
    assert preflight["robot_operation_plan_run"]["passed"] is False
    assert "readiness" not in preflight["robot_operation_plan_run"]
    phases = [event["event"]["phase"] for event in events]
    assert "runtime_start" not in phases
    assert "tcp_authorization" in phases


def test_tcp_authorization_precedes_real_session_creation(
    tmp_path, monkeypatch
):
    order: list[str] = []
    real_authorize = piper_run.authorize_tcp

    def authorize(contract, **kwargs):
        order.append("authorize")
        return real_authorize(contract, **kwargs)

    monkeypatch.setattr(piper_run, "authorize_tcp", authorize)
    created = []

    def factory(session_dir, args, robot_ids):
        order.append("create_session")
        session = FakeMultiRobotSession(session_dir, calls=[])
        session.robot_ids = tuple(robot_ids)
        created.append(session)
        return session

    monkeypatch.setattr(piper_run, "_create_session", factory)
    code, line = run_with_operation_plan(tmp_path)

    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    assert order == ["authorize", "create_session"]
    assert len(created) == 1
    assert created[0].robot_ids == PLAN_ROBOTS
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["tcp"]["authorization"]["decision"] == "ALLOW"
    phases = [event["event"]["phase"] for event in events]
    assert phases.index("tcp_authorization") < phases.index("runtime_start")


def test_operation_plan_bytes_are_captured_verbatim(tmp_path, monkeypatch):
    source = tmp_path / "external-plan.json"
    source.write_bytes(OPERATION_PLAN.read_bytes())
    calls: list = []
    install_fake_multi_robot_session(monkeypatch, calls)
    code, line = run_with_operation_plan(tmp_path, plan_path=source)
    assert code == 10
    evidence_dir, run_document, _events = read_plan_run(line)
    captured = evidence_dir / "robot-operation-plan.json"
    assert captured.read_bytes() == source.read_bytes()
    metadata = json.loads(
        (evidence_dir / "robot-operation-plan.json.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    digest = sha256(source.read_bytes()).hexdigest()
    assert metadata == {"sha256": digest, "bytes": len(source.read_bytes())}
    preflight = json.loads(
        (evidence_dir / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["robot_operation_plan"]["sha256"] == digest
    assert preflight["robot_operation_plan"]["validated"] is True
    assert preflight["robot_operation_plan"]["plan_id"] == (
        "robot-operation-plan-example1-v2"
    )
    assert preflight["robot_operation_plan"]["robots"] == list(PLAN_ROBOTS)
    inputs = preflight["inputs"]
    assert inputs["records"]["robot_operation_plan"]["sha256"] == digest
    assert inputs["immutable_after_all_phases"] is True


def test_operation_plan_mutation_after_dispatch_is_input_mutated(
    tmp_path, monkeypatch
):
    """A plan mutated during dispatch fails closed with INPUT_MUTATED.

    This pins the original gate condition: a post-dispatch mutation is promoted
    to ``INPUT_MUTATED`` only when the dispatch terminal is
    ``EXECUTION_NOT_IMPLEMENTED``. The plan is a legal WAIT-only direct plan, so
    the run stays motion-free; dispatch itself is stubbed to the natural
    "no direct backend" terminal, while the post-phase rehash is the real one.
    """
    source = tmp_path / "external-plan.json"
    source.write_bytes(json.dumps(wait_only_plan_document()).encode("utf-8"))
    complete_start = tmp_path / "complete-structure-start.json"
    complete_start.write_bytes(TASK_GOAL.read_bytes())
    calls: list = []
    install_fake_multi_robot_session(monkeypatch, calls)

    def dispatch_then_mutate(args, record, recorder, contract):
        source.write_bytes(source.read_bytes() + b"\n")
        return (
            RunStatus.EXECUTION_NOT_IMPLEMENTED,
            "the direct operation plan contains a skill without a real "
            "direct-execution backend",
        )

    monkeypatch.setattr(
        piper_run, "_robot_operation_plan_phase", dispatch_then_mutate
    )
    code, line = run_with_operation_plan(
        tmp_path, plan_path=source, start=complete_start
    )
    assert code == 6
    assert line["status"] == "INPUT_MUTATED"
    assert line["passed"] is False
    assert calls == []
    evidence_dir, run_document, events = read_plan_run(line)
    assert run_document["status"] == "INPUT_MUTATED"
    assert run_document["passed"] is False
    preflight = json.loads(
        (evidence_dir / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["checks"]["inputs_immutable_after_validation"] is True
    assert preflight["checks"]["inputs_immutable_after_all_phases"] is False
    dispatch = [
        event for event in events if event["event"]["phase"] == "dispatch"
    ]
    assert dispatch[0]["event"]["status"] == "INPUT_MUTATED"


def test_missing_external_plan_is_invalid_input_with_zero_sessions(
    tmp_path, monkeypatch
):
    calls: list = []
    created = install_fake_multi_robot_session(monkeypatch, calls)
    missing = tmp_path / "missing-plan.json"
    code, line = run_with_operation_plan(tmp_path, plan_path=missing)
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    assert line["passed"] is False
    assert created == []
    assert calls == []
    evidence_dir, preflight_path, events = read_evidence(line)
    assert not (evidence_dir / "robot-operation-plan.json").exists()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert "unreadable" in preflight["robot_operation_plan"]["error"]
    assert preflight["robot_operation_plan"]["validated"] is False
    assert (
        preflight["robot_operation_plan_run"]["status"] == "INVALID_INPUT"
    )
    phases = [event["event"]["phase"] for event in events]
    assert "runtime_start" not in phases
    assert preflight["checks"]["no_motion_dispatched"] is True


def test_readiness_failure_schedules_zero_operations(tmp_path, monkeypatch):
    readiness = FakeReadinessEvidence(
        robots=PLAN_ROBOTS,
        readiness_by_robot={
            "piper_0": {"robot_id": "piper_0", "passed": True},
            "piper_1": {"robot_id": "piper_1", "passed": False},
        },
        all_ready=False,
        passed=False,
    )
    calls: list = []
    install_fake_multi_robot_session(
        monkeypatch, calls, readiness=readiness
    )
    code, line = run_with_operation_plan(tmp_path)
    assert code == 8
    assert line["status"] == "RUNTIME_START_FAILED"
    assert line["passed"] is False
    assert calls == [("start",), ("stop",)]
    evidence_dir, run_document, events = read_plan_run(line)
    assert run_document["scheduler_ran"] is False
    assert run_document["operations"] == {}
    assert run_document["passed"] is False
    assert run_document["cleanup_passed"] is True
    assert run_document["readiness"]["all_ready"] is False
    assert "piper_1" in run_document["readiness"]["readiness_by_robot"]
    preflight = json.loads(
        (evidence_dir / "preflight.json").read_text(encoding="utf-8")
    )
    # Readiness failed before the scheduler, so no scheduler check is
    # recorded at all; the runtime itself was created and then cleaned up.
    assert "robot_operation_plan_scheduler_ran" not in preflight["checks"]
    assert preflight["checks"]["runtime_created"] is True
    assert preflight["runtime"]["cleanup"]["passed"] is True
    by_phase = {
        event["event"]["phase"]: event["event"].get("status")
        for event in events
    }
    assert by_phase["runtime_start"] == "FAILED"
    assert by_phase["runtime_stop"] == "STOPPED"


def test_readiness_missing_a_plan_robot_schedules_zero_operations(
    tmp_path, monkeypatch
):
    partial = FakeReadinessEvidence(
        robots=("piper_0",),
        readiness_by_robot={
            "piper_0": {"robot_id": "piper_0", "passed": True}
        },
        all_ready=True,
        passed=True,
    )
    calls: list = []
    install_fake_multi_robot_session(monkeypatch, calls, readiness=partial)
    code, line = run_with_operation_plan(tmp_path)
    assert code == 8
    assert line["status"] == "RUNTIME_START_FAILED"
    _evidence_dir, run_document, _events = read_plan_run(line)
    assert run_document["scheduler_ran"] is False
    assert run_document["operations"] == {}
    assert "are not exactly the plan robots" in (
        run_document["readiness_rejection"]
    )


def test_duplicate_readiness_robot_entry_schedules_zero_operations(
    tmp_path, monkeypatch
):
    duplicated = FakeReadinessEvidence(
        robots=("piper_0", "piper_0", "piper_1"),
        readiness_by_robot={
            "piper_0": {"robot_id": "piper_0", "passed": True},
            "piper_1": {"robot_id": "piper_1", "passed": True},
        },
        all_ready=True,
        passed=True,
    )
    calls: list = []
    install_fake_multi_robot_session(
        monkeypatch, calls, readiness=duplicated
    )
    code, line = run_with_operation_plan(tmp_path)
    assert code == 8
    assert line["status"] == "RUNTIME_START_FAILED"
    _evidence_dir, run_document, _events = read_plan_run(line)
    assert run_document["scheduler_ran"] is False
    assert run_document["operations"] == {}
    assert "are not exactly the plan robots" in (
        run_document["readiness_rejection"]
    )


def test_extra_readiness_robot_key_schedules_zero_operations(
    tmp_path, monkeypatch
):
    extra = FakeReadinessEvidence(
        robots=PLAN_ROBOTS,
        readiness_by_robot={
            "piper_0": {"robot_id": "piper_0", "passed": True},
            "piper_1": {"robot_id": "piper_1", "passed": True},
            "piper_2": {"robot_id": "piper_2", "passed": True},
        },
        all_ready=True,
        passed=True,
    )
    calls: list = []
    install_fake_multi_robot_session(monkeypatch, calls, readiness=extra)
    code, line = run_with_operation_plan(tmp_path)
    assert code == 8
    assert line["status"] == "RUNTIME_START_FAILED"
    _evidence_dir, run_document, _events = read_plan_run(line)
    assert run_document["scheduler_ran"] is False
    assert run_document["operations"] == {}
    assert "keys are not exactly the plan robots" in (
        run_document["readiness_rejection"]
    )
    assert "extra=['piper_2']" in run_document["readiness_rejection"]


def test_session_start_error_schedules_zero_operations(tmp_path, monkeypatch):
    calls: list = []
    install_fake_multi_robot_session(
        monkeypatch, calls, start_error=RuntimeError("synthetic start failure")
    )
    code, line = run_with_operation_plan(tmp_path)
    assert code == 8
    assert line["status"] == "RUNTIME_START_FAILED"
    assert calls == [("start",), ("stop",)]
    evidence_dir, run_document, events = read_plan_run(line)
    assert run_document["scheduler_ran"] is False
    assert run_document["operations"] == {}
    assert run_document["cleanup_passed"] is True
    preflight = json.loads(
        (evidence_dir / "preflight.json").read_text(encoding="utf-8")
    )
    assert "synthetic start failure" in preflight["reason"]
    by_phase = {
        event["event"]["phase"]: event["event"].get("status")
        for event in events
    }
    assert by_phase["runtime_start"] == "FAILED"
    assert by_phase["runtime_stop"] == "STOPPED"


def test_session_creation_error_is_runtime_start_failed(tmp_path, monkeypatch):
    def factory(_session_dir, _args, _robot_ids):
        raise RuntimeError("synthetic creation failure")

    monkeypatch.setattr(piper_run, "_create_session", factory)
    code, line = run_with_operation_plan(tmp_path)
    assert code == 8
    assert line["status"] == "RUNTIME_START_FAILED"
    assert line["passed"] is False
    evidence_dir, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["runtime"] is None
    assert "synthetic creation failure" in preflight["reason"]
    phases = [event["event"]["phase"] for event in events]
    assert "runtime_start" in phases
    assert "runtime_stop" not in phases
    run_document = json.loads(
        (evidence_dir / "robot-operation-plan-run.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_document["scheduler_ran"] is False
    assert run_document["operations"] == {}
    assert run_document["passed"] is False
    assert preflight["checks"]["no_motion_dispatched"] is True


def test_full_example1_plan_fails_capability_unavailable_without_motion(
    tmp_path, monkeypatch
):
    calls: list = []
    created = install_fake_multi_robot_session(monkeypatch, calls)
    code, line, stdout, _stderr = invoke(plan_argv(tmp_path, OPERATION_PLAN))

    assert code == 10
    assert set(line) == {
        "schema",
        "status",
        "passed",
        "run_dir",
        "result_path",
    }
    assert line["schema"] == RESULT_SCHEMA
    assert line["status"] == "EXECUTION_FAILED"
    assert line["status"] != "EXECUTED"
    assert line["passed"] is False
    assert stdout.count("\n") == 1
    assert len(created) == 1
    assert created[0].robot_ids == PLAN_ROBOTS
    assert calls[0] == ("start",)
    assert calls[-1] == ("stop",)
    observed_robots = {
        call[1] for call in calls if call[0] == "observe"
    }
    assert observed_robots <= {"piper_1"}

    evidence_dir, preflight_path, events = read_evidence(line)
    run_document = json.loads(
        (evidence_dir / "robot-operation-plan-run.json").read_text(
            encoding="utf-8"
        )
    )
    operations = run_document["operations"]
    assert operations["example1-piper0-pick"]["status"] == "FAILED"
    assert "CAPABILITY_UNAVAILABLE" in (
        operations["example1-piper0-pick"]["reason"]
    )
    assert operations["example1-piper0-pick"]["motion_sent"] is False
    assert operations["example1-piper1-wait-stable"]["status"] == "CANCELLED"
    for operation_id in ("example1-piper0-place-down",):
        assert operations[operation_id]["status"] == "BLOCKED"
    assert all(
        item["motion_sent"] is not True for item in operations.values()
    )
    assert run_document["passed"] is False
    assert run_document["scheduler_passed"] is False
    assert run_document["unsafe_terminals"] == []
    assert run_document["motion_sent"] is False
    assert run_document["cleanup_passed"] is True
    assert run_document["cleanup"]["cleanup"]["passed"] is True
    assert [
        item["operation_id"] for item in run_document["cancellations"]
    ] == ["example1-piper1-wait-stable"]
    assert run_document["cancellations"][0]["safe"] is True
    assert run_document["cancellations"][0]["terminal"] is True
    assert run_document["resources"]

    # The direct path binds the formal structure carrier only: no
    # assembly-plan/v2 STAP source plan exists, so the assembly-effect document
    # is the documented trivial one and no STAP authorization is fabricated.
    assembly = run_document["stap_assembly_effects"]
    assert assembly["schema"] == "rocobrick-piper-run-stap-assembly-effects/v1"
    assert assembly["mode"] == "robot_operation_plan_primary"
    assert assembly["passed"] is True
    assert assembly["pending_task_ids"] == []
    assert assembly["missing_task_ids"] == []

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["status"] == "EXECUTION_FAILED"
    assert preflight["passed"] is False
    assert preflight["exit_code"] == 10
    assert preflight["dispatch"]["implemented"] is True
    assert preflight["dispatch"]["dispatch_requested"] is True
    assert preflight["dispatch"]["runtime_created"] is True
    assert preflight["dispatch"]["stap_authorization"] is None
    assert preflight["dispatch"]["status"] == "EXECUTION_FAILED"
    assert preflight["dispatch"]["reason"] == preflight["reason"]
    assert "CAPABILITY_UNAVAILABLE" in preflight["reason"]
    checks = preflight["checks"]
    assert checks["robot_operation_plan_validated"] is True
    assert checks["robot_operation_plan_cross_bound"] is True
    assert checks["robot_operation_plan_readiness_passed"] is True
    assert checks["robot_operation_plan_scheduler_ran"] is True
    assert checks["robot_operation_plan_scheduler_passed"] is False
    assert checks["robot_operation_plan_cleanup_passed"] is True
    # The direct path has no assembly-plan/v2 binding, so the trivial
    # assembly-effect document is passed and confirms no pending task.
    assert checks["robot_operation_plan_pending_tasks_confirmed"] is True
    assert checks["runtime_created"] is True
    assert checks["inputs_immutable_after_all_phases"] is True
    assert checks["motion_sent"] is False
    assert checks["no_motion_dispatched"] is True

    result = json.loads(Path(line["result_path"]).read_text(encoding="utf-8"))
    assert result["schema"] == RESULT_SCHEMA
    assert result["passed"] is False
    assert result["execution_safe"] is False
    assert result["robot_operation_plan_run"]["passed"] is False
    assert result["runtime_created"] is True
    assert result["robot_operation_plan_run"]["cleanup_passed"] is True

    phases = list(
        dict.fromkeys(event["event"]["phase"] for event in events)
    )
    assert phases == [
        "immutable_input",
        "input_aliasing",
        "robot_operation_plan",
        "structure_scene",
        "immutability",
        "tcp_gate",
        "tcp_authorization",
        "runtime_start",
        "runtime_stop",
        "stap_assembly_effects",
        "robot_operation_plan_run",
        "dispatch",
        "result",
    ]
    ready_event = [
        event["event"]
        for event in events
        if event["event"]["phase"] == "runtime_start"
        and event["event"]["status"] == "READY"
    ][0]
    assert ready_event["readiness"]["robots"] == list(PLAN_ROBOTS)
    assert ready_event["readiness"]["readiness_by_robot"]["piper_1"][
        "passed"
    ] is True


class ConfirmedPlaceDownExecutor:
    """Executor stand-in that confirms PLACE_DOWN with assembly evidence.

    The stub never sends motion: it answers a synthetic CONFIRMED result for
    every ``PLACE_DOWN`` operation carrying the assembly evidence labels, so
    the piper-run final semantic check can be exercised end to end.
    """

    def __init__(self, robot_ids, observation_transport):
        """Store the validated robots and the injected transport."""
        self.robot_ids = tuple(robot_ids)
        self.transport = observation_transport

    async def execute(self, operation):
        """Confirm one PLACE_DOWN operation with its evidence labels.

        Args:
            operation: PLACE_DOWN operation dispatched by the scheduler.

        Returns:
            A synthetic CONFIRMED execution carrying the assembly labels.
        """
        assert operation.skill == "PLACE_DOWN", operation.skill
        values = operation.target.values
        evidence = {
            "connection_confirmed": {
                "connection_id": values["connection_id"],
                "upper_physical_instance_id": values[
                    "upper_physical_instance_id"
                ],
                "lower_physical_instance_id": values[
                    "lower_physical_instance_id"
                ],
            },
            "release_confirmed": {"opening_m": 0.04},
            "retreat_confirmed": {"retreat_clear": True},
        }
        return OperationExecution(
            operation_id=operation.operation_id,
            robot_id=operation.executor_robot_id,
            status=OperationStatus.CONFIRMED,
            passed=True,
            terminal=True,
            motion_sent=False,
            evidence={
                label: evidence[label]
                for label in operation.evidence_requirements
            },
            reason="synthetic confirmed assembly",
        )

    async def cancel(self, operation_id):
        """Acknowledge a cancel as terminal and safe (no motion exists).

        Args:
            operation_id: Operation the scheduler asked to cancel.

        Returns:
            A safe terminal cancel acknowledgement.
        """
        return CancelEvidence(
            operation_id=operation_id,
            acknowledged=True,
            terminal=True,
            safe=True,
            evidence={"mode": "no_motion"},
        )


def test_cleanup_failure_blocks_an_otherwise_confirmed_run(
    tmp_path, monkeypatch
):
    plan_path = write_operation_plan(tmp_path, wait_only_plan_document())
    calls: list = []
    install_fake_multi_robot_session(
        monkeypatch,
        calls,
        scene_reader=advancing_scene_reader(),
        cleanup_passed=False,
    )
    complete_start = tmp_path / "complete-structure-start.json"
    complete_start.write_bytes(TASK_GOAL.read_bytes())
    code, line = run_with_operation_plan(
        tmp_path, plan_path=plan_path, start=complete_start
    )
    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    assert line["passed"] is False
    assert calls[-1] == ("stop",)
    evidence_dir, preflight_path, _events = read_evidence(line)
    run_document = json.loads(
        (evidence_dir / "robot-operation-plan-run.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_document["scheduler_passed"] is True
    assert run_document["cleanup_passed"] is False
    assert run_document["passed"] is False
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert "cleanup did not pass" in preflight["reason"]
    assert preflight["runtime"]["cleanup"]["passed"] is False


def test_cleanup_without_termination_evidence_blocks_the_run(
    tmp_path, monkeypatch
):
    """A bare cleanup bool is not session-owned termination evidence."""
    plan_path = write_operation_plan(tmp_path, wait_only_plan_document())
    calls: list = []
    install_fake_multi_robot_session(
        monkeypatch,
        calls,
        scene_reader=advancing_scene_reader(),
        cleanup_document={"passed": True},
    )
    complete_start = tmp_path / "complete-structure-start.json"
    complete_start.write_bytes(TASK_GOAL.read_bytes())
    code, line = run_with_operation_plan(
        tmp_path, plan_path=plan_path, start=complete_start
    )
    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    assert line["passed"] is False
    _evidence_dir, run_document, _events = read_plan_run(line)
    assert run_document["scheduler_passed"] is True
    assert run_document["cleanup_passed"] is False
    assert run_document["passed"] is False


def test_cleanup_leftovers_block_the_run(tmp_path, monkeypatch):
    """Session-scoped leftovers invalidate the cleanup verdict."""
    plan_path = write_operation_plan(tmp_path, wait_only_plan_document())
    leftovers = session_cleanup_document(True)
    leftovers["leftovers"] = {
        "session_containers": ["rocobrick-runtime-session-fake"],
        "session_processes": [],
    }
    calls: list = []
    install_fake_multi_robot_session(
        monkeypatch,
        calls,
        scene_reader=advancing_scene_reader(),
        cleanup_document=leftovers,
    )
    complete_start = tmp_path / "complete-structure-start.json"
    complete_start.write_bytes(TASK_GOAL.read_bytes())
    code, line = run_with_operation_plan(
        tmp_path, plan_path=plan_path, start=complete_start
    )
    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    _evidence_dir, run_document, _events = read_plan_run(line)
    assert run_document["scheduler_passed"] is True
    assert run_document["cleanup_passed"] is False


def test_session_stop_error_is_audited_and_fails_the_run(tmp_path, monkeypatch):
    plan_path = write_operation_plan(tmp_path, wait_only_plan_document())
    calls: list = []
    install_fake_multi_robot_session(
        monkeypatch,
        calls,
        scene_reader=advancing_scene_reader(),
        stop_error=RuntimeError("synthetic stop failure"),
    )
    complete_start = tmp_path / "complete-structure-start.json"
    complete_start.write_bytes(TASK_GOAL.read_bytes())
    code, line = run_with_operation_plan(
        tmp_path, plan_path=plan_path, start=complete_start
    )
    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    assert line["passed"] is False
    evidence_dir, preflight_path, _events = read_evidence(line)
    run_document = json.loads(
        (evidence_dir / "robot-operation-plan-run.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_document["scheduler_passed"] is True
    assert run_document["cleanup_passed"] is False
    assert run_document["cleanup"]["cleanup_passed"] is False
    assert "synthetic stop failure" in run_document["cleanup"]["cleanup_error"]
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert "cleanup did not pass" in preflight["reason"]


class RaisingExecutor:
    """Executor stand-in whose every dispatch raises (strict-mode check)."""

    def __init__(self, robot_ids, observation_transport):
        """Store the validated robots and the injected transport."""
        self.robot_ids = tuple(robot_ids)
        self.transport = observation_transport

    async def execute(self, operation):
        """Fail every dispatched operation with an exception."""
        raise RuntimeError("synthetic executor failure")

    async def cancel(self, operation_id):
        """Acknowledge a cancel as terminal and safe (no motion exists)."""
        return CancelEvidence(
            operation_id=operation_id,
            acknowledged=True,
            terminal=True,
            safe=True,
            evidence={"mode": "no_motion"},
        )


def test_executor_error_reports_execution_failed_with_cleanup(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        executor_module, "SkillOperationExecutor", RaisingExecutor
    )
    calls: list = []
    install_fake_multi_robot_session(monkeypatch, calls)
    code, line = run_with_operation_plan(tmp_path)
    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    assert line["passed"] is False
    assert calls[0] == ("start",)
    assert calls[-1] == ("stop",)
    evidence_dir, preflight_path, _events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert "synthetic executor failure" in preflight["reason"]
    assert preflight["runtime"]["cleanup"]["passed"] is True
    assert preflight["checks"]["no_motion_dispatched"] is True
    run_document = json.loads(
        (evidence_dir / "robot-operation-plan-run.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_document["scheduler_ran"] is True
    assert run_document["scheduler_passed"] is False
    assert run_document["passed"] is False
    assert run_document["motion_sent"] is False
    assert run_document["operations"][
        "example1-piper0-pick"
    ]["reason"].startswith("executor raised RuntimeError")


def test_strict_results_reject_a_legacy_executor(tmp_path, monkeypatch):
    """The runner must run with allow_legacy_results=False."""

    class LegacyExecutor:
        def __init__(self, robot_ids, observation_transport):
            """Store the validated robots (transport is unused)."""
            self.robot_ids = tuple(robot_ids)

        async def execute(self, operation):
            """Return a non-contract (legacy) result that strict mode rejects."""
            return {"passed": True, "operation_id": operation.operation_id}

        async def cancel(self, operation_id):
            """Return a non-contract (legacy) cancel acknowledgement."""
            return {"passed": True}

    monkeypatch.setattr(
        executor_module, "SkillOperationExecutor", LegacyExecutor
    )
    calls: list = []
    install_fake_multi_robot_session(monkeypatch, calls)
    code, line = run_with_operation_plan(tmp_path)
    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    evidence_dir, _preflight_path, _events = read_evidence(line)
    run_document = json.loads(
        (evidence_dir / "robot-operation-plan-run.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(
        item.get("legacy") for item in run_document["operations"].values()
    ) is False
    assert run_document["passed"] is False
    pick = run_document["operations"]["example1-piper0-pick"]
    assert "non-contract executor result rejected" in pick["reason"]
    assert pick["motion_sent"] is not True


def test_production_factory_builds_the_real_a6_session(tmp_path):
    """The production injection seam constructs the final A6 session.

    Only construction is exercised: no process, container, DDS profile or
    motion is ever started here.
    """
    from rocobrick.multi_robot_runtime import (  # noqa: PLC0415
        MultiRobotIsaacMoveItSession,
    )

    args = piper_run._Arguments(
        backend="isaac",
        robot_id=ROBOT,
        run_dir=None,
        preflight_only=False,
        robot_operation_plan_path=OPERATION_PLAN,
        scene_manifest=tmp_path / "scene-manifest.json",
        scene_snapshot_out=tmp_path / "scene-snapshot.json",
        host_instance_id="host-example1",
        config_path=tmp_path / "user-config.json",
    )
    session = piper_run._create_session(
        tmp_path / "session", args, PLAN_ROBOTS
    )
    assert isinstance(session, MultiRobotIsaacMoveItSession)
    assert session.config.repo_root == piper_run.REPO_ROOT
    assert session.config.robot_ids == PLAN_ROBOTS
    assert session.config.scene_manifest == args.scene_manifest
    assert session.config.scene_snapshot_out == args.scene_snapshot_out
    assert session.config.host_instance_id == "host-example1"
    assert session.config.plan_path == OPERATION_PLAN
    assert session.config.config_path == args.config_path
    assert session.session_dir == tmp_path / "session"
    assert session.status.value == "NOT_STARTED"


def test_a6_evidence_serialization_satisfies_the_exact_checks():
    """The finalized A6 evidence classes pass the exact A8 checks."""
    from rocobrick.multi_robot_runtime import (  # noqa: PLC0415
        MultiRobotReadinessEvidence,
        MultiRobotSessionEvidence,
        RuntimeSessionStatus,
    )

    ready = MultiRobotReadinessEvidence(
        robots=PLAN_ROBOTS,
        readiness_by_robot={
            "piper_0": {"robot_id": "piper_0", "passed": True},
            "piper_1": {"robot_id": "piper_1", "passed": True},
        },
        all_ready=True,
    )
    document = piper_run._evidence_document(ready, "readiness")
    assert document["passed"] is True
    assert "not_ready_robots" in document
    assert piper_run._readiness_rejection(document, PLAN_ROBOTS) is None

    unready = MultiRobotReadinessEvidence(
        robots=PLAN_ROBOTS,
        readiness_by_robot={
            "piper_0": {"robot_id": "piper_0", "passed": True},
            "piper_1": {"robot_id": "piper_1", "passed": False},
        },
        all_ready=False,
    )
    document = piper_run._evidence_document(unready, "readiness")
    assert document["passed"] is False
    assert piper_run._readiness_rejection(document, PLAN_ROBOTS) is not None

    duplicate = MultiRobotReadinessEvidence(
        robots=("piper_0", "piper_0", "piper_1"),
        readiness_by_robot={
            "piper_0": {"robot_id": "piper_0", "passed": True},
            "piper_1": {"robot_id": "piper_1", "passed": True},
        },
        all_ready=True,
    )
    document = piper_run._evidence_document(duplicate, "readiness")
    assert piper_run._readiness_rejection(document, PLAN_ROBOTS) is not None

    evidence = MultiRobotSessionEvidence(
        status=RuntimeSessionStatus.STOPPED,
        robots=PLAN_ROBOTS,
        readiness=ready,
        host_pid=4242,
        container_name="rocobrick-runtime-session-fake",
        cleanup={
            "scope": {"container": "rocobrick-runtime-session-fake"},
            "passed": True,
            "container_removed": True,
            "host_stopped": True,
            "host_evidence_orderly": True,
            "leftovers": {"session_containers": [], "session_processes": []},
        },
    )
    document = piper_run._evidence_document(evidence, "cleanup")
    assert document["status"] == "STOPPED"
    assert document["cleanup"]["passed"] is True
    assert piper_run._cleanup_passed(document) is True

    assert piper_run._cleanup_passed({"cleanup": {"passed": True}}) is False
    partial_cleanup = {
        "cleanup": {"passed": True, "container_removed": True}
    }
    assert piper_run._cleanup_passed(partial_cleanup) is False
    leftover = piper_run._evidence_document(evidence, "cleanup")
    leftover["cleanup"]["leftovers"]["session_processes"] = [1234]
    assert piper_run._cleanup_passed(leftover) is False


# ---------------------------------------------------------------------------
# Phase-4 guardrails: the retired STAP CLI surface stays retired
# ---------------------------------------------------------------------------


def test_retired_stap_cli_flags_are_gone(tmp_path):
    """--task/--plan/--structure-dir are absent and fail closed if supplied."""
    help_text = piper_run.build_parser().format_help()
    for flag in ("--task", "--plan", "--structure-dir"):
        assert flag not in help_text
    for flag in ("--task", "--plan", "--structure-dir"):
        code, line, _, stderr = invoke([
            "--robot-operation-plan", str(OPERATION_PLAN),
            "--structure-goal", str(TASK_GOAL),
            "--structure-start", str(TASK_START),
            "--backend", "isaac",
            "--run-dir", str(tmp_path / "run"),
            flag,
            str(tmp_path / "value"),
        ])
        assert code == 2
        assert line["status"] == "INVALID_INPUT"
        assert line["run_dir"] is None
        assert line["result_path"] is None
        assert "unrecognized arguments" in stderr
        assert not (tmp_path / "run").exists()


def test_missing_structure_goal_or_plan_is_invalid_input(tmp_path, monkeypatch):
    """A missing plan, goal, start, or plan directory fails closed as invalid."""
    calls: list = []
    created = install_fake_multi_robot_session(monkeypatch, calls)
    plan_dir = tmp_path / "plan-dir"
    plan_dir.mkdir()
    cases = (
        (tmp_path / "missing-plan.json", TASK_GOAL, TASK_START),
        (OPERATION_PLAN, tmp_path / "missing-goal.json", TASK_START),
        (plan_dir, TASK_GOAL, TASK_START),
        (OPERATION_PLAN, TASK_GOAL, tmp_path / "missing-start.json"),
    )
    for index, (plan, goal, start) in enumerate(cases):
        code, line, _, _ = invoke(
            plan_argv(
                tmp_path,
                plan,
                "--run-dir", str(tmp_path / f"run-{index}"),
                goal=goal,
                start=start,
            )
        )
        assert code == 2, line
        assert line["status"] == "INVALID_INPUT", line
        assert line["passed"] is False
    assert created == []
    assert calls == []


def test_direct_preflight_still_requires_the_tcp_gate(tmp_path, monkeypatch):
    """The formal direct chain keeps the TCP gate in force before dispatch."""
    patch_artifact(monkeypatch, None, tmp_path)
    code, line = run_example1(tmp_path / "run")
    assert code == 5
    assert line["status"] == "TCP_CONTRACT_INVALID"
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["tcp"]["binding_requires_tcp"] is True
    assert preflight["dispatch"]["dispatch_requested"] is False
    assert preflight["checks"]["runtime_created"] is False
    phases = [event["event"]["phase"] for event in events]
    assert "tcp_gate" in phases
    assert "runtime_start" not in phases


def test_piper_run_source_has_no_retired_stap_binding_imports():
    """The retired STAP binding helpers are neither imported nor called."""
    source = PIPER_RUN_SOURCE.read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    for name in (
        "rocobrick.structural_binding",
        "rocobrick.stap_contract",
        "bind_single_stap_task",
        "binding_requires_tcp",
        "BindingDecision",
        "parse_stap_plan",
        "STAP_PLAN_SCHEMA",
        "StapContractError",
    ):
        assert name not in imported, name
    for call in ("parse_stap_plan(", "bind_single_stap_task(", "source_plan="):
        assert call not in source, call
    assert '"stap_source_plan_provided": False' in source
