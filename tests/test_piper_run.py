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
PLAN = ROOT / "runs/example1-compound-pick-20260915T130000Z/generated-plan.json"
TASK_DIR = ROOT / "tasks/type1/example1"
TASK_START = TASK_DIR / "structure_start.json"
TASK_GOAL = TASK_DIR / "structure_goal.json"

PLAN_ID = "assembly-plan-25a5d5cc586a1e5c65f2141d"
TASK_ID = "assemble-768de5a17827920c48d4"
ROBOT = "piper_0"

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
    "rocobrick.stap_contract",
    "rocobrick.structural_binding",
    "rocobrick.tcp_capability",
    "rocobrick.tcp_contract",
    # The INTERNAL --operation bridge (R4.2): imported lazily, only after
    # every immutable gate has passed, and only for explicit dispatch.
    "rocobrick.isaac_backend",
    "rocobrick.runtime_session",
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
    "plan.json",
    "plan.json.metadata.json",
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
    """Hash the three repository input files from disk.

    Returns:
        The raw SHA-256 digests of plan, structure start, and structure goal.
    """
    return {
        "plan": sha256(PLAN.read_bytes()).hexdigest(),
        "task_start": sha256(TASK_START.read_bytes()).hexdigest(),
        "task_goal": sha256(TASK_GOAL.read_bytes()).hexdigest(),
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
    """Run the CLI against example1 with an explicit run directory.

    Args:
        run_dir: Fresh evidence directory to create.
        *extra: Additional command-line flags.

    Returns:
        The exit code and the parsed stdout JSON line.
    """
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(PLAN),
        "--backend", "isaac",
        "--run-dir", str(run_dir),
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
    assert 'rocobrick-validate = "rocobrick.cli:main"' in section


# ---------------------------------------------------------------------------
# example1 end to end: UNRESOLVED_TCP with full evidence
# ---------------------------------------------------------------------------


def test_example1_passes_tcp_gate_and_reports_execution_not_implemented(tmp_path):
    run_dir = tmp_path / "run"
    code, line = run_example1(run_dir)
    assert code == 7
    assert line["schema"] == RESULT_SCHEMA
    assert line["status"] == "EXECUTION_NOT_IMPLEMENTED"
    assert line["passed"] is False
    assert line["run_dir"] == str(run_dir)
    result_path = Path(line["result_path"])
    assert result_path.is_file()
    assert result_path.name == "result.json"

    evidence_dir, preflight_path, events = read_evidence(line)
    assert evidence_dir.is_relative_to(run_dir)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["schema"] == PREFLIGHT_SCHEMA
    assert preflight["status"] == "EXECUTION_NOT_IMPLEMENTED"
    assert preflight["passed"] is False
    assert preflight["exit_code"] == 7
    assert preflight["backend"] == "isaac"
    assert preflight["robot_id"] == ROBOT
    assert preflight["preflight_only"] is False
    assert preflight["plan_id"] == PLAN_ID
    assert preflight["task_id"] == TASK_ID
    assert preflight["motion_geometry"] == "omitted"
    assert preflight["inputs"]["sha256"] == input_hashes()
    assert preflight["binding"]["accepted"] is True
    assert preflight["binding"]["binding"]["grid_xyz"] == [0, 0, 1]
    assert preflight["binding"]["binding"]["plan_sha256"] == input_hashes()["plan"]
    assert preflight["tcp"]["serialized_status"] == "RESOLVED"
    assert preflight["tcp"]["derived_status"] == "RESOLVED"
    assert preflight["tcp"]["status_matches_serialized"] is True
    assert preflight["tcp"]["binding_requires_tcp"] is True
    assert preflight["tcp"]["gate"] == "PASSED"
    assert preflight["tcp"]["require_resolved_tcp"] == "satisfied"
    assert preflight["tcp"]["derived_transform"]["child_frame"] == "rocobrick_tcp"
    assert preflight["tcp"]["derived_blockers"] == []
    assert preflight["dispatch"] == {
        "implemented": False,
        "dispatch_requested": True,
        "blocked": True,
        "status": "EXECUTION_NOT_IMPLEMENTED",
        "reason": preflight["reason"],
    }
    assert preflight["checks"]["no_motion_dispatched"] is True
    assert preflight["checks"]["tcp_resolved"] is True
    assert preflight["checks"]["inputs_immutable_after_all_phases"] is True

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema"] == RESULT_SCHEMA
    assert result["status"] == "EXECUTION_NOT_IMPLEMENTED"
    assert result["passed"] is False
    assert result["exit_code"] == 7
    assert result["input_valid"] is True
    assert result["tasks_complete"] is True
    assert result["structure_valid"] is True
    assert result["execution_safe"] is False
    assert result["evidence_complete"] is True
    assert "no dispatch path is implemented" in result["reason"].lower()

    phases = [event["event"]["phase"] for event in events]
    assert phases == [
        "immutable_input",
        "stap_validated",
        "structural_binding",
        "tcp_gate",
        "dispatch_blocked",
        "result",
    ]
    by_phase = {event["event"]["phase"]: event["event"] for event in events}
    assert by_phase["immutable_input"]["sha256"] == input_hashes()
    assert by_phase["stap_validated"]["plan_id"] == PLAN_ID
    assert by_phase["structural_binding"]["status"] == "ACCEPTED"
    assert by_phase["structural_binding"]["robot_id"] == ROBOT
    assert by_phase["structural_binding"]["motion_geometry"] == "omitted"
    assert by_phase["tcp_gate"]["status"] == "PASSED"
    assert by_phase["tcp_gate"]["tcp"]["derived_status"] == "RESOLVED"
    assert by_phase["tcp_gate"]["tcp"]["binding_requires_tcp"] is True
    assert by_phase["dispatch_blocked"]["status"] == "EXECUTION_NOT_IMPLEMENTED"


def test_example1_preflight_only_reports_preflight_only(tmp_path):
    code, line = run_example1(tmp_path / "run", "--preflight-only")
    assert code == 0
    assert line["status"] == "PREFLIGHT_ONLY"
    assert line["passed"] is True


def test_example1_run_imports_no_runtime_modules(tmp_path):
    before = set(sys.modules)
    code, _line = run_example1(tmp_path / "run")
    assert code == 7
    new_modules = set(sys.modules) - before
    offenders = sorted(
        name for name in new_modules if name.startswith(BANNED_RUNTIME_PREFIXES)
    )
    assert offenders == []


def test_subprocess_cli_example1_reports_execution_not_implemented(tmp_path):
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
            "--task", str(TASK_DIR),
            "--plan", str(PLAN),
            "--backend", "isaac",
            "--run-dir", str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 7, proc.stderr
    lines = [raw for raw in proc.stdout.splitlines() if raw.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert set(payload) == {"schema", "status", "passed", "run_dir", "result_path"}
    assert payload["schema"] == RESULT_SCHEMA
    assert payload["status"] == "EXECUTION_NOT_IMPLEMENTED"
    assert payload["passed"] is False
    assert Path(payload["result_path"]).is_file()


def test_python_dash_m_module_entry_reports_execution_not_implemented(tmp_path):
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
            "--task", str(TASK_DIR),
            "--plan", str(PLAN),
            "--backend", "isaac",
            "--run-dir", str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 7, proc.stderr
    lines = [raw for raw in proc.stdout.splitlines() if raw.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert set(payload) == {"schema", "status", "passed", "run_dir", "result_path"}
    assert payload["schema"] == RESULT_SCHEMA
    assert payload["status"] == "EXECUTION_NOT_IMPLEMENTED"
    assert payload["passed"] is False
    assert Path(payload["result_path"]).is_file()
    _, preflight_path, events = read_evidence(payload)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["status"] == "EXECUTION_NOT_IMPLEMENTED"
    assert preflight["exit_code"] == 7
    phases = [event["event"]["phase"] for event in events]
    assert phases == [
        "immutable_input",
        "stap_validated",
        "structural_binding",
        "tcp_gate",
        "dispatch_blocked",
        "result",
    ]


# ---------------------------------------------------------------------------
# Immutability, run directories, and evidence hygiene
# ---------------------------------------------------------------------------


def test_inputs_unchanged_and_run_dir_holds_copies_only(tmp_path):
    before = input_hashes()
    run_dir = tmp_path / "run"
    code, line = run_example1(run_dir)
    assert code == 7
    assert input_hashes() == before
    evidence_dir = Path(line["result_path"]).parent
    assert (evidence_dir / "plan.json").read_bytes() == PLAN.read_bytes()
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
    argv = ["--task", str(TASK_DIR), "--plan", str(PLAN), "--backend", "isaac"]
    first_code, first_line, _, _ = invoke(argv)
    second_code, second_line, _, _ = invoke(argv)
    assert first_code == second_code == 7
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
        "--task", str(TASK_DIR),
        "--plan", str(PLAN),
        "--backend", "isaac",
        "--run-dir", str(existing),
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


def test_wrong_robot_is_binding_rejected(tmp_path):
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(PLAN),
        "--backend", "isaac",
        "--robot-id", "piper_1",
        "--run-dir", str(tmp_path / "run"),
    ])
    assert code == 3
    assert line["status"] == "BINDING_REJECTED"
    evidence_dir, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["binding"]["accepted"] is False
    assert "piper_1" in preflight["binding"]["reason"]
    assert preflight["passed"] is False
    by_phase = {event["event"]["phase"]: event["event"] for event in events}
    assert by_phase["structural_binding"]["status"] == "REJECTED"
    assert "piper_1" in by_phase["structural_binding"]["reason"]


def test_wrong_robot_run_leaves_inputs_untouched(tmp_path):
    before = input_hashes()
    code, _line = run_example1(tmp_path / "run", "--robot-id", "piper_1")
    assert code == 3
    assert input_hashes() == before


def test_changed_task_bytes_are_binding_rejected(tmp_path):
    task_dir = copy_task_dir(tmp_path / "task")
    goal = json.loads((task_dir / "structure_goal.json").read_text(encoding="utf-8"))
    goal["2"]["color"] = "Green"
    (task_dir / "structure_goal.json").write_text(
        json.dumps(goal), encoding="utf-8"
    )
    code, line, _, _ = invoke([
        "--task", str(task_dir),
        "--plan", str(PLAN),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
    ])
    assert code == 3
    assert line["status"] == "BINDING_REJECTED"


def test_changed_plan_bytes_fail_stap_validation(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["plan_id"] = "tampered-plan"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(plan_path),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
    ])
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    _, _, events = read_evidence(line)
    phases = [event["event"]["phase"] for event in events]
    assert "structural_binding" not in phases


def test_malformed_plan_is_invalid_input(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(b"{")
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(plan_path),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
    ])
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    evidence_dir, _, events = read_evidence(line)
    phases = [event["event"]["phase"] for event in events]
    assert "immutable_input" in phases
    assert "stap_validated" not in phases
    preflight = json.loads(
        (evidence_dir / "preflight.json").read_text(encoding="utf-8")
    )
    assert "not valid UTF-8 JSON" in preflight["reason"]


def test_metric_shaped_plan_rejected_by_stap_parser(tmp_path):
    metric_plan = {
        "schema": "assembly-plan/v2",
        "plan_id": "metric-plan",
        "frame_id": "assembly_world",
        "units": {"length": "m", "angle": "rad"},
        "robots": ["piper_0"],
        "tasks": [
            {
                "task_id": "place:0",
                "part_instance_id": "brick-0",
                "brick_id": 2,
                "target": {
                    "frame_id": "assembly_world",
                    "position_m": [0.0, 0.0, 0.01],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "skill_type": "PLACE_DOWN",
                "primary_robot": "piper_0",
                "supporting_robot": None,
                "predecessors": [],
            }
        ],
    }
    plan_path = tmp_path / "metric-plan.json"
    plan_path.write_text(json.dumps(metric_plan), encoding="utf-8")
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(plan_path),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
    ])
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    evidence_dir, _, events = read_evidence(line)
    phases = [event["event"]["phase"] for event in events]
    assert "stap_validated" not in phases
    assert "structural_binding" not in phases
    preflight = json.loads(
        (evidence_dir / "preflight.json").read_text(encoding="utf-8")
    )
    assert "STAP validation" in preflight["reason"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--task", str(TASK_DIR), "--plan", str(PLAN), "--backend", "rosbag"],
        ["--task", str(TASK_DIR), "--plan", str(PLAN)],
    ],
    ids=["bad-backend", "missing-backend"],
)
def test_invalid_usage_reports_invalid_input(argv):
    code, line, _, _ = invoke(argv)
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    assert line["run_dir"] is None
    assert line["result_path"] is None


def test_missing_structure_files_or_plan_are_invalid_input(tmp_path):
    incomplete = copy_task_dir(tmp_path / "incomplete")
    (incomplete / "structure_start.json").unlink()
    plan_dir = tmp_path / "plan-dir"
    plan_dir.mkdir()
    for argv in (
        ["--task", str(incomplete), "--plan", str(PLAN)],
        ["--task", str(TASK_DIR), "--plan", str(plan_dir)],
        ["--task", str(tmp_path / "nope"), "--plan", str(PLAN)],
    ):
        code, line, _, _ = invoke([*argv, "--backend", "isaac"])
        assert code == 2
        assert line["status"] == "INVALID_INPUT"


def test_extra_task_files_are_allowed_but_ignored(tmp_path):
    task_dir = copy_task_dir(tmp_path / "task")
    (task_dir / "operator-notes.txt").write_text("extra", encoding="utf-8")
    code, line = run_example1(tmp_path / "run")
    assert code == 7
    assert line["status"] == "EXECUTION_NOT_IMPLEMENTED"


# ---------------------------------------------------------------------------
# TCP gate: injected artifacts
# ---------------------------------------------------------------------------


def test_fake_resolved_tcp_reaches_execution_not_implemented(tmp_path, monkeypatch):
    patch_artifact(monkeypatch, fake_artifact(), tmp_path)
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(PLAN),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
    ])
    assert code == 7
    assert line["status"] == "EXECUTION_NOT_IMPLEMENTED"
    assert line["passed"] is False
    evidence_dir, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["tcp"]["derived_status"] == "RESOLVED"
    assert preflight["tcp"]["serialized_status"] == "RESOLVED"
    assert preflight["tcp"]["gate"] == "PASSED"
    assert preflight["tcp"]["require_resolved_tcp"] == "satisfied"
    assert preflight["tcp"]["binding_requires_tcp"] is True
    assert preflight["tcp"]["derived_transform"]["child_frame"] == "tool0"
    assert preflight["dispatch"]["implemented"] is False
    assert preflight["dispatch"]["blocked"] is True
    assert preflight["checks"]["no_motion_dispatched"] is True
    result = json.loads(Path(line["result_path"]).read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert result["execution_safe"] is False
    by_phase = {event["event"]["phase"]: event["event"] for event in events}
    assert by_phase["dispatch_blocked"]["status"] == "EXECUTION_NOT_IMPLEMENTED"


def test_fake_resolved_tcp_preflight_only_passes(tmp_path, monkeypatch):
    patch_artifact(monkeypatch, fake_artifact(), tmp_path)
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(PLAN),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
        "--preflight-only",
    ])
    assert code == 0
    assert line["status"] == "PREFLIGHT_ONLY"
    assert line["passed"] is True
    _, preflight_path, _events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["status"] == "PREFLIGHT_ONLY"
    assert preflight["dispatch"]["dispatch_requested"] is False
    result = json.loads(Path(line["result_path"]).read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["execution_safe"] is True


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
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(PLAN),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
    ])
    assert code == 4
    assert line["status"] == "UNRESOLVED_TCP"
    assert line["passed"] is False
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["tcp"]["serialized_status"] == "UNRESOLVED"
    assert preflight["tcp"]["derived_status"] == "UNRESOLVED"
    assert preflight["tcp"]["status_matches_serialized"] is True
    assert preflight["tcp"]["gate"] == "UNRESOLVED_TCP"
    assert preflight["tcp"]["derived_blockers"]
    assert preflight["dispatch"]["status"] == "UNRESOLVED_TCP"
    assert preflight["checks"]["no_motion_dispatched"] is True
    by_phase = {event["event"]["phase"]: event["event"] for event in events}
    assert by_phase["dispatch_blocked"]["status"] == "UNRESOLVED_TCP"


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
    resolved = piper_run._resolve_and_read(TASK_DIR, PLAN)
    assert resolved.plan_bytes == PLAN.read_bytes()
    assert resolved.start_bytes == TASK_START.read_bytes()
    assert resolved.goal_bytes == TASK_GOAL.read_bytes()


# ---------------------------------------------------------------------------
# --operation: the internal R4.2 runtime session bridge
# ---------------------------------------------------------------------------


from rocobrick.isaac_backend import (  # noqa: E402
    OPERATION_SCHEMA as OPERATION_SCHEMA_CONST,
)
from rocobrick.isaac_backend import RuntimeOperation  # noqa: E402
from rocobrick.runtime_session import (  # noqa: E402
    RuntimeReadinessEvidence,
    RuntimeSessionError,
    RuntimeSessionEvidence,
    RuntimeSessionStatus,
)

ADMITTED_JOINT_POSITIONS = {
    "joint1": 0.0,
    "joint2": 0.5,
    "joint3": -0.5,
    "joint4": 0.3,
    "joint5": 0.0,
    "joint6": 0.0,
}


def joint_operation_document(**overrides) -> dict:
    """A strictly admissible JOINT_TARGET RuntimeOperation document."""
    document = {
        "schema": OPERATION_SCHEMA_CONST,
        "kind": "JOINT_TARGET",
        "payload": {
            "positions": dict(ADMITTED_JOINT_POSITIONS),
            "goal_tolerance": 0.01,
        },
        "timeout_s": 60.0,
        "request_id": "op-joint-1",
    }
    document.update(overrides)
    return document


def link6_pose_operation_document(**overrides) -> dict:
    """A strictly admissible LINK6_POSE RuntimeOperation document."""
    document = {
        "schema": OPERATION_SCHEMA_CONST,
        "kind": "LINK6_POSE",
        "payload": {
            "frame_id": "base_link",
            "position": [0.2, 0.0, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "position_tolerance_m": 0.005,
            "orientation_tolerance_rad": 0.015,
            "tip_link": "link6",
            "tcp_status": "unresolved",
        },
        "timeout_s": 60.0,
        "request_id": "op-pose-1",
    }
    document.update(overrides)
    return document


def passing_result(document: dict) -> dict:
    """A passing RuntimeOperationResult document answering ``document``."""
    return {
        "schema": "rocobrick-runtime-operation-result/v1",
        "request_id": document["request_id"],
        "kind": document["kind"],
        "passed": True,
        "execution": {
            "status": "SUCCEEDED",
            "is_success": True,
            "command_messages": 3,
            "moveit_error_code": 1,
            "planned_waypoints": 4,
            "final_errors": {},
        },
        "evidence": {"waited_ok": True, "passed": True, "snapshot": {}},
        "error": None,
    }


class FakeSession:
    """Minimal IsaacMoveItSession stand-in recording every lifecycle call."""

    def __init__(
        self,
        session_dir,
        *,
        calls,
        result=None,
        start_error=None,
        run_error=None,
        cleanup_passed=True,
        mutate_path=None,
    ):
        self.session_dir = Path(session_dir)
        self._calls = calls
        self._result = result
        self._start_error = start_error
        self._run_error = run_error
        self._cleanup_passed = cleanup_passed
        self._mutate_path = mutate_path

    def start(self):
        self._calls.append(("start",))
        if self._start_error is not None:
            raise RuntimeSessionError(self._start_error)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        return RuntimeReadinessEvidence(
            host_marker=True,
            container_started=True,
            moveit_ready=True,
            controllers_active=True,
            adapter_ready=True,
        )

    def run_operation(self, request_path, result_path):
        self._calls.append(
            ("run_operation", Path(request_path).name, Path(result_path).name)
        )
        operation = RuntimeOperation.parse(
            json.loads(Path(request_path).read_text(encoding="utf-8"))
        )
        assert operation.request_id == self._result["request_id"]
        if self._mutate_path is not None:
            self._mutate_path.write_text(
                json.dumps({"mutated": True}), encoding="utf-8"
            )
        if self._run_error is not None:
            raise RuntimeSessionError(self._run_error)
        (Path(result_path)).write_text(json.dumps(self._result), encoding="utf-8")
        return self._result

    def stop(self):
        self._calls.append(("stop",))
        failed = self._start_error is not None or self._run_error is not None
        return RuntimeSessionEvidence(
            status=(
                RuntimeSessionStatus.FAILED
                if failed
                else RuntimeSessionStatus.STOPPED
            ),
            readiness=None,
            host_pid=4242,
            container_name="rocobrick-runtime-session-fake",
            cleanup={"passed": self._cleanup_passed},
            errors=(),
        )


def install_fake_session(monkeypatch, calls, **kwargs) -> list:
    """Patch ``piper_run._create_session`` with a FakeSession factory."""
    created = []

    def factory(session_dir):
        session = FakeSession(session_dir, calls=calls, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(piper_run, "_create_session", factory)
    return created


def run_with_operation(tmp_path, document, *extra) -> tuple[int, dict]:
    """Write the operation document and run the CLI with ``--operation``."""
    operation_path = tmp_path / "operation.json"
    operation_path.write_text(json.dumps(document), encoding="utf-8")
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(PLAN),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
        "--operation", str(operation_path),
        *extra,
    ])
    return code, line


def test_preflight_only_cannot_combine_with_operation(tmp_path):
    operation_path = tmp_path / "operation.json"
    operation_path.write_text(json.dumps(joint_operation_document()))
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(PLAN),
        "--backend", "isaac",
        "--preflight-only",
        "--operation", str(operation_path),
    ])
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    assert line["run_dir"] is None
    assert line["result_path"] is None


@pytest.mark.parametrize(
    "argv_extra",
    [["--preflight-only"], []],
    ids=["preflight-only", "default"],
)
def test_gates_run_before_the_session_starts(
    tmp_path, monkeypatch, argv_extra
):
    """An invalid plan or TCP failure means the runner is never created."""
    calls: list = []
    install_fake_session(monkeypatch, calls)
    tampered = tmp_path / "plan.json"
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["plan_id"] = "tampered-plan"
    tampered.write_text(json.dumps(plan), encoding="utf-8")
    operation_path = tmp_path / "operation.json"
    operation_path.write_text(json.dumps(joint_operation_document()))
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(tampered),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
        "--operation", str(operation_path),
        *argv_extra,
    ])
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    assert calls == []


def test_tcp_failure_with_operation_starts_zero_sessions(
    tmp_path, monkeypatch
):
    calls: list = []
    install_fake_session(monkeypatch, calls)
    patch_artifact(monkeypatch, None, tmp_path)
    code, line = run_with_operation(tmp_path, joint_operation_document())
    assert code == 5
    assert line["status"] == "TCP_CONTRACT_INVALID"
    assert calls == []


def _runtime_binding_hashes() -> tuple[str, str]:
    """Return the exact runtime and calibration hashes from the artifact."""
    artifact = json.loads(
        piper_run.TCP_ARTIFACT.read_text(encoding="utf-8")
    )
    runtime_sha = next(
        entry["sha256"]
        for entry in artifact["inputs"]["pinned_files"]
        if entry["path"] == piper_run.RUNTIME_USD_PINNED_PATH
    )
    calibration_sha = artifact["tcp_contract"]["transform"]["sha256"]
    return runtime_sha, calibration_sha


def test_tcp_authorization_precedes_session_and_records_verified_provenance(
    tmp_path, monkeypatch
):
    document = joint_operation_document()
    calls: list = []
    order: list[str] = []
    observed: list[tuple[object, dict]] = []
    real_authorize = piper_run.authorize_tcp

    def authorize(contract, **kwargs):
        order.append("authorize")
        observed.append((contract, kwargs))
        return real_authorize(contract, **kwargs)

    monkeypatch.setattr(piper_run, "authorize_tcp", authorize)
    created: list[FakeSession] = []

    def factory(session_dir):
        order.append("create_session")
        session = FakeSession(
            session_dir,
            calls=calls,
            result=passing_result(document),
        )
        created.append(session)
        return session

    monkeypatch.setattr(piper_run, "_create_session", factory)
    before = input_hashes()
    code, line = run_with_operation(tmp_path, document)

    assert code == 0
    assert line["status"] == "EXECUTED"
    assert order == ["authorize", "create_session"]
    assert len(created) == 1
    assert calls == [
        ("start",),
        ("run_operation", "operation-request.json", "operation-result.json"),
        ("stop",),
    ]
    assert input_hashes() == before

    runtime_sha, calibration_sha = _runtime_binding_hashes()
    assert len(observed) == 1
    contract, kwargs = observed[0]
    assert contract.status.value == "RESOLVED"
    assert contract.transform.sha256 == calibration_sha
    assert kwargs == {
        "backend": "isaac",
        "runtime_asset_sha256": runtime_sha,
        "calibration_artifact_sha256": calibration_sha,
    }

    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    authorization = preflight["tcp"]["authorization"]
    assert preflight["operation"]["tcp_authorization"] == authorization
    assert authorization["decision"] == "ALLOW"
    assert authorization["allowed"] is True
    assert authorization["status"] == "TCP_AUTHORIZED"
    assert authorization["checks"]
    assert authorization["backend"] == "isaac"
    assert authorization["provenance"] == "verified_binding_artifact"
    assert authorization["runtime_rehash_performed"] is False
    assert authorization["runtime_asset_sha256"] == runtime_sha
    assert authorization["calibration_artifact_sha256"] == calibration_sha
    phases = [event["event"]["phase"] for event in events]
    assert phases.index("operation_admission") < phases.index(
        "tcp_authorization"
    ) < phases.index("runtime_start")


@pytest.mark.parametrize("mode", ["deny", "exception", "malformed"])
def test_tcp_authorization_failures_create_zero_sessions(
    tmp_path, monkeypatch, mode
):
    if mode == "deny":
        def fake_authorize(*_args, **_kwargs):
            return SimpleNamespace(
                allowed=False,
                status="UNRESOLVED_TCP",
                reason="synthetic deny",
                checks={"synthetic_check": False},
            )
    elif mode == "exception":
        def fake_authorize(*_args, **_kwargs):
            raise RuntimeError("synthetic authorize failure")
    else:
        def fake_authorize(*_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    monkeypatch.setattr(piper_run, "authorize_tcp", fake_authorize)
    calls: list = []
    created = install_fake_session(
        monkeypatch,
        calls,
        result=passing_result(joint_operation_document()),
    )
    code, line = run_with_operation(tmp_path, joint_operation_document())

    assert code == 4
    assert line["status"] == "UNRESOLVED_TCP"
    assert calls == []
    assert created == []
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    authorization = preflight["tcp"]["authorization"]
    assert authorization["decision"] == "DENY"
    assert authorization["allowed"] is False
    assert authorization["backend"] == "isaac"
    assert authorization["provenance"] == "verified_binding_artifact"
    assert authorization["runtime_rehash_performed"] is False
    assert authorization["runtime_asset_sha256"] is not None
    assert authorization["calibration_artifact_sha256"] is not None
    if mode == "deny":
        assert authorization["status"] == "UNRESOLVED_TCP"
        assert authorization["reason"] == "synthetic deny"
        assert authorization["checks"] == {"synthetic_check": False}
    else:
        assert "failed closed" in authorization["reason"]
        assert authorization["checks"] == {}
    phases = [event["event"]["phase"] for event in events]
    assert "operation_admission" in phases
    assert "tcp_authorization" in phases
    assert "runtime_start" not in phases


@pytest.mark.parametrize(
    ("case", "value"),
    [
        ("zero-candidates", None),
        ("multiple-candidates", None),
        ("missing-hash", None),
        ("none-hash", None),
        ("integer-hash", 123),
        ("empty-hash", ""),
        ("short-hash", "0" * 63),
        ("long-hash", "0" * 65),
        ("uppercase-hash", "A" * 64),
        ("nonhex-hash", "g" * 64),
    ],
)
def test_verified_runtime_hash_extraction_fails_closed(
    tmp_path, monkeypatch, case, value
):
    real_rederive = piper_run._rederive_tcp

    def rederive(record):
        contract = real_rederive(record)
        pinned_files = record["verified_binding_record"]["inputs"]["pinned_files"]
        runtime_entries = [
            entry
            for entry in pinned_files
            if entry["path"] == piper_run.RUNTIME_USD_PINNED_PATH
        ]
        assert len(runtime_entries) == 1
        if case == "zero-candidates":
            pinned_files[:] = [
                entry
                for entry in pinned_files
                if entry["path"] != piper_run.RUNTIME_USD_PINNED_PATH
            ]
        elif case == "multiple-candidates":
            pinned_files.append(dict(runtime_entries[0]))
        elif case == "missing-hash":
            runtime_entries[0].pop("sha256")
        else:
            runtime_entries[0]["sha256"] = value
        return contract

    monkeypatch.setattr(piper_run, "_rederive_tcp", rederive)

    def forbidden_authorize(*_args, **_kwargs):
        pytest.fail("authorize_tcp must not run after failed hash extraction")

    monkeypatch.setattr(piper_run, "authorize_tcp", forbidden_authorize)
    calls: list = []
    created = install_fake_session(monkeypatch, calls)
    before = input_hashes()
    code, line = run_with_operation(tmp_path, joint_operation_document())

    assert code == 4
    assert line["status"] == "UNRESOLVED_TCP"
    assert calls == []
    assert created == []
    assert input_hashes() == before
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    authorization = preflight["tcp"]["authorization"]
    assert authorization["decision"] == "DENY"
    assert authorization["allowed"] is False
    assert authorization["status"] == "UNRESOLVED_TCP"
    assert authorization["provenance"] == "verified_binding_artifact"
    assert authorization["runtime_rehash_performed"] is False
    assert authorization["runtime_asset_sha256"] is None
    assert authorization["calibration_artifact_sha256"] is None
    assert any(
        event["event"]["phase"] == "tcp_authorization"
        for event in events
    )


def test_runtime_operation_cli_accepts_only_exact_isaac_backend():
    assert piper_run.BACKENDS == ("isaac",)


def test_valid_joint_operation_executes_and_reports_executed(
    tmp_path, monkeypatch
):
    document = joint_operation_document()
    calls: list = []
    install_fake_session(
        monkeypatch, calls, result=passing_result(document)
    )
    before = input_hashes()
    code, line = run_with_operation(tmp_path, document)
    assert code == 0
    assert line["status"] == "EXECUTED"
    assert line["passed"] is True
    assert calls == [
        ("start",),
        ("run_operation", "operation-request.json", "operation-result.json"),
        ("stop",),
    ]
    assert input_hashes() == before
    evidence_dir, preflight_path, events = read_evidence(line)
    assert (evidence_dir / "operation-request.json").is_file()
    request = json.loads(
        (evidence_dir / "operation-request.json").read_text(encoding="utf-8")
    )
    assert request == document
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["status"] == "EXECUTED"
    assert preflight["passed"] is True
    assert preflight["exit_code"] == 0
    assert preflight["dispatch"]["implemented"] is True
    assert preflight["dispatch"]["blocked"] is False
    assert preflight["operation"]["kind"] == "JOINT_TARGET"
    assert preflight["operation"]["passed"] is True
    assert preflight["operation"]["cleanup_passed"] is True
    assert preflight["runtime"]["cleanup"]["passed"] is True
    assert preflight["runtime"]["container_name"] == (
        "rocobrick-runtime-session-fake"
    )
    assert preflight["checks"]["no_motion_dispatched"] is False
    assert preflight["checks"]["operation_dispatched"] is True
    assert preflight["checks"]["inputs_immutable_after_runtime"] is True
    phases = [event["event"]["phase"] for event in events]
    assert "operation_admission" in phases
    assert "runtime_start" in phases
    assert "operation" in phases
    assert "runtime_stop" in phases
    result = json.loads(Path(line["result_path"]).read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["execution_safe"] is True
    assert result["operation"]["request_id"] == "op-joint-1"
    assert result["runtime_session"]["evidence_path"].endswith(
        "runtime-session.json"
    )


def test_valid_link6_operation_delegates_to_the_session(tmp_path, monkeypatch):
    document = link6_pose_operation_document()
    calls: list = []
    install_fake_session(monkeypatch, calls, result=passing_result(document))
    code, line = run_with_operation(tmp_path, document)
    assert code == 0
    assert line["status"] == "EXECUTED"
    _, preflight_path, _ = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["operation"]["kind"] == "LINK6_POSE"
    assert preflight["operation"]["passed"] is True


def test_admission_rejected_operation_starts_zero_sessions(
    tmp_path, monkeypatch
):
    document = joint_operation_document(
        payload={
            "positions": {**ADMITTED_JOINT_POSITIONS, "joint2": 4.0},
            "goal_tolerance": 0.01,
        }
    )
    calls: list = []
    install_fake_session(monkeypatch, calls)
    code, line = run_with_operation(tmp_path, document)
    assert code == 9
    assert line["status"] == "OPERATION_REJECTED"
    assert line["passed"] is False
    assert calls == []
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["operation"]["kind"] == "JOINT_TARGET"
    assert preflight["operation"]["passed"] is False
    assert preflight["runtime"] is None
    assert preflight["checks"]["no_motion_dispatched"] is False
    phases = [event["event"]["phase"] for event in events]
    assert "runtime_start" not in phases


def test_readiness_kind_is_not_dispatchable(tmp_path, monkeypatch):
    document = {
        "schema": OPERATION_SCHEMA_CONST,
        "kind": "READINESS",
        "payload": {},
        "timeout_s": 30.0,
        "request_id": "op-readiness-1",
    }
    calls: list = []
    install_fake_session(monkeypatch, calls)
    code, line = run_with_operation(tmp_path, document)
    assert code == 9
    assert line["status"] == "OPERATION_REJECTED"
    assert calls == []


def test_malformed_operation_document_is_invalid_input(tmp_path, monkeypatch):
    operation_path = tmp_path / "operation.json"
    operation_path.write_text("{ not json", encoding="utf-8")
    calls: list = []
    install_fake_session(monkeypatch, calls)
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(PLAN),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
        "--operation", str(operation_path),
    ])
    assert code == 2
    assert line["status"] == "INVALID_INPUT"
    assert calls == []


def test_runtime_start_failure_reports_runtime_start_failed(
    tmp_path, monkeypatch
):
    calls: list = []
    install_fake_session(
        monkeypatch, calls, start_error="host_marker failed: marker not seen"
    )
    code, line = run_with_operation(
        tmp_path, joint_operation_document()
    )
    assert code == 8
    assert line["status"] == "RUNTIME_START_FAILED"
    assert line["passed"] is False
    assert calls == [("start",), ("stop",)]
    _, preflight_path, events = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["runtime"]["cleanup"]["passed"] is True
    assert preflight["operation"]["passed"] is False
    assert preflight["passed"] is False
    by_phase = {
        event["event"]["phase"]: event["event"]["status"] for event in events
    }
    assert by_phase["runtime_start"] == "FAILED"
    assert by_phase["runtime_stop"] == "FAILED"


def test_runtime_operation_failure_reports_execution_failed(
    tmp_path, monkeypatch
):
    document = joint_operation_document()
    failed = {
        **passing_result(document),
        "passed": False,
        "execution": {
            "status": "FAILED",
            "is_success": False,
            "command_messages": 0,
        },
        "error": "the joint goal did not pass the execution gates",
    }
    calls: list = []
    install_fake_session(monkeypatch, calls, result=failed)
    code, line = run_with_operation(tmp_path, document)
    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    assert line["passed"] is False
    assert calls == [
        ("start",),
        ("run_operation", "operation-request.json", "operation-result.json"),
        ("stop",),
    ]
    _, preflight_path, _ = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["operation"]["passed"] is False
    assert preflight["passed"] is False


def test_runtime_rejected_operation_reports_operation_rejected(
    tmp_path, monkeypatch
):
    document = joint_operation_document()
    rejected = {
        **passing_result(document),
        "passed": False,
        "execution": {
            "status": "REJECTED",
            "is_success": False,
            "reason": "runtime readiness failed; no motion is allowed",
            "command_messages": 0,
        },
        "error": "the JOINT_TARGET goal did not pass the execution gates",
    }
    calls: list = []
    install_fake_session(monkeypatch, calls, result=rejected)
    code, line = run_with_operation(tmp_path, document)
    assert code == 9
    assert line["status"] == "OPERATION_REJECTED"


def test_failed_cleanup_blocks_executed(tmp_path, monkeypatch):
    document = joint_operation_document()
    calls: list = []
    install_fake_session(
        monkeypatch,
        calls,
        result=passing_result(document),
        cleanup_passed=False,
    )
    code, line = run_with_operation(tmp_path, document)
    assert code == 10
    assert line["status"] == "EXECUTION_FAILED"
    assert line["passed"] is False
    _, preflight_path, _ = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["operation"]["passed"] is True
    assert preflight["operation"]["cleanup_passed"] is False
    assert preflight["passed"] is False


def test_input_mutation_after_runtime_is_input_mutated(tmp_path, monkeypatch):
    plan_copy = tmp_path / "plan-copy.json"
    plan_copy.write_bytes(PLAN.read_bytes())
    operation_path = tmp_path / "operation.json"
    operation_path.write_text(json.dumps(joint_operation_document()))
    calls: list = []
    install_fake_session(
        monkeypatch,
        calls,
        result=passing_result(joint_operation_document()),
        mutate_path=plan_copy,
    )
    code, line, _, _ = invoke([
        "--task", str(TASK_DIR),
        "--plan", str(plan_copy),
        "--backend", "isaac",
        "--run-dir", str(tmp_path / "run"),
        "--operation", str(operation_path),
    ])
    assert code == 6
    assert line["status"] == "INPUT_MUTATED"
    assert calls == [
        ("start",),
        ("run_operation", "operation-request.json", "operation-result.json"),
        ("stop",),
    ]
    _, preflight_path, _ = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["checks"]["inputs_immutable_after_runtime"] is False
    assert preflight["operation"]["passed"] is True
    assert preflight["passed"] is False


def test_operation_without_operation_file_leaves_exit7_unchanged(
    tmp_path, monkeypatch
):
    calls: list = []
    install_fake_session(monkeypatch, calls)
    code, line = run_example1(tmp_path / "run")
    assert code == 7
    assert line["status"] == "EXECUTION_NOT_IMPLEMENTED"
    assert calls == []
    _, preflight_path, _ = read_evidence(line)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["operation"] is None
    assert preflight["runtime"] is None
    assert preflight["checks"]["no_motion_dispatched"] is True
    assert preflight["checks"]["operation_dispatched"] is False
