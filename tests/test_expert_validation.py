"""Tests for headless expert-validation discovery and reports."""

import json
from pathlib import Path

from rocobrick.policy.expert_validation import (
    TaskValidationResult,
    deterministic_validation_yaws,
    discover_tasks,
    failure_from_log,
    progress_bar,
    write_report,
)


def _task(root: Path, family: str, sample: str) -> None:
    directory = root / family / sample
    directory.mkdir(parents=True)
    for name in ("structure_start.json", "structure_goal.json"):
        (directory / name).write_text("{}\n", encoding="utf-8")


def test_discover_tasks_is_natural_and_family_filtered(tmp_path: Path) -> None:
    """Only complete task directories are returned in human numeric order."""
    _task(tmp_path, "basic", "10")
    _task(tmp_path, "basic", "2")
    _task(tmp_path, "dense", "1")
    incomplete = tmp_path / "basic/3"
    incomplete.mkdir(parents=True)
    (incomplete / "structure_start.json").write_text("{}")

    tasks = discover_tasks(tmp_path, ("basic",))
    assert [path.name for path in tasks] == ["2", "10"]


def test_report_contains_progress_results_and_log_link(tmp_path: Path) -> None:
    """HTML and JSON expose results outside terminal output."""
    results = [
        TaskValidationResult(
            task="dense/1",
            family="dense",
            status="failed",
            duration_seconds=12.5,
            failure="IK <failed>",
            log_name="task_logs/dense_1.log",
            initial_yaw_degrees=30.0,
        ),
        TaskValidationResult(task="bridge/3", family="bridge"),
    ]
    report = write_report(tmp_path / "report", tmp_path, results, running=True)
    document = report.read_text(encoding="utf-8")
    assert "1/2" in document
    assert "IK &lt;failed&gt;" in document
    assert "task_logs/dense_1.log" in document
    assert "30.000°" in document
    assert 'http-equiv="refresh"' in document
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["failed"] == 1
    assert summary["running"] is True


def test_failure_parser_and_progress_bar() -> None:
    """Concise failures and progress remain useful in batch output."""
    log = "rocobrick.policy.gt_assembly.SafeStartError: transport: slipped\n"
    assert failure_from_log(log, 1) == "transport: slipped"
    assert progress_bar(1, 4, width=4) == "[#---] 1/4"


def test_validation_yaws_are_continuous_task_stable_and_bounded() -> None:
    """Batch samples remain reproducible without depending on task order."""
    first = deterministic_validation_yaws("dense/1", count=4, seed=7)
    assert first == deterministic_validation_yaws("dense/1", count=4, seed=7)
    assert first != deterministic_validation_yaws("dense/2", count=4, seed=7)
    assert first != deterministic_validation_yaws("dense/1", count=4, seed=8)
    assert all(-180.0 <= value < 180.0 for value in first)
