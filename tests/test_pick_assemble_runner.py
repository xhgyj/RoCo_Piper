"""Pure checks for the unified Task-1 Pick -> PlaceDown runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _runner_module():
    """Load the runner without importing Isaac Sim.

    Returns:
        Loaded runner module.
    """
    path = ROOT / "run/test_pick_assemble.py"
    spec = importlib.util.spec_from_file_location("pick_assemble_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task(directory: Path) -> None:
    """Create the two files required for Task-1 discovery."""
    directory.mkdir(parents=True)
    for name in ("structure_start.json", "structure_goal.json"):
        (directory / name).write_text("{}\n", encoding="utf-8")


def test_discovers_explicit_and_recursive_tasks_once(tmp_path: Path) -> None:
    """Explicit and recursive references resolve to one deterministic path."""
    runner = _runner_module()
    first = tmp_path / "family" / "1"
    second = tmp_path / "family" / "2"
    _task(first)
    _task(second)
    discovered = runner._discover_tasks([second], [tmp_path])
    assert discovered == (first.resolve(), second.resolve())


def test_explicit_incomplete_task_is_rejected(tmp_path: Path) -> None:
    """A misspelled or absent start file fails before Isaac Sim startup."""
    runner = _runner_module()
    task = tmp_path / "incomplete"
    task.mkdir()
    (task / "structure_goal.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="structure_start.json"):
        runner._discover_tasks([task], [])


def test_temporary_config_overrides_only_task_selection(tmp_path: Path) -> None:
    """Runtime task selection leaves the repository configuration unchanged."""
    runner = _runner_module()
    task = tmp_path / "task"
    _task(task)
    output = tmp_path / "config"
    output.mkdir()
    base = {
        "Task_Config": {"Task_Path": "old", "Task_Type": "2"},
        "Robot_Config": {"Robots": []},
    }
    path = runner._temporary_user_config(base, task, output)
    generated = json.loads(path.read_text(encoding="utf-8"))
    assert generated["Task_Config"]["Task_Path"] == str(task)
    assert generated["Task_Config"]["Task_Type"] == "1"
    assert generated["Robot_Config"] == base["Robot_Config"]
    assert base["Task_Config"]["Task_Path"] == "old"
