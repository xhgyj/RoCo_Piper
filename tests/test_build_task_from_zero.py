"""Pure checks for the unified from-zero sequence runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "run/build_task_from_zero.py"
    spec = importlib.util.spec_from_file_location("build_task_from_zero", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_part_order_waits_for_every_support() -> None:
    """A bridge is scheduled only after both of its pillars are assembled."""
    runner = _runner_module()
    topology = {
        "parts": [{"id": item} for item in range(4)],
        "pose_hints": [{"part": 0}],
        "connections": [
            {"id": 0, "stud_id": 0, "hole_id": 1},
            {"id": 1, "stud_id": 1, "hole_id": 3},
            {"id": 2, "stud_id": 0, "hole_id": 2},
            {"id": 3, "stud_id": 2, "hole_id": 3},
        ],
    }
    assert runner._part_order(topology) == (1, 2, 3)
