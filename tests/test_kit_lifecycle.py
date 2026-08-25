"""Tests for deterministic shutdown of BrickSim command workflows."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from rocobrick.env.lifecycle import close_kit_app

ROOT = Path(__file__).resolve().parents[1]


def test_close_kit_app_pauses_and_requests_uncancellable_quit(monkeypatch) -> None:
    """A completed workflow must release both simulation and Kit process."""
    calls = []

    class Environment:
        async def pause(self) -> None:
            calls.append("pause")

    app = SimpleNamespace(
        post_uncancellable_quit=lambda code: calls.append(("quit", code))
    )
    app_module = ModuleType("omni.kit.app")
    app_module.get_app = lambda: app
    kit_module = ModuleType("omni.kit")
    kit_module.app = app_module
    omni_module = ModuleType("omni")
    omni_module.kit = kit_module
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.kit", kit_module)
    monkeypatch.setitem(sys.modules, "omni.kit.app", app_module)

    asyncio.run(close_kit_app(Environment(), 1))
    assert calls == ["pause", ("quit", 1)]


def test_every_bricksim_entry_closes_kit_in_finally() -> None:
    """All Env-based commands must close Kit on success and exceptions."""
    entries = (
        "demo.py",
        "main.py",
        "demo_gt_assembly.py",
        "demo_symbolic_assembly.py",
        "check_cameras.py",
    )
    for name in entries:
        source = (ROOT / "run" / name).read_text(encoding="utf-8")
        assert "finally:" in source
        assert "await close_kit_app(env, return_code)" in source
