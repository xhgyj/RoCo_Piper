"""Opt-in observer-viewport frame capture without starting Isaac."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def bridge_module():
    """Import lightweight bridge helpers without starting Isaac.

    Returns:
        The ordinary host module with deferred simulator imports.
    """
    path = Path(__file__).parents[1] / "isaac" / "p05_host_bridge.py"
    spec = importlib.util.spec_from_file_location("frame_capture_bridge_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_UNSET = object()


def _mock_viewport_sdk(
    monkeypatch, *, viewport, captured, fail=None, schedule=_UNSET
):
    """Install a fake viewport SDK.

    ``fail`` is raised from the capture's awaitable result; ``schedule``
    replaces the object ``capture_viewport_to_file`` hands back.
    """

    class Capture:
        def __init__(self, api, file_path):
            self._api = api
            self._file_path = file_path

        async def wait_for_result(self, completion_frames=2):
            if fail is not None:
                raise fail
            captured.append((self._api, self._file_path))

    def capture_viewport_to_file(api, file_path=None):
        if schedule is not _UNSET:
            return schedule
        return Capture(api, file_path)

    monkeypatch.setitem(
        sys.modules,
        "omni.kit.viewport.utility",
        SimpleNamespace(
            get_active_viewport=lambda: viewport,
            capture_viewport_to_file=capture_viewport_to_file,
        ),
    )


def test_frame_capture_is_opt_in(monkeypatch, tmp_path):
    """An unset or blank directory leaves capture disabled."""
    monkeypatch.delenv("ROCOBRICK_CAPTURE_DIR", raising=False)
    module = bridge_module()
    assert module.frame_capture_directory() is None

    monkeypatch.setenv("ROCOBRICK_CAPTURE_DIR", "   ")
    assert module.frame_capture_directory() is None

    target = tmp_path / "frames"
    monkeypatch.setenv("ROCOBRICK_CAPTURE_DIR", str(target))
    assert module.frame_capture_directory() == target
    assert target.is_dir()


def test_frame_capture_stride_defaults_and_validates(monkeypatch):
    """The stride is a positive integer that defaults to twelve."""
    monkeypatch.delenv("ROCOBRICK_CAPTURE_STRIDE", raising=False)
    module = bridge_module()
    assert module.frame_capture_stride() == 12

    monkeypatch.setenv("ROCOBRICK_CAPTURE_STRIDE", "5")
    assert module.frame_capture_stride() == 5

    for invalid in ("0", "-3", "later"):
        monkeypatch.setenv("ROCOBRICK_CAPTURE_STRIDE", invalid)
        with pytest.raises(ValueError, match="positive integer"):
            module.frame_capture_stride()


def test_capture_is_skipped_when_disabled(monkeypatch):
    """A disabled capture never reaches the Kit viewport utilities."""
    monkeypatch.delenv("ROCOBRICK_CAPTURE_DIR", raising=False)
    module = bridge_module()
    _mock_viewport_sdk(
        monkeypatch, viewport=SimpleNamespace(), captured=[]
    )
    state = module.BridgeState()

    asyncio.run(state.capture_viewport_frame())

    assert state.capture_renders == 0
    assert state.capture_count == 0


def test_capture_writes_every_stride_frame_with_sim_time(monkeypatch, tmp_path):
    """Frames land on disk at the stride and record the simulator time."""
    monkeypatch.setenv("ROCOBRICK_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("ROCOBRICK_CAPTURE_STRIDE", "2")
    module = bridge_module()
    viewport = SimpleNamespace()
    captured = []
    _mock_viewport_sdk(monkeypatch, viewport=viewport, captured=captured)
    state = module.BridgeState()
    state.world = SimpleNamespace(current_time=4.5)

    async def run():
        for _ in range(4):
            await state.capture_viewport_frame()

    asyncio.run(run())

    assert state.capture_renders == 4
    assert state.capture_count == 2
    assert state.capture_errors == 0
    names = [path.rsplit("/", 1)[-1] for _, path in captured]
    assert names == ["frame_000000.png", "frame_000001.png"]
    records = [
        json.loads(line)
        for line in (tmp_path / "frames.jsonl").read_text().splitlines()
    ]
    assert [record["file"] for record in records] == names
    assert [record["sim_time_s"] for record in records] == [4.5, 4.5]


def test_capture_failure_is_counted_and_never_raised(monkeypatch, tmp_path):
    """A failing capture records the cause instead of failing the run."""
    monkeypatch.setenv("ROCOBRICK_CAPTURE_DIR", str(tmp_path))
    monkeypatch.delenv("ROCOBRICK_CAPTURE_STRIDE", raising=False)
    module = bridge_module()
    _mock_viewport_sdk(
        monkeypatch,
        viewport=SimpleNamespace(),
        captured=[],
        fail=RuntimeError("no render product"),
    )
    state = module.BridgeState()

    asyncio.run(state.capture_viewport_frame())

    assert state.capture_count == 0
    assert state.capture_errors == 1
    assert state.capture_last_error == "RuntimeError: no render product"
    assert not list(tmp_path.glob("*.png"))


def test_unscheduled_capture_is_counted(monkeypatch, tmp_path):
    """A viewport that refuses to schedule a capture is reported, not fatal."""
    monkeypatch.setenv("ROCOBRICK_CAPTURE_DIR", str(tmp_path))
    monkeypatch.delenv("ROCOBRICK_CAPTURE_STRIDE", raising=False)
    module = bridge_module()
    _mock_viewport_sdk(
        monkeypatch, viewport=SimpleNamespace(), captured=[], schedule=None
    )
    state = module.BridgeState()

    asyncio.run(state.capture_viewport_frame())

    assert state.capture_count == 0
    assert state.capture_errors == 1
    assert state.capture_last_error == (
        "RuntimeError: viewport capture was not scheduled"
    )


def test_missing_viewport_is_counted_as_a_capture_error(monkeypatch, tmp_path):
    """A headed run without a viewport cannot capture frames."""
    monkeypatch.setenv("ROCOBRICK_CAPTURE_DIR", str(tmp_path))
    module = bridge_module()
    _mock_viewport_sdk(monkeypatch, viewport=None, captured=[])
    state = module.BridgeState()

    asyncio.run(state.capture_viewport_frame())

    assert state.capture_count == 0
    assert state.capture_errors == 1
    assert state.capture_last_error == "no active viewport to capture"
