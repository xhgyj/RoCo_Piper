"""Offline regression for PhysX-to-render synchronization without Kit reentry."""

import asyncio
import importlib.util
import inspect
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
    spec = importlib.util.spec_from_file_location("viewport_bridge_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fabric_sync_copies_current_transform_without_stepping():
    """Only the supported Fabric force-update gets called."""
    calls = []
    world = SimpleNamespace(get_physics_dt=lambda: 1 / 60, current_time=66.4)
    fabric = SimpleNamespace(force_update=lambda dt, now: calls.append((dt, now)))
    bridge_module().sync_viewport_physics(world, fabric)
    assert calls == [(1 / 60, 66.4)]


def test_viewport_refresh_uses_cached_fabric_and_counts_updates():
    """Repeated refreshes need no Kit imports or reentrant app update."""
    module = bridge_module()
    state = module.BridgeState()
    calls = []
    state.viewport_fabric = SimpleNamespace(
        force_update=lambda dt, now: calls.append((dt, now))
    )
    world = SimpleNamespace(get_physics_dt=lambda: 1 / 60, current_time=66.4)
    state.refresh_viewport(world)
    world.current_time = 66.5
    state.refresh_viewport(world)
    assert state.viewport_sync_count == 2
    assert calls == [(1 / 60, 66.4), (1 / 60, 66.5)]


def test_both_headed_loops_refresh_before_kit_yield():
    """Readiness and hold use identical no-reentry render synchronization."""
    source = (Path(__file__).parents[1] / "isaac" / "p05_host_bridge.py").read_text()
    blocks = source.split("if render_enabled and steps % render_stride == 0:")[1:]
    assert len(blocks) == 2
    for block in blocks:
        assert block.lstrip().startswith("await state.render_viewport(world)")
    module = bridge_module()
    renderer = inspect.getsource(module.BridgeState.render_viewport)
    assert "lambda: self.refresh_viewport(world)" in renderer
    helper = inspect.getsource(module.render_viewport_without_physics)
    assert helper.index("refresh()") < helper.index("await app.next_update_async()")
    assert helper.index("settings.set_bool(key, False)") < helper.index("refresh()")
    assert "finally:" in helper and "settings.set_bool(key, previous)" in helper


def _render_harness(*, on_capture=None):
    """Build a world/settings/app triple for the render-only helper.

    Returns:
        The triple plus the recorded app update count and capture flags.
    """
    world = SimpleNamespace(
        get_physics_dt=lambda: 1 / 60,
        current_time=10.0,
        current_time_step_index=600,
    )
    settings = SimpleNamespace(values={"/app/player/playSimulations": True})

    def get(key):
        return settings.values[key]

    def set_bool(key, value):
        settings.values[key] = value

    settings.get = get
    settings.set_bool = set_bool
    app = SimpleNamespace(updates=0)

    async def next_update_async():
        app.updates += 1

    app.next_update_async = next_update_async
    flags = []

    async def capture():
        flags.append(settings.values["/app/player/playSimulations"])
        app.updates += 2
        if on_capture is not None:
            on_capture(world)

    return world, settings, app, flags, capture


def test_frame_capture_runs_inside_the_physics_guard():
    """Capture's extra Kit frames never step physics under the bridge loop."""
    module = bridge_module()
    world, settings, app, flags, capture = _render_harness()

    asyncio.run(
        module.render_viewport_without_physics(
            world, settings, app, lambda: None, capture
        )
    )

    assert flags == [False]
    assert app.updates == 3
    assert settings.values["/app/player/playSimulations"] is True


def test_frame_capture_that_advances_physics_fails_closed():
    """A capture that steps physics is reported and the setting restored."""
    module = bridge_module()

    def advance(world):
        world.current_time += 1 / 60
        world.current_time_step_index += 1

    world, settings, app, _flags, capture = _render_harness(on_capture=advance)

    with pytest.raises(RuntimeError, match="frame capture advanced physics"):
        asyncio.run(
            module.render_viewport_without_physics(
                world, settings, app, lambda: None, capture
            )
        )

    assert settings.values["/app/player/playSimulations"] is True


def test_headed_keeps_bricksim_fabric_delegate(monkeypatch):
    """Robot visual repair must not disable BrickSim's Fabric render data."""
    module = bridge_module()
    args = module.parse_args([])
    monkeypatch.delenv("ROCOBRICK_HEADED", raising=False)
    default = module._kit_launch_argv(args)
    assert "--/app/useFabricSceneDelegate=false" not in default
    monkeypatch.setenv("ROCOBRICK_HEADED", "1")
    headed = module._kit_launch_argv(args)
    assert "--/app/useFabricSceneDelegate=false" not in headed
    assert "--/app/window/enabled=true" in headed
    assert "--/rtx/post/dof/enabled=false" in headed
    assert "--/rtx/post/motionblur/enabled=false" in headed
