"""Offline regression for PhysX-to-render synchronization without Kit reentry."""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace


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
