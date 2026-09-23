import pytest
from types import SimpleNamespace
import rocobrick.runtime_host as rh

class App:
    def __init__(self): self.calls=0
    def shutdown(self): self.calls += 1

def scene(app=None):
    app=app or App()
    part=SimpleNamespace(prim_path="/World/managed/Parts/Part_2", brick_id=2)
    return SimpleNamespace(app=app, world=object(), robots={}, measured_robots={}, parts={"brick-1":part}, topology_paths={2:part.prim_path})

@pytest.mark.asyncio
async def test_bootstrap_calls_loader_once_and_maps(monkeypatch):
    app=App(); calls=[]
    async def fake(*args): calls.append(args); return scene(app)
    monkeypatch.setattr(rh, "load_verified_scene", fake)
    host=await rh.bootstrap_runtime_host(object(), object(), {}, object())
    assert len(calls)==1
    assert host.app is app
    assert host.physical_instance_paths["brick-1"] == "/World/managed/Parts/Part_2"
    assert host.physical_instance_brick_ids["brick-1"] == 2

def test_close_idempotent():
    app=App(); host=rh.LoadedRuntimeHost(scene(app), {"brick-1":"/p"}, {"brick-1":2})
    host.close(); host.close(); assert app.calls == 1

@pytest.mark.asyncio
async def test_bad_scene_fails_closed(monkeypatch):
    async def fake(*args): return SimpleNamespace(app=App(), world=object(), robots={}, measured_robots={}, parts={}, topology_paths={})
    monkeypatch.setattr(rh, "load_verified_scene", fake)
    with pytest.raises(rh.RuntimeHostBootstrapError):
        await rh.bootstrap_runtime_host(object(), object(), {}, object())
