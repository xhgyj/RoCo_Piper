from __future__ import annotations

import json
from pathlib import Path

import pytest

from rocobrick.isaac_snapshot_host_adapter import (
    SceneSnapshotHostAdapterError,
    build_host_binding,
    collect_stage_snapshot,
    load_scene_snapshot_manifest,
    snapshot_artifact_payload,
    write_snapshot_artifact_atomic,
)
from rocobrick.scene_snapshot_collector import SceneObjectManifest, SceneSnapshotManifest

H = "a" * 64

def host():
    return build_host_binding(host_instance_id="host-1", ros_domain_id=0,
        host_profile_sha256=H, host_bridge_schema="rocobrick-host-bridge-smoke/v1",
        runtime_usd_path="/robot.usd", runtime_usd_sha256=H, robot_id="piper_l",
        robot_prim_path="/World/Robot_0", articulation_prim_path="/World/Robot_0/root_joint",
        joint_dofs=("j1",), participant_count=1)

def manifest():
    return SceneSnapshotManifest(schema="rocobrick-scene-snapshot-manifest/v1", scene_id="s",
        frame_id="world", objects=(SceneObjectManifest("part-1", 1, "/World/Parts/part-1", H, {"source":"m"}),))

def stage():
    class Prim:
        def IsValid(self): return True
    class Stage:
        def GetPrimAtPath(self, path): return Prim()
    return Stage()

def test_manifest_roundtrip(tmp_path):
    p=tmp_path/"m.json"
    p.write_text(json.dumps({"schema":"rocobrick-scene-snapshot-manifest/v1","scene_id":"s","frame_id":"world","objects":[{"physical_instance_id":"part-1","brick_id":1,"prim_path":"/World/Parts/part-1","asset_revision_sha256":H,"provenance":{"source":"m"}}]}))
    got=load_scene_snapshot_manifest(p)
    assert got.scene_id == "s" and got.objects[0].physical_instance_id == "part-1"

def test_manifest_invalid(tmp_path):
    p=tmp_path/"bad.json"; p.write_text("{}")
    with pytest.raises(SceneSnapshotHostAdapterError): load_scene_snapshot_manifest(p)

def test_collect_and_atomic_artifact(tmp_path, monkeypatch):
    import sys, types
    class Range:
        def GetMin(self): return (0.0, 0.0, 0.0)
        def GetMax(self): return (1.0, 1.0, 1.0)
    class Bound:
        def ComputeAlignedRange(self): return Range()
    class Cache:
        def __init__(self, *args): pass
        def ComputeWorldBound(self, prim): return Bound()
    fake_usd = types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda: object()))
    fake_geom = types.SimpleNamespace(BBoxCache=Cache, Tokens=types.SimpleNamespace(default_=object()))
    monkeypatch.setitem(sys.modules, "pxr", types.SimpleNamespace(Usd=fake_usd, UsdGeom=fake_geom))
    snapshot=collect_stage_snapshot(stage(), manifest(), host())
    payload=snapshot_artifact_payload(snapshot)
    assert payload["snapshot_sha256"] == snapshot.snapshot_sha256
    out=tmp_path/"snapshot.json"; write_snapshot_artifact_atomic(snapshot, out)
    assert json.loads(out.read_text())["snapshot_sha256"] == snapshot.snapshot_sha256

def test_bad_stage_fails_closed():
    class Bad:
        pass
    with pytest.raises(SceneSnapshotHostAdapterError): collect_stage_snapshot(Bad(), manifest(), host())
