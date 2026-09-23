from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

from rocobrick.planning_scene import box_from_aabb
from rocobrick.scene_snapshot import (
    RuntimeHostBinding,
    SceneObjectSnapshot,
    SceneSnapshot,
)
from rocobrick.scene_snapshot_collector import (
    SceneObjectManifest,
    SceneSnapshotManifest,
    collect_scene_snapshot,
)

__all__ = [
    "SceneSnapshotHostAdapterError", "load_scene_snapshot_manifest",
    "build_host_binding", "make_world_aabb_reader", "collect_stage_snapshot",
    "snapshot_artifact_payload", "write_snapshot_artifact_atomic",
]

class SceneSnapshotHostAdapterError(ValueError):
    pass


def _fail(kind: str, detail: str, cause: BaseException | None = None):
    err = SceneSnapshotHostAdapterError(f"{kind}: {detail}")
    if cause is not None:
        raise err from cause
    raise err


def load_scene_snapshot_manifest(path: str | os.PathLike[str]) -> SceneSnapshotManifest:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict): _fail("manifest", "top-level must be object")
        objects = raw.get("objects")
        if not isinstance(objects, list): _fail("manifest", "objects must be array")
        parsed = []
        for i, item in enumerate(objects):
            if not isinstance(item, dict): _fail("manifest", f"object[{i}] must be object")
            parsed.append(SceneObjectManifest(
                physical_instance_id=item["physical_instance_id"], brick_id=item["brick_id"],
                prim_path=item["prim_path"], asset_revision_sha256=item["asset_revision_sha256"],
                provenance=item["provenance"],
            ))
        return SceneSnapshotManifest(schema=raw["schema"], scene_id=raw["scene_id"],
                                     frame_id=raw["frame_id"], objects=tuple(parsed))
    except SceneSnapshotHostAdapterError:
        raise
    except Exception as exc:
        _fail("manifest", "invalid JSON or schema", exc)


def build_host_binding(*, host_instance_id: str, ros_domain_id: int,
                       host_profile_sha256: str, host_bridge_schema: str,
                       runtime_usd_path: str, runtime_usd_sha256: str,
                       robot_id: str, robot_prim_path: str,
                       articulation_prim_path: str, joint_dofs: tuple[str, ...] | list[str],
                       participant_count: int = 1) -> RuntimeHostBinding:
    try:
        return RuntimeHostBinding(host_instance_id=host_instance_id, ros_domain_id=ros_domain_id,
            host_profile_sha256=host_profile_sha256, host_bridge_schema=host_bridge_schema,
            runtime_usd_path=runtime_usd_path, runtime_usd_sha256=runtime_usd_sha256,
            robot_id=robot_id, robot_prim_path=robot_prim_path,
            articulation_prim_path=articulation_prim_path, joint_dofs=tuple(joint_dofs),
            participant_count=participant_count)
    except Exception as exc:
        _fail("host", "invalid binding", exc)


def make_world_aabb_reader(stage):
    try:
        from pxr import Usd, UsdGeom
    except Exception as exc:
        _fail("bounds", "pxr unavailable", exc)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    def reader(prim_path: str):
        try:
            prim = stage.GetPrimAtPath(prim_path)
            if prim is None or hasattr(prim, "IsValid") and not prim.IsValid():
                _fail("bounds", f"invalid prim {prim_path}")
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            return tuple(float(v) for v in aligned.GetMin()), tuple(float(v) for v in aligned.GetMax())
        except SceneSnapshotHostAdapterError:
            raise
        except Exception as exc:
            _fail("bounds", f"failed for {prim_path}", exc)
    return reader


def collect_stage_snapshot(stage, manifest: SceneSnapshotManifest, host: RuntimeHostBinding) -> SceneSnapshot:
    if not hasattr(stage, "GetPrimAtPath"):
        _fail("stage", "missing GetPrimAtPath")
    try:
        reader = make_world_aabb_reader(stage)
        return collect_scene_snapshot(manifest, host, reader)
    except SceneSnapshotHostAdapterError:
        raise
    except Exception as exc:
        _fail("stage", "snapshot collection failed", exc)


def snapshot_artifact_payload(snapshot: SceneSnapshot) -> dict[str, Any]:
    if not isinstance(snapshot, SceneSnapshot): _fail("artifact", "invalid snapshot")
    payload = snapshot.to_canonical_payload()
    payload["snapshot_sha256"] = snapshot.snapshot_sha256
    return payload


def write_snapshot_artifact_atomic(snapshot: SceneSnapshot, path: str | os.PathLike[str]) -> None:
    target = Path(path)
    tmp = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmpname = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
        tmp = Path(tmpname)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(snapshot_artifact_payload(snapshot), fh, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, target); tmp = None
    except Exception as exc:
        if tmp is not None:
            try: tmp.unlink()
            except OSError: pass
        _fail("artifact", "atomic write failed", exc)
