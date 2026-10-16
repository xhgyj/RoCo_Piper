"""The declared example1 scene manifest artifact must stay real and parseable.

The artifact is a declared expectation document: the host bridge validates its
prim-path map and brick ids fail-closed against the live BrickSim scene at
session start. These tests pin the on-disk bytes to the real example1 values
(observed prim paths, plan-derived asset revision) so no silent drift can
weaken the declared binding. Offline only; no Isaac, ROS, or Docker.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rocobrick.scene_snapshot_collector import (
    SceneObjectManifest,
    SceneSnapshotManifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "scene" / "example1-scene-manifest.json"
PLAN_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "assembly_plans"
    / "example1-compound-pick.json"
)

EXPECTED_OBJECTS = (
    ("brick-4bbba49bec24496bb21e", 2, "/World/managed/Parts/Part_1"),
    ("brick-768de5a17827920c48d4", 2, "/World/managed/Parts/Part_2"),
)


def _load_manifest() -> SceneSnapshotManifest:
    raw = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    objects = [
        SceneObjectManifest(
            physical_instance_id=item["physical_instance_id"],
            brick_id=item["brick_id"],
            prim_path=item["prim_path"],
            asset_revision_sha256=item["asset_revision_sha256"],
            provenance=item["provenance"],
        )
        for item in raw["objects"]
    ]
    return SceneSnapshotManifest(
        schema=raw["schema"],
        scene_id=raw["scene_id"],
        frame_id=raw["frame_id"],
        objects=objects,
    )


def test_artifact_parses_through_the_strict_manifest_type() -> None:
    manifest = _load_manifest()
    assert manifest.scene_id == "example1-compound-pick-20260915T130000Z"
    assert manifest.frame_id == "world"


def test_artifact_carries_the_real_example1_instances_in_order() -> None:
    manifest = _load_manifest()
    ids = tuple(item.physical_instance_id for item in manifest.objects)
    assert ids == tuple(expected[0] for expected in EXPECTED_OBJECTS)
    assert ids == tuple(sorted(ids))
    for item, (instance_id, brick_id, prim_path) in zip(
        manifest.objects, EXPECTED_OBJECTS
    ):
        assert item.brick_id == brick_id
        assert item.prim_path == prim_path


def test_asset_revision_is_the_defining_plan_digest() -> None:
    import hashlib

    plan_sha256 = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    for item in _load_manifest().objects:
        assert item.asset_revision_sha256 == plan_sha256
        assert re.fullmatch(r"[0-9a-f]{64}", item.asset_revision_sha256)
        assert item.provenance == {"source": "example1-declared-manifest"}


def test_artifact_is_sorted_canonical_json() -> None:
    text = ARTIFACT_PATH.read_text(encoding="utf-8")
    assert text.endswith("\n")
    raw = json.loads(text)
    assert text == json.dumps(raw, indent=2, sort_keys=True) + "\n"
