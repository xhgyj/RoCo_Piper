from __future__ import annotations

import ast
import dataclasses
import inspect
from collections.abc import Mapping
from types import MappingProxyType

import pytest

from rocobrick import planning_scene
from rocobrick import scene_snapshot_collector as collector
from rocobrick.scene_snapshot import (
    SCENE_SNAPSHOT_SCHEMA,
    RuntimeHostBinding,
    SceneSnapshot,
)
from rocobrick.structural_binding import canonical_json_digest

SCHEMA = collector.SCENE_SNAPSHOT_MANIFEST_SCHEMA
HASH_A = "a" * 64
HASH_B = "b" * 64
FRAME = "world"
SCENE_ID = "scene-1"


def make_object(
    identifier: str,
    brick_id: int = 1,
    *,
    prim_path: str | None = None,
    asset_revision_sha256: str = HASH_A,
    provenance: Mapping[str, str] | None = None,
) -> collector.SceneObjectManifest:
    return collector.SceneObjectManifest(
        physical_instance_id=identifier,
        brick_id=brick_id,
        prim_path=prim_path or f"/World/{identifier}",
        asset_revision_sha256=asset_revision_sha256,
        provenance={"source": "fixture"} if provenance is None else provenance,
    )


def make_manifest(
    *objects: collector.SceneObjectManifest,
    scene_id: str = SCENE_ID,
    frame_id: str = FRAME,
) -> collector.SceneSnapshotManifest:
    return collector.SceneSnapshotManifest(
        schema=SCHEMA,
        scene_id=scene_id,
        frame_id=frame_id,
        objects=objects,
    )


def make_host(**overrides: object) -> RuntimeHostBinding:
    values: dict[str, object] = {
        "host_instance_id": "host-1",
        "ros_domain_id": 73,
        "host_profile_sha256": HASH_A,
        "host_bridge_schema": "rocobrick-host-bridge-smoke/v1",
        "runtime_usd_path": "/assets/piper.usd",
        "runtime_usd_sha256": HASH_B,
        "robot_id": "robot-0",
        "robot_prim_path": "/World/Robot_0",
        "articulation_prim_path": "/World/Robot_0/root_joint",
        "joint_dofs": ("joint1", "joint2"),
        "participant_count": 1,
    }
    values.update(overrides)
    return RuntimeHostBinding(**values)


def test_manifest_schema_and_collection_happy_path_preserve_order_and_identity() -> None:
    objects = (
        make_object("part-a", 7, provenance={"source": "manifest", "kind": "brick"}),
        make_object("part-b", 8, asset_revision_sha256=HASH_B),
    )
    manifest = make_manifest(*objects)
    host = make_host()
    bounds = {
        "/World/part-a": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        "/World/part-b": ((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
    }
    calls: list[str] = []

    def reader(prim_path: str) -> object:
        calls.append(prim_path)
        return bounds[prim_path]

    snapshot = collector.collect_scene_snapshot(manifest, host, reader)

    assert SCHEMA == "rocobrick-scene-snapshot-manifest/v1"
    assert isinstance(snapshot, SceneSnapshot)
    assert snapshot.schema == SCENE_SNAPSHOT_SCHEMA
    assert snapshot.scene_id == manifest.scene_id
    assert snapshot.frame_id == manifest.frame_id
    assert snapshot.host is host
    assert [item.physical_instance_id for item in snapshot.objects] == [
        "part-a",
        "part-b",
    ]
    assert [item.brick_id for item in snapshot.objects] == [7, 8]
    assert [item.prim_path for item in snapshot.objects] == [
        "/World/part-a",
        "/World/part-b",
    ]
    assert [dict(item.provenance) for item in snapshot.objects] == [
        {"source": "manifest", "kind": "brick"},
        {"source": "fixture"},
    ]
    assert calls == ["/World/part-a", "/World/part-b"]


def test_reader_is_called_once_per_prim_and_builder_receives_actual_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = make_manifest(make_object("part-a"), make_object("part-b", 2))
    host = make_host()
    expected = {
        "/World/part-a": ((0.0, 1.0, 2.0), (1.0, 3.0, 4.0)),
        "/World/part-b": ((4.0, 5.0, 6.0), (7.0, 8.0, 9.0)),
    }
    calls: list[tuple[str, object, object, object]] = []
    original = planning_scene.box_from_aabb

    def spy(
        *,
        object_id: str,
        frame_id: str,
        minimum_m: object,
        maximum_m: object,
    ) -> object:
        calls.append((object_id, frame_id, minimum_m, maximum_m))
        return original(
            object_id=object_id,
            frame_id=frame_id,
            minimum_m=minimum_m,
            maximum_m=maximum_m,
        )

    monkeypatch.setattr(planning_scene, "box_from_aabb", spy)
    reader_calls: list[str] = []

    def reader(prim_path: str) -> object:
        reader_calls.append(prim_path)
        return expected[prim_path]

    snapshot = collector.collect_scene_snapshot(manifest, host, reader)

    assert reader_calls == ["/World/part-a", "/World/part-b"]
    assert calls == [
        ("part-a", FRAME, (0.0, 1.0, 2.0), (1.0, 3.0, 4.0)),
        ("part-b", FRAME, (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)),
    ]
    assert [item.aabb.minimum_m for item in snapshot.objects] == [
        (0.0, 1.0, 2.0),
        (4.0, 5.0, 6.0),
    ]


def test_manifest_object_is_frozen_and_defensively_copies_provenance() -> None:
    provenance = {"source": "fixture"}
    item = make_object("part-a", provenance=provenance)
    provenance["source"] = "changed"

    assert item.provenance["source"] == "fixture"
    assert isinstance(item.provenance, MappingProxyType)
    assert not hasattr(item, "aabb")
    with pytest.raises(TypeError):
        item.provenance["new"] = "value"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.brick_id = 2  # type: ignore[misc]


def test_manifest_is_frozen_tuple_defensive_and_does_not_mutate_inputs() -> None:
    first = make_object("part-a")
    second = make_object("part-b", 2)
    objects = [first, second]
    manifest = make_manifest(*objects)
    original_payload = manifest.to_canonical_payload()

    objects.clear()
    assert manifest.objects == (first, second)
    assert isinstance(manifest.objects, tuple)
    assert manifest.to_canonical_payload() == original_payload
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.scene_id = "other"  # type: ignore[misc]


def test_manifest_payload_and_hash_are_fresh_and_exclude_geometry_and_plan_fields() -> None:
    manifest = make_manifest(make_object("part-a"))
    payload = manifest.to_canonical_payload()
    assert payload == {
        "schema": SCHEMA,
        "scene_id": SCENE_ID,
        "frame_id": FRAME,
        "objects": [
            {
                "physical_instance_id": "part-a",
                "brick_id": 1,
                "prim_path": "/World/part-a",
                "asset_revision_sha256": HASH_A,
                "provenance": {"source": "fixture"},
            }
        ],
    }
    assert manifest.manifest_sha256 == canonical_json_digest(payload)
    payload["objects"].append({"unexpected": "mutation"})
    payload["objects"][0]["provenance"]["changed"] = "value"
    assert manifest.to_canonical_payload() == {
        "schema": SCHEMA,
        "scene_id": SCENE_ID,
        "frame_id": FRAME,
        "objects": [
            {
                "physical_instance_id": "part-a",
                "brick_id": 1,
                "prim_path": "/World/part-a",
                "asset_revision_sha256": HASH_A,
                "provenance": {"source": "fixture"},
            }
        ],
    }
    forbidden_keys = {"plan", "task", "pose", "target", "freshness"}
    assert not forbidden_keys.intersection(payload)
    assert not forbidden_keys.intersection(payload["objects"][0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("physical_instance_id", ""),
        ("physical_instance_id", " part-a"),
        ("brick_id", True),
        ("brick_id", -1),
        ("prim_path", "World/part-a"),
        ("asset_revision_sha256", "A" * 64),
        ("asset_revision_sha256", "short"),
        ("provenance", {}),
        ("provenance", {"": "value"}),
        ("provenance", {"key": " "}),
        ("provenance", {1: "value"}),
    ],
)
def test_manifest_object_rejects_each_invalid_field(
    field: str, value: object
) -> None:
    values: dict[str, object] = {
        "physical_instance_id": "part-a",
        "brick_id": 1,
        "prim_path": "/World/part-a",
        "asset_revision_sha256": HASH_A,
        "provenance": {"source": "fixture"},
    }
    values[field] = value

    with pytest.raises(ValueError, match=field.split(".")[0]):
        collector.SceneObjectManifest(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "other/v1"),
        ("scene_id", ""),
        ("frame_id", " "),
        ("objects", []),
        ("objects", "not-a-sequence"),
        ("objects", b"not-a-sequence"),
    ],
)
def test_manifest_rejects_invalid_top_level_fields(field: str, value: object) -> None:
    values: dict[str, object] = {
        "schema": SCHEMA,
        "scene_id": SCENE_ID,
        "frame_id": FRAME,
        "objects": (make_object("part-a"),),
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        collector.SceneSnapshotManifest(**values)


def test_manifest_rejects_generator_objects_without_consuming_an_open_ended_source() -> None:
    def generated() -> object:
        yield make_object("part-a")

    with pytest.raises(ValueError, match="objects"):
        collector.SceneSnapshotManifest(SCHEMA, SCENE_ID, FRAME, generated())


def test_manifest_rejects_unsorted_and_duplicate_ids_and_prim_paths() -> None:
    part_a = make_object("part-a")
    part_b = make_object("part-b", 2)
    with pytest.raises(ValueError, match="ordered"):
        make_manifest(part_b, part_a)
    with pytest.raises(ValueError, match="physical_instance_id"):
        make_manifest(part_a, make_object("part-a", 9, prim_path="/World/other"))
    with pytest.raises(ValueError, match="prim_path"):
        make_manifest(part_a, make_object("part-b", 2, prim_path=part_a.prim_path))


@pytest.mark.parametrize("bad_reader", [None, object(), []])
def test_collection_requires_callable_reader(bad_reader: object) -> None:
    manifest = make_manifest(make_object("part-a"))

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(manifest, make_host(), bad_reader)  # type: ignore[arg-type]

    assert caught.value.physical_instance_id == "<reader>"
    assert caught.value.category == "reader_type"
    assert "reader_type" in str(caught.value)


def test_collection_requires_exact_manifest_and_host_types() -> None:
    class ManifestChild(collector.SceneSnapshotManifest):
        pass

    class HostChild(RuntimeHostBinding):
        pass

    manifest = make_manifest(make_object("part-a"))
    child_manifest = ManifestChild(
        manifest.schema, manifest.scene_id, manifest.frame_id, manifest.objects
    )
    child_host = HostChild(
        host_instance_id="host-1",
        ros_domain_id=73,
        host_profile_sha256=HASH_A,
        host_bridge_schema="rocobrick-host-bridge-smoke/v1",
        runtime_usd_path="/assets/piper.usd",
        runtime_usd_sha256=HASH_B,
        robot_id="robot-0",
        robot_prim_path="/World/Robot_0",
        articulation_prim_path="/World/Robot_0/root_joint",
        joint_dofs=("joint1", "joint2"),
        participant_count=1,
    )

    with pytest.raises(collector.SceneSnapshotCollectionError) as manifest_error:
        collector.collect_scene_snapshot(child_manifest, child_host, lambda _: [])

    assert manifest_error.value.category == "manifest_type"
    with pytest.raises(collector.SceneSnapshotCollectionError) as host_error:
        collector.collect_scene_snapshot(manifest, child_host, lambda _: [])

    assert host_error.value.category == "host_type"


def test_reader_exception_is_wrapped_once_with_object_id() -> None:
    manifest = make_manifest(make_object("part-a"))
    calls: list[str] = []

    def reader(path: str) -> object:
        calls.append(path)
        raise RuntimeError("reader failed")

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(manifest, make_host(), reader)

    assert calls == ["/World/part-a"]
    assert caught.value.physical_instance_id == "part-a"
    assert caught.value.category == "reader_error"


@pytest.mark.parametrize(
    "bad_bounds",
    [
        {"minimum": (0.0, 0.0, 0.0), "maximum": (1.0, 1.0, 1.0)},
        "not-bounds",
        b"not-bounds",
        ((0.0, 0.0, 0.0),),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
    ],
)
def test_reader_result_must_be_a_strict_two_element_sequence(
    bad_bounds: object,
) -> None:
    manifest = make_manifest(make_object("part-a"))

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(manifest, make_host(), lambda _: bad_bounds)

    assert caught.value.physical_instance_id == "part-a"
    assert caught.value.category == "bounds_shape"


def test_generator_reader_result_is_rejected() -> None:
    manifest = make_manifest(make_object("part-a"))

    def reader(_: str) -> object:
        def generated() -> object:
            yield (0.0, 0.0, 0.0)
            yield (1.0, 1.0, 1.0)

        return generated()

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(manifest, make_host(), reader)

    assert caught.value.category == "bounds_shape"


@pytest.mark.parametrize(
    "bad_bounds",
    [
        ((float("nan"), 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0), (float("inf"), 1.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
    ],
)
def test_aabb_numeric_validation_is_delegated_to_box_builder(
    bad_bounds: object,
) -> None:
    manifest = make_manifest(make_object("part-a"))

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(manifest, make_host(), lambda _: bad_bounds)

    assert caught.value.physical_instance_id == "part-a"
    assert caught.value.category == "aabb_validation"


def test_second_object_failure_has_no_partial_return_and_preserves_calls() -> None:
    manifest = make_manifest(make_object("part-a"), make_object("part-b", 2))
    calls: list[str] = []

    def reader(path: str) -> object:
        calls.append(path)
        if path == "/World/part-b":
            return ((0.0, 0.0, 0.0), (0.0, 1.0, 1.0))
        return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(manifest, make_host(), reader)

    assert calls == ["/World/part-a", "/World/part-b"]
    assert caught.value.physical_instance_id == "part-b"
    assert caught.value.category == "aabb_validation"


def test_bounds_change_changes_snapshot_hash_and_payload() -> None:
    manifest = make_manifest(make_object("part-a"))
    host = make_host()

    def reader_one(_: str) -> object:
        return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    def reader_two(_: str) -> object:
        return ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))

    first = collector.collect_scene_snapshot(manifest, host, reader_one)
    second = collector.collect_scene_snapshot(manifest, host, reader_two)

    assert first.snapshot_sha256 == canonical_json_digest(first.to_canonical_payload())
    assert second.snapshot_sha256 == canonical_json_digest(second.to_canonical_payload())
    assert first.snapshot_sha256 != second.snapshot_sha256
    assert first.to_canonical_payload()["objects"][0]["aabb"] != (
        second.to_canonical_payload()["objects"][0]["aabb"]
    )


def test_collection_does_not_mutate_manifest_or_reader_inputs() -> None:
    provenance = {"source": "fixture"}
    object_a = make_object("part-a", provenance=provenance)
    object_b = make_object("part-b", 2)
    input_objects = [object_a, object_b]
    manifest = make_manifest(*input_objects)
    bounds = {
        "/World/part-a": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        "/World/part-b": [(2.0, 0.0, 0.0), (3.0, 1.0, 1.0)],
    }
    before_bounds = {
        path: [tuple(corner) for corner in pair]
        for path, pair in bounds.items()
    }
    before_manifest = manifest.to_canonical_payload()

    snapshot = collector.collect_scene_snapshot(
        manifest, make_host(), lambda path: bounds[path]
    )

    assert snapshot.objects[0].provenance["source"] == "fixture"
    assert bounds == before_bounds
    assert manifest.to_canonical_payload() == before_manifest
    assert input_objects == [object_a, object_b]
    assert provenance == {"source": "fixture"}


def test_snapshot_has_only_explicit_metadata_and_host_fields() -> None:
    manifest = make_manifest(
        make_object(
            "part-a",
            42,
            asset_revision_sha256=HASH_B,
            provenance={"source": "explicit", "revision": "r1"},
        )
    )
    host = make_host()
    snapshot = collector.collect_scene_snapshot(
        manifest,
        host,
        lambda _: ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    )
    item = snapshot.objects[0]

    assert item.physical_instance_id == "part-a"
    assert item.brick_id == 42
    assert item.prim_path == "/World/part-a"
    assert item.asset_revision_sha256 == HASH_B
    assert dict(item.provenance) == {"source": "explicit", "revision": "r1"}
    assert item.aabb.object_id == "part-a"
    assert item.aabb.frame_id == FRAME
    assert snapshot.host is host
    assert not hasattr(item, "plan_id")
    assert not hasattr(item, "task_id")
    assert not hasattr(item, "pose")
    assert not hasattr(item, "target")


def test_collector_ast_has_no_forbidden_imports_or_operations() -> None:
    source = inspect.getsource(collector)
    tree = ast.parse(source)
    forbidden_roots = {
        "pathlib",
        "subprocess",
        "socket",
        "pxr",
        "omni",
        "isaac",
        "rclpy",
        "runtime_session",
        "piper_run",
        "tools",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert not imported_roots.intersection(forbidden_roots)
    for token in (
        "json.load",
        "open(",
        "pregrasp",
        "approach",
        "PICK",
        "PLACE",
        "0.1358",
    ):
        assert token not in source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "open"
        for node in ast.walk(tree)
    )



def _forged_manifest(objects: object) -> collector.SceneSnapshotManifest:
    manifest = object.__new__(collector.SceneSnapshotManifest)
    object.__setattr__(manifest, "schema", SCHEMA)
    object.__setattr__(manifest, "scene_id", SCENE_ID)
    object.__setattr__(manifest, "frame_id", FRAME)
    object.__setattr__(manifest, "objects", objects)
    return manifest


def _forged_manifest_object(
    *, provenance: object = MappingProxyType({"source": "fixture"})
) -> collector.SceneObjectManifest:
    item = object.__new__(collector.SceneObjectManifest)
    object.__setattr__(item, "physical_instance_id", "part-a")
    object.__setattr__(item, "brick_id", 1)
    object.__setattr__(item, "prim_path", "/World/part-a")
    object.__setattr__(item, "asset_revision_sha256", HASH_A)
    object.__setattr__(item, "provenance", provenance)
    return item


def _forged_host(**overrides: object) -> RuntimeHostBinding:
    values: dict[str, object] = {
        "host_instance_id": "host-1",
        "ros_domain_id": 73,
        "host_profile_sha256": HASH_A,
        "host_bridge_schema": "rocobrick-host-bridge-smoke/v1",
        "runtime_usd_path": "/assets/piper.usd",
        "runtime_usd_sha256": HASH_B,
        "robot_id": "robot-0",
        "robot_prim_path": "/World/Robot_0",
        "articulation_prim_path": "/World/Robot_0/root_joint",
        "joint_dofs": ("joint1", "joint2"),
        "participant_count": 1,
    }
    values.update(overrides)
    host = object.__new__(RuntimeHostBinding)
    for field_name, value in values.items():
        object.__setattr__(host, field_name, value)
    return host


@pytest.mark.parametrize("objects", [None, [make_object("part-a")], ()])
def test_low_level_manifest_container_forgery_fails_closed(
    objects: object,
) -> None:
    manifest = _forged_manifest(objects)
    calls: list[str] = []

    def reader(path: str) -> object:
        calls.append(path)
        return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(manifest, make_host(), reader)

    assert caught.value.category == "manifest_validation"
    assert calls == []


@pytest.mark.parametrize(
    "item",
    [
        object(),
        object.__new__(collector.SceneObjectManifest),
        _forged_manifest_object(provenance={"source": "mutable"}),
        _forged_manifest_object(provenance=MappingProxyType({})),
    ],
)
def test_low_level_manifest_item_and_provenance_forgery_fails_closed(
    item: object,
) -> None:
    manifest = _forged_manifest((item,))

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(
            manifest,
            make_host(),
            lambda _: ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    assert caught.value.category == "manifest_validation"
    assert "physical_instance_id" in str(caught.value) or (
        caught.value.physical_instance_id == "<manifest>"
    )


def test_low_level_manifest_field_forgery_is_revalidated() -> None:
    manifest = _forged_manifest((make_object("part-a"),))
    object.__setattr__(manifest, "scene_id", None)

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(
            manifest,
            make_host(),
            lambda _: ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    assert caught.value.category == "manifest_validation"


@pytest.mark.parametrize(
    "host",
    [
        object.__new__(RuntimeHostBinding),
        _forged_host(joint_dofs=["joint1", "joint2"]),
        _forged_host(participant_count=2),
        _forged_host(runtime_usd_path="relative.usd"),
    ],
)
def test_low_level_host_forgery_fails_closed_without_reader_call(
    host: RuntimeHostBinding,
) -> None:
    manifest = make_manifest(make_object("part-a"))
    calls: list[str] = []

    def reader(path: str) -> object:
        calls.append(path)
        return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(manifest, host, reader)

    assert caught.value.category == "host_validation"
    assert calls == []


def test_ordinary_reader_exception_is_preserved_as_collection_cause() -> None:
    manifest = make_manifest(make_object("part-a"))
    cause = RuntimeError("reader failed")

    def reader(_: str) -> object:
        raise cause

    with pytest.raises(collector.SceneSnapshotCollectionError) as caught:
        collector.collect_scene_snapshot(manifest, make_host(), reader)

    assert caught.value.category == "reader_error"
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize(
    "control_flow",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_reader_control_flow_baseexceptions_are_not_wrapped(
    control_flow: type[BaseException],
) -> None:
    manifest = make_manifest(make_object("part-a"))
    cause = control_flow("control flow")

    def reader(_: str) -> object:
        raise cause

    with pytest.raises(control_flow) as caught:
        collector.collect_scene_snapshot(manifest, make_host(), reader)

    assert caught.value is cause
