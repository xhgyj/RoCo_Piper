from __future__ import annotations

import ast
import dataclasses
import inspect
from types import MappingProxyType

import pytest

from rocobrick.planning_scene import AxisAlignedBox, box_from_aabb
from rocobrick.structural_binding import StructuralBinding, canonical_json_digest
from rocobrick import scene_snapshot
from rocobrick.scene_snapshot import (
    SCENE_SNAPSHOT_SCHEMA,
    BoundSceneObject,
    RuntimeHostBinding,
    SceneObjectSnapshot,
    SceneSnapshot,
    SceneSnapshotContractError,
    bind_scene_object,
    build_planning_scene_update,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
FRAME = "world"
USD_PATH = "/assets/piper.usd"
ROBOT_ID = "robot-0"
SCENE_ID = "scene-1"
DOFS = ("joint1", "joint2")


def make_box(identifier: str, frame: str = FRAME, offset: float = 0.0) -> AxisAlignedBox:
    return box_from_aabb(
        object_id=identifier,
        frame_id=frame,
        minimum_m=(offset, 0.0, 0.0),
        maximum_m=(offset + 1.0, 1.0, 1.0),
    )


def make_object(
    identifier: str,
    brick_id: int = 1,
    *,
    frame: str = FRAME,
    prim_path: str | None = None,
    asset_revision_sha256: str = HASH_A,
    provenance: dict[str, str] | None = None,
) -> SceneObjectSnapshot:
    return SceneObjectSnapshot(
        physical_instance_id=identifier,
        brick_id=brick_id,
        prim_path=prim_path or f"/World/{identifier}",
        aabb=make_box(identifier, frame=frame, offset=float(brick_id)),
        asset_revision_sha256=asset_revision_sha256,
        provenance={"source": "fixture"} if provenance is None else provenance,
    )


def make_host(**overrides: object) -> RuntimeHostBinding:
    values: dict[str, object] = {
        "host_instance_id": "host-1",
        "ros_domain_id": 73,
        "host_profile_sha256": HASH_A,
        "host_bridge_schema": "rocobrick-host-bridge-smoke/v1",
        "runtime_usd_path": USD_PATH,
        "runtime_usd_sha256": HASH_B,
        "robot_id": ROBOT_ID,
        "robot_prim_path": "/World/Robot_0",
        "articulation_prim_path": "/World/Robot_0/root_joint",
        "joint_dofs": DOFS,
        "participant_count": 1,
    }
    values.update(overrides)
    return RuntimeHostBinding(**values)


def payload_for(
    host: RuntimeHostBinding,
    objects: tuple[SceneObjectSnapshot, ...],
    *,
    schema: str = SCENE_SNAPSHOT_SCHEMA,
    scene_id: str = SCENE_ID,
    frame_id: str = FRAME,
) -> dict[str, object]:
    return {
        "schema": schema,
        "scene_id": scene_id,
        "frame_id": frame_id,
        "host": {
            "host_instance_id": host.host_instance_id,
            "ros_domain_id": host.ros_domain_id,
            "host_profile_sha256": host.host_profile_sha256,
            "host_bridge_schema": host.host_bridge_schema,
            "runtime_usd_path": host.runtime_usd_path,
            "runtime_usd_sha256": host.runtime_usd_sha256,
            "robot_id": host.robot_id,
            "robot_prim_path": host.robot_prim_path,
            "articulation_prim_path": host.articulation_prim_path,
            "joint_dofs": list(host.joint_dofs),
            "participant_count": host.participant_count,
        },
        "objects": [
            {
                "physical_instance_id": item.physical_instance_id,
                "brick_id": item.brick_id,
                "prim_path": item.prim_path,
                "aabb": {
                    "object_id": item.aabb.object_id,
                    "frame_id": item.aabb.frame_id,
                    "minimum_m": list(item.aabb.minimum_m),
                    "maximum_m": list(item.aabb.maximum_m),
                    "geometry_digest": item.aabb.geometry_digest,
                },
                "asset_revision_sha256": item.asset_revision_sha256,
                "provenance": dict(item.provenance),
            }
            for item in objects
        ],
    }


def make_snapshot(
    objects: tuple[SceneObjectSnapshot, ...] | None = None,
    *,
    host: RuntimeHostBinding | None = None,
    schema: str = SCENE_SNAPSHOT_SCHEMA,
    scene_id: str = SCENE_ID,
    frame_id: str = FRAME,
    snapshot_sha256: str | None = None,
) -> SceneSnapshot:
    host = make_host() if host is None else host
    objects = (
        (make_object("part-a", 1), make_object("part-b", 2))
        if objects is None
        else objects
    )
    payload = payload_for(
        host, objects, schema=schema, scene_id=scene_id, frame_id=frame_id
    )
    digest = canonical_json_digest(payload)
    return SceneSnapshot(
        schema=schema,
        scene_id=scene_id,
        frame_id=frame_id,
        host=host,
        objects=list(objects),
        snapshot_sha256=digest if snapshot_sha256 is None else snapshot_sha256,
    )


def make_binding(identifier: str, brick_id: int = 1, *, robot_id: str = ROBOT_ID):
    return StructuralBinding(
        plan_id="plan-1",
        task_id=f"task-{identifier}",
        physical_instance_id=identifier,
        brick_id=brick_id,
        robot_id=robot_id,
        source_goal_key="goal-1",
        grid_xyz=(0, 0, 0),
        orientation=0,
        color="red",
        motion_geometry="omitted",
        plan_sha256=HASH_A,
        task_start_sha256=HASH_B,
        task_goal_sha256=HASH_A,
    )


def bind_args(snapshot: SceneSnapshot) -> dict[str, str]:
    return {
        "expected_scene_id": snapshot.scene_id,
        "expected_frame_id": snapshot.frame_id,
        "expected_runtime_usd_path": snapshot.host.runtime_usd_path,
        "expected_runtime_usd_sha256": snapshot.host.runtime_usd_sha256,
        "expected_asset_revision_sha256": HASH_A,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("physical_instance_id", ""),
        ("physical_instance_id", " part-a"),
        ("brick_id", True),
        ("brick_id", -1),
        ("prim_path", "World/part-a"),
        ("aabb", object()),
        ("asset_revision_sha256", "A" * 64),
        ("asset_revision_sha256", "short"),
        ("provenance", {}),
        ("provenance", {"": "value"}),
        ("provenance", {"key": " "}),
        ("provenance", {1: "value"}),
    ],
)
def test_scene_object_rejects_each_invalid_field(field: str, value: object) -> None:
    values: dict[str, object] = {
        "physical_instance_id": "part-a",
        "brick_id": 1,
        "prim_path": "/World/part-a",
        "aabb": make_box("part-a"),
        "asset_revision_sha256": HASH_A,
        "provenance": {"source": "fixture"},
    }
    values[field] = value
    with pytest.raises(SceneSnapshotContractError, match=field.split(".")[0]):
        SceneObjectSnapshot(**values)


def test_scene_object_requires_matching_aabb_object_id() -> None:
    with pytest.raises(SceneSnapshotContractError, match="aabb.object_id"):
        SceneObjectSnapshot(
            "part-a", 1, "/World/part-a", make_box("part-b"), HASH_A, {"source": "x"}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host_instance_id", ""),
        ("host_instance_id", " host"),
        ("ros_domain_id", -1),
        ("ros_domain_id", 233),
        ("ros_domain_id", True),
        ("host_profile_sha256", "bad"),
        ("host_bridge_schema", "other/v1"),
        ("runtime_usd_path", "relative.usd"),
        ("runtime_usd_sha256", "bad"),
        ("robot_id", " "),
        ("robot_prim_path", "World/Robot"),
        ("articulation_prim_path", "root_joint"),
        ("joint_dofs", []),
        ("joint_dofs", ["joint1", "joint1"]),
        ("joint_dofs", ["joint1", " "]),
        ("participant_count", 0),
        ("participant_count", 2),
        ("participant_count", True),
    ],
)
def test_host_rejects_each_invalid_field(field: str, value: object) -> None:
    with pytest.raises(SceneSnapshotContractError, match=field.split(".")[0]):
        make_host(**{field: value})


def test_host_normalizes_sequence_and_rejects_mutation() -> None:

    dofs = ["joint1", "joint2"]
    host = make_host(joint_dofs=dofs)
    dofs.append("joint3")
    assert host.joint_dofs == ("joint1", "joint2")
    assert isinstance(host, RuntimeHostBinding)
    with pytest.raises(dataclasses.FrozenInstanceError):
        host.robot_id = "other"  # type: ignore[misc]


def test_scene_snapshot_rejects_invalid_top_level_and_object_structure() -> None:
    objects = (make_object("part-a", 1), make_object("part-b", 2))
    with pytest.raises(SceneSnapshotContractError, match="schema"):
        make_snapshot(objects, schema="other/v1")
    with pytest.raises(SceneSnapshotContractError, match="scene_id"):
        make_snapshot(objects, scene_id=" ")
    with pytest.raises(SceneSnapshotContractError, match="frame_id"):
        make_snapshot(objects, frame_id=" ")
    with pytest.raises(SceneSnapshotContractError, match="host"):
        SceneSnapshot(
            SCENE_SNAPSHOT_SCHEMA, SCENE_ID, FRAME, object(), objects, HASH_A
        )
    with pytest.raises(SceneSnapshotContractError, match="non-empty"):
        SceneSnapshot(SCENE_SNAPSHOT_SCHEMA, SCENE_ID, FRAME, make_host(), (), HASH_A)
    with pytest.raises(SceneSnapshotContractError, match="ordered"):
        make_snapshot((objects[1], objects[0]))
    with pytest.raises(SceneSnapshotContractError, match="unique"):
        make_snapshot((objects[0], make_object("part-a", 9, prim_path="/World/other")))
    with pytest.raises(SceneSnapshotContractError, match="prim_path"):
        make_snapshot(
            (objects[0], make_object("part-b", 2, prim_path=objects[0].prim_path))
        )
    wrong_frame = (make_object("part-a", 1, frame="other"), make_object("part-b", 2))
    with pytest.raises(SceneSnapshotContractError, match="aabb.frame_id"):
        make_snapshot(wrong_frame)
    with pytest.raises(SceneSnapshotContractError, match="snapshot_sha256"):
        make_snapshot(objects, snapshot_sha256=HASH_B)


def test_snapshot_is_frozen_and_defensively_copies_sequences_and_mappings() -> None:
    provenance = {"source": "fixture"}
    object_a = make_object("part-a", provenance=provenance)
    objects = [object_a, make_object("part-b", 2)]
    snapshot = make_snapshot(tuple(objects))
    provenance["source"] = "changed"
    objects.clear()
    assert object_a.provenance["source"] == "fixture"
    assert snapshot.objects == (object_a, snapshot.objects[1])
    assert isinstance(object_a.provenance, MappingProxyType)
    with pytest.raises(TypeError):
        object_a.provenance["new"] = "value"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.scene_id = "other"  # type: ignore[misc]


def test_canonical_payload_is_fresh_and_hash_covers_every_field() -> None:
    snapshot = make_snapshot()
    payload = snapshot.to_canonical_payload()
    assert payload["objects"][0]["aabb"]["minimum_m"] == [1.0, 0.0, 0.0]
    assert snapshot.snapshot_sha256 == canonical_json_digest(payload)
    copied = snapshot.to_canonical_payload()
    copied["host"]["joint_dofs"].append("joint3")
    copied["objects"][0]["provenance"]["extra"] = "value"
    assert snapshot.to_canonical_payload() == payload

    mutations = [
        ("schema", lambda value: value.__setitem__("schema", "other/v1")),
        ("scene_id", lambda value: value.__setitem__("scene_id", "other")),
        ("frame_id", lambda value: value.__setitem__("frame_id", "other")),
        (
            "host_instance_id",
            lambda value: value["host"].__setitem__("host_instance_id", "other"),
        ),
        (
            "ros_domain_id",
            lambda value: value["host"].__setitem__("ros_domain_id", 74),
        ),
        (
            "host_profile_sha256",
            lambda value: value["host"].__setitem__("host_profile_sha256", HASH_B),
        ),
        (
            "host_bridge_schema",
            lambda value: value["host"].__setitem__(
                "host_bridge_schema", "other/v1"
            ),
        ),
        (
            "runtime_usd_path",
            lambda value: value["host"].__setitem__("runtime_usd_path", "/other.usd"),
        ),
        (
            "runtime_usd_sha256",
            lambda value: value["host"].__setitem__("runtime_usd_sha256", HASH_A),
        ),
        ("robot_id", lambda value: value["host"].__setitem__("robot_id", "other")),
        (
            "robot_prim_path",
            lambda value: value["host"].__setitem__("robot_prim_path", "/Other"),
        ),
        (
            "articulation_prim_path",
            lambda value: value["host"].__setitem__(
                "articulation_prim_path", "/Other/root"
            ),
        ),
        (
            "joint_dofs",
            lambda value: value["host"]["joint_dofs"].append("joint3"),
        ),
        (
            "participant_count",
            lambda value: value["host"].__setitem__("participant_count", 2),
        ),
        (
            "physical_instance_id",
            lambda value: value["objects"][0].__setitem__(
                "physical_instance_id", "other"
            ),
        ),
        ("brick_id", lambda value: value["objects"][0].__setitem__("brick_id", 9)),
        (
            "prim_path",
            lambda value: value["objects"][0].__setitem__("prim_path", "/Other"),
        ),
        (
            "aabb_min",
            lambda value: value["objects"][0]["aabb"]["minimum_m"].__setitem__(0, 9.0),
        ),
        (
            "aabb_max",
            lambda value: value["objects"][0]["aabb"]["maximum_m"].__setitem__(0, 9.0),
        ),
        (
            "aabb_geometry_digest",
            lambda value: value["objects"][0]["aabb"].__setitem__(
                "geometry_digest", HASH_B
            ),
        ),
        (
            "asset_revision_sha256",
            lambda value: value["objects"][0].__setitem__(
                "asset_revision_sha256", HASH_B
            ),
        ),
        (
            "provenance",
            lambda value: value["objects"][0]["provenance"].__setitem__(
                "new", "value"
            ),
        ),
    ]
    for _name, mutate in mutations:
        changed = snapshot.to_canonical_payload()
        mutate(changed)
        assert canonical_json_digest(changed) != snapshot.snapshot_sha256



def _make_valid_bound_for_low_level_tests() -> tuple[SceneSnapshot, BoundSceneObject]:
    snapshot = make_snapshot()
    bound = bind_scene_object(
        make_binding("part-a", 1), snapshot, **bind_args(snapshot)
    )
    return snapshot, bound


def _forge_without_factory(bound: BoundSceneObject) -> BoundSceneObject:
    forged = object.__new__(BoundSceneObject)
    object.__setattr__(forged, "binding", bound.binding)
    object.__setattr__(forged, "snapshot", bound.snapshot)
    object.__setattr__(forged, "object", bound.object)
    object.__setattr__(forged, "checks", dict(bound.checks))
    return forged


def test_bound_scene_object_public_construction_is_disabled() -> None:
    with pytest.raises(TypeError):
        BoundSceneObject()
    with pytest.raises(TypeError):
        BoundSceneObject(None, None, None, {})


def test_build_rejects_low_level_all_true_forgery() -> None:
    _snapshot, bound = _make_valid_bound_for_low_level_tests()
    forged = _forge_without_factory(bound)
    assert forged.checks == dict(bound.checks)
    with pytest.raises(SceneSnapshotContractError):
        build_planning_scene_update([forged])


@pytest.mark.parametrize(
    ("field", "value"),
    [("brick_id", 99), ("robot_id", "other-robot")],
)
def test_build_rechecks_binding_identity_fields(field: str, value: object) -> None:
    _snapshot, bound = _make_valid_bound_for_low_level_tests()
    original = getattr(bound.binding, field)
    object.__setattr__(bound.binding, field, value)
    try:
        with pytest.raises(SceneSnapshotContractError):
            build_planning_scene_update([bound])
    finally:
        object.__setattr__(bound.binding, field, original)


def test_build_rejects_foreign_object_and_snapshot() -> None:
    snapshot, bound = _make_valid_bound_for_low_level_tests()
    foreign_snapshot = make_snapshot()
    original_object = bound.object
    object.__setattr__(bound, "object", foreign_snapshot.objects[0])
    try:
        with pytest.raises(SceneSnapshotContractError):
            build_planning_scene_update([bound])
    finally:
        object.__setattr__(bound, "object", original_object)

    original_snapshot = bound.snapshot
    object.__setattr__(bound, "snapshot", foreign_snapshot)
    try:
        with pytest.raises(SceneSnapshotContractError):
            build_planning_scene_update([bound])
    finally:
        object.__setattr__(bound, "snapshot", original_snapshot)
    assert snapshot is bound.snapshot


def test_build_rejects_forged_checks_even_when_all_true() -> None:
    _snapshot, bound = _make_valid_bound_for_low_level_tests()
    original_checks = bound.checks
    object.__setattr__(bound, "checks", MappingProxyType(dict(original_checks)))
    try:
        assert all(bound.checks.values())
        with pytest.raises(SceneSnapshotContractError):
            build_planning_scene_update([bound])
    finally:
        object.__setattr__(bound, "checks", original_checks)


def test_build_rechecks_snapshot_schema_and_aabb_contract() -> None:
    snapshot, bound = _make_valid_bound_for_low_level_tests()
    object.__setattr__(snapshot, "schema", "other/v1")
    try:
        with pytest.raises(SceneSnapshotContractError):
            build_planning_scene_update([bound])
    finally:
        object.__setattr__(snapshot, "schema", SCENE_SNAPSHOT_SCHEMA)

    original_aabb = bound.object.aabb
    object.__setattr__(bound.object, "aabb", make_box("part-a", frame="other"))
    try:
        with pytest.raises(SceneSnapshotContractError):
            build_planning_scene_update([bound])
    finally:
        object.__setattr__(bound.object, "aabb", original_aabb)

