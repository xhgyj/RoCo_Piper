"""Immutable scene identity and planning-scene construction contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from rocobrick.planning_scene import (
    AxisAlignedBox,
    PlanningSceneUpdate,
    scene_digest,
)
from rocobrick.structural_binding import StructuralBinding, canonical_json_digest

SCENE_SNAPSHOT_SCHEMA = "rocobrick-scene-snapshot/v1"
_HOST_BRIDGE_SCHEMA = "rocobrick-host-bridge-smoke/v1"
_BOUND_SCENE_OBJECT_MARKER = object()
_CHECK_NAMES = (
    "snapshot_schema_valid",
    "scene_id_matches",
    "frame_id_matches",
    "runtime_usd_path_matches",
    "runtime_usd_sha256_matches",
    "robot_id_matches",
    "physical_instance_present_once",
    "brick_id_matches",
    "aabb_object_id_matches",
    "aabb_frame_matches",
    "asset_revision_matches",
)


class SceneSnapshotContractError(ValueError):
    """Raised when an immutable scene contract is malformed or inconsistent."""


def _fail(field_name: str, message: str) -> None:
    raise SceneSnapshotContractError(f"{field_name}: {message}")


def _string(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(field_name, "must be a non-empty string without leading/trailing whitespace")
    return value


def _sha256(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        _fail(field_name, "must be a lowercase 64-character SHA-256 hex digest")
    return value


def _integer(value: object, field_name: str) -> int:
    if type(value) is not int:
        _fail(field_name, "must be an integer")
    return value


def _absolute_path(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if not value.startswith("/"):
        _fail(field_name, "must be an absolute path")
    return value


def _sequence(value: object, field_name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(field_name, "must be a sequence")
    return tuple(value)


def _string_mapping(value: object, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        _fail(field_name, "must be a non-empty mapping")
    copied = dict(value)
    for key, item in copied.items():
        _string(key, f"{field_name} key")
        _string(item, f"{field_name}[{key}] value")
    return MappingProxyType(copied)


def _aabb_payload(aabb: AxisAlignedBox) -> dict[str, object]:
    return {
        "object_id": aabb.object_id,
        "frame_id": aabb.frame_id,
        "minimum_m": list(aabb.minimum_m),
        "maximum_m": list(aabb.maximum_m),
        "geometry_digest": aabb.geometry_digest,
    }


def _object_payload(scene_object: SceneObjectSnapshot) -> dict[str, object]:
    return {
        "physical_instance_id": scene_object.physical_instance_id,
        "brick_id": scene_object.brick_id,
        "prim_path": scene_object.prim_path,
        "aabb": _aabb_payload(scene_object.aabb),
        "asset_revision_sha256": scene_object.asset_revision_sha256,
        "provenance": dict(scene_object.provenance),
    }


def _host_payload(host: RuntimeHostBinding) -> dict[str, object]:
    return {
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
    }


def _snapshot_payload(
    *,
    schema: str,
    scene_id: str,
    frame_id: str,
    host: RuntimeHostBinding,
    objects: Sequence[SceneObjectSnapshot],
) -> dict[str, object]:
    return {
        "schema": schema,
        "scene_id": scene_id,
        "frame_id": frame_id,
        "host": _host_payload(host),
        "objects": [_object_payload(scene_object) for scene_object in objects],
    }


@dataclass(frozen=True)
class SceneObjectSnapshot:
    """One immutable physical object record in a scene snapshot."""

    physical_instance_id: str
    brick_id: int
    prim_path: str
    aabb: AxisAlignedBox
    asset_revision_sha256: str
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        physical_instance_id = _string(
            self.physical_instance_id, "physical_instance_id"
        )
        object.__setattr__(self, "physical_instance_id", physical_instance_id)

        brick_id = _integer(self.brick_id, "brick_id")
        if brick_id < 0:
            _fail("brick_id", "must be non-negative")
        object.__setattr__(self, "brick_id", brick_id)

        object.__setattr__(
            self, "prim_path", _absolute_path(self.prim_path, "prim_path")
        )

        if type(self.aabb) is not AxisAlignedBox:
            _fail("aabb", "must be exactly an AxisAlignedBox")
        if self.aabb.object_id != physical_instance_id:
            _fail("aabb.object_id", "must equal physical_instance_id")

        object.__setattr__(
            self,
            "asset_revision_sha256",
            _sha256(self.asset_revision_sha256, "asset_revision_sha256"),
        )
        object.__setattr__(
            self,
            "provenance",
            _string_mapping(self.provenance, "provenance"),
        )


@dataclass(frozen=True)
class RuntimeHostBinding:
    """Immutable host and asset identity required by a scene snapshot."""

    host_instance_id: str
    ros_domain_id: int
    host_profile_sha256: str
    host_bridge_schema: str
    runtime_usd_path: str
    runtime_usd_sha256: str
    robot_id: str
    robot_prim_path: str
    articulation_prim_path: str
    joint_dofs: tuple[str, ...]
    participant_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "host_instance_id", _string(self.host_instance_id, "host_instance_id")
        )

        ros_domain_id = _integer(self.ros_domain_id, "ros_domain_id")
        if not 0 <= ros_domain_id <= 232:
            _fail("ros_domain_id", "must be between 0 and 232 inclusive")
        object.__setattr__(self, "ros_domain_id", ros_domain_id)

        object.__setattr__(
            self,
            "host_profile_sha256",
            _sha256(self.host_profile_sha256, "host_profile_sha256"),
        )

        host_bridge_schema = _string(self.host_bridge_schema, "host_bridge_schema")
        if host_bridge_schema != _HOST_BRIDGE_SCHEMA:
            _fail("host_bridge_schema", f"must equal {_HOST_BRIDGE_SCHEMA}")
        object.__setattr__(self, "host_bridge_schema", host_bridge_schema)

        object.__setattr__(
            self,
            "runtime_usd_path",
            _absolute_path(self.runtime_usd_path, "runtime_usd_path"),
        )
        object.__setattr__(
            self,
            "runtime_usd_sha256",
            _sha256(self.runtime_usd_sha256, "runtime_usd_sha256"),
        )
        object.__setattr__(self, "robot_id", _string(self.robot_id, "robot_id"))
        object.__setattr__(
            self,
            "robot_prim_path",
            _absolute_path(self.robot_prim_path, "robot_prim_path"),
        )
        object.__setattr__(
            self,
            "articulation_prim_path",
            _absolute_path(self.articulation_prim_path, "articulation_prim_path"),
        )

        joint_dofs = _sequence(self.joint_dofs, "joint_dofs")
        if not joint_dofs:
            _fail("joint_dofs", "must be non-empty")
        normalized_dofs = tuple(
            _string(dof, f"joint_dofs[{index}]")
            for index, dof in enumerate(joint_dofs)
        )
        if len(set(normalized_dofs)) != len(normalized_dofs):
            _fail("joint_dofs", "must not contain duplicates")
        object.__setattr__(self, "joint_dofs", normalized_dofs)

        participant_count = _integer(self.participant_count, "participant_count")
        if participant_count != 1:
            _fail("participant_count", "must equal 1")
        object.__setattr__(self, "participant_count", participant_count)


@dataclass(frozen=True)
class SceneSnapshot:
    """Immutable, content-addressed scene snapshot."""

    schema: str
    scene_id: str
    frame_id: str
    host: RuntimeHostBinding
    objects: tuple[SceneObjectSnapshot, ...]
    snapshot_sha256: str

    def __post_init__(self) -> None:
        schema = _string(self.schema, "schema")
        if schema != SCENE_SNAPSHOT_SCHEMA:
            _fail("schema", f"must equal {SCENE_SNAPSHOT_SCHEMA}")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "scene_id", _string(self.scene_id, "scene_id"))
        object.__setattr__(self, "frame_id", _string(self.frame_id, "frame_id"))

        if type(self.host) is not RuntimeHostBinding:
            _fail("host", "must be exactly a RuntimeHostBinding")

        objects = _sequence(self.objects, "objects")
        if not objects:
            _fail("objects", "must be non-empty")
        for index, scene_object in enumerate(objects):
            if type(scene_object) is not SceneObjectSnapshot:
                _fail(f"objects[{index}]", "must be exactly a SceneObjectSnapshot")
        normalized_objects = tuple(objects)
        object.__setattr__(self, "objects", normalized_objects)

        physical_ids = tuple(
            scene_object.physical_instance_id for scene_object in normalized_objects
        )
        if len(set(physical_ids)) != len(physical_ids):
            _fail("objects", "physical_instance_id values must be unique")
        if any(left >= right for left, right in zip(physical_ids, physical_ids[1:])):
            _fail("objects", "must already be strictly ordered by physical_instance_id")

        prim_paths = tuple(scene_object.prim_path for scene_object in normalized_objects)
        if len(set(prim_paths)) != len(prim_paths):
            _fail("objects", "prim_path values must be unique")
        for index, scene_object in enumerate(normalized_objects):
            if scene_object.aabb.frame_id != self.frame_id:
                _fail(f"objects[{index}].aabb.frame_id", "must equal frame_id")

        snapshot_sha256 = _sha256(self.snapshot_sha256, "snapshot_sha256")
        expected = canonical_json_digest(
            _snapshot_payload(
                schema=self.schema,
                scene_id=self.scene_id,
                frame_id=self.frame_id,
                host=self.host,
                objects=self.objects,
            )
        )
        if snapshot_sha256 != expected:
            _fail("snapshot_sha256", "does not match the canonical snapshot payload")
        object.__setattr__(self, "snapshot_sha256", snapshot_sha256)

    def to_canonical_payload(self) -> dict[str, object]:
        """Return a fresh JSON-compatible payload excluding the snapshot hash."""
        return _snapshot_payload(
            schema=self.schema,
            scene_id=self.scene_id,
            frame_id=self.frame_id,
            host=self.host,
            objects=self.objects,
        )


@dataclass(frozen=True, init=False)
class BoundSceneObject:
    """An object accepted by every structural scene-binding check."""

    binding: StructuralBinding
    snapshot: SceneSnapshot
    object: SceneObjectSnapshot
    checks: Mapping[str, bool]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("BoundSceneObject instances are created by bind_scene_object")

    def __post_init__(self) -> None:
        if type(self.binding) is not StructuralBinding:
            _fail("binding", "must be exactly a StructuralBinding")
        if type(self.snapshot) is not SceneSnapshot:
            _fail("snapshot", "must be exactly a SceneSnapshot")
        if type(self.object) is not SceneObjectSnapshot:
            _fail("object", "must be exactly a SceneObjectSnapshot")
        if not isinstance(self.checks, Mapping):
            _fail("checks", "must be a mapping")
        checks = dict(self.checks)
        if tuple(checks) != _CHECK_NAMES:
            _fail("checks", "keys must use the required exact order")
        for name, value in checks.items():
            if type(value) is not bool:
                _fail(f"checks[{name}]", "must be a boolean")
        object.__setattr__(self, "checks", MappingProxyType(checks))

    @property
    def allowed(self) -> bool:
        """Whether every required binding check passed."""
        return all(self.checks.values())



def _make_bound_scene_object(
    *,
    binding: StructuralBinding,
    snapshot: SceneSnapshot,
    scene_object: SceneObjectSnapshot,
    checks: Mapping[str, bool],
) -> BoundSceneObject:
    bound = object.__new__(BoundSceneObject)
    object.__setattr__(bound, "binding", binding)
    object.__setattr__(bound, "snapshot", snapshot)
    object.__setattr__(bound, "object", scene_object)
    object.__setattr__(bound, "checks", checks)
    object.__setattr__(bound, "_bound_scene_object_marker", _BOUND_SCENE_OBJECT_MARKER)
    BoundSceneObject.__post_init__(bound)
    object.__setattr__(
        bound, "_bound_scene_object_seal", (binding, snapshot, scene_object, bound.checks, scene_object.aabb)
    )
    return bound


def bind_scene_object(
    binding: StructuralBinding,
    snapshot: SceneSnapshot,
    *,
    expected_scene_id: str,
    expected_frame_id: str,
    expected_runtime_usd_path: str,
    expected_runtime_usd_sha256: str,
    expected_asset_revision_sha256: str,
) -> BoundSceneObject:
    """Validate one structural binding against one immutable snapshot.

    Raises:
        SceneSnapshotContractError: If an input type, expected value, or
            ordered binding check is invalid.
    """
    if type(binding) is not StructuralBinding:
        _fail("binding", "must be exactly a StructuralBinding")
    if type(snapshot) is not SceneSnapshot:
        _fail("snapshot", "must be exactly a SceneSnapshot")

    expected_scene_id = _string(expected_scene_id, "expected_scene_id")
    expected_frame_id = _string(expected_frame_id, "expected_frame_id")
    expected_runtime_usd_path = _absolute_path(
        expected_runtime_usd_path, "expected_runtime_usd_path"
    )
    expected_runtime_usd_sha256 = _sha256(
        expected_runtime_usd_sha256, "expected_runtime_usd_sha256"
    )
    expected_asset_revision_sha256 = _sha256(
        expected_asset_revision_sha256, "expected_asset_revision_sha256"
    )

    matching_objects = tuple(
        scene_object
        for scene_object in snapshot.objects
        if scene_object.physical_instance_id == binding.physical_instance_id
    )
    physical_instance_present_once = len(matching_objects) == 1
    matched_object = matching_objects[0] if physical_instance_present_once else None

    checks = {
        "snapshot_schema_valid": snapshot.schema == SCENE_SNAPSHOT_SCHEMA,
        "scene_id_matches": snapshot.scene_id == expected_scene_id,
        "frame_id_matches": snapshot.frame_id == expected_frame_id,
        "runtime_usd_path_matches": (
            snapshot.host.runtime_usd_path == expected_runtime_usd_path
        ),
        "runtime_usd_sha256_matches": (
            snapshot.host.runtime_usd_sha256 == expected_runtime_usd_sha256
        ),
        "robot_id_matches": snapshot.host.robot_id == binding.robot_id,
        "physical_instance_present_once": physical_instance_present_once,
        "brick_id_matches": (
            matched_object is not None and matched_object.brick_id == binding.brick_id
        ),
        "aabb_object_id_matches": (
            matched_object is not None
            and matched_object.aabb.object_id == binding.physical_instance_id
        ),
        "aabb_frame_matches": (
            matched_object is not None
            and matched_object.aabb.frame_id == snapshot.frame_id
        ),
        "asset_revision_matches": (
            matched_object is not None
            and matched_object.asset_revision_sha256 == expected_asset_revision_sha256
        ),
    }
    ordered_checks = {name: checks[name] for name in _CHECK_NAMES}
    failures = [name for name in _CHECK_NAMES if not ordered_checks[name]]
    if failures:
        _fail("bind_scene_object", "failed checks: " + ", ".join(failures))
    return _make_bound_scene_object(
        binding=binding,
        snapshot=snapshot,
        scene_object=matched_object,
        checks=ordered_checks,
    )


def _validate_bound_scene_object(item: object, index: int) -> None:
    if type(item) is not BoundSceneObject:
        _fail(f"bound_objects[{index}]", "must be exactly a BoundSceneObject")
    if getattr(item, "_bound_scene_object_marker", None) is not _BOUND_SCENE_OBJECT_MARKER:
        _fail(f"bound_objects[{index}]", "was not created by bind_scene_object")

    binding = getattr(item, "binding", None)
    snapshot = getattr(item, "snapshot", None)
    scene_object = getattr(item, "object", None)
    checks = getattr(item, "checks", None)
    seal = getattr(item, "_bound_scene_object_seal", None)
    if type(checks) is not MappingProxyType:
        _fail(f"bound_objects[{index}].checks", "must be an immutable mapping")
    if (
        type(seal) is not tuple
        or len(seal) != 5
        or seal[0] is not binding
        or seal[1] is not snapshot
        or seal[2] is not scene_object
        or seal[3] is not checks
        or seal[4] is not getattr(scene_object, "aabb", None)
    ):
        _fail(f"bound_objects[{index}]", "integrity seal is invalid")
    if type(binding) is not StructuralBinding:
        _fail(f"bound_objects[{index}].binding", "must be exactly a StructuralBinding")
    if type(snapshot) is not SceneSnapshot:
        _fail(f"bound_objects[{index}].snapshot", "must be exactly a SceneSnapshot")
    if getattr(snapshot, "schema", None) != SCENE_SNAPSHOT_SCHEMA:
        _fail(f"bound_objects[{index}].snapshot.schema", "is invalid")
    host = getattr(snapshot, "host", None)
    if type(host) is not RuntimeHostBinding:
        _fail(f"bound_objects[{index}].snapshot.host", "has an invalid type")
    snapshot_objects = getattr(snapshot, "objects", None)
    if type(snapshot_objects) is not tuple or any(
        type(candidate) is not SceneObjectSnapshot for candidate in snapshot_objects
    ):
        _fail(f"bound_objects[{index}].snapshot.objects", "has invalid types")
    if type(scene_object) is not SceneObjectSnapshot:
        _fail(f"bound_objects[{index}].object", "must be exactly a SceneObjectSnapshot")
    aabb = getattr(scene_object, "aabb", None)
    if type(aabb) is not AxisAlignedBox:
        _fail(f"bound_objects[{index}].object.aabb", "must be exactly an AxisAlignedBox")
    if not isinstance(checks, Mapping):
        _fail(f"bound_objects[{index}].checks", "must be a mapping")
    if tuple(checks) != _CHECK_NAMES:
        _fail(f"bound_objects[{index}].checks", "keys have the wrong order")
    if any(type(value) is not bool or not value for value in checks.values()):
        _fail(f"bound_objects[{index}].checks", "all checks must be true booleans")

    physical_instance_id = getattr(scene_object, "physical_instance_id", None)
    identity_matches = tuple(
        candidate for candidate in snapshot_objects if candidate is scene_object
    )
    id_matches = tuple(
        candidate
        for candidate in snapshot_objects
        if getattr(candidate, "physical_instance_id", None) == physical_instance_id
    )
    if len(identity_matches) != 1 or len(id_matches) != 1:
        _fail(f"bound_objects[{index}].object", "must occur exactly once by identity and ID")
    if getattr(binding, "physical_instance_id", None) != physical_instance_id:
        _fail(f"bound_objects[{index}].binding", "physical instance does not match object")
    if getattr(binding, "brick_id", None) != getattr(scene_object, "brick_id", None):
        _fail(f"bound_objects[{index}].binding", "brick ID does not match object")
    if getattr(binding, "robot_id", None) != getattr(host, "robot_id", None):
        _fail(f"bound_objects[{index}].binding", "robot ID does not match snapshot host")
    if getattr(identity_matches[0], "aabb", None) is not aabb:
        _fail(f"bound_objects[{index}].object.aabb", "must preserve AABB identity")
    if getattr(aabb, "object_id", None) != physical_instance_id:
        _fail(f"bound_objects[{index}].object.aabb", "object ID does not match")
    if getattr(aabb, "frame_id", None) != getattr(snapshot, "frame_id", None):
        _fail(f"bound_objects[{index}].object.aabb", "frame does not match")

def build_planning_scene_update(
    bound_objects: Sequence[BoundSceneObject],
    *,
    allowed_collision_pairs: Sequence[tuple[str, str]] = (),
) -> PlanningSceneUpdate:
    """Build a deterministic scene update with explicit contact allowances."""
    items = _sequence(bound_objects, "bound_objects")
    if not items:
        _fail("bound_objects", "must be non-empty")
    for index, item in enumerate(items):
        _validate_bound_scene_object(item, index)

    snapshot = items[0].snapshot
    if any(item.snapshot is not snapshot for item in items[1:]):
        _fail("bound_objects", "all items must share the same snapshot identity")

    physical_ids = tuple(item.object.physical_instance_id for item in items)
    if len(set(physical_ids)) != len(physical_ids):
        _fail("bound_objects", "physical_instance_id values must be unique")
    if any(left >= right for left, right in zip(physical_ids, physical_ids[1:])):
        _fail("bound_objects", "must already be strictly ordered by physical_instance_id")

    boxes = tuple(item.object.aabb for item in items)
    canonical_pairs = tuple(
        sorted((str(first), str(second)) for first, second in allowed_collision_pairs)
    )
    placeholder = PlanningSceneUpdate(
        scene_id=snapshot.scene_id,
        boxes=boxes,
        expected_scene_digest="0" * 64,
        allowed_collision_pairs=canonical_pairs,
    )
    expected_scene_digest = scene_digest(placeholder)
    return PlanningSceneUpdate(
        scene_id=snapshot.scene_id,
        boxes=boxes,
        expected_scene_digest=expected_scene_digest,
        allowed_collision_pairs=canonical_pairs,
    )


__all__ = [
    "SCENE_SNAPSHOT_SCHEMA",
    "SceneSnapshotContractError",
    "SceneObjectSnapshot",
    "RuntimeHostBinding",
    "SceneSnapshot",
    "BoundSceneObject",
    "bind_scene_object",
    "build_planning_scene_update",
]
