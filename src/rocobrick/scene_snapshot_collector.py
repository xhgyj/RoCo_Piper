"""Pure host-side collection seam for immutable scene snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from rocobrick import planning_scene
from rocobrick.scene_snapshot import (
    RuntimeHostBinding,
    SceneObjectSnapshot,
    SceneSnapshot,
    SCENE_SNAPSHOT_SCHEMA,
)
from rocobrick.structural_binding import canonical_json_digest

SCENE_SNAPSHOT_MANIFEST_SCHEMA: str = "rocobrick-scene-snapshot-manifest/v1"


class SceneSnapshotCollectionError(ValueError):
    """A deterministic failure while collecting one scene snapshot."""

    def __init__(self, physical_instance_id: str, category: str) -> None:
        self.physical_instance_id = physical_instance_id
        self.category = category
        self.failure_category = category
        super().__init__(
            f"{category}: physical_instance_id={physical_instance_id}"
        )


class WorldBoundsReader(Protocol):
    """Callable seam returning one prim's world-space minimum and maximum."""

    def __call__(self, prim_path: str) -> Sequence[object]:
        """Return exactly (minimum, maximum) for prim_path."""
        ...


def _string(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without "
            "leading/trailing whitespace"
        )
    return value


def _integer(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


def _sha256(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 hex digest"
        )
    return value


def _absolute_path(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if not value.startswith("/"):
        raise ValueError(f"{field_name} must be an absolute path")
    return value


def _sequence(value: object, field_name: str) -> tuple[object, ...]:
    if (
        isinstance(value, Mapping)
        or isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
    ):
        raise ValueError(f"{field_name} must be a sequence")
    return tuple(value)


def _string_mapping(value: object, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    copied = dict(value)
    for key, item in copied.items():
        _string(key, f"{field_name} key")
        _string(item, f"{field_name}[{key}] value")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class SceneObjectManifest:
    """Explicit metadata for one physical object, without geometry."""

    physical_instance_id: str
    brick_id: int
    prim_path: str
    asset_revision_sha256: str
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "physical_instance_id",
            _string(self.physical_instance_id, "physical_instance_id"),
        )

        brick_id = _integer(self.brick_id, "brick_id")
        if brick_id < 0:
            raise ValueError("brick_id must be non-negative")
        object.__setattr__(self, "brick_id", brick_id)

        object.__setattr__(
            self,
            "prim_path",
            _absolute_path(self.prim_path, "prim_path"),
        )
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


def _manifest_object_payload(item: SceneObjectManifest) -> dict[str, object]:
    return {
        "physical_instance_id": item.physical_instance_id,
        "brick_id": item.brick_id,
        "prim_path": item.prim_path,
        "asset_revision_sha256": item.asset_revision_sha256,
        "provenance": dict(item.provenance),
    }


@dataclass(frozen=True)
class SceneSnapshotManifest:
    """Immutable, explicit metadata for one scene snapshot collection."""

    schema: str
    scene_id: str
    frame_id: str
    objects: tuple[SceneObjectManifest, ...]

    def __post_init__(self) -> None:
        schema = _string(self.schema, "schema")
        if schema != SCENE_SNAPSHOT_MANIFEST_SCHEMA:
            raise ValueError(
                f"schema must equal {SCENE_SNAPSHOT_MANIFEST_SCHEMA}"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "scene_id", _string(self.scene_id, "scene_id"))
        object.__setattr__(self, "frame_id", _string(self.frame_id, "frame_id"))

        objects = _sequence(self.objects, "objects")
        if not objects:
            raise ValueError("objects must be non-empty")
        for index, item in enumerate(objects):
            if type(item) is not SceneObjectManifest:
                raise ValueError(
                    f"objects[{index}] must be exactly a SceneObjectManifest"
                )
        normalized_objects = tuple(objects)
        object.__setattr__(self, "objects", normalized_objects)

        physical_ids = tuple(item.physical_instance_id for item in normalized_objects)
        if len(set(physical_ids)) != len(physical_ids):
            raise ValueError("objects physical_instance_id values must be unique")
        if any(left >= right for left, right in zip(physical_ids, physical_ids[1:])):
            raise ValueError(
                "objects must already be strictly ordered by physical_instance_id"
            )

        prim_paths = tuple(item.prim_path for item in normalized_objects)
        if len(set(prim_paths)) != len(prim_paths):
            raise ValueError("objects prim_path values must be unique")

    def to_canonical_payload(self) -> dict[str, object]:
        """Return a fresh ordinary JSON-compatible manifest payload."""
        return {
            "schema": self.schema,
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "objects": [
                _manifest_object_payload(item) for item in self.objects
            ],
        }

    @property
    def manifest_sha256(self) -> str:
        """Return the canonical digest of this manifest's public payload."""
        return canonical_json_digest(self.to_canonical_payload())


def _aabb_payload(scene_object: SceneObjectSnapshot) -> dict[str, object]:
    aabb = scene_object.aabb
    return {
        "object_id": aabb.object_id,
        "frame_id": aabb.frame_id,
        "minimum_m": list(aabb.minimum_m),
        "maximum_m": list(aabb.maximum_m),
        "geometry_digest": aabb.geometry_digest,
    }


def _snapshot_object_payload(scene_object: SceneObjectSnapshot) -> dict[str, object]:
    return {
        "physical_instance_id": scene_object.physical_instance_id,
        "brick_id": scene_object.brick_id,
        "prim_path": scene_object.prim_path,
        "aabb": _aabb_payload(scene_object),
        "asset_revision_sha256": scene_object.asset_revision_sha256,
        "provenance": dict(scene_object.provenance),
    }


def _snapshot_host_payload(host: RuntimeHostBinding) -> dict[str, object]:
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
        "host": _snapshot_host_payload(host),
        "objects": [_snapshot_object_payload(item) for item in objects],
    }


def _strict_bounds_pair(value: object) -> tuple[object, object]:
    if (
        isinstance(value, Mapping)
        or isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
    ):
        raise ValueError("world bounds must be a two-element sequence")
    if len(value) != 2:
        raise ValueError("world bounds must contain exactly minimum and maximum")
    return value[0], value[1]


def _collection_error(
    physical_instance_id: str, category: str
) -> SceneSnapshotCollectionError:
    return SceneSnapshotCollectionError(physical_instance_id, category)


def collect_scene_snapshot(
    manifest: SceneSnapshotManifest,
    host: RuntimeHostBinding,
    world_bounds_reader: WorldBoundsReader,
) -> SceneSnapshot:
    """Collect one immutable snapshot from explicit metadata and host bounds."""
    if type(manifest) is not SceneSnapshotManifest:
        raise _collection_error("<manifest>", "manifest_type")
    if type(host) is not RuntimeHostBinding:
        raise _collection_error("<host>", "host_type")
    if not callable(world_bounds_reader):
        raise _collection_error("<reader>", "reader_type")
    try:
        _validate_manifest_invariants(manifest)
    except _ManifestValidationError as exc:
        raise _collection_error(
            exc.physical_instance_id, "manifest_validation"
        ) from exc
    except Exception as exc:
        raise _collection_error("<manifest>", "manifest_validation") from exc
    try:
        _validate_host_invariants(host)
    except Exception as exc:
        raise _collection_error("<host>", "host_validation") from exc

    scene_objects: list[SceneObjectSnapshot] = []
    current_physical_instance_id = "<snapshot>"
    for item in manifest.objects:
        current_physical_instance_id = item.physical_instance_id
        try:
            bounds = world_bounds_reader(item.prim_path)
        except Exception as exc:
            raise _collection_error(
                current_physical_instance_id, "reader_error"
            ) from exc

        try:
            minimum_m, maximum_m = _strict_bounds_pair(bounds)
        except Exception as exc:
            raise _collection_error(
                current_physical_instance_id, "bounds_shape"
            ) from exc

        try:
            aabb = planning_scene.box_from_aabb(
                object_id=item.physical_instance_id,
                frame_id=manifest.frame_id,
                minimum_m=minimum_m,
                maximum_m=maximum_m,
            )
        except Exception as exc:
            raise _collection_error(
                current_physical_instance_id, "aabb_validation"
            ) from exc

        try:
            scene_objects.append(
                SceneObjectSnapshot(
                    physical_instance_id=item.physical_instance_id,
                    brick_id=item.brick_id,
                    prim_path=item.prim_path,
                    aabb=aabb,
                    asset_revision_sha256=item.asset_revision_sha256,
                    provenance=item.provenance,
                )
            )
        except Exception as exc:
            raise _collection_error(
                current_physical_instance_id, "scene_object_validation"
            ) from exc

    try:
        payload = _snapshot_payload(
            schema=SCENE_SNAPSHOT_SCHEMA,
            scene_id=manifest.scene_id,
            frame_id=manifest.frame_id,
            host=host,
            objects=scene_objects,
        )
        snapshot_sha256 = canonical_json_digest(payload)
        return SceneSnapshot(
            schema=SCENE_SNAPSHOT_SCHEMA,
            scene_id=manifest.scene_id,
            frame_id=manifest.frame_id,
            host=host,
            objects=scene_objects,
            snapshot_sha256=snapshot_sha256,
        )
    except Exception as exc:
        raise _collection_error(
            current_physical_instance_id, "snapshot_validation"
        ) from exc


__all__ = [
    "SCENE_SNAPSHOT_MANIFEST_SCHEMA",
    "SceneSnapshotCollectionError",
    "SceneObjectManifest",
    "SceneSnapshotManifest",
    "WorldBoundsReader",
    "collect_scene_snapshot",
]
class _ManifestValidationError(ValueError):
    def __init__(self, physical_instance_id: str, message: str) -> None:
        self.physical_instance_id = physical_instance_id
        super().__init__(message)


def _safe_manifest_physical_instance_id(item: object) -> str:
    try:
        candidate = item.physical_instance_id  # type: ignore[attr-defined]
    except Exception:
        return "<manifest>"
    return candidate if type(candidate) is str else "<manifest>"


def _validate_manifest_object(item: object) -> tuple[str, str]:
    if type(item) is not SceneObjectManifest:
        raise ValueError("must be exactly a SceneObjectManifest")

    physical_instance_id = _string(
        item.physical_instance_id, "physical_instance_id"
    )
    brick_id = _integer(item.brick_id, "brick_id")
    if brick_id < 0:
        raise ValueError("brick_id must be non-negative")
    prim_path = _absolute_path(item.prim_path, "prim_path")
    _sha256(item.asset_revision_sha256, "asset_revision_sha256")

    provenance = item.provenance
    if type(provenance) is not MappingProxyType or not provenance:
        raise ValueError("provenance must be a non-empty immutable mapping")
    for key, value in provenance.items():
        _string(key, "provenance key")
        _string(value, f"provenance[{key}] value")
    return physical_instance_id, prim_path


def _validate_manifest_invariants(manifest: object) -> None:
    if type(manifest) is not SceneSnapshotManifest:
        raise ValueError("must be exactly a SceneSnapshotManifest")

    schema = _string(manifest.schema, "schema")
    if schema != SCENE_SNAPSHOT_MANIFEST_SCHEMA:
        raise ValueError(f"schema must equal {SCENE_SNAPSHOT_MANIFEST_SCHEMA}")
    _string(manifest.scene_id, "scene_id")
    _string(manifest.frame_id, "frame_id")

    objects = manifest.objects
    if type(objects) is not tuple:
        raise ValueError("objects must be exactly a tuple")
    if not objects:
        raise ValueError("objects must be non-empty")

    physical_ids: list[str] = []
    prim_paths: list[str] = []
    for index, item in enumerate(objects):
        physical_instance_id = _safe_manifest_physical_instance_id(item)
        try:
            item_id, prim_path = _validate_manifest_object(item)
        except Exception as exc:
            raise _ManifestValidationError(
                physical_instance_id,
                f"objects[{index}] is invalid: {exc}",
            ) from exc
        physical_ids.append(item_id)
        prim_paths.append(prim_path)

    if len(set(physical_ids)) != len(physical_ids):
        raise _ManifestValidationError(
            physical_ids[-1], "physical_instance_id values must be unique"
        )
    if any(left >= right for left, right in zip(physical_ids, physical_ids[1:])):
        raise _ManifestValidationError(
            physical_ids[-1],
            "objects must already be strictly ordered by physical_instance_id",
        )
    if len(set(prim_paths)) != len(prim_paths):
        raise _ManifestValidationError(
            physical_ids[-1], "prim_path values must be unique"
        )


_HOST_BINDING_FIELDS = (
    "host_instance_id",
    "ros_domain_id",
    "host_profile_sha256",
    "host_bridge_schema",
    "runtime_usd_path",
    "runtime_usd_sha256",
    "robot_id",
    "robot_prim_path",
    "articulation_prim_path",
    "joint_dofs",
    "participant_count",
)


def _validate_host_invariants(host: object) -> RuntimeHostBinding:
    if type(host) is not RuntimeHostBinding:
        raise ValueError("must be exactly a RuntimeHostBinding")

    values = {
        field_name: getattr(host, field_name)  # type: ignore[arg-type]
        for field_name in _HOST_BINDING_FIELDS
    }
    validated = RuntimeHostBinding(**values)
    for field_name, value in values.items():
        normalized = getattr(validated, field_name)
        if type(value) is not type(normalized) or value != normalized:
            raise ValueError(f"{field_name} is not in its normalized immutable form")
    return validated
