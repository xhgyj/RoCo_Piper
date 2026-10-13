"""Formal planning-scene collision-object contract (pure, R5.A).

Frozen value types for axis-aligned box collision objects and whole-scene
updates, plus a deterministic SHA-256 digest over the canonical geometry
identity, so a caller can state the scene it expects and any applier result
can be verified against that claim. Pure data only: no ROS, no MoveIt, no
transport imports; the ROS adapter hook lives in
:class:`rocobrick.ros_moveit_runtime.RosMoveItRuntime`
(``planning_scene_applier`` / ``apply_planning_scene``).

Scope guard: this contract only ADDs standalone BOX collision objects. No
attached objects, no contact allowances, no SRDF semantics.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

SCHEMA: str = "rocobrick-planning-scene-update/v1"

# Canonical identity tags hashed into the digests (stable contract versions).
_BOX_IDENTITY_TAG: str = "rocobrick-planning-scene-box/v1"
_SCENE_IDENTITY_TAG: str = "rocobrick-planning-scene-digest/v1"
_ALLOWED_COLLISION_PAIR_TAG: str = "rocobrick-planning-scene-allowed-collision-pair/v1"


def _sha256_text(text: str) -> str:
    """Lowercase SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    """True for a lowercase 64-character SHA-256 hex digest string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _point3(value: object, field_name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of 3 finite numbers")
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 numbers")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field_name} entries must be numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} entries must be finite")
        numbers.append(number)
    return numbers[0], numbers[1], numbers[2]


def _canonical_point(point: tuple[float, float, float]) -> str:
    """Round-trip-exact canonical text for one 3-vector (repr floats)."""
    return "[{0}]".format(", ".join(repr(value) for value in point))


def box_geometry_digest(
    *,
    object_id: str,
    frame_id: str,
    minimum_m: tuple[float, float, float],
    maximum_m: tuple[float, float, float],
) -> str:
    """Deterministic SHA-256 over the canonical box identity.

    The identity is the newline-joined canonical form of the box version tag,
    object id, planning frame, and the repr-exact minimum/maximum corners, so
    equal boxes built by different callers always agree and any coordinate or
    naming change changes the digest.
    """
    identity = "\n".join(
        (
            _BOX_IDENTITY_TAG,
            f"object_id={object_id}",
            f"frame_id={frame_id}",
            f"minimum_m={_canonical_point(minimum_m)}",
            f"maximum_m={_canonical_point(maximum_m)}",
        )
    )
    return _sha256_text(identity)


def scene_digest(update: PlanningSceneUpdate) -> str:
    """Deterministic SHA-256 over boxes and explicit contact allowances."""
    identity = "\n".join(
        (
            _SCENE_IDENTITY_TAG,
            f"schema={update.schema}",
            f"scene_id={update.scene_id}",
            *[f"box={box.geometry_digest}" for box in update.boxes],
            *[
                f"allowed_collision_pair={first}|{second}"
                for first, second in update.allowed_collision_pairs
            ],
            *[
                f"forbidden_collision_pair={first}|{second}"
                for first, second in update.forbidden_collision_pairs
            ],
        )
    )
    return _sha256_text(identity)


@dataclass(frozen=True)
class AxisAlignedBox:
    """One axis-aligned BOX collision object with its provenance digest.

    ``minimum_m``/``maximum_m`` are the inclusive corners in ``frame_id``;
    every axis of the maximum corner must be strictly greater than the
    minimum (degenerate boxes are rejected). ``geometry_digest`` must equal
    :func:`box_geometry_digest` over the identity, so it can never diverge
    from the coordinates it claims to digest.
    """

    object_id: str
    frame_id: str
    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]
    geometry_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _string(self.object_id, "object_id"))
        object.__setattr__(self, "frame_id", _string(self.frame_id, "frame_id"))
        object.__setattr__(
            self, "minimum_m", _point3(self.minimum_m, "minimum_m")
        )
        object.__setattr__(
            self, "maximum_m", _point3(self.maximum_m, "maximum_m")
        )
        for axis, (low, high) in enumerate(zip(self.minimum_m, self.maximum_m)):
            if not high > low:
                raise ValueError(
                    f"maximum_m must exceed minimum_m on axis {axis} "
                    f"({high!r} <= {low!r})"
                )
        if not _is_sha256(self.geometry_digest):
            raise ValueError("geometry_digest must be a lowercase SHA-256 digest")
        expected = box_geometry_digest(
            object_id=self.object_id,
            frame_id=self.frame_id,
            minimum_m=self.minimum_m,
            maximum_m=self.maximum_m,
        )
        if self.geometry_digest != expected:
            raise ValueError(
                "geometry_digest does not match the canonical box identity"
            )

    @property
    def center_m(self) -> tuple[float, float, float]:
        """Center point of the box in its frame."""
        return tuple(
            (low + high) / 2.0 for low, high in zip(self.minimum_m, self.maximum_m)
        )

    @property
    def dimensions_m(self) -> tuple[float, float, float]:
        """Full side lengths of the box (maximum minus minimum per axis)."""
        return tuple(
            high - low for low, high in zip(self.minimum_m, self.maximum_m)
        )


def box_from_aabb(
    *,
    object_id: str,
    frame_id: str,
    minimum_m: tuple[float, float, float],
    maximum_m: tuple[float, float, float],
) -> AxisAlignedBox:
    """Pure builder: validate the AABB corners and digest them.

    Raises:
        ValueError: when the identity strings are empty, the corners are not
            three finite numbers each, any maximum component does not exceed
            its minimum, or the derived digest would be malformed.
    """
    minimum = _point3(minimum_m, "minimum_m")
    maximum = _point3(maximum_m, "maximum_m")
    return AxisAlignedBox(
        object_id=_string(object_id, "object_id"),
        frame_id=_string(frame_id, "frame_id"),
        minimum_m=minimum,
        maximum_m=maximum,
        geometry_digest=box_geometry_digest(
            object_id=object_id,
            frame_id=frame_id,
            minimum_m=minimum,
            maximum_m=maximum,
        ),
    )


@dataclass(frozen=True)
class PlanningSceneUpdate:
    """A whole-scene update claim: boxes plus the digest they must produce.

    ``expected_scene_digest`` is the caller's pinned claim over
    :func:`scene_digest`; an applier that cannot reproduce it reports a
    failed result instead of pretending the scene was applied.
    """

    scene_id: str
    boxes: tuple[AxisAlignedBox, ...]
    expected_scene_digest: str
    allowed_collision_pairs: tuple[tuple[str, str], ...] = ()
    schema: str = SCHEMA
    forbidden_collision_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _string(self.scene_id, "scene_id"))
        if isinstance(self.boxes, (str, bytes)) or not isinstance(
            self.boxes, Sequence
        ):
            raise ValueError("boxes must be a sequence of AxisAlignedBox items")
        boxes = tuple(self.boxes)
        for index, box in enumerate(boxes):
            if not isinstance(box, AxisAlignedBox):
                raise ValueError(f"boxes[{index}] must be an AxisAlignedBox")
        object.__setattr__(self, "boxes", boxes)
        pairs = tuple(
            (str(first), str(second))
            for first, second in self.allowed_collision_pairs
        )
        if any(
            not first
            or not second
            or first >= second
            for first, second in pairs
        ):
            raise ValueError(
                "allowed_collision_pairs must contain canonical non-empty "
                "strictly ordered string pairs"
            )
        if len(set(pairs)) != len(pairs) or tuple(sorted(pairs)) != pairs:
            raise ValueError(
                "allowed_collision_pairs must be unique and lexicographically ordered"
            )
        object.__setattr__(self, "allowed_collision_pairs", pairs)
        forbidden = tuple((str(first), str(second))
                          for first, second in self.forbidden_collision_pairs)
        if (any(not first or not second or first >= second
                for first, second in forbidden)
                or len(set(forbidden)) != len(forbidden)
                or tuple(sorted(forbidden)) != forbidden):
            raise ValueError("forbidden_collision_pairs must be canonical unique ordered pairs")
        if set(forbidden) & set(pairs):
            raise ValueError("collision pairs cannot be both allowed and forbidden")
        object.__setattr__(self, "forbidden_collision_pairs", forbidden)
        if not _is_sha256(self.expected_scene_digest):
            raise ValueError(
                "expected_scene_digest must be a lowercase SHA-256 digest"
            )
        object.__setattr__(self, "schema", _string(self.schema, "schema"))


@dataclass(frozen=True)
class PlanningSceneResult:
    """Outcome of one planning-scene application with its evidence."""

    passed: bool
    applied_object_ids: tuple[str, ...] = ()
    scene_digest: str = ""
    details: Mapping[str, object] = field(default_factory=dict)


__all__ = [
    "SCHEMA",
    "AxisAlignedBox",
    "PlanningSceneResult",
    "PlanningSceneUpdate",
    "box_from_aabb",
    "box_geometry_digest",
    "scene_digest",
]
