"""Pure identity bindings and canonical digests for task structure bytes.

``StructuralBinding`` is an identity statement, not a pose: it ties one
assembly task to one physical instance, brick and robot at integer grid
coordinates with canonical plan and task-structure digests. The binding carries
no metric pose, frame, or TCP value and never repairs or infers one, and this
module imports nothing from ROS, Isaac, the metric plan contracts, or execution
code. ``canonical_json_digest`` is the single canonical JSON recipe the rest of
the repository uses for structure and snapshot digests.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

MOTION_GEOMETRY_OMITTED = "omitted"
_BINDING_SHA_FIELDS = ("plan_sha256", "task_start_sha256", "task_goal_sha256")


def canonical_json_digest(payload: object) -> str:
    """Return the lowercase SHA-256 of canonical JSON bytes for ``payload``.

    The canonical form is the exact recipe the STAP contract already uses for
    plan identifiers: ``json.dumps`` with sorted keys, comma separators, and
    ASCII-only output, encoded as UTF-8. Task structure files are hashed with
    the same recipe so a plan's ``source`` digests are comparable byte-for-byte.

    Args:
        payload: A JSON-compatible object.

    Returns:
        The 64-character lowercase SHA-256 hex digest.
    """
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _sha(value: str, field: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


@dataclass(frozen=True)
class StructuralBinding:
    """Grid-level identity of one pending STAP task against one robot.

    All fields are structural: grid coordinates are integer stud/layer units,
    ``orientation`` is the quarter-turn count, and the three digests pin the
    exact plan and task structure bytes this binding was derived from.
    ``motion_geometry`` is always ``"omitted"``: a structural binding never
    carries metric pose information.
    """

    plan_id: str
    task_id: str
    physical_instance_id: str
    brick_id: int
    robot_id: str
    source_goal_key: str
    grid_xyz: tuple[int, int, int]
    orientation: int
    color: str
    motion_geometry: str
    plan_sha256: str
    task_start_sha256: str
    task_goal_sha256: str

    def __post_init__(self) -> None:
        for field in ("plan_id", "task_id", "physical_instance_id", "robot_id",
                      "source_goal_key", "color"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        object.__setattr__(self, "brick_id", _int(self.brick_id, "brick_id"))
        if self.brick_id < 0:
            raise ValueError("brick_id must not be negative")
        grid = self.grid_xyz
        if (
            not isinstance(grid, Sequence)
            or isinstance(grid, (str, bytes))
            or len(grid) != 3
        ):
            raise ValueError("grid_xyz must contain exactly three integers")
        object.__setattr__(
            self,
            "grid_xyz",
            tuple(_int(value, "grid_xyz") for value in grid),
        )
        if any(value < 0 for value in self.grid_xyz):
            raise ValueError("grid_xyz entries must not be negative")
        object.__setattr__(self, "orientation", _int(self.orientation, "orientation"))
        if self.orientation not in (0, 1):
            raise ValueError("orientation must be a quarter-turn count of 0 or 1")
        if self.motion_geometry != MOTION_GEOMETRY_OMITTED:
            raise ValueError(
                f"motion_geometry must be {MOTION_GEOMETRY_OMITTED!r}; a structural "
                "binding never carries metric motion data"
            )
        for field in _BINDING_SHA_FIELDS:
            object.__setattr__(self, field, _sha(getattr(self, field), field))
