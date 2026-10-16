"""Build the runtime scene contract directly from task structure files.

This module is the structure-to-scene seam of ``piper-run``: it turns task
structure files plus one external ``robot-operation-plan/v1`` document into the
internal :class:`~rocobrick.stap_contract.StapPlan` the runtime host and the
BrickSim topology builder consume.

Entry points
------------

* :func:`build_structure_scene_plan` (``structure_dir``, ``robot_plan``): the
  pre-existing host-bridge entry, preserved unchanged. It resolves
  ``structure_goal.json`` (or ``structure.json``) and an optional
  ``structure_start.json`` inside one directory and keeps its original
  traversal-order id assignment, so the current host ``--structure-dir`` call
  behaves exactly as before.
* :func:`build_structure_scene_plan_from_files` (``start``, ``goal``,
  ``plan``, ``manifest``): the strict explicit-files entry. It reads exactly
  the named files, derives canonical physical instance ids from row content,
  and accepts a mapping only when it can prove it.

Scope
-----

Each returned ``StapPlan`` is a topology carrier only: it carries the goal part
geometry, the initially assembled ids, the declared robots, and one pending
``PLACE_DOWN`` task per newly placed part. Each task carries the operation's
``source_task_id``, part, executor, and directly evidenced initial support; no
STAP schedule, dependency, or H-LTL semantics are claimed, so the neutral
``level``/``schedule_slot``/``predecessors`` values and the empty hierarchy are
recorded as such. The carrier is never a validated ``assembly-plan/v2``
document, this module never parses one, and nothing here authorizes dispatch,
motion, or an ``EXECUTED`` verdict. It is offline validation and scene
preparation evidence only.

Strict inputs
-------------

* ``goal`` (required): a JSON object mapping source goal keys to brick rows
  ``{brick_id, x, y, z, ori, color}``; it must not be empty.
* ``start`` (optional): the same row shape. An omitted path, an empty file, or
  ``{}`` all mean EMPTY initial structure and are recorded canonically as
  ``start_path=None``, ``start_sha256=None``, ``start_bytes=0`` and
  ``initial_entries=0``. Every start row must be exactly one goal row: an
  initially assembled part is a goal part and is never placed.
* ``plan``: the raw ``robot-operation-plan/v1`` mapping, validated with
  :func:`~rocobrick.robot_operation_plan.parse_robot_operation_plan`.
* ``manifest`` (optional): a ``rocobrick-scene-snapshot-manifest/v1`` document.
  Its object ids must equal the goal part ids exactly, and every object's
  ``brick_id`` must match its goal row.

Strict binding
--------------

Structure rows carry no ids; the external plan refers to parts by physical
instance id. The binding is deterministic evidence, never operation traversal
order:

* the canonical physical instance id of a row is ``brick-`` followed by the
  first 20 hex characters of the SHA-256 of the compact JSON array
  ``[brick_id, x, y, z, ori, color]``, the derivation every generated
  assembly plan in this repository already uses;
* when every physical id the plan references is one of those content digests,
  every goal row is bound by that digest (mode ``"content-digest"``);
* otherwise a singleton role proof may apply: when exactly one goal row is
  initially assembled and exactly one goal part is newly placed, the plan ids
  are exactly one never-placed support plus one placed part, and none of them
  is a content digest, the bijection is forced (mode ``"singleton"``);
* any other unproven mapping raises :class:`StructureSceneError`. Ids are never
  assigned by operation order, by source key order, or by list position, and a
  plan id that is not a goal part id is never accepted.

Strict placements are validated against the goal structure: the placed uppers
are exactly the goal parts absent from the initial structure and each is placed
once, every named lower is the goal part directly beneath its upper (same x/y,
``z-1``) and is either initially assembled or placed in an earlier execution
layer and schedule slot, and every placing robot is a declared plan robot
carrying at most one supporting robot. A part at ``z == 0`` has no lower that
``robot-operation-plan/v1`` can express, so a baseplate placement fails closed
as unprovable instead of being guessed.

Metadata
--------

The strict metadata is JSON-safe evidence of the exact input bytes and of the
proven binding; it is also carried verbatim as ``StapPlan.raw``. The legacy
entry keeps its original metadata keys
(``structure_dir``/``goal_path``/``start_path``/``goal_sha256``/
``start_sha256``/``goal_entries``/``initial_entries``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from rocobrick.models import SkillType
from rocobrick.robot_operation_plan import (
    RobotOperationPlan,
    RobotOperationPlanError,
    parse_robot_operation_plan,
)
from rocobrick.scene_snapshot_collector import SCENE_SNAPSHOT_MANIFEST_SCHEMA
from rocobrick.stap_contract import StapPlan, StructuralPart, StructuralTask
from rocobrick.structural_binding import canonical_json_digest

_ROW_NAMES = ("brick_id", "x", "y", "z", "ori", "color")
_INTEGER_NAMES = ("brick_id", "x", "y", "z", "ori")
_ORIENTATIONS = (0, 1)
_INSTANCE_ID_PREFIX = "brick-"
_INSTANCE_ID_DIGEST_CHARS = 20
_CARRIER_PLAN_ID_PREFIX = "robot-operation-structure-"

# One strict brick row: (brick_id, x, y, z, ori, color).
_Row = tuple[int, int, int, int, int, str]


class StructureSceneError(ValueError):
    """Task structure files and one operation plan cannot be bound."""


# ---------------------------------------------------------------------------
# Directory-based entry point (preserved for the host bridge scene loader)
# ---------------------------------------------------------------------------


def _read(path: Path) -> tuple[bytes, Mapping[str, object]]:
    """Read one legacy structure file, strictly read-only.

    Args:
        path: The structure file to read.

    Returns:
        The exact file bytes and the decoded JSON object.

    Raises:
        StructureSceneError: If the file is unreadable or not a JSON object.
    """
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise StructureSceneError(f"cannot read structure file {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise StructureSceneError(f"structure file {path} must contain an object")
    return raw, value


def _record(value: object, key: str) -> dict[str, object]:
    """Validate one legacy structure row by required field types.

    Args:
        value: The decoded row.
        key: The source structure key, for error messages.

    Returns:
        The row as a dict, preserving any extra fields it carries.

    Raises:
        StructureSceneError: If a required field is missing or mistyped.
    """
    if not isinstance(value, Mapping):
        raise StructureSceneError(f"structure entry {key!r} must be an object")
    result = dict(value)
    for name in ("brick_id", "x", "y", "z", "ori"):
        if type(result.get(name)) is not int:
            raise StructureSceneError(
                f"structure entry {key!r}.{name} must be an integer"
            )
    if type(result.get("color")) is not str or not result["color"]:
        raise StructureSceneError(
            f"structure entry {key!r}.color must be a non-empty string"
        )
    return result


def _structure_digest(value: bytes) -> str:
    """Return the raw SHA-256 digest the legacy metadata records.

    Args:
        value: The exact file bytes.

    Returns:
        The lowercase SHA-256 hex digest.
    """
    return sha256(value).hexdigest()


def build_structure_scene_plan(
    structure_dir: Path,
    robot_plan: Mapping[str, object],
) -> tuple[StapPlan, dict[str, object]]:
    """Create the structural plan consumed by the existing scene loader.

    This is the compatibility entry point of the host bridge ``--structure-dir``
    call, preserved with its original semantics: it resolves
    ``structure_goal.json`` (or ``structure.json``) and an optional
    ``structure_start.json`` inside one directory and assigns the plan-local
    physical instance ids by operation traversal order. New callers should use
    :func:`build_structure_scene_plan_from_files`, which takes the explicit
    files and binds the ids by proven evidence instead.

    Args:
        structure_dir: Directory holding the task structure files.
        robot_plan: The ``robot-operation-plan/v1`` document as a mapping.

    Returns:
        The ``StapPlan`` consumed by the runtime scene loader and its metadata
        evidence.

    Raises:
        StructureSceneError: If the structure files or the plan cannot be read
            or bound.
    """
    goal_path = structure_dir / "structure_goal.json"
    start_path = structure_dir / "structure_start.json"
    if not goal_path.exists():
        goal_path = structure_dir / "structure.json"
    if not goal_path.exists():
        raise StructureSceneError(
            f"no structure_goal.json or structure.json in {structure_dir}"
        )
    goal_bytes, goal_doc = _read(goal_path)
    start_bytes: bytes | None = None
    start_doc: Mapping[str, object] = {}
    if start_path.exists():
        # A start structure is optional.  An empty file is the on-disk
        # representation of an EMPTY initial structure; the goal file remains
        # strict and is still rejected when blank or empty.
        raw_start = start_path.read_bytes()
        if raw_start.strip():
            start_bytes, start_doc = _read(start_path)
        else:
            start_bytes, start_doc = raw_start, {}
    goal_rows = {str(key): _record(value, str(key)) for key, value in goal_doc.items()}
    start_rows = {
        str(key): _record(value, str(key)) for key, value in start_doc.items()
    }
    if not goal_rows:
        raise StructureSceneError("goal structure must not be empty")
    robots = tuple(
        item for item in robot_plan.get("robots", []) if isinstance(item, str)
    )
    if not robots:
        raise StructureSceneError("robot operation plan has no robots")
    target_ids: list[str] = []
    lower_id: str | None = None
    for operation in robot_plan.get("operations", []):
        if not isinstance(operation, Mapping):
            continue
        target = operation.get("target")
        if not isinstance(target, Mapping):
            continue
        for name in (
            "physical_instance_id",
            "upper_physical_instance_id",
            "lower_physical_instance_id",
        ):
            value = target.get(name)
            if isinstance(value, str) and value not in target_ids:
                target_ids.append(value)
        if operation.get("skill") == "PLACE_DOWN":
            value = target.get("lower_physical_instance_id")
            if isinstance(value, str):
                lower_id = value
    if len(target_ids) < len(goal_rows):
        raise StructureSceneError(
            "robot operation plan does not identify every structural part"
        )
    initial_ids = ([lower_id] if lower_id else []) + [
        item for item in target_ids if item != lower_id
    ][: max(0, len(start_rows) - (1 if lower_id else 0))]
    moving_ids = [item for item in target_ids if item not in initial_ids]
    if len(initial_ids) != len(start_rows) or len(moving_ids) != (
        len(goal_rows) - len(start_rows)
    ):
        raise StructureSceneError(
            "cannot bind structure entries to physical instance ids"
        )
    key_to_id: dict[str, str] = {}
    for key in start_rows:
        key_to_id[key] = initial_ids.pop(0)
    for key in goal_rows:
        if key not in key_to_id:
            key_to_id[key] = moving_ids.pop(0)
    task_robot: dict[str, str] = {}
    for operation in robot_plan.get("operations", []):
        if isinstance(operation, Mapping) and operation.get("skill") == "PLACE_DOWN":
            task_id = operation.get("source_task_id")
            robot_id = operation.get("executor_robot_id")
            if isinstance(task_id, str) and isinstance(robot_id, str):
                task_robot[task_id] = robot_id
    goal_parts: dict[str, StructuralPart] = {}
    initially: set[str] = set()
    for key, row in goal_rows.items():
        instance_id = key_to_id[key]
        goal_parts[instance_id] = StructuralPart(
            physical_instance_id=instance_id,
            source_goal_key=key,
            source_start_key=key if key in start_rows else None,
            brick_id=row["brick_id"],
            x=row["x"],
            y=row["y"],
            z=row["z"],
            orientation_quarter_turns=row["ori"],
            color=row["color"],
        )
        if key in start_rows:
            initially.add(instance_id)
    tasks: dict[str, StructuralTask] = {}
    for instance_id in goal_parts:
        if instance_id in initially:
            continue
        task_id = next(
            (
                str(operation.get("source_task_id"))
                for operation in robot_plan.get("operations", [])
                if isinstance(operation, Mapping)
                and operation.get("skill") == "PLACE_DOWN"
                and isinstance(operation.get("target"), Mapping)
                and operation["target"].get("upper_physical_instance_id") == instance_id
            ),
            f"assemble-{instance_id.removeprefix('brick-')}",
        )
        tasks[task_id] = StructuralTask(
            task_id,
            instance_id,
            SkillType.PLACE_DOWN,
            task_robot.get(task_id, robots[0]),
            None,
            frozenset(),
            frozenset(),
            0,
            0,
        )
    goal_sha256 = _structure_digest(goal_bytes)
    metadata: dict[str, object] = {
        "structure_dir": str(structure_dir),
        "goal_path": str(goal_path),
        "start_path": str(start_path) if start_bytes is not None else None,
        "goal_sha256": goal_sha256,
        "start_sha256": (
            _structure_digest(start_bytes) if start_bytes is not None else None
        ),
        "goal_entries": len(goal_rows),
        "initial_entries": len(start_rows),
    }
    plan = StapPlan(
        plan_id="robot-operation-structure-" + goal_sha256[:24],
        robots=robots,
        source_goal_sha256=goal_sha256,
        source_start_sha256=(
            metadata["start_sha256"]
            if isinstance(metadata["start_sha256"], str)
            else None
        ),
        goal_parts=goal_parts,
        initially_assembled=frozenset(initially),
        tasks=tasks,
        hltl_hierarchy=(),
        hltl_root_formula="true",
        raw=metadata,
    )
    return plan, metadata


# ---------------------------------------------------------------------------
# Strict explicit-files entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlanFacts:
    """The physical facts one validated operation plan states.

    Attributes:
        referenced: Every physical instance id the plan references.
        placed: ``upper -> lower`` for every ``PLACE_DOWN`` operation.
        placement_operation: ``upper -> operation_id`` of that operation.
        connections: ``upper -> connection_id`` established by it.
    """

    referenced: frozenset[str]
    placed: Mapping[str, str]
    placement_operation: Mapping[str, str]
    placement_skill: Mapping[str, str]
    connections: Mapping[str, str]


def _read_bytes(path: Path, label: str) -> bytes:
    """Read one input file exactly once.

    Args:
        path: The file to read.
        label: The evidence label of the input.

    Returns:
        The exact file bytes.

    Raises:
        StructureSceneError: If the file cannot be read.
    """
    try:
        return path.read_bytes()
    except OSError as exc:
        raise StructureSceneError(f"cannot read {label} {path}: {exc}") from exc


def _decode_object(raw: bytes, path: Path, label: str) -> Mapping[str, object]:
    """Decode one JSON input and require a single object.

    Args:
        raw: The exact input bytes.
        path: The input path, for error messages.
        label: The evidence label of the input.

    Returns:
        The decoded JSON object.

    Raises:
        StructureSceneError: If the bytes are not a JSON object.
    """
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise StructureSceneError(f"{label} {path} is not valid JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise StructureSceneError(f"{label} {path} must contain a JSON object")
    return document


def _rows(document: Mapping[str, object], path: Path, label: str) -> dict[str, _Row]:
    """Extract the strict brick rows of one structure document.

    Args:
        document: The decoded structure object.
        path: The structure path, for error messages.
        label: The evidence label of the structure.

    Returns:
        The brick rows in source key order.

    Raises:
        StructureSceneError: If any entry key or brick row is malformed.
    """
    rows: dict[str, _Row] = {}
    for key, value in document.items():
        if not isinstance(key, str) or not key:
            raise StructureSceneError(
                f"{label} structure {path} has an invalid entry key {key!r}"
            )
        if not isinstance(value, Mapping):
            raise StructureSceneError(
                f"{label} structure entry {key!r} must be an object"
            )
        names = sorted(str(name) for name in value)
        if names != sorted(_ROW_NAMES):
            raise StructureSceneError(
                f"{label} structure entry {key!r} must carry exactly "
                f"{list(_ROW_NAMES)}; found {names}"
            )
        numbers: list[int] = []
        for name in _INTEGER_NAMES:
            item = value.get(name)
            if type(item) is not int:
                raise StructureSceneError(
                    f"{label} structure entry {key!r}.{name} must be an integer"
                )
            numbers.append(item)
        color = value.get("color")
        if not isinstance(color, str) or not color:
            raise StructureSceneError(
                f"{label} structure entry {key!r}.color must be a non-empty string"
            )
        brick_id, x, y, z, orientation = numbers
        if brick_id < 0 or z < 0 or orientation not in _ORIENTATIONS:
            raise StructureSceneError(
                f"{label} structure entry {key!r} has an invalid brick_id, z, "
                "or orientation"
            )
        rows[key] = (brick_id, x, y, z, orientation, color)
    return rows


def _physical_instance_id(row: _Row) -> str:
    """Return the canonical physical instance id of one brick row.

    The derivation is the one every generated assembly plan in this repository
    already uses: ``brick-`` plus the first 20 hex characters of the SHA-256
    over the compact JSON array ``[brick_id, x, y, z, ori, color]``.

    Args:
        row: The strict brick row.

    Returns:
        The canonical ``brick-<20 hex>`` physical instance id.
    """
    payload = json.dumps(list(row), separators=(",", ":"))
    digest = sha256(payload.encode("utf-8")).hexdigest()
    return _INSTANCE_ID_PREFIX + digest[:_INSTANCE_ID_DIGEST_CHARS]


def _read_manifest(path: Path) -> tuple[bytes, dict[str, int]]:
    """Read one scene manifest and extract its object ids and brick ids.

    Args:
        path: The manifest path.

    Returns:
        The exact manifest bytes and its ``physical_instance_id -> brick_id``
        objects.

    Raises:
        StructureSceneError: If the manifest is unreadable, malformed, or
            repeats a physical instance id.
    """
    raw = _read_bytes(path, "scene manifest")
    document = _decode_object(raw, path, "scene manifest")
    if document.get("schema") != SCENE_SNAPSHOT_MANIFEST_SCHEMA:
        raise StructureSceneError(
            f"scene manifest {path} schema must be exactly "
            f"{SCENE_SNAPSHOT_MANIFEST_SCHEMA!r}"
        )
    objects = document.get("objects")
    if (
        not isinstance(objects, Sequence)
        or isinstance(objects, (str, bytes, bytearray))
        or not objects
    ):
        raise StructureSceneError(f"scene manifest {path} carries no objects")
    brick_ids: dict[str, int] = {}
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise StructureSceneError(
                f"scene manifest {path} objects[{index}] must be an object"
            )
        instance_id = item.get("physical_instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise StructureSceneError(
                f"scene manifest {path} objects[{index}] carries no "
                "physical_instance_id"
            )
        if instance_id in brick_ids:
            raise StructureSceneError(
                f"scene manifest {path} repeats physical_instance_id {instance_id!r}"
            )
        brick_id = item.get("brick_id")
        if type(brick_id) is not int or brick_id < 0:
            raise StructureSceneError(
                f"scene manifest {path} objects[{index}].brick_id must be a "
                "non-negative integer"
            )
        brick_ids[instance_id] = brick_id
    return raw, brick_ids


def _target_values(operation_id: str, plan: RobotOperationPlan) -> Mapping[str, object]:
    """Return the validated target values of one non-``WAIT`` operation.

    Args:
        operation_id: The operation to inspect.
        plan: The validated operation plan.

    Returns:
        The operation's target values.

    Raises:
        StructureSceneError: If the operation carries no target.
    """
    target = plan.operations[operation_id].target
    if target is None:
        raise StructureSceneError(f"operation {operation_id!r} carries no target")
    return target.values


def _target_text(values: Mapping[str, object], name: str, path: str) -> str:
    """Return one non-empty string target value.

    Args:
        values: The target values mapping.
        name: The target key to read.
        path: The evidence path, for error messages.

    Returns:
        The non-empty string value.

    Raises:
        StructureSceneError: If the value is not a non-empty string.
    """
    item = values.get(name)
    if not isinstance(item, str) or not item:
        raise StructureSceneError(f"{path}.{name} must be a non-empty string")
    return item


def _plan_facts(plan: RobotOperationPlan) -> _PlanFacts:
    """Collect the physical facts one validated operation plan states.

    Args:
        plan: The validated ``robot-operation-plan/v1`` document.

    Returns:
        The referenced instance ids and the ``PLACE_DOWN`` placements.

    Raises:
        StructureSceneError: If a part is placed twice or onto itself, or a
            declared reference is not a non-empty string.
    """
    referenced: set[str] = set()
    placed: dict[str, str] = {}
    placement_operation: dict[str, str] = {}
    placement_skill: dict[str, str] = {}
    connections: dict[str, str] = {}
    for operation_id, operation in plan.operations.items():
        path = f"operation {operation_id!r}"
        if operation.skill in {"PLACE_DOWN", "PLACE_UP"}:
            values = _target_values(operation_id, plan)
            upper = _target_text(values, "upper_physical_instance_id", path)
            lower = _target_text(values, "lower_physical_instance_id", path)
            if upper == lower:
                raise StructureSceneError(
                    f"{path} {operation.skill} names the same part twice"
                )
            if upper in placed:
                raise StructureSceneError(
                    f"part {upper!r} is placed by both "
                    f"{placement_operation[upper]!r} and {operation_id!r}"
                )
            placed[upper] = lower
            placement_operation[upper] = operation_id
            placement_skill[upper] = operation.skill
            connections[upper] = _target_text(values, "connection_id", path)
            referenced.update((upper, lower))
        elif operation.skill == "PICK":
            values = _target_values(operation_id, plan)
            referenced.add(_target_text(values, "physical_instance_id", path))
        else:
            observation = operation.parameters.get("observation")
            if observation is not None:
                if not isinstance(observation, Mapping):
                    raise StructureSceneError(
                        f"{path}.parameters.observation must be an object"
                    )
                referenced.add(
                    _target_text(
                        observation,
                        "physical_instance_id",
                        f"{path}.parameters.observation",
                    )
                )
    return _PlanFacts(
        frozenset(referenced),
        dict(placed),
        dict(placement_operation),
        dict(placement_skill),
        dict(connections),
    )


def _forced_role_binding(
    goal_rows: Mapping[str, _Row],
    start_key_by_goal_key: Mapping[str, str],
    facts: _PlanFacts,
    digest_ids: frozenset[str],
) -> dict[str, str] | None:
    """Prove the id mapping when each structure role has exactly one member.

    An initially assembled part is never placed, so the plan ids that no
    ``PLACE_DOWN`` operation places (the supports) are exactly the initially
    assembled parts, and every placed upper is exactly one goal part absent
    from the initial structure. When each side of that partition has exactly
    one member, the bijection is forced even when the plan ids carry no content
    digest -- but only when none of the plan ids is a content digest, because a
    digest that names a different row would contradict the forced mapping.

    Args:
        goal_rows: The goal structure rows in source key order.
        start_key_by_goal_key: The ``goal key -> start key`` mapping of
            initially assembled parts.
        facts: The physical facts of the operation plan.
        digest_ids: The content-derived ids of every goal row.

    Returns:
        The forced ``goal key -> physical instance id`` binding, or ``None``
        when the partition does not force one.
    """
    if facts.referenced & digest_ids:
        return None
    supports = sorted(facts.referenced - set(facts.placed))
    initial_rows = sorted(start_key_by_goal_key)
    moving_rows = sorted(key for key in goal_rows if key not in start_key_by_goal_key)
    if len(initial_rows) != len(supports) or len(initial_rows) > 1:
        return None
    if len(moving_rows) != len(facts.placed) or len(moving_rows) > 1:
        return None
    binding: dict[str, str] = {}
    if initial_rows:
        binding[initial_rows[0]] = supports[0]
    if moving_rows:
        binding[moving_rows[0]] = sorted(facts.placed)[0]
    return binding


def _bind_goal_rows(
    goal_rows: Mapping[str, _Row],
    start_key_by_goal_key: Mapping[str, str],
    facts: _PlanFacts,
) -> tuple[dict[str, str], str]:
    """Bind every goal row to one plan-local physical instance id.

    Args:
        goal_rows: The goal structure rows in source key order.
        start_key_by_goal_key: The ``goal key -> start key`` mapping of
            initially assembled parts.
        facts: The physical facts of the operation plan.

    Returns:
        The ``goal key -> physical instance id`` binding and its proof mode,
        ``"content-digest"`` or ``"singleton"``.

    Raises:
        StructureSceneError: If the mapping cannot be proven.
    """
    derived = {key: _physical_instance_id(row) for key, row in goal_rows.items()}
    digest_ids = frozenset(derived.values())
    if len(digest_ids) != len(derived):
        raise StructureSceneError(
            "goal structure repeats a brick row; physical instance ids would "
            "be ambiguous"
        )
    if facts.referenced <= digest_ids:
        return dict(derived), "content-digest"
    forced = _forced_role_binding(goal_rows, start_key_by_goal_key, facts, digest_ids)
    if forced is not None:
        return forced, "singleton"
    unproven_rows = [key for key in goal_rows if derived[key] not in facts.referenced]
    unproven_ids = sorted(facts.referenced - digest_ids)
    raise StructureSceneError(
        "cannot bind the goal structure to the plan-local physical instance "
        f"ids: {len(unproven_rows)} goal entries and {len(unproven_ids)} plan "
        f"ids remain unproven (unproven plan ids: {unproven_ids})"
    )


def _operation_positions(
    plan: RobotOperationPlan,
) -> tuple[dict[str, int], dict[str, int]]:
    """Index every operation by its execution layer and schedule slot.

    Args:
        plan: The validated operation plan.

    Returns:
        The ``operation_id -> layer index`` and ``operation_id -> slot index``
        mappings.
    """
    layers: dict[str, int] = {}
    for index, layer in enumerate(plan.execution_layers):
        for operation_id in layer:
            layers[operation_id] = index
    slots: dict[str, int] = {}
    for index, slot in enumerate(plan.planned_schedule):
        for entry in slot:
            slots[entry.operation_id] = index
    return layers, slots


def _start_key_by_goal_key(
    goal_rows: Mapping[str, _Row], start_rows: Mapping[str, _Row]
) -> dict[str, str]:
    """Bind every initial structure row to exactly one goal structure row.

    Args:
        goal_rows: The goal structure rows in source key order.
        start_rows: The initial structure rows in source key order.

    Returns:
        The ``goal key -> start key`` mapping of initially assembled parts.

    Raises:
        StructureSceneError: If a start row repeats a goal row or is not a goal
            row at all.
    """
    goal_key_by_row = {row: key for key, row in goal_rows.items()}
    start_key_by_goal_key: dict[str, str] = {}
    for start_key, row in start_rows.items():
        goal_key = goal_key_by_row.get(row)
        if goal_key is None:
            raise StructureSceneError(
                f"start structure entry {start_key!r} is not a goal structure "
                "entry; an initially assembled part must be a goal part"
            )
        if goal_key in start_key_by_goal_key:
            raise StructureSceneError(
                f"start structure entry {start_key!r} repeats goal structure "
                f"entry {goal_key!r}"
            )
        start_key_by_goal_key[goal_key] = start_key
    return start_key_by_goal_key


def _validate_placements(
    facts: _PlanFacts,
    row_by_instance: Mapping[str, _Row],
    initial_ids: frozenset[str],
    layers: Mapping[str, int],
    slots: Mapping[str, int],
    allow_partial: bool = False,
) -> None:
    """Validate every placement against the goal and initial structures.

    Args:
        facts: The physical facts of the operation plan.
        row_by_instance: The goal rows indexed by physical instance id.
        initial_ids: The initially assembled physical instance ids.
        layers: The ``operation_id -> layer index`` mapping.
        slots: The ``operation_id -> slot index`` mapping.

    Raises:
        StructureSceneError: If a part is both initially assembled and placed,
            a goal part is never placed, a lower part is not the goal part
            directly beneath its upper, or that lower is not placed first.
    """
    placed_parts = {
        upper if facts.placement_skill[upper] == "PLACE_DOWN" else facts.placed[upper]
        for upper in facts.placed
    }
    both = sorted(placed_parts & initial_ids)
    if both:
        raise StructureSceneError(
            f"parts are both initially assembled and placed: {both}"
        )
    # A robot-operation-plan may intentionally be a skill-level prefix such
    # as TRANSIT -> PICK, with placement deferred to a later run. Once a
    # PLACE_DOWN or PLACE_UP operation is present, every non-initial goal part
    # must be accounted for by one placement operation.
    unplaced = sorted(set(row_by_instance) - initial_ids - placed_parts)
    if unplaced and not allow_partial:
        raise StructureSceneError(
            f"goal parts are not placed by any PLACE_DOWN operation: {unplaced}"
        )
    for upper in sorted(facts.placed):
        lower = facts.placed[upper]
        upper_row = row_by_instance[upper]
        lower_row = row_by_instance[lower]
        operation_id = facts.placement_operation[upper]
        skill = facts.placement_skill[upper]
        # PLACE_DOWN keeps the historical exact x/y support relation. PLACE_UP
        # represents task C: the moving lower is one layer below the fixed upper;
        # exact x/y/interface offset is established by the runtime connection.
        if skill == "PLACE_UP":
            if lower_row[3] != upper_row[3] - 1:
                raise StructureSceneError(
                    f"PLACE_UP for {upper!r} requires moving lower {lower!r} "
                    "to be one layer below the supported upper"
                )
        else:
            beneath = (upper_row[1], upper_row[2], upper_row[3] - 1)
            if (lower_row[1], lower_row[2], lower_row[3]) != beneath:
                hint = (
                    "; a part at z=0 has no goal part beneath it and cannot be "
                    "grounded by robot-operation-plan/v1"
                    if upper_row[3] == 0
                    else ""
                )
                raise StructureSceneError(
                    f"PLACE_DOWN for {upper!r} names support {lower!r} at "
                    f"(x={lower_row[1]}, y={lower_row[2]}, z={lower_row[3]}), not "
                    f"the goal part directly beneath {upper!r} at "
                    f"(x={upper_row[1]}, y={upper_row[2]}, z={upper_row[3]}){hint}"
                )
        if lower not in facts.placed:
            continue
        before = facts.placement_operation[lower]
        after = facts.placement_operation[upper]
        if layers[before] >= layers[after] or slots[before] >= slots[after]:
            raise StructureSceneError(
                f"part {lower!r} must be placed before {upper!r} rests on it, "
                f"but operation {before!r} does not precede {after!r} in the "
                "execution layers and planned schedule"
            )


def _validate_manifest(
    path: Path,
    manifest_brick_ids: Mapping[str, int],
    row_by_instance: Mapping[str, _Row],
) -> None:
    """Validate one scene manifest against the goal structure.

    Args:
        path: The manifest path, for error messages.
        manifest_brick_ids: The manifest ``physical_instance_id -> brick_id``
            objects.
        row_by_instance: The goal rows indexed by physical instance id.

    Raises:
        StructureSceneError: If the manifest ids are not exactly the goal part
            ids, or a manifest brick id does not match its goal row.
    """
    manifest_ids = set(manifest_brick_ids)
    goal_ids = set(row_by_instance)
    if manifest_ids != goal_ids:
        raise StructureSceneError(
            f"scene manifest {path} ids must equal the goal structure ids "
            f"exactly: missing {sorted(goal_ids - manifest_ids)}, unexpected "
            f"{sorted(manifest_ids - goal_ids)}"
        )
    mismatched = sorted(
        (instance_id, manifest_brick_ids[instance_id], row_by_instance[instance_id][0])
        for instance_id in manifest_ids
        if manifest_brick_ids[instance_id] != row_by_instance[instance_id][0]
    )
    if mismatched:
        raise StructureSceneError(
            f"scene manifest {path} brick ids do not match the goal structure "
            f"(physical_instance_id, manifest, goal): {mismatched}"
        )


def build_structure_scene_plan_from_files(
    start: Path | None,
    goal: Path,
    plan: Mapping[str, object],
    *,
    manifest: Path | None = None,
) -> tuple[StapPlan, dict[str, object]]:
    """Build the topology carrier of one structure assembly from exact inputs.

    Args:
        start: Initial structure file, or ``None`` for EMPTY initial
            structure. An empty file or ``{}`` is the same EMPTY structure.
        goal: Goal structure file; it must not be empty.
        plan: The external ``robot-operation-plan/v1`` document. It is
            validated here and never repaired or reordered.
        manifest: Optional ``rocobrick-scene-snapshot-manifest/v1`` document.
            When given, its object ids and brick ids must match the goal.

    Returns:
        The ``StapPlan`` topology carrier and its JSON-safe metadata evidence.
        The carrier is not an ``assembly-plan/v2`` document and authorizes no
        motion.

    Raises:
        StructureSceneError: If any input is unreadable or malformed, the
            external plan is invalid, the physical ids cannot be proven, the
            goal/start relation is inconsistent, or the manifest disagrees.
    """
    if not isinstance(plan, Mapping):
        raise StructureSceneError("plan must be a robot-operation-plan/v1 object")

    goal_bytes = _read_bytes(goal, "goal structure")
    goal_document = _decode_object(goal_bytes, goal, "goal")
    goal_rows = _rows(goal_document, goal, "goal")
    if not goal_rows:
        raise StructureSceneError("goal structure must not be empty")

    start_path: Path | None = None
    start_bytes = b""
    start_document: Mapping[str, object] | None = None
    start_rows: dict[str, _Row] = {}
    if start is not None:
        raw_start = _read_bytes(start, "start structure")
        if raw_start.strip():
            document = _decode_object(raw_start, start, "start")
            rows = _rows(document, start, "start")
            if rows:
                start_path = start
                start_bytes = raw_start
                start_document = document
                start_rows = rows

    manifest_bytes = b""
    manifest_brick_ids: dict[str, int] = {}
    if manifest is not None:
        manifest_bytes, manifest_brick_ids = _read_manifest(manifest)

    try:
        operation_plan = parse_robot_operation_plan(plan)
    except RobotOperationPlanError as exc:
        raise StructureSceneError(f"robot operation plan is invalid: {exc}") from exc

    facts = _plan_facts(operation_plan)
    start_key_by_goal_key = _start_key_by_goal_key(goal_rows, start_rows)
    key_to_id, binding_mode = _bind_goal_rows(goal_rows, start_key_by_goal_key, facts)
    row_by_instance = {key_to_id[key]: goal_rows[key] for key in sorted(goal_rows)}
    initial_ids = frozenset(key_to_id[goal_key] for goal_key in start_key_by_goal_key)
    layers, slots = _operation_positions(operation_plan)
    # Once a placement operation is present, require complete placement. A
    # A PICK-only prefix remains valid for scene preparation.
    allow_partial = (
        any(
            operation.skill == "PICK"
            for operation in operation_plan.operations.values()
        )
        and not facts.placed
    )
    _validate_placements(
        facts,
        row_by_instance,
        initial_ids,
        layers,
        slots,
        allow_partial=allow_partial,
    )
    if manifest is not None:
        _validate_manifest(manifest, manifest_brick_ids, row_by_instance)

    goal_parts: dict[str, StructuralPart] = {}
    for goal_key in sorted(goal_rows):
        row = goal_rows[goal_key]
        instance_id = key_to_id[goal_key]
        goal_parts[instance_id] = StructuralPart(
            physical_instance_id=instance_id,
            source_goal_key=goal_key,
            source_start_key=start_key_by_goal_key.get(goal_key),
            brick_id=row[0],
            x=row[1],
            y=row[2],
            z=row[3],
            orientation_quarter_turns=row[4],
            color=row[5],
        )

    tasks: dict[str, StructuralTask] = {}
    for upper in sorted(facts.placed):
        operation = operation_plan.operations[facts.placement_operation[upper]]
        skill = facts.placement_skill[upper]
        moving_part = upper if skill == "PLACE_DOWN" else facts.placed[upper]
        supporting = tuple(
            robot
            for robot in operation.participants
            if robot != operation.executor_robot_id
        )
        if len(supporting) > 1:
            raise StructureSceneError(
                f"operation {operation.operation_id!r} names more than one "
                f"supporting robot; one support robot is allowed"
            )
        if operation.source_task_id in tasks:
            raise StructureSceneError(
                f"source task {operation.source_task_id!r} places more than one part"
            )
        lower = facts.placed[upper]
        support_fact = lower if skill == "PLACE_DOWN" else upper
        tasks[operation.source_task_id] = StructuralTask(
            operation.source_task_id,
            moving_part,
            SkillType(skill),
            operation.executor_robot_id,
            supporting[0] if supporting else None,
            frozenset(),
            frozenset({support_fact}) if support_fact in initial_ids else frozenset(),
            0,
            0,
        )

    goal_canonical_sha256 = canonical_json_digest(goal_document)
    start_canonical_sha256 = (
        canonical_json_digest(start_document) if start_document is not None else None
    )
    plan_id = _CARRIER_PLAN_ID_PREFIX + goal_canonical_sha256[:24]
    placements: list[dict[str, object]] = []
    for upper in sorted(facts.placed):
        operation = operation_plan.operations[facts.placement_operation[upper]]
        placements.append(
            {
                "operation_id": operation.operation_id,
                "source_task_id": operation.source_task_id,
                "executor_robot_id": operation.executor_robot_id,
                "upper_physical_instance_id": upper,
                "lower_physical_instance_id": facts.placed[upper],
                "connection_id": facts.connections[upper],
            }
        )
        if operation.skill == "PLACE_UP":
            placements[-1]["skill"] = operation.skill
            placements[-1]["moving_physical_instance_id"] = facts.placed[upper]
    metadata: dict[str, object] = {
        "plan_id": plan_id,
        "robots": list(operation_plan.robots),
        "binding": binding_mode,
        "goal_path": str(goal),
        "goal_sha256": sha256(goal_bytes).hexdigest(),
        "goal_bytes": len(goal_bytes),
        "goal_canonical_sha256": goal_canonical_sha256,
        "goal_entries": len(goal_rows),
        "start_path": str(start_path) if start_path is not None else None,
        "start_sha256": (
            sha256(start_bytes).hexdigest() if start_path is not None else None
        ),
        "start_bytes": len(start_bytes),
        "start_canonical_sha256": start_canonical_sha256,
        "initial_entries": len(start_rows),
        "manifest_path": str(manifest) if manifest is not None else None,
        "manifest_sha256": (
            sha256(manifest_bytes).hexdigest() if manifest is not None else None
        ),
        "manifest_bytes": len(manifest_bytes),
        "manifest_entries": len(manifest_brick_ids),
        "physical_instance_ids": {key: key_to_id[key] for key in sorted(key_to_id)},
        "initial_instance_ids": sorted(initial_ids),
        "placed_instance_ids": sorted(facts.placed),
        "placements": placements,
        "robot_plan_id": operation_plan.plan_id,
        "robot_source_plan_id": operation_plan.source_plan_id,
    }
    carrier = StapPlan(
        plan_id=plan_id,
        robots=operation_plan.robots,
        source_goal_sha256=goal_canonical_sha256,
        source_start_sha256=start_canonical_sha256,
        goal_parts=goal_parts,
        initially_assembled=initial_ids,
        tasks=tasks,
        hltl_hierarchy=(),
        hltl_root_formula="true",
        raw=metadata,
    )
    return carrier, metadata
