"""Generate and validate one-step symbolic brick-assembly tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Literal, TypedDict

from bricksim.colors import parse_color
from bricksim.topology.legolization import legolization_json_to_topology_json

StructureFamily = Literal["basic", "adjacent", "multilevel", "dense", "bridge"]
DatasetSplit = Literal["train", "validation", "test_structure", "test_relation"]

BRICK_DIMENSIONS: dict[int, tuple[int, int]] = {
    2: (2, 4),
    3: (2, 6),
    4: (1, 8),
    5: (1, 4),
    6: (1, 6),
    9: (1, 2),
    12: (2, 2),
}
BRICK_ID_BY_CANONICAL_DIMS = {
    tuple(sorted(dimensions)): brick_id
    for brick_id, dimensions in BRICK_DIMENSIONS.items()
}
COLORS = ("Blue", "Green", "Red", "Yellow", "Dark Turquoise", "Orange")
BASE_PLATE_SIZE = (32, 32)


class BrickJson(TypedDict):
    """One brick in the challenge task JSON format."""

    x: int
    y: int
    z: int
    ori: int
    brick_id: int
    color: str


StructureJson = dict[str, BrickJson]


@dataclass(frozen=True)
class AssemblyConnection:
    """One expected stud-to-hole relation for the target brick."""

    reference_part_id: int
    target_part_id: int
    stud_iface: int
    hole_iface: int
    offset: tuple[int, int]
    yaw: int
    overlap_studs: int


@dataclass(frozen=True)
class AssemblyStep:
    """A single target-brick action, possibly with multiple connections."""

    target_part_id: int
    primary_connection: AssemblyConnection
    additional_connections: tuple[AssemblyConnection, ...]
    context_part_ids: tuple[int, ...]
    structure_family: StructureFamily
    reference_level: int

    def to_json(self) -> dict[str, object]:
        """Return a JSON-compatible execution-plan record."""
        return {
            "target_part_id": self.target_part_id,
            "primary_connection": asdict(self.primary_connection),
            "additional_connections": [
                asdict(connection) for connection in self.additional_connections
            ],
            "context_part_ids": list(self.context_part_ids),
            "structure_family": self.structure_family,
            "reference_level": self.reference_level,
        }


@dataclass(frozen=True)
class GeneratedTask:
    """One generated Task-1 start/goal pair and its execution step."""

    sample_id: str
    split: DatasetSplit
    family: StructureFamily
    seed: int
    start: StructureJson
    goal: StructureJson
    topology: dict[str, object]
    step: AssemblyStep
    relation_signature: str


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for a deterministic symbolic task corpus."""

    output_dir: Path
    seed: int = 0
    count_per_family: int = 1
    families: tuple[StructureFamily, ...] = (
        "basic",
        "adjacent",
        "multilevel",
        "dense",
        "bridge",
    )


@dataclass(frozen=True)
class GenerationReport:
    """Summary of generated and validated artifacts."""

    output_dir: Path
    total_tasks: int
    family_counts: dict[str, int]
    split_counts: dict[str, int]


@dataclass(frozen=True)
class ValidationReport:
    """Validation result for one generated task."""

    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class GoalValidationReport:
    """BrickSim-oriented validation result for one goal structure."""

    valid: bool
    errors: tuple[str, ...]
    brick_count: int
    connection_count: int


@dataclass(frozen=True)
class _Relation:
    reference_brick_id: int
    target_brick_id: int
    reference_ori: int
    target_ori: int
    delta_x: int
    delta_y: int


def oriented_dimensions(brick_id: int, ori: int) -> tuple[int, int]:
    """Return the footprint dimensions along task-grid x and y."""
    length, width = BRICK_DIMENSIONS[brick_id]
    return (width, length) if ori % 2 else (length, width)


def enumerate_relations() -> tuple[_Relation, ...]:
    """Enumerate all one-reference relations with non-empty stud overlap.

    Returns:
        Stable relation catalog covering all supported brick pairs.
    """
    relations: list[_Relation] = []
    for reference_id in sorted(BRICK_DIMENSIONS):
        for target_id in sorted(BRICK_DIMENSIONS):
            for reference_ori in (0, 1):
                reference_length, reference_width = oriented_dimensions(
                    reference_id, reference_ori
                )
                for target_ori in (0, 1):
                    target_length, target_width = oriented_dimensions(
                        target_id, target_ori
                    )
                    for delta_x in range(-target_length + 1, reference_length):
                        for delta_y in range(-target_width + 1, reference_width):
                            relations.append(
                                _Relation(
                                    reference_id,
                                    target_id,
                                    reference_ori,
                                    target_ori,
                                    delta_x,
                                    delta_y,
                                )
                            )
    return tuple(relations)


def build_single_step_plan(
    start: StructureJson,
    goal: StructureJson,
    family: StructureFamily = "basic",
) -> AssemblyStep:
    """Build one target-centric step from a Task-1 start/goal pair.

    Returns:
        The single target-centric assembly step.
    """
    added = sorted(set(goal) - set(start), key=int)
    if len(added) != 1:
        raise ValueError(f"expected exactly one target brick, found {added}")
    target_key = added[0]
    target_id = int(target_key)
    topology = _to_topology(goal)
    connections: list[AssemblyConnection] = []
    for raw in topology["connections"]:  # type: ignore[index]
        if raw["hole_id"] != target_id:
            continue
        reference_id = int(raw["stud_id"])
        if reference_id == 0:
            continue
        overlap = _overlap_studs(goal[str(reference_id)], goal[target_key])
        connections.append(
            AssemblyConnection(
                reference_part_id=reference_id,
                target_part_id=target_id,
                stud_iface=int(raw["stud_iface"]),
                hole_iface=int(raw["hole_iface"]),
                offset=(int(raw["offset"][0]), int(raw["offset"][1])),
                yaw=int(raw["yaw"]),
                overlap_studs=overlap,
            )
        )
    if not connections:
        raise ValueError("target brick has no reference connection")
    connections.sort(key=lambda item: (-item.overlap_studs, item.reference_part_id))
    target_z = goal[target_key]["z"]
    return AssemblyStep(
        target_part_id=target_id,
        primary_connection=connections[0],
        additional_connections=tuple(connections[1:]),
        context_part_ids=tuple(sorted(int(key) for key in start)),
        structure_family=family,
        reference_level=target_z,
    )


def validate_generated_task(task: GeneratedTask) -> ValidationReport:
    """Validate geometry, one-step semantics, and topology consistency.

    Returns:
        Validation status and all discovered errors.
    """
    errors: list[str] = []
    added = set(task.goal) - set(task.start)
    if added != {str(task.step.target_part_id)}:
        errors.append("start and goal must differ by exactly the planned target")
    if not set(task.start).issubset(task.goal):
        errors.append("start is not a subset of goal")
    if sorted(int(key) for key in task.goal) != list(range(1, len(task.goal) + 1)):
        errors.append("part ids must be contiguous and start at one")
    if task.goal:
        minimum_xyz = tuple(
            min(brick[axis] for brick in task.goal.values())
            for axis in ("x", "y", "z")
        )
        if minimum_xyz != (0, 0, 0):
            errors.append(
                "generated structure must be normalized to local origin; "
                f"found minimum xyz={minimum_xyz}"
            )
    for first_key, first in task.goal.items():
        if not _inside_base_plate(first):
            errors.append(f"part {first_key} lies outside the base plate")
        for second_key, second in task.goal.items():
            if int(second_key) <= int(first_key):
                continue
            if first["z"] == second["z"] and _overlap_studs(first, second):
                errors.append(f"parts {first_key} and {second_key} overlap")
    try:
        rebuilt = build_single_step_plan(task.start, task.goal, task.family)
        if rebuilt != task.step:
            errors.append("execution plan does not match derived topology")
    except ValueError as exc:
        errors.append(str(exc))
    return ValidationReport(not errors, tuple(errors))


def validate_goal_structure(goal: StructureJson) -> GoalValidationReport:
    """Validate a goal JSON without requiring a start structure or plan.

    Returns:
        Field, geometry, support, and BrickSim topology validation result.
    """
    errors: list[str] = []
    connection_count = 0
    if not goal:
        errors.append("goal must contain at least one brick")
        return GoalValidationReport(False, tuple(errors), 0, 0)
    numeric_keys = sorted((key for key in goal if key.isdigit()), key=int)
    keys_valid = len(numeric_keys) == len(goal)
    if not keys_valid:
        errors.append("all top-level keys must be numeric brick ids")
    expected_keys = [str(index) for index in range(1, len(goal) + 1)]
    if numeric_keys != expected_keys:
        errors.append("brick ids must be contiguous strings starting at '1'")

    required_fields = {"x", "y", "z", "ori", "brick_id", "color"}
    structurally_valid = keys_valid
    for key in numeric_keys:
        brick = goal[key]
        brick_valid = True
        missing = required_fields - set(brick)
        extra = set(brick) - required_fields
        if missing:
            errors.append(f"brick {key} is missing fields: {sorted(missing)}")
            structurally_valid = False
            brick_valid = False
        if extra:
            errors.append(f"brick {key} has unknown fields: {sorted(extra)}")
        if not brick_valid:
            continue
        if brick["brick_id"] not in BRICK_DIMENSIONS:
            errors.append(f"brick {key} has unsupported brick_id {brick['brick_id']}")
            structurally_valid = False
            brick_valid = False
            continue
        for coordinate in ("x", "y", "z", "ori"):
            if not isinstance(brick[coordinate], int):
                errors.append(f"brick {key} field {coordinate} must be an integer")
                structurally_valid = False
                brick_valid = False
        if not brick_valid:
            continue
        if brick["z"] < 0:
            errors.append(f"brick {key} has negative z layer")
        if brick["ori"] not in (0, 1):
            errors.append(f"brick {key} ori must be 0 or 1")
        if not isinstance(brick["color"], str):
            errors.append(f"brick {key} color must be a supported color name")
        else:
            try:
                parse_color(brick["color"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"brick {key} has invalid color: {exc}")
        if not _inside_base_plate(brick):
            errors.append(f"brick {key} lies outside the 32x32 base plate")

    if structurally_valid:
        for first_key, first in goal.items():
            for second_key, second in goal.items():
                if int(second_key) <= int(first_key):
                    continue
                if first["z"] == second["z"] and _overlap_studs(first, second):
                    errors.append(
                        f"bricks {first_key} and {second_key} overlap on layer "
                        f"{first['z']}"
                    )
        for key, brick in goal.items():
            if brick["z"] == 0:
                continue
            supported = any(
                support["z"] == brick["z"] - 1
                and _overlap_studs(support, brick) > 0
                for support_key, support in goal.items()
                if support_key != key
            )
            if not supported:
                errors.append(f"brick {key} is unsupported at layer {brick['z']}")
        try:
            topology = _to_topology(goal)
            connections = topology["connections"]
            connection_count = len(connections)  # type: ignore[arg-type]
            reachable = {0}
            changed = True
            while changed:
                changed = False
                for connection in connections:  # type: ignore[union-attr]
                    stud_id = int(connection["stud_id"])
                    hole_id = int(connection["hole_id"])
                    if stud_id in reachable and hole_id not in reachable:
                        reachable.add(hole_id)
                        changed = True
            missing_parts = set(range(1, len(goal) + 1)) - reachable
            if missing_parts:
                errors.append(
                    f"parts are disconnected from the base plate: "
                    f"{sorted(missing_parts)}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"BrickSim topology conversion failed: {exc}")
    return GoalValidationReport(
        valid=not errors,
        errors=tuple(errors),
        brick_count=len(goal),
        connection_count=connection_count,
    )


def generate_symbolic_tasks(config: GenerationConfig) -> GenerationReport:
    """Generate, validate, and write Task-1 start/goal pairs by family.

    The output layout is ``<output>/<family>/<sequence>/``. Each sequence
    directory contains only ``structure_start.json`` and
    ``structure_goal.json`` so it can be consumed directly as a Task-1 task.

    Returns:
        Counts and output location for the generated corpus.
    """
    if config.count_per_family < 1:
        raise ValueError("count_per_family must be positive")
    relations = enumerate_relations()
    rng = Random(config.seed)
    relation_indices = list(range(len(relations)))
    rng.shuffle(relation_indices)
    family_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    total_tasks = 0
    cursor = 0
    for family in config.families:
        family_counts[family] = 0
        for index in range(config.count_per_family):
            relation_offset = cursor + index // 3
            relation = relations[
                relation_indices[relation_offset % len(relation_indices)]
            ]
            task_seed = config.seed * 100_003 + cursor * 31 + index + 1
            task = _generate_task(family, relation, task_seed, index)
            validation = validate_generated_task(task)
            if not validation.valid:
                raise ValueError(
                    f"generated invalid task {task.sample_id}: {validation.errors}"
                )
            _write_task_pair(config.output_dir, task, index + 1)
            total_tasks += 1
            family_counts[family] += 1
            split_counts[task.split] = split_counts.get(task.split, 0) + 1
        cursor += (config.count_per_family + 2) // 3
    return GenerationReport(
        output_dir=config.output_dir,
        total_tasks=total_tasks,
        family_counts=family_counts,
        split_counts=split_counts,
    )


def generate_demo_task(family: StructureFamily, seed: int) -> GeneratedTask:
    """Generate one deterministic in-memory task for the visual demo.

    Returns:
        Validated generated task.
    """
    relations = enumerate_relations()
    relation = relations[seed % len(relations)]
    task = _generate_task(family, relation, seed, 0)
    validation = validate_generated_task(task)
    if not validation.valid:
        raise ValueError(f"invalid demo task: {validation.errors}")
    return task


def write_generated_task(output_dir: Path, task: GeneratedTask) -> Path:
    """Write one task and return its TaskConfig-compatible directory.

    Returns:
        Directory containing the Task-1 start and goal files.
    """
    task_dir = output_dir / task.split / task.family / task.sample_id
    _write_task(output_dir, task)
    return task_dir


def _generate_task(
    family: StructureFamily,
    relation: _Relation,
    seed: int,
    index: int,
) -> GeneratedTask:
    rng = Random(seed)
    if family == "bridge":
        start, goal = _bridge_structure(rng, relation, index)
    else:
        start, goal = _single_reference_structure(family, relation, rng)
    start, goal = _normalize_structure_pair(start, goal)
    step = build_single_step_plan(start, goal, family)
    topology = _to_topology(goal)
    signature = _relation_signature(goal, step)
    split = _select_split(signature, index)
    digest = hashlib.sha256(
        f"{family}:{seed}:{signature}".encode()
    ).hexdigest()[:12]
    return GeneratedTask(
        sample_id=f"{family}-{digest}",
        split=split,
        family=family,
        seed=seed,
        start=start,
        goal=goal,
        topology=topology,
        step=step,
        relation_signature=signature,
    )


def _single_reference_structure(
    family: StructureFamily, relation: _Relation, rng: Random
) -> tuple[StructureJson, StructureJson]:
    reference_length, reference_width = oriented_dimensions(
        relation.reference_brick_id, relation.reference_ori
    )
    target_length, target_width = oriented_dimensions(
        relation.target_brick_id, relation.target_ori
    )
    min_dx = min(0, relation.delta_x)
    min_dy = min(0, relation.delta_y)
    max_dx = max(reference_length, relation.delta_x + target_length)
    max_dy = max(reference_width, relation.delta_y + target_width)
    minimum_origin_x = -min_dx
    maximum_origin_x = BASE_PLATE_SIZE[0] - max_dx
    minimum_origin_y = -min_dy
    maximum_origin_y = BASE_PLATE_SIZE[1] - max_dy
    if family in {"adjacent", "dense"}:
        minimum_origin_x += 2
        maximum_origin_x -= 2
        minimum_origin_y += 2
        maximum_origin_y -= 2
    origin_x = rng.randint(minimum_origin_x, maximum_origin_x)
    origin_y = rng.randint(minimum_origin_y, maximum_origin_y)
    reference_level = rng.choice((1, 2, 3)) if family == "multilevel" else 1
    bricks: list[BrickJson] = []
    for layer in range(reference_level):
        bricks.append(
            _brick(
                origin_x,
                origin_y,
                layer,
                relation.reference_ori,
                relation.reference_brick_id,
                rng,
            )
        )
    reference = bricks[-1]
    target = _brick(
        origin_x + relation.delta_x,
        origin_y + relation.delta_y,
        reference_level,
        relation.target_ori,
        relation.target_brick_id,
        rng,
    )
    if family == "adjacent":
        _add_ground_context(bricks, reference, target, rng, rng.randint(1, 4))
    elif family == "dense":
        _add_dense_context(bricks, target, rng)
    start = _renumber(bricks)
    goal = dict(start)
    goal[str(len(goal) + 1)] = target
    return start, goal


def _bridge_structure(
    rng: Random, relation: _Relation, variant_index: int
) -> tuple[StructureJson, StructureJson]:
    patterns = (
        ((1, 4), (1, 2), (1, 2)),
        ((1, 6), (1, 2), (1, 4)),
        ((1, 8), (1, 4), (1, 4)),
        ((2, 4), (2, 2), (2, 2)),
        ((2, 6), (2, 2), (2, 4)),
    )
    pattern_index = variant_index % len(patterns)
    target_dims, left_dims, right_dims = patterns[pattern_index]
    rotate = (relation.reference_ori + relation.target_ori) % 2
    target_id, target_ori = _brick_type_for_dims(target_dims, rotate)
    left_id, left_ori = _brick_type_for_dims(left_dims, rotate)
    right_id, right_ori = _brick_type_for_dims(right_dims, rotate)
    x, y = 12, 14
    left_length, left_width = oriented_dimensions(left_id, left_ori)
    right_length, right_width = oriented_dimensions(right_id, right_ori)
    target_length, target_width = oriented_dimensions(target_id, target_ori)
    if (
        left_length + right_length == target_length
        and left_width == right_width == target_width
    ):
        right_x, right_y = x + left_length, y
    elif (
        left_width + right_width == target_width
        and left_length == right_length == target_length
    ):
        right_x, right_y = x, y + left_width
    else:
        raise ValueError("bridge supports do not tile the target footprint")
    bricks = [
        _brick(x, y, 0, left_ori, left_id, rng),
        _brick(right_x, right_y, 0, right_ori, right_id, rng),
    ]
    target = _brick(x, y, 1, target_ori, target_id, rng)
    start = _renumber(bricks)
    goal = dict(start)
    goal["3"] = target
    return start, goal


def _add_ground_context(
    bricks: list[BrickJson],
    reference: BrickJson,
    target: BrickJson,
    rng: Random,
    count: int,
) -> None:
    candidates = _neighbor_candidates(target, brick_id=9)
    rng.shuffle(candidates)
    for candidate in candidates:
        if len(bricks) >= count + 1:
            return
        if any(
            brick["z"] == candidate["z"] and _overlap_studs(brick, candidate)
            for brick in bricks
        ):
            continue
        if _overlap_studs(candidate, target):
            continue
        candidate["color"] = rng.choice(COLORS)
        bricks.append(candidate)


def _add_dense_context(
    bricks: list[BrickJson], target: BrickJson, rng: Random
) -> None:
    target_length, target_width = oriented_dimensions(target["brick_id"], target["ori"])
    obstacle_id = 12
    obstacle_length, obstacle_width = oriented_dimensions(obstacle_id, 0)
    # Block one randomly selected grasp axis while keeping the orthogonal axis
    # open.  This produces episodes that require a 90-degree gripper rotation
    # without making a top-down parallel-jaw grasp geometrically impossible.
    position_groups = [
        (
            (target["x"] - obstacle_length - 1, target["y"]),
            (
                target["x"] + target_length + 1,
                target["y"] + target_width - obstacle_width,
            ),
        ),
        (
            (target["x"], target["y"] - obstacle_width - 1),
            (
                target["x"] + target_length - obstacle_length,
                target["y"] + target_width + 1,
            ),
        ),
    ]
    rng.shuffle(position_groups)
    for positions in position_groups:
        additions = []
        for x, y in positions:
            support = _brick(x, y, 0, 0, obstacle_id, rng)
            obstacle = _brick(x, y, target["z"], 0, obstacle_id, rng)
            if not all(_inside_base_plate(item) for item in (support, obstacle)):
                continue
            if any(
                brick["z"] == support["z"] and _overlap_studs(brick, support)
                for brick in (*bricks, *additions)
            ):
                continue
            additions.extend((support, obstacle))
        if additions:
            bricks.extend(additions)
            return
    raise ValueError("could not place dense context around target")


def _neighbor_candidates(target: BrickJson, brick_id: int) -> list[BrickJson]:
    target_length, target_width = oriented_dimensions(target["brick_id"], target["ori"])
    length, width = oriented_dimensions(brick_id, 0)
    positions = (
        (target["x"] - length, target["y"]),
        (target["x"] + target_length, target["y"]),
        (target["x"], target["y"] - width),
        (target["x"], target["y"] + target_width),
    )
    return [
        BrickJson(x=x, y=y, z=0, ori=0, brick_id=brick_id, color="Blue")
        for x, y in positions
    ]


def _brick_type_for_dims(dims: tuple[int, int], rotate: int) -> tuple[int, int]:
    canonical = tuple(sorted(dims))
    brick_id = BRICK_ID_BY_CANONICAL_DIMS[canonical]
    base_dims = BRICK_DIMENSIONS[brick_id]
    desired = (dims[1], dims[0]) if rotate else dims
    ori = 0 if base_dims == desired else 1
    return brick_id, ori


def _brick(
    x: int,
    y: int,
    z: int,
    ori: int,
    brick_id: int,
    rng: Random,
) -> BrickJson:
    return BrickJson(
        x=x,
        y=y,
        z=z,
        ori=ori,
        brick_id=brick_id,
        color=rng.choice(COLORS),
    )


def _renumber(bricks: list[BrickJson]) -> StructureJson:
    return {str(index): brick for index, brick in enumerate(bricks, start=1)}


def _normalize_structure_pair(
    start: StructureJson, goal: StructureJson
) -> tuple[StructureJson, StructureJson]:
    """Translate a start/goal pair to one shared local grid origin.

    Returns:
        Independently copied start and goal structures with shared translation.
    """
    minimum_x = min(brick["x"] for brick in goal.values())
    minimum_y = min(brick["y"] for brick in goal.values())
    minimum_z = min(brick["z"] for brick in goal.values())

    def translated(structure: StructureJson) -> StructureJson:
        return {
            key: BrickJson(
                **{
                    **brick,
                    "x": brick["x"] - minimum_x,
                    "y": brick["y"] - minimum_y,
                    "z": brick["z"] - minimum_z,
                }
            )
            for key, brick in structure.items()
        }

    return translated(start), translated(goal)


def _to_topology(structure: StructureJson) -> dict[str, object]:
    colors = [
        parse_color(structure[str(index)]["color"])
        for index in range(1, len(structure) + 1)
    ]
    return legolization_json_to_topology_json(
        structure,
        color=colors,
        include_base_plate=True,
        base_plate_size=BASE_PLATE_SIZE,
        base_plate_color=parse_color("Light Gray"),
    )


def _overlap_studs(first: BrickJson, second: BrickJson) -> int:
    first_length, first_width = oriented_dimensions(first["brick_id"], first["ori"])
    second_length, second_width = oriented_dimensions(second["brick_id"], second["ori"])
    overlap_x = max(
        0,
        min(first["x"] + first_length, second["x"] + second_length)
        - max(first["x"], second["x"]),
    )
    overlap_y = max(
        0,
        min(first["y"] + first_width, second["y"] + second_width)
        - max(first["y"], second["y"]),
    )
    return overlap_x * overlap_y


def _inside_base_plate(brick: BrickJson) -> bool:
    length, width = oriented_dimensions(brick["brick_id"], brick["ori"])
    return (
        0 <= brick["x"]
        and 0 <= brick["y"]
        and brick["x"] + length <= BASE_PLATE_SIZE[0]
        and brick["y"] + width <= BASE_PLATE_SIZE[1]
    )


def _relation_signature(goal: StructureJson, step: AssemblyStep) -> str:
    target = goal[str(step.target_part_id)]
    items = []
    for connection in (step.primary_connection, *step.additional_connections):
        reference = goal[str(connection.reference_part_id)]
        items.append(
            (
                reference["brick_id"],
                reference["ori"],
                connection.offset,
                connection.yaw,
                connection.overlap_studs,
            )
        )
    payload = (target["brick_id"], target["ori"], tuple(sorted(items)))
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def _select_split(signature: str, variant_index: int) -> DatasetSplit:
    bucket = int(signature[:8], 16) % 10
    if bucket < 2:
        return "test_relation"
    context_bucket = variant_index % 3
    if context_bucket == 1:
        return "validation"
    if context_bucket == 2:
        return "test_structure"
    return "train"


def _write_task(output_dir: Path, task: GeneratedTask) -> None:
    task_dir = output_dir / task.split / task.family / task.sample_id
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_json(task_dir / "structure_start.json", task.start)
    _write_json(task_dir / "structure_goal.json", task.goal)
    _write_json(task_dir / "topology.json", task.topology)
    _write_json(
        task_dir / "execution_plan.json",
        {
            "schema": "rocobrick/assembly_execution_plan@1",
            "steps": [task.step.to_json()],
        },
    )
    _write_json(
        task_dir / "metadata.json",
        {
            "sample_id": task.sample_id,
            "split": task.split,
            "family": task.family,
            "seed": task.seed,
            "relation_signature": task.relation_signature,
        },
    )


def _write_task_pair(
    output_dir: Path, task: GeneratedTask, sequence_number: int
) -> None:
    """Write the minimal on-disk representation used under tasks/type1."""
    task_dir = output_dir / task.family / str(sequence_number)
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_json(task_dir / "structure_start.json", task.start)
    _write_json(task_dir / "structure_goal.json", task.goal)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
