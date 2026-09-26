"""Offline tests for the structure-to-scene carrier binder.

Every test here is pure: it reads repository fixtures or writes temporary
structure files, never starts Isaac, ROS, Docker, or motion, and never modifies
a repository input byte.
"""

from __future__ import annotations

import ast
import inspect
import json
from hashlib import sha256
from pathlib import Path

import pytest

from rocobrick.models import SkillType
from rocobrick.plan.structure_scene import (
    StructureSceneError,
    build_structure_scene_plan,
    build_structure_scene_plan_from_files,
)
from rocobrick.structural_binding import canonical_json_digest

ROOT = Path(__file__).parents[1]
MODULE = ROOT / "src/rocobrick/plan/structure_scene.py"
TASK_DIR = ROOT / "tasks/type1/example1"
START = TASK_DIR / "structure_start.json"
GOAL = TASK_DIR / "structure_goal.json"
OPERATION_PLAN = ROOT / "artifacts/scene/example1-robot-operation-plan.json"
MANIFEST = ROOT / "artifacts/scene/example1-scene-manifest.json"
STAP_PLAN = (
    ROOT / "runs/example1-compound-pick-20260915T130000Z/generated-plan.json"
)

ROBOT = "piper_0"
BOTTOM = "brick-4bbba49bec24496bb21e"
TOP = "brick-768de5a17827920c48d4"
TASK_ID = "assemble-768de5a17827920c48d4"
BOTTOM_ROW = {"brick_id": 2, "x": 0, "y": 0, "z": 0, "ori": 0, "color": "Red"}
TOP_ROW = {"brick_id": 2, "x": 0, "y": 0, "z": 1, "ori": 0, "color": "Blue"}
UPPER_ROW = {"brick_id": 2, "x": 0, "y": 0, "z": 2, "ori": 0, "color": "Green"}


def _read_json(path: Path) -> dict:
    """Load one JSON object from disk.

    Args:
        path: The file to read.

    Returns:
        The decoded document.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, document: object) -> Path:
    """Write one JSON document into a temporary location.

    Args:
        path: The destination path.
        document: The JSON-compatible document.

    Returns:
        The destination path.
    """
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _instance_id(row: dict) -> str:
    """Derive the canonical physical instance id of one brick row.

    The derivation is re-implemented here, independently of the module under
    test, so the convention itself is pinned: this helper must reproduce the
    ids of the repository's generated plans.

    Args:
        row: The brick row.

    Returns:
        The canonical ``brick-<20 hex>`` physical instance id.
    """
    payload = [row[name] for name in ("brick_id", "x", "y", "z", "ori", "color")]
    digest = sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return "brick-" + digest.hexdigest()[:20]


def _place_down_operation(
    index: int, upper: str, lower: str, robot: str = ROBOT
) -> dict:
    """Build one valid ``PLACE_DOWN`` operation.

    Args:
        index: The operation index, used for its ids.
        upper: The placed physical instance id.
        lower: The support physical instance id.
        robot: The executor robot.

    Returns:
        The operation document.
    """
    operation_id = f"place-down-{index}"
    connection = f"conn-{index}"
    return {
        "operation_id": operation_id,
        "source_task_id": f"assemble-task-{index}",
        "skill": "PLACE_DOWN",
        "executor_robot_id": robot,
        "participants": [robot],
        "role_bindings": {"executor": robot},
        "target": {
            "connection_id": connection,
            "upper_physical_instance_id": upper,
            "lower_physical_instance_id": lower,
        },
        "parameters": {},
        "preconditions": [],
        "effects": [],
        "resources": [
            f"robot:{robot}",
            f"connection:{connection}",
            f"part:{upper}",
            f"part:{lower}",
        ],
        "evidence_requirements": ["connection_confirmed"],
    }


def _place_down_plan(
    placements: list[tuple[str, str]],
    *,
    robots: tuple[str, ...] = (ROBOT,),
    layers: list[list[str]] | None = None,
    schedule: list[list[tuple[str, str]]] | None = None,
) -> dict:
    """Build a minimal valid ``robot-operation-plan/v1`` PLACE_DOWN document.

    Args:
        placements: ``(upper, lower)`` pairs in operation order.
        robots: The declared plan robots.
        layers: Optional explicit execution layers.
        schedule: Optional explicit planned schedule.

    Returns:
        The operation plan document.
    """
    operations = [
        _place_down_operation(index, upper, lower)
        for index, (upper, lower) in enumerate(placements)
    ]
    operation_ids = [operation["operation_id"] for operation in operations]
    if layers is None:
        layers = [[operation_id] for operation_id in operation_ids]
    if schedule is None:
        schedule = [[(operation_id, ROBOT)] for operation_id in operation_ids]
    return {
        "schema": "robot-operation-plan/v1",
        "plan_id": "robot-operation-plan-synthetic",
        "source_plan_id": "assembly-plan-synthetic",
        "robots": list(robots),
        "operations": operations,
        "dependencies": [],
        "execution_layers": layers,
        "planned_schedule": [
            [
                {"operation_id": operation_id, "robot_id": robot}
                for operation_id, robot in slot
            ]
            for slot in schedule
        ],
        "evidence_requirements": [
            {"operation_id": operation_id, "evidence": "connection_confirmed"}
            for operation_id in operation_ids
        ],
    }


def test_public_apis_and_error_type_are_frozen() -> None:
    """Both entries expose exactly the agreed signatures and error type."""
    legacy = inspect.signature(build_structure_scene_plan).parameters
    assert list(legacy) == ["structure_dir", "robot_plan"]
    assert legacy["structure_dir"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert legacy["robot_plan"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    strict = inspect.signature(build_structure_scene_plan_from_files).parameters
    assert list(strict) == ["start", "goal", "plan", "manifest"]
    assert strict["manifest"].kind is inspect.Parameter.KEYWORD_ONLY
    assert strict["manifest"].default is None
    assert issubclass(StructureSceneError, ValueError)


def test_module_never_parses_an_external_assembly_plan() -> None:
    """The carrier must never import or call the STAP plan parser."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "parse_stap_plan" not in imported
    assert "parse_stap_plan" not in called
    assert "parse_plan" not in imported
    assert "parse_plan" not in called


def test_example1_binds_goal_start_and_plan_by_content_digest() -> None:
    """The frozen example1 inputs bind to the ids their plan already uses."""
    plan, metadata = build_structure_scene_plan_from_files(
        START, GOAL, _read_json(OPERATION_PLAN), manifest=MANIFEST
    )

    assert metadata["binding"] == "content-digest"
    assert metadata["physical_instance_ids"] == {"1": BOTTOM, "2": TOP}
    assert metadata["initial_instance_ids"] == [BOTTOM]
    assert metadata["placed_instance_ids"] == [TOP]
    assert metadata["robots"] == ["piper_0", "piper_1"]
    assert metadata["placements"] == [
        {
            "operation_id": "example1-piper0-place-down",
            "source_task_id": TASK_ID,
            "executor_robot_id": "piper_0",
            "upper_physical_instance_id": TOP,
            "lower_physical_instance_id": BOTTOM,
            "connection_id": "conn-example1-upper-lower",
        }
    ]
    assert metadata["robot_plan_id"] == "robot-operation-plan-example1-v1"
    assert (
        metadata["robot_source_plan_id"]
        == "assembly-plan-25a5d5cc586a1e5c65f2141d"
    )

    assert set(plan.goal_parts) == {BOTTOM, TOP}
    bottom = plan.goal_parts[BOTTOM]
    assert (bottom.source_goal_key, bottom.source_start_key) == ("1", "1")
    assert (
        bottom.brick_id,
        bottom.x,
        bottom.y,
        bottom.z,
        bottom.orientation_quarter_turns,
        bottom.color,
    ) == (2, 0, 0, 0, 0, "Red")
    top = plan.goal_parts[TOP]
    assert (top.source_goal_key, top.source_start_key) == ("2", None)
    assert (top.x, top.y, top.z, top.color) == (0, 0, 1, "Blue")

    assert plan.initially_assembled == frozenset({BOTTOM})
    assert plan.robots == ("piper_0", "piper_1")
    assert set(plan.tasks) == {TASK_ID}
    task = plan.tasks[TASK_ID]
    assert task.skill is SkillType.PLACE_DOWN
    assert task.physical_instance_id == TOP
    assert task.primary_robot == "piper_0"
    assert task.supporting_robot is None
    assert task.initial_supports == frozenset({BOTTOM})
    assert plan.raw is metadata


def test_metadata_records_exact_bytes_and_comparable_canonical_digests() -> None:
    """Metadata pins the exact input bytes and the STAP-comparable digests."""
    metadata = build_structure_scene_plan_from_files(
        START, GOAL, _read_json(OPERATION_PLAN), manifest=MANIFEST
    )[1]
    goal_bytes = GOAL.read_bytes()
    start_bytes = START.read_bytes()
    manifest_bytes = MANIFEST.read_bytes()

    assert metadata["goal_path"] == str(GOAL)
    assert metadata["goal_sha256"] == sha256(goal_bytes).hexdigest()
    assert metadata["goal_bytes"] == len(goal_bytes)
    assert metadata["goal_canonical_sha256"] == canonical_json_digest(
        _read_json(GOAL)
    )
    assert metadata["goal_entries"] == 2

    assert metadata["start_path"] == str(START)
    assert metadata["start_sha256"] == sha256(start_bytes).hexdigest()
    assert metadata["start_bytes"] == len(start_bytes)
    assert metadata["start_canonical_sha256"] == canonical_json_digest(
        _read_json(START)
    )
    assert metadata["initial_entries"] == 1

    assert metadata["manifest_path"] == str(MANIFEST)
    assert metadata["manifest_sha256"] == sha256(manifest_bytes).hexdigest()
    assert metadata["manifest_bytes"] == len(manifest_bytes)
    assert metadata["manifest_entries"] == 2

    # The canonical digests equal the source digests recorded by the generated
    # assembly plan, so provenance is comparable without parsing that plan.
    source = _read_json(STAP_PLAN)["source"]
    assert metadata["goal_canonical_sha256"] == source["goal_sha256"]
    assert metadata["start_canonical_sha256"] == source["start_sha256"]

    bare = build_structure_scene_plan_from_files(
        START, GOAL, _read_json(OPERATION_PLAN)
    )[1]
    assert bare["manifest_path"] is None
    assert bare["manifest_sha256"] is None
    assert bare["manifest_bytes"] == 0
    assert bare["manifest_entries"] == 0


def test_binding_ignores_operation_order_and_source_keys(tmp_path: Path) -> None:
    """Neither operation order nor structure keys may influence the binding."""
    document = _read_json(OPERATION_PLAN)
    baseline = build_structure_scene_plan_from_files(START, GOAL, document)[1]
    reversed_document = dict(
        document, operations=list(reversed(document["operations"]))
    )
    reordered = build_structure_scene_plan_from_files(START, GOAL, reversed_document)[1]
    assert reordered["physical_instance_ids"] == baseline["physical_instance_ids"]
    assert reordered["binding"] == "content-digest"
    assert reordered["plan_id"] == baseline["plan_id"]

    goal = _read_json(GOAL)
    start = _read_json(START)
    renamed_goal = {"b": goal["2"], "a": goal["1"]}
    renamed_start = {"a": start["1"]}
    plan, metadata = build_structure_scene_plan_from_files(
        _write_json(tmp_path / "start.json", renamed_start),
        _write_json(tmp_path / "goal.json", renamed_goal),
        document,
    )
    assert metadata["physical_instance_ids"] == {"a": BOTTOM, "b": TOP}
    assert plan.goal_parts[BOTTOM].source_goal_key == "a"
    assert plan.goal_parts[BOTTOM].source_start_key == "a"
    assert plan.goal_parts[TOP].source_goal_key == "b"
    assert plan.goal_parts[TOP].source_start_key is None


def test_singleton_partition_proves_opaque_plan_ids(tmp_path: Path) -> None:
    """One support and one newly placed part force the id mapping."""
    goal = _write_json(
        tmp_path / "goal.json", {"1": BOTTOM_ROW, "2": TOP_ROW}
    )
    start = _write_json(tmp_path / "start.json", {"1": BOTTOM_ROW})
    plan_document = _place_down_plan(
        [("brick-opaque-top", "brick-opaque-bottom")]
    )

    plan, metadata = build_structure_scene_plan_from_files(start, goal, plan_document)

    assert metadata["binding"] == "singleton"
    assert metadata["physical_instance_ids"] == {
        "1": "brick-opaque-bottom",
        "2": "brick-opaque-top",
    }
    assert plan.initially_assembled == frozenset({"brick-opaque-bottom"})
    assert set(plan.tasks) == {"assemble-task-0"}
    assert plan.tasks["assemble-task-0"].physical_instance_id == "brick-opaque-top"


def test_multi_part_opaque_mapping_fails_closed(tmp_path: Path) -> None:
    """Two supports or two placed parts with opaque ids are never guessed."""
    goal = _write_json(
        tmp_path / "goal.json",
        {"1": BOTTOM_ROW, "2": TOP_ROW, "3": UPPER_ROW},
    )
    start = _write_json(tmp_path / "start.json", {"1": BOTTOM_ROW})
    plan_document = _place_down_plan(
        [("brick-x", "brick-a"), ("brick-y", "brick-x")]
    )

    with pytest.raises(StructureSceneError, match="remain unproven"):
        build_structure_scene_plan_from_files(start, goal, plan_document)

    canonical = _place_down_plan(
        [
            (_instance_id(TOP_ROW), _instance_id(BOTTOM_ROW)),
            (_instance_id(UPPER_ROW), _instance_id(TOP_ROW)),
        ]
    )
    plan, metadata = build_structure_scene_plan_from_files(start, goal, canonical)
    assert metadata["binding"] == "content-digest"
    assert plan.initially_assembled == frozenset({_instance_id(BOTTOM_ROW)})
    assert {
        task.physical_instance_id for task in plan.tasks.values()
    } == {_instance_id(TOP_ROW), _instance_id(UPPER_ROW)}


def test_plan_must_cover_and_ground_every_goal_part(tmp_path: Path) -> None:
    """Uncovered, misplaced, late, and duplicated placements all fail closed."""
    document = _read_json(OPERATION_PLAN)

    with pytest.raises(StructureSceneError, match="not placed by any PLACE_DOWN"):
        build_structure_scene_plan_from_files(None, GOAL, document)

    empty = _write_json(tmp_path / "empty_start.json", {})
    with pytest.raises(StructureSceneError, match="not placed by any PLACE_DOWN"):
        build_structure_scene_plan_from_files(empty, GOAL, document)

    # A part at z=0 cannot be grounded by robot-operation-plan/v1.
    red = {"brick_id": 2, "x": 0, "y": 0, "z": 0, "ori": 0, "color": "Red"}
    blue = {"brick_id": 2, "x": 1, "y": 0, "z": 0, "ori": 0, "color": "Blue"}
    ground_goal = _write_json(
        tmp_path / "ground_goal.json", {"1": red, "2": blue}
    )
    ground_plan = _place_down_plan(
        [
            (_instance_id(blue), _instance_id(red)),
            (_instance_id(red), _instance_id(blue)),
        ]
    )
    with pytest.raises(
        StructureSceneError, match="cannot be grounded by robot-operation-plan/v1"
    ):
        build_structure_scene_plan_from_files(None, ground_goal, ground_plan)

    # A lower part that is not the goal part directly beneath its upper.
    green = {"brick_id": 2, "x": 2, "y": 0, "z": 1, "ori": 0, "color": "Green"}
    wrong_goal = _write_json(
        tmp_path / "wrong_goal.json",
        {"1": BOTTOM_ROW, "2": TOP_ROW, "3": green},
    )
    wrong_start = _write_json(tmp_path / "wrong_start.json", {"1": BOTTOM_ROW})
    wrong_plan = _place_down_plan(
        [
            (_instance_id(TOP_ROW), _instance_id(BOTTOM_ROW)),
            (_instance_id(green), _instance_id(BOTTOM_ROW)),
        ]
    )
    with pytest.raises(StructureSceneError, match="directly beneath"):
        build_structure_scene_plan_from_files(wrong_start, wrong_goal, wrong_plan)

    # A support that is itself placed later than the part resting on it.
    chain_goal = _write_json(
        tmp_path / "chain_goal.json",
        {"1": BOTTOM_ROW, "2": TOP_ROW, "3": UPPER_ROW},
    )
    chain_start = _write_json(tmp_path / "chain_start.json", {"1": BOTTOM_ROW})
    late = _place_down_plan(
        [
            (_instance_id(TOP_ROW), _instance_id(BOTTOM_ROW)),
            (_instance_id(UPPER_ROW), _instance_id(TOP_ROW)),
        ],
        layers=[["place-down-1"], ["place-down-0"]],
    )
    with pytest.raises(StructureSceneError, match="does not precede"):
        build_structure_scene_plan_from_files(chain_start, chain_goal, late)

    # A goal part that no operation places.
    partial = _place_down_plan(
        [(_instance_id(TOP_ROW), _instance_id(BOTTOM_ROW))]
    )
    with pytest.raises(StructureSceneError, match="not placed by any PLACE_DOWN"):
        build_structure_scene_plan_from_files(chain_start, chain_goal, partial)

    # One part placed twice: the same pair and connection id twice, so the
    # operation-plan contract itself accepts it and the binder must not.
    duplicated = _place_down_plan(
        [
            (_instance_id(TOP_ROW), _instance_id(BOTTOM_ROW)),
            (_instance_id(TOP_ROW), _instance_id(BOTTOM_ROW)),
        ]
    )
    duplicate_target = duplicated["operations"][0]["target"]
    second = duplicated["operations"][1]
    second["target"]["connection_id"] = duplicate_target["connection_id"]
    second["resources"] = list(duplicated["operations"][0]["resources"])
    with pytest.raises(StructureSceneError, match="placed by both"):
        build_structure_scene_plan_from_files(
            _write_json(tmp_path / "dup_start.json", {"1": BOTTOM_ROW}),
            _write_json(tmp_path / "dup_goal.json", {"1": BOTTOM_ROW, "2": TOP_ROW}),
            duplicated,
        )

    # A part that is both initially assembled and placed.
    with pytest.raises(StructureSceneError, match="initially assembled and placed"):
        build_structure_scene_plan_from_files(
            _write_json(
                tmp_path / "full_start.json", {"1": BOTTOM_ROW, "2": TOP_ROW}
            ),
            _write_json(
                tmp_path / "full_goal.json", {"1": BOTTOM_ROW, "2": TOP_ROW}
            ),
            _place_down_plan(
                [(_instance_id(TOP_ROW), _instance_id(BOTTOM_ROW))]
            ),
        )


def test_manifest_ids_and_brick_ids_must_match_exactly(tmp_path: Path) -> None:
    """A scene manifest is accepted only when it matches the goal exactly."""
    document = _read_json(OPERATION_PLAN)
    manifest = _read_json(MANIFEST)
    build_structure_scene_plan_from_files(START, GOAL, document, manifest=MANIFEST)

    missing = dict(manifest, objects=list(manifest["objects"])[:1])
    with pytest.raises(StructureSceneError, match="ids must equal"):
        build_structure_scene_plan_from_files(
            START, GOAL, document,
            manifest=_write_json(tmp_path / "missing.json", missing),
        )

    extra = dict(
        manifest,
        objects=list(manifest["objects"])
        + [{"physical_instance_id": "brick-extra", "brick_id": 2}],
    )
    with pytest.raises(StructureSceneError, match="unexpected"):
        build_structure_scene_plan_from_files(
            START, GOAL, document,
            manifest=_write_json(tmp_path / "extra.json", extra),
        )

    swapped = dict(manifest, objects=[dict(item) for item in manifest["objects"]])
    swapped["objects"][0]["brick_id"] = 7
    with pytest.raises(StructureSceneError, match="brick ids do not match"):
        build_structure_scene_plan_from_files(
            START, GOAL, document,
            manifest=_write_json(tmp_path / "swapped.json", swapped),
        )

    wrong_schema = dict(manifest, schema="rocobrick-scene-snapshot-manifest/v2")
    with pytest.raises(StructureSceneError, match="schema must be exactly"):
        build_structure_scene_plan_from_files(
            START, GOAL, document,
            manifest=_write_json(tmp_path / "schema.json", wrong_schema),
        )


def test_structure_rows_and_inputs_are_strict(tmp_path: Path) -> None:
    """Malformed rows, files, and duplicate rows are rejected as-is."""
    document = _read_json(OPERATION_PLAN)

    with pytest.raises(StructureSceneError, match="must contain a JSON object"):
        build_structure_scene_plan_from_files(
            None, _write_json(tmp_path / "list.json", [BOTTOM_ROW]), document
        )

    with pytest.raises(StructureSceneError, match="must not be empty"):
        build_structure_scene_plan_from_files(
            None, _write_json(tmp_path / "empty_goal.json", {}), document
        )

    missing_key = {
        key: value for key, value in BOTTOM_ROW.items() if key != "ori"
    }
    with pytest.raises(StructureSceneError, match="must carry exactly"):
        build_structure_scene_plan_from_files(
            None,
            _write_json(tmp_path / "missing_key.json", {"1": missing_key}),
            document,
        )

    with pytest.raises(StructureSceneError, match="must carry exactly"):
        build_structure_scene_plan_from_files(
            None,
            _write_json(
                tmp_path / "extra_key.json",
                {"1": dict(BOTTOM_ROW, studs=4)},
            ),
            document,
        )

    with pytest.raises(StructureSceneError, match="x must be an integer"):
        build_structure_scene_plan_from_files(
            None,
            _write_json(tmp_path / "bool.json", {"1": dict(BOTTOM_ROW, x=True)}),
            document,
        )

    with pytest.raises(StructureSceneError, match="invalid brick_id"):
        build_structure_scene_plan_from_files(
            None,
            _write_json(
                tmp_path / "negative.json", {"1": dict(BOTTOM_ROW, brick_id=-1)}
            ),
            document,
        )

    with pytest.raises(StructureSceneError, match="invalid brick_id"):
        build_structure_scene_plan_from_files(
            None,
            _write_json(tmp_path / "ori.json", {"1": dict(BOTTOM_ROW, ori=2)}),
            document,
        )

    with pytest.raises(StructureSceneError, match="color must be a non-empty"):
        build_structure_scene_plan_from_files(
            None,
            _write_json(tmp_path / "color.json", {"1": dict(BOTTOM_ROW, color="")}),
            document,
        )

    with pytest.raises(StructureSceneError, match="repeats a brick row"):
        build_structure_scene_plan_from_files(
            None,
            _write_json(
                tmp_path / "duplicate.json",
                {"1": BOTTOM_ROW, "2": dict(BOTTOM_ROW)},
            ),
            document,
        )

    with pytest.raises(StructureSceneError, match="cannot read start structure"):
        build_structure_scene_plan_from_files(tmp_path / "absent.json", GOAL, document)

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(StructureSceneError, match="is not valid JSON"):
        build_structure_scene_plan_from_files(broken, GOAL, document)

    with pytest.raises(StructureSceneError, match="is not a goal structure entry"):
        build_structure_scene_plan_from_files(
            _write_json(
                tmp_path / "foreign.json", {"1": dict(BOTTOM_ROW, color="Green")}
            ),
            GOAL,
            document,
        )


def test_invalid_operation_plans_and_foreign_ids_fail_closed() -> None:
    """The external plan is validated and every referenced id must be a goal."""
    with pytest.raises(StructureSceneError, match="robot-operation-plan/v1"):
        build_structure_scene_plan_from_files(START, GOAL, [])

    bad_schema = dict(
        _read_json(OPERATION_PLAN), schema="robot-operation-plan/v2"
    )
    with pytest.raises(StructureSceneError, match="robot operation plan is invalid"):
        build_structure_scene_plan_from_files(START, GOAL, bad_schema)

    foreign = _read_json(OPERATION_PLAN)
    for operation in foreign["operations"]:
        if operation["skill"] == "WAIT":
            operation["parameters"]["observation"]["physical_instance_id"] = (
                "brick-foreign"
            )
    with pytest.raises(StructureSceneError, match="remain unproven"):
        build_structure_scene_plan_from_files(START, GOAL, foreign)


def test_a_task_carries_at_most_one_supporting_robot(tmp_path: Path) -> None:
    """A PLACE_DOWN with two non-executors cannot be carried or silently cut."""
    plan_document = _place_down_plan(
        [(_instance_id(TOP_ROW), _instance_id(BOTTOM_ROW))],
        robots=(ROBOT, "piper_1", "piper_2"),
    )
    operation = plan_document["operations"][0]
    operation["participants"] = [ROBOT, "piper_1", "piper_2"]
    operation["role_bindings"] = {
        "executor": ROBOT,
        "support_1": "piper_1",
        "support_2": "piper_2",
    }
    operation["resources"] = [
        f"robot:{ROBOT}",
        "robot:piper_1",
        "robot:piper_2",
        *operation["resources"][1:],
    ]

    with pytest.raises(StructureSceneError, match="more than one supporting robot"):
        build_structure_scene_plan_from_files(
            _write_json(tmp_path / "start.json", {"1": BOTTOM_ROW}),
            _write_json(tmp_path / "goal.json", {"1": BOTTOM_ROW, "2": TOP_ROW}),
            plan_document,
        )


# ---------------------------------------------------------------------------
# Legacy directory-based entry point (host bridge --structure-dir)
# ---------------------------------------------------------------------------


def _pick_plan(instance_id: str) -> dict:
    """Build a minimal valid ``robot-operation-plan/v1`` PICK-only document.

    Args:
        instance_id: The picked physical instance id.

    Returns:
        The operation plan document.
    """
    return {
        "schema": "robot-operation-plan/v1",
        "plan_id": "robot-operation-plan-synthetic-pick",
        "source_plan_id": "assembly-plan-synthetic",
        "robots": [ROBOT],
        "operations": [
            {
                "operation_id": "pick-0",
                "source_task_id": "assemble-other",
                "skill": "PICK",
                "executor_robot_id": ROBOT,
                "participants": [ROBOT],
                "role_bindings": {"executor": ROBOT},
                "target": {"physical_instance_id": instance_id},
                "parameters": {},
                "preconditions": [],
                "effects": [],
                "resources": [f"robot:{ROBOT}", f"part:{instance_id}"],
                "evidence_requirements": [],
            }
        ],
        "dependencies": [],
        "execution_layers": [["pick-0"]],
        "planned_schedule": [[{"operation_id": "pick-0", "robot_id": ROBOT}]],
        "evidence_requirements": [],
    }


def test_legacy_directory_entry_keeps_its_call_semantics() -> None:
    """The host bridge's two-argument call still binds example1 unchanged."""
    plan, metadata = build_structure_scene_plan(
        TASK_DIR, _read_json(OPERATION_PLAN)
    )
    goal_bytes = GOAL.read_bytes()
    start_bytes = START.read_bytes()

    assert metadata == {
        "structure_dir": str(TASK_DIR),
        "goal_path": str(GOAL),
        "start_path": str(START),
        "goal_sha256": sha256(goal_bytes).hexdigest(),
        "start_sha256": sha256(start_bytes).hexdigest(),
        "goal_entries": 2,
        "initial_entries": 1,
    }
    assert plan.plan_id == "robot-operation-structure-" + sha256(
        goal_bytes
    ).hexdigest()[:24]
    assert plan.source_goal_sha256 == sha256(goal_bytes).hexdigest()
    assert plan.source_start_sha256 == sha256(start_bytes).hexdigest()
    assert plan.robots == ("piper_0", "piper_1")
    assert set(plan.goal_parts) == {BOTTOM, TOP}
    assert plan.goal_parts[BOTTOM].source_goal_key == "1"
    assert plan.goal_parts[BOTTOM].source_start_key == "1"
    assert plan.goal_parts[TOP].source_goal_key == "2"
    assert plan.goal_parts[TOP].source_start_key is None
    assert plan.initially_assembled == frozenset({BOTTOM})
    assert set(plan.tasks) == {TASK_ID}
    assert plan.tasks[TASK_ID].physical_instance_id == TOP
    assert plan.tasks[TASK_ID].primary_robot == "piper_0"
    assert plan.raw is metadata

    # For the frozen example1 inputs the legacy traversal-order assignment and
    # the strict evidence binding agree, so the host path is unchanged.
    strict_plan, _ = build_structure_scene_plan_from_files(
        START, GOAL, _read_json(OPERATION_PLAN)
    )
    assert set(plan.goal_parts) == set(strict_plan.goal_parts)
    assert plan.initially_assembled == strict_plan.initially_assembled


def test_legacy_directory_entry_falls_back_to_structure_json(
    tmp_path: Path,
) -> None:
    """``structure.json`` and a missing start file keep their old meaning."""
    directory = tmp_path / "task"
    directory.mkdir()
    _write_json(directory / "structure.json", {"1": BOTTOM_ROW})

    plan, metadata = build_structure_scene_plan(
        directory, _pick_plan(BOTTOM)
    )

    assert metadata["goal_path"] == str(directory / "structure.json")
    assert metadata["start_path"] is None
    assert metadata["start_sha256"] is None
    assert metadata["initial_entries"] == 0
    assert plan.initially_assembled == frozenset()
    assert set(plan.goal_parts) == {BOTTOM}
    assert plan.goal_parts[BOTTOM].source_start_key is None
    assert set(plan.tasks) == {"assemble-4bbba49bec24496bb21e"}
    task = plan.tasks["assemble-4bbba49bec24496bb21e"]
    assert task.physical_instance_id == BOTTOM
    assert task.primary_robot == ROBOT


def test_legacy_directory_entry_stays_lenient_and_fails_closed(
    tmp_path: Path,
) -> None:
    """The legacy entry keeps its original leniency and error messages."""
    directory = tmp_path / "task"
    directory.mkdir()
    _write_json(directory / "structure_start.json", {"1": BOTTOM_ROW})
    _write_json(
        directory / "structure_goal.json",
        {"1": dict(BOTTOM_ROW, studs=4), "2": TOP_ROW},
    )
    document = _read_json(OPERATION_PLAN)

    plan, _ = build_structure_scene_plan(directory, document)
    assert set(plan.goal_parts) == {BOTTOM, TOP}

    with pytest.raises(StructureSceneError, match="must carry exactly"):
        build_structure_scene_plan_from_files(
            directory / "structure_start.json",
            directory / "structure_goal.json",
            document,
        )

    with pytest.raises(StructureSceneError, match="no structure_goal.json"):
        build_structure_scene_plan(tmp_path / "absent", document)

    with pytest.raises(StructureSceneError, match="has no robots"):
        build_structure_scene_plan(
            TASK_DIR, {"robots": [], "operations": []}
        )

    with pytest.raises(StructureSceneError, match="does not identify every"):
        build_structure_scene_plan(TASK_DIR, {"robots": [ROBOT], "operations": []})

    goal_only = tmp_path / "goal-only"
    goal_only.mkdir()
    _write_json(goal_only / "structure.json", {"1": BOTTOM_ROW, "2": TOP_ROW})
    with pytest.raises(StructureSceneError, match="cannot bind structure entries"):
        build_structure_scene_plan(goal_only, document)
