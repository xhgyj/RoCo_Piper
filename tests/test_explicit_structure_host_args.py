"""Offline tests for the explicit structure-files host bridge arguments.

These tests pin the additive scene-loading-only ``--structure-goal`` /
``--structure-start`` path end to end without booting Kit, Isaac, rclpy,
ROS, or Docker:

* the parent and Kit-child bridge parsers recognize the flags, forward them
  through ``_argv_from_args`` and reject inconsistent combinations
  (start without goal, goal with ``--structure-dir``, explicit paths outside
  scene mode);
* the ``run_bridge`` scene branch parses the external
  ``robot-operation-plan/v1`` strictly and binds the explicit goal (and
  optional start) files into the internal topology carrier with
  ``build_structure_scene_plan_from_files`` BEFORE ``bootstrap_runtime_host``,
  so any invalid input fails before a scene can load;
* the existing ``--structure-dir`` and pure-STAP branches stay unchanged;
* ``MultiRobotSessionConfig`` carries the optional
  ``structure_goal_path``/``structure_start_path`` fields with their
  validation and forwards them verbatim in ``_host_argv`` (the start flag
  only when it is not ``None``; nothing is inferred from a directory).

Nothing here dispatches an operation or authorizes motion: the carrier is a
topology input to scene loading alone.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rocobrick.plan.structure_scene import (
    StructureSceneError,
    build_structure_scene_plan_from_files,
)
from rocobrick.robot_operation_plan import (
    RobotOperationPlanError,
    parse_robot_operation_plan,
)
from rocobrick.runtime_session import (
    MultiRobotIsaacMoveItSession,
    MultiRobotSessionConfig,
)

ROOT = Path(__file__).parents[1]
BRIDGE_PATH = ROOT / "isaac" / "p05_host_bridge.py"

# Frozen repository fixtures (read-only provenance inputs): the example1
# robot-operation-plan/v1, its explicit structure files and scene manifest
# bind together exactly (pinned by tests/test_structure_scene.py).
OPERATION_PLAN = (
    ROOT / "tests/fixtures/robot_operation_plans/example1-robot-operation-plan.json"
)
TASK_DIR = ROOT / "tasks/type1/example1"
START = TASK_DIR / "structure_start.json"
GOAL = TASK_DIR / "structure_goal.json"
MANIFEST = ROOT / "artifacts/scene/example1-scene-manifest.json"

CARRIER_PLAN_ID_PREFIX = "robot-operation-structure-"

# Minimal valid scene-mode base: the explicit flags are legal only here.
SCENE_BASE = (
    "--scene-manifest", "/tmp/scene-manifest.json",
    "--scene-snapshot-out", "/tmp/scene-snapshot.json",
    "--host-instance-id", "repro-001",
    "--plan", "/tmp/plan.json",
    "--config", "/tmp/config.json",
)


@pytest.fixture(scope="module")
def bridge_module():
    """Import the bridge module without booting isaacsim, omni, or rclpy.

    Returns:
        The imported bridge module.
    """
    spec = importlib.util.spec_from_file_location(
        "p05_host_bridge_explicit_structure", BRIDGE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bridge_source() -> str:
    """Return the raw bridge script text.

    Returns:
        The bridge script source.
    """
    return BRIDGE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def run_bridge_body(bridge_source: str) -> str:
    """Return the ``run_bridge`` body slice of the bridge source.

    Args:
        bridge_source: The raw bridge script text.

    Returns:
        The source between the ``run_bridge`` and ``write_evidence``
        definitions.
    """
    start = bridge_source.index("def run_bridge")
    end = bridge_source.index("def write_evidence")
    return bridge_source[start:end]


def _flag_values(argv: list[str], flag: str) -> list[str]:
    """Return every value passed for one CLI flag.

    Args:
        argv: The command line to scan.
        flag: The exact flag token (for example ``--structure-goal``).

    Returns:
        The values following each occurrence of the flag.
    """
    return [
        argv[index + 1]
        for index, token in enumerate(argv)
        if token == flag and index + 1 < len(argv)
    ]


def _session(tmp_path: Path, **config_overrides) -> MultiRobotIsaacMoveItSession:
    """Build one offline session whose config carries the given overrides.

    Args:
        tmp_path: Temporary directory for the session and its inputs.
        **config_overrides: Overrides applied to the base configuration.

    Returns:
        The session (never started; no process, container or motion).
    """

    class _NullRunner:
        """Stand-in runner; ``_host_argv`` never launches anything."""

    kwargs = {
        "repo_root": ROOT,
        "robot_ids": ("piper_0", "piper_1"),
        "scene_manifest": tmp_path / "scene-manifest.json",
        "scene_snapshot_out": tmp_path / "scene-snapshot.json",
        "host_instance_id": "host-explicit-structure-test",
        "plan_path": OPERATION_PLAN,
        "config_path": tmp_path / "user-config.json",
    }
    kwargs.update(config_overrides)
    config = MultiRobotSessionConfig(**kwargs)
    return MultiRobotIsaacMoveItSession(
        config,
        runner=_NullRunner(),
        session_dir=tmp_path / "session",
    )


# ------------------------------------------------------------ bridge parser


def test_explicit_structure_flags_parse_as_paths(bridge_module) -> None:
    """Scene mode accepts an explicit goal with an optional explicit start."""
    args = bridge_module.parse_args(
        [*SCENE_BASE, "--structure-goal", "/tmp/goal.json"]
    )
    assert args.structure_goal == Path("/tmp/goal.json")
    assert args.structure_start is None

    both = bridge_module.parse_args(
        [
            *SCENE_BASE,
            "--structure-goal", "/tmp/goal.json",
            "--structure-start", "/tmp/start.json",
        ]
    )
    assert both.structure_goal == Path("/tmp/goal.json")
    assert both.structure_start == Path("/tmp/start.json")


def test_explicit_structure_flags_default_to_none(bridge_module) -> None:
    """Omitted flags stay ``None``: no default path is ever inferred."""
    args = bridge_module.parse_args([*SCENE_BASE])
    assert args.structure_goal is None
    assert args.structure_start is None


def test_explicit_flags_round_trip_through_kit_child_argv(bridge_module) -> None:
    """The rebuilt Kit-child argv preserves (or omits) the explicit flags."""
    explicit = bridge_module.parse_args(
        [
            *SCENE_BASE,
            "--structure-goal", "/tmp/goal.json",
            "--structure-start", "/tmp/start.json",
        ]
    )
    rebuilt = bridge_module.parse_args(
        bridge_module._argv_from_args(explicit)
    )
    assert rebuilt.structure_goal == explicit.structure_goal
    assert rebuilt.structure_start == explicit.structure_start

    goal_only = bridge_module.parse_args(
        [*SCENE_BASE, "--structure-goal", "/tmp/goal.json"]
    )
    argv = bridge_module._argv_from_args(goal_only)
    assert _flag_values(argv, "--structure-goal") == ["/tmp/goal.json"]
    assert "--structure-start" not in argv

    minimal = bridge_module.parse_args(["--evidence-out", "/tmp/e.json"])
    argv = bridge_module._argv_from_args(minimal)
    assert "--structure-goal" not in argv
    assert "--structure-start" not in argv


def test_start_without_goal_is_rejected(bridge_module) -> None:
    """An explicit start without an explicit goal fails the parent parser."""
    with pytest.raises(SystemExit):
        bridge_module.parse_args(
            [*SCENE_BASE, "--structure-start", "/tmp/start.json"]
        )


def test_goal_with_structure_dir_is_rejected(bridge_module) -> None:
    """An explicit goal cannot be combined with the directory mode."""
    with pytest.raises(SystemExit):
        bridge_module.parse_args(
            [
                *SCENE_BASE,
                "--structure-dir", "/tmp/structure",
                "--structure-goal", "/tmp/goal.json",
            ]
        )


def test_explicit_paths_outside_scene_mode_are_rejected(bridge_module) -> None:
    """The explicit flags are scene-mode only and fail closed elsewhere."""
    with pytest.raises(SystemExit):
        bridge_module.parse_args(
            ["--evidence-out", "/tmp/e.json", "--structure-goal", "/tmp/g.json"]
        )
    with pytest.raises(SystemExit):
        bridge_module.parse_args(
            ["--evidence-out", "/tmp/e.json", "--structure-start", "/tmp/s.json"]
        )


# ---------------------------------------------------- run_bridge scene branch


def test_explicit_branch_strict_parse_and_carrier_build_precede_bootstrap(
    run_bridge_body: str,
) -> None:
    """The explicit-files branch fails closed before the scene can load.

    Inside the scene branch the external plan is parsed strictly first, then
    the explicit files build the carrier through
    ``build_structure_scene_plan_from_files`` (with the scene manifest), and
    only then may ``bootstrap_runtime_host`` run.
    """
    assert "elif args.structure_goal is not None:" in run_bridge_body
    strict = run_bridge_body.index("parse_robot_operation_plan(plan_document)")
    build = run_bridge_body.index("build_structure_scene_plan_from_files(")
    bootstrap = run_bridge_body.index("bootstrap_runtime_host(")
    assert strict < build < bootstrap
    call = run_bridge_body[build:bootstrap]
    assert "args.structure_start," in call
    assert "args.structure_goal," in call
    assert "plan_document," in call
    assert "manifest=(" in call
    assert "args.scene_manifest.is_file()" in call
    assert "else None" in call
    assert 'evidence["scene_runtime"]["structure"] = structure_evidence' in call


def test_existing_stap_and_structure_dir_branches_unchanged(
    run_bridge_body: str,
) -> None:
    """The pre-existing directory and pure-STAP branches keep their shape."""
    directory = run_bridge_body.index("if args.structure_dir is not None:")
    explicit = run_bridge_body.index("elif args.structure_goal is not None:")
    fallback = run_bridge_body.index("plan = parse_stap_plan(plan_document)")
    assert directory < explicit < fallback
    assert (
        "build_structure_scene_plan(args.structure_dir, plan_document)"
        in run_bridge_body
    )
    assert run_bridge_body.count("parse_stap_plan(plan_document)") == 1
    assert (
        run_bridge_body.count(
            'evidence["scene_runtime"]["structure"] = structure_evidence'
        )
        == 2
    )


def test_explicit_branch_adds_no_dispatch_or_motion_path(
    run_bridge_body: str,
) -> None:
    """The branch loads a scene topology and never touches a motion path."""
    start = run_bridge_body.index("elif args.structure_goal is not None:")
    end = run_bridge_body.index("else:", start)
    branch = run_bridge_body[start:end]
    for token in (
        "create_publisher",
        "create_subscription",
        "create_service",
        "run_motion_operation",
        "Executor",
        "action_client",
        "send_goal",
        "TrajectoryExecution",
    ):
        assert token not in branch, token


# ------------------------------------------- explicit-files branch behaviour


def test_invalid_operation_plan_fails_before_carrier_build() -> None:
    """The strict plan parse rejects a non-plan before any binder runs."""
    document = {"schema": "robot-operation-plan/v1", "plan_id": 1}
    with pytest.raises(RobotOperationPlanError):
        parse_robot_operation_plan(document)
    # The binder wraps the same strict parse and never repairs the document.
    with pytest.raises(StructureSceneError, match="robot operation plan is invalid"):
        build_structure_scene_plan_from_files(START, GOAL, document)


def test_invalid_goal_file_fails_before_bootstrap(tmp_path: Path) -> None:
    """An unreadable or malformed explicit goal fails the carrier build."""
    goal = tmp_path / "goal.json"
    goal.write_text("{not json", encoding="utf-8")
    document = json.loads(OPERATION_PLAN.read_text(encoding="utf-8"))
    with pytest.raises(StructureSceneError):
        build_structure_scene_plan_from_files(None, goal, document)

    empty_goal = tmp_path / "empty-goal.json"
    empty_goal.write_text("{}", encoding="utf-8")
    with pytest.raises(StructureSceneError, match="goal structure must not be empty"):
        build_structure_scene_plan_from_files(START, empty_goal, document)


def test_explicit_files_build_carrier_and_record_structure_evidence(
    bridge_module, tmp_path: Path
) -> None:
    """The branch's exact call sequence binds the carrier and its evidence.

    The frozen example1 inputs (plan, explicit start, explicit goal and scene
    manifest) are resolved exactly as ``run_bridge`` does: strict plan parse,
    ``build_structure_scene_plan_from_files`` with the scene manifest, and the
    metadata recorded under ``scene_runtime.structure``. The evidence stays
    JSON-safe and the carrier authorizes no motion by itself.
    """
    args = bridge_module.parse_args([
        "--scene-manifest", str(MANIFEST),
        "--scene-snapshot-out", str(tmp_path / "scene-snapshot.json"),
        "--host-instance-id", "repro-001",
        "--plan", str(OPERATION_PLAN),
        "--config", str(tmp_path / "config.json"),
        "--structure-goal", str(GOAL),
        "--structure-start", str(START),
    ])
    evidence = bridge_module.initial_evidence(args)
    plan_bytes = args.plan.read_bytes()
    evidence["scene_runtime"]["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    plan_document = json.loads(plan_bytes)
    parse_robot_operation_plan(plan_document)
    carrier, structure_evidence = build_structure_scene_plan_from_files(
        args.structure_start,
        args.structure_goal,
        plan_document,
        manifest=args.scene_manifest,
    )
    evidence["scene_runtime"]["structure"] = structure_evidence

    assert structure_evidence["plan_id"].startswith(CARRIER_PLAN_ID_PREFIX)
    assert structure_evidence["goal_path"] == str(GOAL)
    assert structure_evidence["start_path"] == str(START)
    assert structure_evidence["manifest_path"] == str(MANIFEST)
    assert structure_evidence["manifest_entries"] == 2
    assert structure_evidence["robots"] == ["piper_0", "piper_1"]
    assert carrier.robots == ("piper_0", "piper_1")
    assert carrier.tasks and carrier.goal_parts
    assert json.dumps(evidence, sort_keys=True)


def test_start_none_means_empty_initial_without_directory_inference() -> None:
    """A ``None`` start is EMPTY initial structure, never a directory default.

    The goal file lives in a task directory that also contains
    ``structure_start.json``; loading it implicitly would succeed, so the
    required failure proves the sibling file was not inferred.
    """
    assert START.is_file()
    document = json.loads(OPERATION_PLAN.read_text(encoding="utf-8"))
    with pytest.raises(
        StructureSceneError, match="not placed by any PLACE_DOWN"
    ):
        build_structure_scene_plan_from_files(None, GOAL, document)


# ------------------------------------------------------- session config argv


def test_config_explicit_structure_fields_default_to_none(tmp_path: Path) -> None:
    """The new fields exist, default to ``None`` and are never inferred."""
    names = {field.name for field in dataclasses.fields(MultiRobotSessionConfig)}
    assert {"structure_goal_path", "structure_start_path"} <= names
    session = _session(tmp_path, structure_dir=tmp_path / "structure")
    assert session.config.structure_dir == tmp_path / "structure"
    assert session.config.structure_goal_path is None
    assert session.config.structure_start_path is None


def test_config_rejects_non_path_structure_fields(tmp_path: Path) -> None:
    """Both explicit structure fields must be ``Path`` or ``None``."""
    with pytest.raises(ValueError, match="structure_goal_path must be a Path"):
        _session(tmp_path, structure_goal_path="/tmp/goal.json")
    goal = tmp_path / "goal.json"
    with pytest.raises(ValueError, match="structure_start_path must be a Path"):
        _session(tmp_path, structure_goal_path=goal, structure_start_path="/tmp/s.json")


def test_config_rejects_start_without_goal(tmp_path: Path) -> None:
    """A start without a goal is a configuration error."""
    with pytest.raises(ValueError, match="structure_start_path requires"):
        _session(tmp_path, structure_start_path=tmp_path / "start.json")


def test_config_rejects_goal_with_structure_dir(tmp_path: Path) -> None:
    """The explicit goal cannot be combined with the legacy directory."""
    with pytest.raises(ValueError, match="cannot be combined with structure_dir"):
        _session(
            tmp_path,
            structure_dir=tmp_path / "structure",
            structure_goal_path=tmp_path / "goal.json",
        )


def test_config_accepts_explicit_goal_and_start_paths(tmp_path: Path) -> None:
    """A goal with or without a start is accepted and stored verbatim."""
    goal = tmp_path / "goal.json"
    start = tmp_path / "start.json"
    goal_only = _session(tmp_path, structure_goal_path=goal)
    assert goal_only.config.structure_goal_path == goal
    assert goal_only.config.structure_start_path is None
    both = _session(
        tmp_path, structure_goal_path=goal, structure_start_path=start
    )
    assert both.config.structure_goal_path == goal
    assert both.config.structure_start_path == start


def test_host_argv_forwards_explicit_structure_flags(
    bridge_module, tmp_path: Path
) -> None:
    """The session argv carries both flags and parses in the bridge parser."""
    goal = tmp_path / "goal.json"
    start = tmp_path / "start.json"
    argv = _session(
        tmp_path, structure_goal_path=goal, structure_start_path=start
    )._host_argv()
    assert _flag_values(argv, "--structure-goal") == [str(goal)]
    assert _flag_values(argv, "--structure-start") == [str(start)]
    args = bridge_module.parse_args(argv[4:])
    assert args.structure_goal == goal
    assert args.structure_start == start


def test_host_argv_omits_start_flag_when_start_is_none(
    bridge_module, tmp_path: Path
) -> None:
    """The start flag is forwarded only when the start is not ``None``."""
    goal = tmp_path / "goal.json"
    argv = _session(tmp_path, structure_goal_path=goal)._host_argv()
    assert _flag_values(argv, "--structure-goal") == [str(goal)]
    assert "--structure-start" not in argv
    args = bridge_module.parse_args(argv[4:])
    assert args.structure_goal == goal
    assert args.structure_start is None


def test_host_argv_has_no_explicit_structure_flags_by_default(
    bridge_module, tmp_path: Path
) -> None:
    """Without the explicit fields no structure flag is forwarded at all."""
    argv = _session(tmp_path)._host_argv()
    assert "--structure-goal" not in argv
    assert "--structure-start" not in argv
    assert "--structure-dir" not in argv
    args = bridge_module.parse_args(argv[4:])
    assert args.structure_goal is None
    assert args.structure_start is None
    assert args.structure_dir is None
