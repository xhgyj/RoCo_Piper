"""Offline tests for the structure-only and direct robot-plan paths.

The tests drive ``piper-run`` with ``--robot-operation-plan`` and explicit
structure inputs. Without the complete runtime option group this is a
structure-only preflight: no runtime session or motion is allowed. With the
complete group, plans containing only currently implemented direct skills may
enter the real dispatch seam; this suite uses a guarded fake session to verify
that routing without STAP is intentional and auditable.

Every run goes through the documented session construction seam
(``piper_run._create_session``), which is replaced by an autouse guard that
records and fails on any invocation, so a regression that tries to dispatch
from the structure-only path cannot start anything real. All mutable inputs
are temporary copies; an autouse guard also re-hashes the repository
provenance inputs after every test so no fixture byte can change.

Covered acceptance properties:

* the ``--structure-goal``/``--structure-start`` options are exposed and a
  missing, directory, or empty goal is rejected before any session exists,
  with ``events.jsonl`` evidence left behind;
* a valid example1 structure preflight (goal, real start, scene manifest,
  frozen external plan) is audited, non-success, and dispatches nothing, and
  every supplied input is captured verbatim with its raw SHA-256;
* an omitted or literal-empty ``--structure-start`` means EMPTY start, is
  audited as such, and fails closed for the frozen example1 plan instead of
  silently defaulting to the repository start file;
* the preflight inputs record the start choice explicitly
  (``start_provided``/``start_empty``) and never invent a SHA-256 for an
  absent start;
* two supplied inputs that resolve to the same file (operation plan,
  structure goal, supplied structure start, scene manifest or config,
  including through a symlink or a ``..`` spelling) are ``INVALID_INPUT``
  before the binder runs, with the full audit trail and no session;
* a foreign or otherwise mismatched scene manifest is ``INVALID_INPUT``;
* an input mutated after capture is detected by the post-run re-hash;
* every early exit keeps ``events.jsonl``, ``preflight.json`` and
  ``result.json``;
* a WAIT-only external plan can never produce a false ``EXECUTED``.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator, Mapping, Sequence
from hashlib import sha256
from io import StringIO
from pathlib import Path

import pytest

from rocobrick import piper_run
from rocobrick.piper_run import EXIT_CODES, RunStatus, main
from rocobrick.plan import structure_scene

ROOT = Path(__file__).parents[1]
TASK_DIR = ROOT / "tasks/type1/example1"
TASK_START = TASK_DIR / "structure_start.json"
TASK_GOAL = TASK_DIR / "structure_goal.json"
SCENE_MANIFEST = ROOT / "artifacts/scene/example1-scene-manifest.json"
OPERATION_PLAN = (
    ROOT / "tests/fixtures/robot_operation_plans/example1-robot-operation-plan.json"
)

# Every repository input this suite must leave byte-identical.
PROVENANCE_INPUTS = (TASK_GOAL, TASK_START, SCENE_MANIFEST, OPERATION_PLAN)

# The terminal statuses that mean "this invocation was rejected".
REJECTION_STATUSES = frozenset({"INVALID_INPUT", "BINDING_REJECTED"})

# Sentinel distinguishing "flag omitted" from an explicit empty string.
_OMITTED = object()


# ---------------------------------------------------------------------------
# Guards and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def session_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[tuple[object, ...]]]:
    """Fail every invocation of the runtime session constructor.

    Yields:
        The (expected to stay empty) list of attempted constructions.
    """
    invocations: list[tuple[object, ...]] = []

    def guard(*args: object, **kwargs: object) -> object:
        invocations.append((args, kwargs))
        raise AssertionError(
            "the structure-only preflight must never create a runtime session"
        )

    monkeypatch.setattr(piper_run, "_create_session", guard)
    yield invocations


@pytest.fixture(autouse=True)
def provenance_guard() -> Iterator[None]:
    """Fail the test if any repository provenance input changes on disk.

    Yields:
        Nothing; the assertion runs after the test body.
    """
    before = _digests(PROVENANCE_INPUTS)
    yield
    assert _digests(PROVENANCE_INPUTS) == before


def _read_json(path: Path) -> dict:
    """Load one JSON document from disk.

    Args:
        path: The file to read.

    Returns:
        The decoded document.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _digests(paths: Sequence[Path]) -> dict[str, str]:
    """Hash every given file from disk.

    Args:
        paths: The files to hash.

    Returns:
        Mapping of absolute path string to raw SHA-256 hex digest.
    """
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}


def _walk_items(value: object) -> Iterator[tuple[str, object]]:
    """Yield every mapping entry of one nested JSON value, depth first.

    Args:
        value: The JSON value to walk.

    Yields:
        Every ``(key, value)`` pair found in mappings and sequences.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_items(item)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            yield from _walk_items(item)


def _values_for_key(document: object, key: str) -> list[object]:
    """Collect every value stored under one key in a nested JSON value.

    Args:
        document: The JSON value to search.
        key: The exact mapping key to collect.

    Returns:
        Every value found for that key, in traversal order.
    """
    return [value for name, value in _walk_items(document) if name == key]


def _invoke(argv: Sequence[str]) -> tuple[int, dict, str]:
    """Run ``piper_run.main`` with captured stdout and stderr.

    Args:
        argv: The command-line arguments for ``main``.

    Returns:
        The exit code, the single parsed stdout JSON line, and stderr.

    Raises:
        AssertionError: If stdout does not carry exactly one JSON line.
    """
    stdout, stderr = StringIO(), StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(list(argv))
    lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1, stdout.getvalue()
    return code, json.loads(lines[0]), stderr.getvalue()


def _structure_argv(
    run_dir: Path,
    *,
    goal: Path | str,
    start: object = _OMITTED,
    plan: Path = OPERATION_PLAN,
    manifest: Path | None = SCENE_MANIFEST,
    preflight_only: bool = True,
) -> list[str]:
    """Build a structure-only preflight invocation.

    Args:
        run_dir: Fresh evidence directory to create.
        goal: ``--structure-goal`` value; a path or a raw string.
        start: ``--structure-start`` value, or the omit sentinel.
        plan: External ``robot-operation-plan/v1`` file.
        manifest: Scene manifest file, or ``None`` to omit the option.
        preflight_only: Whether to request the explicit no-runtime preflight.

    Returns:
        The argv list for ``piper_run.main``.
    """
    argv = [
        "--robot-operation-plan",
        str(plan),
        "--backend",
        "isaac",
        "--run-dir",
        str(run_dir),
        "--structure-goal",
        str(goal),
    ]
    if preflight_only:
        argv.append("--preflight-only")
    if start is not _OMITTED:
        assert isinstance(start, (Path, str)), start
        argv += ["--structure-start", str(start)]
    if manifest is not None:
        argv += ["--scene-manifest", str(manifest)]
    return argv


def _assert_no_session(invocations: Sequence[tuple[object, ...]]) -> None:
    """Assert the runtime session constructor was never invoked.

    Args:
        invocations: The recorded session constructions.
    """
    assert list(invocations) == [], (
        "the structure-only preflight created a runtime session"
    )


def _assert_rejected(code: int, line: Mapping[str, object]) -> None:
    """Assert one invocation was rejected without claiming success.

    Args:
        code: The process exit code.
        line: The parsed stdout JSON line.
    """
    assert code != 0, line
    assert line.get("status") in REJECTION_STATUSES, line
    assert line.get("status") != "EXECUTED", line
    assert line.get("passed") is False, line


def _assert_early_exit_evidence(line: Mapping[str, object], run_dir: Path) -> Path:
    """Assert one early exit left the audit trail it owes.

    Args:
        line: The parsed stdout JSON line.
        run_dir: The requested run directory.

    Returns:
        The evidence directory of the run.

    Raises:
        AssertionError: If the result path, events, or documents are missing.
    """
    result_path = line.get("result_path")
    assert isinstance(result_path, str) and result_path, (
        f"the run left no result document: {dict(line)}"
    )
    path = Path(result_path)
    assert path.name == "result.json", path
    assert path.is_file(), path
    assert path.is_relative_to(run_dir), path
    evidence_dir = path.parent
    events_path = evidence_dir / "events.jsonl"
    assert events_path.is_file(), (
        f"the early exit left no events.jsonl in {evidence_dir}"
    )
    events = [
        json.loads(raw)
        for raw in events_path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    assert events, "events.jsonl is empty"
    preflight_path = evidence_dir / "preflight.json"
    assert preflight_path.is_file(), preflight_path
    preflight = _read_json(preflight_path)
    assert preflight.get("status") == line.get("status"), preflight
    result = _read_json(path)
    assert result.get("status") == line.get("status"), result
    assert result.get("passed") is False, result
    return evidence_dir


def _json_documents(evidence_dir: Path) -> list[object]:
    """Load every top-level JSON document of one evidence directory.

    Args:
        evidence_dir: The evidence directory to scan.

    Returns:
        The decoded documents in file-name order.
    """
    return [
        _read_json(path)
        for path in sorted(evidence_dir.glob("*.json"))
        if path.is_file()
    ]


def _assert_inputs_captured(
    evidence_dir: Path, supplied: Mapping[str, Path]
) -> None:
    """Assert every supplied input is copied verbatim and raw-SHA audited.

    Args:
        evidence_dir: The evidence directory of the run.
        supplied: Label to source path of each supplied input.
    """
    copies = {
        path.name: path.read_bytes()
        for path in evidence_dir.iterdir()
        if path.is_file()
    }
    audited_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(evidence_dir.glob("*.json"))
        if path.is_file()
    )
    for label, source in supplied.items():
        raw = source.read_bytes()
        digest = sha256(raw).hexdigest()
        matching = [
            name for name, data in copies.items() if data == raw and raw
        ]
        assert matching, f"the {label} bytes were not captured verbatim"
        assert digest in audited_text, (
            f"the raw SHA-256 of the {label} is not audited"
        )
    for metadata_path in sorted(evidence_dir.glob("*.metadata.json")):
        payload = _read_json(metadata_path)
        sibling = evidence_dir / metadata_path.name.removesuffix(
            ".metadata.json"
        )
        assert sibling.is_file(), sibling
        raw = sibling.read_bytes()
        if "sha256" in payload:
            assert payload["sha256"] == sha256(raw).hexdigest(), metadata_path
        if "bytes" in payload:
            assert payload["bytes"] == len(raw), metadata_path


_STRUCTURE_START_KEYS = frozenset({
    "start",
    "start_path",
    "start_sha256",
    "start_bytes",
    "start_entries",
    "start_canonical_sha256",
    "task_start",
})


def _is_structure_start_key(key: str) -> bool:
    """Return whether one mapping key audits the structure start.

    Deliberately narrow: timestamps such as ``started_at`` or retry counters
    such as ``restart_count`` are not start-audit keys.

    Args:
        key: The mapping key to classify.

    Returns:
        True only for keys that name the structure start input.
    """
    lowered = key.lower()
    return "structure_start" in lowered or lowered in _STRUCTURE_START_KEYS


def _assert_empty_start_audited(documents: Sequence[object]) -> None:
    """Assert the audit records the structure start as EMPTY.

    The real example1 start file must never appear: neither its path nor its
    raw digest may be recorded, and any structure-start key must carry an
    empty (``None``/``0``) value instead of a file.

    Args:
        documents: Every decoded evidence document of the run.
    """
    entries = [
        (key, value)
        for document in documents
        for key, value in _walk_items(document)
        if _is_structure_start_key(key)
    ]
    assert entries, "the audit does not record the (empty) structure start"
    real_path = str(TASK_START)
    real_sha = sha256(TASK_START.read_bytes()).hexdigest()
    for key, value in entries:
        assert value != real_path, key
        assert value != real_sha, key
        if isinstance(value, Mapping):
            for nested_key, nested in value.items():
                name = str(nested_key).lower()
                assert nested != real_path, (key, nested_key)
                assert nested != real_sha, (key, nested_key)
                if "path" in name or "sha256" in name:
                    assert nested is None, (key, nested_key, nested)
                if "bytes" in name or "entries" in name:
                    assert nested in (None, 0), (key, nested_key, nested)
        else:
            assert value is None or value is False or value == 0 or value == "", (
                key,
                value,
            )


def _assert_no_motion_claim(evidence_dir: Path) -> None:
    """Assert no evidence document claims motion was sent.

    Args:
        evidence_dir: The evidence directory of the run.
    """
    documents = _json_documents(evidence_dir)
    claims = [
        value
        for document in documents
        for value in _values_for_key(document, "motion_sent")
    ]
    assert all(claim is not True for claim in claims), claims


def _structure_workspace(tmp_path: Path) -> dict[str, Path]:
    """Copy the four structure preflight inputs into a temporary directory.

    Args:
        tmp_path: Temporary directory root.

    Returns:
        The ``goal``/``start``/``manifest``/``plan`` temporary copies.
    """
    destination = tmp_path / "inputs"
    destination.mkdir()
    sources = {
        "goal": TASK_GOAL,
        "start": TASK_START,
        "manifest": SCENE_MANIFEST,
        "plan": OPERATION_PLAN,
    }
    copies: dict[str, Path] = {}
    for label, source in sources.items():
        target = destination / source.name
        target.write_bytes(source.read_bytes())
        copies[label] = target
    return copies


def _wait_only_plan() -> dict:
    """Build a strict single-WAIT plan from the frozen example1 document.

    The operation document is copied verbatim from the committed
    ``robot-operation-plan/v1`` artifact, so its target, resources and
    evidence shapes stay exactly what example1 declares.

    Returns:
        A one-layer, one-operation ``robot-operation-plan/v1`` document.
    """
    document = _read_json(OPERATION_PLAN)
    operation = next(
        item for item in document["operations"] if item["skill"] == "WAIT"
    )
    return {
        "schema": "robot-operation-plan/v1",
        "plan_id": "structure-preflight-wait-only",
        "source_plan_id": document["source_plan_id"],
        "robots": list(document["robots"]),
        "operations": [operation],
        "dependencies": [],
        "execution_layers": [[operation["operation_id"]]],
        "planned_schedule": [
            [
                {
                    "operation_id": operation["operation_id"],
                    "robot_id": operation["executor_robot_id"],
                }
            ]
        ],
        "evidence_requirements": [
            {
                "operation_id": operation["operation_id"],
                "evidence": operation["evidence_requirements"][0],
            }
        ],
    }


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_structure_preflight_flags_are_exposed() -> None:
    """The parser exposes the structure-only options by their exact names."""
    options = {
        option
        for action in piper_run.build_parser()._actions
        for option in action.option_strings
    }
    assert "--structure-goal" in options
    assert "--structure-start" in options


# ---------------------------------------------------------------------------
# Missing / unreadable goal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("goal_kind", ["absent", "directory", "empty-string"])
def test_missing_structure_goal_rejects_before_session(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
    goal_kind: str,
) -> None:
    """A missing goal is rejected before any session and leaves evidence."""
    if goal_kind == "directory":
        goal: Path | str = tmp_path / "goal-dir"
        goal.mkdir()
    elif goal_kind == "empty-string":
        goal = ""
    else:
        goal = tmp_path / "missing" / "structure_goal.json"
    run_dir = tmp_path / "run"
    code, line, _ = _invoke(_structure_argv(run_dir, goal=goal))
    _assert_no_session(session_guard)
    _assert_rejected(code, line)
    _assert_early_exit_evidence(line, run_dir)


# ---------------------------------------------------------------------------
# Valid example1 structure preflight
# ---------------------------------------------------------------------------


def test_example1_structure_preflight_is_audited_and_never_dispatches(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
) -> None:
    """Valid example1 inputs produce a non-success, audited, no-motion run."""
    run_dir = tmp_path / "run"
    code, line, _ = _invoke(
        _structure_argv(
            run_dir,
            goal=TASK_GOAL,
            start=TASK_START,
            plan=OPERATION_PLAN,
            manifest=SCENE_MANIFEST,
        )
    )
    _assert_no_session(session_guard)

    # The valid inputs must validate, report PREFLIGHT_ONLY, and never dispatch.
    assert line.get("run_dir") == str(run_dir), line
    assert code == 0, line
    assert line.get("status") == "PREFLIGHT_ONLY", line
    assert line.get("passed") is False, line

    evidence_dir = _assert_early_exit_evidence(line, run_dir)
    _assert_inputs_captured(
        evidence_dir,
        {
            "goal": TASK_GOAL,
            "start": TASK_START,
            "manifest": SCENE_MANIFEST,
            "plan": OPERATION_PLAN,
        },
    )
    _assert_no_motion_claim(evidence_dir)


def test_complete_runtime_group_dispatches_direct_pick_without_stap(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete direct plan reaches the shared runtime dispatch seam.

    This test does not claim physical motion: the dispatch seam is replaced by
    a sentinel. It proves that no-STAP routing no longer stops at
    ``EXECUTION_NOT_IMPLEMENTED`` when the external plan contains only skills
    with implemented direct backends.
    """
    source = _read_json(OPERATION_PLAN)
    allowed = {
        "example1-piper1-wait-stable",
        "example1-piper0-pick",
    }
    operations = [
        operation
        for operation in source["operations"]
        if operation["operation_id"] in allowed
    ]
    operation_plan = dict(source)
    operation_plan["plan_id"] = "direct-pick-no-stap"
    operation_plan["operations"] = operations
    operation_plan["evidence_requirements"] = [
        requirement
        for requirement in source["evidence_requirements"]
        if requirement["operation_id"] in allowed
    ]
    operation_plan["dependencies"] = [
        dependency
        for dependency in source["dependencies"]
        if dependency["predecessor_operation_id"] in allowed
        and dependency["successor_operation_id"] in allowed
    ]
    operation_plan["execution_layers"] = [
        ["example1-piper1-wait-stable", "example1-piper0-pick"],
    ]
    operation_plan["planned_schedule"] = [
        [
            {"operation_id": "example1-piper1-wait-stable", "robot_id": "piper_1"},
            {"operation_id": "example1-piper0-pick", "robot_id": "piper_0"},
        ],
    ]
    plan_path = tmp_path / "direct-transit-pick.json"
    plan_path.write_text(json.dumps(operation_plan), encoding="utf-8")
    complete_start = tmp_path / "complete-structure-start.json"
    complete_start.write_bytes(TASK_GOAL.read_bytes())
    calls: list[object] = []

    def dispatch(args, record, recorder, contract, stap_plan):
        calls.append((args, record, recorder, contract, stap_plan))
        return None

    monkeypatch.setattr(piper_run, "_robot_operation_plan_phase", dispatch)
    run_dir = tmp_path / "run"
    code, line, _ = _invoke(
        _structure_argv(
            run_dir,
            goal=TASK_GOAL,
            start=complete_start,
            plan=plan_path,
            manifest=SCENE_MANIFEST,
            preflight_only=False,
        )
    )

    assert code == 0, line
    assert line["status"] == "EXECUTED", line
    assert line["passed"] is True, line
    assert len(calls) == 1
    assert calls[0][4] is None
    assert session_guard == []
    result_path = Path(str(line["result_path"]))
    assert result_path.is_file()
    result = _read_json(result_path)
    evidence_dir = result_path.parent
    assert result["status"] == "EXECUTED"
    assert result["passed"] is True
    preflight = _read_json(evidence_dir / "preflight.json")
    assert preflight["dispatch"]["implemented"] is True
    assert preflight["dispatch"]["stap_authorization"] is None
    assert preflight["checks"]["runtime_created"] is False
    assert preflight["checks"]["stap_source_plan_provided"] is False


def _direct_handover_plan(*, with_pick: bool) -> dict:
    """Build a direct HANDOVER plan without requiring a PICK predecessor.

    Args:
        with_pick: Whether to include an explicit donor PICK operation.

    Returns:
        A strict plan for HANDOVER-only or sequential PICK and HANDOVER.
    """
    source = _read_json(OPERATION_PLAN)
    pick = next(item for item in source["operations"] if item["skill"] == "PICK")
    donor = pick["executor_robot_id"]
    receiver = next(robot for robot in source["robots"] if robot != donor)
    physical_id = pick["target"]["physical_instance_id"]
    handover = {
        "operation_id": "direct-handover",
        "source_task_id": pick["source_task_id"],
        "skill": "HANDOVER",
        "executor_robot_id": receiver,
        "participants": [donor, receiver],
        "role_bindings": {"executor": receiver, "donor": donor, "receiver": receiver},
        "target": {
            "physical_instance_id": physical_id,
            "donor_robot_id": donor,
            "receiver_robot_id": receiver,
        },
        "parameters": {
            "grasp_profile_id": "handover-bottom-coordinated-v1",
            "co_hold_timeout_s": 10.0,
            "follow_samples": 4,
        },
        "preconditions": [f"holding:{physical_id}"],
        "effects": [f"holding:{physical_id}"],
        "resources": [f"robot:{donor}", f"robot:{receiver}", f"part:{physical_id}"],
        "evidence_requirements": [
            "receiver_free_space",
            "receiver_grasp_feedback",
            "co_hold_confirmed",
            "scene_transfer_committed",
            "donor_release_feedback",
            "receiver_holding_follow",
            "owner_transfer_committed",
        ],
    }
    operations = [pick, handover] if with_pick else [handover]
    return {
        "schema": source["schema"],
        "plan_id": "direct-pick-handover" if with_pick else "direct-handover-only",
        "source_plan_id": source["source_plan_id"],
        "robots": source["robots"],
        "operations": operations,
        "dependencies": [
            {
                "predecessor_operation_id": pick["operation_id"],
                "successor_operation_id": handover["operation_id"],
                "reason": "confirmed donor holding before transfer",
            }
        ] if with_pick else [],
        "execution_layers": [[item["operation_id"]] for item in operations],
        "planned_schedule": [
            [{
                "operation_id": item["operation_id"],
                "robot_id": item["executor_robot_id"],
            }]
            for item in operations
        ],
        "evidence_requirements": [
            {
                "operation_id": item["operation_id"],
                "evidence": item["evidence_requirements"][0],
            }
            for item in operations
        ],
    }


@pytest.mark.parametrize(
    "with_pick", [False, True], ids=["handover-only", "pick-handover"]
)
@pytest.mark.parametrize("preflight_only", [False, True], ids=["dispatch", "preflight"])
def test_direct_handover_routes_without_stap(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
    monkeypatch: pytest.MonkeyPatch,
    with_pick: bool,
    preflight_only: bool,
) -> None:
    """HANDOVER reaches dispatch without STAP; preflight never starts runtime.

    A sentinel failure proves routing only, not motion or physical holding.
    The real backend must still verify donor holding and all transfer evidence.
    """
    plan = tmp_path / "handover.json"
    plan.write_text(
        json.dumps(_direct_handover_plan(with_pick=with_pick)), encoding="utf-8"
    )
    # HANDOVER-only does not promise to assemble the pending example1 part.
    # This structural fixture is not evidence of donor holding at runtime.
    complete_start = tmp_path / "complete-start.json"
    complete_start.write_bytes(TASK_GOAL.read_bytes())
    calls: list[object] = []

    def dispatch(args, record, recorder, contract, stap_plan):
        calls.append(stap_plan)
        assert record["checks"]["direct_runtime_skills_supported"] is True
        assert record["checks"]["tcp_resolved"] is True
        return (
            RunStatus.RUNTIME_START_FAILED,
            "offline dispatch sentinel; no runtime created",
        )

    monkeypatch.setattr(piper_run, "_robot_operation_plan_phase", dispatch)
    run_dir = tmp_path / "run"
    code, line, _ = _invoke(
        _structure_argv(
            run_dir, goal=TASK_GOAL,
            start=TASK_START if with_pick else complete_start, plan=plan,
            preflight_only=preflight_only,
        )
    )
    expected = (
        RunStatus.PREFLIGHT_ONLY if preflight_only else RunStatus.RUNTIME_START_FAILED
    )
    assert line["status"] == expected, line
    assert code == EXIT_CODES[expected], line
    assert calls == ([] if preflight_only else [None])
    assert session_guard == []
    evidence_dir = Path(line["result_path"]).parent
    preflight = _read_json(evidence_dir / "preflight.json")
    assert preflight["robot_operation_plan"]["source_plan_bound"] is False
    captured_plan = evidence_dir / "robot-operation-plan.json"
    assert captured_plan.read_bytes() == plan.read_bytes()
    _assert_no_motion_claim(evidence_dir)


@pytest.mark.parametrize("rejection", ["bad-roles", "tcp-denied", "removed-transit"])
def test_direct_handover_preserves_pre_dispatch_gates(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
    monkeypatch: pytest.MonkeyPatch,
    rejection: str,
) -> None:
    """Adding HANDOVER does not bypass strict roles, TCP or TRANSIT rejection."""
    document = _direct_handover_plan(with_pick=False)
    if rejection == "bad-roles":
        document["operations"][0]["role_bindings"]["donor"] = "foreign-robot"
    elif rejection == "removed-transit":
        document["operations"][0]["skill"] = "TRANSIT"
    else:
        def deny_tcp(contract):
            raise piper_run.UnresolvedTcpError("offline TCP denial")

        monkeypatch.setattr(piper_run, "require_resolved_tcp", deny_tcp)
    plan = tmp_path / "handover.json"
    plan.write_text(json.dumps(document), encoding="utf-8")
    complete_start = tmp_path / "complete-start.json"
    complete_start.write_bytes(TASK_GOAL.read_bytes())
    calls: list[object] = []

    def forbidden_dispatch(*args, **kwargs):
        calls.append(args)
        raise AssertionError("rejected direct plan reached runtime dispatch")

    monkeypatch.setattr(piper_run, "_robot_operation_plan_phase", forbidden_dispatch)
    code, line, _ = _invoke(_structure_argv(
        tmp_path / "run", goal=TASK_GOAL, start=complete_start,
        plan=plan, preflight_only=False,
    ))
    expected = (
        RunStatus.UNRESOLVED_TCP
        if rejection == "tcp-denied" else RunStatus.INVALID_INPUT
    )
    assert line["status"] == expected, line
    assert code == EXIT_CODES[expected], line
    assert calls == []
    assert session_guard == []
    _assert_no_motion_claim(Path(line["result_path"]).parent)


@pytest.mark.parametrize("with_pick", [False, True])
@pytest.mark.parametrize("profile_id", [
    "brick-side-grasp-v1", "handover-bottom-coordinated-v1",
])
def test_handover_profile_admission_prevents_all_runtime_creation(
    tmp_path, session_guard, with_pick, profile_id,
):
    """Unknown or unqualified profiles reject even a preceding donor PICK."""
    document = _direct_handover_plan(with_pick=with_pick)
    document["operations"][-1]["parameters"]["grasp_profile_id"] = profile_id
    plan = tmp_path / "handover.json"
    plan.write_text(json.dumps(document), encoding="utf-8")
    complete_start = tmp_path / "complete-start.json"
    complete_start.write_bytes(TASK_GOAL.read_bytes())
    code, line, _ = _invoke(_structure_argv(
        tmp_path / "run", goal=TASK_GOAL,
        start=TASK_START if with_pick else complete_start,
        plan=plan, preflight_only=False,
    ))
    expected = (
        RunStatus.INVALID_INPUT if profile_id == "brick-side-grasp-v1"
        else RunStatus.EXECUTION_NOT_IMPLEMENTED
    )
    assert line["status"] == expected, line
    assert code == EXIT_CODES[expected]
    assert session_guard == []
    _assert_no_motion_claim(Path(line["result_path"]).parent)


def test_profile_flag_cannot_activate_legacy_handover_handler(
    tmp_path, session_guard, monkeypatch,
):
    """Qualification alone never substitutes for new production providers."""
    from dataclasses import replace

    from rocobrick.skills import handover_profile

    qualified = replace(
        handover_profile.resolve_handover_profile("handover-bottom-coordinated-v1"),
        execution_qualified=True,
    )
    monkeypatch.setattr(
        handover_profile, "resolve_handover_profile", lambda _: qualified
    )
    plan = tmp_path / "handover.json"
    plan.write_text(json.dumps(_direct_handover_plan(with_pick=True)), encoding="utf-8")
    code, line, _ = _invoke(_structure_argv(
        tmp_path / "run", goal=TASK_GOAL, start=TASK_START, plan=plan,
        preflight_only=False,
    ))
    assert line["status"] == RunStatus.EXECUTION_NOT_IMPLEMENTED
    assert code == EXIT_CODES[RunStatus.EXECUTION_NOT_IMPLEMENTED]
    assert session_guard == []
    _assert_no_motion_claim(Path(line["result_path"]).parent)


def test_omitted_and_empty_start_mean_empty_and_fail_closed(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
) -> None:
    """Omitting the start or passing '' both mean EMPTY, and both fail closed."""
    omitted_run = tmp_path / "run-omitted"
    omitted_code, omitted_line, _ = _invoke(
        _structure_argv(omitted_run, goal=TASK_GOAL)
    )
    empty_run = tmp_path / "run-empty"
    empty_code, empty_line, _ = _invoke(
        _structure_argv(empty_run, goal=TASK_GOAL, start="")
    )
    _assert_no_session(session_guard)

    # An empty initial structure cannot assemble the frozen example1 goal with
    # its single PLACE_DOWN, so both invocations must fail closed and agree.
    _assert_rejected(omitted_code, omitted_line)
    _assert_rejected(empty_code, empty_line)
    assert omitted_line.get("status") == empty_line.get("status"), (
        omitted_line,
        empty_line,
    )
    assert omitted_code == empty_code

    for run_dir, line in (
        (omitted_run, omitted_line),
        (empty_run, empty_line),
    ):
        evidence_dir = _assert_early_exit_evidence(line, run_dir)
        _assert_empty_start_audited(_json_documents(evidence_dir))
        _assert_no_motion_claim(evidence_dir)
        start_bytes = TASK_START.read_bytes()
        copies = [
            path.read_bytes()
            for path in evidence_dir.iterdir()
            if path.is_file()
        ]
        assert start_bytes not in copies, (
            "the empty start fell back to the repository structure_start.json"
        )


def test_preflight_inputs_record_explicit_start_evidence(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
) -> None:
    """The preflight inputs say ``start_provided``/``start_empty`` explicitly.

    An omitted flag and the literal empty string are both EMPTY with no
    invented SHA-256, a supplied blank start file is EMPTY with its real
    (zero-content) digest, and a supplied real start is provided non-empty.
    """
    blank_start = tmp_path / "blank-start.json"
    blank_start.write_bytes(b"")
    variants = (
        ("omitted", _OMITTED, TASK_GOAL),
        ("empty-string", "", TASK_GOAL),
        ("blank-file", blank_start, TASK_GOAL),
        ("real-start", TASK_START, SCENE_MANIFEST),
    )
    runs: list[tuple[str, Path, int, dict]] = []
    for label, start, manifest in variants:
        run_dir = tmp_path / f"run-{label}"
        code, line, _ = _invoke(
            _structure_argv(
                run_dir, goal=TASK_GOAL, start=start, manifest=manifest
            )
        )
        runs.append((label, run_dir, code, line))
    _assert_no_session(session_guard)

    for label, run_dir, code, line in runs:
        evidence_dir = _assert_early_exit_evidence(line, run_dir)
        inputs = _read_json(evidence_dir / "preflight.json")["inputs"]
        record = inputs["records"]["structure_start"]
        if label == "real-start":
            assert code == 0, line
            assert line["status"] == "PREFLIGHT_ONLY", line
        else:
            # An EMPTY initial structure cannot assemble the example1 goal.
            _assert_rejected(code, line)
        assert inputs["start_provided"] == record["start_provided"], inputs
        assert inputs["start_empty"] == record["start_empty"], inputs
        if label in ("omitted", "empty-string"):
            assert record["start_provided"] is False, record
            assert record["start_empty"] is True, record
            assert record["present"] is False, record
            assert record["path"] is None, record
            assert record["sha256"] is None, (
                "the absent start must not receive an invented SHA-256"
            )
            assert record["bytes"] is None, record
            assert record["captured_path"] is None, record
        elif label == "blank-file":
            assert record["start_provided"] is True, record
            assert record["start_empty"] is True, record
            assert record["present"] is True, record
            assert record["bytes"] == 0, record
            assert record["sha256"] == sha256(b"").hexdigest(), record
            assert record["captured_path"] is None, record
        else:
            assert record["start_provided"] is True, record
            assert record["start_empty"] is False, record
            assert record["present"] is True, record
            assert record["sha256"] == sha256(
                TASK_START.read_bytes()
            ).hexdigest(), record
            assert record["captured_path"] is not None, record


# ---------------------------------------------------------------------------
# Input aliasing
# ---------------------------------------------------------------------------


def _forbid_structure_binder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the structure binder is ever invoked.

    The aliasing rejection must happen strictly before the binder; any binder
    invocation becomes a hard test failure.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "the structure binder ran despite an input aliasing rejection"
        )

    monkeypatch.setattr(
        structure_scene,
        "build_structure_scene_plan_from_files",
        forbidden,
    )
    if hasattr(piper_run, "build_structure_scene_plan_from_files"):
        monkeypatch.setattr(
            piper_run,
            "build_structure_scene_plan_from_files",
            forbidden,
        )


@pytest.mark.parametrize(
    "case",
    [
        "plan-is-goal",
        "start-is-goal",
        "start-is-plan",
        "manifest-is-goal",
        "config-is-plan",
        "start-symlink-to-goal",
        "start-dotdot-to-goal",
    ],
)
def test_same_file_input_aliasing_is_invalid_input(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Two supplied inputs resolving to one file are rejected pre-binder."""
    _forbid_structure_binder(monkeypatch)
    goal: Path = TASK_GOAL
    start: object = _OMITTED
    manifest: Path | None = SCENE_MANIFEST
    extra: list[str] = []
    if case == "plan-is-goal":
        plan: Path = TASK_GOAL
    elif case == "start-is-goal":
        plan, start = OPERATION_PLAN, TASK_GOAL
    elif case == "start-is-plan":
        plan, start = OPERATION_PLAN, OPERATION_PLAN
    elif case == "manifest-is-goal":
        plan, manifest = OPERATION_PLAN, TASK_GOAL
    elif case == "config-is-plan":
        plan = OPERATION_PLAN
        extra = ["--config", str(OPERATION_PLAN)]
    elif case == "start-symlink-to-goal":
        plan = OPERATION_PLAN
        link = tmp_path / "start-alias.json"
        link.symlink_to(TASK_GOAL)
        start = link
    elif case == "start-dotdot-to-goal":
        plan = OPERATION_PLAN
        start = (
            TASK_GOAL.parent / ".." / TASK_GOAL.parent.name / TASK_GOAL.name
        )
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError(case)
    run_dir = tmp_path / "run"
    code, line, _ = _invoke(
        _structure_argv(
            run_dir, goal=goal, start=start, plan=plan, manifest=manifest
        )
        + extra
    )
    _assert_no_session(session_guard)
    assert code == EXIT_CODES[RunStatus.INVALID_INPUT], line
    assert line.get("status") == "INVALID_INPUT", line
    assert line.get("passed") is False, line
    evidence_dir = _assert_early_exit_evidence(line, run_dir)

    result = _read_json(evidence_dir / "result.json")
    reason = result.get("reason")
    assert isinstance(reason, str), reason
    assert "resolve to the same file" in reason, reason

    events = [
        record["event"]
        for record in (
            json.loads(raw)
            for raw in (evidence_dir / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if raw.strip()
        )
    ]
    alias_events = [
        event for event in events if event.get("phase") == "input_aliasing"
    ]
    assert len(alias_events) == 1, events
    assert alias_events[0].get("status") == "FAILED", alias_events[0]
    assert "resolve to the same file" in str(
        alias_events[0].get("reason")
    ), alias_events[0]
    resolved = alias_events[0].get("resolved_paths")
    assert isinstance(resolved, dict), alias_events[0]
    assert len(set(resolved.values())) < len(resolved), resolved
    assert [
        event for event in events if event.get("phase") == "structure_scene"
    ] == [], "the binder ran despite the input aliasing rejection"

    preflight = _read_json(evidence_dir / "preflight.json")
    assert preflight.get("binding") is None, preflight.get("binding")
    assert preflight.get("structure") is None, preflight.get("structure")
    checks = preflight.get("checks")
    assert isinstance(checks, dict), checks
    assert checks.get("inputs_non_aliasing") is False, checks
    assert checks.get("structure_scene_validated") is not True, checks
    assert checks.get("no_motion_dispatched") is True, checks
    assert preflight["inputs"]["inputs_non_aliasing"] is False, preflight
    assert (
        preflight["inputs"]["resolved_input_paths"] == resolved
    ), preflight["inputs"]

    run_document = _read_json(evidence_dir / "robot-operation-plan-run.json")
    assert run_document.get("status") == "INVALID_INPUT", run_document
    assert run_document.get("passed") is False, run_document
    assert run_document.get("runtime_created") is False, run_document
    assert run_document.get("motion_sent") is False, run_document
    _assert_no_motion_claim(evidence_dir)


# ---------------------------------------------------------------------------
# Manifest mismatches
# ---------------------------------------------------------------------------


def _mismatched_manifest(tmp_path: Path, variant: str) -> Path:
    """Write one manifest that must not validate against the example1 goal.

    Args:
        tmp_path: Temporary directory root.
        variant: ``foreign-id``, ``brick-mismatch`` or ``missing-object``.

    Returns:
        The written manifest path.
    """
    document = _read_json(SCENE_MANIFEST)
    objects = document["objects"]
    if variant == "foreign-id":
        objects[0]["physical_instance_id"] = "brick-ffffffffffffffffffff"
    elif variant == "brick-mismatch":
        objects[0]["brick_id"] = 99
    elif variant == "missing-object":
        document["objects"] = objects[1:]
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError(variant)
    target = tmp_path / f"{variant}-manifest.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


@pytest.mark.parametrize(
    "variant", ["foreign-id", "brick-mismatch", "missing-object"]
)
def test_mismatched_manifest_is_invalid_input(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
    variant: str,
) -> None:
    """A foreign or mismatched manifest is invalid input, not a valid run."""
    manifest = _mismatched_manifest(tmp_path, variant)
    manifest_bytes = manifest.read_bytes()
    run_dir = tmp_path / "run"
    code, line, _ = _invoke(
        _structure_argv(
            run_dir,
            goal=TASK_GOAL,
            start=TASK_START,
            manifest=manifest,
        )
    )
    _assert_no_session(session_guard)
    assert code == EXIT_CODES[RunStatus.INVALID_INPUT], line
    assert line.get("status") == "INVALID_INPUT", line
    assert line.get("passed") is False, line
    assert manifest.read_bytes() == manifest_bytes
    _assert_early_exit_evidence(line, run_dir)


# ---------------------------------------------------------------------------
# Mutation after capture
# ---------------------------------------------------------------------------


def test_input_mutation_after_capture_is_detected(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A goal mutated after the structure build is caught by the re-hash."""
    inputs = _structure_workspace(tmp_path)
    goal = inputs["goal"]
    original_build = structure_scene.build_structure_scene_plan_from_files
    state = {"mutated": False, "built": 0}

    def mutating_build(*args: object, **kwargs: object) -> object:
        result = original_build(*args, **kwargs)
        state["built"] += 1
        if not state["mutated"]:
            state["mutated"] = True
            goal.write_bytes(b'{"mutated": true}')
        return result

    monkeypatch.setattr(
        structure_scene,
        "build_structure_scene_plan_from_files",
        mutating_build,
    )
    if hasattr(piper_run, "build_structure_scene_plan_from_files"):
        monkeypatch.setattr(
            piper_run,
            "build_structure_scene_plan_from_files",
            mutating_build,
        )

    run_dir = tmp_path / "run"
    code, line, _ = _invoke(
        _structure_argv(
            run_dir,
            goal=goal,
            start=inputs["start"],
            plan=inputs["plan"],
            manifest=inputs["manifest"],
        )
    )
    _assert_no_session(session_guard)
    assert state["built"] >= 1, (
        "the structure binder was never invoked; the mutation was not injected"
    )
    assert state["mutated"] is True
    assert line.get("status") == "INPUT_MUTATED", line
    assert code == EXIT_CODES[RunStatus.INPUT_MUTATED], line
    assert line.get("passed") is False, line
    _assert_early_exit_evidence(line, run_dir)


# ---------------------------------------------------------------------------
# Early exits always keep events.jsonl
# ---------------------------------------------------------------------------


def _break_input_for_early_exit(inputs: Mapping[str, Path], case: str) -> None:
    """Sabotage exactly one temporary input to force one early exit.

    Args:
        inputs: The temporary input copies.
        case: The early-exit case name.
    """
    if case == "goal-not-json":
        inputs["goal"].write_bytes(b"{")
    elif case == "goal-empty-object":
        inputs["goal"].write_bytes(b"{}")
    elif case == "start-not-json":
        inputs["start"].write_bytes(b"{")
    elif case == "plan-not-json":
        inputs["plan"].write_bytes(b"not json")
    elif case == "plan-missing":
        inputs["plan"].unlink()
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "goal-not-json",
        "goal-empty-object",
        "start-not-json",
        "plan-not-json",
        "plan-missing",
    ],
)
def test_early_exit_always_writes_events(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
    case: str,
) -> None:
    """Every early rejection keeps events.jsonl, preflight.json, result.json."""
    inputs = _structure_workspace(tmp_path)
    _break_input_for_early_exit(inputs, case)
    before = _digests([path for path in inputs.values() if path.exists()])
    run_dir = tmp_path / "run"
    code, line, _ = _invoke(
        _structure_argv(
            run_dir,
            goal=inputs["goal"],
            start=inputs["start"],
            plan=inputs["plan"],
            manifest=inputs["manifest"],
        )
    )
    _assert_no_session(session_guard)
    _assert_rejected(code, line)
    _assert_early_exit_evidence(line, run_dir)
    assert _digests([path for path in inputs.values() if path.exists()]) == before


# ---------------------------------------------------------------------------
# WAIT-only must never be EXECUTED
# ---------------------------------------------------------------------------


def test_wait_only_plan_never_produces_executed(
    tmp_path: Path,
    session_guard: list[tuple[object, ...]],
) -> None:
    """A WAIT-only plan cannot pass a structure preflight as EXECUTED."""
    plan = tmp_path / "wait-only-plan.json"
    plan.write_text(json.dumps(_wait_only_plan()), encoding="utf-8")
    run_dir = tmp_path / "run"
    code, line, _ = _invoke(
        _structure_argv(
            run_dir,
            goal=TASK_GOAL,
            start=TASK_START,
            plan=plan,
            manifest=SCENE_MANIFEST,
        )
    )
    _assert_no_session(session_guard)
    _assert_rejected(code, line)
    assert line.get("status") != "EXECUTED", line
    evidence_dir = _assert_early_exit_evidence(line, run_dir)
    _assert_no_motion_claim(evidence_dir)
