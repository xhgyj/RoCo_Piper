"""Formal ``piper-run`` immutable STAP ingest and preflight composition.

``piper-run`` is the offline composition entry point between the accepted
contract libraries and a future execution backend. Every invocation:

1. Reads the plan and the two task structure files exactly once into memory
   and hashes the raw bytes before anything is parsed (IMMUTABLE_INPUT). The
   input paths are never written to.
2. Validates the plan authoritatively with ``parse_stap_plan``. The metric
   plan contract is never imported or called, and no plan is ever repaired,
   reassigned, reordered, or rewritten (STAP_VALIDATED).
3. Binds the plan's single pending task to the exact supplied bytes with
   ``bind_single_stap_task`` and stops on any rejection (STRUCTURAL_BINDING).
4. Re-derives the Piper L TCP claim from the read-only model-audit artifact
   with ``assess_piper_l_tcp``, requires the re-derived status to equal the
   serialized status and ``tcp_execution_allowed`` to be consistent, and —
   because every PLACE_DOWN binding requires a tool point —
   ``require_resolved_tcp`` (TCP_GATE).
5. Without ``--robot-operation-plan``, never dispatches: the default path is
   still the preflight bridge. A run whose gates all pass reports
   ``EXECUTION_NOT_IMPLEMENTED`` and exits nonzero; only ``--preflight-only``
   on a plan whose gates all pass reports ``PREFLIGHT_ONLY`` with exit code
   0. No ROS, Isaac, Docker, GPU, MoveIt action, or motion is ever started
   or sent.
6. ``--robot-operation-plan PLAN_JSON`` is the only production dispatch
   entry: the external ``robot-operation-plan/v1`` bytes are copied
   verbatim into ``robot-operation-plan.json`` inside the run directory and
   validated against the authoritative STAP plan (cross-binding: plan id,
   robot set, task allocation, parts, connections, layers, schedule and
   resources). Every gate — including the strict
   :func:`rocobrick.tcp_capability.authorize_tcp` decision — runs BEFORE any
   runtime is created. Only then is the real multi-robot session
   (:class:`rocobrick.multi_robot_runtime.MultiRobotIsaacMoveItSession`,
   one host and one container for all validated plan robots) created and
   started; it needs the explicit scene group (``--scene-manifest``,
   ``--scene-snapshot-out``, ``--host-instance-id`` and ``--config``) and
   never falls back to a fake transport. ``start()`` must prove readiness
   for every plan robot; otherwise no operation is ever scheduled.
7. With a READY session the plan runs through
   :class:`rocobrick.robot_operation_runner.OperationPlanRunner` in strict
   mode (``allow_legacy_results=False``) over
   :class:`rocobrick.execution.robot_operation_executor.SkillOperationExecutor`, which
   observes the real scene through the session's read-only ``observe``
   transport. WAIT, TRANSIT, PICK and PLACE_DOWN use their evidence backends;
   PLACE_DOWN proves connection, release and retreat separately before any
   assembled state is published. The session is stopped with cleanup evidence in
   a ``finally`` block on every path, and ``robot-operation-plan-run.json``
   (schema ``rocobrick-operation-plan-run/v1``) records readiness, per-robot
   isolated operations, layers, cancellations, resources, cleanup, the
   immutable input digests and the aggregate ``passed`` verdict. STAP-backed
   runs additionally require pending assembly effects to be evidenced by
   CONFIRMED PLACE_DOWN operations; direct robot-operation-plan runs have no
   STAP assembly effect to evaluate.
   Terminal statuses: ``INVALID_INPUT`` (2) for a malformed or unbindable
   external plan, ``UNRESOLVED_TCP`` (4) when the TCP authorization denies,
   ``RUNTIME_START_FAILED`` (8) when the session or any robot readiness
   fails (zero operations scheduled), ``EXECUTION_FAILED`` (10) when the
   scheduler, the assembly-effect check or cleanup does not pass, and
   ``EXECUTED`` (0) only when every gate, every robot readiness, every
   operation, every pending assembly effect and the cleanup passed and the
   input digests are unchanged.
8. Re-hashes the original input files from disk after all phases — including
   the external operation plan and after the runtime — and fails with
   ``INPUT_MUTATED`` if any digest changed (IMMUTABILITY).
9. ``--robot-operation-plan`` WITHOUT ``--task``/``--plan`` is the
   structure-bound direct path. It requires ``--structure-goal`` (which is
   never defaulted); ``--structure-start`` is optional and an omitted flag
   or the literal empty string both mean EMPTY initial structure, and the
   inputs record that choice explicitly (``start_provided`` and
   ``start_empty``) without inventing a digest for an absent start. The
   named original inputs must also stay distinct files: the plan, the goal,
   a supplied start and the optional scene manifest and config are rejected
   with ``INVALID_INPUT`` (2) when any two of them resolve to the same file,
   strictly before the binder runs. The external plan is read, copied
   verbatim, hashed and validated with ``parse_robot_operation_plan(source_plan=None)``,
   then bound to the exact structure files with
   :func:`rocobrick.plan.structure_scene.build_structure_scene_plan_from_files`
   and only afterwards is the TCP claim re-derived. By default this path
   executes: the scene manifest, snapshot, host id and config paths are
   filled from the fresh run directory and repository defaults, and the
   Isaac host generates the manifest from its loaded scene before PICK
   dispatch. With ``--preflight-only`` the same invocation becomes the
   no-runtime preflight: no scene path is invented, no session is created,
   no operation is scheduled and no motion is sent.

The program is offline and pure apart from its evidence directory and the
explicit ``--task``/``--plan`` STAP or structure-bound
``--robot-operation-plan`` runtime bridge: without dispatch (no
``--robot-operation-plan``, or the ``--preflight-only`` direct form) it
imports no runtime adapter module. It writes evidence (input copies, phase
events, ``preflight.json``, and ``result.json``) only inside its own run
directory. Exactly one final JSON line is printed to stdout; diagnostics go
to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from traceback import format_exc
from uuid import uuid4

from rocobrick.compat import StrEnum
from rocobrick.evidence import EvidenceRecorder, FinalResult
from rocobrick.robot_operation_plan import (
    RobotOperationPlanError,
    parse_robot_operation_plan,
)
from rocobrick.stap_contract import (
    STAP_PLAN_SCHEMA,
    StapContractError,
    parse_stap_plan,
)
from rocobrick.structural_binding import (
    StructuralBinding,
    bind_single_stap_task,
    binding_requires_tcp,
)
from rocobrick.tcp_capability import authorize_tcp
from rocobrick.tcp_contract import (
    TcpContract,
    TcpStatus,
    UnresolvedTcpError,
    assess_piper_l_tcp,
    require_resolved_tcp,
)

RESULT_SCHEMA = "rocobrick-piper-run-result/v1"
PREFLIGHT_SCHEMA = "rocobrick-piper-run-preflight/v1"
TCP_ARTIFACT_SCHEMA = "rocobrick-piper-l-tcp-binding-contract/v1"
PLAN_CAPTURE_NAME = "robot-operation-plan.json"
PLAN_RUN_NAME = "robot-operation-plan-run.json"
SCENE_MANIFEST_CAPTURE_NAME = "scene-manifest.json"
CONFIG_CAPTURE_NAME = "config.json"
# Direct ``--robot-operation-plan`` runs never synthesize a STAP source plan.
# Explicit preflight-only runs stop before dispatch; direct runtime runs still
# require structure binding, implemented skill backends and TCP authorization.
STRUCTURE_ONLY_MODE = "structure-only-preflight"
NO_STAP_AUTHORIZATION = "no authoritative STAP authorization"
OPERATION_TIMEOUT_S = 60.0
CANCEL_TIMEOUT_S = 5.0
ASSEMBLY_EFFECT_SCHEMA = "rocobrick-piper-run-stap-assembly-effects/v1"
PENDING_TASK_SKILL = "PLACE_DOWN"
ASSEMBLY_EVIDENCE_LABELS = (
    "connection_confirmed",
    "release_confirmed",
    "retreat_confirmed",
)
UPWARD_TASK_SKILL = "PLACE_UP"


REPO_ROOT = Path(__file__).resolve().parents[2]
TCP_ARTIFACT = (
    REPO_ROOT / "artifacts" / "model-audit" / "piper_l_tcp_binding_contract.json"
)
TASK_START_NAME = "structure_start.json"
TASK_GOAL_NAME = "structure_goal.json"
DEFAULT_ROBOT_ID = "piper_0"
BACKENDS = ("isaac",)
RUNS_DIR_NAME = "runs"
RUN_DIR_PREFIX = "piper-run"
RUNTIME_USD_PINNED_PATH = "robot_assets/piper_l/USD/piper_l_gripper_d435.usd"


class RunStatus(StrEnum):
    """Exact terminal status vocabulary of a ``piper-run`` invocation."""

    PREFLIGHT_ONLY = "PREFLIGHT_ONLY"
    INVALID_INPUT = "INVALID_INPUT"
    BINDING_REJECTED = "BINDING_REJECTED"
    UNRESOLVED_TCP = "UNRESOLVED_TCP"
    TCP_CONTRACT_INVALID = "TCP_CONTRACT_INVALID"
    INPUT_MUTATED = "INPUT_MUTATED"
    EXECUTION_NOT_IMPLEMENTED = "EXECUTION_NOT_IMPLEMENTED"
    RUNTIME_START_FAILED = "RUNTIME_START_FAILED"
    OPERATION_REJECTED = "OPERATION_REJECTED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTED = "EXECUTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


EXIT_CODES: Mapping[RunStatus, int] = {
    RunStatus.PREFLIGHT_ONLY: 0,
    RunStatus.EXECUTED: 0,
    RunStatus.INTERNAL_ERROR: 1,
    RunStatus.INVALID_INPUT: 2,
    RunStatus.BINDING_REJECTED: 3,
    RunStatus.UNRESOLVED_TCP: 4,
    RunStatus.TCP_CONTRACT_INVALID: 5,
    RunStatus.INPUT_MUTATED: 6,
    RunStatus.EXECUTION_NOT_IMPLEMENTED: 7,
    RunStatus.RUNTIME_START_FAILED: 8,
    RunStatus.OPERATION_REJECTED: 9,
    RunStatus.EXECUTION_FAILED: 10,
}


class _UsageError(Exception):
    """Raised when the command line itself is invalid."""


class _RunDirError(Exception):
    """Raised when the evidence run directory cannot be created freshly."""

    def __init__(self, reason: str, path: Path) -> None:
        """Store the reason and the path the caller asked for.

        Args:
            reason: Human-readable explanation for the refusal.
            path: The run directory path involved in the refusal.
        """
        super().__init__(reason)
        self.reason = reason
        self.path = path


class _TcpGateError(Exception):
    """Raised when the TCP gate refuses the run with a terminal status."""

    def __init__(self, status: RunStatus, reason: str) -> None:
        """Store the terminal status demanded by the gate and its reason.

        Args:
            status: The terminal run status (``TCP_CONTRACT_INVALID``).
            reason: Human-readable explanation for the refusal.
        """
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class RunOutcome:
    """Terminal result of one invocation, mirrored in the stdout JSON line."""

    status: RunStatus
    reason: str
    exit_code: int
    run_dir: Path | None
    result_path: Path | None
    passed: bool | None = None


@dataclass(frozen=True)
class _Arguments:
    """Parsed command-line arguments."""

    task_dir: Path | None
    plan_path: Path | None
    backend: str
    robot_id: str
    run_dir: Path | None
    preflight_only: bool
    robot_operation_plan_path: Path | None = None
    scene_manifest: Path | None = None
    scene_snapshot_out: Path | None = None
    host_instance_id: str | None = None
    config_path: Path | None = None
    structure_dir: Path | None = None
    structure_goal: Path | None = None
    structure_start: Path | None = None


@dataclass(frozen=True)
class _Inputs:
    """The exact input bytes read once, with their raw digests."""

    task_dir: Path
    start_path: Path
    goal_path: Path
    plan_path: Path
    plan_bytes: bytes
    start_bytes: bytes
    goal_bytes: bytes
    sha256: Mapping[str, str]
    sizes: Mapping[str, int]


class _ArgumentParser(argparse.ArgumentParser):
    """An argument parser that raises instead of exiting the process."""

    def error(self, message: str) -> None:
        """Raise ``_UsageError`` so ``main`` can emit its JSON line.

        Args:
            message: The usage error description from argparse.
        """
        raise _UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``piper-run`` argument parser.

    Returns:
        The parser implementing the exact ``piper-run`` CLI interface.
    """
    parser = _ArgumentParser(
        prog="piper-run",
        description=(
            "Immutable STAP ingest, structural binding, TCP gate, and "
            "evidence composition preflight. Starts no runtime and no motion."
        ),
    )
    parser.add_argument(
        "--task",
        required=False,
        type=Path,
        metavar="TASK_DIR",
        help=(
            "task directory containing structure_start.json and "
            "structure_goal.json (extra files are ignored); read-only, "
            "together with --plan"
        ),
    )
    parser.add_argument(
        "--plan",
        required=False,
        type=Path,
        metavar="PLAN_JSON",
        help=(
            "assembly-plan/v2 STAP plan JSON file; read-only, together with "
            "--task; omit both to select the structure-bound "
            "--robot-operation-plan path (add --preflight-only for the "
            "no-runtime preflight)"
        ),
    )
    parser.add_argument(
        "--backend",
        required=False,
        default=None,
        choices=BACKENDS,
        help=(
            "execution backend selected for this run; required by the "
            f"--task/--plan STAP path (default for the structure-only "
            f"preflight: {BACKENDS[0]})"
        ),
    )
    parser.add_argument(
        "--robot-id",
        default=DEFAULT_ROBOT_ID,
        metavar="ROBOT_ID",
        help=(
            "robot the plan must allocate its single pending task to "
            f"(default: {DEFAULT_ROBOT_ID})"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "evidence directory to create; it must not already exist "
            "(default: a unique runs/piper-run-<UTC timestamp>-<8hex>)"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "stop after the gates and never consider dispatch; an unresolved "
            "required TCP still fails nonzero"
        ),
    )
    parser.add_argument("--scene-manifest", type=Path, default=None)
    parser.add_argument("--scene-snapshot-out", type=Path, default=None)
    parser.add_argument("--host-instance-id", default=None)
    parser.add_argument("--config", type=Path, default=None, metavar="USER_CONFIG_JSON")
    parser.add_argument(
        "--structure-dir",
        type=Path,
        default=None,
        metavar="STRUCTURE_DIR",
    )
    parser.add_argument(
        "--structure-goal",
        type=Path,
        default=None,
        metavar="GOAL_JSON",
        dest="structure_goal",
        help=(
            "goal structure JSON of a structure-bound "
            "--robot-operation-plan invocation (no --task/--plan STAP "
            "source); required in that mode and never defaulted"
        ),
    )
    parser.add_argument(
        "--structure-start",
        default=None,
        metavar="START_JSON",
        dest="structure_start",
        help=(
            "optional initial structure JSON of a structure-bound "
            "--robot-operation-plan invocation; an omitted flag or the "
            "literal empty string both mean EMPTY initial structure"
        ),
    )
    parser.add_argument(
        "--robot-operation-plan",
        default=None,
        type=Path,
        dest="robot_operation_plan",
        metavar="PLAN_JSON",
        help=(
            "the ONLY production dispatch entry: an external "
            "robot-operation-plan/v1 document, validated against the "
            "assembly-plan/v2 binding (with --task/--plan) or the explicit "
            "structure inputs, then executed through the real multi-robot "
            "session; scene paths default into the run directory"
        ),
    )
    return parser


def _diagnostic(message: str) -> None:
    """Print a diagnostics line to stderr.

    Args:
        message: The diagnostics text.
    """
    print(message, file=sys.stderr)


def _prepare_run_dir(requested: Path | None) -> Path:
    """Create the evidence run directory without ever overwriting.

    Args:
        requested: The caller's directory, or ``None`` for a unique
            ``runs/piper-run-<UTC timestamp>-<8hex>`` directory.

    Returns:
        The freshly created run directory.

    Raises:
        _RunDirError: If the directory exists or cannot be created.
    """
    if requested is not None:
        requested = requested.expanduser().resolve()
        try:
            requested.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise _RunDirError(
                f"run directory {requested} already exists; refusing to overwrite it",
                requested,
            ) from error
        except OSError as error:
            raise _RunDirError(
                f"cannot create run directory {requested}: {error}", requested
            ) from error
        return requested
    base = Path(RUNS_DIR_NAME).expanduser().resolve()
    for _ in range(64):
        candidate = base / (
            f"{RUN_DIR_PREFIX}-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        )
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:  # pragma: no cover - 8hex collision
            continue
        except OSError as error:
            raise _RunDirError(
                f"cannot create a run directory under {base}: {error}", base
            ) from error
        return candidate
    raise _RunDirError(  # pragma: no cover - statistically unreachable
        f"cannot create a unique run directory under {base}", base
    )


def _resolve_and_read(task_dir: Path, plan_path: Path) -> _Inputs:
    """Check the input layout and read every input exactly once.

    Args:
        task_dir: Directory that must contain both task structure files.
        plan_path: Plan file that must exist.

    Returns:
        The exact input bytes with their raw SHA-256 digests and sizes.

    Raises:
        ValueError: If the task directory, structure files, or plan file do
            not satisfy the required layout.
    """
    if not task_dir.is_dir():
        raise ValueError(
            f"task directory {task_dir} does not exist or is not a directory"
        )
    start_path = task_dir / TASK_START_NAME
    goal_path = task_dir / TASK_GOAL_NAME
    for path in (start_path, goal_path):
        if not path.is_file():
            raise ValueError(f"required task structure file {path} is missing")
    if not plan_path.is_file():
        raise ValueError(f"plan {plan_path} does not exist or is not a file")
    plan_bytes = plan_path.read_bytes()
    start_bytes = start_path.read_bytes()
    goal_bytes = goal_path.read_bytes()
    return _Inputs(
        task_dir=task_dir,
        start_path=start_path,
        goal_path=goal_path,
        plan_path=plan_path,
        plan_bytes=plan_bytes,
        start_bytes=start_bytes,
        goal_bytes=goal_bytes,
        sha256={
            "plan": sha256(plan_bytes).hexdigest(),
            "task_start": sha256(start_bytes).hexdigest(),
            "task_goal": sha256(goal_bytes).hexdigest(),
        },
        sizes={
            "plan": len(plan_bytes),
            "task_start": len(start_bytes),
            "task_goal": len(goal_bytes),
        },
    )


def _load_tcp_artifact(path: Path) -> Mapping[str, object]:
    """Read the TCP model-audit artifact JSON, strictly read-only.

    Args:
        path: The artifact path.

    Returns:
        The parsed artifact mapping.

    Raises:
        ValueError: If the artifact is not valid UTF-8 JSON or not an object.
        OSError: If the artifact cannot be read.
    """
    payload = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("artifact must be a JSON object")
    return payload


def _tcp_evidence_from_artifact(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    """Extract the structured evidence mapping the TCP contract expects.

    This is a format adapter only: classification is delegated entirely to
    ``assess_piper_l_tcp``; nothing is classified, judged, or repaired here.

    Args:
        artifact: The parsed model-audit artifact.

    Returns:
        The evidence mapping with ``robot_model``, ``pinned_sources``, and
        ``items`` exactly as ``assess_piper_l_tcp`` requires.

    Raises:
        KeyError: If the artifact lacks the required sections.
        TypeError: If sections have unexpected shapes.
    """
    pinned_files = artifact["inputs"]["pinned_files"]
    return {
        "robot_model": artifact["model"],
        "pinned_sources": {entry["path"]: entry["sha256"] for entry in pinned_files},
        "items": artifact["evidence"],
    }


def _rederive_tcp(tcp_record: dict[str, object]) -> TcpContract:
    """Re-derive the TCP contract from the read-only audit artifact.

    The artifact's schema, its own ``passed`` verdict, its serialized status,
    and its ``tcp_execution_allowed`` claim are all verified against a fresh
    ``assess_piper_l_tcp`` classification; no single field is trusted alone.

    Args:
        tcp_record: The mutable TCP section of the preflight record, updated
            in place with everything the gate observed.

    Returns:
        The re-derived ``TcpContract``.

    Raises:
        _TcpGateError: With ``TCP_CONTRACT_INVALID`` when the artifact is
            missing, malformed, failed its own audit, or disagrees with the
            strict re-derivation.
    """
    tcp_record["artifact"] = str(TCP_ARTIFACT)
    try:
        artifact = _load_tcp_artifact(TCP_ARTIFACT)
    except (OSError, ValueError) as error:
        raise _TcpGateError(
            RunStatus.TCP_CONTRACT_INVALID,
            f"the TCP binding artifact {TCP_ARTIFACT} is unreadable or "
            f"malformed: {error}",
        ) from error
    tcp_record["artifact_schema"] = artifact.get("schema")
    tcp_record["artifact_checks"] = artifact.get("checks")
    if artifact.get("schema") != TCP_ARTIFACT_SCHEMA:
        raise _TcpGateError(
            RunStatus.TCP_CONTRACT_INVALID,
            f"the TCP binding artifact schema {artifact.get('schema')!r} is "
            f"not {TCP_ARTIFACT_SCHEMA!r}",
        )
    tcp_record["artifact_passed"] = artifact.get("passed")
    if artifact.get("passed") is not True:
        raise _TcpGateError(
            RunStatus.TCP_CONTRACT_INVALID,
            "the TCP binding artifact records passed=false; its own audit failed",
        )
    serialized = artifact.get("tcp_contract")
    if not isinstance(serialized, Mapping) or not isinstance(
        serialized.get("status"), str
    ):
        raise _TcpGateError(
            RunStatus.TCP_CONTRACT_INVALID,
            "the TCP binding artifact carries no serialized tcp_contract.status string",
        )
    tcp_record["serialized_status"] = serialized["status"]
    tcp_record["serialized_blockers"] = serialized.get("blockers")
    try:
        evidence = _tcp_evidence_from_artifact(artifact)
        contract = assess_piper_l_tcp(evidence)
    except (KeyError, TypeError, ValueError) as error:
        raise _TcpGateError(
            RunStatus.TCP_CONTRACT_INVALID,
            f"the TCP binding artifact evidence cannot be re-derived: {error}",
        ) from error
    tcp_record["derived_status"] = str(contract.status)
    tcp_record["derived_candidate_frames"] = list(contract.candidate_frames)
    tcp_record["derived_blockers"] = list(contract.blockers)
    tcp_record["derived_transform"] = (
        asdict(contract.transform) if contract.transform is not None else None
    )
    if contract.status.value != serialized["status"]:
        raise _TcpGateError(
            RunStatus.TCP_CONTRACT_INVALID,
            f"the re-derived TCP status {contract.status.value} does not "
            f"equal the serialized status {serialized['status']}",
        )
    tcp_record["status_matches_serialized"] = True
    allowed = artifact.get("tcp_execution_allowed")
    if not isinstance(allowed, bool) or allowed is not (
        contract.status is TcpStatus.RESOLVED
    ):
        raise _TcpGateError(
            RunStatus.TCP_CONTRACT_INVALID,
            f"tcp_execution_allowed={allowed!r} contradicts the re-derived "
            f"status {contract.status.value}",
        )
    tcp_record["tcp_execution_allowed_consistent"] = True
    verified_inputs = artifact.get("inputs")
    if not isinstance(verified_inputs, Mapping):
        raise _TcpGateError(
            RunStatus.TCP_CONTRACT_INVALID,
            "the TCP binding artifact carries no verified inputs mapping",
        )
    tcp_record["verified_binding_record"] = {
        "inputs": {
            "pinned_files": verified_inputs.get("pinned_files"),
        },
    }
    return contract


def _derive_pick_collision_filter(
    args: "_Arguments",
    robot_ids: tuple[str, ...],
) -> tuple[str | None, tuple[str, ...]]:
    """Derive the authorized scene-contact set for direct skill execution.

    Covers every PICK target and every PLACE_DOWN upper/lower part.

    Returns:
        The unique executor robot and its authorized physical-instance ids;
        ambiguous multi-robot assignments return ``(None, ())``.
    """
    plan_path = getattr(args, "robot_operation_plan_path", None)
    if plan_path is None or not plan_path.is_file():
        return None, ()
    try:
        document = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, ()
    operations = document.get("operations")
    if not isinstance(operations, list):
        return None, ()
    by_robot: dict[str, set[str]] = {}
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        skill = operation.get("skill")
        robot_id = operation.get("executor_robot_id")
        target = operation.get("target")
        physical_ids: set[str] = set()
        if skill == "PICK" and isinstance(target, Mapping):
            physical_id = target.get("physical_instance_id")
            if isinstance(physical_id, str) and physical_id:
                physical_ids.add(physical_id)
        elif skill in {"PLACE_DOWN", "PLACE_UP"} and isinstance(target, Mapping):
            for name in (
                "upper_physical_instance_id",
                "lower_physical_instance_id",
            ):
                value = target.get(name)
                if isinstance(value, str) and value:
                    physical_ids.add(value)
        if isinstance(robot_id, str) and robot_id in robot_ids and physical_ids:
            by_robot.setdefault(robot_id, set()).update(physical_ids)
    if len(by_robot) != 1:
        return None, ()
    robot_id, instance_ids = next(iter(by_robot.items()))
    return robot_id, tuple(sorted(instance_ids))


def _derive_observation_connection_pairs(
    args: "_Arguments",
    robot_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Derive the observed connection pairs of direct PLACE_DOWN targets.

    Every PLACE_DOWN upper/lower pair of the plan becomes one observed
    connection pair, so each scene observation carries the live bricksim
    ``are_parts_connected`` verdict the placement evidence requires.  Pairs
    are unique and order-normalized.

    Returns:
        The unique order-normalized connection pairs of the plan.
    """
    plan_path = getattr(args, "robot_operation_plan_path", None)
    if plan_path is None or not plan_path.is_file():
        return ()
    try:
        document = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    operations = document.get("operations")
    if not isinstance(operations, list):
        return ()
    pairs: set[tuple[str, str]] = set()
    for operation in operations:
        if not isinstance(operation, Mapping) or operation.get("skill") not in {
            "PLACE_DOWN", "PLACE_UP"
        }:
            continue
        if operation.get("executor_robot_id") not in robot_ids:
            continue
        target = operation.get("target")
        if not isinstance(target, Mapping):
            continue
        upper = target.get("upper_physical_instance_id")
        lower = target.get("lower_physical_instance_id")
        if isinstance(upper, str) and isinstance(lower, str) and upper and lower:
            pairs.add((upper, lower))
    return tuple(sorted(pairs))


def _create_session(
    session_dir: Path,
    args: "_Arguments",
    robot_ids: tuple[str, ...],
):
    """Create the real multi-robot session (injection seam for offline tests).

    The production construction is deliberately trivial and lazy: the real
    :class:`rocobrick.multi_robot_runtime.MultiRobotIsaacMoveItSession` is
    imported only here, after every gate and the TCP authorization have
    passed, with the validated plan robots and the explicit scene option
    group of the invocation. There is no fake fallback: a missing scene group
    is rejected as a usage error before any run directory exists, and a
    construction failure is reported as a runtime start failure.

    Args:
        session_dir: Directory (inside the run evidence directory) that the
            session owns for its lifecycle evidence.
        args: Parsed command-line arguments carrying the scene group.
        robot_ids: Validated ``robots`` of the parsed operation plan; every
            one of them is started and must become ready.

    Returns:
        A session exposing ``start()``, ``async observe(robot_id)`` and
        ``stop()``.
    """
    from rocobrick.multi_robot_runtime import (
        MultiRobotIsaacMoveItSession,
        MultiRobotSessionConfig,
        SubprocessRunner,
    )

    collision_filter_robot_id, collision_filter_instance_ids = (
        _derive_pick_collision_filter(args, robot_ids)
    )
    config = MultiRobotSessionConfig(
        repo_root=REPO_ROOT,
        robot_ids=robot_ids,
        scene_manifest=args.scene_manifest,
        scene_snapshot_out=args.scene_snapshot_out,
        host_instance_id=args.host_instance_id,
        plan_path=(
            args.plan_path
            or args.robot_operation_plan_path
            or REPO_ROOT / "artifacts/scene/example1-robot-operation-plan.json"
        ),
        config_path=args.config_path or REPO_ROOT / "config/user_config.json",
        structure_dir=args.structure_dir,
        structure_goal_path=args.structure_goal,
        structure_start_path=args.structure_start,
        collision_filter_robot_id=collision_filter_robot_id,
        collision_filter_instance_ids=collision_filter_instance_ids,
        observation_connection_pairs=_derive_observation_connection_pairs(
            args, robot_ids
        ),
        verify_plan_robots=not (
            args.robot_operation_plan_path is not None and args.plan_path is None
        ),
    )
    return MultiRobotIsaacMoveItSession(
        config, runner=SubprocessRunner(), session_dir=session_dir
    )


def _authorize_tcp_dispatch(
    args: "_Arguments",
    record: dict[str, object],
    recorder: EvidenceRecorder,
    contract: TcpContract | None = None,
) -> tuple[str | None, str | None] | tuple[RunStatus, str]:
    """Authorize calibrated-TCP dispatch from the verified binding record.

    Used by the ``--robot-operation-plan`` dispatch: extracts the runtime USD
    and calibration SHA-256 provenance from the verified binding artifact
    record, runs ``authorize_tcp`` fail-closed, and records the full decision
    into ``record["tcp"]["authorization"]`` plus one ``tcp_authorization``
    evidence event. Nothing is sent here and no session is created; a denial
    happens strictly before any runtime exists.

    Args:
        args: Parsed command-line arguments.
        record: The mutable preflight record (its ``tcp`` section carries the
            verified binding record and receives the authorization decision).
        recorder: The run evidence recorder.
        contract: The re-derived TCP contract from the TCP gate.

    Returns:
        ``(runtime_sha256, calibration_sha256)`` when dispatch is allowed,
        otherwise ``(RunStatus.UNRESOLVED_TCP, reason)``.
    """
    tcp_authorization: dict[str, object] = {
        "decision": "DENY",
        "allowed": False,
        "status": str(RunStatus.UNRESOLVED_TCP),
        "reason": None,
        "checks": {},
        "backend": args.backend,
        "provenance": "verified_binding_artifact",
        "runtime_rehash_performed": False,
        "runtime_asset_sha256": None,
        "calibration_artifact_sha256": None,
    }
    tcp_record = record.get("tcp")
    if isinstance(tcp_record, dict):
        tcp_record["authorization"] = tcp_authorization

    runtime_sha: str | None = None
    calibration_sha: str | None = None
    extraction_error: str | None = None
    try:
        if not isinstance(tcp_record, Mapping):
            raise ValueError("the verified TCP binding record is missing")
        verified_record = tcp_record.get("verified_binding_record")
        if not isinstance(verified_record, Mapping):
            raise ValueError("the verified TCP binding record is missing")
        verified_inputs = verified_record.get("inputs")
        if not isinstance(verified_inputs, Mapping):
            raise ValueError("the verified binding inputs are missing")
        pinned_files = verified_inputs.get("pinned_files")
        if not isinstance(pinned_files, list):
            raise ValueError("verified binding pinned_files must be an array")
        for index, entry in enumerate(pinned_files):
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"verified binding pinned_files[{index}] is not an object"
                )
            path = entry.get("path")
            digest = entry.get("sha256")
            if not isinstance(path, str):
                raise ValueError(
                    f"verified binding pinned_files[{index}].path is not a string"
                )
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"^[0-9a-f]{64}$", digest) is None
            ):
                raise ValueError(
                    f"verified binding pinned_files[{index}].sha256 is not a "
                    "lowercase SHA-256 digest"
                )
        runtime_candidates = [
            entry
            for entry in pinned_files
            if entry.get("path") == RUNTIME_USD_PINNED_PATH
        ]
        if len(runtime_candidates) != 1:
            raise ValueError(
                "verified binding must contain exactly one runtime USD pinned "
                f"entry at {RUNTIME_USD_PINNED_PATH!r}; found "
                f"{len(runtime_candidates)}"
            )
        runtime_sha = runtime_candidates[0]["sha256"]
        if not isinstance(contract, TcpContract) or contract.transform is None:
            raise ValueError("the verified re-derived TCP contract has no transform")
        calibration_sha = contract.transform.sha256
        if (
            not isinstance(calibration_sha, str)
            or re.fullmatch(r"^[0-9a-f]{64}$", calibration_sha) is None
        ):
            raise ValueError(
                "the re-derived TCP calibration transform has no valid SHA-256 digest"
            )
        tcp_authorization["runtime_asset_sha256"] = runtime_sha
        tcp_authorization["calibration_artifact_sha256"] = calibration_sha
    except Exception as error:
        extraction_error = (
            f"TCP capability authorization inputs could not be verified: {error}"
        )

    if extraction_error is not None:
        tcp_authorization["reason"] = extraction_error
        recorder.append_event(
            {
                "phase": "tcp_authorization",
                **tcp_authorization,
            }
        )
        return (
            RunStatus.UNRESOLVED_TCP,
            extraction_error,
        )

    backend = args.backend
    try:
        decision = authorize_tcp(
            contract,
            backend=backend,
            runtime_asset_sha256=runtime_sha,
            calibration_artifact_sha256=calibration_sha,
        )
        allowed = getattr(decision, "allowed")
        status = getattr(decision, "status")
        reason = getattr(decision, "reason")
        checks = getattr(decision, "checks")
        if (
            type(allowed) is not bool
            or not isinstance(status, str)
            or not isinstance(reason, str)
            or not isinstance(checks, Mapping)
            or any(
                not isinstance(key, str) or type(value) is not bool
                for key, value in checks.items()
            )
        ):
            raise ValueError("authorize_tcp returned a malformed decision")
        checks_copy = dict(checks)
        tcp_authorization.update(
            {
                "decision": "ALLOW" if allowed is True else "DENY",
                "allowed": allowed,
                "status": status,
                "reason": reason,
                "checks": checks_copy,
            }
        )
    except Exception as error:
        authorization_error = f"TCP capability authorization failed closed: {error}"
        tcp_authorization["reason"] = authorization_error
        recorder.append_event(
            {
                "phase": "tcp_authorization",
                **tcp_authorization,
            }
        )
        return (
            RunStatus.UNRESOLVED_TCP,
            authorization_error,
        )

    recorder.append_event(
        {
            "phase": "tcp_authorization",
            **tcp_authorization,
        }
    )
    if tcp_authorization["allowed"] is not True:
        return (
            RunStatus.UNRESOLVED_TCP,
            f"TCP capability authorization denied: {tcp_authorization['reason']}",
        )
    return runtime_sha, calibration_sha


def _json_value(value: object) -> object:
    """Copy one evidence value into strict-JSON-compatible structures.

    Dataclasses (readiness and session evidence), enums, paths, mappings and
    sequences are converted; nothing is judged or repaired here.

    Args:
        value: Any value carried by readiness, cleanup or scheduler evidence.

    Returns:
        A ``dict``/``list``/scalar copy accepted by ``json.dumps``.

    Raises:
        TypeError: If the value cannot be represented as strict JSON.
    """
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {str(key): _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"evidence value of type {type(value).__name__} is not JSON-safe")


def _evidence_document(evidence: object, name: str) -> dict[str, object]:
    """Normalize one session evidence object into a JSON-safe mapping.

    The evidence object's own serialization wins when it provides one
    (``serializable()`` or ``serialize()``), so computed verdicts such as
    ``MultiRobotReadinessEvidence.passed`` survive; plain dataclasses fall
    back to ``dataclasses.asdict``.

    Args:
        evidence: Mapping, serializing object, or dataclass.
        name: Evidence name used in error messages.

    Returns:
        The normalized evidence mapping.

    Raises:
        ValueError: If the evidence is not a serializable mapping.
    """
    if isinstance(evidence, Mapping):
        document = _json_value(dict(evidence))
    elif callable(getattr(evidence, "serializable", None)):
        document = _json_value(evidence.serializable())
    elif callable(getattr(evidence, "serialize", None)):
        document = _json_value(evidence.serialize())
    elif is_dataclass(evidence) and not isinstance(evidence, type):
        document = _json_value(evidence)
    else:
        raise ValueError(f"the {name} evidence is not a serializable mapping")
    if not isinstance(document, dict):
        raise ValueError(f"the {name} evidence is not a mapping")
    return document


def _readiness_rejection(
    document: Mapping[str, object], robots: tuple[str, ...]
) -> str | None:
    """Return why readiness evidence cannot authorize the plan, if any.

    Fail-closed and exact: ``passed`` and ``all_ready`` must both be ``True``;
    the reported robot list must equal the validated plan robots exactly
    (same order, no duplicate, no extra, no missing entry); the
    ``readiness_by_robot`` mapping must carry exactly the plan robot keys; and
    every robot's own report must explicitly say ``passed is True``. Nothing
    is inferred from a missing or duplicated value.

    Args:
        document: Normalized readiness evidence of the session start.
        robots: Validated ``robots`` of the parsed operation plan.

    Returns:
        ``None`` only when every robot is proven ready, otherwise the reason.
    """
    if document.get("passed") is not True or document.get("all_ready") is not True:
        return (
            "the session readiness evidence does not report passed=True and "
            f"all_ready=True: {dict(document)}"
        )
    reported = document.get("robots")
    if not isinstance(reported, (list, tuple)) or list(reported) != list(robots):
        return (
            f"the readiness evidence robots {reported!r} are not exactly the "
            f"plan robots {list(robots)}"
        )
    by_robot = document.get("readiness_by_robot")
    if not isinstance(by_robot, Mapping):
        return "the readiness evidence carries no readiness_by_robot mapping"
    if set(by_robot) != set(robots):
        extra = sorted(set(by_robot) - set(robots))
        missing = sorted(set(robots) - set(by_robot))
        return (
            "the readiness evidence readiness_by_robot keys are not exactly "
            f"the plan robots: missing={missing} extra={extra}"
        )
    for robot in robots:
        entry = by_robot.get(robot)
        if not isinstance(entry, Mapping):
            return f"the readiness evidence carries no report for robot {robot!r}"
        if entry.get("passed") is not True:
            return f"robot {robot!r} is not ready: {dict(entry)}"
    return None


def _cleanup_passed(document: Mapping[str, object]) -> bool:
    """Return whether session cleanup evidence proves a clean scoped stop.

    Fail-closed: the session's own cleanup verdict must be ``True`` AND the
    explicit termination evidence of the resources THIS session owned must
    prove a verified container removal and a reaped host stop with no
    session-scoped leftovers. A bare boolean or an empty mapping never counts
    as termination evidence.

    Args:
        document: Normalized session stop evidence.

    Returns:
        True only when every owned-resource termination flag is explicitly
        proven passing.
    """
    cleanup = document.get("cleanup")
    if not isinstance(cleanup, Mapping):
        return False
    if cleanup.get("passed") is not True:
        return False
    if cleanup.get("container_removed") is not True:
        return False
    if cleanup.get("host_stopped") is not True:
        return False
    leftovers = cleanup.get("leftovers")
    if isinstance(leftovers, Mapping) and (
        leftovers.get("session_containers") or leftovers.get("session_processes")
    ):
        return False
    return True


def _plan_failure_reason(document: Mapping[str, object]) -> str:
    """Summarize the first non-confirmed operation of a run document.

    Args:
        document: Serialized ``rocobrick-operation-plan-run/v1`` document.

    Returns:
        A human-readable summary naming the first operation that did not end
        confirmed, with its status and recorded reason when available.
    """
    operations = document.get("operations")
    if not isinstance(operations, Mapping) or not operations:
        return "no operation was confirmed by the scheduler"
    for operation_id, item in operations.items():
        if not isinstance(item, Mapping):
            continue
        if item.get("status") == "CONFIRMED" and item.get("passed") is True:
            continue
        reason = item.get("reason")
        return (
            f"operation {operation_id!r} on robot {item.get('robot_id')!r} "
            f"ended {item.get('status')}: "
            f"{reason if reason is not None else 'no reason recorded'}"
        )
    return "not every operation was confirmed"


def _place_down_upper_part(operation: object) -> str | None:
    """Return the placed/moving part targeted by PLACE_DOWN or PLACE_UP.

    Returns:
        The upper part for PLACE_DOWN, or the lower moving part for PLACE_UP.
    """
    target = getattr(operation, "target", None)
    values = getattr(target, "values", None)
    if not isinstance(values, Mapping):
        return None
    skill = getattr(operation, "skill", None)
    key = (
        "lower_physical_instance_id"
        if skill == "PLACE_UP"
        else "upper_physical_instance_id"
    )
    value = values.get(key)
    return value if isinstance(value, str) and value else None


def _evidence_label_proven(evidence: object, label: str) -> bool:
    """Return whether one required evidence label carries an explicit proof.

    Mirrors the frozen scheduler predicate: the label must be present in the
    evidence mapping and its value must be neither ``None`` nor ``False``.

    Args:
        evidence: The serialized per-operation evidence mapping (or anything
            else found under the ``evidence`` key).
        label: Required evidence label name.

    Returns:
        True only when the label is present with an explicit value.
    """
    if not isinstance(evidence, Mapping) or label not in evidence:
        return False
    value = evidence[label]
    return value is not None and value is not False


def _pending_assembly_effect_check(
    stap_plan: object,
    operation_plan: object,
    run_document: Mapping[str, object],
) -> dict[str, object]:
    """Verify every pending STAP task's assembly effect is evidenced.

    This is the final semantic gate of a dispatch run: ``EXECUTED`` requires
    each pending STAP task to have exactly one ``PLACE_DOWN`` operation in the
    external plan that executes that task's part on that task's allocated
    robot, whose serialized scheduler result is ``CONFIRMED``/``passed`` and
    whose evidence carries every required connection, release and retreat
    label. A binding of the external plan to the STAP plan, a CONFIRMED
    observation of another skill, or a missing operation is never assembly
    effect evidence, and the check is fail-closed on any missing or
    malformed input.

    Args:
        stap_plan: The authoritative validated STAP plan.
        operation_plan: The parsed external operation plan.
        run_document: The serialized ``rocobrick-operation-plan-run/v1``
            document of the scheduler run.

    Returns:
        A JSON-safe evidence document with ``passed``, the pending, evidenced
        and missing task ids, and one per-task verification entry.
    """
    tasks = getattr(stap_plan, "tasks", None)
    operations = getattr(operation_plan, "operations", None)
    run_operations = run_document.get("operations")
    run_operations = run_operations if isinstance(run_operations, Mapping) else {}
    document: dict[str, object] = {
        "schema": ASSEMBLY_EFFECT_SCHEMA,
        "required_evidence_labels": list(ASSEMBLY_EVIDENCE_LABELS),
        "pending_task_ids": [],
        "evidenced_task_ids": [],
        "missing_task_ids": [],
        "tasks": [],
        "passed": False,
        "reason": None,
    }
    if not isinstance(tasks, Mapping) or not isinstance(operations, Mapping):
        document["reason"] = (
            "the authoritative STAP tasks or the parsed operation plan are "
            "not available for the assembly-effect check"
        )
        return document
    entries: list[dict[str, object]] = []
    pending: list[str] = []
    evidenced: list[str] = []
    missing: list[str] = []
    for task_id, task in tasks.items():
        task_name = str(task_id)
        part_id = getattr(task, "physical_instance_id", None)
        robot_id = getattr(task, "primary_robot", None)
        skill = str(getattr(task, "skill", ""))
        entry: dict[str, object] = {
            "task_id": task_name,
            "physical_instance_id": part_id,
            "primary_robot": robot_id,
            "skill": skill,
            "required_evidence_labels": list(ASSEMBLY_EVIDENCE_LABELS),
            "operation_id": None,
            "operation_status": None,
            "operation_passed": None,
            "missing_evidence_labels": list(ASSEMBLY_EVIDENCE_LABELS),
            "confirmed": False,
            "reason": None,
        }
        pending.append(task_name)
        matching = [
            operation
            for operation in operations.values()
            if getattr(operation, "skill", None)
            in {PENDING_TASK_SKILL, UPWARD_TASK_SKILL}
            and getattr(operation, "source_task_id", None) == task_id
            and getattr(operation, "executor_robot_id", None) == robot_id
            and _place_down_upper_part(operation) == part_id
        ]
        if len(matching) != 1:
            entry["reason"] = (
                f"the external plan has {len(matching)} PLACE_DOWN/PLACE_UP "
                f"placement operations for STAP task "
                f"{task_name!r} (part {part_id!r} on "
                f"robot {robot_id!r}); exactly one is required"
            )
            missing.append(task_name)
            entries.append(entry)
            continue
        operation = matching[0]
        operation_id = getattr(operation, "operation_id", None)
        entry["operation_id"] = operation_id
        result = (
            run_operations.get(operation_id) if isinstance(operation_id, str) else None
        )
        if not isinstance(result, Mapping):
            entry["reason"] = (
                f"operation {operation_id!r} has no scheduler result in the "
                "run document"
            )
            missing.append(task_name)
            entries.append(entry)
            continue
        status = result.get("status")
        passed = result.get("passed") is True
        entry["operation_status"] = status
        entry["operation_passed"] = passed
        missing_labels = [
            label
            for label in ASSEMBLY_EVIDENCE_LABELS
            if not _evidence_label_proven(result.get("evidence"), label)
        ]
        entry["missing_evidence_labels"] = missing_labels
        if status != "CONFIRMED" or not passed:
            entry["reason"] = (
                f"operation {operation_id!r} for STAP task {task_name!r} "
                f"ended {status!r} (passed={passed!r}) instead of CONFIRMED"
            )
            missing.append(task_name)
        elif missing_labels:
            entry["reason"] = (
                f"operation {operation_id!r} for STAP task {task_name!r} is "
                f"CONFIRMED but lacks the assembly evidence labels "
                f"{missing_labels}"
            )
            missing.append(task_name)
        else:
            entry["confirmed"] = True
            evidenced.append(task_name)
        entries.append(entry)
    document["pending_task_ids"] = pending
    document["evidenced_task_ids"] = evidenced
    document["missing_task_ids"] = missing
    document["tasks"] = entries
    document["passed"] = not missing
    if missing:
        document["reason"] = (
            "pending STAP assembly effects are not evidenced for tasks "
            f"{missing}; each pending task requires one CONFIRMED "
            f"{PENDING_TASK_SKILL} operation for its part and robot with the "
            f"evidence labels {list(ASSEMBLY_EVIDENCE_LABELS)}"
        )
    return document


def _robot_operation_plan_phase(
    args: "_Arguments",
    record: dict[str, object],
    recorder: EvidenceRecorder,
    contract: TcpContract | None,
    stap_plan: object,
) -> tuple[RunStatus, str] | None:
    """Validate the external plan, authorize, run the session, aggregate.

    Every gate — external plan capture/parse/cross-binding and the strict TCP
    authorization — completes before the real multi-robot session is created.
    The session is started for every validated plan robot; a readiness
    failure schedules zero operations. The plan itself runs through
    ``OperationPlanRunner`` in strict mode over ``SkillOperationExecutor``,
    which observes the session's real read-only scene transport, and the
    session is stopped with cleanup evidence in a ``finally`` block on every
    path. A final semantic check then requires every pending STAP task's
    assembly effect to be evidenced by one CONFIRMED ``PLACE_DOWN`` operation
    for that exact task/part/robot with the required connection, release and
    retreat labels; ``robot-operation-plan-run.json`` is left in ``record``
    for the caller to finalize with the post-phase input digests.

    Args:
        args: Parsed command-line arguments with the external plan and the
            explicit scene option group.
        record: The mutable preflight record, updated in place.
        recorder: The run evidence recorder.
        contract: The re-derived TCP contract from the TCP gate.
        stap_plan: The authoritative validated STAP plan.

    Returns:
        ``None`` when every robot was ready, every operation was confirmed,
        every pending STAP assembly effect is evidenced, the cleanup passed
        and the inputs stayed immutable (the caller reports ``EXECUTED``),
        otherwise the terminal status and reason.
    """
    from rocobrick.adapters.moveit.transit import (
        SessionTcpMotionAdapter,
        load_transit_contexts,
    )
    from rocobrick.execution.operation_state_ledger import OperationStateLedger
    from rocobrick.execution.robot_operation_executor import SkillOperationExecutor
    from rocobrick.robot_operation_runner import (
        RUN_SCHEMA,
        OperationPlanRunner,
        serialize_run_result,
    )
    from rocobrick.skills.pick import PickHandler
    from rocobrick.skills.place_down import PlaceDownHandler
    from rocobrick.skills.place_up import PlaceUpHandler
    from rocobrick.skills.support_top import SupportTopHandler
    from rocobrick.tcp_contract import EXPECTED_RUNTIME_MODEL_SHA256
    from rocobrick.unified_runtime_facade import UnifiedDualArmRuntimeFacade

    plan_record: dict[str, object] = {
        "path": str(args.robot_operation_plan_path),
        "captured_path": str(recorder.directory / PLAN_CAPTURE_NAME),
        "schema": None,
        "plan_id": None,
        "source_plan_id": None,
        "robots": None,
        "operations": [],
        "execution_layers": [],
        "planned_schedule": [],
        "sha256": None,
        "bytes": None,
        "validated": False,
        "source_plan_bound": stap_plan is not None,
        "error": None,
    }
    record["robot_operation_plan"] = plan_record
    try:
        # The original bytes are captured verbatim into the run directory and
        # the capture is what gets parsed, so the audited copy and the parsed
        # bytes can never diverge.
        recorder.capture_input(PLAN_CAPTURE_NAME, args.robot_operation_plan_path)
        raw = (recorder.directory / PLAN_CAPTURE_NAME).read_bytes()
    except OSError as error:
        reason = f"the robot operation plan is unreadable: {error}"
        plan_record["error"] = reason
        recorder.append_event(
            {
                "phase": "robot_operation_plan",
                "status": "REJECTED",
                "reason": reason,
            }
        )
        return RunStatus.INVALID_INPUT, reason
    plan_record["sha256"] = sha256(raw).hexdigest()
    plan_record["bytes"] = len(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        reason = f"the robot operation plan is not valid UTF-8 JSON: {error}"
        plan_record["error"] = reason
        recorder.append_event(
            {
                "phase": "robot_operation_plan",
                "status": "REJECTED",
                "reason": reason,
                "sha256": plan_record["sha256"],
            }
        )
        return RunStatus.INVALID_INPUT, reason
    try:
        operation_plan = parse_robot_operation_plan(payload, source_plan=stap_plan)
    except RobotOperationPlanError as error:
        reason = (
            f"the robot operation plan failed strict validation against the "
            f"STAP binding: {error}"
        )
        plan_record["error"] = reason
        recorder.append_event(
            {
                "phase": "robot_operation_plan",
                "status": "REJECTED",
                "reason": reason,
                "sha256": plan_record["sha256"],
            }
        )
        return RunStatus.INVALID_INPUT, reason
    from rocobrick.skills.handover_profile import resolve_handover_profile

    try:
        handover_profiles = [
            resolve_handover_profile(op.parameters["grasp_profile_id"])
            for op in operation_plan.operations.values()
            if op.skill == "HANDOVER"
        ]
    except ValueError as error:
        plan_record["error"] = str(error)
        return RunStatus.INVALID_INPUT, str(error)
    plan_record["handover_profiles"] = [
        {
            "profile_id": p.profile_id,
            "sha256": p.sha256,
            "execution_qualified": p.execution_qualified,
        }
        for p in handover_profiles
    ]
    plan_record.update(
        {
            "schema": operation_plan.schema,
            "plan_id": operation_plan.plan_id,
            "source_plan_id": operation_plan.source_plan_id,
            "robots": list(operation_plan.robots),
            "operations": [
                {
                    "operation_id": operation.operation_id,
                    "source_task_id": operation.source_task_id,
                    "skill": operation.skill,
                    "executor_robot_id": operation.executor_robot_id,
                    "participants": list(operation.participants),
                    "resources": list(operation.resources),
                    "evidence_requirements": list(operation.evidence_requirements),
                    "target": (
                        dict(operation.target.values)
                        if operation.target is not None
                        else None
                    ),
                }
                for operation in operation_plan.operations.values()
            ],
            "execution_layers": [
                list(layer) for layer in operation_plan.execution_layers
            ],
            "planned_schedule": [
                [
                    {
                        "operation_id": item.operation_id,
                        "robot_id": item.robot_id,
                    }
                    for item in slot
                ]
                for slot in operation_plan.planned_schedule
            ],
            "validated": True,
        }
    )
    record["checks"]["robot_operation_plan_validated"] = True
    record["checks"]["robot_operation_plan_cross_bound"] = True
    recorder.append_event(
        {
            "phase": "robot_operation_plan",
            "status": "VALIDATED",
            "schema": operation_plan.schema,
            "plan_id": operation_plan.plan_id,
            "source_plan_id": operation_plan.source_plan_id,
            "robots": list(operation_plan.robots),
            "operation_ids": list(operation_plan.operations),
            "execution_layers": [
                list(layer) for layer in operation_plan.execution_layers
            ],
            "sha256": plan_record["sha256"],
            "captured_path": plan_record["captured_path"],
        }
    )

    authorized = _authorize_tcp_dispatch(args, record, recorder, contract)
    if isinstance(authorized[0], RunStatus):
        return authorized

    handover_simulation = os.environ.get("ROCOBRICK_HANDOVER_SIMULATION") == "1"
    if handover_simulation and args.backend != "isaac":
        return RunStatus.INVALID_INPUT, "HANDOVER simulation trial requires the Isaac backend"
    if any(not p.execution_qualified for p in handover_profiles) and not handover_simulation:
        reason = (
            "HANDOVER_PROFILE_NOT_QUALIFIED: coordinated bottom handover is "
            "offline-planning only until contact, constraints and synchronized "
            "execution are verified; no runtime created and no motion dispatched"
        )
        recorder.append_event(
            {"phase": "handover_admission", "status": "REJECTED", "reason": reason}
        )
        return RunStatus.EXECUTION_NOT_IMPLEMENTED, reason

    # Do not allow changing a profile flag to select the legacy PICK-target
    # handler. The new production providers must be integrated explicitly.
    if handover_profiles and not handover_simulation:
        reason = (
            "HANDOVER_RUNTIME_CAPABILITY_UNAVAILABLE: isolated joint planner, "
            "physics pad-contact reports and constrained synchronized execution "
            "providers are not integrated; no runtime or motion dispatched"
        )
        recorder.append_event(
            {"phase": "handover_admission", "status": "REJECTED", "reason": reason}
        )
        return RunStatus.EXECUTION_NOT_IMPLEMENTED, reason

    run_document: dict[str, object] = {
        "schema": RUN_SCHEMA,
        "passed": False,
        "scheduler_ran": False,
        "scheduler_passed": False,
        "operations": {},
        "layers": [],
        "evidence": [],
        "resources": [],
        "cancellations": [],
        "unsafe_terminals": [],
        "readiness": None,
        "cleanup": None,
        "cleanup_passed": False,
        "motion_sent": False,
    }
    record["robot_operation_plan_run"] = run_document

    session = None
    readiness_document: dict[str, object] | None = None
    cleanup_document: dict[str, object] | None = None
    cleanup_passed = False
    runtime_terminal: tuple[RunStatus, str] | None = None
    scheduler_result = None
    try:
        try:
            session = _create_session(
                recorder.directory / "runtime-session",
                args,
                operation_plan.robots,
            )
        except Exception as error:
            reason = f"the multi-robot session could not be created: {error}"
            recorder.append_event(
                {
                    "phase": "runtime_start",
                    "status": "FAILED",
                    "reason": reason,
                    "session_dir": None,
                }
            )
            runtime_terminal = (RunStatus.RUNTIME_START_FAILED, reason)
        if runtime_terminal is None:
            recorder.append_event(
                {
                    "phase": "runtime_start",
                    "status": "STARTING",
                    "session_dir": str(session.session_dir),
                }
            )
            try:
                readiness = session.start()
            except Exception as error:
                reason = f"the multi-robot session failed to start: {error}"
                recorder.append_event(
                    {
                        "phase": "runtime_start",
                        "status": "FAILED",
                        "reason": reason,
                        "session_dir": str(session.session_dir),
                    }
                )
                runtime_terminal = (RunStatus.RUNTIME_START_FAILED, reason)
            else:
                try:
                    readiness_document = _evidence_document(readiness, "readiness")
                except (TypeError, ValueError) as error:
                    reason = (
                        "the multi-robot session readiness evidence is not "
                        f"serializable: {error}"
                    )
                    run_document["readiness_rejection"] = reason
                    recorder.append_event(
                        {
                            "phase": "runtime_start",
                            "status": "FAILED",
                            "reason": reason,
                            "session_dir": str(session.session_dir),
                        }
                    )
                    runtime_terminal = (
                        RunStatus.RUNTIME_START_FAILED,
                        reason,
                    )
                else:
                    rejection = _readiness_rejection(
                        readiness_document, operation_plan.robots
                    )
                    if rejection is not None:
                        run_document["readiness_rejection"] = rejection
                        recorder.append_event(
                            {
                                "phase": "runtime_start",
                                "status": "FAILED",
                                "reason": rejection,
                                "readiness": readiness_document,
                                "session_dir": str(session.session_dir),
                            }
                        )
                        runtime_terminal = (
                            RunStatus.RUNTIME_START_FAILED,
                            rejection,
                        )
                    else:
                        record["checks"]["robot_operation_plan_readiness_passed"] = True
                        recorder.append_event(
                            {
                                "phase": "runtime_start",
                                "status": "READY",
                                "passed": True,
                                "readiness": readiness_document,
                                "session_dir": str(session.session_dir),
                            }
                        )
        if runtime_terminal is None:
            try:
                layer_timeout_overrides = {}
                handover_simulation_evidence = None
                if hasattr(session, "run_motion_operation"):
                    if contract is None or contract.transform is None:
                        raise RuntimeError(
                            "authorized TCP contract has no calibration transform"
                        )
                    transit_contexts = load_transit_contexts(
                        args.config_path or REPO_ROOT / "config/user_config.json",
                        REPO_ROOT / "config/skill_params.json",
                        operation_plan.robots,
                        runtime_model_sha256=EXPECTED_RUNTIME_MODEL_SHA256,
                        source_sha256=contract.transform.sha256,
                    )
                    transit_adapters = {
                        robot_id: SessionTcpMotionAdapter(session, robot_id)
                        for robot_id in operation_plan.robots
                    }
                    state_ledger = OperationStateLedger(operation_plan.robots)
                    unified_runtime = UnifiedDualArmRuntimeFacade(
                        session,
                        transit_contexts,
                        state_ledger=state_ledger,
                    )
                    pick_backends = {}
                    place_backends = {}
                    place_up_backends = {}
                    support_backends = {}
                    handover_backend = None
                    if any(
                        item.skill
                        in ("PICK", "HANDOVER", "PLACE_DOWN", "PLACE_UP", "SUPPORT_TOP")
                        for item in operation_plan.operations.values()
                    ):
                        from rocobrick.isaac_snapshot_host_adapter import (
                            load_scene_snapshot_manifest,
                        )

                        if args.scene_manifest is None:
                            raise RuntimeError(
                                "PICK/PLACE/SUPPORT operations require a scene manifest"
                            )
                        manifest = load_scene_snapshot_manifest(args.scene_manifest)
                        if any(
                            item.skill == "PICK"
                            for item in operation_plan.operations.values()
                        ):
                            top_grasps = {}
                            if os.environ.get("ROCOBRICK_TOP_OFFSET_PICK") == "1":
                                from rocobrick.plan.structure_scene import (
                                    build_structure_scene_plan_from_files,
                                )
                                from rocobrick.skills.top_offset_grasp import (
                                    top_offset_bindings,
                                )

                                structure = stap_plan
                                if structure is None:
                                    structure, _ = (
                                        build_structure_scene_plan_from_files(
                                            recorder.directory / "structure_start.json"
                                            if args.structure_start is not None
                                            else None,
                                            recorder.directory / "structure_goal.json",
                                            payload,
                                        )
                                    )
                                library = json.loads(
                                    (REPO_ROOT / "config/lego_library.json").read_text()
                                )
                                top_grasps = top_offset_bindings(
                                    structure, operation_plan, library
                                )
                            pick_backends = {
                                robot_id: PickHandler(
                                    session,
                                    state_ledger,
                                    session_id=args.host_instance_id,
                                    manifest_sha256=manifest.manifest_sha256,
                                    motion_transport=session,
                                    free_space_transport=unified_runtime,
                                    context=transit_contexts[robot_id],
                                    top_offset_grasps=top_grasps,
                                )
                                for robot_id in operation_plan.robots
                            }
                        if any(
                            item.skill == "PLACE_DOWN"
                            for item in operation_plan.operations.values()
                        ):
                            # Explicit validation switch: probe stops at the
                            # FIRST measured interface contact, withdraws and
                            # fails closed; it never publishes a placed state.
                            place_probe = (
                                os.environ.get("ROCOBRICK_PLACE_CONTACT_PROBE") == "1"
                            )
                            place_backends = {
                                robot_id: PlaceDownHandler(
                                    session,
                                    state_ledger,
                                    session_id=args.host_instance_id,
                                    manifest_sha256=manifest.manifest_sha256,
                                    motion_transport=session,
                                    free_space_transport=unified_runtime,
                                    context=transit_contexts[robot_id],
                                    probe_contact=place_probe,
                                )
                                for robot_id in operation_plan.robots
                            }
                        if any(
                            item.skill == "PLACE_UP"
                            for item in operation_plan.operations.values()
                        ):
                            place_up_backends = {
                                robot_id: PlaceUpHandler(
                                    session,
                                    state_ledger,
                                    session_id=args.host_instance_id,
                                    manifest_sha256=manifest.manifest_sha256,
                                    motion_transport=session,
                                    free_space_transport=unified_runtime,
                                    context=transit_contexts[robot_id],
                                )
                                for robot_id in operation_plan.robots
                            }
                        if any(
                            item.skill == "SUPPORT_TOP"
                            for item in operation_plan.operations.values()
                        ):
                            support_backends = {
                                robot_id: SupportTopHandler(
                                    session,
                                    state_ledger,
                                    session_id=args.host_instance_id,
                                    manifest_sha256=manifest.manifest_sha256,
                                    motion_transport=session,
                                    free_space_transport=unified_runtime,
                                    context=transit_contexts[robot_id],
                                )
                                for robot_id in operation_plan.robots
                            }
                    # Candidate planning and continuous upward insertion need
                    # their actual phase budgets, not the legacy 60 s layer
                    # deadline. Geometry/contact safety checks stay unchanged.
                    for index, layer in enumerate(operation_plan.execution_layers):
                        for oid in layer:
                            item = operation_plan.operations[oid]
                            budget = 0.0
                            if item.skill == "PICK" and item.parameters.get("successor_skill") == "PLACE_UP":
                                budget = 300.0
                            elif item.skill == "PLACE_UP":
                                budget = 360.0
                            if budget:
                                layer_timeout_overrides[index] = max(layer_timeout_overrides.get(index, 0.0), budget)
                    if handover_profiles:
                        from rocobrick.skills.direct_handover import (
                            DirectSimulationHandover,
                        )
                        config_path = args.config_path or REPO_ROOT / "config/user_config.json"
                        config_document = json.loads(config_path.read_text(encoding="utf-8"))
                        robot_rows = config_document["Robot_Config"]["Robots"]
                        package_dirs = tuple(sorted({
                            (config_path.parent / row["Robot_Package_Dir"]).resolve()
                            for row in robot_rows if row["Name"] in operation_plan.robots
                        }))
                        live_handover = DirectSimulationHandover(
                            session, state_ledger, transit_contexts,
                            session_id=args.host_instance_id,
                            manifest_sha256=manifest.manifest_sha256,
                            package_dirs=package_dirs,
                            profile_id=handover_profiles[0].profile_id,
                        )
                        # Direct paired preparation uses the proven combined
                        # MoveIt path; the formal backend continues from its
                        # measured arrival through contact and transfer.
                        handover_backend = live_handover.backend
                        layer_timeout_overrides.update({
                            index: 1200.0
                            for index, layer in enumerate(operation_plan.execution_layers)
                            if any(
                                operation_plan.operations[oid].skill == "HANDOVER"
                                for oid in layer
                            )
                        })
                        handover_simulation_evidence = {
                            "explicit_opt_in": True,
                            "hardware_qualified": False,
                            "profile_sha256": live_handover.profile.sha256,
                            "contact_position_tolerance_m": live_handover.profile.contact_position_tolerance_m,
                            "finger_contact_policy": asdict(live_handover.finger_policy),
                            "finger_contact_policy_sha256": live_handover.finger_policy.sha256,
                            "path": "original_pick_lift_then_direct_preparation_formal_handover",
                            "completion_scope": "full_handover_with_cohold_and_owner_transfer",
                            "layer_timeout_s_by_index": {
                                str(k): v for k, v in layer_timeout_overrides.items()
                            },
                        }

                    executor = SkillOperationExecutor(
                        operation_plan.robots,
                        session,
                        transit_adapters=transit_adapters,
                        transit_contexts=transit_contexts,
                        pick_backends=pick_backends,
                        place_backends=place_backends,
                        place_up_backends=place_up_backends,
                        support_backends=support_backends,
                        handover_backend=handover_backend,
                        pre_phase_runtime=unified_runtime,
                        state_ledger=state_ledger,
                    )
                else:
                    # Keep the narrow observation-only injection seam used by
                    # offline tests and legacy callers. Real sessions expose
                    # run_motion_operation and always take the branch above.
                    executor = SkillOperationExecutor(operation_plan.robots, session)
                runner = OperationPlanRunner(
                    plan=operation_plan,
                    executor=executor,
                    canceler=executor,
                    operation_timeout_s=OPERATION_TIMEOUT_S,
                    cancel_timeout_s=CANCEL_TIMEOUT_S,
                    allow_legacy_results=False,
                    layer_timeout_s_by_index=layer_timeout_overrides,
                )
                scheduler_result = asyncio.run(runner.run())
                run_document = serialize_run_result(
                    scheduler_result, readiness=readiness_document
                )
                run_document["scheduler_ran"] = True
                run_document["scheduler_passed"] = bool(scheduler_result.passed)
                if handover_simulation_evidence is not None:
                    run_document["handover_simulation"] = handover_simulation_evidence
            except Exception as error:
                reason = (
                    "the robot operation plan run failed before a result was "
                    f"produced: {error}"
                )
                recorder.append_event(
                    {
                        "phase": "robot_operation_plan_run",
                        "status": "FAILED",
                        "reason": reason,
                    }
                )
                runtime_terminal = (RunStatus.EXECUTION_FAILED, reason)
            else:
                record["checks"]["robot_operation_plan_scheduler_ran"] = True
                record["checks"]["robot_operation_plan_scheduler_passed"] = bool(
                    scheduler_result.passed
                )
    finally:
        if session is not None:
            try:
                stop_evidence = session.stop()
                cleanup_document = _evidence_document(stop_evidence, "cleanup")
                cleanup_passed = _cleanup_passed(cleanup_document)
            except Exception as error:  # noqa: BLE001 - cleanup is audited
                cleanup_document = {
                    "cleanup_passed": False,
                    "cleanup_error": str(error),
                }
                cleanup_passed = False
            record["runtime"] = cleanup_document
            record["checks"]["robot_operation_plan_cleanup_passed"] = cleanup_passed
            recorder.append_event(
                {
                    "phase": "runtime_stop",
                    "status": "STOPPED" if cleanup_passed else "FAILED",
                    "cleanup_passed": cleanup_passed,
                    "cleanup": cleanup_document,
                }
            )
            if not cleanup_passed and runtime_terminal is None:
                runtime_terminal = (
                    RunStatus.EXECUTION_FAILED,
                    "the multi-robot session cleanup did not pass; the run "
                    "is not verified",
                )
    record["robot_operation_plan_run"] = run_document
    run_document["readiness"] = readiness_document
    run_document["cleanup"] = cleanup_document
    run_document["cleanup_passed"] = cleanup_passed
    operations_document = run_document.get("operations")
    motion_sent = any(
        isinstance(item, Mapping) and item.get("motion_sent") is True
        for item in (
            operations_document.values()
            if isinstance(operations_document, Mapping)
            else ()
        )
    )
    run_document["motion_sent"] = motion_sent
    record["checks"]["motion_sent"] = motion_sent

    # Final semantic check: EXECUTED requires every pending STAP task's
    # assembly effect to be evidenced by a CONFIRMED PLACE_DOWN operation for
    # that exact task/part/robot carrying the required connection, release and
    # retreat labels. The STAP binding alone, or a CONFIRMED observation of
    # another skill, never substitutes for the assembly effect.
    assembly_effects = (
        _pending_assembly_effect_check(stap_plan, operation_plan, run_document)
        if stap_plan is not None
        else {
            "schema": ASSEMBLY_EFFECT_SCHEMA,
            "mode": "robot_operation_plan_primary",
            "passed": True,
            "pending_task_ids": [],
            "evidenced_task_ids": [],
            "missing_task_ids": [],
            "tasks": [],
            "reason": (
                "no assembly-plan/v2 binding is required for "
                "robot-operation-plan simulation"
            ),
        }
    )
    assembly_effects_passed = assembly_effects.get("passed") is True
    run_document["stap_assembly_effects"] = assembly_effects
    record["checks"]["robot_operation_plan_pending_tasks_confirmed"] = (
        assembly_effects_passed
    )
    recorder.append_event(
        {
            "phase": "stap_assembly_effects",
            "status": "CONFIRMED" if assembly_effects_passed else "FAILED",
            "passed": assembly_effects_passed,
            "pending_task_ids": assembly_effects.get("pending_task_ids"),
            "evidenced_task_ids": assembly_effects.get("evidenced_task_ids"),
            "missing_task_ids": assembly_effects.get("missing_task_ids"),
            "tasks": assembly_effects.get("tasks"),
            "reason": assembly_effects.get("reason"),
        }
    )

    if runtime_terminal is not None:
        terminal = runtime_terminal
    elif run_document.get("scheduler_passed") is not True:
        terminal = (
            RunStatus.EXECUTION_FAILED,
            "the robot operation plan did not pass: "
            + _plan_failure_reason(run_document),
        )
    elif not assembly_effects_passed:
        terminal = (
            RunStatus.EXECUTION_FAILED,
            "the run does not evidence every pending STAP assembly effect: "
            + str(
                assembly_effects.get("reason")
                or "missing CONFIRMED PLACE_DOWN evidence"
            ),
        )
    else:
        terminal = None
    recorder.append_event(
        {
            "phase": "robot_operation_plan_run",
            "status": "CONFIRMED" if terminal is None else str(terminal[0]),
            "scheduler_ran": bool(run_document.get("scheduler_ran")),
            "scheduler_passed": bool(run_document.get("scheduler_passed")),
            "pending_stap_tasks_confirmed": assembly_effects_passed,
            "missing_assembly_task_ids": assembly_effects.get("missing_task_ids"),
            "readiness_passed": bool(
                record["checks"].get("robot_operation_plan_readiness_passed")
            ),
            "cleanup_passed": cleanup_passed,
            "motion_sent": motion_sent,
            "operations": {
                operation_id: {
                    "status": item.get("status"),
                    "passed": item.get("passed"),
                    "terminal": item.get("terminal"),
                    "motion_sent": item.get("motion_sent"),
                    "reason": item.get("reason"),
                }
                for operation_id, item in (
                    operations_document.items()
                    if isinstance(operations_document, Mapping)
                    else ()
                )
                if isinstance(item, Mapping)
            },
            "unsafe_terminals": list(run_document.get("unsafe_terminals") or ()),
        }
    )
    return terminal


def _capture_input_record(
    recorder: EvidenceRecorder,
    name: str,
    path: Path | None,
    capture_name: str,
    *,
    capture_blank: bool = True,
) -> tuple[bytes | None, dict[str, object]]:
    """Read, hash and copy one original input of a structure-only run.

    The exact bytes are read once and hashed, and the same file is copied
    verbatim into the run directory under ``capture_name`` so the audited
    copy and the recorded digest can never silently diverge. A read or copy
    failure is recorded in ``entry['error']`` instead of being raised, so the
    caller decides whether that input is required.

    Args:
        recorder: The run evidence recorder that owns the copy.
        name: Evidence name of the input (for example
            ``robot_operation_plan`` or ``structure_goal``).
        path: The input path, or ``None`` when the invocation omitted it.
        capture_name: File name of the verbatim copy in the run directory.
        capture_blank: Whether a blank (empty or whitespace-only) file is
            still copied; a blank start structure is not, so an omitted or
            empty start stays explicit as EMPTY initial structure.

    Returns:
        The exact input bytes (``None`` when the input was omitted or could
        not be read) and the JSON-safe input record.
    """
    entry: dict[str, object] = {
        "path": str(path) if path is not None else None,
        "present": False,
        "blank": True,
        "sha256": None,
        "bytes": None,
        "captured_path": None,
        "error": None,
    }
    if path is None:
        return None, entry
    try:
        raw = path.read_bytes()
    except OSError as error:
        entry["error"] = f"the {name} input {path} is unreadable: {error}"
        return None, entry
    entry["present"] = True
    entry["blank"] = not raw.strip()
    entry["sha256"] = sha256(raw).hexdigest()
    entry["bytes"] = len(raw)
    if entry["blank"] and not capture_blank:
        return raw, entry
    try:
        recorder.capture_input(capture_name, path)
    except OSError as error:
        entry["error"] = (
            f"the {name} input {path} could not be copied into the run "
            f"directory: {error}"
        )
    else:
        entry["captured_path"] = str(recorder.directory / capture_name)
    return raw, entry


def _audit_input_aliasing(
    supplied: Sequence[tuple[str, Path | None]],
) -> tuple[dict[str, str], str | None]:
    """Resolve every supplied input path and detect two names on one file.

    The operation plan, the structure goal, a supplied structure start and
    the optional scene manifest and config are distinct evidence roles; two
    of them resolving to one file (directly, through a symlink, or through
    two spellings of one path) would let one byte stream serve two audited
    roles, so a structure-only preflight must reject that before the binder
    runs. A ``None`` path is an omitted input and never collides.

    Args:
        supplied: ``(input name, path)`` pairs in fixed audit order; a
            ``None`` path means the input was omitted and is skipped.

    Returns:
        The resolved path string of every supplied input, indexed by input
        name (an unresolvable path falls back to its raw spelling), and the
        rejection reason of the first colliding pair, or ``None``.
    """
    resolved: dict[str, str] = {}
    first_owner: dict[str, tuple[str, Path]] = {}
    for name, path in supplied:
        if path is None:
            continue
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError):  # pragma: no cover - defensive
            resolved_path = Path(path)
        resolved[name] = str(resolved_path)
        collision = first_owner.get(resolved[name])
        if collision is not None:
            return resolved, (
                f"the {collision[0]} input {collision[1]} and the {name} "
                f"input {path} resolve to the same file {resolved_path}; the "
                "robot operation plan, the structure goal, a supplied "
                "structure start and the optional scene manifest and config "
                "must be distinct files"
            )
        first_owner[resolved[name]] = (name, path)
    return resolved, None


def _rehash_original_inputs(
    paths: Mapping[str, Path],
) -> tuple[dict[str, str], str | None]:
    """Re-read and re-hash the original input files of a run.

    Args:
        paths: The original input paths indexed by their evidence names.

    Returns:
        The fresh digest of every re-readable path (partial on failure) and
        the first read error, or ``None`` when every path was re-read.
    """
    digests: dict[str, str] = {}
    for name, path in paths.items():
        try:
            digests[name] = sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            return digests, f"the {name} input {path} could not be re-read: {error}"
    return digests, None


def _mutation_detail(
    before: Mapping[str, str],
    after: Mapping[str, str],
    read_error: str | None,
) -> str:
    """Describe why the original input digests of a run are not unchanged.

    Args:
        before: The digests recorded before the phases.
        after: The digests re-read from disk after the phases.
        read_error: The first re-read error, or ``None``.

    Returns:
        The read error, or the exact changed input names.
    """
    if read_error is not None:
        return read_error
    changed = sorted(name for name in before if before[name] != after.get(name))
    return f"the original inputs changed on disk: {changed}"


def _structure_carrier_summary(
    carrier: object, metadata: Mapping[str, object]
) -> dict[str, object]:
    """Summarize one bound structure carrier as JSON-safe evidence.

    The carrier is built from the exact structure files and the validated
    external plan, carrying the goal geometry, the initially assembled ids
    and one PLACE_DOWN task per newly placed part. It is NOT an
    ``assembly-plan/v2`` document and authorizes no dispatch, session, or
    motion.

    Args:
        carrier: The ``StapPlan`` topology carrier, read duck-typed.
        metadata: The builder's JSON-safe binding metadata.

    Returns:
        The JSON-safe structure evidence of the run document.
    """
    tasks = getattr(carrier, "tasks", None)
    parts = getattr(carrier, "goal_parts", None)
    task_evidence: list[dict[str, object]] = []
    if isinstance(tasks, Mapping):
        for task_id, task in sorted(tasks.items(), key=lambda item: str(item[0])):
            task_evidence.append(
                {
                    "task_id": str(task_id),
                    "physical_instance_id": getattr(task, "physical_instance_id", None),
                    "skill": _json_value(getattr(task, "skill", None)),
                    "primary_robot": getattr(task, "primary_robot", None),
                    "supporting_robot": getattr(task, "supporting_robot", None),
                    "initial_supports": sorted(
                        getattr(task, "initial_supports", ()) or ()
                    ),
                }
            )
    return {
        "authoritative": False,
        "authorization": None,
        "note": (
            "structure topology carrier only; it is not an assembly-plan/v2 "
            "document and authorizes no dispatch, session, or motion"
        ),
        "plan_id": getattr(carrier, "plan_id", None),
        "robots": list(getattr(carrier, "robots", ()) or ()),
        "goal_part_ids": sorted(parts) if isinstance(parts, Mapping) else [],
        "initially_assembled": sorted(
            getattr(carrier, "initially_assembled", ()) or ()
        ),
        "tasks": task_evidence,
        "binding": _json_value(metadata),
    }


def _run_robot_operation_plan_primary(
    args: _Arguments, run_dir: Path, recorder: EvidenceRecorder
) -> RunOutcome:
    """Validate and, when authorized, execute a direct operation plan.

    Without an authoritative STAP source this path still validates and binds
    the external operation plan to explicit structure inputs. A complete scene
    option group then permits the shared real runtime for skills with available
    backends; the direct path does not synthesize STAP authorization or
    assembly-effect evidence.
    The explicit structure inputs are captured and bound before the TCP claim is
    re-derived; every step is audited as an event. Before the binder runs, the
    named original inputs are audited for aliasing: the plan, the goal, a
    supplied start and the optional scene manifest and config must resolve to
    distinct files, and an aliased pair ends ``INVALID_INPUT`` with no binding.
    The inputs also record the start choice explicitly; an omitted or
    empty-string start is EMPTY initial structure and never receives an
    invented SHA-256. With a complete runtime option group, the shared runtime
    dispatch is authorized by the same TCP gate and executor used by the STAP
    path. Without that group, the invocation remains a no-runtime preflight.
    A direct run does not synthesize an ``assembly-plan/v2`` STAP plan or
    evaluate STAP assembly effects.


    Args:
        args: Parsed arguments of a structure-only invocation.
        run_dir: The freshly created evidence run directory.
        recorder: The run evidence recorder.

    Returns:
        The terminal non-success outcome, including its exit code.
    """
    from rocobrick.plan.structure_scene import (
        StructureSceneError,
        build_structure_scene_plan_from_files,
    )
    from rocobrick.robot_operation_runner import RUN_SCHEMA

    plan_path = args.robot_operation_plan_path
    goal_path = args.structure_goal
    start_path = args.structure_start
    checks: dict[str, object] = {}
    tcp_record: dict[str, object] = {
        "binding_requires_tcp": True,
        "gate": "NOT_EVALUATED",
        "authorization": {
            "decision": "DENY",
            "allowed": False,
            "requested": False,
            "status": str(RunStatus.EXECUTION_NOT_IMPLEMENTED),
            "reason": (
                f"{NO_STAP_AUTHORIZATION}; the structure-only preflight "
                "never authorizes dispatch"
            ),
            "backend": args.backend,
            "provenance": "no_stap_source_plan",
            "runtime_asset_sha256": None,
            "calibration_artifact_sha256": None,
        },
    }
    plan_record: dict[str, object] = {
        "path": str(plan_path) if plan_path is not None else None,
        "captured_path": str(recorder.directory / PLAN_CAPTURE_NAME),
        "mode": STRUCTURE_ONLY_MODE,
        "schema": None,
        "plan_id": None,
        "source_plan_id": None,
        "robots": None,
        "operations": [],
        "execution_layers": [],
        "planned_schedule": [],
        "sha256": None,
        "bytes": None,
        "validated": False,
        "source_plan_bound": False,
        "stap_authorization": None,
        "error": None,
    }
    record: dict[str, object] = {
        "schema": PREFLIGHT_SCHEMA,
        "mode": STRUCTURE_ONLY_MODE,
        "backend": args.backend,
        "robot_id": args.robot_id,
        "preflight_only": False,
        "robot_operation_plan_path": plan_record["path"],
        "task_dir": None,
        "plan_path": None,
        "plan_id": None,
        "task_id": None,
        "structure_goal_path": (str(goal_path) if goal_path is not None else None),
        "structure_start_path": (str(start_path) if start_path is not None else None),
        "motion_geometry": None,
        "inputs": None,
        "binding": None,
        "structure": None,
        "stap_authorization": None,
        "tcp": tcp_record,
        "dispatch": None,
        "robot_operation_plan": plan_record,
        "robot_operation_plan_run": None,
        "runtime": None,
        "checks": checks,
    }
    recorder.capture_provenance(
        REPO_ROOT,
        {
            "tool": "piper-run",
            "mode": STRUCTURE_ONLY_MODE,
            "backend": args.backend,
            "robot_id": args.robot_id,
            "robot_operation_plan": plan_record["path"],
            "structure_goal": record["structure_goal_path"],
            "structure_start": record["structure_start_path"],
        },
    )
    terminal: tuple[RunStatus, str] | None = None
    input_records: dict[str, dict[str, object]] = {}
    original_paths: dict[str, Path] = {}
    digest_before: dict[str, str] = {}
    plan_bytes: bytes | None = None
    payload: Mapping[str, object] | None = None
    operation_plan = None
    start_path_for_builder: Path | None = None
    # Scene paths are materialized into the fresh run directory by
    # _materialize_direct_defaults, so direct runtime dispatch is available
    # unless the caller explicitly requested the no-runtime preflight.
    direct_dispatch = not args.preflight_only

    # Phase 1: IMMUTABLE_INPUT. Every original input is read once, hashed and
    # copied verbatim; the goal is required and the start, scene manifest and
    # config are captured only when the invocation supplies them.
    if plan_path is None or goal_path is None:
        terminal = (
            RunStatus.INVALID_INPUT,
            "the structure-only --robot-operation-plan preflight requires "
            "both --robot-operation-plan and --structure-goal",
        )
    else:
        plan_bytes, plan_entry = _capture_input_record(
            recorder, "robot_operation_plan", plan_path, PLAN_CAPTURE_NAME
        )
        input_records["robot_operation_plan"] = plan_entry
        _, goal_entry = _capture_input_record(
            recorder, "structure_goal", goal_path, TASK_GOAL_NAME
        )
        input_records["structure_goal"] = goal_entry
        for name, entry in (
            ("robot_operation_plan", plan_entry),
            ("structure_goal", goal_entry),
        ):
            if entry["present"] is not True or entry["captured_path"] is None:
                terminal = (
                    RunStatus.INVALID_INPUT,
                    str(entry["error"] or f"the {name} input could not be captured"),
                )
                break
            digest_before[name] = str(entry["sha256"])
            original_paths[name] = Path(str(entry["path"]))
        _, start_entry = _capture_input_record(
            recorder,
            "structure_start",
            start_path,
            TASK_START_NAME,
            capture_blank=False,
        )
        input_records["structure_start"] = start_entry
        # Explicit start evidence: an omitted flag or the literal empty
        # string means EMPTY initial structure, and a supplied blank file is
        # EMPTY too. An absent start keeps sha256=None: no digest is ever
        # invented for bytes that were never read.
        start_entry["start_provided"] = start_path is not None
        start_entry["start_empty"] = start_path is None or (
            start_entry["present"] is True and start_entry["blank"] is True
        )
        if start_entry["present"] is True:
            if start_entry["blank"] is False:
                if start_entry["captured_path"] is None:
                    terminal = (
                        RunStatus.INVALID_INPUT,
                        str(start_entry["error"]),
                    )
                else:
                    start_path_for_builder = start_path
                    digest_before["structure_start"] = str(start_entry["sha256"])
                    original_paths["structure_start"] = Path(str(start_entry["path"]))
        elif start_entry["error"] is not None:
            terminal = (RunStatus.INVALID_INPUT, str(start_entry["error"]))
        for name, path, capture_name in (
            (
                "scene_manifest",
                args.scene_manifest,
                SCENE_MANIFEST_CAPTURE_NAME,
            ),
            ("config", args.config_path, CONFIG_CAPTURE_NAME),
        ):
            _, entry = _capture_input_record(recorder, name, path, capture_name)
            input_records[name] = entry
            if entry["captured_path"] is not None and path is not None:
                digest_before[name] = str(entry["sha256"])
                original_paths[name] = path
    recorder.append_event(
        {
            "phase": "immutable_input",
            "status": "OK" if terminal is None else "FAILED",
            "mode": STRUCTURE_ONLY_MODE,
            "reason": terminal[1] if terminal is not None else None,
            "inputs": input_records,
        }
    )

    # Phase 1b: INPUT_ALIASING. The named original inputs must stay distinct
    # files; two roles resolving to one file would let one byte stream serve
    # two audited roles, and is rejected strictly before the binder, the TCP
    # gate and any session can exist.
    resolved_input_paths, alias_reason = _audit_input_aliasing(
        (
            ("robot_operation_plan", plan_path),
            ("structure_goal", goal_path),
            ("structure_start", start_path),
            ("scene_manifest", args.scene_manifest),
            ("config", args.config_path),
        )
    )
    checks["inputs_non_aliasing"] = alias_reason is None
    if alias_reason is not None and terminal is None:
        terminal = (RunStatus.INVALID_INPUT, alias_reason)
    recorder.append_event(
        {
            "phase": "input_aliasing",
            "status": "FAILED" if alias_reason is not None else "OK",
            "mode": STRUCTURE_ONLY_MODE,
            "reason": alias_reason,
            "resolved_paths": resolved_input_paths,
        }
    )

    # Phase 2: the external plan is validated with no STAP source plan and is
    # never repaired, reordered, or synthesized into one.
    if terminal is None and plan_bytes is not None:
        plan_record["sha256"] = digest_before["robot_operation_plan"]
        plan_record["bytes"] = len(plan_bytes)
        decoded: object = None
        try:
            decoded = json.loads(plan_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            terminal = (
                RunStatus.INVALID_INPUT,
                f"the robot operation plan is not valid UTF-8 JSON: {error}",
            )
        if terminal is None:
            if isinstance(decoded, Mapping):
                payload = decoded
            else:
                terminal = (
                    RunStatus.INVALID_INPUT,
                    "the robot operation plan must be a JSON object",
                )
    if terminal is None and payload is not None:
        try:
            operation_plan = parse_robot_operation_plan(payload, source_plan=None)
            from rocobrick.skills.handover_profile import resolve_handover_profile

            plan_record["handover_profiles"] = [
                {
                    "profile_id": profile.profile_id,
                    "sha256": profile.sha256,
                    "execution_qualified": profile.execution_qualified,
                }
                for op in operation_plan.operations.values()
                if op.skill == "HANDOVER"
                for profile in [
                    resolve_handover_profile(op.parameters["grasp_profile_id"])
                ]
            ]
        except RobotOperationPlanError as error:
            terminal = (
                RunStatus.INVALID_INPUT,
                "the robot operation plan failed strict structural "
                f"validation without a STAP source plan: {error}",
            )
        except ValueError as error:
            operation_plan = None
            terminal = (RunStatus.INVALID_INPUT, str(error))
    if operation_plan is None:
        plan_record["error"] = terminal[1] if terminal is not None else "not validated"
        recorder.append_event(
            {
                "phase": "robot_operation_plan",
                "status": "REJECTED",
                "mode": STRUCTURE_ONLY_MODE,
                "reason": plan_record["error"],
                "sha256": plan_record["sha256"],
                "bytes": plan_record["bytes"],
                "source_plan_bound": False,
            }
        )
    else:
        plan_record.update(
            {
                "schema": operation_plan.schema,
                "plan_id": operation_plan.plan_id,
                "source_plan_id": operation_plan.source_plan_id,
                "robots": list(operation_plan.robots),
                "operations": [
                    {
                        "operation_id": operation.operation_id,
                        "source_task_id": operation.source_task_id,
                        "skill": operation.skill,
                        "executor_robot_id": operation.executor_robot_id,
                        "participants": list(operation.participants),
                        "resources": list(operation.resources),
                        "evidence_requirements": list(operation.evidence_requirements),
                        "target": (
                            dict(operation.target.values)
                            if operation.target is not None
                            else None
                        ),
                    }
                    for operation in operation_plan.operations.values()
                ],
                "execution_layers": [
                    list(layer) for layer in operation_plan.execution_layers
                ],
                "planned_schedule": [
                    [
                        {
                            "operation_id": item.operation_id,
                            "robot_id": item.robot_id,
                        }
                        for item in slot
                    ]
                    for slot in operation_plan.planned_schedule
                ],
                "validated": True,
            }
        )
        checks["robot_operation_plan_validated"] = True
        direct_dispatch = direct_dispatch and all(
            operation.skill in {
                "HANDOVER",
                "WAIT",
                "PICK",
                "PLACE_DOWN",
                "PLACE_UP",
                "SUPPORT_TOP",
            }
            for operation in operation_plan.operations.values()
        )
        checks["direct_runtime_skills_supported"] = direct_dispatch
        recorder.append_event(
            {
                "phase": "robot_operation_plan",
                "status": "VALIDATED",
                "mode": STRUCTURE_ONLY_MODE,
                "schema": operation_plan.schema,
                "plan_id": operation_plan.plan_id,
                "source_plan_id": operation_plan.source_plan_id,
                "robots": list(operation_plan.robots),
                "operation_ids": list(operation_plan.operations),
                "execution_layers": [
                    list(layer) for layer in operation_plan.execution_layers
                ],
                "sha256": plan_record["sha256"],
                "captured_path": plan_record["captured_path"],
                "source_plan_bound": False,
            }
        )

    # Phase 3: STRUCTURE_SCENE binding, still strictly before any TCP
    # re-derivation and any session creation.
    binding_accepted: bool | None = None
    if operation_plan is not None and terminal is None:
        try:
            carrier, raw_metadata = build_structure_scene_plan_from_files(
                start_path_for_builder,
                goal_path,
                payload,
                manifest=(
                    args.scene_manifest
                    if args.scene_manifest is not None and args.scene_manifest.is_file()
                    else None
                ),
            )
        except StructureSceneError as error:
            binding_accepted = False
            terminal = (
                RunStatus.INVALID_INPUT,
                "the task structure files and the robot operation plan cannot "
                f"be bound: {error}",
            )
        else:
            binding_accepted = True
            normalized = _json_value(raw_metadata)
            metadata: dict[str, object] = (
                normalized if isinstance(normalized, dict) else {"metadata": normalized}
            )
            summary = _structure_carrier_summary(carrier, metadata)
            record["structure"] = summary
            record["binding"] = {
                "accepted": True,
                "kind": "structure-scene",
                "reason": (
                    "the external plan is bound to the exact goal and start "
                    "structure files by physical instance id; no "
                    "assembly-plan/v2 STAP plan was supplied, so this is a "
                    "topology carrier only"
                ),
                "authoritative_stap": None,
                "structure": metadata,
            }
            checks["structure_scene_validated"] = True
            recorder.append_event(
                {
                    "phase": "structure_scene",
                    "status": "VALIDATED",
                    "mode": STRUCTURE_ONLY_MODE,
                    "goal_path": str(goal_path),
                    "start_path": metadata.get("start_path"),
                    "binding": metadata.get("binding"),
                    "plan_id": summary["plan_id"],
                    "robots": summary["robots"],
                    "goal_part_ids": summary["goal_part_ids"],
                    "initial_instance_ids": metadata.get("initial_instance_ids"),
                    "placed_instance_ids": metadata.get("placed_instance_ids"),
                    "goal_sha256": metadata.get("goal_sha256"),
                    "start_sha256": metadata.get("start_sha256"),
                    "manifest_sha256": metadata.get("manifest_sha256"),
                    "authoritative": False,
                }
            )
    if binding_accepted is False:
        record["binding"] = {
            "accepted": False,
            "kind": "structure-scene",
            "reason": terminal[1] if terminal is not None else None,
            "authoritative_stap": None,
            "structure": None,
        }
        recorder.append_event(
            {
                "phase": "structure_scene",
                "status": "REJECTED",
                "mode": STRUCTURE_ONLY_MODE,
                "reason": terminal[1] if terminal is not None else None,
                "authoritative": False,
            }
        )

    # Phase 4: IMMUTABILITY of the original inputs after validation. No gate
    # or session may proceed on bytes that changed while they were validated.
    digest_after_validation, rehash_error = _rehash_original_inputs(original_paths)
    immutable_after_validation = (
        rehash_error is None and digest_after_validation == digest_before
    )
    checks["inputs_immutable_after_validation"] = immutable_after_validation
    recorder.append_event(
        {
            "phase": "immutability",
            "stage": "after_validation",
            "mode": STRUCTURE_ONLY_MODE,
            "status": "OK" if immutable_after_validation else "FAILED",
            "sha256_before": digest_before,
            "sha256_after": digest_after_validation,
            "reason": rehash_error,
        }
    )
    if terminal is None and not immutable_after_validation:
        terminal = (
            RunStatus.INPUT_MUTATED,
            "the original inputs of the structure-only preflight are not "
            "immutable after validation: "
            f"{_mutation_detail(digest_before, digest_after_validation, rehash_error)}",
        )

    # Phase 5: TCP_GATE. The calibrated TCP claim is re-derived and required
    # to resolve; nothing is sent and no session exists for it to authorize.
    if terminal is None:
        gate_status = "PASSED"
        try:
            contract = _rederive_tcp(tcp_record)
            require_resolved_tcp(contract)
            tcp_record["require_resolved_tcp"] = "satisfied"
            checks["tcp_resolved"] = True
        except UnresolvedTcpError as error:
            gate_status = str(RunStatus.UNRESOLVED_TCP)
            terminal = (
                RunStatus.UNRESOLVED_TCP,
                f"{NO_STAP_AUTHORIZATION}: required TCP is unresolved: "
                f"{error}; no session was created and no motion was sent",
            )
        except _TcpGateError as failure:
            gate_status = str(failure.status)
            terminal = (
                failure.status,
                f"{NO_STAP_AUTHORIZATION}: {failure.reason}; no session was "
                "created and no motion was sent",
            )
        tcp_record["gate"] = gate_status
        recorder.append_event(
            {
                "phase": "tcp_gate",
                "status": gate_status,
                "mode": STRUCTURE_ONLY_MODE,
                "tcp": tcp_record,
            }
        )
    else:
        recorder.append_event(
            {
                "phase": "tcp_gate",
                "status": "NOT_EVALUATED",
                "mode": STRUCTURE_ONLY_MODE,
                "reason": terminal[1],
            }
        )

    # Phase 6: execute the validated no-STAP path for skills with real
    # backends, including HANDOVER. The runtime retains all evidence gates.
    if terminal is None and direct_dispatch:
        runtime_terminal = _robot_operation_plan_phase(
            args, record, recorder, contract, None
        )
        terminal = runtime_terminal or (
            RunStatus.EXECUTED,
            "robot operation plan completed with confirmed evidence and "
            "successful cleanup",
        )
    elif terminal is None:
        terminal = (
            (
                RunStatus.PREFLIGHT_ONLY,
                "preflight-only run: all gates evaluated and no "
                "capability is unresolved; dispatch not requested",
            )
            if args.preflight_only
            else (
                RunStatus.EXECUTION_NOT_IMPLEMENTED,
                "the direct operation plan contains a skill without a "
                "real direct-execution backend",
            )
        )

    # Phase 7: IMMUTABILITY after all phases and the terminal documents.
    digest_after_all, final_error = _rehash_original_inputs(original_paths)
    immutable_after_all = final_error is None and digest_after_all == digest_before
    checks["inputs_immutable_after_all_phases"] = immutable_after_all
    if not immutable_after_all and terminal[0] is RunStatus.EXECUTION_NOT_IMPLEMENTED:
        terminal = (
            RunStatus.INPUT_MUTATED,
            "the original inputs of the structure-only preflight are not "
            "immutable after all phases: "
            f"{_mutation_detail(digest_before, digest_after_all, final_error)}",
        )
    start_record = input_records.get("structure_start")
    start_provided = start_path is not None
    start_empty = not start_provided or (
        isinstance(start_record, dict) and start_record.get("start_empty") is True
    )
    record["inputs"] = {
        "records": input_records,
        "start_provided": start_provided,
        "start_empty": start_empty,
        "inputs_non_aliasing": alias_reason is None,
        "resolved_input_paths": resolved_input_paths,
        "sha256_after_validation": digest_after_validation,
        "immutable_after_validation": immutable_after_validation,
        "sha256_after_all_phases": digest_after_all,
        "immutable_after_all_phases": immutable_after_all,
    }
    status, reason = terminal
    if direct_dispatch:
        run_document = record.get("robot_operation_plan_run")
        if not isinstance(run_document, dict):
            run_document = {
                "schema": RUN_SCHEMA,
                "passed": False,
                "operations": {},
                "motion_sent": False,
            }
            record["robot_operation_plan_run"] = run_document
        motion_sent = bool(run_document.get("motion_sent"))
        run_document["status"] = str(status)
        run_document["reason"] = reason
        run_document["passed"] = status is RunStatus.EXECUTED
        record["dispatch"] = {
            "implemented": True,
            "dispatch_requested": True,
            "blocked": False,
            "runtime_created": run_document.get("readiness") is not None,
            "stap_authorization": None,
            "status": str(status),
            "reason": reason,
        }
        checks.update(
            {
                "structure_only": False,
                "stap_source_plan_provided": False,
                "stap_authorization": False,
                "runtime_created": run_document.get("readiness") is not None,
                "motion_sent": motion_sent,
                "no_motion_dispatched": not motion_sent,
            }
        )
        recorder.append_event(
            {
                "phase": "dispatch",
                "status": str(status),
                "mode": "robot_operation_plan_primary",
                "dispatch_requested": True,
                "runtime_created": run_document.get("readiness") is not None,
                "motion_sent": motion_sent,
                "reason": reason,
            }
        )
        record["status"] = str(status)
        record["passed"] = status is RunStatus.EXECUTED
        record["reason"] = reason
        record["exit_code"] = EXIT_CODES[status]
        record["evidence_dir"] = str(recorder.directory)
        recorder.write_json(PLAN_RUN_NAME, run_document)
        recorder.write_json("preflight.json", record)
        result = FinalResult(
            input_valid=(
                status not in (RunStatus.INVALID_INPUT, RunStatus.INPUT_MUTATED)
            ),
            tasks_complete=checks.get("structure_scene_validated") is True,
            structure_valid=checks.get("structure_scene_validated") is True,
            execution_safe=status is RunStatus.EXECUTED,
            retreat_clear=status is RunStatus.EXECUTED,
            evidence_complete=True,
            reason=reason,
        )
        recorder.finalize(result)
        result_path = recorder.directory / "result.json"
        result_document = {
            "schema": RESULT_SCHEMA,
            "status": str(status),
            "passed": result.passed,
            "exit_code": EXIT_CODES[status],
            "run_dir": str(run_dir),
            "preflight_path": str(recorder.directory / "preflight.json"),
            "mode": "robot_operation_plan_primary",
            "runtime_created": run_document.get("readiness") is not None,
            "motion_sent": motion_sent,
            "structure": record.get("structure"),
            "robot_operation_plan": record.get("robot_operation_plan"),
            "robot_operation_plan_run": run_document,
            **asdict(result),
        }
        recorder.write_json("result.json", result_document)
        recorder.append_event(
            {
                "phase": "result",
                "status": str(status),
                "mode": "robot_operation_plan_primary",
                "passed": result.passed,
                "reason": reason,
                "exit_code": EXIT_CODES[status],
            }
        )
        return RunOutcome(
            status=status,
            reason=reason,
            exit_code=EXIT_CODES[status],
            run_dir=run_dir,
            result_path=result_path,
            passed=result.passed,
        )
    record["dispatch"] = {
        "implemented": False,
        "dispatch_requested": False,
        "blocked": True,
        "runtime_created": False,
        "stap_authorization": None,
        "status": str(status),
        "reason": reason,
    }
    checks.update(
        {
            "structure_only": True,
            "stap_source_plan_provided": False,
            "stap_authorization": False,
            "runtime_created": False,
            "motion_sent": False,
            "no_motion_dispatched": True,
        }
    )
    recorder.append_event(
        {
            "phase": "dispatch_blocked",
            "status": str(status),
            "mode": STRUCTURE_ONLY_MODE,
            "dispatch_requested": False,
            "runtime_created": False,
            "stap_authorization": None,
            "reason": reason,
        }
    )
    run_document = _json_value(
        {
            "schema": RUN_SCHEMA,
            "mode": STRUCTURE_ONLY_MODE,
            "status": str(status),
            "reason": reason,
            "passed": False,
            "runtime_created": False,
            "scheduler_ran": False,
            "scheduler_passed": False,
            "motion_sent": False,
            "assembly_passed": False,
            "operations": {},
            "layers": [],
            "evidence": [],
            "resources": [],
            "cancellations": [],
            "unsafe_terminals": [],
            "readiness": None,
            "cleanup": None,
            "cleanup_passed": False,
            "retreat_clear": False,
            "stap_authorization": None,
            "stap_assembly_effects": None,
            "structure": record["structure"],
            "robot_operation_plan": plan_record,
            "tcp": tcp_record,
            "inputs": {
                "sha256_before_validation": digest_before,
                "sha256_after_validation": digest_after_validation,
                "immutable_after_validation": immutable_after_validation,
                "sha256_after_all_phases": digest_after_all,
                "immutable_after_all_phases": immutable_after_all,
            },
        }
    )
    record["robot_operation_plan_run"] = run_document
    recorder.write_json(PLAN_RUN_NAME, run_document)
    record["status"] = str(status)
    record["passed"] = False
    record["reason"] = reason
    record["exit_code"] = EXIT_CODES[status]
    record["evidence_dir"] = str(recorder.directory)
    recorder.write_json("preflight.json", record)
    result = FinalResult(
        input_valid=status not in (RunStatus.INVALID_INPUT, RunStatus.INPUT_MUTATED),
        tasks_complete=False,
        structure_valid=checks.get("structure_scene_validated") is True,
        execution_safe=False,
        retreat_clear=False,
        evidence_complete=True,
        reason=reason,
    )
    recorder.finalize(result)
    result_path = recorder.directory / "result.json"
    result_document: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": str(status),
        "passed": result.passed,
        "exit_code": EXIT_CODES[status],
        "run_dir": str(run_dir),
        "preflight_path": str(recorder.directory / "preflight.json"),
        "mode": STRUCTURE_ONLY_MODE,
        "stap_authorization": None,
        "runtime_created": False,
        "motion_sent": False,
        "assembly_passed": False,
        "structure": record["structure"],
        "robot_operation_plan": plan_record,
        "robot_operation_plan_run": run_document,
        **asdict(result),
    }
    recorder.write_json("result.json", result_document)
    recorder.append_event(
        {
            "phase": "result",
            "status": str(status),
            "mode": STRUCTURE_ONLY_MODE,
            "passed": result.passed,
            "reason": reason,
            "exit_code": EXIT_CODES[status],
        }
    )
    return RunOutcome(
        status=status,
        reason=reason,
        exit_code=EXIT_CODES[status],
        run_dir=run_dir,
        result_path=result_path,
        passed=False,
    )


def _materialize_direct_defaults(args: _Arguments, run_dir: Path) -> _Arguments:
    """Fill internal runtime paths for a direct robot plan.

    Args:
        args: Parsed invocation arguments.
        run_dir: Fresh evidence directory used for generated paths.

    Returns:
        The arguments with direct-runtime defaults materialized; an explicit
        no-runtime preflight request is passed through unchanged so no scene
        path is ever invented for it.
    """
    if args.preflight_only:
        return args
    if args.robot_operation_plan_path is None:
        return args
    return replace(
        args,
        scene_manifest=(
            args.scene_manifest or run_dir / "scene-manifest.generated.json"
        ).resolve(),
        scene_snapshot_out=(
            args.scene_snapshot_out or run_dir / "scene-snapshot.json"
        ).resolve(),
        host_instance_id=(args.host_instance_id or f"piper-run-{run_dir.name}"),
        config_path=(
            args.config_path or REPO_ROOT / "config/user_config.json"
        ).resolve(),
    )


def _run(args: _Arguments) -> RunOutcome:
    """Execute the preflight phases and write the run evidence.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The terminal outcome of the run, including its exit code.

    Raises:
        _RunDirError: If the requested run directory cannot be created
            freshly.
    """
    run_dir = _prepare_run_dir(args.run_dir)
    try:
        recorder = EvidenceRecorder(root=run_dir)
    except OSError as error:
        return RunOutcome(
            status=RunStatus.INTERNAL_ERROR,
            reason=f"cannot open the evidence recorder in {run_dir}: {error}",
            exit_code=EXIT_CODES[RunStatus.INTERNAL_ERROR],
            run_dir=run_dir,
            result_path=None,
        )
    if args.robot_operation_plan_path is not None and args.plan_path is None:
        args = _materialize_direct_defaults(args, run_dir)
        return _run_robot_operation_plan_primary(args, run_dir, recorder)
    recorder.capture_provenance(
        REPO_ROOT,
        {
            "tool": "piper-run",
            "backend": args.backend,
            "robot_id": args.robot_id,
            "preflight_only": args.preflight_only,
            "robot_operation_plan": (
                str(args.robot_operation_plan_path)
                if args.robot_operation_plan_path is not None
                else None
            ),
        },
    )
    record: dict[str, object] = {
        "schema": PREFLIGHT_SCHEMA,
        "backend": args.backend,
        "robot_id": args.robot_id,
        "preflight_only": args.preflight_only,
        "robot_operation_plan_path": (
            str(args.robot_operation_plan_path)
            if args.robot_operation_plan_path is not None
            else None
        ),
        "task_dir": str(args.task_dir),
        "plan_path": str(args.plan_path),
        "plan_id": None,
        "task_id": None,
        "motion_geometry": None,
        "inputs": None,
        "binding": None,
        "tcp": None,
        "dispatch": None,
        "robot_operation_plan": None,
        "robot_operation_plan_run": None,
        "runtime": None,
        "checks": {},
    }
    terminal: tuple[RunStatus, str] | None = None
    inputs: _Inputs | None = None
    binding: StructuralBinding | None = None
    contract: TcpContract | None = None
    try:
        # Phase 1: IMMUTABLE_INPUT + authoritative STAP validation.
        try:
            inputs = _resolve_and_read(args.task_dir, args.plan_path)
        except ValueError as error:
            terminal = (RunStatus.INVALID_INPUT, str(error))
        if inputs is not None:
            record["inputs"] = {
                "task_dir": str(inputs.task_dir),
                "plan_path": str(inputs.plan_path),
                "sha256": dict(inputs.sha256),
                "bytes": dict(inputs.sizes),
            }
            record["checks"]["task_structure_files_present"] = True
            record["checks"]["plan_path_is_file"] = True
            recorder.capture_input("plan.json", inputs.plan_path)
            recorder.capture_input("structure_start.json", inputs.start_path)
            recorder.capture_input("structure_goal.json", inputs.goal_path)
            recorder.append_event(
                {
                    "phase": "immutable_input",
                    "status": "OK",
                    "sha256": dict(inputs.sha256),
                    "bytes": dict(inputs.sizes),
                    "note": (
                        "plan and task structure bytes read exactly once and "
                        "hashed before any parsing; the input paths are never "
                        "written"
                    ),
                }
            )
            try:
                payload = json.loads(inputs.plan_bytes.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as error:
                terminal = (
                    RunStatus.INVALID_INPUT,
                    f"plan is not valid UTF-8 JSON: {error}",
                )
            if terminal is None:
                try:
                    plan = parse_stap_plan(payload)
                except StapContractError as error:
                    terminal = (
                        RunStatus.INVALID_INPUT,
                        f"plan failed authoritative STAP validation: {error}",
                    )
                else:
                    record["plan_id"] = plan.plan_id
                    record["checks"]["stap_validated_authoritatively"] = True
                    recorder.append_event(
                        {
                            "phase": "stap_validated",
                            "status": "OK",
                            "schema": STAP_PLAN_SCHEMA,
                            "plan_id": plan.plan_id,
                            "parser": (
                                "rocobrick.stap_contract.parse_stap_plan "
                                "(the metric plan contract is never imported)"
                            ),
                            "robots": list(plan.robots),
                        }
                    )
        # Phase 2: STRUCTURAL_BINDING.
        if terminal is None:
            decision = bind_single_stap_task(
                plan_bytes=inputs.plan_bytes,
                task_start_bytes=inputs.start_bytes,
                task_goal_bytes=inputs.goal_bytes,
                robot_id=args.robot_id,
            )
            binding = decision.binding
            record["binding"] = {
                "accepted": decision.accepted,
                "reason": decision.reason,
                "binding": asdict(binding) if binding is not None else None,
                "details": dict(decision.details),
            }
            if binding is not None:
                record["plan_id"] = binding.plan_id
                record["task_id"] = binding.task_id
                record["motion_geometry"] = binding.motion_geometry
                record["checks"]["binding_accepted"] = True
            recorder.append_event(
                {
                    "phase": "structural_binding",
                    "status": "ACCEPTED" if decision.accepted else "REJECTED",
                    "reason": decision.reason,
                    "robot_id": args.robot_id,
                    "motion_geometry": (
                        binding.motion_geometry if binding is not None else None
                    ),
                    "binding": asdict(binding) if binding is not None else None,
                    "details": dict(decision.details),
                }
            )
            if not decision.accepted:
                terminal = (
                    RunStatus.BINDING_REJECTED,
                    f"structural binding rejected: {decision.reason}",
                )
        # Phase 3: TCP_GATE.
        requires_tcp = binding_requires_tcp(binding) if binding is not None else None
        tcp_record: dict[str, object] = {"binding_requires_tcp": requires_tcp}
        record["tcp"] = tcp_record
        if terminal is None:
            if requires_tcp:
                gate_status = "PASSED"
                try:
                    contract = _rederive_tcp(tcp_record)
                    require_resolved_tcp(contract)
                    tcp_record["require_resolved_tcp"] = "satisfied"
                    record["checks"]["tcp_resolved"] = True
                except UnresolvedTcpError as error:
                    gate_status = str(RunStatus.UNRESOLVED_TCP)
                    terminal = (
                        RunStatus.UNRESOLVED_TCP,
                        f"required TCP is unresolved: {error}",
                    )
                except _TcpGateError as failure:
                    gate_status = str(failure.status)
                    terminal = (failure.status, failure.reason)
                tcp_record["gate"] = gate_status
            else:
                tcp_record["gate"] = "not_required"
            recorder.append_event(
                {
                    "phase": "tcp_gate",
                    "status": str(terminal[0]) if terminal is not None else "PASSED",
                    "tcp": tcp_record,
                }
            )
        # Phase 4: CAPABILITY / DISPATCH decision. ``--robot-operation-plan``
        # is the only production dispatch entry; without it nothing is ever
        # sent to any runtime. The plan path captures and validates the
        # external document and authorizes the calibrated TCP strictly before
        # any real session exists.
        if binding is not None:
            if (
                terminal is None
                and args.robot_operation_plan_path is not None
                and not args.preflight_only
            ):
                record["dispatch"] = {
                    "implemented": True,
                    "dispatch_requested": True,
                    "blocked": False,
                    "status": "RUNTIME_DISPATCH",
                    "reason": (
                        "explicit --robot-operation-plan dispatch through the "
                        "real multi-robot session"
                    ),
                }
                recorder.append_event(
                    {
                        "phase": "dispatch",
                        "status": "RUNTIME_DISPATCH",
                        "robot_operation_plan": str(args.robot_operation_plan_path),
                    }
                )
                terminal = _robot_operation_plan_phase(
                    args, record, recorder, contract, plan
                )
                if terminal is None:
                    terminal = (
                        RunStatus.EXECUTED,
                        "the robot operation plan executed: every robot was "
                        "ready, every operation was confirmed, every pending "
                        "STAP assembly effect is evidenced, and the "
                        "multi-robot session stopped with passing cleanup",
                    )
                else:
                    record["dispatch"]["blocked"] = True
                record["dispatch"]["status"] = str(terminal[0])
                record["dispatch"]["reason"] = terminal[1]
            elif terminal is None:
                terminal = (
                    (
                        RunStatus.PREFLIGHT_ONLY,
                        "preflight-only run: all gates evaluated and no "
                        "capability is unresolved; dispatch not requested",
                    )
                    if args.preflight_only
                    else (
                        RunStatus.EXECUTION_NOT_IMPLEMENTED,
                        "all gates passed but no dispatch path is implemented "
                        "in piper-run; no command or action message was sent",
                    )
                )
            if record.get("dispatch") is None:
                record["dispatch"] = {
                    "implemented": False,
                    "dispatch_requested": not args.preflight_only,
                    "blocked": True,
                    "status": str(terminal[0]),
                    "reason": terminal[1],
                }
                recorder.append_event(
                    {
                        "phase": "dispatch_blocked",
                        "status": str(terminal[0]),
                        "dispatch_requested": not args.preflight_only,
                        "reason": terminal[1],
                    }
                )
    except Exception as error:
        _diagnostic(format_exc())
        terminal = (
            RunStatus.INTERNAL_ERROR,
            f"unexpected internal error: {error}",
        )
    # Phase 5: IMMUTABILITY of the original input files, including the
    # captured external operation plan when this run requested one.
    run_document = record.get("robot_operation_plan_run")
    plan_record = record.get("robot_operation_plan")
    if inputs is not None:
        originals = {
            "plan": inputs.plan_path,
            "task_start": inputs.start_path,
            "task_goal": inputs.goal_path,
        }
        expected = dict(inputs.sha256)
        sizes = dict(inputs.sizes)
        if (
            isinstance(plan_record, Mapping)
            and isinstance(plan_record.get("sha256"), str)
            and isinstance(plan_record.get("path"), str)
        ):
            originals["robot_operation_plan"] = Path(str(plan_record["path"]))
            expected["robot_operation_plan"] = str(plan_record["sha256"])
            sizes["robot_operation_plan"] = plan_record.get("bytes")
        after: dict[str, str] = {}
        immutable = False
        try:
            after = {
                name: sha256(path.read_bytes()).hexdigest()
                for name, path in originals.items()
            }
        except OSError as error:
            record["checks"]["inputs_immutable_after_all_phases"] = False
            terminal = (
                RunStatus.INPUT_MUTATED,
                f"input files could not be re-read after the phases: {error}",
            )
        else:
            immutable = after == expected
            record["checks"]["inputs_immutable_after_all_phases"] = immutable
            if not immutable:
                terminal = (
                    RunStatus.INPUT_MUTATED,
                    f"input files changed on disk during the run: "
                    f"before={expected} after={after}",
                )
        if isinstance(run_document, dict):
            # The run document is finalized only after every phase, so its
            # ``passed`` verdict can require unchanged inputs and the written
            # copy can carry the post-run digests.
            run_document["inputs"] = {
                "sha256_before": expected,
                "bytes": sizes,
                "sha256_after_all_phases": after,
                "immutable_after_all_phases": immutable,
            }
            run_document["passed"] = bool(
                run_document.get("scheduler_ran") is True
                and run_document.get("scheduler_passed") is True
                and record["checks"].get("robot_operation_plan_readiness_passed")
                is True
                and record["checks"].get("robot_operation_plan_cleanup_passed") is True
                and record["checks"].get("robot_operation_plan_pending_tasks_confirmed")
                is True
                and immutable
            )
    status, reason = (
        terminal
        if terminal is not None
        else (
            RunStatus.INTERNAL_ERROR,
            "the run ended without a terminal status",
        )
    )
    motion_sent = record["checks"].get("motion_sent") is True
    record["checks"]["robot_operation_plan_dispatched"] = (
        args.robot_operation_plan_path is not None
    )
    record["checks"]["runtime_dispatched"] = isinstance(record.get("runtime"), Mapping)
    record["checks"]["no_motion_dispatched"] = not motion_sent
    record["checks"]["inputs_immutable_after_runtime"] = record["checks"].get(
        "inputs_immutable_after_all_phases", False
    )
    robot_plan_passed = bool(
        isinstance(run_document, Mapping) and run_document.get("passed") is True
    )
    record["status"] = str(status)
    record["passed"] = status is RunStatus.PREFLIGHT_ONLY or (
        status is RunStatus.EXECUTED and robot_plan_passed
    )
    record["reason"] = reason
    record["exit_code"] = EXIT_CODES[status]
    record["evidence_dir"] = str(recorder.directory)
    # This phase implements no retreat capability, so the retreat_clear
    # boolean honestly mirrors the overall run result for a plan dispatch.
    retreat_clear = True
    if args.robot_operation_plan_path is not None:
        retreat_clear = bool(record["passed"])
        if isinstance(run_document, dict):
            run_document["retreat_clear"] = retreat_clear
            recorder.write_json(PLAN_RUN_NAME, run_document)
    recorder.write_json("preflight.json", record)
    result = FinalResult(
        input_valid=status not in (RunStatus.INVALID_INPUT, RunStatus.INPUT_MUTATED),
        tasks_complete=binding is not None,
        structure_valid=binding is not None,
        execution_safe=record["passed"],
        retreat_clear=retreat_clear,
        evidence_complete=True,
        reason=reason,
    )
    recorder.finalize(result)
    result_path = recorder.directory / "result.json"
    result_document: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": str(status),
        "passed": result.passed,
        "exit_code": record["exit_code"],
        "run_dir": str(run_dir),
        "preflight_path": str(recorder.directory / "preflight.json"),
        **asdict(result),
    }
    if args.robot_operation_plan_path is not None:
        # Link the external plan record, the aggregated run document (which
        # carries readiness, operations, layers, cancellations, resources,
        # cleanup and the immutable input digests) and the runtime lifecycle
        # from the run result document; the full documents live in the run
        # directory.
        result_document["robot_operation_plan"] = record.get("robot_operation_plan")
        result_document["robot_operation_plan_run"] = record.get(
            "robot_operation_plan_run"
        )
        result_document["runtime_session"] = record.get("runtime")
    recorder.write_json("result.json", result_document)
    recorder.append_event(
        {
            "phase": "result",
            "status": str(status),
            "passed": result.passed,
            "reason": reason,
            "exit_code": record["exit_code"],
        }
    )
    return RunOutcome(
        status=status,
        reason=reason,
        exit_code=EXIT_CODES[status],
        run_dir=run_dir,
        result_path=result_path,
        passed=bool(record["passed"]),
    )


def _structure_start_path(value: object) -> Path | None:
    """Resolve the optional structure start of a structure-only invocation.

    ``--structure-start`` is deliberately not declared with ``type=Path``: an
    omitted flag and the literal empty string both mean EMPTY initial
    structure, and ``Path("")`` would silently erase that distinction as
    ``Path(".")``. Any other value is the caller's start file path.

    Args:
        value: The raw ``--structure-start`` value, or ``None`` when omitted.

    Returns:
        The start path, or ``None`` for EMPTY initial structure.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value != "":
        return Path(value)
    return None


def _parse_arguments(argv: list[str] | None) -> _Arguments:
    """Parse the command line into an ``_Arguments`` value.

    ``--robot-operation-plan`` without ``--task``/``--plan`` selects the
    structure-bound direct path: it requires ``--structure-goal`` (never
    defaulted) and, unless ``--preflight-only`` is passed, executes through
    the real session with scene paths materialized into the run directory;
    ``--preflight-only`` turns the same invocation into the no-runtime
    preflight. With ``--task``/``--plan`` the established STAP path is
    preserved unchanged, including its required ``--backend``, its full scene
    option group when it also carries ``--robot-operation-plan`` and the
    rejection of ``--preflight-only`` combinations with dispatch.

    Args:
        argv: Raw arguments, or ``None`` to use ``sys.argv[1:]``.

    Returns:
        The parsed arguments.

    Raises:
        _UsageError: If the command line is invalid.
    """
    namespace = build_parser().parse_args(argv)
    direct = namespace.robot_operation_plan is not None
    if (
        direct
        and namespace.preflight_only
        and (namespace.task is not None or namespace.plan is not None)
    ):
        raise _UsageError(
            "--robot-operation-plan cannot be combined with --preflight-only"
        )
    structure_only = (
        direct
        and namespace.task is None
        and namespace.plan is None
        and namespace.preflight_only
    )
    if structure_only:
        if namespace.structure_goal is None:
            raise _UsageError(
                "--structure-goal is required for a structure-only "
                "--robot-operation-plan invocation; piper-run never defaults "
                "the goal structure"
            )
        return _Arguments(
            task_dir=None,
            plan_path=None,
            backend=namespace.backend or BACKENDS[0],
            robot_id=namespace.robot_id,
            run_dir=namespace.run_dir,
            preflight_only=True,
            robot_operation_plan_path=namespace.robot_operation_plan,
            scene_manifest=namespace.scene_manifest,
            scene_snapshot_out=namespace.scene_snapshot_out,
            host_instance_id=namespace.host_instance_id,
            config_path=namespace.config,
            structure_dir=namespace.structure_dir,
            structure_goal=namespace.structure_goal,
            structure_start=_structure_start_path(namespace.structure_start),
        )
    if direct and namespace.task is None and namespace.plan is None:
        if namespace.structure_goal is None:
            raise _UsageError(
                "direct robot-operation-plan execution requires "
                "--structure-goal; the goal structure is never inferred"
            )
        return _Arguments(
            task_dir=None,
            plan_path=None,
            backend=namespace.backend or BACKENDS[0],
            robot_id=namespace.robot_id,
            run_dir=namespace.run_dir,
            preflight_only=False,
            robot_operation_plan_path=namespace.robot_operation_plan,
            scene_manifest=namespace.scene_manifest,
            scene_snapshot_out=namespace.scene_snapshot_out,
            host_instance_id=namespace.host_instance_id,
            config_path=namespace.config,
            structure_dir=None,
            structure_goal=namespace.structure_goal,
            structure_start=_structure_start_path(namespace.structure_start),
        )
    if namespace.task is None or namespace.plan is None:
        raise _UsageError(
            "--task and --plan are required together; --robot-operation-plan "
            "alone selects the structure-only preflight and then requires "
            "--structure-goal"
        )
    if namespace.backend is None:
        raise _UsageError("--backend is required")
    if direct and (
        namespace.scene_manifest is None
        or namespace.scene_snapshot_out is None
        or namespace.host_instance_id is None
        or namespace.config is None
    ):
        raise _UsageError(
            "--robot-operation-plan requires --scene-manifest, "
            "--scene-snapshot-out, --host-instance-id and --config with "
            "--task/--plan; there is no fake runtime fallback"
        )
    scene_fields = (
        namespace.scene_manifest,
        namespace.scene_snapshot_out,
        namespace.host_instance_id,
    )
    if any(value is not None for value in scene_fields) and not all(
        value is not None for value in scene_fields
    ):
        raise _UsageError(
            "--scene-manifest, --scene-snapshot-out and --host-instance-id "
            "must be provided together"
        )
    if namespace.scene_manifest is not None and namespace.config is None:
        raise _UsageError(
            "--config must be provided together with --scene-manifest, "
            "--scene-snapshot-out and --host-instance-id"
        )
    return _Arguments(
        task_dir=namespace.task,
        plan_path=namespace.plan,
        backend=namespace.backend,
        robot_id=namespace.robot_id,
        run_dir=namespace.run_dir,
        preflight_only=namespace.preflight_only,
        robot_operation_plan_path=namespace.robot_operation_plan,
        scene_manifest=namespace.scene_manifest,
        scene_snapshot_out=namespace.scene_snapshot_out,
        host_instance_id=namespace.host_instance_id,
        config_path=namespace.config,
        structure_dir=namespace.structure_dir,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the ``piper-run`` preflight CLI.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code (see ``EXIT_CODES``).

    Raises:
        Nothing: every failure is reported through the returned exit code and
        exactly one final JSON line on stdout.
    """
    try:
        args = _parse_arguments(argv)
    except _UsageError as error:
        _diagnostic(f"piper-run: {error}")
        outcome = RunOutcome(
            status=RunStatus.INVALID_INPUT,
            reason=f"invalid usage: {error}",
            exit_code=EXIT_CODES[RunStatus.INVALID_INPUT],
            run_dir=None,
            result_path=None,
        )
    else:
        try:
            outcome = _run(args)
        except _RunDirError as error:
            _diagnostic(f"piper-run: {error}")
            outcome = RunOutcome(
                status=RunStatus.INVALID_INPUT,
                reason=error.reason,
                exit_code=EXIT_CODES[RunStatus.INVALID_INPUT],
                run_dir=error.path,
                result_path=None,
            )
        except Exception as error:  # pragma: no cover - defensive
            _diagnostic(format_exc())
            outcome = RunOutcome(
                status=RunStatus.INTERNAL_ERROR,
                reason=f"unexpected internal error: {error}",
                exit_code=EXIT_CODES[RunStatus.INTERNAL_ERROR],
                run_dir=None,
                result_path=None,
            )
    print(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "status": str(outcome.status),
                "passed": (
                    outcome.status is RunStatus.PREFLIGHT_ONLY
                    if outcome.passed is None
                    else outcome.passed
                ),
                "run_dir": (
                    str(outcome.run_dir) if outcome.run_dir is not None else None
                ),
                "result_path": (
                    str(outcome.result_path)
                    if outcome.result_path is not None
                    else None
                ),
            }
        )
    )
    return outcome.exit_code


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
