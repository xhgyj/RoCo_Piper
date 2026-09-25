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
   :class:`rocobrick.robot_operation_executor.SkillOperationExecutor`, which
   observes the real scene through the session's read-only ``observe``
   transport. In this phase only a genuinely observed WAIT can be
   confirmed; PICK, TRANSIT and PLACE_DOWN fail closed with
   ``CAPABILITY_UNAVAILABLE`` and ``motion_sent=False``. The session is
   stopped with cleanup evidence in a ``finally`` block on every path, and
   ``robot-operation-plan-run.json`` (schema
   ``rocobrick-operation-plan-run/v1``) records readiness, per-robot
   isolated operations, layers, cancellations, resources, cleanup, the
   immutable input digests and the aggregate ``passed`` verdict. A final
   semantic check additionally requires every pending STAP task's assembly
   effect to be evidenced by one CONFIRMED ``PLACE_DOWN`` operation for that
   exact task/part/robot carrying the required connection, release and
   retreat labels; binding the external plan to the STAP plan is never
   assembly evidence, so a run that only confirms an observation (for
   example a WAIT) stays ``EXECUTION_FAILED`` while any task is unassembled.
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

The program is offline and pure apart from its evidence directory and the
explicit ``--robot-operation-plan`` runtime bridge: without it, it imports
no runtime adapter module, and it writes evidence (input copies, phase
events, ``preflight.json``, and ``result.json``) only inside its own run
directory. Exactly one final JSON line is printed to stdout; diagnostics go
to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
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
OPERATION_TIMEOUT_S = 60.0
CANCEL_TIMEOUT_S = 5.0
ASSEMBLY_EFFECT_SCHEMA = "rocobrick-piper-run-stap-assembly-effects/v1"
PENDING_TASK_SKILL = "PLACE_DOWN"
ASSEMBLY_EVIDENCE_LABELS = (
    "connection_confirmed",
    "release_confirmed",
    "retreat_confirmed",
)


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

    task_dir: Path
    plan_path: Path
    backend: str
    robot_id: str
    run_dir: Path | None
    preflight_only: bool
    robot_operation_plan_path: Path | None = None
    scene_manifest: Path | None = None
    scene_snapshot_out: Path | None = None
    host_instance_id: str | None = None
    config_path: Path | None = None


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
        required=True,
        type=Path,
        metavar="TASK_DIR",
        help=(
            "task directory containing structure_start.json and "
            "structure_goal.json (extra files are ignored); read-only"
        ),
    )
    parser.add_argument(
        "--plan",
        required=True,
        type=Path,
        metavar="PLAN_JSON",
        help="assembly-plan/v2 STAP plan JSON file; read-only",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=BACKENDS,
        help="execution backend selected for this run",
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
        "--robot-operation-plan",
        default=None,
        type=Path,
        dest="robot_operation_plan",
        metavar="PLAN_JSON",
        help=(
            "the ONLY production dispatch entry: an external "
            "robot-operation-plan/v1 document, validated against the "
            "read-only assembly-plan/v2 binding and then executed through "
            "the real multi-robot session; requires the full scene option "
            "group and cannot be combined with --preflight-only"
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
        try:
            requested.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise _RunDirError(
                f"run directory {requested} already exists; refusing to "
                "overwrite it",
                requested,
            ) from error
        except OSError as error:
            raise _RunDirError(
                f"cannot create run directory {requested}: {error}", requested
            ) from error
        return requested
    base = Path(RUNS_DIR_NAME)
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
        "pinned_sources": {
            entry["path"]: entry["sha256"] for entry in pinned_files
        },
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
            "the TCP binding artifact records passed=false; its own audit "
            "failed",
        )
    serialized = artifact.get("tcp_contract")
    if (
        not isinstance(serialized, Mapping)
        or not isinstance(serialized.get("status"), str)
    ):
        raise _TcpGateError(
            RunStatus.TCP_CONTRACT_INVALID,
            "the TCP binding artifact carries no serialized "
            "tcp_contract.status string",
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

    config = MultiRobotSessionConfig(
        repo_root=REPO_ROOT,
        robot_ids=robot_ids,
        scene_manifest=args.scene_manifest,
        scene_snapshot_out=args.scene_snapshot_out,
        host_instance_id=args.host_instance_id,
        plan_path=args.plan_path,
        config_path=args.config_path,
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
        if (
            not isinstance(contract, TcpContract)
            or contract.transform is None
        ):
            raise ValueError(
                "the verified re-derived TCP contract has no transform"
            )
        calibration_sha = contract.transform.sha256
        if (
            not isinstance(calibration_sha, str)
            or re.fullmatch(r"^[0-9a-f]{64}$", calibration_sha) is None
        ):
            raise ValueError(
                "the re-derived TCP calibration transform has no valid "
                "SHA-256 digest"
            )
        tcp_authorization["runtime_asset_sha256"] = runtime_sha
        tcp_authorization["calibration_artifact_sha256"] = calibration_sha
    except Exception as error:
        extraction_error = (
            "TCP capability authorization inputs could not be verified: "
            f"{error}"
        )

    if extraction_error is not None:
        tcp_authorization["reason"] = extraction_error
        recorder.append_event({
            "phase": "tcp_authorization",
            **tcp_authorization,
        })
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
        tcp_authorization.update({
            "decision": "ALLOW" if allowed is True else "DENY",
            "allowed": allowed,
            "status": status,
            "reason": reason,
            "checks": checks_copy,
        })
    except Exception as error:
        authorization_error = (
            "TCP capability authorization failed closed: "
            f"{error}"
        )
        tcp_authorization["reason"] = authorization_error
        recorder.append_event({
            "phase": "tcp_authorization",
            **tcp_authorization,
        })
        return (
            RunStatus.UNRESOLVED_TCP,
            authorization_error,
        )

    recorder.append_event({
        "phase": "tcp_authorization",
        **tcp_authorization,
    })
    if tcp_authorization["allowed"] is not True:
        return (
            RunStatus.UNRESOLVED_TCP,
            "TCP capability authorization denied: "
            f"{tcp_authorization['reason']}",
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
        return {
            str(key): _json_value(item)
            for key, item in asdict(value).items()
        }
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
    raise TypeError(
        f"evidence value of type {type(value).__name__} is not JSON-safe"
    )


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
    if (
        document.get("passed") is not True
        or document.get("all_ready") is not True
    ):
        return (
            "the session readiness evidence does not report passed=True and "
            f"all_ready=True: {dict(document)}"
        )
    reported = document.get("robots")
    if not isinstance(reported, (list, tuple)) or list(reported) != list(
        robots
    ):
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
            return (
                f"the readiness evidence carries no report for robot {robot!r}"
            )
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
        leftovers.get("session_containers")
        or leftovers.get("session_processes")
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
    """Return the upper part a PLACE_DOWN operation targets, when declared.

    Args:
        operation: One parsed external ``RobotOperation`` (duck-typed).

    Returns:
        The non-empty ``upper_physical_instance_id`` of the operation target,
        or ``None`` when the operation carries no such target value.
    """
    target = getattr(operation, "target", None)
    values = getattr(target, "values", None)
    if not isinstance(values, Mapping):
        return None
    upper = values.get("upper_physical_instance_id")
    return upper if isinstance(upper, str) and upper else None


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
    run_operations = (
        run_operations if isinstance(run_operations, Mapping) else {}
    )
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
            if getattr(operation, "skill", None) == PENDING_TASK_SKILL
            and getattr(operation, "source_task_id", None) == task_id
            and getattr(operation, "executor_robot_id", None) == robot_id
            and _place_down_upper_part(operation) == part_id
        ]
        if len(matching) != 1:
            entry["reason"] = (
                f"the external plan has {len(matching)} {PENDING_TASK_SKILL} "
                f"operations for STAP task {task_name!r} (part {part_id!r} on "
                f"robot {robot_id!r}); exactly one is required"
            )
            missing.append(task_name)
            entries.append(entry)
            continue
        operation = matching[0]
        operation_id = getattr(operation, "operation_id", None)
        entry["operation_id"] = operation_id
        result = (
            run_operations.get(operation_id)
            if isinstance(operation_id, str)
            else None
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
    from rocobrick.robot_operation_executor import SkillOperationExecutor
    from rocobrick.robot_operation_runner import (
        RUN_SCHEMA,
        OperationPlanRunner,
        serialize_run_result,
    )

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
        recorder.capture_input(
            PLAN_CAPTURE_NAME, args.robot_operation_plan_path
        )
        raw = (recorder.directory / PLAN_CAPTURE_NAME).read_bytes()
    except OSError as error:
        reason = f"the robot operation plan is unreadable: {error}"
        plan_record["error"] = reason
        recorder.append_event({
            "phase": "robot_operation_plan",
            "status": "REJECTED",
            "reason": reason,
        })
        return RunStatus.INVALID_INPUT, reason
    plan_record["sha256"] = sha256(raw).hexdigest()
    plan_record["bytes"] = len(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        reason = (
            f"the robot operation plan is not valid UTF-8 JSON: {error}"
        )
        plan_record["error"] = reason
        recorder.append_event({
            "phase": "robot_operation_plan",
            "status": "REJECTED",
            "reason": reason,
            "sha256": plan_record["sha256"],
        })
        return RunStatus.INVALID_INPUT, reason
    try:
        operation_plan = parse_robot_operation_plan(
            payload, source_plan=stap_plan
        )
    except RobotOperationPlanError as error:
        reason = (
            f"the robot operation plan failed strict validation against the "
            f"STAP binding: {error}"
        )
        plan_record["error"] = reason
        recorder.append_event({
            "phase": "robot_operation_plan",
            "status": "REJECTED",
            "reason": reason,
            "sha256": plan_record["sha256"],
        })
        return RunStatus.INVALID_INPUT, reason
    plan_record.update({
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
                "evidence_requirements": list(
                    operation.evidence_requirements
                ),
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
    })
    record["checks"]["robot_operation_plan_validated"] = True
    record["checks"]["robot_operation_plan_cross_bound"] = True
    recorder.append_event({
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
    })

    authorized = _authorize_tcp_dispatch(args, record, recorder, contract)
    if isinstance(authorized[0], RunStatus):
        return authorized

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
            reason = (
                f"the multi-robot session could not be created: {error}"
            )
            recorder.append_event({
                "phase": "runtime_start",
                "status": "FAILED",
                "reason": reason,
                "session_dir": None,
            })
            runtime_terminal = (RunStatus.RUNTIME_START_FAILED, reason)
        if runtime_terminal is None:
            recorder.append_event({
                "phase": "runtime_start",
                "status": "STARTING",
                "session_dir": str(session.session_dir),
            })
            try:
                readiness = session.start()
            except Exception as error:
                reason = (
                    f"the multi-robot session failed to start: {error}"
                )
                recorder.append_event({
                    "phase": "runtime_start",
                    "status": "FAILED",
                    "reason": reason,
                    "session_dir": str(session.session_dir),
                })
                runtime_terminal = (RunStatus.RUNTIME_START_FAILED, reason)
            else:
                try:
                    readiness_document = _evidence_document(
                        readiness, "readiness"
                    )
                except (TypeError, ValueError) as error:
                    reason = (
                        "the multi-robot session readiness evidence is not "
                        f"serializable: {error}"
                    )
                    run_document["readiness_rejection"] = reason
                    recorder.append_event({
                        "phase": "runtime_start",
                        "status": "FAILED",
                        "reason": reason,
                        "session_dir": str(session.session_dir),
                    })
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
                        recorder.append_event({
                            "phase": "runtime_start",
                            "status": "FAILED",
                            "reason": rejection,
                            "readiness": readiness_document,
                            "session_dir": str(session.session_dir),
                        })
                        runtime_terminal = (
                            RunStatus.RUNTIME_START_FAILED,
                            rejection,
                        )
                    else:
                        record["checks"][
                            "robot_operation_plan_readiness_passed"
                        ] = True
                        recorder.append_event({
                            "phase": "runtime_start",
                            "status": "READY",
                            "passed": True,
                            "readiness": readiness_document,
                            "session_dir": str(session.session_dir),
                        })
        if runtime_terminal is None:
            try:
                executor = SkillOperationExecutor(
                    operation_plan.robots, session
                )
                runner = OperationPlanRunner(
                    plan=operation_plan,
                    executor=executor,
                    canceler=executor,
                    operation_timeout_s=OPERATION_TIMEOUT_S,
                    cancel_timeout_s=CANCEL_TIMEOUT_S,
                    allow_legacy_results=False,
                )
                scheduler_result = asyncio.run(runner.run())
                run_document = serialize_run_result(
                    scheduler_result, readiness=readiness_document
                )
                run_document["scheduler_ran"] = True
                run_document["scheduler_passed"] = bool(
                    scheduler_result.passed
                )
            except Exception as error:
                reason = (
                    "the robot operation plan run failed before a result was "
                    f"produced: {error}"
                )
                recorder.append_event({
                    "phase": "robot_operation_plan_run",
                    "status": "FAILED",
                    "reason": reason,
                })
                runtime_terminal = (RunStatus.EXECUTION_FAILED, reason)
            else:
                record["checks"][
                    "robot_operation_plan_scheduler_ran"
                ] = True
                record["checks"][
                    "robot_operation_plan_scheduler_passed"
                ] = bool(scheduler_result.passed)
    finally:
        if session is not None:
            try:
                stop_evidence = session.stop()
                cleanup_document = _evidence_document(
                    stop_evidence, "cleanup"
                )
                cleanup_passed = _cleanup_passed(cleanup_document)
            except Exception as error:  # noqa: BLE001 - cleanup is audited
                cleanup_document = {
                    "cleanup_passed": False,
                    "cleanup_error": str(error),
                }
                cleanup_passed = False
            record["runtime"] = cleanup_document
            record["checks"][
                "robot_operation_plan_cleanup_passed"
            ] = cleanup_passed
            recorder.append_event({
                "phase": "runtime_stop",
                "status": "STOPPED" if cleanup_passed else "FAILED",
                "cleanup_passed": cleanup_passed,
                "cleanup": cleanup_document,
            })
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
    assembly_effects = _pending_assembly_effect_check(
        stap_plan, operation_plan, run_document
    )
    assembly_effects_passed = assembly_effects.get("passed") is True
    run_document["stap_assembly_effects"] = assembly_effects
    record["checks"]["robot_operation_plan_pending_tasks_confirmed"] = (
        assembly_effects_passed
    )
    recorder.append_event({
        "phase": "stap_assembly_effects",
        "status": "CONFIRMED" if assembly_effects_passed else "FAILED",
        "passed": assembly_effects_passed,
        "pending_task_ids": assembly_effects.get("pending_task_ids"),
        "evidenced_task_ids": assembly_effects.get("evidenced_task_ids"),
        "missing_task_ids": assembly_effects.get("missing_task_ids"),
        "tasks": assembly_effects.get("tasks"),
        "reason": assembly_effects.get("reason"),
    })

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
    recorder.append_event({
        "phase": "robot_operation_plan_run",
        "status": "CONFIRMED" if terminal is None else str(terminal[0]),
        "scheduler_ran": bool(run_document.get("scheduler_ran")),
        "scheduler_passed": bool(run_document.get("scheduler_passed")),
        "pending_stap_tasks_confirmed": assembly_effects_passed,
        "missing_assembly_task_ids": assembly_effects.get(
            "missing_task_ids"
        ),
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
        "unsafe_terminals": list(
            run_document.get("unsafe_terminals") or ()
        ),
    })
    return terminal


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
            recorder.append_event({
                "phase": "immutable_input",
                "status": "OK",
                "sha256": dict(inputs.sha256),
                "bytes": dict(inputs.sizes),
                "note": (
                    "plan and task structure bytes read exactly once and "
                    "hashed before any parsing; the input paths are never "
                    "written"
                ),
            })
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
                    recorder.append_event({
                        "phase": "stap_validated",
                        "status": "OK",
                        "schema": STAP_PLAN_SCHEMA,
                        "plan_id": plan.plan_id,
                        "parser": (
                            "rocobrick.stap_contract.parse_stap_plan "
                            "(the metric plan contract is never imported)"
                        ),
                        "robots": list(plan.robots),
                    })
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
            recorder.append_event({
                "phase": "structural_binding",
                "status": "ACCEPTED" if decision.accepted else "REJECTED",
                "reason": decision.reason,
                "robot_id": args.robot_id,
                "motion_geometry": (
                    binding.motion_geometry if binding is not None else None
                ),
                "binding": asdict(binding) if binding is not None else None,
                "details": dict(decision.details),
            })
            if not decision.accepted:
                terminal = (
                    RunStatus.BINDING_REJECTED,
                    f"structural binding rejected: {decision.reason}",
                )
        # Phase 3: TCP_GATE.
        requires_tcp = (
            binding_requires_tcp(binding) if binding is not None else None
        )
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
            recorder.append_event({
                "phase": "tcp_gate",
                "status": str(terminal[0]) if terminal is not None else "PASSED",
                "tcp": tcp_record,
            })
        # Phase 4: CAPABILITY / DISPATCH decision. ``--robot-operation-plan``
        # is the only production dispatch entry; without it nothing is ever
        # sent to any runtime. The plan path captures and validates the
        # external document and authorizes the calibrated TCP strictly before
        # any real session exists.
        if binding is not None:
            if terminal is None and args.robot_operation_plan_path is not None:
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
                recorder.append_event({
                    "phase": "dispatch",
                    "status": "RUNTIME_DISPATCH",
                    "robot_operation_plan": str(
                        args.robot_operation_plan_path
                    ),
                })
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
                recorder.append_event({
                    "phase": "dispatch_blocked",
                    "status": str(terminal[0]),
                    "dispatch_requested": not args.preflight_only,
                    "reason": terminal[1],
                })
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
            originals["robot_operation_plan"] = Path(
                str(plan_record["path"])
            )
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
                and record["checks"].get(
                    "robot_operation_plan_readiness_passed"
                )
                is True
                and record["checks"].get(
                    "robot_operation_plan_cleanup_passed"
                )
                is True
                and record["checks"].get(
                    "robot_operation_plan_pending_tasks_confirmed"
                )
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
    record["checks"]["runtime_dispatched"] = (
        isinstance(record.get("runtime"), Mapping)
    )
    record["checks"]["no_motion_dispatched"] = not motion_sent
    record["checks"]["inputs_immutable_after_runtime"] = record["checks"].get(
        "inputs_immutable_after_all_phases", False
    )
    robot_plan_passed = bool(
        isinstance(run_document, Mapping)
        and run_document.get("passed") is True
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
        input_valid=status
        not in (RunStatus.INVALID_INPUT, RunStatus.INPUT_MUTATED),
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
        result_document["robot_operation_plan"] = record.get(
            "robot_operation_plan"
        )
        result_document["robot_operation_plan_run"] = record.get(
            "robot_operation_plan_run"
        )
        result_document["runtime_session"] = record.get("runtime")
    recorder.write_json("result.json", result_document)
    recorder.append_event({
        "phase": "result",
        "status": str(status),
        "passed": result.passed,
        "reason": reason,
        "exit_code": record["exit_code"],
    })
    return RunOutcome(
        status=status,
        reason=reason,
        exit_code=EXIT_CODES[status],
        run_dir=run_dir,
        result_path=result_path,
        passed=bool(record["passed"]),
    )


def _parse_arguments(argv: list[str] | None) -> _Arguments:
    """Parse the command line into an ``_Arguments`` value.

    Args:
        argv: Raw arguments, or ``None`` to use ``sys.argv[1:]``.

    Returns:
        The parsed arguments.

    Raises:
        _UsageError: If the command line is invalid.
    """
    namespace = build_parser().parse_args(argv)
    if namespace.robot_operation_plan is not None:
        if namespace.preflight_only:
            raise _UsageError(
                "--robot-operation-plan cannot be combined with "
                "--preflight-only"
            )
        if (
            namespace.scene_manifest is None
            or namespace.scene_snapshot_out is None
            or namespace.host_instance_id is None
            or namespace.config is None
        ):
            raise _UsageError(
                "--robot-operation-plan requires --scene-manifest, "
                "--scene-snapshot-out, --host-instance-id and --config; "
                "there is no fake runtime fallback"
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
        json.dumps({
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
        })
    )
    return outcome.exit_code


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
