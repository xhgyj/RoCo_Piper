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
5. Without ``--operation``, never dispatches: the default path is still the
   preflight bridge. A run whose gates all pass reports
   ``EXECUTION_NOT_IMPLEMENTED`` and exits nonzero; only ``--preflight-only``
   on a plan whose gates all pass reports ``PREFLIGHT_ONLY`` with exit code
   0. No ROS, Isaac, Docker, GPU, MoveIt action, or motion is ever started
   or sent.
6. With the INTERNAL/advanced ``--operation REQUEST_JSON`` bridge (until R5),
   the same immutable gates run FIRST; only then is the formal R4.2 runtime
   session (:class:`rocobrick.runtime_session.IsaacMoveItSession`) started,
   the operation (JOINT_TARGET or LINK6_POSE only, strictly admitted) is
   executed against the live stack, and the session is stopped with cleanup
   evidence. Terminal statuses: ``RUNTIME_START_FAILED`` (8),
   ``OPERATION_REJECTED`` (9), ``EXECUTION_FAILED`` (10), ``EXECUTED`` (0,
   only when the operation passed AND the cleanup passed). This path never
   infers an example1 STAP metric target, and the runtime lifecycle evidence
   plus the operation result are linked from the run result document.
7. Re-hashes the original input files from disk after all phases — including
   after the runtime — and fails with ``INPUT_MUTATED`` if any digest changed
   (IMMUTABILITY).

The program is offline and pure apart from its evidence directory and the
explicit ``--operation`` runtime bridge: without an operation it imports no
runtime adapter module, and it writes evidence (input copies, phase events,
``preflight.json``, and ``result.json``) only inside its own run directory.
Exactly one final JSON line is printed to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from traceback import format_exc
from uuid import uuid4

from rocobrick.compat import StrEnum
from rocobrick.evidence import EvidenceRecorder, FinalResult
from rocobrick.tcp_capability import authorize_tcp
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
    operation_path: Path | None = None
    scene_manifest: Path | None = None
    scene_snapshot_out: Path | None = None
    host_instance_id: str | None = None


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
    parser.add_argument(
        "--operation",
        default=None,
        type=Path,
        metavar="REQUEST_JSON",
        help=(
            "INTERNAL/advanced bridge until R5: a strict RuntimeOperation "
            "request (JOINT_TARGET or LINK6_POSE) executed against the live "
            "Isaac/MoveIt runtime session after every gate passes; cannot be "
            "combined with --preflight-only"
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


def _create_session(session_dir: Path, args: "_Arguments" | None = None):
    """Create the R4.2 runtime session (injection seam for offline tests).

    The production construction is deliberately trivial: the real Isaac/
    MoveIt session with the subprocess runner. Tests monkeypatch this hook
    with a fake session so the full preflight-before-session ordering stays
    verifiable offline.

    Args:
        session_dir: Directory (inside the run evidence directory) that the
            session owns for its lifecycle evidence and operation documents.

    Returns:
        A session exposing ``start()``, ``run_operation()`` and ``stop()``.
    """
    from rocobrick.runtime_session import (
        IsaacMoveItSession,
        RuntimeSessionConfig,
        SubprocessRunner,
    )

    config = RuntimeSessionConfig(
        repo_root=REPO_ROOT,
        scene_manifest=args.scene_manifest if args is not None else None,
        scene_snapshot_out=args.scene_snapshot_out if args is not None else None,
        host_instance_id=args.host_instance_id if args is not None else None,
    )
    return IsaacMoveItSession(
        config, runner=SubprocessRunner(), session_dir=session_dir
    )


def _runtime_operation_phase(
    args: "_Arguments",
    record: dict[str, object],
    recorder: EvidenceRecorder,
    contract: TcpContract | None = None,
) -> tuple[RunStatus, str] | None:
    """Start the runtime session, execute the operation, and stop cleanly.

    Every immutable gate has already passed when this runs. The operation
    document is validated strictly and strictly admitted BEFORE any session
    exists (a rejection therefore starts zero processes). Runtime lifecycle
    and operation evidence are appended as ``runtime_start`` / ``operation`` /
    ``runtime_stop`` events and linked from the record.

    Args:
        args: Parsed command-line arguments with the operation path.
        record: The mutable preflight record, updated in place.
        recorder: The run evidence recorder.

    Returns:
        ``None`` when the operation passed and the cleanup passed (the caller
        reports ``EXECUTED``), otherwise the terminal status and reason.
    """
    from rocobrick.isaac_backend import (
        OPERATION_SCHEMA,
        OperationContractError,
        OperationKind,
        RuntimeOperation,
        target_for_operation,
    )
    from rocobrick.runtime_session import RuntimeSessionError

    operation_record: dict[str, object] = {
        "source": str(args.operation_path),
        "schema": None,
        "kind": None,
        "request_id": None,
        "sha256": None,
        "timeout_s": None,
        "passed": False,
        "result": None,
        "result_path": None,
    }
    record["operation"] = operation_record
    try:
        raw = args.operation_path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        operation = RuntimeOperation.parse(document)
    except (OSError, ValueError) as error:
        return (
            RunStatus.INVALID_INPUT,
            f"the operation document is not a valid RuntimeOperation: {error}",
        )
    operation_record.update(
        {
            "schema": OPERATION_SCHEMA,
            "kind": str(operation.kind.value),
            "request_id": operation.request_id,
            "sha256": sha256(raw).hexdigest(),
            "timeout_s": operation.timeout_s,
        }
    )
    if operation.kind not in (OperationKind.JOINT_TARGET, OperationKind.LINK6_POSE):
        return (
            RunStatus.OPERATION_REJECTED,
            f"operation kind {operation.kind} is not dispatchable; only "
            "JOINT_TARGET and LINK6_POSE are allowed",
        )
    try:
        target_for_operation(operation)
    except OperationContractError as error:
        return (
            RunStatus.OPERATION_REJECTED,
            f"operation target rejected before dispatch: {error}",
        )
    operation_record["target_admitted"] = True
    recorder.write_json("operation-request.json", operation.serialize())
    recorder.append_event({
        "phase": "operation_admission",
        "status": "ACCEPTED",
        "kind": str(operation.kind.value),
        "request_id": operation.request_id,
        "timeout_s": operation.timeout_s,
    })

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
    operation_record["tcp_authorization"] = tcp_authorization
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

    session = None
    readiness = None
    result_map: Mapping[str, object] | None = None
    runtime_terminal: tuple[RunStatus, str] | None = None
    try:
        try:
            # Preserve the historical one-argument seam for callers/tests that
            # provide a session factory, while forwarding scene options to the
            # real factory when the CLI actually requested them.
            if args.scene_manifest is None:
                session = _create_session(recorder.directory / "runtime-session")
            else:
                session = _create_session(recorder.directory / "runtime-session", args)
            recorder.append_event({
                "phase": "runtime_start",
                "status": "STARTING",
                "session_dir": str(session.session_dir),
            })
            readiness = session.start()
            recorder.append_event({
                "phase": "runtime_start",
                "status": "READY",
                "passed": readiness.passed,
                "readiness": asdict(readiness),
                "session_evidence_path": str(
                    session.session_dir / "runtime-session.json"
                ),
            })
        except RuntimeSessionError as error:
            recorder.append_event({
                "phase": "runtime_start",
                "status": "FAILED",
                "reason": str(error),
                "session_dir": (
                    str(session.session_dir) if session is not None else None
                ),
            })
            runtime_terminal = (
                RunStatus.RUNTIME_START_FAILED,
                f"the runtime session failed to start: {error}",
            )
        if runtime_terminal is None:
            result_path = session.session_dir / "operation-result.json"
            try:
                result_map = session.run_operation(
                    recorder.directory / "operation-request.json", result_path
                )
                operation_record["passed"] = result_map.get("passed") is True
                operation_record["result"] = dict(result_map)
                operation_record["result_path"] = str(result_path)
                execution = result_map.get("execution")
                recorder.append_event({
                    "phase": "operation",
                    "status": (
                        str(execution.get("status"))
                        if isinstance(execution, Mapping)
                        else "UNKNOWN"
                    ),
                    "passed": result_map.get("passed") is True,
                    "request_id": result_map.get("request_id"),
                    "kind": result_map.get("kind"),
                    "result_path": str(result_path),
                })
            except RuntimeSessionError as error:
                recorder.append_event({
                    "phase": "operation",
                    "status": "FAILED",
                    "reason": str(error),
                })
                runtime_terminal = (
                    RunStatus.EXECUTION_FAILED,
                    f"the operation failed at runtime: {error}",
                )
    finally:
        # The stop is unconditional: cleanup evidence is recorded even after
        # a partial start or an unexpected error.
        cleanup_passed = False
        if session is not None:
            stop_evidence = session.stop()
            cleanup_passed = bool(stop_evidence.cleanup.get("passed"))
            record["runtime"] = {
                **stop_evidence.serializable(),
                "evidence_path": str(session.session_dir / "runtime-session.json"),
            }
            recorder.append_event({
                "phase": "runtime_stop",
                "status": str(stop_evidence.status),
                "cleanup": dict(stop_evidence.cleanup),
                "errors": list(stop_evidence.errors),
            })
    operation_record["cleanup_passed"] = cleanup_passed
    if runtime_terminal is not None:
        return runtime_terminal
    if result_map is None:  # pragma: no cover - defensive
        return (
            RunStatus.EXECUTION_FAILED,
            "no operation result was produced by the runtime session",
        )
    execution = result_map.get("execution")
    if result_map.get("passed") is not True:
        rejected = (
            isinstance(execution, Mapping)
            and execution.get("status") == "REJECTED"
        )
        if rejected:
            return (
                RunStatus.OPERATION_REJECTED,
                "the operation was rejected before any motion: "
                f"{execution.get('reason')}",
            )
        return (
            RunStatus.EXECUTION_FAILED,
            "the operation did not pass the execution gates: "
            f"{result_map.get('error')}",
        )
    if not cleanup_passed:
        return (
            RunStatus.EXECUTION_FAILED,
            "the operation executed but the session cleanup did not pass; "
            "the run is not verified",
        )
    return None


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
            "operation": (
                str(args.operation_path)
                if args.operation_path is not None
                else None
            ),
        },
    )
    record: dict[str, object] = {
        "schema": PREFLIGHT_SCHEMA,
        "backend": args.backend,
        "robot_id": args.robot_id,
        "preflight_only": args.preflight_only,
        "operation_path": (
            str(args.operation_path) if args.operation_path is not None else None
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
        "operation": None,
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
        # Phase 4: CAPABILITY / DISPATCH decision. Without --operation the
        # dispatch path is deliberately not implemented; nothing is ever sent
        # to any runtime. With the internal --operation bridge the formal
        # R4.2 runtime session runs after every gate has passed.
        if binding is not None:
            if terminal is None and args.operation_path is not None:
                record["dispatch"] = {
                    "implemented": True,
                    "dispatch_requested": True,
                    "blocked": False,
                    "status": "RUNTIME_DISPATCH",
                    "reason": (
                        "explicit --operation dispatch through the R4.2 "
                        "Isaac/MoveIt runtime session"
                    ),
                }
                recorder.append_event({
                    "phase": "dispatch",
                    "status": "RUNTIME_DISPATCH",
                    "operation": str(args.operation_path),
                })
                terminal = _runtime_operation_phase(
                    args, record, recorder, contract
                )
                if terminal is None:
                    terminal = (
                        RunStatus.EXECUTED,
                        "the operation executed against the live runtime and "
                        "the session stopped with passing cleanup",
                    )
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
    # Phase 5: IMMUTABILITY of the original input files.
    if inputs is not None:
        originals = {
            "plan": inputs.plan_path,
            "task_start": inputs.start_path,
            "task_goal": inputs.goal_path,
        }
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
            immutable = after == dict(inputs.sha256)
            record["checks"]["inputs_immutable_after_all_phases"] = immutable
            if not immutable:
                terminal = (
                    RunStatus.INPUT_MUTATED,
                    f"input files changed on disk during the run: "
                    f"before={dict(inputs.sha256)} after={after}",
                )
    status, reason = (
        terminal
        if terminal is not None
        else (
            RunStatus.INTERNAL_ERROR,
            "the run ended without a terminal status",
        )
    )
    record["checks"]["no_motion_dispatched"] = args.operation_path is None
    record["checks"]["operation_dispatched"] = args.operation_path is not None
    record["checks"]["inputs_immutable_after_runtime"] = record["checks"].get(
        "inputs_immutable_after_all_phases", False
    )
    operation_record = record.get("operation")
    runtime_record = record.get("runtime")
    operation_passed = bool(
        isinstance(operation_record, Mapping)
        and operation_record.get("passed") is True
    )
    cleanup_passed = bool(
        isinstance(runtime_record, Mapping)
        and isinstance(runtime_record.get("cleanup"), Mapping)
        and runtime_record["cleanup"].get("passed") is True
    )
    record["status"] = str(status)
    record["passed"] = status is RunStatus.PREFLIGHT_ONLY or (
        status is RunStatus.EXECUTED and operation_passed and cleanup_passed
    )
    record["reason"] = reason
    record["exit_code"] = EXIT_CODES[status]
    record["evidence_dir"] = str(recorder.directory)
    recorder.write_json("preflight.json", record)
    result = FinalResult(
        input_valid=status
        not in (RunStatus.INVALID_INPUT, RunStatus.INPUT_MUTATED),
        tasks_complete=binding is not None,
        structure_valid=binding is not None,
        execution_safe=record["passed"],
        retreat_clear=True,
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
    if args.operation_path is not None:
        # Link the runtime lifecycle evidence and the operation result from
        # the run result document (the full documents live in the run dir).
        result_document["operation"] = record.get("operation")
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
    if namespace.preflight_only and namespace.operation is not None:
        raise _UsageError(
            "--preflight-only cannot be combined with --operation"
        )
    scene_fields = (namespace.scene_manifest, namespace.scene_snapshot_out, namespace.host_instance_id)
    if any(value is not None for value in scene_fields) and not all(value is not None for value in scene_fields):
        raise _UsageError("--scene-manifest, --scene-snapshot-out and --host-instance-id must be provided together")
    return _Arguments(
        task_dir=namespace.task,
        plan_path=namespace.plan,
        backend=namespace.backend,
        robot_id=namespace.robot_id,
        run_dir=namespace.run_dir,
        preflight_only=namespace.preflight_only,
        operation_path=namespace.operation,
        scene_manifest=namespace.scene_manifest,
        scene_snapshot_out=namespace.scene_snapshot_out,
        host_instance_id=namespace.host_instance_id,
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
