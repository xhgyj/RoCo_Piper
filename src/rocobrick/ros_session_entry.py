"""Container child entry for the runtime session (R4.2).

Runs INSIDE the already-started rootless ROS Humble container through::

    python3 -m rocobrick.ros_session_entry --request /session/request.json \
        --result /session/result.json

It creates exactly one rclpy node with a
:class:`rocobrick.ros_moveit_runtime.RosMoveItRuntime` plus the accepted
:class:`rocobrick.ros_moveit_executor.RosMoveItExecutor`, registers the
transport, waits for readiness through the R1 predicates, executes the one
operation carried by the strict :class:`rocobrick.isaac_backend.RuntimeOperation`
request, writes the strict
:class:`rocobrick.isaac_backend.RuntimeOperationResult` document atomically
(always, also on failure), prints exactly one JSON summary line on stdout,
and shuts the executor/node/rclpy down cleanly.

READINESS is a no-motion operation: nothing is ever published or sent.
JOINT_TARGET / LINK6_POSE targets are rebuilt through the formal dataclasses
and re-checked with strict admission immediately before the executor send;
no target derivation, no plan parsing, and no TCP semantics exist here. The
repository ``src`` tree is consumed exactly as mounted read-only.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path

from rocobrick.isaac_backend import (
    OPERATION_RESULT_SCHEMA,
    OperationContractError,
    OperationKind,
    OperationRejected,
    RuntimeOperation,
    RuntimeOperationResult,
    target_for_operation,
)
from rocobrick.moveit_contract import JointTarget, Link6PoseTarget
from rocobrick.moveit_execution import ExecutionResult, is_success

SUMMARY_SCHEMA = "rocobrick-runtime-operation-summary/v1"
DEFAULT_NODE_NAME = "rocobrick_session_entry"
FK_TIMEOUT_S = 10.0
THREAD_JOIN_TIMEOUT_S = 2.0


def build_parser() -> argparse.ArgumentParser:
    """Build the child entry argument parser."""
    parser = argparse.ArgumentParser(
        prog="python3 -m rocobrick.ros_session_entry",
        description=(
            "Execute one bounded runtime operation inside the running "
            "MoveIt container; writes one strict result document and prints "
            "exactly one JSON summary line."
        ),
    )
    parser.add_argument(
        "--request",
        required=True,
        type=Path,
        metavar="PATH",
        help="RuntimeOperation request JSON (mounted session directory)",
    )
    parser.add_argument(
        "--result",
        required=True,
        type=Path,
        metavar="PATH",
        help="RuntimeOperationResult destination JSON (always written)",
    )
    parser.add_argument(
        "--node-name",
        default=DEFAULT_NODE_NAME,
        metavar="NAME",
        help=f"rclpy node name (default: {DEFAULT_NODE_NAME})",
    )
    return parser


def load_request(path: Path) -> RuntimeOperation:
    """Load and strictly parse the operation request document."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return RuntimeOperation.parse(document)


def write_result(path: Path, result: RuntimeOperationResult) -> None:
    """Write the result document atomically (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.serialize(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _jsonable(value: object) -> object:
    """Best-effort JSON-safety for details/limits mappings."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def execution_record(outcome: ExecutionResult) -> dict[str, object]:
    """JSON-safe evidence record of one ExecutionResult (no verdict here)."""
    return {
        "status": str(outcome.status.value),
        "moveit_error_code": outcome.moveit_error_code,
        "planned_waypoints": outcome.planned_waypoints,
        "command_messages": outcome.command_messages,
        "final_errors": {
            name: float(value) for name, value in outcome.final_errors.items()
        },
        "is_success": is_success(outcome),
        "details": _jsonable(dict(outcome.details)),
    }


def _rejected_record(reason: str) -> dict[str, object]:
    """Execution record for a target rejected before any send."""
    return {
        "status": "REJECTED",
        "sent": False,
        "reason": reason,
        "command_messages": 0,
    }


def perform_operation(
    operation: RuntimeOperation,
    *,
    runtime: object,
    executor: object,
    fk_timeout_s: float = FK_TIMEOUT_S,
) -> RuntimeOperationResult:
    """Execute one operation against an (already spinning) runtime.

    ``runtime`` is a :class:`rocobrick.ros_moveit_runtime.RosMoveItRuntime`
    and ``executor`` a
    :class:`rocobrick.ros_moveit_executor.RosMoveItExecutor`; both are passed
    in (not constructed) so offline tests can drive this function with
    fakes. READINESS never sends motion; motion operations rebuild the
    formal target and re-run strict admission before delegating exactly once
    to the executor.
    """
    evidence: dict[str, object] = {}
    try:
        runtime.register_subscriptions()
        runtime.register_clients()
        waited_ok = bool(runtime.wait_until_ready(timeout_s=operation.timeout_s))
        snapshot = runtime.readiness_snapshot()
        snapshot_map: dict[str, object] = dict(asdict(snapshot))
        readiness_passed = bool(waited_ok and snapshot.passed)
        evidence.update(
            {
                "waited_ok": waited_ok,
                "passed": readiness_passed,
                "snapshot": snapshot_map,
            }
        )
        if operation.kind is OperationKind.READINESS:
            evidence["motion_sent"] = False
            if not readiness_passed:
                return RuntimeOperationResult(
                    request_id=operation.request_id,
                    kind=operation.kind,
                    passed=False,
                    execution={"motion_sent": False, "goals_sent": 0},
                    evidence=evidence,
                    error=(
                        "readiness phase failed: the runtime snapshot did not "
                        "pass every component"
                    ),
                )
            return RuntimeOperationResult(
                request_id=operation.request_id,
                kind=operation.kind,
                passed=True,
                execution={
                    "motion_sent": False,
                    "goals_sent": 0,
                    "command_messages": 0,
                },
                evidence=evidence,
                error=None,
            )
        if not readiness_passed:
            evidence["motion_sent"] = False
            return RuntimeOperationResult(
                request_id=operation.request_id,
                kind=operation.kind,
                passed=False,
                execution=_rejected_record(
                    "runtime readiness failed; no motion is allowed until the "
                    "snapshot passes every component"
                ),
                evidence=evidence,
                error="readiness phase failed before the goal send",
            )
        target = target_for_operation(operation)
        if isinstance(target, JointTarget):
            outcome = executor.execute_joint(
                target, timeout_s=operation.timeout_s
            )
        elif isinstance(target, Link6PoseTarget):
            outcome = executor.execute_link6_pose(
                target, timeout_s=operation.timeout_s
            )
        else:  # pragma: no cover - target_for_operation narrows the type
            raise OperationContractError(
                f"unsupported target type {type(target).__name__}"
            )
        record = execution_record(outcome)
        evidence["motion_sent"] = bool(outcome.command_messages >= 1)
        return RuntimeOperationResult(
            request_id=operation.request_id,
            kind=operation.kind,
            passed=bool(record["is_success"]),
            execution=record,
            evidence=evidence,
            error=(
                None
                if record["is_success"]
                else f"the {str(operation.kind.value)} goal did not pass the "
                "execution gates"
            ),
        )
    except OperationRejected as error:
        return RuntimeOperationResult(
            request_id=operation.request_id,
            kind=operation.kind,
            passed=False,
            execution=_rejected_record(str(error)),
            evidence={**evidence, "motion_sent": False},
            error=str(error),
        )
    except Exception as error:  # noqa: BLE001 - the child must always answer
        return RuntimeOperationResult(
            request_id=operation.request_id,
            kind=operation.kind,
            passed=False,
            execution={"motion_sent": None},
            evidence=dict(evidence),
            error=f"{type(error).__name__}: {error}",
        )


def _make_fk_hook(
    holder: dict, fk_timeout_s: float
) -> Callable[[Mapping[str, float]], object]:
    """Runtime ``fk_verifier`` hook delegating to the live /compute_fk."""

    def fk_hook(state: Mapping[str, float]):
        runtime = holder.get("runtime")
        if runtime is None:
            return None
        return runtime.compute_link6_fk(state, timeout_s=fk_timeout_s)

    return fk_hook


def _spin_up(
    node_name: str, fk_timeout_s: float
) -> tuple[object, object, object, threading.Thread, object]:
    """Create the node, background executor, runtime and executor adapter."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    from rocobrick.ros_moveit_executor import RosMoveItExecutor
    from rocobrick.ros_moveit_runtime import RosMoveItRuntime

    rclpy.init()
    node = rclpy.create_node(node_name)
    background = SingleThreadedExecutor()
    background.add_node(node)
    thread = threading.Thread(target=background.spin, daemon=True)
    thread.start()
    holder: dict = {}
    runtime = RosMoveItRuntime(node, fk_verifier=_make_fk_hook(holder, fk_timeout_s))
    holder["runtime"] = runtime
    executor = RosMoveItExecutor(runtime)
    return runtime, executor, background, thread, node


def _spin_down(
    runtime: object, background: object, thread: threading.Thread, node: object
) -> None:
    """Orderly teardown: quiesce the spinning thread before destroying nodes."""
    import rclpy

    runtime.shutdown()
    background.shutdown()
    thread.join(timeout=THREAD_JOIN_TIMEOUT_S)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _fallback_result(error: str) -> RuntimeOperationResult:
    """Schema-conform failure answer for an unparseable request."""
    return RuntimeOperationResult(
        request_id="",
        kind=OperationKind.READINESS,
        passed=False,
        execution={"sent": False},
        evidence={},
        error=error,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the child entry; exactly one stdout JSON summary line is printed."""
    args = build_parser().parse_args(argv)
    try:
        operation = load_request(args.request)
    except (OSError, OperationContractError, ValueError) as error:
        result = _fallback_result(f"request could not be loaded: {error}")
        write_result(args.result, result)
        print(json.dumps(_summary(result, args.result), sort_keys=True), flush=True)
        return 2
    runtime_object = executor_object = background = node = None
    thread: threading.Thread | None = None
    try:
        (
            runtime_object,
            executor_object,
            background,
            thread,
            node,
        ) = _spin_up(args.node_name, min(FK_TIMEOUT_S, operation.timeout_s))
        result = perform_operation(
            operation, runtime=runtime_object, executor=executor_object
        )
    except Exception as error:  # noqa: BLE001 - the child must always answer
        result = RuntimeOperationResult(
            request_id=operation.request_id,
            kind=operation.kind,
            passed=False,
            execution={"sent": False},
            evidence={},
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        if runtime_object is not None:
            try:
                _spin_down(runtime_object, background, thread, node)
            except Exception:  # pragma: no cover - defensive teardown
                pass
    write_result(args.result, result)
    print(json.dumps(_summary(result, args.result), sort_keys=True), flush=True)
    return 0 if result.passed else 1


def _summary(result: RuntimeOperationResult, result_path: Path) -> dict[str, object]:
    """The single stdout summary line payload."""
    return {
        "schema": SUMMARY_SCHEMA,
        "result_schema": OPERATION_RESULT_SCHEMA,
        "request_id": result.request_id,
        "kind": str(result.kind.value),
        "passed": result.passed,
        "result_path": str(result_path),
        "error": result.error,
    }


if __name__ == "__main__":  # pragma: no cover - container entry point
    raise SystemExit(main())
