"""Contract tests for the A2 scheduler, result types, and run serialization."""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from rocobrick.multi_robot_scheduler import (
    CancelEvidence,
    ExecutionContractError,
    MultiRobotScheduler,
    OperationExecution,
    OperationStatus,
    RobotOperationExecutor,
    is_json_safe,
    schedule,
    to_json_value,
)
from rocobrick.robot_operation_plan import (
    RobotOperation,
    RobotOperationPlan,
    ScheduledOperation,
    parse_robot_operation_plan,
)
from rocobrick.unified_dual_arm_runtime import (
    SuccessorPrePhaseEvidence,
    SuccessorPrePhaseRequest,
)
from rocobrick.robot_operation_runner import (
    RUN_SCHEMA,
    OperationPlanRunner,
    serialize_run_result,
)


def _operation(
    operation_id: str,
    robot_id: str,
    *,
    evidence_requirements: tuple[str, ...] = (),
    resources: tuple[str, ...] | None = None,
) -> dict:
    return {
        "operation_id": operation_id,
        "source_task_id": f"task-{operation_id}",
        "skill": "WAIT",
        "executor_robot_id": robot_id,
        "participants": [robot_id],
        "role_bindings": {"executor": robot_id},
        "target": None,
        "parameters": {},
        "preconditions": [],
        "effects": [],
        "resources": (
            list(resources)
            if resources is not None
            else [f"robot:{robot_id}"]
        ),
        "evidence_requirements": list(evidence_requirements),
    }


def _plan(
    operations: list[dict],
    layers: list[list[str]],
    *,
    planned_schedule: list[list[dict]] | None = None,
    dependencies: list[dict] | None = None,
    evidence_requirements: list[dict] | None = None,
    robots: tuple[str, ...] = ("piper_0", "piper_1"),
) -> RobotOperationPlan:
    if planned_schedule is None:
        planned_schedule = [
            [
                {
                    "operation_id": operation_id,
                    "robot_id": next(
                        item["executor_robot_id"]
                        for item in operations
                        if item["operation_id"] == operation_id
                    ),
                }
                for operation_id in layer
            ]
            for layer in layers
        ]
    return parse_robot_operation_plan(
        {
            "schema": "robot-operation-plan/v1",
            "plan_id": "plan-1",
            "source_plan_id": "assembly-1",
            "robots": list(robots),
            "operations": operations,
            "dependencies": dependencies or [],
            "execution_layers": layers,
            "planned_schedule": planned_schedule,
            "evidence_requirements": evidence_requirements or [],
        }
    )


def _execution(
    operation,
    *,
    status: OperationStatus = OperationStatus.CONFIRMED,
    passed: bool = True,
    terminal: bool = True,
    motion_sent: bool = False,
    evidence: dict | None = None,
    reason: str | None = None,
    operation_id: str | None = None,
    robot_id: str | None = None,
) -> OperationExecution:
    return OperationExecution(
        operation_id=operation_id or operation.operation_id,
        robot_id=robot_id or operation.executor_robot_id,
        status=status,
        passed=passed,
        terminal=terminal,
        motion_sent=motion_sent,
        evidence=evidence or {},
        reason=reason,
    )


class StrictExecutor:
    """Scripted formal executor implementing the RobotOperationExecutor API."""

    def __init__(
        self,
        outcomes: dict[str, object] | None = None,
        *,
        delays: dict[str, float] | None = None,
        cancel_result: object | None = None,
        cancel_error: BaseException | None = None,
    ) -> None:
        self.outcomes = outcomes or {}
        self.delays = delays or {}
        self.cancel_result = cancel_result
        self.cancel_error = cancel_error
        self.dispatched: list[str] = []
        self.cancelled: list[str] = []
        self.events: list[tuple[str, str]] = []

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        self.dispatched.append(operation.operation_id)
        self.events.append(("start", operation.operation_id))
        delay = self.delays.get(operation.operation_id)
        if delay:
            await asyncio.sleep(delay)
        outcome = self.outcomes.get(operation.operation_id)
        if outcome is None:
            outcome = _execution(operation)
        elif callable(outcome):
            outcome = outcome(operation)
        if isinstance(outcome, BaseException):
            raise outcome
        self.events.append(("end", operation.operation_id))
        return outcome

    async def execute_pre_phase(
        self, request: SuccessorPrePhaseRequest
    ) -> SuccessorPrePhaseEvidence:
        """Return terminal-safe successor pre-phase evidence."""
        return SuccessorPrePhaseEvidence(
            predecessor_operation_id=request.predecessor_operation_id,
            successor_operation_ids=request.successor_operation_ids,
            participants=request.participants,
            scene_digest=request.scene_digest,
            planned=True,
            executed=True,
            terminal=True,
            safe=True,
            resource_transfer_confirmed=True,
            evidence={"pre_phase": True},
            terminal_cleanup=request.terminal_cleanup,
        )

    async def cancel(self, operation_id: str) -> object:
        self.cancelled.append(operation_id)
        self.events.append(("cancel", operation_id))
        if self.cancel_error is not None:
            raise self.cancel_error
        if callable(self.cancel_result) and not isinstance(
            self.cancel_result, CancelEvidence
        ):
            return self.cancel_result(operation_id)
        return self.cancel_result


class LegacyExecutor:
    """Pre-contract stub returning a plain boolean."""

    def __init__(self, *, fail: str | None = None, delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay
        self.dispatched: list[str] = []
        self.events: list[tuple[str, str]] = []

    async def execute(self, operation: RobotOperation) -> object:
        self.dispatched.append(operation.operation_id)
        self.events.append(("start", operation.operation_id))
        await asyncio.sleep(self.delay)
        self.events.append(("end", operation.operation_id))
        return operation.operation_id != self.fail


def _raw_operation(
    operation_id: str, robot_id: str, resources: tuple[str, ...]
) -> RobotOperation:
    return RobotOperation(
        operation_id=operation_id,
        source_task_id="task-raw",
        skill="WAIT",
        executor_robot_id=robot_id,
        participants=(robot_id,),
        role_bindings={"executor": robot_id},
        target=None,
        parameters={},
        preconditions=(),
        effects=(),
        resources=resources,
        evidence_requirements=(),
    )


def _raw_plan(
    operations: list[RobotOperation],
    layers: tuple[tuple[str, ...], ...],
) -> RobotOperationPlan:
    robots = tuple(dict.fromkeys(op.executor_robot_id for op in operations))
    mapping = {op.operation_id: op for op in operations}
    planned = tuple(
        tuple(
            ScheduledOperation(
                slot=index,
                operation_id=operation_id,
                robot_id=mapping[operation_id].executor_robot_id,
            )
            for operation_id in layer
        )
        for index, layer in enumerate(layers)
    )
    return RobotOperationPlan(
        schema="robot-operation-plan/v1",
        plan_id="raw",
        source_plan_id="assembly-raw",
        robots=robots,
        operations=mapping,
        dependencies=(),
        execution_layers=layers,
        planned_schedule=planned,
        evidence_requirements=(),
    )


def test_formal_result_types_are_frozen_and_strict() -> None:
    execution = OperationExecution(
        operation_id="a",
        robot_id="piper_0",
        status=OperationStatus.CONFIRMED,
        passed=True,
        terminal=True,
        motion_sent=False,
        evidence={"confirmed": True},
    )
    assert dataclasses.is_dataclass(execution)
    with pytest.raises(dataclasses.FrozenInstanceError):
        execution.passed = False  # type: ignore[misc]
    for bad in (
        {"passed": 1},
        {"terminal": 0},
        {"motion_sent": "yes"},
        {"status": "CONFIRMED"},
        {"reason": 7},
        {"evidence": {"result": object()}},
        {"evidence": {"result": float("nan")}},
    ):
        fields = {
            "operation_id": "a",
            "robot_id": "piper_0",
            "status": OperationStatus.CONFIRMED,
            "passed": True,
            "terminal": True,
            "motion_sent": False,
            "evidence": {"confirmed": True},
        }
        fields.update(bad)
        with pytest.raises(ExecutionContractError):
            OperationExecution(**fields)

    cancel = CancelEvidence(
        operation_id="a",
        acknowledged=True,
        terminal=True,
        safe=True,
        evidence={"stopped": True},
    )
    assert dataclasses.is_dataclass(cancel)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cancel.safe = False  # type: ignore[misc]
    with pytest.raises(ExecutionContractError):
        CancelEvidence(
            operation_id="a",
            acknowledged=1,
            terminal=True,
            safe=True,
            evidence={},
        )


def test_protocol_and_field_order_are_frozen() -> None:
    assert asyncio.iscoroutinefunction(RobotOperationExecutor.execute)
    assert asyncio.iscoroutinefunction(RobotOperationExecutor.cancel)
    assert list(OperationExecution.__dataclass_fields__) == [
        "operation_id",
        "robot_id",
        "status",
        "passed",
        "terminal",
        "motion_sent",
        "evidence",
        "reason",
    ]
    assert list(CancelEvidence.__dataclass_fields__) == [
        "operation_id",
        "acknowledged",
        "terminal",
        "safe",
        "evidence",
    ]


def test_json_helpers_reject_unsafe_values() -> None:
    assert is_json_safe({"a": [1, True, None, "x"]})
    assert not is_json_safe({"a": object()})
    assert not is_json_safe({"a": float("inf")})
    assert not is_json_safe({1: "x"})
    assert to_json_value({"a": (1, 2)}) == {"a": [1, 2]}
    with pytest.raises(ValueError):
        to_json_value({1: "x"})


def test_same_layer_is_concurrent_and_layer_barriers() -> None:
    plan = _plan(
        [
            _operation("a", "piper_0"),
            _operation("b", "piper_1"),
            _operation("c", "piper_0"),
        ],
        [["a", "b"], ["c"]],
    )
    executor = StrictExecutor(delays={"a": 0.02, "b": 0.02})
    result = asyncio.run(schedule(plan, executor))
    assert result.passed
    assert executor.events[:2] == [("start", "a"), ("start", "b")]
    start_c = executor.events.index(("start", "c"))
    assert start_c > executor.events.index(("end", "a"))
    assert start_c > executor.events.index(("end", "b"))


def test_planned_schedule_controls_evidence_order_only() -> None:
    plan = _plan(
        [_operation("a", "piper_0"), _operation("b", "piper_1")],
        [["a"], ["b"]],
        planned_schedule=[
            [{"operation_id": "b", "robot_id": "piper_1"}],
            [{"operation_id": "a", "robot_id": "piper_0"}],
        ],
    )
    executor = StrictExecutor()
    result = asyncio.run(schedule(plan, executor))
    assert result.passed
    assert executor.dispatched == ["a", "b"]
    operation_entries = [
        entry for entry in result.evidence if entry["kind"] == "operation"
    ]
    assert [entry["operation_id"] for entry in operation_entries] == ["b", "a"]


def test_all_predecessors_are_required() -> None:
    plan = _plan(
        [
            _operation("a", "piper_0"),
            _operation("b", "piper_1"),
            _operation("c", "piper_0"),
        ],
        [["a"], ["b"], ["c"]],
        dependencies=[
            {
                "predecessor_operation_id": "a",
                "successor_operation_id": "c",
                "reason": "needs a",
            },
            {
                "predecessor_operation_id": "b",
                "successor_operation_id": "c",
                "reason": "needs b",
            },
        ],
    )
    ok = StrictExecutor()
    confirmed = asyncio.run(schedule(plan, ok))
    assert confirmed.passed
    assert ok.dispatched == ["a", "b", "c"]

    failing = StrictExecutor(
        outcomes={
            "b": lambda op: _execution(
                op,
                status=OperationStatus.FAILED,
                passed=False,
                reason="b failed",
            )
        }
    )
    blocked = asyncio.run(schedule(plan, failing))
    assert not blocked.passed
    assert blocked.operations["c"].status is OperationStatus.BLOCKED
    assert blocked.operations["c"].terminal is False
    assert "c" not in failing.dispatched
    assert blocked.layers[-1].status is OperationStatus.BLOCKED


def test_strict_evidence_labels_from_operation_and_plan() -> None:
    plan = _plan(
        [
            _operation(
                "a",
                "piper_0",
                evidence_requirements=("operation_label", "plan_label"),
            )
        ],
        [["a"]],
        evidence_requirements=[
            {"operation_id": "a", "evidence": "plan_label"}
        ],
    )
    missing = StrictExecutor(
        outcomes={
            "a": lambda op: _execution(
                op, evidence={"operation_label": True}
            )
        }
    )
    result = asyncio.run(schedule(plan, missing))
    assert not result.passed
    assert result.operations["a"].status is OperationStatus.FAILED
    assert "plan_label" in (result.operations["a"].reason or "")
    assert result.resources[0].acquired is True
    assert result.resources[0].released is False

    falsy = StrictExecutor(
        outcomes={
            "a": lambda op: _execution(
                op,
                evidence={"operation_label": None, "plan_label": False},
            )
        }
    )
    assert not asyncio.run(schedule(plan, falsy)).passed

    satisfied = StrictExecutor(
        outcomes={
            "a": lambda op: _execution(
                op,
                evidence={
                    "operation_label": True,
                    "plan_label": {"source": "runtime"},
                },
            )
        }
    )
    confirmed = asyncio.run(schedule(plan, satisfied))
    assert confirmed.passed
    assert confirmed.resources[0].released is True


def test_identity_and_consistency_fail_closed() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])
    cases = {
        "wrong_operation_id": lambda op: _execution(op, operation_id="other"),
        "wrong_robot_id": lambda op: _execution(op, robot_id="piper_1"),
        "passed_false": lambda op: _execution(op, passed=False),
        "not_terminal": lambda op: _execution(op, terminal=False),
        "non_terminal_status": lambda op: _execution(
            op,
            status=OperationStatus.RUNNING,
            passed=False,
            terminal=False,
            reason="still running",
        ),
    }
    for name, outcome in cases.items():
        result = asyncio.run(
            schedule(plan, StrictExecutor(outcomes={"a": outcome}))
        )
        assert not result.passed, name
        assert (
            result.operations["a"].status is not OperationStatus.CONFIRMED
        ), name

    def invalid(op: RobotOperation) -> OperationExecution:
        return OperationExecution(
            operation_id=op.operation_id,
            robot_id=op.executor_robot_id,
            status=OperationStatus.CONFIRMED,
            passed=1,
            terminal=True,
            motion_sent=False,
            evidence={},
        )

    raised = asyncio.run(
        schedule(plan, StrictExecutor(outcomes={"a": invalid}))
    )
    assert not raised.passed
    assert "ExecutionContractError" in (raised.operations["a"].reason or "")


def test_handoff_releases_resources_only_after_terminal_safe_transfer() -> None:
    """A deferred skill keeps its lease until the next transport handoff."""
    first = _operation("a", "piper_0", resources=("robot:piper_0",))
    second = _operation("b", "piper_0", resources=("robot:piper_0",))
    plan = _plan([first, second], [["a"], ["b"]])
    executor = StrictExecutor(
        outcomes={
            "a": lambda op: _execution(
                op,
                evidence={"handoff_required": True},
            )
        }
    )
    result = asyncio.run(schedule(plan, executor))
    assert result.passed
    assert result.operations["a"].resources_released is True
    assert result.operations["a"].evidence["handoff_confirmed"]["pre_phase"] is True
    assert result.resources[0].released is True


class FailingPrePhaseExecutor(StrictExecutor):
    """Executor whose successor pre-phase never proves terminal safety."""

    async def execute_pre_phase(
        self, request: SuccessorPrePhaseRequest
    ) -> SuccessorPrePhaseEvidence:
        return SuccessorPrePhaseEvidence(
            predecessor_operation_id=request.predecessor_operation_id,
            successor_operation_ids=request.successor_operation_ids,
            participants=request.participants,
            scene_digest=request.scene_digest,
            planned=False,
            executed=False,
            terminal=False,
            safe=False,
            resource_transfer_confirmed=False,
            evidence={"pre_phase": False},
            terminal_cleanup=request.terminal_cleanup,
            reason="pre-phase blocked",
        )


def test_failed_handoff_keeps_previous_resources_held() -> None:
    """A failed next-transport handoff cannot release the prior lease."""
    first = _operation("a", "piper_0", resources=("robot:piper_0",))
    second = _operation("b", "piper_0", resources=("robot:piper_0",))
    plan = _plan([first, second], [["a"], ["b"]])
    result = asyncio.run(
        schedule(
            plan,
            FailingPrePhaseExecutor(
                outcomes={"a": lambda op: _execution(op, evidence={"handoff_required": True})}
            ),
        )
    )
    assert not result.passed
    assert result.resources[0].released is False
    assert result.operations["a"].resources_released is False


    plan = _plan(
        [_operation("a", "piper_0"), _operation("b", "piper_0")],
        [["a"], ["b"]],
    )
    executor = StrictExecutor()
    result = asyncio.run(schedule(plan, executor))
    assert result.passed
    assert [
        (record.operation_id, record.acquired, record.released)
        for record in result.resources
    ] == [("a", True, True), ("b", True, True)]


def test_failed_operation_holds_resources_and_blocks_successor() -> None:
    plan = _plan(
        [_operation("a", "piper_0"), _operation("b", "piper_0")],
        [["a"], ["b"]],
    )
    executor = StrictExecutor(
        outcomes={
            "a": lambda op: _execution(
                op,
                status=OperationStatus.FAILED,
                passed=False,
                terminal=True,
                motion_sent=True,
                reason="motion fault",
            )
        }
    )
    result = asyncio.run(schedule(plan, executor))
    assert not result.passed
    records = {
        (record.operation_id, record.resource): record
        for record in result.resources
    }
    assert records[("a", "robot:piper_0")].released is False
    assert result.operations["b"].status is OperationStatus.BLOCKED
    assert executor.dispatched == ["a"]
    assert result.unsafe_terminals == ("a",)


def test_dispatch_time_conflict_fails_closed_without_dispatch() -> None:
    plan = _raw_plan(
        [
            _raw_operation("a", "piper_0", ("robot:shared",)),
            _raw_operation("b", "piper_1", ("robot:shared",)),
        ],
        (("a", "b"),),
    )
    executor = StrictExecutor()
    result = asyncio.run(schedule(plan, executor))
    assert not result.passed
    assert executor.dispatched == ["a"]
    assert result.operations["b"].status is OperationStatus.FAILED
    assert "resource unavailable" in (result.operations["b"].reason or "")
    refused = {
        (record.operation_id, record.resource): record
        for record in result.resources
    }
    assert refused[("b", "robot:shared")].acquired is False


def test_timeout_requires_terminal_safe_cancel_to_release() -> None:
    plan = _plan(
        [_operation("a", "piper_0"), _operation("b", "piper_1")],
        [["a", "b"]],
    )
    executor = StrictExecutor(
        delays={"a": 10.0, "b": 10.0},
        cancel_result=lambda operation_id: CancelEvidence(
            operation_id=operation_id,
            acknowledged=True,
            terminal=True,
            safe=True,
            evidence={"stopped": True},
        ),
    )
    scheduler = MultiRobotScheduler(operation_timeout_s=0.05, cancel_timeout_s=0.5)
    result = asyncio.run(scheduler.schedule(plan, executor))
    assert not result.passed
    assert set(executor.cancelled) == {"a", "b"}
    assert [item.operation_id for item in result.cancellations] == ["a", "b"]
    assert all(
        item.status is OperationStatus.CANCELLED
        for item in result.operations.values()
    )
    assert all(item.terminal for item in result.operations.values())
    assert all(item.motion_sent is None for item in result.operations.values())
    assert all(item.deadline_exceeded for item in result.operations.values())
    assert all(
        item.resources_released for item in result.operations.values()
    )
    assert all(record.released for record in result.resources)
    assert result.unsafe_terminals == ()
    assert "deadline" in (result.operations["a"].reason or "")
    deadline_events = [
        entry for entry in result.evidence if entry["kind"] == "deadline_exceeded"
    ]
    assert [entry["operation_ids"] for entry in deadline_events] == [["a", "b"]]


def test_late_terminal_confirmation_cannot_restore_deadline_success() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])

    class LateConfirm:
        async def execute(self, operation: RobotOperation) -> OperationExecution:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                return _execution(operation, reason="late terminal confirmation")
            raise AssertionError("unreachable")

        async def cancel(self, operation_id: str) -> CancelEvidence:
            return CancelEvidence(
                operation_id=operation_id,
                acknowledged=True,
                terminal=True,
                safe=True,
                evidence={"requested": True},
            )

    scheduler = MultiRobotScheduler(
        operation_timeout_s=0.05, cancel_timeout_s=0.2
    )
    result = asyncio.run(scheduler.schedule(plan, LateConfirm()))
    assert result.passed is False
    item = result.operations["a"]
    assert item.status is OperationStatus.CANCELLED
    assert item.passed is False
    assert item.terminal is True
    assert item.motion_sent is False
    assert item.deadline_exceeded is True
    assert item.late_terminal_confirmation is True
    # The validated terminal result releases the lease, but it cannot make the
    # deadline an operation or run success.
    assert item.resources_released is True
    assert result.resources[0].released is True
    assert result.unsafe_terminals == ()
    assert result.layers[0].status is OperationStatus.FAILED
    deadline_events = [
        entry for entry in result.evidence if entry["kind"] == "deadline_exceeded"
    ]
    assert [entry["operation_ids"] for entry in deadline_events] == [["a"]]
    document = serialize_run_result(result)
    assert document["passed"] is False
    assert document["operations"]["a"]["deadline_exceeded"] is True
    assert document["operations"]["a"]["late_terminal_confirmation"] is True
    assert document["operations"]["a"]["status"] == "CANCELLED"
    assert document["operations"]["a"]["passed"] is False
    assert document["resources"][0]["released"] is True


def test_cooperative_cancel_attaches_safe_evidence_and_releases_resource() -> None:
    plan = _plan(
        [_operation("bad", "piper_0"), _operation("waiting", "piper_1")],
        [["bad", "waiting"]],
    )
    stopped = asyncio.Event()

    class CooperativeExecutor:
        async def execute(self, operation: RobotOperation) -> OperationExecution:
            if operation.operation_id == "bad":
                await asyncio.sleep(0)
                return _execution(
                    operation,
                    status=OperationStatus.FAILED,
                    passed=False,
                    reason="capability unavailable",
                )
            await stopped.wait()
            return _execution(
                operation,
                status=OperationStatus.CANCELLED,
                passed=False,
                reason="cancelled cooperatively",
            )

        async def cancel(self, operation_id: str) -> CancelEvidence:
            stopped.set()
            return CancelEvidence(
                operation_id=operation_id,
                acknowledged=True,
                terminal=True,
                safe=True,
                evidence={"motion_sent": False},
            )

    result = asyncio.run(
        MultiRobotScheduler(allow_legacy_results=False).schedule(
            plan, CooperativeExecutor()
        )
    )
    waiting = result.operations["waiting"]
    assert result.passed is False
    assert waiting.status is OperationStatus.CANCELLED
    assert waiting.cancel is not None and waiting.cancel.safe is True
    assert waiting.resources_released is True
    assert waiting.operation_id not in result.unsafe_terminals
    assert next(
        record for record in result.resources if record.operation_id == "waiting"
    ).released is True
    assert serialize_run_result(result)["operations"]["waiting"]["cancel"][
        "acknowledged"
    ] is True


def test_timeout_without_terminal_safe_cancel_holds_resources() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])
    executor = StrictExecutor(
        delays={"a": 10.0},
        cancel_result=lambda operation_id: CancelEvidence(
            operation_id=operation_id,
            acknowledged=True,
            terminal=False,
            safe=False,
            evidence={"accepted": True},
        ),
    )
    scheduler = MultiRobotScheduler(operation_timeout_s=0.05, cancel_timeout_s=0.5)
    result = asyncio.run(scheduler.schedule(plan, executor))
    assert not result.passed
    item = result.operations["a"]
    assert item.status is OperationStatus.CANCELLED
    assert item.terminal is False
    assert item.motion_sent is None
    assert item.deadline_exceeded is True
    assert result.unsafe_terminals == ("a",)
    assert result.resources[0].released is False


def test_cancel_failure_is_recorded_not_swallowed() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])
    executor = StrictExecutor(
        delays={"a": 10.0},
        cancel_error=RuntimeError("cancel transport down"),
    )
    scheduler = MultiRobotScheduler(operation_timeout_s=0.05, cancel_timeout_s=0.5)
    result = asyncio.run(scheduler.schedule(plan, executor))
    assert not result.passed
    assert result.unsafe_terminals == ("a",)
    assert len(result.cancellations) == 1
    assert "cancel transport down" in str(result.cancellations[0].evidence)
    assert result.cancellations[0].safe is False
    assert result.resources[0].released is False


def test_legacy_cancel_is_not_motion_safety() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])
    executor = StrictExecutor(delays={"a": 10.0})
    scheduler = MultiRobotScheduler(operation_timeout_s=0.05, cancel_timeout_s=0.5)
    result = asyncio.run(scheduler.schedule(plan, executor))
    assert not result.passed
    item = result.operations["a"]
    assert item.motion_sent is None
    assert item.terminal is False
    assert result.unsafe_terminals == ("a",)
    assert result.cancellations[0].acknowledged is False
    assert result.cancellations[0].safe is False


def test_cancel_evidence_identity_mismatch_fails_closed() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])
    executor = StrictExecutor(
        delays={"a": 10.0},
        cancel_result=CancelEvidence(
            operation_id="other",
            acknowledged=True,
            terminal=True,
            safe=True,
            evidence={"stopped": True},
        ),
    )
    scheduler = MultiRobotScheduler(operation_timeout_s=0.05, cancel_timeout_s=0.5)
    result = asyncio.run(scheduler.schedule(plan, executor))
    assert not result.passed
    assert result.cancellations[0].acknowledged is False
    assert "does not match" in str(result.cancellations[0].evidence)
    assert result.unsafe_terminals == ("a",)
    assert result.resources[0].released is False


def test_blocked_layers_after_failure() -> None:
    plan = _plan(
        [
            _operation("a", "piper_0"),
            _operation("b", "piper_1"),
            _operation("c", "piper_0"),
        ],
        [["a"], ["b"], ["c"]],
    )
    executor = StrictExecutor(
        outcomes={
            "a": lambda op: _execution(
                op,
                status=OperationStatus.FAILED,
                passed=False,
                reason="nope",
            )
        }
    )
    result = asyncio.run(schedule(plan, executor))
    assert not result.passed
    assert [layer.status for layer in result.layers] == [
        OperationStatus.FAILED,
        OperationStatus.BLOCKED,
        OperationStatus.BLOCKED,
    ]
    assert result.operations["b"].status is OperationStatus.BLOCKED
    assert result.operations["c"].status is OperationStatus.BLOCKED
    assert executor.dispatched == ["a"]


def test_scheduler_state_is_isolated_between_runs() -> None:
    scheduler = MultiRobotScheduler(operation_timeout_s=0.05, cancel_timeout_s=0.5)
    hanging = StrictExecutor(delays={"a": 10.0})
    first = asyncio.run(
        scheduler.schedule(_plan([_operation("a", "piper_0")], [["a"]]), hanging)
    )
    assert not first.passed
    assert first.unsafe_terminals == ("a",)

    second = asyncio.run(
        scheduler.schedule(
            _plan([_operation("b", "piper_0")], [["b"]]), StrictExecutor()
        )
    )
    assert second.passed
    assert set(second.operations) == {"b"}
    assert second.unsafe_terminals == ()
    assert second.cancellations == ()
    assert [record.released for record in second.resources] == [True]


def test_executor_exception_is_recorded_as_failure() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])
    executor = StrictExecutor(outcomes={"a": RuntimeError("driver exploded")})
    result = asyncio.run(schedule(plan, executor))
    assert not result.passed
    assert result.operations["a"].status is OperationStatus.FAILED
    assert "driver exploded" in (result.operations["a"].reason or "")
    assert result.unsafe_terminals == ("a",)
    assert result.resources[0].released is False


def test_legacy_results_are_marked_and_strict_mode_rejects() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])
    result = asyncio.run(schedule(plan, LegacyExecutor()))
    assert result.passed
    assert result.operations["a"].legacy is True
    assert result.operations["a"].evidence["legacy_unvalidated"] is True
    assert result.resources[0].released is True

    strict = asyncio.run(
        schedule(plan, LegacyExecutor(), allow_legacy_results=False)
    )
    assert not strict.passed
    assert "rejected" in (strict.operations["a"].reason or "")


def test_legacy_result_skips_strict_evidence_labels_by_design() -> None:
    plan = _plan(
        [_operation("a", "piper_0", evidence_requirements=("confirmed",))],
        [["a"]],
        evidence_requirements=[{"operation_id": "a", "evidence": "confirmed"}],
    )

    class DictExecutor:
        async def execute(self, operation: RobotOperation) -> dict:
            return {
                "passed": True,
                "motion_sent": False,
                "evidence": ["scheduler_dispatch_confirmed"],
            }

    result = asyncio.run(schedule(plan, DictExecutor()))
    assert result.passed
    assert result.operations["a"].legacy is True


def test_legacy_non_boolean_passed_fails_closed() -> None:
    class BadExecutor:
        async def execute(self, operation: RobotOperation) -> dict:
            return {"passed": "yes"}

    plan = _plan([_operation("a", "piper_0")], [["a"]])
    result = asyncio.run(schedule(plan, BadExecutor()))
    assert not result.passed
    assert "explicit boolean" in (result.operations["a"].reason or "")


def test_runner_document_is_v1_json_safe_with_optional_extension() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])
    runner = OperationPlanRunner(plan, StrictExecutor())
    result = asyncio.run(runner.run())
    document = runner.serialize(result)
    assert document["schema"] == RUN_SCHEMA == "rocobrick-operation-plan-run/v1"
    assert {
        "schema",
        "passed",
        "operations",
        "layers",
        "evidence",
    } <= set(document)
    assert "readiness" not in document
    assert "cleanup" not in document
    assert document["passed"] is True
    assert document["operations"]["a"]["terminal"] is True
    assert document["operations"]["a"]["motion_sent"] is False
    assert {
        "operation_id",
        "robot_id",
        "status",
        "passed",
        "detail",
    } <= set(document["operations"]["a"])
    assert {
        "layer_index",
        "status",
        "operation_ids",
    } <= set(document["layers"][0])
    json.dumps(document, allow_nan=False)

    extended = serialize_run_result(
        result,
        readiness={"robots": {"piper_0": {"ready": True}}},
        cleanup={"cleanup_passed": True},
    )
    assert extended["readiness"] == {"robots": {"piper_0": {"ready": True}}}
    assert extended["cleanup"] == {"cleanup_passed": True}
    with pytest.raises(ValueError):
        serialize_run_result(result, readiness={"bad": object()})


def test_runner_attaches_configured_readiness_and_cleanup() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])
    runner = OperationPlanRunner(
        plan,
        StrictExecutor(),
        readiness={"ready": True},
        cleanup={"cleanup_passed": True},
    )
    document = runner.serialize(asyncio.run(runner.run()))
    assert document["readiness"] == {"ready": True}
    assert document["cleanup"] == {"cleanup_passed": True}


def test_runner_timeout_serializes_unsafe_terminals() -> None:
    plan = _plan([_operation("a", "piper_0")], [["a"]])
    runner = OperationPlanRunner(
        plan,
        StrictExecutor(delays={"a": 10.0}),
        operation_timeout_s=0.05,
        cancel_timeout_s=0.5,
    )
    result = asyncio.run(runner.run())
    document = runner.serialize(result)
    assert document["passed"] is False
    assert document["unsafe_terminals"] == ["a"]
    assert document["operations"]["a"]["motion_sent"] is None
    assert document["operations"]["a"]["deadline_exceeded"] is True
    assert document["cancellations"][0]["safe"] is False


def test_scheduler_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        MultiRobotScheduler(operation_timeout_s=0)
    with pytest.raises(ValueError):
        MultiRobotScheduler(cancel_timeout_s=-1)
