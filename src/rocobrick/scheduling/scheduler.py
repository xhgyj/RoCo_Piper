"""Deterministic offline operation-event admission model.

An admission is ONLY permission to request coordination, never permission to
send motion. External dependencies require whole-operation physical success.
The injected evidence port must validate session, identity, attempt, freshness,
scene and physical/terminal evidence. No production verifier is supplied here.
This reducer has no threads, clocks, framework imports or execution side effects.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from rocobrick.contracts.phases import ResourceKind, ResourceRequirement
from rocobrick.scheduling.resource_manager import ResourceManager
from rocobrick.task_plan.dependency_graph import DependencyGraph
from rocobrick.task_plan.schema import TaskOperation, TaskPlan


class OperationState(StrEnum):
    """Admission states, not controller action states."""

    WAITING = "waiting"
    ADMITTED = "admitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    UNCERTAIN = "uncertain"


class EventKind(StrEnum):
    """Only independently verified completion can satisfy a dependency."""

    PHYSICAL_SUCCESS = "physical_success"
    FAILURE = "failure"
    CANCEL_REQUESTED = "cancel_requested"
    FACT_REVOKED = "fact_revoked"


@dataclass(frozen=True)
class OperationEvent:
    """Evidence references scoped to one session and operation attempt."""

    event_id: str
    session_id: str
    plan_id: str
    operation_id: str
    attempt: int
    kind: EventKind
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Enforce strict event identity and immutable reference collections."""
        for value in (self.event_id, self.session_id, self.plan_id, self.operation_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("event identities must be nonempty strings")
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if not isinstance(self.kind, EventKind):
            raise ValueError("kind must be EventKind")
        if not isinstance(self.evidence_ids, tuple) or any(
            not isinstance(value, str) or not value.strip()
            for value in self.evidence_ids
        ):
            raise ValueError("evidence_ids must be a tuple of nonempty strings")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("duplicate evidence reference")
        if self.kind is EventKind.PHYSICAL_SUCCESS and not self.evidence_ids:
            raise ValueError("physical success requires evidence references")


class VerificationPort(Protocol):
    """Consumer-owned port for world and skill-specific physical validators.

    Returning true for completion means ALL required operation phases, physical
    predicates and terminal obligations are verified, not merely plan/action
    success. Returning true for release additionally proves no continuing
    holding/support/contact obligation needs any of the held resources.
    """

    def ready(self, operation: TaskOperation, attempt: int) -> bool:
        """Check fresh physical conditions and consistent world provenance."""
        ...

    def completed(self, operation: TaskOperation, event: OperationEvent) -> bool:
        """Validate whole-operation physical success for this exact attempt."""
        ...

    def releasable(
        self, operation: TaskOperation, attempt: int, evidence_ids: tuple[str, ...]
    ) -> bool:
        """Validate terminal-safe evidence and absence of continuing obligations."""
        ...


class DependencyScheduler:
    """Own offline admission, event identity and task leases for a single plan.

    One attempt per operation in this first slice. Retry, lease transfer,
    persistence, phase expansion and execution are intentionally unavailable.
    A conservative plan-wide quarantine follows revocation/uncertain execution.
    Independent work after a *verified safe* failure can continue after explicit
    release. Unknown terminal state never expires into availability.
    """

    def __init__(
        self, plan: TaskPlan, session_id: str, verification: VerificationPort
    ) -> None:
        """Bind immutable inputs and a verifier; allocate no runtime resources."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be nonempty")
        self.plan = plan
        self.session_id = session_id
        self._verification = verification
        self._graph = DependencyGraph(plan)
        self._operations = {operation.id: operation for operation in plan.operations}
        self._states = dict.fromkeys(self._operations, OperationState.WAITING)
        self._leases = ResourceManager()
        self._events: dict[str, OperationEvent] = {}
        self._quarantined = False

    def state(self, operation_id: str) -> OperationState:
        """Return an operation's current state; unknown identities are errors."""
        return self._states[operation_id]

    def held_resources(self, operation_id: str) -> tuple[ResourceRequirement, ...]:
        """Return immutable retained-lease evidence, not a mutable manager."""
        self._operations[operation_id]
        return self._leases.held(operation_id)

    def _resources(self, operation: TaskOperation) -> tuple[ResourceRequirement, ...]:
        resources = [
            ResourceRequirement(ResourceKind.ROBOT, f"robot:{robot}")
            for robot in operation.participants
        ]
        if operation.object_id is not None:
            resources.append(
                ResourceRequirement(ResourceKind.PART, f"part:{operation.object_id}")
            )
        return tuple(resources)

    def waiting_reasons(self, operation_id: str) -> tuple[str, ...]:
        """Return eligibility blockers without acquiring or sending anything."""
        operation = self._operations[operation_id]
        if self._states[operation_id] is not OperationState.WAITING:
            return (f"state:{self._states[operation_id].value}",)
        reasons: list[str] = []
        if self._quarantined:
            reasons.append("plan_quarantined")
        for predecessor in self._graph.predecessors(operation_id):
            if self._states[predecessor] is not OperationState.SUCCEEDED:
                reasons.append(f"predecessor:{predecessor}")
        for resource in self._leases.conflicts(
            operation_id, self._resources(operation)
        ):
            reasons.append(f"resource:{resource.resource_id}")
        # Exact booleans: a truthy legacy dict is never a verified fact.
        if self._verification.ready(operation, 1) is not True:
            reasons.append("physical_preconditions")
        return tuple(reasons)

    def eligible(self) -> tuple[str, ...]:
        """Return ready requests, independent of array order or layer barriers."""
        return tuple(
            identity
            for identity in sorted(self._operations)
            if not self.waiting_reasons(identity)
        )

    def admit(self, operation_id: str) -> bool:
        """Recheck conditions and atomically claim task resources, not motion.

        Returns:
            Whether the request was admitted to coordination.
        """
        if self.waiting_reasons(operation_id):
            return False
        if not self._leases.claim(
            operation_id, self._resources(self._operations[operation_id])
        ):
            return False
        self._states[operation_id] = OperationState.ADMITTED
        return True

    def apply(self, event: OperationEvent) -> bool:
        """Apply an event once; stale identity or unverified success cannot advance.

        Reusing an event ID with different contents is an audit error. Rejected
        events remain recorded: submit a new event after new evidence arrives.
        An unsafe result keeps leases and quarantines further admissions.

        Returns:
            Whether the event caused a state transition.
        """
        if not isinstance(event, OperationEvent):
            raise ValueError("expected OperationEvent")
        previous = self._events.get(event.event_id)
        if previous is not None:
            if previous != event:
                raise ValueError("event identity reused with different payload")
            return False
        if (
            event.session_id != self.session_id
            or event.plan_id != self.plan.plan_id
            or event.operation_id not in self._operations
            or event.attempt != 1
        ):
            return False
        self._events[event.event_id] = event
        identity = event.operation_id
        state = self._states[identity]
        if event.kind is EventKind.FACT_REVOKED:
            if state not in (OperationState.ADMITTED, OperationState.SUCCEEDED):
                return False
            self._states[identity] = OperationState.UNCERTAIN
            self._quarantined = True
            self._block_descendants(identity)
            return True
        if state is not OperationState.ADMITTED:
            return False
        if event.kind is EventKind.PHYSICAL_SUCCESS:
            if (
                self._verification.completed(self._operations[identity], event)
                is not True
            ):
                return False
            self._states[identity] = OperationState.SUCCEEDED
        elif event.kind is EventKind.FAILURE:
            self._states[identity] = OperationState.FAILED
            # Unknown execution state affects even graph-independent requests.
            self._quarantined = True
            self._block_descendants(identity)
        else:
            self._states[identity] = OperationState.UNCERTAIN
            self._quarantined = True
            self._block_descendants(identity)
        return True

    def _block_descendants(self, operation_id: str) -> None:
        pending = list(self._graph.successors(operation_id))
        seen: set[str] = set()
        while pending:
            identity = pending.pop()
            if identity in seen:
                continue
            seen.add(identity)
            state = self._states[identity]
            if state is OperationState.WAITING:
                self._states[identity] = OperationState.BLOCKED
            elif state in (OperationState.ADMITTED, OperationState.SUCCEEDED):
                self._states[identity] = OperationState.UNCERTAIN
            pending.extend(self._graph.successors(identity))

    def release(self, operation_id: str, evidence_ids: tuple[str, ...]) -> bool:
        """Release only on independent terminal/physical proof, never on timeout.

        Revoked/uncertain cases require future explicit reconciliation, so this
        initial model cannot clear their quarantine or resurrect blocked tasks.

        Returns:
            Whether independent evidence authorized release.
        """
        operation = self._operations[operation_id]
        if self._states[operation_id] not in (
            OperationState.SUCCEEDED,
            OperationState.FAILED,
        ):
            return False
        if (
            not isinstance(evidence_ids, tuple)
            or not evidence_ids
            or any(
                not isinstance(item, str) or not item.strip() for item in evidence_ids
            )
        ):
            raise ValueError("release requires immutable evidence references")
        if self._verification.releasable(operation, 1, evidence_ids) is not True:
            return False
        self._leases.release_authorized(operation_id)
        self._quarantined = any(
            state is OperationState.UNCERTAIN
            or (state is OperationState.FAILED and bool(self._leases.held(identity)))
            for identity, state in self._states.items()
        )
        return True
