"""Deterministic dependency graph for ``robot-task-plan/v2`` operations."""

from __future__ import annotations

from collections.abc import Collection

from .schema import PHYSICAL_SUCCESS, TaskPlan, TaskPlanError


class DependencyGraph:
    """Read-only dependency graph with deterministic operation ordering.

    The graph is derived only from explicit dependencies. Operation array order
    is never treated as an implicit dependency, and every returned ID tuple is
    sorted independently of document order and caller collection order.
    """

    def __init__(self, plan: TaskPlan) -> None:
        """Build and validate the graph from a frozen task plan.

        Args:
            plan: Parsed task plan.

        Raises:
            TaskPlanError: If operation identities are duplicated or unknown to
                a dependency, an edge is a self-edge or duplicate, or the
                dependencies contain a cycle.
        """
        predecessors: dict[str, set[str]] = {}
        successors: dict[str, set[str]] = {}
        for operation in plan.operations:
            if operation.id in predecessors:
                raise TaskPlanError(f"duplicate operation id {operation.id!r}")
            predecessors[operation.id] = set()
            successors[operation.id] = set()
        for dependency in plan.dependencies:
            before = dependency.before
            after = dependency.after
            if dependency.condition != PHYSICAL_SUCCESS:
                raise TaskPlanError(
                    f"dependency {before!r} -> {after!r} has unsupported "
                    f"condition {dependency.condition!r}; expected "
                    f"{PHYSICAL_SUCCESS!r}"
                )
            if before not in predecessors or after not in successors:
                raise TaskPlanError(
                    f"dependency {before!r} -> {after!r} references an "
                    "unknown operation"
                )
            if before == after:
                raise TaskPlanError(f"operation {before!r} depends on itself")
            if after in successors[before]:
                raise TaskPlanError(f"duplicate dependency {before!r} -> {after!r}")
            successors[before].add(after)
            predecessors[after].add(before)
        self._predecessors: dict[str, tuple[str, ...]] = {
            operation_id: tuple(sorted(values))
            for operation_id, values in predecessors.items()
        }
        self._successors: dict[str, tuple[str, ...]] = {
            operation_id: tuple(sorted(values))
            for operation_id, values in successors.items()
        }
        self._operation_ids = frozenset(predecessors)
        self._validate_acyclic(predecessors, successors)

    @staticmethod
    def _validate_acyclic(
        predecessors: dict[str, set[str]],
        successors: dict[str, set[str]],
    ) -> None:
        indegree = {
            operation_id: len(values) for operation_id, values in predecessors.items()
        }
        queue = [operation_id for operation_id, count in indegree.items() if count == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for successor in successors[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)
        if visited != len(indegree):
            blocked = sorted(
                operation_id for operation_id, count in indegree.items() if count > 0
            )
            raise TaskPlanError(f"dependency cycle detected among operations {blocked}")

    @staticmethod
    def _lookup(
        table: dict[str, tuple[str, ...]], operation_id: str
    ) -> tuple[str, ...]:
        if not isinstance(operation_id, str) or operation_id not in table:
            raise TaskPlanError(f"unknown operation id {operation_id!r}")
        return table[operation_id]

    def predecessors(self, operation_id: str) -> tuple[str, ...]:
        """Return sorted direct predecessors of one operation.

        Args:
            operation_id: Known operation identity.

        Returns:
            Sorted operation IDs with a direct edge into ``operation_id``.

        Raises:
            TaskPlanError: If ``operation_id`` is not in the plan.
        """
        return self._lookup(self._predecessors, operation_id)

    def successors(self, operation_id: str) -> tuple[str, ...]:
        """Return sorted direct successors of one operation.

        Args:
            operation_id: Known operation identity.

        Returns:
            Sorted operation IDs directly depending on ``operation_id``.

        Raises:
            TaskPlanError: If ``operation_id`` is not in the plan.
        """
        return self._lookup(self._successors, operation_id)

    def ready(self, completed: Collection[str]) -> tuple[str, ...]:
        """Return sorted operations whose direct predecessors are complete.

        Args:
            completed: Completed operation IDs. Iteration order and duplicate
                entries are irrelevant.

        Returns:
            Sorted non-completed operation IDs that are ready to request
            advancement. Readiness is not motion authorization.

        Raises:
            TaskPlanError: If ``completed`` is a string, contains a non-string,
                or names an operation outside this plan.
        """
        if isinstance(completed, (str, bytes, bytearray)):
            raise TaskPlanError(
                "completed must be a collection of operation IDs, not a string"
            )
        completed_ids: set[str] = set()
        for operation_id in completed:
            if not isinstance(operation_id, str):
                raise TaskPlanError("completed must contain only operation IDs")
            if operation_id not in self._operation_ids:
                raise TaskPlanError(
                    f"completed contains unknown operation id {operation_id!r}"
                )
            completed_ids.add(operation_id)
        return tuple(
            sorted(
                operation_id
                for operation_id in self._operation_ids
                if operation_id not in completed_ids
                and all(
                    predecessor in completed_ids
                    for predecessor in self._predecessors[operation_id]
                )
            )
        )
