"""Focused tests for the deterministic offline dependency graph."""

from __future__ import annotations

import pytest
from plan_documents import graph_document

from rocobrick.task_plan import (
    PHYSICAL_SUCCESS,
    DependencyGraph,
    TaskDependency,
    TaskOperation,
    TaskPlan,
    TaskPlanError,
    parse_task_plan,
)


def _edge(before: str, after: str) -> dict[str, object]:
    return {
        "before": before,
        "after": after,
        "condition": PHYSICAL_SUCCESS,
    }


def _wait(operation_id: str) -> TaskOperation:
    return TaskOperation(operation_id, "WAIT", "robot", ["robot"])


def _graph(order: tuple[str, ...] = ("A", "B", "C")) -> DependencyGraph:
    return DependencyGraph(parse_task_plan(graph_document(order)))


def test_c_is_ready_after_a_without_waiting_for_independent_b() -> None:
    """A -> C only: C may advance after A even when B is unfinished."""
    graph = _graph()
    assert graph.ready(set()) == ("A", "B")
    assert graph.predecessors("C") == ("A",)
    assert graph.successors("A") == ("C",)
    assert graph.ready({"A"}) == ("B", "C")
    assert graph.ready({"A", "B"}) == ("C",)
    assert graph.ready({"A", "B", "C"}) == ()


@pytest.mark.parametrize(
    "order",
    [
        ("A", "B", "C"),
        ("C", "A", "B"),
        ("B", "C", "A"),
        ("C", "B", "A"),
    ],
)
def test_operation_array_order_is_never_an_implicit_dependency(
    order: tuple[str, ...],
) -> None:
    """Shuffling the operations array cannot change graph edges or readiness."""
    graph = _graph(order)
    assert graph.ready(set()) == ("A", "B")
    assert graph.predecessors("C") == ("A",)
    assert graph.successors("A") == ("C",)
    assert graph.ready({"A"}) == ("B", "C")


def test_multiple_predecessors_are_sorted_and_deterministic() -> None:
    """D depends on A and B; returned IDs are sorted, not array ordered."""
    document = graph_document()
    operations = document["operations"]
    assert isinstance(operations, list)
    operations.append(
        {
            "id": "D",
            "skill": "WAIT",
            "executor": "robot_alpha",
            "participants": ["robot_alpha"],
        }
    )
    document["dependencies"] = [
        _edge("A", "C"),
        _edge("A", "D"),
        _edge("B", "D"),
    ]
    graph = DependencyGraph(parse_task_plan(document))
    assert graph.predecessors("D") == ("A", "B")
    assert graph.successors("A") == ("C", "D")
    assert graph.ready({"A"}) == ("B", "C")
    assert graph.ready({"B"}) == ("A",)
    assert graph.ready({"A", "B"}) == ("C", "D")
    assert graph.ready({"A", "B", "C"}) == ("D",)


def test_completed_collection_order_and_duplicates_do_not_matter() -> None:
    """Readiness depends on the completed set, not its iteration order."""
    graph = _graph()
    assert graph.ready(["B", "A", "A"]) == ("C",)
    assert graph.ready(("B", "A")) == ("C",)
    assert graph.ready({"A", "B"}) == ("C",)


def test_lookups_and_ready_reject_unknown_or_non_string_ids() -> None:
    """Graph queries fail closed on unknown or malformed operation IDs."""
    graph = _graph()
    with pytest.raises(TaskPlanError, match="unknown operation id"):
        graph.predecessors("Z")
    with pytest.raises(TaskPlanError, match="unknown operation id"):
        graph.successors("Z")
    with pytest.raises(TaskPlanError, match="collection of operation IDs"):
        graph.ready("A")
    with pytest.raises(TaskPlanError, match="only operation IDs"):
        graph.ready([1])
    with pytest.raises(TaskPlanError, match="unknown operation id"):
        graph.ready(["Z"])


def test_graph_rejects_cycles_from_direct_typed_input() -> None:
    """The graph independently rejects a cycle bypassing parser checks."""
    plan = TaskPlan(
        schema="robot-task-plan/v2",
        plan_id="cycle",
        operations=(_wait("A"), _wait("B")),
        dependencies=(
            TaskDependency("A", "B", PHYSICAL_SUCCESS),
            TaskDependency("B", "A", PHYSICAL_SUCCESS),
        ),
    )
    with pytest.raises(TaskPlanError, match="cycle"):
        DependencyGraph(plan)


def test_graph_rejects_duplicate_edges_and_operation_ids() -> None:
    """Direct typed input cannot inject duplicate graph identities or edges."""
    duplicate_edge = TaskPlan(
        schema="robot-task-plan/v2",
        plan_id="duplicate-edge",
        operations=(_wait("A"), _wait("B")),
        dependencies=(
            TaskDependency("A", "B", PHYSICAL_SUCCESS),
            TaskDependency("A", "B", PHYSICAL_SUCCESS),
        ),
    )
    with pytest.raises(TaskPlanError, match="duplicate dependency"):
        DependencyGraph(duplicate_edge)
    duplicate_operation = TaskPlan(
        schema="robot-task-plan/v2",
        plan_id="duplicate-operation",
        operations=(_wait("A"), _wait("A")),
        dependencies=(),
    )
    with pytest.raises(TaskPlanError, match="duplicate operation id"):
        DependencyGraph(duplicate_operation)
