"""Shared robot-task-plan/v2 documents for focused offline tests."""

from __future__ import annotations


def _operation(
    operation_id: str,
    skill: str,
    executor: str,
    object_id: str | None = None,
    target_ref: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": operation_id,
        "skill": skill,
        "executor": executor,
        "participants": [executor],
    }
    if object_id is not None:
        payload["object_id"] = object_id
    if target_ref is not None:
        payload["target_ref"] = target_ref
    return payload


def design_example() -> dict[str, object]:
    """Return the three-operation design example from the architecture note.

    Returns:
        A fresh mutable document; callers may mutate it for negative tests.
    """
    return {
        "schema": "robot-task-plan/v2",
        "plan_id": "assembly-example",
        "operations": [
            _operation("pick_a", "PICK", "piper_0", "brick_a"),
            _operation("place_a", "PLACE_DOWN", "piper_0", "brick_a", "slot_a"),
            _operation("pick_b", "PICK", "piper_1", "brick_b"),
        ],
        "dependencies": [
            {
                "before": "pick_a",
                "after": "place_a",
                "condition": "physical_success",
            }
        ],
    }


def graph_document(order: tuple[str, ...] = ("A", "B", "C")) -> dict[str, object]:
    """Return A -> C with independent B in the requested operation order.

    Args:
        order: Permutation of ``A``, ``B``, and ``C`` for the operations array.

    Returns:
        A fresh mutable document with the explicit A -> C dependency only.
    """
    operations = [
        _operation("A", "PICK", "robot_alpha", "brick_a"),
        _operation("B", "PICK", "robot_beta", "brick_b"),
        _operation("C", "PLACE_DOWN", "robot_alpha", "brick_a", "slot_a"),
    ]
    by_id = {str(operation["id"]): operation for operation in operations}
    return {
        "schema": "robot-task-plan/v2",
        "plan_id": "graph",
        "operations": [by_id[operation_id] for operation_id in order],
        "dependencies": [
            {
                "before": "A",
                "after": "C",
                "condition": "physical_success",
            }
        ],
    }
