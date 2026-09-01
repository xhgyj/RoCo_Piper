"""Upper-level topology planning without execution details."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


def _records(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"topology {name} must be an array")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"topology {name}[{index}] must be an object")
        result.append(item)
    return tuple(result)


def _integer(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"topology field {key} must be an integer")
    return value


@dataclass(frozen=True)
class PlanningProblem:
    """Structure and resources visible to an upper task planner."""

    episode_id: str
    topology: Mapping[str, object]
    initially_assembled_part_ids: tuple[int, ...]
    available_arm_ids: tuple[str, ...]


@dataclass(frozen=True)
class AssemblyTask:
    """One upper-level decision about what an assigned arm must assemble."""

    task_id: str
    target_part_id: int
    connection_ids: tuple[int, ...]
    assigned_arm: str
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class TaskPlan:
    """Deterministic planner output consumed by execution planning."""

    episode_id: str
    tasks: tuple[AssemblyTask, ...]


class TaskPlanner(Protocol):
    """Python boundary implemented by an upper assembly planner."""

    def plan(self, problem: PlanningProblem) -> TaskPlan:
        """Return assignments and dependencies without motion actions."""
        ...


class TopologyTaskPlanner:
    """Reference support-first planner with stable branch ownership."""

    def plan(self, problem: PlanningProblem) -> TaskPlan:
        """Build a deterministic, one-step-prefetchable assembly order.

        Returns:
            Complete task plan preserving structural support dependencies.
        """
        if not problem.available_arm_ids:
            raise ValueError("planning requires at least one available arm")
        parts = _records(problem.topology.get("parts"), "parts")
        connections = _records(problem.topology.get("connections"), "connections")
        all_parts = {_integer(part, "id") for part in parts}
        assembled = set(problem.initially_assembled_part_ids)
        remaining = all_parts - assembled
        incoming: dict[int, list[Mapping[str, object]]] = {
            part_id: [] for part_id in remaining
        }
        first_seen: dict[int, int] = {}
        for index, connection in enumerate(connections):
            target_id = _integer(connection, "hole_id")
            if target_id in incoming:
                incoming[target_id].append(connection)
                first_seen.setdefault(target_id, index)
        missing = sorted(part_id for part_id, items in incoming.items() if not items)
        if missing:
            raise ValueError(
                f"unassembled parts have no incoming connection: {missing}"
            )

        assignments: dict[int, str] = {}
        assigned_children: dict[int, int] = {}
        depth_by_part = {
            part_id: 0 for part_id in problem.initially_assembled_part_ids
        }
        loads = {arm_id: 0 for arm_id in problem.available_arm_ids}
        tasks: list[AssemblyTask] = []
        while remaining:
            ready = [
                part_id
                for part_id in remaining
                if {
                    _integer(connection, "stud_id")
                    for connection in incoming[part_id]
                }
                <= assembled
            ]
            if not ready:
                raise ValueError(
                    "topology has no support-complete next part; "
                    f"assembled={sorted(assembled)} remaining={sorted(remaining)}"
                )
            target_id = min(
                ready,
                key=lambda item: (
                    1
                    + max(
                        depth_by_part[_integer(connection, "stud_id")]
                        for connection in incoming[item]
                    ),
                    first_seen[item],
                    item,
                ),
            )
            structural_parents = sorted(
                {
                    _integer(connection, "stud_id")
                    for connection in incoming[target_id]
                    if _integer(connection, "stud_id") in assignments
                }
            )
            arm_id = self._assign_arm(
                structural_parents,
                assignments,
                assigned_children,
                loads,
                problem.available_arm_ids,
            )
            structural_dependencies = {
                f"assemble_{part_id}" for part_id in structural_parents
            }
            tasks.append(
                AssemblyTask(
                    task_id=f"assemble_{target_id}",
                    target_part_id=target_id,
                    connection_ids=tuple(
                        sorted(
                            _integer(connection, "id")
                            for connection in incoming[target_id]
                        )
                    ),
                    assigned_arm=arm_id,
                    depends_on=tuple(sorted(structural_dependencies)),
                )
            )
            assignments[target_id] = arm_id
            depth_by_part[target_id] = 1 + max(
                depth_by_part[_integer(connection, "stud_id")]
                for connection in incoming[target_id]
            )
            for parent in structural_parents:
                assigned_children[parent] = assigned_children.get(parent, 0) + 1
            loads[arm_id] += 1
            assembled.add(target_id)
            remaining.remove(target_id)
        return TaskPlan(problem.episode_id, tuple(tasks))

    @staticmethod
    def _assign_arm(
        parents: list[int],
        assignments: Mapping[int, str],
        assigned_children: Mapping[int, int],
        loads: Mapping[str, int],
        available_arms: tuple[str, ...],
    ) -> str:
        if len(parents) == 1 and assigned_children.get(parents[0], 0) == 0:
            return assignments[parents[0]]
        candidates = (
            tuple(dict.fromkeys(assignments[parent] for parent in parents))
            if len(parents) > 1
            else available_arms
        )
        arm_order = {arm_id: index for index, arm_id in enumerate(available_arms)}
        return min(candidates, key=lambda arm_id: (loads[arm_id], arm_order[arm_id]))
