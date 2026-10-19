"""Frozen robot-task-plan/v2 ingress contract and dependency graph.

This package is offline-only: it validates task identity, executor assignment,
opaque references, and hard physical-success dependencies. Optional skill
parameters are opaque JSON data, not motion authority. It performs no
structural binding, scheduling, allocation, or motion dispatch.
"""

from .dependency_graph import DependencyGraph
from .parser import parse_task_plan
from .schema import (
    PHYSICAL_SUCCESS,
    PLACEMENT_SKILLS,
    REJECTED_SKILLS,
    SUPPORTED_SKILLS,
    TASK_PLAN_SCHEMA,
    TaskDependency,
    TaskOperation,
    TaskPlan,
    TaskPlanError,
)

__all__ = [
    "PHYSICAL_SUCCESS",
    "PLACEMENT_SKILLS",
    "REJECTED_SKILLS",
    "SUPPORTED_SKILLS",
    "TASK_PLAN_SCHEMA",
    "DependencyGraph",
    "TaskDependency",
    "TaskOperation",
    "TaskPlan",
    "TaskPlanError",
    "parse_task_plan",
]
