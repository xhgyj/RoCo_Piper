"""Shared offline fixtures for strictly bound successor pre-phases."""

from dataclasses import replace

from rocobrick.robot_operation_plan import OperationTarget, RobotOperation
from rocobrick.unified_dual_arm_runtime import (
    SharedSceneSnapshot,
    SuccessorPrePhaseRequest,
    failed_pre_phase,
)


def operation(operation_id="next-op", robot_id="piper_0"):
    """Construct an immutable operation with a static PICK target.

    Returns:
        An offline immutable motion operation with a static PICK target.
    """
    return RobotOperation(
        operation_id,
        "task",
        "PICK",
        robot_id,
        (robot_id,),
        {"executor": robot_id},
        OperationTarget({"physical_instance_id": "part-1"}),
        {},
        (),
        (),
        (f"robot:{robot_id}",),
        (),
    )


def snapshot(participants=("piper_0",), facts=None):
    """Construct one admitted shared observation with known physical facts.

    Returns:
        An admitted offline shared snapshot.
    """
    return SharedSceneSnapshot(
        {
            "session_id": "session-1",
            "manifest_sha256": "a" * 64,
            "scene_version": 1,
            "sim_time_s": 1.0,
            "robots": {
                robot: {
                    "authorized": True,
                    "terminal_safe": True,
                    "holding_known": True,
                    "supporting_known": True,
                    "facts": (facts or {}).get(robot, {}),
                }
                for robot in participants
            },
        },
        1.1,
        0.5,
    )


def request(*, policy=None):
    """Construct a single predecessor/successor transition.

    Returns:
        A bound offline transition request.
    """
    previous = operation("previous-op")
    return SuccessorPrePhaseRequest(
        "transition-1",
        (previous,),
        {"previous-op": {"terminal_safe": True}},
        None if policy else operation(),
        ("piper_0",),
        ("robot:piper_0",),
        {"robot:piper_0": "previous-op"},
        snapshot(),
        policy,
    )


def success(item, final_facts=None):
    """Create physical evidence bound to every request field.

    Returns:
        Scripted terminal-safe arrival evidence for offline tests.
    """
    return replace(
        failed_pre_phase(item, ""),
        planned=True,
        prepared=True,
        started=True,
        arrival_confirmed=True,
        terminal=True,
        safe=True,
        adoption_ready=True,
        final_facts=final_facts or {robot: {} for robot in item.participants},
        evidence={"pre_phase": True},
        reason=None,
    )
