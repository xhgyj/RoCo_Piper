"""Offline tests for the multi-robot session plan-identity gate.

Everything runs against the fake ``ProcessRunner``/clock of
``test_multi_robot_live_session`` - no Docker, no Isaac, no ROS, and no
motion is ever involved. The tests pin both accepted plan schemas of
``MultiRobotIsaacMoveItSession._verify_plan_robots``: the frozen
``assembly-plan/v2`` STAP contract (behavior unchanged) and the strict
structure-only ``robot-operation-plan/v1`` contract parsed with
``source_plan=None``. A plan whose robot set disagrees with the session, an
unknown schema, or a malformed plan must fail closed BEFORE any process
starts, and an explicitly disabled check keeps its legacy record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_multi_robot_live_session import (
    ROBOTS,
    assert_motion_free,
    make_runner,
    make_session,
    write_stap_plan,
)

from rocobrick.runtime_session import (
    MultiRobotSessionError,
    RuntimeSessionStatus,
)

ROBOT_OPERATION_PLAN_SCHEMA = "robot-operation-plan/v1"
OPERATION_PLAN_ID = "operation-plan-identity"
SOURCE_PLAN_ID = "assembly-plan-source"


def wait_operation(operation_id: str, robot_id: str) -> dict:
    """One minimal WAIT operation executed by ``robot_id``.

    Returns:
        The raw operation document of the external plan format.
    """
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
        "resources": [f"robot:{robot_id}"],
        "evidence_requirements": [],
    }


def operation_plan_document() -> dict:
    """One structurally valid plan for exactly the session robots.

    Returns:
        The raw ``robot-operation-plan/v1`` document.
    """
    return {
        "schema": ROBOT_OPERATION_PLAN_SCHEMA,
        "plan_id": OPERATION_PLAN_ID,
        "source_plan_id": SOURCE_PLAN_ID,
        "robots": list(ROBOTS),
        "operations": [
            wait_operation("wait-piper-0", "piper_0"),
            wait_operation("wait-piper-1", "piper_1"),
        ],
        "dependencies": [],
        "execution_layers": [["wait-piper-0", "wait-piper-1"]],
        "planned_schedule": [
            [
                {"operation_id": "wait-piper-0", "robot_id": "piper_0"},
                {"operation_id": "wait-piper-1", "robot_id": "piper_1"},
            ]
        ],
        "evidence_requirements": [],
    }


def write_operation_plan(path: Path, document: dict) -> Path:
    """Write one robot-operation-plan document variant to ``path``.

    Returns:
        The written plan path, for use as ``plan_path``.
    """
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def session_document(tmp_path: Path) -> dict:
    """The lifecycle evidence document of the session under test.

    Returns:
        The decoded ``multi-robot-session.json`` evidence document.
    """
    return json.loads(
        (tmp_path / "session" / "multi-robot-session.json").read_text(
            encoding="utf-8"
        )
    )


def test_operation_plan_robot_mismatch_refuses_before_any_process(tmp_path):
    """A plan robot set that differs from the session never starts a process."""
    plan = write_operation_plan(
        tmp_path / "operation-plan.json", operation_plan_document()
    )
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner, plan_path=plan, robot_ids=("piper_0",))
    with pytest.raises(MultiRobotSessionError, match="plan_robots"):
        session.start()
    assert runner.commands == []
    assert runner.launches == []
    assert runner.execs == []
    assert session.status is RuntimeSessionStatus.FAILED
    assert session.plan_check == {}
    document = session_document(tmp_path)
    assert document["status"] == "FAILED"
    assert document["plan_check"] == {}
    assert any("plan_robots" in error for error in document["errors"])


def test_valid_operation_plan_is_verified_and_start_stays_motion_free(tmp_path):
    """A matching operation plan verifies with its full identity record."""
    plan = write_operation_plan(
        tmp_path / "operation-plan.json", operation_plan_document()
    )
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner, plan_path=plan)
    session.start()
    assert session.status is RuntimeSessionStatus.READY
    expected = {
        "verified": True,
        "schema": ROBOT_OPERATION_PLAN_SCHEMA,
        "plan_id": OPERATION_PLAN_ID,
        "source_plan_id": SOURCE_PLAN_ID,
        "plan_robots": list(ROBOTS),
        "session_robots": list(ROBOTS),
    }
    assert session.plan_check == expected
    assert session_document(tmp_path)["plan_check"] == expected
    # Verification dispatched no motion: one container, no command traffic.
    assert len(runner.container_names) == 1
    assert_motion_free(runner.commands)


def test_stap_plan_verification_shape_is_unchanged(tmp_path):
    """The STAP path keeps its exact legacy plan_check record."""
    runner = make_runner(tmp_path)
    session = make_session(
        tmp_path, runner, plan_path=write_stap_plan(tmp_path / "plan.json")
    )
    session.start()
    assert session.status is RuntimeSessionStatus.READY
    plan_check = session.plan_check
    assert set(plan_check) == {
        "verified",
        "plan_id",
        "plan_robots",
        "session_robots",
    }
    assert plan_check["verified"] is True
    assert plan_check["plan_robots"] == list(ROBOTS)
    assert plan_check["session_robots"] == list(ROBOTS)
    assert isinstance(plan_check["plan_id"], str) and plan_check["plan_id"]
    assert session_document(tmp_path)["plan_check"] == plan_check


def test_malformed_operation_plan_fails_before_any_process(tmp_path):
    """A structurally invalid operation plan fails the same closed gate."""
    document = operation_plan_document()
    document["unexpected"] = True
    plan = write_operation_plan(tmp_path / "operation-plan.json", document)
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner, plan_path=plan)
    with pytest.raises(MultiRobotSessionError, match="plan_robots"):
        session.start()
    assert runner.commands == []
    assert runner.launches == []
    assert runner.execs == []
    assert session.status is RuntimeSessionStatus.FAILED


def test_unknown_operation_plan_schema_fails_before_any_process(tmp_path):
    """An unknown plan schema never bypasses the gate into a process start."""
    document = operation_plan_document()
    document["schema"] = "robot-operation-plan/v2"
    plan = write_operation_plan(tmp_path / "operation-plan.json", document)
    runner = make_runner(tmp_path)
    session = make_session(tmp_path, runner, plan_path=plan)
    with pytest.raises(MultiRobotSessionError, match="plan_robots"):
        session.start()
    assert runner.commands == []
    assert runner.launches == []
    assert runner.execs == []
    assert session.status is RuntimeSessionStatus.FAILED


def test_verification_disabled_keeps_legacy_behavior_for_any_plan(tmp_path):
    """A disabled check records its legacy reason and parses no plan at all."""
    plan = tmp_path / "operation-plan.json"
    plan.write_text("not a plan document at all", encoding="utf-8")
    runner = make_runner(tmp_path)
    session = make_session(
        tmp_path, runner, plan_path=plan, verify_plan_robots=False
    )
    session.start()
    assert session.status is RuntimeSessionStatus.READY
    assert session.plan_check == {
        "verified": False,
        "reason": "disabled by config",
    }
    assert session_document(tmp_path)["plan_check"] == session.plan_check
