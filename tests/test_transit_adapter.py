"""Offline contract tests for the single-dispatch TRANSIT adapter."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from rocobrick.isaac_backend import RuntimeOperation
from rocobrick.moveit_contract import TcpPoseTarget
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.tcp_contract import EXPECTED_RUNTIME_MODEL_SHA256
from rocobrick.transit_adapter import SessionTcpMotionAdapter


class RecordingMotionSession:
    """Capture one robot-scoped dispatch and reject any second attempt."""

    def __init__(self, record: Mapping[str, object]) -> None:
        """Use the supplied runtime operation record as the single reply."""
        self.record = record
        self.calls: list[tuple[str, RuntimeOperation]] = []

    def run_motion_operation(
        self, robot_id: str, operation: RuntimeOperation
    ) -> Mapping[str, object]:
        """Record the dispatch and return its scripted runtime result.

        Returns:
            The scripted response for the single dispatch.
        """
        self.calls.append((robot_id, operation))
        if len(self.calls) != 1:
            raise AssertionError("TRANSIT dispatched more than once")
        return self.record


@pytest.fixture
def target() -> TcpPoseTarget:
    """Return a calibrated TCP target with the strict default tolerances."""
    return TcpPoseTarget(
        frame_id="base_link",
        position=(0.28, 0.318, 0.186),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        runtime_model_sha256=EXPECTED_RUNTIME_MODEL_SHA256,
        source_sha256="a" * 64,
    )


@pytest.fixture
def operation() -> RobotOperation:
    """Return one formal unladen TRANSIT operation."""
    return RobotOperation(
        operation_id="transit-to-pick",
        source_task_id="task-pick",
        skill="TRANSIT",
        executor_robot_id="piper_0",
        participants=("piper_0",),
        role_bindings={"executor": "piper_0"},
        target=None,
        parameters={},
        preconditions=(),
        effects=(),
        resources=("robot:piper_0",),
        evidence_requirements=(),
    )


def test_final_fk_failure_is_returned_without_retry(
    operation: RobotOperation, target: TcpPoseTarget
) -> None:
    """A terminal-safe FK miss preserves the first result without retrying."""
    execution = {
        "goal_accepted": True,
        "action_status": "SUCCEEDED",
        "is_success": False,
        "terminal": True,
        "safe_to_release": True,
        "command_messages": 15,
        "final_errors": {
            "position_error_m": 0.0068,
            "orientation_error_rad": 0.0354,
        },
        "details": {"settled_at_rest": True},
    }
    record = {
        "result": {
            "passed": False,
            "error": "the TCP_POSE goal did not pass the execution gates",
            "execution": execution,
            "evidence": {"robot_id": "piper_0"},
        }
    }
    session = RecordingMotionSession(record)

    result = SessionTcpMotionAdapter(session, "piper_0").execute(operation, target)

    assert len(session.calls) == 1
    assert session.calls[0][0] == "piper_0"
    assert session.calls[0][1].request_id == operation.operation_id
    assert result.passed is False
    assert result.terminal is True
    assert result.safe_to_release is True
    assert result.motion_sent is True
    assert result.reason == "the TCP_POSE goal did not pass the execution gates"
    assert result.evidence["execution"] == execution
    assert result.evidence["runtime"] == {"robot_id": "piper_0"}
    assert result.evidence["final_pose_feedback_confirmed"] is False
    assert "retry_count" not in result.evidence


def test_confirmed_fk_succeeds_with_one_dispatch(
    operation: RobotOperation, target: TcpPoseTarget
) -> None:
    """A confirmed pose remains successful with a single session call."""
    session = RecordingMotionSession({
        "result": {
            "passed": True,
            "error": None,
            "execution": {
                "goal_accepted": True,
                "action_status": "SUCCEEDED",
                "is_success": True,
                "terminal": True,
                "safe_to_release": True,
                "command_messages": 12,
                "final_errors": {
                    "position_error_m": 0.005,
                    "orientation_error_rad": 0.02,
                },
                "details": {"settled_at_rest": True},
            },
            "evidence": {},
        }
    })

    result = SessionTcpMotionAdapter(session, "piper_0").execute(operation, target)

    assert len(session.calls) == 1
    assert result.passed is True
    assert result.terminal is True
    assert result.safe_to_release is True
    assert result.motion_sent is True
    assert result.reason is None
    assert result.evidence["final_pose_feedback_confirmed"] is True
