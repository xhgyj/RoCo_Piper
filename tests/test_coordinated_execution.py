"""Tests for coordinated prepare/start/cancel safety."""

# Test doubles intentionally use compact protocol methods.
# ruff: noqa: D101, D102, D103, D105

from dataclasses import dataclass
from threading import Barrier

import pytest

from rocobrick.coordinated_execution import CoordinatedExecutionCoordinator
from rocobrick.coordinated_moveit import (
    CoordinatedOperation,
    CoordinatedRobotTrajectory,
    CoordinatedTrajectoryPoint,
)


@dataclass
class Participant:
    prepare_ok: bool = True
    start_ok: bool = True
    cancel_safe: bool = True
    calls: list[str] | None = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    def prepare(self, trajectory, *, start_at_sim_time_s):
        self.calls.append("prepare")
        return {"prepared": self.prepare_ok}

    def start(self, *, start_at_sim_time_s):
        self.calls.append("start")
        return {
            "started": self.start_ok,
            "terminal": self.start_ok,
            "safe": self.start_ok,
        }

    def cancel(self):
        self.calls.append("cancel")
        return {"attempted": True, "terminal": True, "safe": self.cancel_safe}


def operation() -> CoordinatedOperation:
    return CoordinatedOperation(
        coordination_id="coord-1",
        scene_digest="scene-1",
        participants=("piper_0", "piper_1"),
        trajectories=tuple(
            CoordinatedRobotTrajectory(
                robot_id,
                (f"{robot_id}_joint1",),
                (
                    CoordinatedTrajectoryPoint(0.0, (0.0,)),
                    CoordinatedTrajectoryPoint(1.0, (0.1,)),
                ),
            )
            for robot_id in ("piper_0", "piper_1")
        ),
        start_at_sim_time_s=10.0,
    )


def test_start_calls_participants_concurrently_at_shared_barrier():
    """A participant cannot finish start before the peer enters start."""
    barrier = Barrier(2)
    entered = []

    @dataclass
    class BarrierParticipant(Participant):
        def start(self, *, start_at_sim_time_s):
            entered.append(start_at_sim_time_s)
            barrier.wait(timeout=1.0)
            self.calls.append("start")
            return {"started": True, "terminal": True, "safe": True}

    first = BarrierParticipant()
    second = BarrierParticipant()
    result = CoordinatedExecutionCoordinator(
        {"piper_0": first, "piper_1": second}
    ).execute(operation())
    assert result.started is True
    assert entered == [10.0, 10.0]


def test_prepare_failure_never_starts_and_cancels_all():
    first = Participant()
    second = Participant(prepare_ok=False)
    result = CoordinatedExecutionCoordinator(
        {"piper_0": first, "piper_1": second}
    ).execute(operation())
    assert result.started is False
    assert first.calls == ["prepare", "cancel"]
    assert second.calls == ["prepare", "cancel"]
    assert result.started is False
    assert result.safe_to_release is True


def test_start_failure_fanouts_cancel_and_records_terminal_safe():
    first = Participant()
    second = Participant(start_ok=False)
    result = CoordinatedExecutionCoordinator(
        {"piper_0": first, "piper_1": second}
    ).execute(operation())
    assert first.calls[:2] == ["prepare", "start"]
    assert second.calls[:2] == ["prepare", "start"]
    assert first.calls[2:] and set(first.calls[2:]) == {"cancel"}
    assert second.calls[2:] and set(second.calls[2:]) == {"cancel"}
    assert result.started is False
    assert result.terminal is True
    assert result.safe_to_release is True
    assert result.error is not None


def test_cancel_failure_keeps_operation_unsafe():
    first = Participant(start_ok=False)
    second = Participant(cancel_safe=False)
    result = CoordinatedExecutionCoordinator(
        {"piper_0": first, "piper_1": second}
    ).execute(operation())
    assert result.terminal is False
    assert result.safe_to_release is False
    assert "terminal-safe" in (result.error or "")


def test_participant_identity_must_match():
    with pytest.raises(ValueError, match="identities"):
        CoordinatedExecutionCoordinator({"piper_0": Participant()}).execute(operation())
