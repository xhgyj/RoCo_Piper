"""Strict prepare/start/cancel lifecycle for coordinated arm trajectories.

The protocol is transport-free. A ROS adapter supplies one participant per arm;
this module owns the all-or-nothing admission and fail-closed evidence rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Protocol

from rocobrick.coordinated_moveit import (
    CoordinatedOperation,
    CoordinatedRobotTrajectory,
)


class CoordinatedParticipant(Protocol):
    """One arm controller participant in a coordinated operation."""

    def prepare(
        self,
        trajectory: CoordinatedRobotTrajectory,
        *,
        start_at_sim_time_s: float,
    ) -> Mapping[str, object]:
        """Prepare this participant's trajectory for the shared start barrier."""
        ...

    def start(self, *, start_at_sim_time_s: float) -> Mapping[str, object]:
        """Start this participant at the shared simulation-time barrier."""
        ...

    def cancel(self) -> Mapping[str, object]:
        """Cancel this participant and return terminal safety evidence."""
        ...


@dataclass(frozen=True)
class CoordinatedLifecycleResult:
    """Aggregate evidence for one prepare/start/cancel lifecycle."""

    coordination_id: str
    prepared: bool
    started: bool
    terminal: bool
    safe_to_release: bool
    participants: Mapping[str, Mapping[str, object]]
    cancel_evidence: Mapping[str, Mapping[str, object]]
    error: str | None = None

    def serialize(self) -> dict[str, object]:
        """Return JSON-safe aggregate lifecycle evidence."""
        return {
            "coordination_id": self.coordination_id,
            "prepared": self.prepared,
            "started": self.started,
            "terminal": self.terminal,
            "safe_to_release": self.safe_to_release,
            "participants": {
                str(robot): dict(evidence)
                for robot, evidence in self.participants.items()
            },
            "cancel_evidence": {
                str(robot): dict(evidence)
                for robot, evidence in self.cancel_evidence.items()
            },
            "error": self.error,
        }


class CoordinatedExecutionCoordinator:
    """Run the strict all-participant lifecycle for one admitted operation."""

    def __init__(self, participants: Mapping[str, CoordinatedParticipant]) -> None:
        """Bind exact participant identities to one coordinated lifecycle."""
        if not participants:
            raise ValueError("coordinated participants must not be empty")
        self._participants = dict(participants)

    def execute(
        self,
        operation: CoordinatedOperation,
        *,
        start_at_sim_time_s: float | None = None,
        require_synchronization_feedback: bool = False,
        before_start: Callable[[], None] | None = None,
        require_completion: bool = True,
    ) -> CoordinatedLifecycleResult:
        """Prepare all, start all at one barrier, and fail closed on any error.

        Returns:
            Aggregate terminal and cancellation evidence.
        """
        expected = tuple(operation.participants)
        if set(self._participants) != set(expected):
            raise ValueError(
                "coordinated participant identities do not match operation"
            )
        start_at = (
            operation.start_at_sim_time_s
            if start_at_sim_time_s is None
            else float(start_at_sim_time_s)
        )
        if (
            type(require_synchronization_feedback) is not bool
            or not math.isfinite(start_at)
            or start_at < 0
        ):
            raise ValueError("coordinated barrier/synchronization policy is invalid")
        prepared: dict[str, Mapping[str, object]] = {}
        cancel_evidence: dict[str, Mapping[str, object]] = {}
        try:
            for trajectory in operation.trajectories:
                evidence = self._participants[trajectory.robot_id].prepare(
                    trajectory, start_at_sim_time_s=start_at
                )
                prepared[trajectory.robot_id] = dict(evidence)
                if evidence.get("prepared") is not True:
                    return self._failed_prepare(
                        operation,
                        prepared,
                        cancel_evidence,
                        f"participant {trajectory.robot_id} did not prepare",
                    )
            started: dict[str, Mapping[str, object]] = {}
            if before_start is not None:
                before_start()
            failure: str | None = None
            cancel_sent = False
            # All participants are already prepared. Start their action goals
            # concurrently against the same simulation-time barrier; iterating
            # start() calls serially would create an unbounded start skew.
            with ThreadPoolExecutor(max_workers=len(expected)) as pool:
                futures = {
                    pool.submit(
                        self._participants[robot_id].start,
                        start_at_sim_time_s=start_at,
                    ): robot_id
                    for robot_id in expected
                }
                for future in as_completed(futures):
                    robot_id = futures[future]
                    try:
                        evidence = future.result()
                    except Exception as error:  # noqa: BLE001 - cancel owned peers
                        evidence = {
                            "started": False,
                            "terminal": False,
                            "safe": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    started[robot_id] = dict(evidence)
                    if evidence.get("started") is not True or (
                        require_completion
                        and (
                            evidence.get("terminal") is not True
                            or evidence.get("safe") is not True
                        )
                        or require_synchronization_feedback
                        and evidence.get("synchronization_feedback_confirmed")
                        is not True
                    ):
                        failure = f"participant {robot_id} did not complete safely"
                        if not cancel_sent and any(not f.done() for f in futures):
                            # Abort immediately even when the failing robot is
                            # second in participant order. Keep waiting for all
                            # action lanes before a FINAL cancellation proof:
                            # an early idle cancel cannot settle a late send.
                            self.cancel(operation)
                            cancel_sent = True
            if failure is not None:
                if require_completion:
                    return self._failed_start(
                        operation, {**prepared, **started}, cancel_evidence, failure
                    )
                failure = None
            if not require_completion:
                return CoordinatedLifecycleResult(
                    coordination_id=operation.coordination_id,
                    prepared=True,
                    started=all(
                        evidence.get("started") is True
                        for evidence in started.values()
                    ),
                    terminal=all(
                        evidence.get("terminal") is True
                        for evidence in started.values()
                    ),
                    safe_to_release=all(
                        evidence.get("safe") is True
                        for evidence in started.values()
                    ),
                    participants={**prepared, **started},
                    cancel_evidence=cancel_evidence,
                    error=None,
                )
            terminal = all(
                evidence.get("terminal") is True and evidence.get("safe") is True
                for evidence in started.values()
            )
            if not terminal:
                return self._failed_start(
                    operation,
                    {**prepared, **started},
                    cancel_evidence,
                    "coordinated participants did not reach terminal-safe evidence",
                )
            if require_synchronization_feedback:
                intervals = []
                for robot_id, evidence in started.items():
                    value = evidence.get("controller_start_sim_time_s")
                    interval = evidence.get("controller_start_interval_s")
                    if evidence.get(
                        "synchronization_feedback_confirmed"
                    ) is not True or (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                        or evidence.get("feedback_source")
                        != "controller_phase_and_stamped_isaac_joints"
                    ):
                        return self._failed_start(
                            operation,
                            {**prepared, **started},
                            cancel_evidence,
                            f"participant {robot_id} lacks verified "
                            "controller/physics synchronization",
                        )
                    if (
                        not isinstance(interval, (list, tuple))
                        or not 1 <= len(interval) <= 2
                        or any(
                            isinstance(v, bool)
                            or not isinstance(v, (int, float))
                            or not math.isfinite(v)
                            for v in interval
                        )
                        or interval[0] != value
                        or interval[-1] < interval[0]
                    ):
                        return self._failed_start(
                            operation,
                            {**prepared, **started},
                            cancel_evidence,
                            "synchronized start interval is invalid",
                        )
                    intervals.extend(float(v) for v in interval)
                skew = max(intervals) - min(intervals)
                if skew > operation.synchronization_tolerance_s or any(
                    abs(value - start_at) > operation.synchronization_tolerance_s
                    for value in intervals
                ):
                    return self._failed_start(
                        operation,
                        {**prepared, **started},
                        cancel_evidence,
                        "measured controller phase start skew exceeds budget",
                    )
                for robot_id, evidence in started.items():
                    started[robot_id] = {
                        **evidence,
                        "measured_controller_start_skew_s": skew,
                    }
            return CoordinatedLifecycleResult(
                coordination_id=operation.coordination_id,
                prepared=True,
                started=True,
                terminal=True,
                safe_to_release=True,
                participants={**prepared, **started},
                cancel_evidence=cancel_evidence,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed with evidence
            return self._failed_start(
                operation,
                prepared,
                cancel_evidence,
                f"{type(exc).__name__}: {exc}",
            )

    def cancel(self, operation: CoordinatedOperation) -> CoordinatedLifecycleResult:
        """Fan out cancellation and require terminal-safe evidence from all.

        Returns:
            Aggregate cancellation evidence and resource-release verdict.
        """
        if set(self._participants) != set(operation.participants):
            raise ValueError(
                "coordinated participant identities do not match operation"
            )
        evidence: dict[str, Mapping[str, object]] = {}
        errors: list[str] = []
        for robot_id in operation.participants:
            try:
                item = dict(self._participants[robot_id].cancel())
            except Exception as exc:  # noqa: BLE001 - preserve every participant
                item = {"attempted": True, "terminal": False, "safe": False}
                errors.append(f"{robot_id}: {type(exc).__name__}: {exc}")
            evidence[robot_id] = item
            if item.get("terminal") is not True or item.get("safe") is not True:
                errors.append(f"{robot_id}: cancellation is not terminal-safe")
        safe = not errors and len(evidence) == len(operation.participants)
        return CoordinatedLifecycleResult(
            coordination_id=operation.coordination_id,
            prepared=False,
            started=False,
            terminal=safe,
            safe_to_release=safe,
            participants={},
            cancel_evidence=evidence,
            error=None if safe else "; ".join(errors),
        )

    def _failed_prepare(
        self,
        operation: CoordinatedOperation,
        evidence: Mapping[str, Mapping[str, object]],
        cancel_evidence: Mapping[str, Mapping[str, object]],
        error: str,
    ) -> CoordinatedLifecycleResult:
        cancelled = self.cancel(operation)
        return CoordinatedLifecycleResult(
            coordination_id=operation.coordination_id,
            prepared=False,
            started=False,
            terminal=cancelled.terminal,
            safe_to_release=cancelled.safe_to_release,
            participants=dict(evidence),
            cancel_evidence=cancelled.cancel_evidence,
            error=error if cancelled.error is None else f"{error}; {cancelled.error}",
        )

    def _failed_start(
        self,
        operation: CoordinatedOperation,
        evidence: Mapping[str, Mapping[str, object]],
        cancel_evidence: Mapping[str, Mapping[str, object]],
        error: str,
    ) -> CoordinatedLifecycleResult:
        cancelled = self.cancel(operation)
        return CoordinatedLifecycleResult(
            coordination_id=operation.coordination_id,
            prepared=True,
            started=False,
            terminal=cancelled.terminal,
            safe_to_release=cancelled.safe_to_release,
            participants=dict(evidence),
            cancel_evidence=cancelled.cancel_evidence,
            error=error if cancelled.error is None else f"{error}; {cancelled.error}",
        )


__all__ = [
    "CoordinatedExecutionCoordinator",
    "CoordinatedLifecycleResult",
    "CoordinatedParticipant",
]
