"""Strict prepare/start/cancel lifecycle for coordinated arm trajectories.

The protocol is transport-free. A ROS adapter supplies one participant per arm;
this module owns the all-or-nothing admission and fail-closed evidence rules.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
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
                        operation, prepared, cancel_evidence,
                        f"participant {trajectory.robot_id} did not prepare",
                    )
            started: dict[str, Mapping[str, object]] = {}
            # All participants are already prepared. Start their action goals
            # concurrently against the same simulation-time barrier; iterating
            # start() calls serially would create an unbounded start skew.
            with ThreadPoolExecutor(max_workers=len(expected)) as pool:
                futures = {
                    robot_id: pool.submit(
                        self._participants[robot_id].start,
                        start_at_sim_time_s=start_at,
                    )
                    for robot_id in expected
                }
                for robot_id in expected:
                    evidence = futures[robot_id].result()
                    started[robot_id] = dict(evidence)
                    if evidence.get("started") is not True:
                        merged = {**prepared, **started}
                        return self._failed_start(
                            operation, merged, cancel_evidence,
                            f"participant {robot_id} did not start",
                        )
            terminal = all(
                evidence.get("terminal") is True and evidence.get("safe") is True
                for evidence in started.values()
            )
            if not terminal:
                return self._failed_start(
                    operation, {**prepared, **started}, cancel_evidence,
                    "coordinated participants did not reach terminal-safe evidence",
                )
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
