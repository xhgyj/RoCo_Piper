"""Controller phase-clock evidence cross-checked with stamped Isaac joints.

Goal acceptance/wall time is never a synchronization sample. The monitor
checks the controller's executing phase, its actual joint feedback and a new
stamped physics-state sample against the admitted joint-linear trajectory.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence

from rocobrick.coordinated_moveit import CoordinatedRobotTrajectory
from rocobrick.execution.prepared_motion import SynchronizationPolicy
from rocobrick.skills.handover_profile import finite


def _seconds(raw: object) -> float:
    sec, nanosec = getattr(raw, "sec", None), getattr(raw, "nanosec", None)
    if (
        type(sec) is not int
        or type(nanosec) is not int
        or not 0 <= nanosec < 1_000_000_000
    ):
        raise ValueError("controller feedback timestamp is unavailable or invalid")
    return float(sec) + nanosec * 1e-9


def _positions(raw: object, length: int) -> tuple[float, ...]:
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != length
    ):
        raise ValueError("controller feedback positions are incomplete")
    return tuple(finite(v, "controller joint") for v in raw)


class SynchronizedFeedbackMonitor:
    """Thread-safe bounded feedback accumulator for one prepared arm goal."""

    def __init__(
        self,
        trajectory: CoordinatedRobotTrajectory,
        policy: SynchronizationPolicy,
        start_at_sim_time_s: float,
    ) -> None:
        """Bind exact joints, phase clock and immutable tracking policy."""
        self.trajectory, self.policy = trajectory, policy
        self.start = finite(start_at_sim_time_s, "execution barrier", positive=True)
        self._lock = threading.Lock()
        self._count = 0
        self._last_count = -1
        self._last_stamp: float | None = None
        self._first_stamp: float | None = None
        self._start_estimates: list[float] = []
        self._tracking = 0.0
        self._max_gap = 0.0
        self._error: str | None = None

    def expected_positions(self, elapsed: float) -> tuple[float, ...]:
        """Return the admitted joint-linear positions at this exact phase time."""
        points = self.trajectory.points
        if elapsed < -1e-6 or elapsed > points[-1].time_from_start_s + 1e-6:
            raise ValueError("feedback phase time is outside the admitted trajectory")
        elapsed = min(max(elapsed, 0.0), points[-1].time_from_start_s)
        for first, last in zip(points, points[1:]):
            if elapsed <= last.time_from_start_s:
                fraction = (elapsed - first.time_from_start_s) / (
                    last.time_from_start_s - first.time_from_start_s
                )
                return tuple(
                    a + (b - a) * fraction
                    for a, b in zip(first.positions, last.positions)
                )
        return points[-1].positions

    def consume(self, message: object, measured: Mapping[str, object] | None) -> None:
        """Validate executing feedback and a fresh physics sample, recording errors."""
        with self._lock:
            if self._error is not None:
                return
            try:
                feedback = getattr(message, "feedback", message)
                names = tuple(getattr(feedback, "joint_names"))
                if names != self.trajectory.joint_names:
                    raise ValueError(
                        "synchronized feedback robot/joint binding mismatch"
                    )
                stamp = _seconds(getattr(getattr(feedback, "header"), "stamp"))
                desired = getattr(feedback, "desired")
                elapsed = _seconds(getattr(desired, "time_from_start"))
                # Pre-barrier reports of the waiting start state are harmless;
                # an advancing phase before the barrier is not.
                if stamp < self.start:
                    if elapsed > 1e-6:
                        raise ValueError(
                            "controller started its phase before the barrier"
                        )
                    return
                estimate = stamp - elapsed
                if abs(estimate - self.start) > self.policy.max_start_skew_s:
                    raise ValueError(
                        "controller execution phase missed the common barrier"
                    )
                duration = self.trajectory.points[-1].time_from_start_s
                wanted = self.expected_positions(min(elapsed, duration))
                desired_positions = _positions(
                    getattr(desired, "positions"), len(names)
                )
                actual = _positions(
                    getattr(getattr(feedback, "actual"), "positions"), len(names)
                )
                error = max(abs(a - b) for a, b in zip(desired_positions, wanted))
                if error > self.policy.max_tracking_error_rad:
                    raise ValueError(
                        "controller interpolation differs from certified linear path"
                    )
                error = max(error, max(abs(a - b) for a, b in zip(actual, wanted)))
                if error > self.policy.max_tracking_error_rad:
                    raise ValueError(
                        "controller actual tracking exceeded prepared motion budget"
                    )
                if not isinstance(measured, Mapping):
                    raise ValueError("stamped physics joint feedback is unavailable")
                count = measured.get("sample_count")
                physical_stamp = finite(
                    measured.get("sim_time_s"), "physics joint timestamp"
                )
                receipt_age = finite(measured.get("age_s"), "physics feedback age")
                if (
                    receipt_age < 0
                    or receipt_age > self.policy.max_feedback_gap_s
                    or (abs(physical_stamp - stamp) > self.policy.max_feedback_gap_s)
                ):
                    raise ValueError(
                        "physics/controller feedback is stale or mixed-clock"
                    )
                if type(count) is not int or count < 1:
                    raise ValueError("physics feedback sample identity is invalid")
                if physical_stamp < self.start or count == self._last_count:
                    return
                if (
                    count < self._last_count
                    or self._last_stamp is not None
                    and physical_stamp <= self._last_stamp
                ):
                    raise ValueError("physics joint feedback did not advance")
                physical_elapsed = physical_stamp - self.start
                if physical_elapsed > self.trajectory.points[-1].time_from_start_s:
                    # Completion/settling reports cannot establish new motion
                    # tracking evidence; final arrival has its own fresh gate.
                    return
                joints = measured.get("joints")
                if not isinstance(joints, Mapping):
                    raise ValueError("physics feedback omits measured joints")
                physical = tuple(
                    finite(joints.get(f"joint{i}"), "physical arm joint")
                    for i in range(1, 7)
                )
                physical_wanted = self.expected_positions(physical_elapsed)
                error = max(
                    error, max(abs(a - b) for a, b in zip(physical, physical_wanted))
                )
                if error > self.policy.max_tracking_error_rad:
                    raise ValueError(
                        "measured arm tracking exceeded prepared motion budget"
                    )
                gap = physical_stamp - (
                    self.start if self._last_stamp is None else self._last_stamp
                )
                if gap > self.policy.max_feedback_gap_s:
                    raise ValueError("executing physics feedback gap exceeded budget")
                self._max_gap = max(gap, self._max_gap)
                self._tracking = max(error, self._tracking)
                self._last_count, self._last_stamp = count, physical_stamp
                if self._first_stamp is None:
                    self._first_stamp = physical_stamp
                self._count += 1
                # Accumulate constant-size extrema, not an unbounded callback log.
                self._start_estimates = (
                    [estimate]
                    if not self._start_estimates
                    else [
                        min(self._start_estimates[0], estimate),
                        max(self._start_estimates[-1], estimate),
                    ]
                )
            except (AttributeError, TypeError, ValueError, IndexError) as error:
                self._error = f"{type(error).__name__}: {error}"

    @property
    def failed(self) -> bool:
        """Return whether an executing feedback sample violated a safety gate."""
        with self._lock:
            return self._error is not None

    def evidence(self) -> Mapping[str, object]:
        """Return measured tracking and controller phase-clock evidence.

        Returns:
            Explicit feedback validity, not an action-acceptance timestamp or
            an inferred physical custody/first-contact proof.
        """
        with self._lock:
            interval = list(self._start_estimates)
            good = (
                self._error is None and self._count >= self.policy.min_feedback_samples
            )
            if self._last_stamp is None or (
                self.start
                + self.trajectory.points[-1].time_from_start_s
                - self._last_stamp
                > self.policy.max_feedback_gap_s
            ):
                good = False
            if interval and interval[-1] - interval[0] > self.policy.max_start_skew_s:
                good = False
            return {
                "synchronization_feedback_confirmed": good,
                "feedback_source": "controller_phase_and_stamped_isaac_joints",
                "controller_start_interval_s": interval,
                "controller_start_sim_time_s": interval[0] if interval else None,
                "first_physics_sample_sim_time_s": self._first_stamp,
                "last_physics_sample_sim_time_s": self._last_stamp,
                "feedback_samples": self._count,
                "max_tracking_error_rad": self._tracking,
                "max_feedback_gap_s": self._max_gap,
                "feedback_error": self._error
                or (None if good else "insufficient executing feedback"),
            }
