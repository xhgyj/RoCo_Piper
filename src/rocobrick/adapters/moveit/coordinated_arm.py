"""One arm's real ROS participant boundary for coordinated free-space motion.

The prepare/start/cancel lifecycle matches
:class:`rocobrick.coordinated_execution.CoordinatedParticipant`:

* ``prepare`` admits the split trajectory (exact joint ownership, monotonic
  times, measured start match) and builds the controller goal WITHOUT
  sending any motion;
* ``start`` sends the prepared FollowJointTrajectory goal, waits for goal
  acceptance and the terminal action result, and proves arrival from the
  measured final joint state — never from the command alone;
* ``cancel`` fans out to the runtime's owned cancel path and settles late
  acceptances before claiming terminal evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from rocobrick.adapters.moveit.synchronized_feedback import SynchronizedFeedbackMonitor
from rocobrick.coordinated_moveit import CoordinatedRobotTrajectory
from rocobrick.execution.prepared_motion import SynchronizationPolicy
from rocobrick.moveit_contract import ARM_JOINT_NAMES

ACCEPT_TIMEOUT_S = 5.0
RESULT_MARGIN_S = 10.0
START_MATCH_TOLERANCE_RAD = 0.01
ARRIVAL_TOLERANCE_RAD = 0.015


class CoordinatedArmParticipant:
    """Wrap one robot runtime as a coordinated free-space participant."""

    def __init__(
        self,
        runtime: object,
        *,
        policy: SynchronizationPolicy | None = None,
    ) -> None:
        """Bind the participant to its robot-scoped arm controller runtime."""
        self._runtime = runtime
        self._goal: object | None = None
        self._trajectory: CoordinatedRobotTrajectory | None = None
        self._policy = policy
        self._monitor: SynchronizedFeedbackMonitor | None = None
        self._prepared_barrier: float | None = None
        self._send_uncertain = False

    def prepare(
        self,
        trajectory: CoordinatedRobotTrajectory,
        *,
        start_at_sim_time_s: float,
    ) -> Mapping[str, object]:
        """Admit one split trajectory and build its goal without motion.

        Returns:
            Admission evidence; ``prepared`` is True only when the trajectory
            names exactly this robot's arm joints, times are monotonic from
            zero, and the measured live state matches the planned start.
        """
        runtime = self._runtime
        self._goal = None
        self._trajectory = None
        self._monitor = None
        self._prepared_barrier = None
        arm_names = tuple(getattr(runtime, "arm_joint_names", ()) or ())
        if tuple(trajectory.joint_names) != arm_names or arm_names != tuple(
            f"{trajectory.robot_id}_joint{i}" for i in range(1, 7)
        ):
            return {
                "prepared": False,
                "error": ("trajectory joints do not exactly match the participant arm"),
            }
        times = tuple(point.time_from_start_s for point in trajectory.points)
        if (
            not times
            or times[0] != 0.0
            or any(right <= left for left, right in zip(times, times[1:]))
        ):
            return {"prepared": False, "error": "trajectory times are invalid"}
        measured = runtime.latest_isaac_state()
        if not measured:
            return {
                "prepared": False,
                "error": "no measured live state for the participant start",
            }
        start_point = trajectory.points[0]
        drift = max(
            abs(measured[canonical] - start_point.positions[index])
            for index, canonical in enumerate(ARM_JOINT_NAMES)
        )
        tolerance = (
            START_MATCH_TOLERANCE_RAD
            if self._policy is None
            else min(START_MATCH_TOLERANCE_RAD, self._policy.max_tracking_error_rad)
        )
        if drift > tolerance:
            return {
                "prepared": False,
                "error": (
                    "measured start drifted from the planned combined start "
                    f"(max {drift:.4f} rad)"
                ),
                "start_drift_rad": drift,
            }
        factories = runtime._resolve_factories()
        goal = factories.follow_joint_trajectory_goal_factory()
        goal.trajectory.joint_names = list(trajectory.joint_names)
        if self._policy is not None:
            clocks = runtime.clock_values()
            sample = runtime.latest_stamped_isaac_state()
            if (
                not clocks
                or not sample
                or sample["age_s"] > self._policy.max_feedback_gap_s
            ):
                return {
                    "prepared": False,
                    "error": "fresh stamped physics state/clock unavailable",
                }
            if not math.isfinite(start_at_sim_time_s) or (
                start_at_sim_time_s
                <= clocks[-1] * 1e-9 + self._policy.max_feedback_gap_s
            ):
                return {
                    "prepared": False,
                    "error": "shared barrier is not safely in the future",
                }
            stamp = goal.trajectory.header.stamp
            stamp.sec = math.floor(start_at_sim_time_s)
            stamp.nanosec = round((start_at_sim_time_s - stamp.sec) * 1e9)
            if stamp.nanosec == 1_000_000_000:
                stamp.sec += 1
                stamp.nanosec = 0
            self._monitor = SynchronizedFeedbackMonitor(
                trajectory, self._policy, start_at_sim_time_s
            )
        for point in trajectory.points:
            item = factories.joint_trajectory_point_factory()
            item.positions = list(point.positions)
            item.time_from_start = runtime._duration_message(
                float(point.time_from_start_s), factories
            )
            goal.trajectory.points.append(item)
        for name in trajectory.joint_names:
            tolerance = factories.joint_tolerance_factory()
            tolerance.name = name
            tolerance.position = (
                0.02 if self._policy is None else self._policy.max_tracking_error_rad
            )
            tolerance.velocity = 0.05
            goal.goal_tolerance.append(tolerance)
        goal.goal_time_tolerance = runtime._duration_message(5.0, factories)
        if self._policy is not None:
            for name in trajectory.joint_names:
                item = factories.joint_tolerance_factory()
                item.name = name
                item.position = self._policy.max_tracking_error_rad
                goal.path_tolerance.append(item)
        self._goal = goal
        self._trajectory = trajectory
        self._prepared_barrier = float(start_at_sim_time_s)
        return {
            "prepared": True,
            "waypoints": len(trajectory.points),
            "duration_s": times[-1],
            "start_at_sim_time_s": float(start_at_sim_time_s),
            "start_drift_rad": drift,
            "motion_sent": False,
        }

    def start(self, *, start_at_sim_time_s: float) -> Mapping[str, object]:
        """Send the prepared goal and prove arrival from measured state.

        Returns:
            Lifecycle evidence with ``started`` from real goal acceptance,
            ``terminal`` from the action result, and ``safe``/``arrival``
            confirmed against the measured final joint state.
        """
        runtime = self._runtime
        if self._goal is None or self._trajectory is None:
            return {"started": False, "error": "participant was not prepared"}
        if self._prepared_barrier != start_at_sim_time_s:
            return {
                "started": False,
                "motion_sent": False,
                "error": "prepared barrier changed",
            }
        factories = runtime._resolve_factories()
        client = runtime._arm_client
        if client is None or not runtime._action_ready(client):
            return {
                "started": False,
                "error": "arm controller action is not ready",
            }
        duration = self._trajectory.points[-1].time_from_start_s
        deadline = runtime._clock_fn() + ACCEPT_TIMEOUT_S + duration + RESULT_MARGIN_S
        send_commands = getattr(runtime, "_command_count", 0)
        if self._policy is not None:
            clocks = runtime.clock_values()
            if not clocks or clocks[-1] * 1e-9 >= start_at_sim_time_s:
                return {
                    "started": False,
                    "motion_sent": False,
                    "error": "prepared barrier expired",
                }
        self._send_uncertain = True
        if self._monitor is None:
            future = client.send_goal_async(self._goal)
        else:
            future = client.send_goal_async(
                self._goal, feedback_callback=self._feedback
            )
        runtime._active_goal_future = future
        self._send_uncertain = False
        if not runtime._wait_future(future, runtime._clock_fn() + ACCEPT_TIMEOUT_S):
            lifecycle = runtime.cancel_active()
            return {
                "started": False,
                "terminal": lifecycle.get("terminal") is True,
                "safe": lifecycle.get("safe") is True,
                "error": "goal acceptance timed out",
                "cancel": lifecycle,
            }
        handle = future.result()
        runtime._active_goal_future = None
        if handle is None or not getattr(handle, "accepted", False):
            return {
                "started": False,
                "terminal": True,
                "safe": True,
                "error": "arm controller rejected the prepared goal",
            }
        runtime._active_handle = handle
        accepted_wall_s = runtime._clock_fn()
        result_future = handle.get_result_async()
        runtime._active_result_future = result_future
        if self._monitor is not None:
            while (
                not result_future.done()
                and runtime._clock_fn() < deadline
                and not self._monitor.failed
            ):
                runtime._spin_fn()
                runtime._sleep_fn(0.01)
            finished = result_future.done() and not self._monitor.failed
        else:
            finished = runtime._wait_future(result_future, deadline)
        if not finished:
            lifecycle = runtime.cancel_active()
            return {
                "started": True,
                "terminal": lifecycle.get("terminal") is True,
                "safe": lifecycle.get("safe") is True,
                "error": "action result timed out or synchronized feedback failed",
                "cancel": lifecycle,
                **(dict(self._monitor.evidence()) if self._monitor is not None else {}),
            }
        wrapper = result_future.result()
        runtime._active_handle = None
        runtime._active_result_future = None
        status = int(getattr(wrapper, "status"))
        goal_status = factories.goal_status_type()
        succeeded = status == int(goal_status.STATUS_SUCCEEDED)
        canceled = status == int(goal_status.STATUS_CANCELED)
        end_point = self._trajectory.points[-1]
        # Fresh measured state after the terminal result decides arrival; a
        # succeeded action alone never proves the physical endpoint.
        result_states = runtime._isaac_state_count
        while (
            runtime._isaac_state_count <= result_states
            and runtime._clock_fn() < deadline
        ):
            runtime._spin_fn()
            runtime._sleep_fn(0.01)
        final = runtime.latest_isaac_state()
        arrival = False
        final_error: float | None = None
        fresh = runtime._isaac_state_count > result_states
        if self._policy is not None:
            sample = runtime.latest_stamped_isaac_state()
            fresh = bool(
                fresh
                and sample
                and sample["sample_count"] == runtime._isaac_state_count
                and sample["age_s"] <= self._policy.max_feedback_gap_s
                and sample["sim_time_s"] >= start_at_sim_time_s + duration
            )
        if final and fresh:
            final_error = max(
                abs(final[canonical] - end_point.positions[index])
                for index, canonical in enumerate(ARM_JOINT_NAMES)
            )
            arrival_tolerance = (
                ARRIVAL_TOLERANCE_RAD
                if self._policy is None
                else min(ARRIVAL_TOLERANCE_RAD, self._policy.max_tracking_error_rad)
            )
            arrival = final_error <= arrival_tolerance
        terminal = succeeded or canceled
        safe = succeeded and arrival
        if canceled and final_error is not None and arrival:
            safe = True
        return {
            "started": True,
            "accepted_wall_s": accepted_wall_s,
            "start_at_sim_time_s": float(start_at_sim_time_s),
            "final_action_status": status,
            "terminal": terminal,
            "safe": safe,
            "arrival_confirmed": arrival,
            "final_joint_error_rad": final_error,
            "final_joint_state": dict(final) if final else None,
            "command_messages": getattr(runtime, "_command_count", 0) - send_commands,
            "waypoints": len(self._trajectory.points),
            "motion_sent": True,
            **(dict(self._monitor.evidence()) if self._monitor is not None else {}),
        }

    def _feedback(self, message: object) -> None:
        """Cross-check controller phase feedback with newly stamped physics state."""
        if self._monitor is not None:
            try:
                sample = self._runtime.latest_stamped_isaac_state()
            except (AttributeError, RuntimeError, ValueError):
                sample = None
            self._monitor.consume(message, sample)

    def cancel(self) -> Mapping[str, object]:
        """Cancel the owned goal and keep ownership until terminal evidence.

        Returns:
            Terminal-safe evidence, or uncertainty for unowned goal dispatch.
        """
        lifecycle = self._runtime.cancel_active()
        final = self._runtime.latest_isaac_state()
        evidence: dict[str, object] = dict(lifecycle)
        evidence["attempted"] = True
        if self._send_uncertain:
            evidence.update(
                terminal=False,
                safe=False,
                error="goal dispatch raised before its acceptance handle was owned",
            )
        if final and self._trajectory is not None:
            evidence["final_joint_state"] = dict(final)
        return evidence


__all__ = ["CoordinatedArmParticipant"]
