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

from rocobrick.coordinated_moveit import CoordinatedRobotTrajectory
from rocobrick.moveit_contract import ARM_JOINT_NAMES

ACCEPT_TIMEOUT_S = 5.0
RESULT_MARGIN_S = 10.0
START_MATCH_TOLERANCE_RAD = 0.01
ARRIVAL_TOLERANCE_RAD = 0.015


class CoordinatedArmParticipant:
    """Wrap one robot runtime as a coordinated free-space participant."""

    def __init__(self, runtime: object) -> None:
        """Bind the participant to its robot-scoped arm controller runtime."""
        self._runtime = runtime
        self._goal: object | None = None
        self._trajectory: CoordinatedRobotTrajectory | None = None

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
        arm_names = tuple(getattr(runtime, "arm_joint_names", ()) or ())
        if tuple(trajectory.joint_names) != arm_names:
            return {
                "prepared": False,
                "error": (
                    "trajectory joints do not exactly match the participant arm"
                ),
            }
        times = tuple(point.time_from_start_s for point in trajectory.points)
        if not times or times[0] != 0.0 or any(
            right <= left for left, right in zip(times, times[1:])
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
        if drift > START_MATCH_TOLERANCE_RAD:
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
            tolerance.position = 0.02
            tolerance.velocity = 0.05
            goal.goal_tolerance.append(tolerance)
        goal.goal_time_tolerance = runtime._duration_message(5.0, factories)
        self._goal = goal
        self._trajectory = trajectory
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
        future = client.send_goal_async(self._goal)
        runtime._active_goal_future = future
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
        if not runtime._wait_future(result_future, deadline):
            lifecycle = runtime.cancel_active()
            return {
                "started": True,
                "terminal": lifecycle.get("terminal") is True,
                "safe": lifecycle.get("safe") is True,
                "error": "action result timed out",
                "cancel": lifecycle,
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
        if final:
            final_error = max(
                abs(final[canonical] - end_point.positions[index])
                for index, canonical in enumerate(ARM_JOINT_NAMES)
            )
            arrival = final_error <= ARRIVAL_TOLERANCE_RAD
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
        }

    def cancel(self) -> Mapping[str, object]:
        """Cancel the owned goal and keep ownership until terminal evidence."""
        lifecycle = self._runtime.cancel_active()
        final = self._runtime.latest_isaac_state()
        evidence: dict[str, object] = dict(lifecycle)
        evidence["attempted"] = True
        if final and self._trajectory is not None:
            evidence["final_joint_state"] = dict(final)
        return evidence


__all__ = ["CoordinatedArmParticipant"]
