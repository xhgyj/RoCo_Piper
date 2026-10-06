"""ROS transport for one combined MoveIt plan-only request."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from rocobrick.coordinated_session import CoordinatedSessionRequest
from rocobrick.ros_moveit_runtime import (
    ALLOWED_PLANNING_TIME_S,
    NUM_PLANNING_ATTEMPTS,
    RosMoveItFactories,
    _terminal_status_name,
)


class CoordinatedMoveItPlanRuntime:
    """Build and admit one combined ``dual_arm`` MoveGroup plan."""

    def __init__(
        self,
        node: object,
        *,
        joint_binding: Mapping[str, Sequence[str]],
        action_name: str = "/dual_arm/move_action",
        factories: RosMoveItFactories | None = None,
        clock: Callable[[], float],
        spin: Callable[[], None],
        sleep: Callable[[float], None],
        wait_step_s: float = 0.05,
    ) -> None:
        """Bind the combined action client and exact prefixed joint ownership."""
        if not joint_binding or set(joint_binding) != {"piper_0", "piper_1"}:
            raise ValueError("coordinated joint binding must cover both Piper arms")
        self.node = node
        self.joint_binding = {
            str(robot): tuple(str(joint) for joint in joints)
            for robot, joints in joint_binding.items()
        }
        self.action_name = str(action_name)
        self._factories = factories
        self._clock = clock
        self._spin = spin
        self._sleep = sleep
        self._wait_step_s = float(wait_step_s)
        self._client: object | None = None
        self._goal_handle: object | None = None
        self._result_future: object | None = None
        self._acceptance_future: object | None = None

    def register(self) -> None:
        """Create the single combined MoveGroup action client."""
        if self._client is not None:
            return
        factories = self._resolve_factories()
        self._client = factories.action_client_factory(
            self.node, factories.move_group_type(), self.action_name
        )

    def ready(self) -> bool:
        """Return action-server readiness without sending a goal."""
        if self._client is None:
            return False
        try:
            return bool(self._client.wait_for_server(timeout_sec=0.0))  # type: ignore[attr-defined]
        except Exception:
            return False

    def plan(
        self,
        request: CoordinatedSessionRequest,
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, object]:
        """Send one plan-only goal and return raw combined trajectory evidence.

        Returns:
            Combined joint names, waypoints, and the successful action status.
        """
        if request.planning_group != "dual_arm":
            raise ValueError("combined runtime requires planning_group='dual_arm'")
        # MoveGroup is the sole combined planner. Even physical pre-phase
        # execution uses a plan-only goal here; our coordinated participants
        # own the two controller goals, synchronization barrier and cancel.
        self.register()
        if self._client is None or not self.ready():
            raise RuntimeError(
                f"combined MoveGroup action is not ready: {self.action_name}"
            )
        factories = self._resolve_factories()
        goal = self._build_goal(request, factories)
        deadline = self._clock() + float(timeout_s or request.timeout_s)
        future = self._client.send_goal_async(goal)  # type: ignore[attr-defined]
        self._acceptance_future = future
        if not self._wait(future, deadline):
            self.cancel()
            raise TimeoutError("combined MoveGroup goal acceptance timed out")
        handle = future.result()
        if handle is None or not getattr(handle, "accepted", False):
            raise RuntimeError("combined MoveGroup rejected the plan-only goal")
        self._goal_handle = handle
        result_future = handle.get_result_async()
        self._result_future = result_future
        if not self._wait(result_future, deadline):
            self.cancel()
            raise TimeoutError("combined MoveGroup plan timed out")
        wrapper = result_future.result()
        self._acceptance_future = None
        self._goal_handle = None
        self._result_future = None
        action_result = wrapper.result
        status = _terminal_status_name(wrapper.status, factories.goal_status_type())
        error_code = int(getattr(action_result.error_code, "val"))
        trajectory = action_result.planned_trajectory.joint_trajectory
        joint_names = tuple(str(name) for name in trajectory.joint_names)
        points = tuple(
            {
                "time_from_start_s": float(point.time_from_start.sec)
                + float(point.time_from_start.nanosec) * 1e-9,
                "positions": tuple(float(value) for value in point.positions),
            }
            for point in trajectory.points
        )
        if status != "SUCCEEDED" or error_code != 1:
            raise RuntimeError(
                "combined MoveGroup plan failed: "
                f"status={status} error_code={error_code}"
            )
        return {"joint_names": joint_names, "points": points, "status": status}

    def cancel(self) -> Mapping[str, object]:
        """Cancel the combined planning goal and require terminal evidence.

        Returns:
            Cancellation acknowledgement and terminal-safe evidence.
        """
        info: dict[str, object] = {
            "attempted": False,
            "goal_accepted": None,
            "terminal": False,
            "safe": False,
        }
        handle = self._goal_handle
        deadline = self._clock() + 10.0
        if handle is None and self._acceptance_future is not None:
            if not self._wait(self._acceptance_future, deadline):
                info["error"] = "late planning acceptance remains unknown"
                return info
            handle = self._acceptance_future.result()
            if handle is not None and getattr(handle, "accepted", False):
                self._goal_handle = handle
                self._result_future = handle.get_result_async()
        if handle is None:
            return info
        info["attempted"] = True
        try:
            cancel_future = handle.cancel_goal_async()
            deadline = self._clock() + 10.0
            if not self._wait(cancel_future, deadline):
                info["error"] = "combined goal cancellation timed out"
                return info
            response = cancel_future.result()
            info["return_code"] = int(getattr(response, "return_code"))
            result_future = self._result_future
            if result_future is None or not self._wait(result_future, deadline):
                info["error"] = "combined result did not settle after cancellation"
                return info
            info["goal_accepted"] = True
            info["terminal"] = True
            info["safe"] = True
            return info
        except Exception as exc:  # noqa: BLE001 - cancellation must be recorded
            info["error"] = f"{type(exc).__name__}: {exc}"
            return info

    def shutdown(self) -> None:
        """Cancel an owned planning goal and destroy the sole combined action client."""
        self.cancel()
        if self._client is not None:
            self._client.destroy()
            self._client = None

    def _build_goal(
        self,
        request: CoordinatedSessionRequest,
        factories: RosMoveItFactories,
    ) -> object:
        goal = factories.move_group_goal_factory()
        motion = factories.motion_plan_request_factory()
        motion.group_name = request.planning_group
        motion.allowed_planning_time = ALLOWED_PLANNING_TIME_S
        motion.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        start_state = getattr(request, "start_state", {})
        if isinstance(start_state, Mapping) and start_state:
            motion.start_state.joint_state.name = list(start_state)
            motion.start_state.joint_state.position = [
                float(start_state[name]) for name in start_state
            ]
        constraints = factories.constraints_factory()
        path_constraints = factories.constraints_factory()
        source = request.scene_snapshot.get("tcp_provenance")
        for robot_id in request.participants:
            target = request.targets[robot_id]
            raw_pose = target.get("tcp_pose")
            if request.scene_snapshot and isinstance(raw_pose, Mapping):
                if not isinstance(source, Mapping) or any(
                    raw_pose.get(key) != source.get(key)
                    for key in ("runtime_model_sha256", "source_sha256")
                ):
                    raise ValueError(
                        "TCP target calibration does not match runtime provenance"
                    )
            raw_positions = target.get("joint_positions")
            expected = self.joint_binding[robot_id]
            joint_space = (
                isinstance(raw_positions, Mapping)
                and set(raw_positions) == set(expected)
            )
            if target.get("maintain_pose") is True and isinstance(raw_pose, Mapping):
                # The measured participant is held by its explicit joint target
                # and combined start_state. A full TCP path constraint is too
                # restrictive for this free-space OMPL transition; contact and
                # support phases remain outside this planner.
                pass
            if joint_space:
                for joint_name in expected:
                    value = raw_positions[joint_name]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError(f"target joint {joint_name} is not finite")
                    item = factories.joint_constraint_factory()
                    item.joint_name = joint_name
                    item.position = float(value)
                    item.tolerance_above = 0.01
                    item.tolerance_below = 0.01
                    item.weight = 1.0
                    constraints.joint_constraints.append(item)
                continue
            if isinstance(raw_pose, Mapping):
                self._append_pose_constraints(
                    constraints, raw_pose, robot_id, factories
                )
                continue
            raise ValueError(
                f"target {robot_id} must carry exact tcp_pose or joint_positions"
            )
        motion.goal_constraints.append(constraints)
        motion.path_constraints = path_constraints
        goal.request = motion
        goal.planning_options.plan_only = True
        goal.planning_options.replan = False
        return goal

    def _append_pose_constraints(
        self,
        constraints: object,
        target: Mapping[str, object],
        robot_id: str,
        factories: RosMoveItFactories,
    ) -> None:
        """Append one robot-scoped calibrated TCP pose constraint."""
        required = {
            "position",
            "orientation_xyzw",
            "tip_link",
            "frame_id",
            "runtime_model_sha256",
            "source_sha256",
            "position_tolerance_m",
            "orientation_tolerance_rad",
        }
        if set(target) != required:
            raise ValueError(f"target {robot_id} tcp_pose keys are invalid")
        position = target["position"]
        orientation = target["orientation_xyzw"]
        if (
            not isinstance(position, Sequence)
            or isinstance(position, (str, bytes))
            or len(position) != 3
            or not isinstance(orientation, Sequence)
            or isinstance(orientation, (str, bytes))
            or len(orientation) != 4
        ):
            raise ValueError(f"target {robot_id} tcp_pose vectors are invalid")
        values = tuple(position) + tuple(orientation)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"target {robot_id} tcp_pose must be finite")
        if abs(sum(float(value) ** 2 for value in orientation) - 1.0) > 1e-6:
            raise ValueError(f"target {robot_id} tcp_pose quaternion is not normalized")
        if target["frame_id"] != f"{robot_id}_base_link":
            raise ValueError(f"target {robot_id} tcp_pose frame is not robot-scoped")
        prefix = f"{robot_id}_"
        if target["tip_link"] != f"{prefix}rocobrick_tcp":
            raise ValueError(f"target {robot_id} tcp_pose tip is not calibrated")
        for digest_name in ("runtime_model_sha256", "source_sha256"):
            digest = target[digest_name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"target {robot_id} {digest_name} is invalid")
        for tolerance_name in ("position_tolerance_m", "orientation_tolerance_rad"):
            tolerance = target[tolerance_name]
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not math.isfinite(float(tolerance))
                or float(tolerance) <= 0.0
            ):
                raise ValueError(f"target {robot_id} {tolerance_name} is invalid")
        position_constraint = factories.position_constraint_factory()
        position_constraint.header.frame_id = str(target["frame_id"])
        position_constraint.link_name = str(target["tip_link"])
        sphere = factories.solid_primitive_factory()
        sphere.type = getattr(type(sphere), "SPHERE", getattr(sphere, "SPHERE", 2))
        sphere.dimensions = [float(target["position_tolerance_m"])]
        pose = factories.pose_factory()
        pose.position.x, pose.position.y, pose.position.z = (
            float(item) for item in position
        )
        pose.orientation.w = 1.0
        position_constraint.constraint_region.primitives.append(sphere)
        position_constraint.constraint_region.primitive_poses.append(pose)
        position_constraint.weight = 1.0
        constraints.position_constraints.append(position_constraint)
        orientation_constraint = factories.orientation_constraint_factory()
        orientation_constraint.header.frame_id = str(target["frame_id"])
        orientation_constraint.link_name = str(target["tip_link"])
        (
            orientation_constraint.orientation.x,
            orientation_constraint.orientation.y,
            orientation_constraint.orientation.z,
            orientation_constraint.orientation.w,
        ) = (float(item) for item in orientation)
        orientation_constraint.absolute_x_axis_tolerance = float(
            target["orientation_tolerance_rad"]
        )
        orientation_constraint.absolute_y_axis_tolerance = float(
            target["orientation_tolerance_rad"]
        )
        orientation_constraint.absolute_z_axis_tolerance = float(
            target["orientation_tolerance_rad"]
        )
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints.append(orientation_constraint)

    def _resolve_factories(self) -> RosMoveItFactories:
        if self._factories is None:
            from rocobrick.ros_moveit_runtime import RosMoveItRuntime

            self._factories = RosMoveItRuntime(self.node)._default_factories()
        return self._factories

    def _wait(self, future: object, deadline: float) -> bool:
        while not future.done():  # type: ignore[attr-defined]
            self._spin()
            if self._clock() >= deadline:
                return False
            self._sleep(self._wait_step_s)
        return True


__all__ = ["CoordinatedMoveItPlanRuntime"]
