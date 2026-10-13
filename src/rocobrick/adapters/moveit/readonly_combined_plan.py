"""One combined pose/joint query without scene writes or controller access."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Protocol

from rocobrick.coordinated_ros_runtime import CombinedMoveItPlanningError
from rocobrick.coordinated_session import (
    CoordinatedSessionRequest,
    CoordinatedSessionResult,
)
from rocobrick.dual_arm_coordinator import DualArmPlanOnlyCoordinator
from rocobrick.skills.handover_profile import digest, finite
from rocobrick.skills.handover_region import validate_region

READONLY_PLAN_SCHEMA = "rocobrick-readonly-combined-plan/v1"


def _validate_pose_target(robot: str, target, snapshot) -> None:
    """Admit calibrated TCP goals only, with bounded handover tolerances."""
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
    if not isinstance(target, Mapping) or set(target) != required:
        raise ValueError("read-only pose query requires a complete calibrated TCP goal")
    source = snapshot.get("tcp_provenance")
    if not isinstance(source, Mapping) or any(
        target[key] != source.get(key)
        for key in ("runtime_model_sha256", "source_sha256")
    ):
        raise ValueError("read-only pose goal calibration differs from snapshot")
    for key in ("runtime_model_sha256", "source_sha256"):
        digest(target[key], key)
    if target["frame_id"] != f"{robot}_base_link" or (
        target["tip_link"] != f"{robot}_rocobrick_tcp"
    ):
        raise ValueError("read-only pose goal frame/tip belongs to another robot")
    for key, length in (("position", 3), ("orientation_xyzw", 4)):
        values = target[key]
        if not isinstance(values, (tuple, list)) or len(values) != length:
            raise ValueError("read-only pose goal vectors are incomplete")
        for value in values:
            finite(value, key)
    if abs(sum(v * v for v in target["orientation_xyzw"]) - 1.0) > 1e-6:
        raise ValueError("read-only pose goal quaternion is not normalized")
    for key, maximum in (
        ("position_tolerance_m", 0.001),
        ("orientation_tolerance_rad", 0.015),
    ):
        if finite(target[key], key, positive=True) > maximum:
            raise ValueError("read-only pose goal tolerance exceeds handover policy")


class ReadOnlyCombinedSession(Protocol):
    """Explicit read/plan-only surface; no apply, execute or controller method."""

    robot_ids: tuple[str, ...]

    def coordinated_readiness(
        self, request: CoordinatedSessionRequest
    ) -> Mapping[str, object]:
        """Read both robots' current readiness and common clock."""
        ...

    @property
    def unified_planning_scene_version(self) -> int:
        """Read complete world/attachment fingerprint without modifying it."""
        ...

    def coordinated_plan_runtime(self) -> object:
        """Return the one combined plan-only MoveGroup client."""
        ...


def perform_readonly_combined_plan(
    session: ReadOnlyCombinedSession,
    request: CoordinatedSessionRequest,
) -> CoordinatedSessionResult:
    """Query an already bound shared scene; certify native geometry separately.

    Returns:
        A no-motion trajectory result with unchanged world/attachment version.
    """
    rejected_trajectory: dict[str, object] = {}
    try:
        admitted = json.dumps(request.serialize(), sort_keys=True, allow_nan=False)
        if (
            request.robots != session.robot_ids
            or not request.plan_only
            or (
                set(request.participants) != {"piper_0", "piper_1"}
                or request.planning_group != "dual_arm"
                or request.operation
            )
        ):
            raise ValueError(
                "read-only HANDOVER query requires both full arms in plan-only mode"
            )
        binding = {
            robot: tuple(f"{robot}_joint{i}" for i in range(1, 7))
            for robot in request.participants
        }
        pose_goals = all(set(t) == {"tcp_pose"} for t in request.targets.values())
        region_goals = all(set(t) == {"tcp_region"} for t in request.targets.values())
        if region_goals:
            regions = [t["tcp_region"] for t in request.targets.values()]
            if (
                not all(isinstance(r, Mapping) for r in regions)
                or regions[0] != regions[1]
            ):
                raise ValueError("both arms must bind the same coupled region")
            validate_region(regions[0], request.scene_snapshot)
        for robot, names in binding.items():
            target = request.targets[robot]
            for name in names:
                finite(request.start_state.get(name), "complete planning start")
            if pose_goals:
                _validate_pose_target(robot, target["tcp_pose"], request.scene_snapshot)
                continue
            if region_goals:
                continue
            if set(target) != {"joint_positions", "joint_goal_tolerance_rad"} or (
                not isinstance(target["joint_positions"], Mapping)
                or set(target["joint_positions"]) != set(names)
            ):
                raise ValueError("read-only query accepts exact six-joint goals only")
            tolerance = finite(
                target["joint_goal_tolerance_rad"], "goal tolerance", positive=True
            )
            if tolerance > 0.01:
                raise ValueError("read-only goal tolerance exceeds model policy")
            for name in names:
                finite(target["joint_positions"][name], "paired joint goal")
        if request.scene_snapshot.get("scene_digest") != request.scene_digest:
            raise ValueError("read-only query snapshot is not bound")
        version = session.unified_planning_scene_version
        if type(version) is not int or version != request.scene_version:
            raise ValueError("read-only query world/attachment version is stale")
        coordinator = DualArmPlanOnlyCoordinator(binding)
        readiness = coordinator.admit_readiness(
            session.coordinated_readiness(request), request
        )
        raw = session.coordinated_plan_runtime().plan(
            request, timeout_s=request.timeout_s
        )
        if not isinstance(raw, Mapping) or raw.get("status") != "SUCCEEDED":
            raise ValueError("combined plan-only query did not succeed")
        # Preserve the exact returned trajectory in any later rejection, so a
        # start/goal mismatch is diagnosable per joint instead of summarized.
        rejected_trajectory.update(
            {
                "joint_names": [str(name) for name in raw.get("joint_names", ())],
                "points": [
                    {
                        "time_from_start_s": float(point["time_from_start_s"]),
                        "positions": [float(value) for value in point["positions"]],
                    }
                    for point in raw.get("points", ())
                ],
            }
        )
        operation = coordinator.admit_plan(
            request, joint_names=raw["joint_names"], points=raw["points"]
        )
        start_deviations: dict[str, float] = {}
        goal_deviations: dict[str, float] = {}
        for path in operation.trajectories:
            target = request.targets[path.robot_id]
            for name, initial, final in zip(
                path.joint_names, path.points[0].positions, path.points[-1].positions
            ):
                start = abs(initial - request.start_state[name])
                if start > 0.002:
                    start_deviations[name] = start
                if not pose_goals and not region_goals:
                    goal = abs(final - target["joint_positions"][name])
                    if goal > target["joint_goal_tolerance_rad"]:
                        goal_deviations[name] = goal
        if start_deviations or goal_deviations:
            raise ValueError(
                "read-only path does not match captured start/paired goal: "
                f"start_deviations={start_deviations} "
                f"goal_deviations={goal_deviations} "
                "(start_tolerance=0.002)"
            )
        if json.dumps(request.serialize(), sort_keys=True, allow_nan=False) != admitted:
            raise ValueError("read-only request mutated during planning")
        final_version = session.unified_planning_scene_version
        if type(final_version) is not int or final_version != version:
            raise ValueError("read-only world/attachment changed during planning")
        return CoordinatedSessionResult(
            request.coordination_id,
            request.robots,
            asdict(readiness),
            {
                "joint_names": list(raw["joint_names"]),
                "points": [dict(p) for p in raw["points"]],
                "coordination_id": operation.coordination_id,
                "scene_digest": request.scene_digest,
                "planning_scene_version": version,
            },
            {
                "plan_only": True,
                "read_only": True,
                "physical_execution": False,
                "terminal": True,
                "safe_to_release": False,
                "goal_mode": "tcp_region"
                if region_goals
                else ("tcp_pose" if pose_goals else "joint_positions"),
                "native_fk_validation_required": pose_goals or region_goals,
            },
            True,
            None,
            request.participants,
        )
    except (
        AssertionError,
        AttributeError,
        KeyError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
    ) as error:
        # Generated ROS message setters raise AssertionError on scalar types.
        # Preserve its no-motion/read-only envelope and original diagnostic;
        # the outer worker's generic failure would drop the read-only marker.
        failure_details = (
            {"moveit_error_code": error.moveit_error_code}
            if isinstance(error, CombinedMoveItPlanningError)
            else {}
        )
        return CoordinatedSessionResult(
            request.coordination_id,
            request.robots,
            {},
            dict(rejected_trajectory),
            {
                "plan_only": True,
                "read_only": True,
                "physical_execution": False,
                "terminal": not isinstance(error, TimeoutError),
                "safe_to_release": False,
                **failure_details,
            },
            False,
            str(error),
            request.participants,
        )
