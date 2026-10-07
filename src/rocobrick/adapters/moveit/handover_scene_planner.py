"""Native isolated HANDOVER geometry gates around one combined ROS plan.

No controller or shared-scene writer is reachable through this provider.
Phase paths use joint-linear interpolation with conservative collision and
Cartesian-tube certificates; they are planning evidence, not custody proof.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from rocobrick.adapters.moveit.handover_geometry import IsolatedHandoverGeometry
from rocobrick.coordinated_moveit import (
    CoordinatedOperation,
    CoordinatedTrajectoryPoint,
)
from rocobrick.coordinated_moveit_adapter import (
    coordinated_operation_from_joint_trajectory,
)
from rocobrick.coordinated_session import (
    CoordinatedSessionRequest,
    CoordinatedSessionResult,
)
from rocobrick.skills.handover_planning import (
    HandoverCandidate,
    HandoverPlanningRequest,
    HandoverRobotContext,
    _hash,
    calibrated_target,
)
from rocobrick.skills.handover_profile import PLANNING_CAPABILITIES, digest, finite
from rocobrick.skills.handover_region import candidate_from_endpoint, preparation_region
from rocobrick.skills.observation import Pose, _vector
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.targets import _quat_rotate_vector


class CombinedPlanRuntime(Protocol):
    """Existing single MoveGroup plan-only action, never an execution facade."""

    def plan(
        self, request: CoordinatedSessionRequest, *, timeout_s: float
    ) -> Mapping[str, object]:
        """Return the one combined ROS trajectory without applying a scene diff."""
        ...


class ReadOnlySessionTransport(Protocol):
    """Host facade of the same persistent session's read-only mailbox lane."""

    def plan_readonly_combined(
        self, request: CoordinatedSessionRequest
    ) -> CoordinatedSessionResult:
        """Return a bound combined plan without scene synchronization."""
        ...


class ReadOnlyPlanRejectedError(ValueError):
    """A failed, recorded MoveGroup query; never a controller failure."""


class ReadOnlyPlannerInternalError(ValueError):
    """An explicitly identified integration failure, not a numeric plan status."""


class SessionCombinedPlanRuntime:
    """Connect native host geometry to the container's existing MoveGroup."""

    def __init__(self, session: ReadOnlySessionTransport) -> None:
        """Bind the existing session, without creating ROS or another host."""
        self.session = session

    def plan(
        self, request: CoordinatedSessionRequest, *, timeout_s: float
    ) -> Mapping[str, object]:
        """Query the read-only lane instead of the scene-synchronizing pre-phase.

        Returns:
            Combined joint trajectory for separate native path certification.
        """
        if not request.plan_only or timeout_s != request.timeout_s:
            raise ValueError("read-only transport budget or plan-only binding mismatch")
        result = self.session.plan_readonly_combined(request)
        if result.coordination_id != request.coordination_id or (
            result.robots != request.robots
            or result.participants != request.participants
            or result.lifecycle.get("physical_execution") is not False
            or result.lifecycle.get("read_only") is not True
        ):
            raise ValueError("read-only combined result identity/capability mismatch")
        if not result.passed:
            # MoveIt FAILURE=99999 also results from ordinary OMPL multi-plan
            # goal sampling exhaustion. A numeric status alone cannot prove
            # an internal integration failure; retain it in worker evidence
            # and reject this candidate within the existing bounded search.
            raise ReadOnlyPlanRejectedError(
                f"read-only combined planning failed: {result.error}"
            )
        if all(set(t) == {"tcp_pose"} for t in request.targets.values()) and (
            result.lifecycle.get("goal_mode") != "tcp_pose"
            or result.lifecycle.get("native_fk_validation_required") is not True
        ):
            raise ValueError(
                "read-only pose result omitted native FK admission boundary"
            )
        if all(set(t) == {"tcp_region"} for t in request.targets.values()) and (
            result.lifecycle.get("goal_mode") != "tcp_region"
            or result.lifecycle.get("native_fk_validation_required") is not True
        ):
            raise ValueError(
                "read-only region result omitted native FK admission boundary"
            )
        return {"status": "SUCCEEDED", **dict(result.trajectory)}


class ReadOnlyCombinedHandoverPlanner:
    """Use the existing combined MoveGroup runtime without scene synchronization."""

    def __init__(
        self,
        runtime: CombinedPlanRuntime,
        *,
        clock: Callable[[], float] = time.monotonic,
        read_static_scene: Callable[[HandoverPlanningRequest], Mapping[str, object]]
        | None = None,
    ):
        """Bind a plan-only transport, not execute_free_space_motion."""
        self.runtime, self.clock = runtime, clock
        self.read_static_scene = read_static_scene
        self.last_planning_failure: str | None = None
        self._region_query_count = 0

    def plan_poses(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        start: Mapping[str, float],
        *,
        deadline_s: float,
        region: Mapping[str, object] | None = None,
    ) -> CoordinatedOperation | None:
        """Let MoveGroup choose both arm joint goals from calibrated TCP poses.

        Returns:
            The returned path, not a native IK proposal or execution permission.
        """
        return self.plan(
            request, candidate, None, start, deadline_s=deadline_s, region=region
        )

    def plan(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        goals: Mapping[str, float] | None,
        start: Mapping[str, float],
        *,
        deadline_s: float,
        region: Mapping[str, object] | None = None,
    ) -> CoordinatedOperation | None:
        """Send one combined pose/joint plan-only request on the common axis.

        Returns:
            The combined trajectory, or None before dispatch if time expired.
        """
        if self.clock() >= deadline_s:
            return None
        # Geometry/candidate binding stays immutable across the bounded search.
        # Live production supplies a fresh observation ONLY after the existing
        # full static-state guard has compared it with that captured geometry.
        shared = (
            self.read_static_scene(request)
            if self.read_static_scene is not None
            else None
        )
        snapshot = (
            shared["scene_snapshot"] if shared is not None else request.scene_snapshot
        )
        version = (
            shared["planning_scene_version"]
            if shared is not None
            else request.scene_version
        )
        remaining = deadline_s - self.clock()
        if remaining <= 0:
            return None
        participants = (request.donor.robot_id, request.receiver.robot_id)
        binding = {
            robot: tuple(f"{robot}_joint{i}" for i in range(1, 7))
            for robot in participants
        }
        goal_tolerance = request.profile.joint_hold_tolerance_rad
        pose_goals = goals is None
        # Leave time for endpoint, contact-phase and swept-path validation;
        # a single unsuccessful pose query cannot consume the whole search.
        if pose_goals:
            remaining = min(remaining, 10.0)
        targets = (
            {
                request.donor.robot_id: calibrated_target(
                    candidate.donor_goal, request.donor, request.profile
                ),
                request.receiver.robot_id: calibrated_target(
                    candidate.receiver_precontact, request.receiver, request.profile
                ),
            }
            if pose_goals
            else {
                robot: {
                    "joint_positions": {n: goals[n] for n in names},
                    "joint_goal_tolerance_rad": goal_tolerance,
                }
                for robot, names in binding.items()
            }
        )
        if region is not None:
            if goals is not None:
                raise ValueError("region query cannot carry native joint goals")
            targets = {r: {"tcp_region": dict(region)} for r in participants}
            self._region_query_count += 1
        query_id = (
            _hash({"region": region, "query": self._region_query_count})
            if region is not None
            else candidate.candidate_id
        )
        envelope = CoordinatedSessionRequest(
            coordination_id=query_id,
            robots=request.session_robots,
            scene_digest=snapshot["scene_digest"],
            planning_group="dual_arm",
            plan_only=True,
            timeout_s=remaining,
            targets=targets,
            scene_snapshot=snapshot,
            scene_version=version,
            start_state=start,
            participants=participants,
        )
        try:
            raw = self.runtime.plan(envelope, timeout_s=remaining)
        except ReadOnlyPlanRejectedError as error:
            if not pose_goals:
                raise
            self.last_planning_failure = str(error)
            return None
        self.last_planning_failure = None
        names, points = raw.get("joint_names"), raw.get("points")
        expected_names = {n for arm in binding.values() for n in arm}
        if (
            raw.get("status") != "SUCCEEDED"
            or not isinstance(names, Sequence)
            or (isinstance(names, (str, bytes)) or set(names) != expected_names)
        ):
            raise ValueError(
                "combined MoveGroup returned an invalid twelve-joint trajectory"
            )
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            raise ValueError("combined MoveGroup omitted trajectory points")
        return coordinated_operation_from_joint_trajectory(
            coordination_id=query_id,
            scene_digest=request.scene_snapshot["scene_digest"],
            participants=participants,
            joint_names=names,
            points=points,
            joint_binding=binding,
            start_at_sim_time_s=0.0,
            synchronization_tolerance_s=request.profile.max_sync_skew_s,
        )


@dataclass(frozen=True)
class HandoverGripperConfigurations:
    """Explicit calibrated finger states; no hardcoded opening conversion.

    Each map covers the owning model's complete non-arm finger joints. Closing
    includes the commanded squeeze state, not merely the nominal brick span.
    A caller must bind these values to its captured calibration and candidate.
    """

    candidate_id: str
    calibration_sha256: str
    receiver_open: Mapping[str, float]
    receiver_closed: Mapping[str, float]
    donor_open: Mapping[str, float]

    def __post_init__(self) -> None:
        """Freeze explicit states and reject booleans/nonfinite joint values."""
        from types import MappingProxyType

        digest(self.candidate_id, "gripper candidate")
        digest(self.calibration_sha256, "gripper calibration")
        for field in ("receiver_open", "receiver_closed", "donor_open"):
            raw = getattr(self, field)
            if (
                not isinstance(raw, Mapping)
                or not raw
                or any(not isinstance(n, str) or not n for n in raw)
            ):
                raise ValueError("calibrated finger state is incomplete")
            object.__setattr__(
                self,
                field,
                MappingProxyType(
                    {n: finite(v, "calibrated finger joint") for n, v in raw.items()},
                ),
            )


@dataclass(frozen=True)
class HandoverPhasePaths:
    """Admitted no-motion paths before any provisional custody or release."""

    candidate_id: str
    receiver_open: tuple[Mapping[str, float], Mapping[str, float]]
    approach: tuple[Mapping[str, float], ...]
    receiver_close: tuple[Mapping[str, float], Mapping[str, float]]
    donor_open: tuple[Mapping[str, float], Mapping[str, float]]
    separation: tuple[Mapping[str, float], ...]
    receiver_follow: tuple[Mapping[str, float], ...]
    planned_receiver_attachment: Pose

    def __post_init__(self) -> None:
        """Keep consumers from rewriting previously certified joint states."""
        from types import MappingProxyType

        digest(self.candidate_id, "phase-path candidate")
        for field in (
            "receiver_open",
            "approach",
            "receiver_close",
            "donor_open",
            "separation",
            "receiver_follow",
        ):
            values = getattr(self, field)
            if len(values) < 2:
                raise ValueError("phase paths require at least two complete states")
            object.__setattr__(
                self, field, tuple(MappingProxyType(dict(v)) for v in values)
            )


class NativeHandoverScenePlanner:
    """Full-model native geometry provider for the HANDOVER planning backend."""

    capabilities = PLANNING_CAPABILITIES

    def __init__(
        self,
        geometry: IsolatedHandoverGeometry,
        planner: ReadOnlyCombinedHandoverPlanner,
        *,
        grippers: Callable[[HandoverCandidate], HandoverGripperConfigurations],
        read_binding: Callable[[], str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Require explicit calibration and read-only live scene/model identity."""
        self.geometry, self.planner = geometry, planner
        self.grippers, self.read_binding, self.clock = grippers, read_binding, clock
        self._paths: dict[str, HandoverPhasePaths] = {}
        self._finger_bindings: dict[str, HandoverGripperConfigurations] = {}
        self._goal_bindings: dict[str, dict[str, float]] = {}
        self._pose_goals: dict[str, dict[str, float]] = {}
        self._ik_cache: dict[tuple[str, Pose], Mapping[str, float] | None] = {}
        self.phase_rejections: dict[str, int] = {}
        self.last_path_failure = None
        self._cartesian_failure = None

    def _reject_path(
        self,
        phase: str,
        failure: str | None,
        *,
        collision_pairs: tuple[tuple[str, str], ...] = (),
    ) -> bool:
        key = phase + ":" + (failure or "UNPROVED_PATH")
        self.phase_rejections[key] = self.phase_rejections.get(key, 0) + 1
        self.last_path_failure = {
            "phase": phase,
            "failure": failure,
            "clearance_pair": self.geometry.last_clearance_failure,
            "contact_region_pair": self.geometry.last_contact_region_failure,
            "collision_pairs": collision_pairs,
        }
        return False

    def _bound(self, request: HandoverPlanningRequest) -> None:
        if request.binding_sha256 != self.geometry.binding:
            raise ValueError("native planner request/model/scene binding changed")

    def _fingers(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
    ) -> HandoverGripperConfigurations:
        values = self.grippers(candidate)
        if not isinstance(values, HandoverGripperConfigurations) or (
            values.candidate_id != candidate.candidate_id
        ):
            raise ValueError("gripper calibration belongs to another candidate")
        previous = self._finger_bindings.get(candidate.candidate_id)
        if previous is not None and values != previous:
            raise ValueError("candidate gripper configuration changed during planning")
        if candidate.candidate_id not in self._finger_bindings and (
            len(self._finger_bindings) >= request.profile.max_evaluations
        ):
            raise ValueError("gripper configuration cache exceeds evaluation budget")
        for robot, fields in (
            (request.receiver.robot_id, (values.receiver_open, values.receiver_closed)),
            (request.donor.robot_id, (values.donor_open,)),
        ):
            expected = {
                n
                for n in self.geometry.model.names
                if n.startswith(robot + "_")
                and n not in {f"{robot}_joint{i}" for i in range(1, 7)}
            }
            if any(set(v) != expected for v in fields):
                raise ValueError(
                    "calibrated finger state must cover exact exported gripper joints"
                )
        self._finger_bindings[candidate.candidate_id] = values
        return values

    def solve_ik(
        self,
        request: HandoverPlanningRequest,
        context: HandoverRobotContext,
        target: Mapping[str, object],
        *,
        deadline_s: float,
    ) -> Mapping[str, float] | None:
        """Use measured-state native IK on this robot's calibrated TCP.

        Returns:
            A bounded six-joint solution, or None after the finite budget.
        """
        self._bound(request)
        if context not in (request.donor, request.receiver):
            raise ValueError("native IK context does not match captured calibration")
        raw = target.get("tcp_pose")
        if set(target) != {"tcp_pose"} or not isinstance(raw, Mapping):
            raise ValueError("calibrated TCP target is required")
        local = Pose(
            _vector(raw.get("position"), 3, "IK position"),
            _vector(raw.get("orientation_xyzw"), 4, "IK quaternion"),
        )
        if abs(sum(v * v for v in local.orientation_xyzw) - 1) > 1e-6:
            raise ValueError("native IK requires a normalized quaternion")
        world = compose(
            Pose(context.base.position_xyz, context.base.orientation_xyzw), local
        )
        if target != calibrated_target(world, context, request.profile):
            # Roundoff is allowed only in coordinate conversion, never identity.
            encoded = calibrated_target(world, context, request.profile)["tcp_pose"]
            if set(raw) != set(encoded) or any(
                raw.get(k) != encoded[k]
                for k in encoded
                if k not in ("position", "orientation_xyzw")
            ):
                raise ValueError("native IK calibration/frame mismatch")
        key = (context.robot_id, world)
        if key in self._ik_cache:
            return self._ik_cache[key]
        solved = self.geometry.solve_tcp_ik(
            context.robot_id,
            world,
            request.start_state,
            deadline_s=deadline_s,
            clock=self.clock,
        )
        if self.clock() < deadline_s:
            if len(self._ik_cache) >= request.profile.max_evaluations * 2:
                raise ValueError("native IK cache exceeds captured evaluation budget")
            self._ik_cache[key] = solved
        return solved

    def validate_pair(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> bool:
        """Check the full paired state with explicitly opened receiving fingers.

        Returns:
            Native full-scene collision and unchanged-limit admission.
        """
        self._bound(request)
        fingers = self._fingers(request, candidate)
        return (
            self.clock() < deadline_s
            and self.geometry.check(
                {**request.start_state, **joints, **fingers.receiver_open},
                holder=request.donor.robot_id,
            ).valid
        )

    def _cartesian(
        self,
        request: HandoverPlanningRequest,
        robot: str,
        start: Mapping[str, float],
        first: Pose,
        last: Pose,
        *,
        deadline_s: float,
        receiver_touch: bool,
        holder: str,
        attachment: Pose | None = None,
        finger_contact: bool = False,
    ) -> tuple[Mapping[str, float], ...] | None:
        self._cartesian_failure = None
        steps = max(
            1, math.ceil(math.dist(first.position_xyz, last.position_xyz) / 0.002)
        )
        if steps > 256:
            self._cartesian_failure = "CARTESIAN_STEP_BUDGET"
            return None
        states = [dict(start)]
        previous = first
        for index in range(1, steps + 1):
            pose = Pose(
                tuple(
                    a + (b - a) * index / steps
                    for a, b in zip(first.position_xyz, last.position_xyz)
                ),
                last.orientation_xyzw,
            )
            solved = self.geometry.solve_tcp_ik(
                robot,
                pose,
                states[-1],
                deadline_s=deadline_s,
                clock=self.clock,
                attempts=1,
            )
            if solved is None:
                self._cartesian_failure = "CARTESIAN_WAYPOINT_IK"
                return None
            state = {**states[-1], **solved}
            if not self.geometry.certify_cartesian_segment(
                states[-1],
                state,
                robot=robot,
                first_pose=previous,
                last_pose=pose,
                deadline_s=deadline_s,
                clock=self.clock,
                receiver_touch=receiver_touch,
                holder=holder,
                receiver_attachment=attachment,
                finger_contact=finger_contact,
            ):
                self._cartesian_failure = self.geometry.last_certificate_failure
                return None
            states.append(state)
            previous = pose
        return tuple(states)

    def _prepare_paths(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> bool:
        self._bound(request)
        if candidate.binding_sha256 != request.binding_sha256:
            raise ValueError("phase-path candidate belongs to another scene")
        self.last_path_failure = None
        self.geometry.last_clearance_failure = None
        self.geometry.last_contact_region_failure = None
        self._fingers(request, candidate)
        if candidate.candidate_id in self._paths:
            if self._goal_bindings[candidate.candidate_id] != dict(joints):
                raise ValueError("paired goal changed after phase-path certification")
            return self.clock() < deadline_s
        fingers = self._fingers(request, candidate)
        giver, receiver = request.donor.robot_id, request.receiver.robot_id
        initial = dict(request.start_state)
        opened = {**initial, **fingers.receiver_open}
        # Establish nominal simultaneous holding geometry FIRST. A reachable
        # open-jaw precontact target is not evidence that the jaws can close
        # around the object without occupying the giver's finger volume.
        # Runtime cohold still requires the original physical pad evidence.
        precontact = {**opened, **joints}
        first_pose, contact_pose = (
            candidate.receiver_precontact,
            candidate.receiver_contact,
        )
        if candidate.candidate_id in self._pose_goals:
            if self._pose_goals[candidate.candidate_id] != dict(joints):
                raise ValueError("MoveGroup endpoint changed before phase validation")
            # Preserve MoveGroup's actual joint solution and tool direction.
            # Translate the short constant-orientation insertion from its real
            # precontact FK; prove its object-relative goal before any sweep.
            first_pose = self.geometry.tcp_pose(receiver, precontact)
            direction = _quat_rotate_vector(first_pose.orientation_xyzw, (0, 0, 1))
            distance = math.dist(
                candidate.receiver_precontact.position_xyz,
                candidate.receiver_contact.position_xyz,
            )
            contact_pose = Pose(
                tuple(
                    p + distance * d for p, d in zip(first_pose.position_xyz, direction)
                ),
                first_pose.orientation_xyzw,
            )
            object_pose = compose(
                self.geometry.tcp_pose(giver, precontact),
                inverse(candidate.object_to_donor_tcp),
            )
            expected = compose(
                compose(object_pose, inverse(candidate.object_goal)),
                candidate.receiver_contact,
            )
            position_error, rotation_error = pose_error(contact_pose, expected)
            if (
                position_error > request.profile.relative_position_tolerance_m
                or rotation_error > request.profile.relative_orientation_tolerance_rad
            ):
                return self._reject_path(
                    "moveit_contact_binding", "ACTUAL_RELATIVE_CONTACT_GOAL_MISMATCH"
                )
        contact_ik = self.geometry.solve_tcp_ik(
            receiver,
            contact_pose,
            precontact,
            deadline_s=deadline_s,
            clock=self.clock,
            attempts=1,
        )
        if contact_ik is None:
            return self._reject_path("cohold_endpoint", "COHOLD_TARGET_IK")
        cohold = {**precontact, **contact_ik, **fingers.receiver_closed}
        cohold_verdict = self.geometry.check(
            cohold,
            holder=giver,
            receiver_touch=True,
            finger_contact=self.geometry.finger_contact_policy is not None,
        )
        if not cohold_verdict.valid:
            return self._reject_path(
                "cohold_endpoint",
                cohold_verdict.reason,
                collision_pairs=cohold_verdict.collision_pairs,
            )
        if not self.geometry.certify_segment(
            initial, opened, holder=giver, deadline_s=deadline_s, clock=self.clock
        ):
            return self._reject_path(
                "receiver_open", self.geometry.last_certificate_failure
            )
        approach = self._cartesian(
            request,
            receiver,
            precontact,
            first_pose,
            contact_pose,
            deadline_s=deadline_s,
            receiver_touch=True,
            holder=giver,
            finger_contact=self.geometry.finger_contact_policy is not None,
        )
        if approach is None:
            return self._reject_path("approach", self._cartesian_failure)
        closed = {**approach[-1], **fingers.receiver_closed}
        released = {**closed, **fingers.donor_open}
        for phase, first, last in (
            ("receiver_close", approach[-1], closed),
            ("donor_open", closed, released),
        ):
            if not self.geometry.certify_segment(
                first,
                last,
                holder=giver,
                receiver_touch=True,
                deadline_s=deadline_s,
                clock=self.clock,
                finger_contact=self.geometry.finger_contact_policy is not None,
            ):
                return self._reject_path(phase, self.geometry.last_certificate_failure)
        # This is a PLANNED attachment for geometry only. Runtime must replace
        # it with verified cohold/follow evidence before final ledger adoption.
        receiver_pose = self.geometry.tcp_pose(receiver, closed)
        object_pose = compose(
            self.geometry.tcp_pose(giver, closed),
            inverse(candidate.object_to_donor_tcp),
        )
        attachment = compose(inverse(receiver_pose), object_pose)
        giver_pose = self.geometry.tcp_pose(giver, released)
        direction = _quat_rotate_vector(giver_pose.orientation_xyzw, (0, 0, -1))
        separation_goal = Pose(
            tuple(
                a + request.profile.precontact_distance_m * d
                for a, d in zip(giver_pose.position_xyz, direction)
            ),
            giver_pose.orientation_xyzw,
        )
        separation = self._cartesian(
            request,
            giver,
            released,
            giver_pose,
            separation_goal,
            deadline_s=deadline_s,
            receiver_touch=False,
            holder=receiver,
            attachment=attachment,
            finger_contact=self.geometry.finger_contact_policy is not None,
        )
        if separation is None:
            return self._reject_path("separation", self._cartesian_failure)
        follow_goal = Pose(
            (
                receiver_pose.position_xyz[0],
                receiver_pose.position_xyz[1],
                receiver_pose.position_xyz[2] + request.profile.verification_distance_m,
            ),
            receiver_pose.orientation_xyzw,
        )
        follow = self._cartesian(
            request,
            receiver,
            separation[-1],
            receiver_pose,
            follow_goal,
            deadline_s=deadline_s,
            receiver_touch=False,
            holder=receiver,
            attachment=attachment,
        )
        if follow is None:
            return self._reject_path("receiver_follow", self._cartesian_failure)
        if len(self._paths) >= request.profile.max_evaluations:
            raise ValueError(
                "native phase-path cache exceeds request evaluation budget"
            )
        self._paths[candidate.candidate_id] = HandoverPhasePaths(
            candidate.candidate_id,
            (initial, opened),
            approach,
            (approach[-1], closed),
            (closed, released),
            separation,
            follow,
            attachment,
        )
        self._goal_bindings[candidate.candidate_id] = dict(joints)
        return True

    def validate_gripper_sweeps(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> bool:
        """Certify opening/closing full sweeps, including both robot bodies.

        Returns:
            Whether the native phase paths have all been certified.
        """
        return self._prepare_paths(request, candidate, joints, deadline_s=deadline_s)

    def validate_approach(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> bool:
        """Require the cached whole-path Cartesian and collision proof.

        Returns:
            Whether the native phase paths have all been certified.
        """
        return self._prepare_paths(request, candidate, joints, deadline_s=deadline_s)

    def validate_separation(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> bool:
        """Require proved separation and receiver verification motion paths.

        Returns:
            Whether the native phase paths have all been certified.
        """
        return self._prepare_paths(request, candidate, joints, deadline_s=deadline_s)

    def plan_combined(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> CoordinatedOperation | None:
        """Plan once for both arms; independent plans are never concatenated.

        Returns:
            One combined ROS plan, or None if the scene/path admission fails.
        """
        self._bound(request)
        if candidate.candidate_id not in self._paths or not self.scene_unchanged(
            request
        ):
            return None
        if not self._prepare_paths(request, candidate, joints, deadline_s=deadline_s):
            return None
        fingers = self._fingers(request, candidate)
        plan = self.planner.plan(
            request,
            candidate,
            joints,
            {**request.start_state, **fingers.receiver_open},
            deadline_s=deadline_s,
        )
        if plan is None:
            return None
        # MoveGroup goal tolerance is not an exact phase boundary. Retain its
        # path and add explicit start/end bridges; continuous certification
        # below checks these bridges too, rather than rounding a waypoint.
        trajectories = []
        for trajectory in plan.trajectories:
            points = [
                CoordinatedTrajectoryPoint(
                    0.0, tuple(request.start_state[n] for n in trajectory.joint_names)
                )
            ]
            points.extend(
                CoordinatedTrajectoryPoint(p.time_from_start_s + 0.2, p.positions)
                for p in trajectory.points
            )
            points.append(
                CoordinatedTrajectoryPoint(
                    points[-1].time_from_start_s + 0.2,
                    tuple(joints[n] for n in trajectory.joint_names),
                )
            )
            trajectories.append(replace(trajectory, points=tuple(points)))
        return replace(plan, trajectories=tuple(trajectories))

    def plan_combined_poses(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        *,
        deadline_s: float,
    ) -> CoordinatedOperation | None:
        """Query MoveGroup BEFORE native IK/contact gates; verify returned FK.

        Returns:
            A structurally bound path with its original endpoint joints, or None.
            Full geometry and contact-phase proof are still mandatory afterward.
        """
        self._bound(request)
        self.last_path_failure = None
        if self.clock() >= deadline_s or not self.scene_unchanged(request):
            return None
        fingers = self._fingers(request, candidate)
        plan = self.planner.plan_poses(
            request,
            candidate,
            {**request.start_state, **fingers.receiver_open},
            deadline_s=deadline_s,
        )
        if plan is None:
            return None
        for trajectory in plan.trajectories:
            if any(
                abs(value - request.start_state[name])
                > request.profile.joint_hold_tolerance_rad
                for name, value in zip(
                    trajectory.joint_names, trajectory.points[0].positions
                )
            ):
                self._reject_path("moveit_start", "CAPTURED_START_MISMATCH")
                return None
        goals = {
            n: v
            for t in plan.trajectories
            for n, v in zip(t.joint_names, t.points[-1].positions)
        }
        state = {**request.start_state, **goals, **fingers.receiver_open}
        for context, target in (
            (request.donor, candidate.donor_goal),
            (request.receiver, candidate.receiver_precontact),
        ):
            position, rotation = pose_error(
                self.geometry.tcp_pose(context.robot_id, state), target
            )
            if position > request.profile.position_tolerance_m or (
                rotation > request.profile.orientation_tolerance_rad
            ):
                self._reject_path(
                    "moveit_endpoint", f"{context.robot_id}_TCP_GOAL_MISMATCH"
                )
                return None
        if len(self._pose_goals) >= request.profile.max_evaluations:
            raise ValueError("MoveGroup endpoint cache exceeds evaluation budget")
        self._pose_goals[candidate.candidate_id] = dict(goals)
        # Bridge only the measured START to the returned path. Never round,
        # clip or replace MoveGroup's final joint solution with native IK.
        return replace(
            plan,
            trajectories=tuple(
                replace(
                    t,
                    points=(
                        CoordinatedTrajectoryPoint(
                            0.0, tuple(request.start_state[n] for n in t.joint_names)
                        ),
                        *(
                            replace(p, time_from_start_s=p.time_from_start_s + 0.2)
                            for p in t.points
                        ),
                    ),
                )
                for t in plan.trajectories
            ),
        )

    def plan_region(self, request: HandoverPlanningRequest, *, deadline_s: float):
        """Ask MoveIt for a coupled region; bind contact to its unmodified endpoint.

        No host IK, point grid, or extra-lift goal is used for preparation.
        Short contact-phase IK/collision certificates remain separate gates.

        Returns:
            The FK-bound candidate and combined path, or None on rejection.
        """
        self._bound(request)
        self.last_path_failure = None
        if self.clock() >= deadline_s or not self.scene_unchanged(request):
            return None
        region = preparation_region(request)
        observed = request.observation(request.donor.robot_id)
        direction = _quat_rotate_vector(
            observed.object_pose.orientation_xyzw, (0, 0, 1)
        )
        center = compose(
            observed.object_pose, Pose(tuple(region["center"]), (0, 0, 0, 1))
        )
        receiver = Pose(
            tuple(
                c
                - d * (region["precontact_distance_m"] + region["contact_clearance_m"])
                for c, d in zip(center.position_xyz, direction)
            ),
            observed.object_pose.orientation_xyzw,
        )
        # This is only a finger-calibration/identity seed, never a MoveIt target.
        seed = candidate_from_endpoint(request, observed.tcp_pose, receiver)
        fingers = self._fingers(request, seed)
        plan = self.planner.plan_poses(
            request,
            seed,
            {**request.start_state, **fingers.receiver_open},
            deadline_s=deadline_s,
            region=region,
        )
        if plan is None:
            return None
        goals = {
            n: v
            for t in plan.trajectories
            for n, v in zip(t.joint_names, t.points[-1].positions)
        }
        state = {**request.start_state, **goals, **fingers.receiver_open}
        try:
            candidate = candidate_from_endpoint(
                request,
                self.geometry.tcp_pose(request.donor.robot_id, state),
                self.geometry.tcp_pose(request.receiver.robot_id, state),
            )
        except ValueError as error:
            self._reject_path("moveit_region_endpoint", str(error))
            return None
        if any(
            abs(v - request.start_state[n]) > request.profile.joint_hold_tolerance_rad
            for t in plan.trajectories
            for n, v in zip(t.joint_names, t.points[0].positions)
        ):
            self._reject_path("moveit_start", "CAPTURED_START_MISMATCH")
            return None
        if len(self._pose_goals) >= request.profile.max_evaluations:
            raise ValueError("MoveGroup endpoint cache exceeds evaluation budget")
        self._pose_goals[candidate.candidate_id] = dict(goals)
        plan = replace(
            plan,
            coordination_id=candidate.candidate_id,
            trajectories=tuple(
                replace(
                    t,
                    points=(
                        CoordinatedTrajectoryPoint(
                            0.0, tuple(request.start_state[n] for n in t.joint_names)
                        ),
                        *(
                            replace(p, time_from_start_s=p.time_from_start_s + 0.2)
                            for p in t.points
                        ),
                    ),
                )
                for t in plan.trajectories
            ),
        )
        return candidate, plan

    def validate_continuous_path(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        plan: CoordinatedOperation,
        *,
        deadline_s: float,
    ) -> bool:
        """Certify every joint-linear segment of the actual combined ROS plan.

        Returns:
            True only after conservative full-path clearance certification.
        """
        self._bound(request)
        if plan.coordination_id != candidate.candidate_id or (
            plan.scene_digest != request.scene_snapshot["scene_digest"]
        ):
            raise ValueError("native path proof received a foreign trajectory")
        by_robot = {t.robot_id: t for t in plan.trajectories}
        participants = (request.donor.robot_id, request.receiver.robot_id)
        if plan.participants != participants or set(by_robot) != set(participants):
            raise ValueError("native combined path omits a participant")
        grid = tuple(p.time_from_start_s for p in by_robot[participants[0]].points)
        if any(
            tuple(p.time_from_start_s for p in t.points) != grid
            or set(t.joint_names) != {f"{r}_joint{i}" for i in range(1, 7)}
            for r, t in by_robot.items()
        ):
            raise ValueError(
                "native combined path requires both six-joint arms and one time grid"
            )
        fingers = self._fingers(request, candidate)
        states = []
        for index in range(len(grid)):
            state = {**request.start_state, **fingers.receiver_open}
            for t in by_robot.values():
                state.update(zip(t.joint_names, t.points[index].positions))
            states.append(state)
        return all(
            self.geometry.certify_segment(
                a,
                b,
                holder=request.donor.robot_id,
                deadline_s=deadline_s,
                clock=self.clock,
            )
            for a, b in zip(states, states[1:])
        )

    def scene_unchanged(self, request: HandoverPlanningRequest) -> bool:
        """Require captured model/mesh hashes and exact read-back request binding.

        Returns:
            Whether the live read-only identity matches the captured request.
        """
        self._bound(request)
        self.geometry.files.validate()
        return self.read_binding() == request.binding_sha256

    def phase_paths(self, candidate: HandoverCandidate) -> HandoverPhasePaths:
        """Return previously certified paths, never inferred physical ownership.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return self._paths[candidate.candidate_id]
