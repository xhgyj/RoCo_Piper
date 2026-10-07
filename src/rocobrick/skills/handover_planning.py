"""Side-effect-free candidate geometry and bounded paired handover planning."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Protocol

from rocobrick.coordinated_moveit import CoordinatedOperation
from rocobrick.skills.handover_profile import (
    HandoverProfile,
    digest,
    finite,
    require_handover_capabilities,
)
from rocobrick.skills.observation import Pose, _pose, _vector, parse_pick_observation
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.targets import (
    RobotBaseFrame,
    TcpProvenance,
    _quat_rotate_vector,
    world_orientation_to_base,
    world_position_to_base,
)


def _hash(raw: object) -> str:
    return hashlib.sha256(
        json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _rotation(axis: str, angle: float) -> Pose:
    v = [0.0, 0.0, 0.0, math.cos(angle / 2)]
    v["xyz".index(axis)] = math.sin(angle / 2)
    return Pose((0.0, 0.0, 0.0), tuple(v))


@dataclass(frozen=True)
class ObjectGraspGeometry:
    """Verified object-local contact box, not an inverted world AABB."""

    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]
    source_sha256: str

    def __post_init__(self) -> None:
        """Require positive local dimensions and immutable geometry provenance."""
        lo = _vector(self.minimum_m, 3, "object local minimum")
        hi = _vector(self.maximum_m, 3, "object local maximum")
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError("object-local contact box must have positive dimensions")
        digest(self.source_sha256, "object geometry source")
        object.__setattr__(self, "minimum_m", lo)
        object.__setattr__(self, "maximum_m", hi)


@dataclass(frozen=True)
class HandoverRobotContext:
    """One explicitly identified base and calibrated TCP binding."""

    robot_id: str
    base: RobotBaseFrame
    provenance: TcpProvenance

    def __post_init__(self) -> None:
        """Reject ambiguous robot names and unnormalized base frames."""
        if not isinstance(self.robot_id, str) or not self.robot_id:
            raise ValueError("robot context identity is required")
        _pose(asdict(Pose(self.base.position_xyz, self.base.orientation_xyzw)), "base")
        digest(self.provenance.runtime_model_sha256, "runtime model")
        digest(self.provenance.source_sha256, "TCP source")


@dataclass(frozen=True)
class HandoverPlanningRequest:
    """Fresh paired starting state for one held physical object."""

    operation_id: str
    physical_instance_id: str
    session_robots: tuple[str, ...]
    donor: HandoverRobotContext
    receiver: HandoverRobotContext
    scene_snapshot: Mapping[str, object]
    scene_version: int
    start_state: Mapping[str, float]
    giver_holding_evidence: Mapping[str, object]
    object_geometry: ObjectGraspGeometry
    profile: HandoverProfile
    follow_samples: int
    co_hold_timeout_s: float
    observed_sim_time_s: float

    def __post_init__(self) -> None:
        """Validate identity, physical admission and both robots' provenance."""
        if not all(
            isinstance(v, str) and v
            for v in (
                self.operation_id,
                self.physical_instance_id,
            )
        ):
            raise ValueError("handover operation and object identity are required")
        if self.donor.robot_id == self.receiver.robot_id:
            raise ValueError("handover requires distinct robots")
        if (
            not isinstance(self.session_robots, tuple)
            or len(self.session_robots) != len(set(self.session_robots))
            or not all(isinstance(x, str) and x for x in self.session_robots)
            or not {self.donor.robot_id, self.receiver.robot_id}
            <= set(self.session_robots)
        ):
            raise ValueError("session must explicitly cover both handover robots")
        if type(self.scene_version) is not int or self.scene_version < 0:
            raise ValueError("scene version must be a nonnegative integer")
        if type(self.follow_samples) is not int or self.follow_samples < 3:
            raise ValueError("follow_samples must be an integer >= 3")
        finite(self.co_hold_timeout_s, "co_hold_timeout_s", positive=True)
        evidence = self.giver_holding_evidence
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("holding_confirmed") is not True
            or evidence.get("physical_instance_id") != self.physical_instance_id
            or evidence.get("robot_id") != self.donor.robot_id
        ):
            raise ValueError("giver holding evidence must identify the exact object")
        if not isinstance(self.start_state, Mapping):
            raise ValueError("complete measured start state is required")
        for ctx in (self.donor, self.receiver):
            if ctx.provenance.runtime_model_sha256 != self.profile.runtime_model_sha256:
                raise ValueError("handover profile does not support the runtime model")
            for i in range(1, 7):
                name = f"{ctx.robot_id}_joint{i}"
                finite(self.start_state.get(name), f"start_state.{name}")
            self.observation(ctx.robot_id)
        digest(self.scene_snapshot.get("scene_digest"), "shared scene")
        sample_time = finite(self.scene_snapshot.get("sim_time_s"), "snapshot time")
        observed_time = finite(self.observed_sim_time_s, "observed simulation time")
        if not 0 <= observed_time - sample_time <= self.profile.max_scene_age_s:
            raise ValueError("handover scene snapshot is stale or from a future clock")
        # Hashing also rejects NaN or non-JSON input before candidate generation.
        self.binding_sha256

    def observation(self, robot_id: str):
        """Parse the robot and object from the exact admitted shared sample.

        Returns:
            An identified, co-sampled robot observation.
        """
        return parse_pick_observation(
            self.scene_snapshot,
            robot_id=robot_id,
            physical_instance_id=self.physical_instance_id,
            session_id=self.scene_snapshot.get("session_id"),
            manifest_sha256=self.scene_snapshot.get("manifest_sha256"),
        )

    @property
    def binding_sha256(self) -> str:
        """Return the full operation/model/policy/scene/start-state binding."""
        return _hash(
            {
                "operation_id": self.operation_id,
                "object": self.physical_instance_id,
                "session_robots": self.session_robots,
                "donor": asdict(self.donor),
                "receiver": asdict(self.receiver),
                "scene": self.scene_snapshot,
                "scene_version": self.scene_version,
                "start_state": self.start_state,
                "holding": self.giver_holding_evidence,
                "geometry": asdict(self.object_geometry),
                "profile": self.profile.sha256,
                "follow_samples": self.follow_samples,
                "co_hold_timeout_s": self.co_hold_timeout_s,
                "observed_sim_time_s": self.observed_sim_time_s,
            }
        )


@dataclass(frozen=True)
class HandoverCandidate:
    """One object-relative paired goal; no IK or execution success is inferred."""

    candidate_id: str
    binding_sha256: str
    object_goal: Pose
    object_to_donor_tcp: Pose
    donor_goal: Pose
    receiver_contact: Pose
    receiver_precontact: Pose
    approach_direction_world: tuple[float, float, float]
    receiver_span_m: float

    def __post_init__(self) -> None:
        """Require valid, bound rigid targets and a bottom approach direction."""
        digest(self.candidate_id, "candidate_id")
        digest(self.binding_sha256, "candidate binding")
        for name in (
            "object_goal",
            "object_to_donor_tcp",
            "donor_goal",
            "receiver_contact",
            "receiver_precontact",
        ):
            _pose(asdict(getattr(self, name)), name)
        direction = _vector(self.approach_direction_world, 3, "approach direction")
        if abs(math.sqrt(sum(v * v for v in direction)) - 1) > 1e-6:
            raise ValueError("approach direction must be normalized")
        if direction[2] <= 1e-6:
            raise ValueError("receiver approach must point upward")
        finite(self.receiver_span_m, "receiver span", positive=True)
        if (
            max(
                pose_error(
                    compose(self.object_goal, self.object_to_donor_tcp), self.donor_goal
                )
            )
            > 1e-7
        ):
            raise ValueError("candidate changed the measured giver grasp transform")
        tool_direction = _quat_rotate_vector(
            self.receiver_contact.orientation_xyzw,
            (0, 0, 1),
        )
        delta = tuple(
            a - b
            for a, b in zip(
                self.receiver_contact.position_xyz,
                self.receiver_precontact.position_xyz,
            )
        )
        distance = sum(a * b for a, b in zip(delta, direction))
        if (
            math.dist(direction, tool_direction) > 1e-6
            or distance <= 0
            or math.dist(delta, tuple(distance * x for x in direction)) > 1e-7
            or pose_error(self.receiver_contact, self.receiver_precontact)[1] > 1e-7
        ):
            raise ValueError(
                "candidate precontact does not follow its tool approach axis"
            )
        if self.candidate_id != _hash(
            {
                name: value
                for name, value in asdict(self).items()
                if name != "candidate_id"
            }
        ):
            raise ValueError("candidate identity does not bind all paired geometry")


def generate_candidates(
    request: HandoverPlanningRequest,
) -> tuple[HandoverCandidate, ...]:
    """Sample bounded paired bottom goals while preserving the actual giver grasp.

    Returns:
        Deterministic geometry candidates, not collision-free motion plans.
    """
    donor = request.observation(request.donor.robot_id)
    profile = request.profile
    held = compose(inverse(donor.object_pose), donor.tcp_pose)
    geom = request.object_geometry
    center = tuple((a + b) / 2 for a, b in zip(geom.minimum_m, geom.maximum_m))
    extents = tuple(b - a for a, b in zip(geom.minimum_m, geom.maximum_m))
    forward = tuple(
        sum(
            _quat_rotate_vector(ctx.base.orientation_xyzw, (1, 0, 0))[i]
            for ctx in (request.donor, request.receiver)
        )
        for i in range(3)
    )
    horizontal_norm = math.hypot(forward[0], forward[1])
    if horizontal_norm < 1e-6:
        raise ValueError("opposed base directions require an explicit workspace policy")
    forward = (forward[0] / horizontal_norm, forward[1] / horizontal_norm, 0.0)
    midpoint = tuple(
        (request.donor.base.position_xyz[i] + request.receiver.base.position_xyz[i]) / 2
        for i in range(3)
    )
    positions = [donor.object_pose.position_xyz] + [
        (
            midpoint[0] + forward[0] * distance,
            midpoint[1] + forward[1] * distance,
            midpoint[2] + height,
        )
        for distance, height in itertools.product(
            profile.forward_offsets_m,
            profile.object_heights_m,
        )
    ]
    selections = list(
        itertools.product(
            positions,
            profile.object_yaws_rad,
            profile.object_tilts_rad,
            profile.receiver_yaws_rad,
            profile.receiver_tilts_rad,
        )
    )
    # Diverse, reproducible coverage rather than exhausting only one location.
    random.Random(request.binding_sha256).shuffle(selections)
    candidates = []
    for (
        position,
        object_yaw,
        object_tilt,
        receiver_yaw,
        receiver_tilt,
    ) in selections:
        depth = profile.receiver_depth_offsets_m[0]
        object_rot = compose(_rotation("z", object_yaw), _rotation("y", object_tilt))
        object_goal = Pose(
            position, compose(object_rot, donor.object_pose).orientation_xyzw
        )
        donor_goal = compose(object_goal, held)
        local_tool = compose(
            _rotation("z", receiver_yaw), _rotation("y", receiver_tilt)
        )
        tool = compose(object_goal, local_tool)
        direction = _quat_rotate_vector(tool.orientation_xyzw, (0, 0, 1))
        if direction[2] <= 1e-6:
            continue
        # Shift ONLY in the tool's tangent/approach plane, never along the
        # closing axis: both opposing jaws must still straddle the brick.
        local_tangent = _quat_rotate_vector(local_tool.orientation_xyzw, (1, 0, 0))
        tangent_span = sum(abs(a) * b for a, b in zip(local_tangent, extents))
        tangent_limit = max(0.0, tangent_span / 2 - profile.observation_noise_m)
        lateral = next(
            (
                value
                for value in profile.receiver_lateral_offsets_m
                if abs(value) <= tangent_limit
            ),
            None,
        )
        if lateral is None:
            continue
        tangent = _quat_rotate_vector(tool.orientation_xyzw, (1, 0, 0))
        world_center = compose(object_goal, Pose(center, (0, 0, 0, 1))).position_xyz
        contact = Pose(
            tuple(
                world_center[i]
                + tangent[i] * lateral
                - direction[i] * (profile.contact_clearance_m + depth)
                for i in range(3)
            ),
            tool.orientation_xyzw,
        )
        precontact = Pose(
            tuple(
                contact.position_xyz[i] - direction[i] * profile.precontact_distance_m
                for i in range(3)
            ),
            tool.orientation_xyzw,
        )
        closing_axis = _quat_rotate_vector(local_tool.orientation_xyzw, (0, 1, 0))
        span = sum(abs(closing_axis[i]) * extents[i] for i in range(3))
        binding = request.binding_sha256
        candidate_id = _hash(
            {
                "binding_sha256": binding,
                "object_goal": asdict(object_goal),
                "object_to_donor_tcp": asdict(held),
                "donor_goal": asdict(donor_goal),
                "receiver_contact": asdict(contact),
                "receiver_precontact": asdict(precontact),
                "approach_direction_world": direction,
                "receiver_span_m": span,
            }
        )
        candidates.append(
            HandoverCandidate(
                candidate_id,
                binding,
                object_goal,
                held,
                donor_goal,
                contact,
                precontact,
                direction,
                span,
            )
        )
        if len(candidates) == profile.max_candidates:
            break
    # Reachability coverage comes before grasp refinement. The planning loop
    # prioritizes neighbors only after paired endpoint admission, so a large
    # grasp grid cannot consume the budget on unreachable world poses.
    anchors = tuple(candidates)
    seen = {c.candidate_id for c in candidates}
    for anchor in anchors:
        if len(candidates) >= profile.max_candidates:
            break
        for variant in grasp_variants(request, anchor):
            if variant.candidate_id in seen:
                continue
            candidates.append(variant)
            seen.add(variant.candidate_id)
            if len(candidates) >= profile.max_candidates:
                break
    return tuple(candidates)


def grasp_variants(
    request: HandoverPlanningRequest, candidate: HandoverCandidate
) -> tuple[HandoverCandidate, ...]:
    """Refine only the frozen profile's tangent/depth grid at a paired goal.

    Returns:
        Bound receiver grasp neighbors with unchanged giver and object targets.
    """
    if candidate.binding_sha256 != request.binding_sha256:
        raise ValueError("grasp refinement belongs to a foreign request")
    geometry, profile = request.object_geometry, request.profile
    center = tuple((a + b) / 2 for a, b in zip(geometry.minimum_m, geometry.maximum_m))
    extents = tuple(b - a for a, b in zip(geometry.minimum_m, geometry.maximum_m))
    world_center = compose(
        candidate.object_goal, Pose(center, (0, 0, 0, 1))
    ).position_xyz
    orientation = candidate.receiver_contact.orientation_xyzw
    tangent = _quat_rotate_vector(orientation, (1, 0, 0))
    local_tool = compose(inverse(candidate.object_goal), candidate.receiver_contact)
    local_tangent = _quat_rotate_vector(local_tool.orientation_xyzw, (1, 0, 0))
    tangent_span = sum(abs(a) * b for a, b in zip(local_tangent, extents))
    direction = candidate.approach_direction_world
    variants = []
    for lateral, depth in itertools.product(
        profile.receiver_lateral_offsets_m, profile.receiver_depth_offsets_m
    ):
        if abs(lateral) > max(0.0, tangent_span / 2 - profile.observation_noise_m):
            continue
        contact = Pose(
            tuple(
                center + lateral * axis - (profile.contact_clearance_m + depth) * d
                for center, axis, d in zip(world_center, tangent, direction)
            ),
            orientation,
        )
        precontact = Pose(
            tuple(
                p - profile.precontact_distance_m * d
                for p, d in zip(contact.position_xyz, direction)
            ),
            orientation,
        )
        payload = {
            key: value
            for key, value in asdict(candidate).items()
            if key != "candidate_id"
        }
        payload.update(
            receiver_contact=asdict(contact), receiver_precontact=asdict(precontact)
        )
        variant = replace(
            candidate,
            candidate_id=_hash(payload),
            receiver_contact=contact,
            receiver_precontact=precontact,
        )
        if variant.candidate_id != candidate.candidate_id:
            variants.append(variant)
    return tuple(variants)


def calibrated_target(
    pose: Pose,
    ctx: HandoverRobotContext,
    profile: HandoverProfile,
) -> Mapping[str, object]:
    """Encode a world goal in this robot's exact calibrated frame.

    Returns:
        A namespaced, provenance-bound TCP pose target.
    """
    return {
        "tcp_pose": {
            "position": list(world_position_to_base(pose.position_xyz, ctx.base)),
            "orientation_xyzw": list(
                world_orientation_to_base(pose.orientation_xyzw, ctx.base)
            ),
            "frame_id": f"{ctx.robot_id}_base_link",
            "tip_link": f"{ctx.robot_id}_rocobrick_tcp",
            "runtime_model_sha256": ctx.provenance.runtime_model_sha256,
            "source_sha256": ctx.provenance.source_sha256,
            "position_tolerance_m": profile.position_tolerance_m,
            "orientation_tolerance_rad": profile.orientation_tolerance_rad,
        }
    }


@dataclass(frozen=True)
class CandidateEvaluation:
    """Explicit read-only feasibility gates supplied by the planning transport."""

    candidate_id: str
    binding_sha256: str
    giver_ik: bool
    receiver_ik: bool
    combined_state_valid: bool
    gripper_sweeps_valid: bool
    approach_valid: bool
    separation_valid: bool
    scene_unchanged: bool
    reason: str | None = None
    joint_path_valid: bool = False
    joint_plan: CoordinatedOperation | None = None

    def __post_init__(self) -> None:
        """Reject incomplete identity, truthy gates and unclassified reasons."""
        digest(self.candidate_id, "evaluation candidate")
        digest(self.binding_sha256, "evaluation binding")
        for name in self.gates:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a real bool")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("evaluation reason must be text or null")
        if self.joint_plan is not None and (
            not isinstance(self.joint_plan, CoordinatedOperation)
            or self.joint_plan.coordination_id != self.candidate_id
        ):
            raise ValueError("joint plan must be bound to the evaluated candidate")
        if self.joint_path_valid and self.joint_plan is None:
            raise ValueError("joint path proof requires a combined trajectory")

    @property
    def gates(self) -> tuple[str, ...]:
        """Return required endpoint, sweep, path and scene gates."""
        return (
            "giver_ik",
            "receiver_ik",
            "combined_state_valid",
            "gripper_sweeps_valid",
            "approach_valid",
            "separation_valid",
            "scene_unchanged",
            "joint_path_valid",
        )

    @property
    def passed(self) -> bool:
        """Return feasibility only, never physical HANDOVER success."""
        return self.reason is None and all(getattr(self, name) for name in self.gates)


class HandoverPlanningTransport(Protocol):
    """Readonly complete-scene endpoint and path validation seam."""

    capabilities: frozenset[str]

    def evaluate_candidate(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        *,
        deadline_s: float,
    ) -> CandidateEvaluation:
        """Evaluate without controller dispatch or shared scene mutation."""
        ...


@dataclass(frozen=True)
class HandoverPlanningResult:
    """A selected feasible candidate or a bounded, classified planning failure."""

    binding_sha256: str
    candidate: HandoverCandidate | None
    evaluation: CandidateEvaluation | None
    evaluated: int
    rejected: Mapping[str, int]
    reason: str | None
    rejection_reasons: Mapping[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return a bound planning result, not an execution verdict."""
        return (
            self.candidate is not None
            and self.evaluation is not None
            and self.evaluation.passed
            and self.candidate.binding_sha256 == self.binding_sha256
            and self.evaluation.binding_sha256 == self.binding_sha256
            and self.evaluation.candidate_id == self.candidate.candidate_id
            and self.reason is None
        )


def plan_handover(
    request: HandoverPlanningRequest,
    transport: HandoverPlanningTransport,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> HandoverPlanningResult:
    """Select a fully evaluated paired goal within a frozen finite budget.

    Returns:
        A no-motion result, preserving why every rejected candidate failed.
    """
    require_handover_capabilities(
        request.profile, transport.capabilities, physical=False
    )
    binding = request.binding_sha256
    deadline = clock() + request.profile.planning_timeout_s
    rejected: dict[str, int] = {}
    rejection_reasons: dict[str, int] = {}
    evaluated = 0
    reason = "NO_FEASIBLE_HANDOVER_IN_PROFILE_BUDGET"
    anchors = iter(generate_candidates(request))
    pending: deque[HandoverCandidate] = deque()
    queued: set[str] = set()
    seen: set[str] = set()
    while True:
        if clock() >= deadline:
            reason = "HANDOVER_PLANNING_BUDGET_EXHAUSTED"
            break
        if evaluated >= request.profile.max_evaluations:
            break
        candidate = pending.popleft() if pending else next(anchors, None)
        if candidate is None:
            break
        queued.discard(candidate.candidate_id)
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        evaluated += 1
        result = transport.evaluate_candidate(request, candidate, deadline_s=deadline)
        if not isinstance(result, CandidateEvaluation):
            raise ValueError("planning transport returned an invalid evaluation")
        if (
            request.binding_sha256 != binding
            or result.binding_sha256 != binding
            or result.candidate_id != candidate.candidate_id
        ):
            raise ValueError("candidate evaluation binding changed or is foreign")
        if clock() >= deadline:
            reason = "HANDOVER_PLANNING_BUDGET_EXHAUSTED"
            break
        if result.passed:
            return HandoverPlanningResult(
                binding,
                candidate,
                result,
                evaluated,
                dict(rejected),
                None,
                dict(rejection_reasons),
            )
        category = result.reason or "MISSING_REQUIRED_PLANNING_PROOF"
        rejection_reasons[category] = rejection_reasons.get(category, 0) + 1
        for name in result.gates:
            if not getattr(result, name):
                rejected[name] = rejected.get(name, 0) + 1
        if result.giver_ik and result.receiver_ik and result.combined_state_valid:
            for neighbor in grasp_variants(request, candidate):
                if neighbor.candidate_id not in seen | queued:
                    pending.append(neighbor)
                    queued.add(neighbor.candidate_id)
    return HandoverPlanningResult(
        binding, None, None, evaluated, dict(rejected), reason, dict(rejection_reasons)
    )
