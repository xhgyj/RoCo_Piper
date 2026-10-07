"""Read-only paired IK and combined-trajectory admission for HANDOVER.

The provider must operate on an isolated planning scene containing both arms,
both grippers, the attached object and environment. There is deliberately no
executor or shared-scene mutation method in this seam. Endpoint validity is
not continuous collision proof and independent arm plans are not accepted.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Protocol

from rocobrick.coordinated_moveit import CoordinatedOperation
from rocobrick.skills.handover_planning import (
    CandidateEvaluation,
    HandoverCandidate,
    HandoverPlanningRequest,
    HandoverRobotContext,
    calibrated_target,
)
from rocobrick.skills.handover_profile import finite, require_handover_capabilities
from rocobrick.skills.observation import Pose


class HandoverScenePlanner(Protocol):
    """Complete geometry provider; absence is not inferred as collision-free."""

    capabilities: frozenset[str]

    def solve_ik(
        self,
        request: HandoverPlanningRequest,
        context: HandoverRobotContext,
        target: Mapping[str, object],
        *,
        deadline_s: float,
    ) -> Mapping[str, float] | None:
        """Solve one calibrated target using the measured full-state seed."""
        ...

    def validate_pair(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> bool:
        """Validate all limits and full pair geometry, not arm-only spheres."""
        ...

    def validate_gripper_sweeps(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> bool:
        """Check receiver closing and giver opening through their full sweeps."""
        ...

    def validate_approach(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> bool:
        """Check the Cartesian bottom approach with giver joints held fixed."""
        ...

    def validate_separation(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> bool:
        """Check giver separation and receiver-follow feasibility after release."""
        ...

    def plan_combined(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        joints: Mapping[str, float],
        *,
        deadline_s: float,
    ) -> CoordinatedOperation | None:
        """Plan one combined path from measured start to the paired IK goal."""
        ...

    def validate_continuous_path(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        plan: CoordinatedOperation,
        *,
        deadline_s: float,
    ) -> bool:
        """Validate entire swept path, including grippers and carried object."""
        ...

    def scene_unchanged(self, request: HandoverPlanningRequest) -> bool:
        """Read back exact scene identity/version without applying scene diffs."""
        ...


def _joints(context: HandoverRobotContext) -> tuple[str, ...]:
    return tuple(f"{context.robot_id}_joint{i}" for i in range(1, 7))


def _solution(
    value: Mapping[str, float] | None,
    context: HandoverRobotContext,
) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(_joints(context)):
        raise ValueError("IK solution must name exactly its own six arm joints")
    return {name: finite(value[name], "IK joint") for name in _joints(context)}


def _validate_plan(
    request: HandoverPlanningRequest,
    candidate: HandoverCandidate,
    plan: CoordinatedOperation,
    goal: Mapping[str, float],
) -> None:
    if not isinstance(plan, CoordinatedOperation):
        raise ValueError("combined planner must return a coordinated trajectory")
    participants = (request.donor.robot_id, request.receiver.robot_id)
    if (
        plan.coordination_id != candidate.candidate_id
        or plan.scene_digest != request.scene_snapshot["scene_digest"]
        or plan.participants != participants
        or len(plan.trajectories) != len(participants)
        or plan.synchronization_tolerance_s > request.profile.max_sync_skew_s
    ):
        raise ValueError("combined plan identity/scene/participants/skew mismatch")
    by_robot = {t.robot_id: t for t in plan.trajectories}
    if set(by_robot) != set(participants):
        raise ValueError("combined trajectory participant binding is incomplete")
    grids = []
    for context in (request.donor, request.receiver):
        trajectory = by_robot[context.robot_id]
        if set(trajectory.joint_names) != set(_joints(context)):
            raise ValueError(
                "combined trajectory must contain both full six-joint arms"
            )
        grids.append(tuple(p.time_from_start_s for p in trajectory.points))
        for name, initial, final in zip(
            trajectory.joint_names,
            trajectory.points[0].positions,
            trajectory.points[-1].positions,
        ):
            if (
                abs(initial - request.start_state[name])
                > request.profile.joint_hold_tolerance_rad
            ):
                raise ValueError("combined trajectory does not start at measured state")
            if abs(final - goal[name]) > request.profile.joint_hold_tolerance_rad:
                raise ValueError("combined trajectory does not end at paired IK goals")
    if grids[0] != grids[1]:
        raise ValueError("combined trajectory must retain one identical time grid")


class HandoverJointPlanningBackend:
    """Evaluate both goals and one validated joint plan, without dispatch."""

    def __init__(
        self,
        provider: HandoverScenePlanner,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind an explicitly capable isolated-scene planner and budget clock."""
        self._provider = provider
        self._clock = clock
        self.capabilities = provider.capabilities

    def evaluate_candidate(
        self,
        request: HandoverPlanningRequest,
        candidate: HandoverCandidate,
        *,
        deadline_s: float,
    ) -> CandidateEvaluation:
        """Plan paired goals only after sweep and contact-path validation.

        Returns:
            Bound no-motion feasibility gates and the admitted combined plan.
        """
        require_handover_capabilities(
            request.profile, self.capabilities, physical=False
        )
        binding = request.binding_sha256
        if candidate.binding_sha256 != binding:
            raise ValueError("candidate belongs to a foreign handover request")
        gates = {
            name: False
            for name in (
                "giver_ik",
                "receiver_ik",
                "combined_state_valid",
                "gripper_sweeps_valid",
                "approach_valid",
                "separation_valid",
                "scene_unchanged",
                "joint_path_valid",
            )
        }
        plan = None

        def finish(reason: str | None) -> CandidateEvaluation:
            if request.binding_sha256 != binding:
                raise ValueError("planning provider mutated the frozen request")
            return CandidateEvaluation(
                candidate.candidate_id,
                candidate.binding_sha256,
                **gates,
                reason=reason,
                joint_plan=plan,
            )

        def expired() -> bool:
            return self._clock() >= deadline_s

        goals: dict[str, float] = {}
        targets: tuple[tuple[HandoverRobotContext, Pose, str], ...] = (
            (request.donor, candidate.donor_goal, "giver_ik"),
            (request.receiver, candidate.receiver_precontact, "receiver_ik"),
        )
        for context, pose, gate in targets:
            if expired():
                return finish("HANDOVER_PLANNING_BUDGET_EXHAUSTED")
            solution = _solution(
                self._provider.solve_ik(
                    request,
                    context,
                    calibrated_target(pose, context, request.profile),
                    deadline_s=deadline_s,
                ),
                context,
            )
            if solution is None:
                return finish(f"{gate.upper()}_FAILED")
            gates[gate] = True
            goals.update(solution)
        checks = (
            ("combined_state_valid", self._provider.validate_pair),
            ("gripper_sweeps_valid", self._provider.validate_gripper_sweeps),
            ("approach_valid", self._provider.validate_approach),
            ("separation_valid", self._provider.validate_separation),
        )
        for gate, check in checks:
            if expired():
                return finish("HANDOVER_PLANNING_BUDGET_EXHAUSTED")
            outcome = check(request, candidate, goals, deadline_s=deadline_s)
            if type(outcome) is not bool:
                raise ValueError(f"{gate} must return a real bool")
            gates[gate] = outcome
            if not outcome:
                return finish(f"{gate.upper()}_FAILED")
        if expired():
            return finish("HANDOVER_PLANNING_BUDGET_EXHAUSTED")
        plan = self._provider.plan_combined(
            request, candidate, goals, deadline_s=deadline_s
        )
        if plan is None:
            return finish("COMBINED_PLAN_FAILED")
        _validate_plan(request, candidate, plan, goals)
        if expired():
            return finish("HANDOVER_PLANNING_BUDGET_EXHAUSTED")
        gates["joint_path_valid"] = self._provider.validate_continuous_path(
            request,
            candidate,
            plan,
            deadline_s=deadline_s,
        )
        gates["scene_unchanged"] = self._provider.scene_unchanged(request)
        if expired():
            return finish("HANDOVER_PLANNING_BUDGET_EXHAUSTED")
        return finish(
            None if all(gates.values()) else "PATH_OR_SCENE_VALIDATION_FAILED"
        )
