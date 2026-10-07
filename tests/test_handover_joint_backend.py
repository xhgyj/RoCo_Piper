"""Offline complete-pair path admission tests; no ROS/Isaac is launched."""

from dataclasses import replace

import pytest
from test_handover_joint_planning import joint_plan, planning_request

from rocobrick.coordinated_moveit import CoordinatedTrajectoryPoint
from rocobrick.skills.handover_joint_backend import HandoverJointPlanningBackend
from rocobrick.skills.handover_planning import generate_candidates
from rocobrick.skills.handover_profile import PLANNING_CAPABILITIES


class Provider:
    """Explicit geometry/path stub, not evidence of physical feasibility."""

    capabilities = PLANNING_CAPABILITIES

    def __init__(self):
        """Track seam calls and optional injected failures."""
        self.calls = []
        self.plan_mutator = lambda plan: plan
        self.reject = None

    def solve_ik(self, request, context, target, *, deadline_s):
        """Return an exactly identified synthetic six-joint solution."""
        self.calls.append((context.robot_id, target["tcp_pose"]))
        return {f"{context.robot_id}_joint{i}": 0.1 for i in range(1, 7)}

    def _validate(self, gate, joints):
        self.calls.append(gate)
        assert len(joints) == 12
        return gate != self.reject

    def validate_pair(self, request, candidate, joints, *, deadline_s):
        """Return the injected full-pair endpoint verdict."""
        return self._validate("pair", joints)

    def validate_gripper_sweeps(self, request, candidate, joints, *, deadline_s):
        """Return the injected closing/opening sweep verdict."""
        return self._validate("sweeps", joints)

    def validate_approach(self, request, candidate, joints, *, deadline_s):
        """Return the injected bottom approach verdict."""
        return self._validate("approach", joints)

    def validate_separation(self, request, candidate, joints, *, deadline_s):
        """Return the injected giver separation verdict."""
        return self._validate("separation", joints)

    def plan_combined(self, request, candidate, joints, *, deadline_s):
        """Return a synthetic common-time pair plan, optionally corrupted.

        Returns:
            A coordinated plan for admission testing only.
        """
        self.calls.append("plan")
        assert len(joints) == 12
        return self.plan_mutator(joint_plan(request, candidate))

    def validate_continuous_path(self, request, candidate, plan, *, deadline_s):
        """Return the injected continuous path verdict."""
        self.calls.append("continuous")
        return self.reject != "continuous"

    def scene_unchanged(self, request):
        """Return the injected scene-readback verdict."""
        self.calls.append("scene")
        return self.reject != "scene"


def evaluate(provider):
    """Run a deterministic, never-expiring candidate admission.

    Returns:
        Candidate evaluation from the synthetic provider.
    """
    request = planning_request()
    candidate = generate_candidates(request)[0]
    return HandoverJointPlanningBackend(provider, clock=lambda: 0.0).evaluate_candidate(
        request,
        candidate,
        deadline_s=1.0,
    )


def test_backend_plans_both_measured_calibrated_targets_on_one_joint_path():
    """Donor relocation and receiver bottom precontact are planned together."""
    provider = Provider()
    result = evaluate(provider)
    assert result.passed
    assert result.joint_plan is not None
    assert len(result.joint_plan.trajectories) == 2
    donor, receiver = provider.calls[:2]
    assert donor[0] == "piper_1" and receiver[0] == "piper_0"
    assert donor[1]["frame_id"] == "piper_1_base_link"
    assert receiver[1]["frame_id"] == "piper_0_base_link"
    assert provider.calls[2:] == [
        "pair",
        "sweeps",
        "approach",
        "separation",
        "plan",
        "continuous",
        "scene",
    ]


@pytest.mark.parametrize("stage", ["pair", "sweeps", "approach", "separation"])
def test_geometry_failure_prevents_combined_planning(stage):
    """Individual IK success does not admit the collision/contact paths."""
    provider = Provider()
    provider.reject = stage
    result = evaluate(provider)
    assert not result.passed and result.joint_plan is None
    assert "plan" not in provider.calls


@pytest.mark.parametrize("stage", ["continuous", "scene"])
def test_planned_path_requires_continuous_geometry_and_scene_readback(stage):
    """A trajectory alone cannot claim all path and binding proofs passed."""
    provider = Provider()
    provider.reject = stage
    result = evaluate(provider)
    assert not result.passed
    assert result.joint_plan is not None


@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "scene",
        "sync",
        "start",
        "goal",
        "time-grid",
        "partial-arm",
    ],
)
def test_combined_path_rejects_foreign_or_independent_arm_evidence(fault):
    """Full start/goal/identity coverage and a common time grid are required."""

    def mutate(plan):
        if fault == "identity":
            return replace(plan, coordination_id="f" * 64)
        if fault == "scene":
            return replace(plan, scene_digest="f" * 64)
        if fault == "sync":
            return replace(plan, synchronization_tolerance_s=0.1)
        trajectory = plan.trajectories[0]
        points = trajectory.points
        if fault == "partial-arm":
            trajectory = replace(
                trajectory,
                joint_names=trajectory.joint_names[:5],
                points=tuple(replace(p, positions=p.positions[:5]) for p in points),
            )
        else:
            first = points[0]
            last = points[-1]
            if fault == "start":
                first = CoordinatedTrajectoryPoint(0.0, (0.02,) * 6)
            elif fault == "goal":
                last = CoordinatedTrajectoryPoint(1.0, (0.2,) * 6)
            else:
                last = replace(last, time_from_start_s=2.0)
            trajectory = replace(trajectory, points=(first, last))
        return replace(plan, trajectories=(trajectory, plan.trajectories[1]))

    provider = Provider()
    provider.plan_mutator = mutate
    with pytest.raises(ValueError):
        evaluate(provider)
    assert "continuous" not in provider.calls


def test_endpoint_truth_without_combined_trajectory_is_rejected():
    """Even explicit true gates need their combined trajectory artifact."""
    result = evaluate(Provider())
    with pytest.raises(ValueError, match="combined trajectory"):
        replace(result, joint_plan=None)


def test_provider_cannot_mutate_scene_while_reporting_unchanged():
    """A read-only contract also checks its complete frozen input binding."""

    class MutatingProvider(Provider):
        def scene_unchanged(self, request):
            """Return a false claim after changing the requested scene version."""
            request.scene_snapshot["sequence"] += 1
            return True

    with pytest.raises(ValueError, match="mutated"):
        evaluate(MutatingProvider())


def test_candidate_deadline_is_checked_between_provider_stages():
    """A slow IK cannot cause later planning calls after the shared deadline."""
    provider = Provider()
    request = planning_request()
    candidate = generate_candidates(request)[0]
    ticks = iter((0.0, 2.0))
    result = HandoverJointPlanningBackend(
        provider, clock=lambda: next(ticks)
    ).evaluate_candidate(
        request,
        candidate,
        deadline_s=1.0,
    )
    assert not result.passed and len(provider.calls) == 1
    assert result.reason == "HANDOVER_PLANNING_BUDGET_EXHAUSTED"
