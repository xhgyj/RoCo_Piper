"""Native plan compilation is separate from controllers and physical qualification."""

import asyncio
from dataclasses import replace

import pytest
from test_coordinated_handover_lifecycle import operation, program
from test_handover_joint_planning import planning_request
from test_handover_scene_planner import native_provider

from rocobrick.adapters.moveit.handover_geometry import IsolatedHandoverGeometry
from rocobrick.adapters.moveit.handover_program import NativeCoordinatedHandoverFactory
from rocobrick.adapters.moveit.handover_scene_planner import (
    HandoverGripperConfigurations,
    NativeHandoverScenePlanner,
    ReadOnlyCombinedHandoverPlanner,
)
from rocobrick.skills.handover_profile import EXECUTION_CAPABILITIES


def test_unqualified_profile_rejected_before_native_provider_or_controller():
    """The compiler cannot promote a planning-only policy to motion."""
    request = planning_request()

    def forbidden(_):
        pytest.fail("unqualified profile must be rejected before provider creation")

    factory = NativeCoordinatedHandoverFactory(
        lambda _: request,
        forbidden,
        forbidden,
        execution_capabilities=EXECUTION_CAPABILITIES,
        phase_timeout_s=60,
    )
    with pytest.raises(ValueError, match="NOT_QUALIFIED"):
        factory._compile(operation(program()))


def test_native_compiler_preserves_the_exact_full_model_phase_chain(tmp_path):
    """Actual native geometry compiles all phases; only MoveGroup is synthetic."""
    previous, captured, _, runtime = native_provider(tmp_path)
    request = replace(
        captured, profile=replace(captured.profile, execution_qualified=True)
    )
    geometry = IsolatedHandoverGeometry(previous.geometry.files, request)
    runtime.geometry, runtime.captured = geometry, request

    def fingers(candidate):
        """Return complete synthetic finger calibration for this exact candidate."""
        return HandoverGripperConfigurations(
            candidate.candidate_id,
            "f" * 64,
            {"piper_0_gripper_joint1": 0.02, "piper_0_gripper_joint2": -0.02},
            {"piper_0_gripper_joint1": 0.006, "piper_0_gripper_joint2": -0.006},
            {"piper_1_gripper_joint1": 0.02, "piper_1_gripper_joint2": -0.02},
        )

    provider = NativeHandoverScenePlanner(
        geometry,
        ReadOnlyCombinedHandoverPlanner(runtime),
        grippers=fingers,
        read_binding=lambda: request.binding_sha256,
    )
    factory = NativeCoordinatedHandoverFactory(
        lambda _: request,
        lambda _: provider,
        lambda _: ((0.015, 0.017), (0.015, 0.017), (0.075, 0.085)),
        execution_capabilities=EXECUTION_CAPABILITIES,
        phase_timeout_s=60,
    )
    result = asyncio.run(factory.prepare(operation(program())))
    assert result.request is request
    assert result.paths.candidate_id == result.candidate.candidate_id
    assert dict(result.paths.receiver_open[0]) == dict(request.start_state)
    assert result.rendezvous.coordination_id == result.candidate.candidate_id
    assert result.current_binding() == result.binding_sha256
    assert runtime.requests
    assert all(
        set(t) == {"tcp_region"} for q in runtime.requests for t in q.targets.values()
    )


@pytest.mark.parametrize(
    "change", ["partial", "jump", "held_arm", "scene", "start", "time_grid"]
)
def test_program_rejects_broken_or_discontinuous_certified_phase_chain(change):
    """No execution starts with an incomplete arm or a discontinuous phase path."""
    p = program()
    if change in {"partial", "jump", "held_arm"}:
        states = [dict(v) for v in p.paths.approach]
        if change == "partial":
            states[0].pop("piper_0_joint1")
        elif change == "jump":
            states[-1]["piper_0_joint1"] = 0.05
        else:
            states[1]["piper_1_joint1"] = 0.05
        with pytest.raises(ValueError):
            replace(p, paths=replace(p.paths, approach=tuple(states)))
    else:
        motion = p.rendezvous
        if change == "scene":
            motion = replace(motion, scene_digest="a" * 64)
        elif change == "start":
            first = motion.trajectories[0]
            point = replace(first.points[0], positions=(0.1,) * 6)
            first = replace(first, points=(point, first.points[-1]))
            motion = replace(motion, trajectories=(first, motion.trajectories[1]))
        else:
            first = motion.trajectories[0]
            point = replace(first.points[-1], time_from_start_s=1.5)
            first = replace(first, points=(first.points[0], point))
            motion = replace(motion, trajectories=(first, motion.trajectories[1]))
        with pytest.raises(ValueError):
            replace(p, rendezvous=motion)
