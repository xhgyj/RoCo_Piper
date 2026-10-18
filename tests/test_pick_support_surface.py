"""Measured support-level resolution and protected platform planning boxes."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_pick_handler import Observer, document, operation
from test_place_down_handler import context

from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.feeding_platforms import FeedingPlatform
from rocobrick.skills.grasp_region import supporting_surface_top
from rocobrick.skills.pick import PickHandler
from rocobrick.skills.target_context import load_target_contexts

PLATFORM = FeedingPlatform("piper_0", (0.35, 0.06), (0.24, 0.24), 0.08, 0.08)


def test_support_uses_platform_not_global_floor():
    """The actual observed level replaces the lower global-ground proposal."""
    ctx = replace(context(), feeding_platforms=(PLATFORM,))
    obs = SimpleNamespace(
        aabb_min=(0.33, 0.04, 0.0800001), aabb_max=(0.37, 0.08, 0.1024)
    )
    assert supporting_surface_top(obs, ctx) == 0.08
    for bottom in (0.07, 0.09):
        with pytest.raises(ValueError, match="SUPPORT_UNRESOLVED"):
            supporting_surface_top(
                SimpleNamespace(aabb_min=(0.33, 0.04, bottom), aabb_max=obs.aabb_max),
                ctx,
            )
    with pytest.raises(ValueError, match="SUPPORT_UNRESOLVED"):
        supporting_surface_top(
            obs, replace(ctx, feeding_platforms=(PLATFORM, PLATFORM))
        )


def test_context_loads_same_platforms_for_subset_robot_contexts():
    """Immutable contexts use the simulator's existing platform parser."""
    from pathlib import Path

    root = Path(__file__).parents[1]
    ctx = load_target_contexts(
        root / "config/user_config_type1_scaled2_C.json",
        root / "config/skill_params.json",
        ("piper_0",),
        runtime_model_sha256="a" * 64,
        source_sha256="b" * 64,
    )["piper_0"]
    assert len(ctx.feeding_platforms) == 2
    assert ctx.feeding_platforms[0] == PLATFORM
    assert ctx.ground.top_z_m == -0.001


def test_platforms_never_receive_finger_contact_exemptions():
    """Final grasp exemptions affect the target only, never its support box."""
    ledger = OperationStateLedger(("piper_0",))
    ledger.confirm(
        "piper_0",
        "robot_at_pick_region",
        {"arrival": True, "physical_instance_id": "brick-a"},
    )
    handler = PickHandler(
        Observer(document()), ledger, session_id="session-1", manifest_sha256="a" * 64
    )
    admission = asyncio.run(handler.preflight(operation()))
    ctx = replace(context(), feeding_platforms=(PLATFORM,))
    payload = handler._planning_payload(admission, ctx, allow_contact=True)
    boxes = {box["object_id"]: box for box in payload["boxes"]}
    assert boxes[PLATFORM.prim_path]["maximum_m"][2] == 0.08
    assert not any(
        PLATFORM.prim_path in pair for pair in payload["allowed_collision_pairs"]
    )


def test_experimental_margin_only_applies_to_platform_supported_pick():
    """Serialize one shared margin for the native sampler and path hull gates."""
    from test_pick_grasp_region import sample
    from rocobrick.skills.grasp_region import place_up_grasp_region
    from rocobrick.skills.observation import Pose

    geometry = {"pad_vertices_tcp": [[0.0, 0.0, 0.0]], "flange_to_tcp_m": 0.13}
    ground = place_up_grasp_region(sample(), context(), geometry=geometry)
    assert ground["floor_margin_m"] == 0.005
    observed = replace(
        sample(),
        aabb_min=(0.34, 0.05, 0.08),
        aabb_max=(0.36, 0.07, 0.10),
        object_pose=Pose((0.35, 0.06, 0.09), (0.0, 0.0, 0.0, 1.0)),
    )
    platform = place_up_grasp_region(
        observed, replace(context(), feeding_platforms=(PLATFORM,)), geometry=geometry
    )
    assert platform["floor_margin_m"] == 0.004
    assert platform["ground_top_world_m"] == 0.08
    assert platform["contact_margin_m"] == ground["contact_margin_m"]
