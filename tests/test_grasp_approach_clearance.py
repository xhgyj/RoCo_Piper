"""Stage-specific hull safety: platform margin is not a grasp exemption."""

from types import SimpleNamespace

import numpy as np
import pytest

from rocobrick.clearance.grasp_approach import grasp_approach_clearance
from rocobrick.clearance.hulls import box_hull


def query(height):
    """Return minimal fixed-link kinematics and measured world geometry."""
    links = ("gripper_base", "gripper_link1", "gripper_link2")
    model = '<robot name="test"><link name="world"/>'
    for link in links:
        model += (
            f'<link name="r_{link}"/><joint name="{link}" type="fixed">'
            f'<parent link="world"/><child link="r_{link}"/>'
            f'<origin xyz="0 0 {height}"/></joint>'
        )
    model += "</robot>"
    request = SimpleNamespace(
        targets={
            "r": {
                "grasp_region": {
                    "physical_instance_id": "b",
                    "ground_top_world_m": 0.08,
                    "floor_margin_m": 0.005,
                    "contact_margin_m": 0.001,
                }
            }
        },
        start_state={},
        scene_snapshot={
            "bricks": {"b": {"aabb_min": [1, 1, 0.08], "aabb_max": [1.02, 1.02, 0.1]}},
            "environment_colliders": {
                "/World/feeding_platforms/r": {
                    "aabb_min": [-0.1, -0.1, 0],
                    "aabb_max": [0.1, 0.1, 0.08],
                }
            },
        },
    )
    trajectory = {"joint_names": ["unused"], "points": [{"positions": [0]}]}
    return model, request, trajectory


def test_platform_near_miss_is_blocked_before_dispatch(monkeypatch):
    """Positive 1 mm geometric gap still violates the existing 5 mm margin."""
    hull = box_hull("stub", np.array([-0.001] * 3), np.array([0.001] * 3))
    monkeypatch.setattr(
        "rocobrick.clearance.grasp_approach.load_hulls",
        lambda: SimpleNamespace(hull=lambda _: hull),
    )
    with pytest.raises(ValueError, match="GRASP_APPROACH_WORLD_CLEARANCE"):
        grasp_approach_clearance(*query(0.082))
    result = grasp_approach_clearance(*query(0.09))
    assert result["checked_pairs"] == 6
    assert result["contact_exemptions"] == []


def test_other_skill_contact_policies_are_not_overridden():
    """Only explicit grasp-region approach requests enter this gate."""
    assert grasp_approach_clearance("", SimpleNamespace(targets={}), {}) == {
        "applicable": False,
    }


@pytest.mark.parametrize("target_key", ["grasp_region", "grasp_approach_region"])
def test_platform_margin_is_request_scoped_and_still_rejects_contact(monkeypatch, target_key):
    """Endpoint, hover and local path gates honor the same serialized margin."""
    hull = box_hull("stub", np.array([-0.001] * 3), np.array([0.001] * 3))
    monkeypatch.setattr(
        "rocobrick.clearance.grasp_approach.load_hulls",
        lambda: SimpleNamespace(hull=lambda _: hull),
    )
    model, request, trajectory = query(0.085344)
    with pytest.raises(ValueError, match="GRASP_APPROACH_WORLD_CLEARANCE"):
        grasp_approach_clearance(model, request, trajectory)
    region = request.targets["r"].pop("grasp_region")
    region["floor_margin_m"] = 0.004
    request.targets["r"][target_key] = region
    result = grasp_approach_clearance(model, request, trajectory)
    assert result["worst"]["distance_m"] == pytest.approx(0.004344)
    assert result["worst"]["required_margin_m"] == 0.004
    assert result["contact_exemptions"] == []
    # Reducing positive clearance must never turn penetration into admission.
    for height in (0.084, 0.0805):
        model, request, trajectory = query(height)
        request.targets["r"]["grasp_region"]["floor_margin_m"] = 0.004
        with pytest.raises(ValueError, match="GRASP_APPROACH_WORLD_CLEARANCE"):
            grasp_approach_clearance(model, request, trajectory)
