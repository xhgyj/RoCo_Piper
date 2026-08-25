"""Tests for the privileged single-step GT assembly expert."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rocobrick.policy import gt_assembly

ROOT = Path(__file__).resolve().parents[1]


def _environment() -> SimpleNamespace:
    return SimpleNamespace(
        topology={
            "parts": [
                {"id": 0, "payload": {"L": 32, "W": 32, "H": 1}},
                {"id": 1, "payload": {"L": 2, "W": 2, "H": 3}},
                {"id": 2, "payload": {"L": 2, "W": 2, "H": 3}},
                {"id": 3, "payload": {"L": 4, "W": 2, "H": 3}},
            ],
            "connections": [
                {
                    "stud_id": 1,
                    "stud_iface": 1,
                    "hole_id": 3,
                    "hole_iface": 0,
                    "offset": [0, 0],
                    "yaw": 0,
                },
                {
                    "stud_id": 2,
                    "stud_iface": 1,
                    "hole_id": 3,
                    "hole_iface": 0,
                    "offset": [-2, 0],
                    "yaw": 0,
                },
            ],
        },
        pre_placed_parts={0: "/plate", 1: "/left", 2: "/right"},
        to_place_placed={3: "/target"},
    )


def test_resolver_builds_one_target_with_all_references() -> None:
    """A bridge remains one action while retaining both GT references."""
    task = gt_assembly.resolve_single_step_task(_environment())
    assert task.target_id == 3
    assert task.target_path == "/target"
    assert {item.reference_id for item in task.connections} == {1, 2}
    assert task.dimensions == {"L": 4, "W": 2, "H": 3}


def test_reference_orientation_rotates_local_connection_offset(monkeypatch) -> None:
    """The reference world transform maps a local offset into world axes."""
    core = ModuleType("bricksim.core")
    core.compute_connection_transform = lambda **kwargs: (
        (1.0, 0.0, 0.0, 0.0),
        (0.008, 0.0, 0.0096),
    )
    monkeypatch.setitem(sys.modules, "bricksim.core", core)
    world_t_reference = np.eye(4)
    world_t_reference[:3, :3] = Rotation.from_euler(
        "z", 90, degrees=True
    ).as_matrix()
    world_t_reference[:3, 3] = [0.1, -0.2, 0.01]
    env = SimpleNamespace(
        get_prim_world_T=lambda path: world_t_reference,
    )
    connection = gt_assembly.GTConnection(
        1, 2, "/reference", "/target", 1, 0, (1, 0), 0, 2
    )
    goal = gt_assembly._goal_from_connection(env, connection)
    np.testing.assert_allclose(goal[:3, 3], [0.1, -0.192, 0.0196], atol=1e-9)


def test_multi_reference_goal_disagreement_is_rejected(monkeypatch) -> None:
    """Execution cannot start when bridge references imply different poses."""
    task = gt_assembly.resolve_single_step_task(_environment())

    def goal_from_connection(env, connection):
        result = np.eye(4)
        if connection.reference_id == 2:
            result[0, 3] = 0.002
        return result

    monkeypatch.setattr(gt_assembly, "_goal_from_connection", goal_from_connection)
    with pytest.raises(ValueError, match="inconsistent multi-reference"):
        gt_assembly.compute_goal_brick_pose(SimpleNamespace(), task)


def test_checked_in_expert_config_preserves_safe_start_contract() -> None:
    """The sole expert keeps the agreed control rate and 60 mm safe start."""
    config = gt_assembly.load_expert_config(
        ROOT / "config/gt_assembly_expert.json"
    )
    assert config.control_hz == 30
    assert config.history_steps == 10
    assert config.max_alignment_rotation_step == pytest.approx(np.deg2rad(4.0))
    assert config.max_alignment_rotation_lead == pytest.approx(np.deg2rad(6.0))
    assert gt_assembly.SAFE_HEIGHT == pytest.approx(0.06)


def test_wrong_snapped_connection_is_reported(monkeypatch) -> None:
    """An active interface with the wrong grid pose fails immediately."""
    core = ModuleType("bricksim.core")
    core.lookup_physics_connection = lambda **kwargs: SimpleNamespace(
        offset=(1, 0), yaw=0
    )
    monkeypatch.setitem(sys.modules, "bricksim.core", core)
    task = gt_assembly.resolve_single_step_task(_environment())
    error = gt_assembly._connection_conflict(task)
    assert error is not None
    assert "actual=(1, 0)/0" in error


def test_grasp_axis_rotates_away_from_side_obstacle() -> None:
    """A neighbor beside local x blocks x grasp while leaving y grasp open."""
    dimensions = {"L": 2, "W": 4, "H": 3}
    # Target occupies x=[-8, 8] mm and y=[-16, 16] mm.  This obstacle
    # touches its +x side and overlaps its y span.
    obstacle_bounds = ((0.008, 0.024, -0.008, 0.008),)
    clearance_x = gt_assembly._grasp_axis_clearance(
        dimensions, obstacle_bounds, 0
    )
    clearance_y = gt_assembly._grasp_axis_clearance(
        dimensions, obstacle_bounds, 1
    )
    assert clearance_x == pytest.approx(0.0)
    assert np.isinf(clearance_y)


def test_grasp_axis_clearance_rejects_unknown_axis() -> None:
    """Only the target brick's local x and y axes are valid choices."""
    with pytest.raises(ValueError, match="grasp_axis"):
        gt_assembly._grasp_axis_clearance(
            {"L": 2, "W": 2, "H": 3}, (), 2
        )


def test_pick_grasp_transform_is_preserved_at_rotated_goal() -> None:
    """Transport preserves brick-to-TCP while applying the goal yaw."""
    pickup = np.eye(4)
    pickup[:3, 3] = [0.15, 0.02, 0.0]
    goal = np.eye(4)
    goal[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    goal[:3, 3] = [0.0, -0.2, 0.03]
    pick_tcp = gt_assembly._grasp_tcp_for_brick(
        pickup,
        grasp_axis=0,
        sign=1.0,
        tcp_height=gt_assembly.PICK_TCP_HEIGHT,
    )
    brick_t_tcp = np.linalg.inv(pickup) @ pick_tcp
    safe_brick = gt_assembly._offset_along_local_z(goal, 0.06)
    safe_tcp = safe_brick @ brick_t_tcp
    np.testing.assert_allclose(
        np.linalg.inv(safe_brick) @ safe_tcp,
        brick_t_tcp,
        atol=1e-12,
    )
    np.testing.assert_allclose(safe_brick[:3, 3], [0.0, -0.2, 0.09])


def test_safe_start_is_directly_above_goal_and_preserves_pickup_yaw() -> None:
    """Preparation changes goal XY/height but leaves yaw for local alignment."""
    goal = np.eye(4)
    goal[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    goal[:3, 3] = [0.0, -0.2, 0.03]
    pickup = np.eye(4)
    pickup[:3, :3] = Rotation.from_euler("z", -25, degrees=True).as_matrix()
    safe = gt_assembly._safe_start_brick_pose(goal, pickup, 0.06)

    local_position = goal[:3, :3].T @ (safe[:3, 3] - goal[:3, 3])
    np.testing.assert_allclose(local_position[:2], [0.0, 0.0], atol=1e-12)
    assert local_position[2] == pytest.approx(0.06)
    np.testing.assert_allclose(safe[:3, :3], pickup[:3, :3], atol=1e-12)


def test_pick_grasp_maps_selected_axis_to_real_finger_motion() -> None:
    """The selected brick axis maps to Piper tool y, its jaw-motion axis."""
    brick = np.eye(4)
    tcp_x = gt_assembly._grasp_tcp_for_brick(
        brick, 0, 1.0, gt_assembly.PICK_TCP_HEIGHT
    )
    tcp_y = gt_assembly._grasp_tcp_for_brick(
        brick, 1, 1.0, gt_assembly.PICK_TCP_HEIGHT
    )
    np.testing.assert_allclose(tcp_x[:3, 1], brick[:3, 0])
    np.testing.assert_allclose(tcp_y[:3, 1], brick[:3, 1])
    np.testing.assert_allclose(tcp_x[:3, 2], -brick[:3, 2])


def test_long_brick_opening_uses_half_width_per_finger() -> None:
    """Opening is tailored to brick width instead of always using full travel."""
    assert gt_assembly._open_gripper_joint_position(0.064) == pytest.approx(
        0.038
    )
    assert gt_assembly._open_gripper_joint_position(0.016) == pytest.approx(
        0.014
    )


def test_transport_seating_limit_scales_with_grasp_width() -> None:
    """Long-axis grasps can seat farther without weakening small-brick checks."""
    assert gt_assembly._transport_jaw_drift_limit(0.008) == pytest.approx(0.004)
    assert gt_assembly._transport_jaw_drift_limit(0.016) == pytest.approx(0.008)
    assert gt_assembly._transport_jaw_drift_limit(0.064) == pytest.approx(0.008)
    assert gt_assembly.GRASP_DRIFT_COMPARISON_TOLERANCE == pytest.approx(0.0002)


def test_fast_alignment_is_disabled_only_for_one_by_two_targets() -> None:
    """A 1x2 target avoids rotation lead while other footprints retain it."""
    assert not gt_assembly._fast_alignment_allowed({"L": 1, "W": 2})
    assert not gt_assembly._fast_alignment_allowed({"L": 2, "W": 1})
    assert gt_assembly._fast_alignment_allowed({"L": 2, "W": 2})
    assert gt_assembly._fast_alignment_allowed({"L": 2, "W": 4})


def test_long_brick_prefers_long_axis_for_yaw_stability() -> None:
    """High-aspect-ratio targets use a wider grasp to resist yaw slip."""
    assert gt_assembly._preferred_grasp_axis({"L": 8, "W": 1, "H": 3}) == 0
    assert gt_assembly._preferred_grasp_axis({"L": 1, "W": 6, "H": 3}) == 1
    assert gt_assembly._preferred_grasp_axis({"L": 4, "W": 2, "H": 3}) == 1


def test_pick_grasp_rejects_non_planar_axis() -> None:
    """The physical pickup uses only the two brick-local side axes."""
    with pytest.raises(ValueError, match="grasp_axis"):
        gt_assembly._grasp_tcp_for_brick(
            np.eye(4), 2, 1.0, gt_assembly.PICK_TCP_HEIGHT
        )


def test_already_aligned_safe_start_does_not_require_pi_hint(monkeypatch) -> None:
    """A zero IK probe means no rotation is needed, not unreachable IK."""
    robot = SimpleNamespace()
    env = SimpleNamespace(
        robot_pins=[robot],
        get_prim_world_T=lambda path: np.eye(4),
    )
    prepared = SimpleNamespace(
        arm_index=0,
        task=SimpleNamespace(target_path="/target"),
        world_t_goal_brick=np.eye(4),
    )
    monkeypatch.setattr(
        gt_assembly, "_arm_configuration", lambda env, arm_index: np.zeros(6)
    )
    monkeypatch.setattr(
        gt_assembly,
        "_tcp_world",
        lambda env, arm_index, q: np.eye(4),
    )
    monkeypatch.setattr(
        gt_assembly,
        "_try_verified_ik",
        lambda robot, target, seed: seed.copy(),
    )

    reachable, hint, branch = gt_assembly._alignment_rotation_hint(
        env, prepared, np.deg2rad(4.0)
    )
    assert reachable
    assert hint is None
    assert branch is not None
    np.testing.assert_allclose(branch.configurations[0], np.zeros(6))


def test_alignment_branch_seed_advances_one_rotation_step() -> None:
    """The runtime chooses the verified seed nearest its next target pose."""
    branch = gt_assembly.AlignmentIKBranch(
        rotation_errors=np.deg2rad(np.array([0.0, 4.0, 8.0])),
        configurations=(
            np.array([0.0, 0.0]),
            np.array([1.0, -2.0]),
            np.array([2.0, -4.0]),
        ),
    )
    seed = gt_assembly._alignment_branch_seed(
        branch,
        np.deg2rad(5.0),
    )
    np.testing.assert_allclose(seed, np.array([1.0, -2.0]))
