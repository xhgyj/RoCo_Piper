"""Pure invariance checks for structure-independent geometry planning."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rocobrick.execution import HeldObject, HeldObjectState
from rocobrick.planning.geometry import obb_intersects
from rocobrick.planning.models import GraspRegion, ObjectGeometry, OrientedBox
from rocobrick.planning.planners import _adaptive_ik_path
from rocobrick.policy.bricksim_grounder import BrickSimActionGrounder


def _pose(x: float, y: float, yaw: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    pose[:3, 3] = [x, y, 0.0]
    return pose


class _BranchingRobot:
    """IK double that would change branch even for the seed pose."""

    def __init__(self) -> None:
        self.solve_calls = 0

    @property
    def arm_configuration_indices(self) -> tuple[int, ...]:
        """Expose one controlled joint."""
        return (0,)

    def solve_ik(self, world_t_tcp: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Return a different branch on every unnecessary IK invocation."""
        del world_t_tcp
        self.solve_calls += 1
        return seed + 0.05

    def configuration_is_safe(self, q: np.ndarray) -> bool:
        """Accept finite test configurations.

        Returns:
            Whether every test joint value is finite.
        """
        return bool(np.isfinite(q).all())


def test_obb_collision_is_invariant_to_world_rigid_transform_and_names() -> None:
    """Collision depends on relative geometry, not IDs or world alignment."""
    left = OrientedBox("long_brick", _pose(0.0, 0.0, 0.4), np.array([0.03, 0.01, 0.01]))
    right = OrientedBox("wall", _pose(0.025, 0.012, -0.2), np.array([0.01, 0.02, 0.02]))
    expected = obb_intersects(left, right)
    world_t_shift = np.eye(4)
    world_t_shift[:3, :3] = Rotation.from_euler("z", 1.1).as_matrix()
    world_t_shift[:3, 3] = [0.7, -0.3, 0.2]
    renamed_left = OrientedBox(
        "arbitrary_a", world_t_shift @ left.world_t_box, left.half_extents
    )
    renamed_right = OrientedBox(
        "arbitrary_b", world_t_shift @ right.world_t_box, right.half_extents
    )
    assert obb_intersects(renamed_left, renamed_right) is expected


def test_held_state_accumulates_drift_without_rebasing() -> None:
    """Several small slips remain visible as cumulative grasp drift."""
    held = HeldObject("brick", "robot", np.eye(4), 0, 0.016)
    state = HeldObjectState.from_pick(held)
    for x in (0.001, 0.002, 0.003):
        observed = np.eye(4)
        observed[0, 3] = x
        state.observe(observed)
    assert np.isclose(state.cumulative_position_drift, 0.003)
    assert np.allclose(state.acquisition_object_t_tcp, np.eye(4))


def test_object_collision_frame_is_independent_from_grasp_region() -> None:
    """Asset-specific grasp calibration does not move the collision OBB."""
    object_pose = _pose(0.3, -0.1, 0.4)
    object_t_collision = np.eye(4)
    object_t_collision[2, 3] = 0.01
    object_t_grasp = np.eye(4)
    object_t_grasp[2, 3] = -0.004
    geometry = ObjectGeometry(
        "part",
        object_pose,
        object_t_collision,
        np.array([0.02, 0.01, 0.005]),
        (
            GraspRegion(
                object_t_grasp,
                np.array([0.02, 0.01, 0.003]),
            ),
        ),
    )
    assert np.allclose(
        geometry.collision_box().world_t_box,
        object_pose @ object_t_collision,
    )
    assert not np.allclose(object_t_collision, object_t_grasp)


def test_bricksim_goal_id_is_scoped_to_the_exact_target() -> None:
    """Two target IDs cannot silently resolve to the same assembly goal."""
    first = BrickSimActionGrounder.assembly_goal_id("/World/Part_1")
    second = BrickSimActionGrounder.assembly_goal_id("/World/Part_2")
    assert first != second
    assert first == "bricksim:place_down:/World/Part_1"


def test_adaptive_ik_preserves_known_start_configuration() -> None:
    """The path start must not be re-solved onto another IK branch."""
    robot = _BranchingRobot()
    start = _pose(0.0, 0.0, 0.0)
    goal = _pose(0.01, 0.0, 0.0)
    seed = np.array([0.2])
    path = _adaptive_ik_path(robot, (start, goal), seed, "test", 0.1, 3)
    assert robot.solve_calls == 1
    assert np.array_equal(path.configurations[0], seed)
