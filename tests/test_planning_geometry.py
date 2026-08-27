"""Pure invariance checks for structure-independent geometry planning."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rocobrick.backends.base import RobotState
from rocobrick.execution import HeldObject, HeldObjectState
from rocobrick.planning import geometry as geometry_module
from rocobrick.planning.geometry import obb_intersects, path_clearance
from rocobrick.planning.models import (
    GraspRegion,
    GripperGeometry,
    ObjectGeometry,
    OrientedBox,
    SceneGeometry,
)
from rocobrick.planning.planners import GraspPlanner, PlannerConfig, _adaptive_ik_path
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


class _LimitedIkRobot:
    """Robot double that permits only one three-endpoint probe chain."""

    def __init__(self) -> None:
        self.solve_calls = 0
        self._q = np.zeros(8)
        self._q[2] = 0.10

    @property
    def robot_id(self) -> str:
        """Return a stable test identifier."""
        return "limited"

    @property
    def arm_configuration_indices(self) -> tuple[int, ...]:
        """Expose the six arm joints."""
        return (0, 1, 2, 3, 4, 5)

    def read_state(self) -> RobotState:
        """Return the fixed initial state.

        Returns:
            Current test configuration and TCP pose.
        """
        return RobotState(
            self._q.copy(),
            self.forward_kinematics(self._q),
            self._q[6:].copy(),
        )

    def solve_ik(self, world_t_tcp: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        """Solve only the first candidate's three endpoint probes.

        Returns:
            Pose-encoded configuration, or None after three calls.
        """
        self.solve_calls += 1
        if self.solve_calls > 3:
            return None
        result = seed.copy()
        result[:3] = world_t_tcp[:3, 3]
        result[3:6] = Rotation.from_matrix(world_t_tcp[:3, :3]).as_rotvec()
        return result

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Decode a test configuration into a TCP pose.

        Returns:
            World-frame pose represented by the first six joints.
        """
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        pose[:3, 3] = q[:3]
        return pose

    def configuration_is_safe(self, q: np.ndarray) -> bool:
        """Accept finite configurations.

        Returns:
            Whether all joint values are finite.
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


def test_adaptive_ik_reuses_verified_goal_configuration() -> None:
    """A cached endpoint avoids solving the same endpoint a second time."""
    robot = _BranchingRobot()
    start = _pose(0.0, 0.0, 0.0)
    goal = _pose(0.01, 0.0, 0.0)
    seed = np.array([0.2])
    verified_goal = np.array([0.25])
    path = _adaptive_ik_path(
        robot,
        (start, goal),
        seed,
        "test",
        0.1,
        3,
        verified_goal,
    )
    assert robot.solve_calls == 0
    assert np.array_equal(path.configurations[-1], verified_goal)


def test_grasp_planner_stops_after_first_fully_feasible_candidate() -> None:
    """Lower-ranked candidates do not consume IK after a complete success."""
    robot = _LimitedIkRobot()
    extents = np.array([0.008, 0.016, 0.006])
    target = ObjectGeometry(
        "brick",
        np.eye(4),
        np.eye(4),
        extents,
        (GraspRegion(np.eye(4), extents),),
    )
    planner = GraspPlanner(PlannerConfig(pick_clearance=0.002, precision_ik_step=0.002))
    plan = planner.plan(robot, SceneGeometry(target, ()))
    assert plan.robot_id == robot.robot_id
    assert robot.solve_calls == 3


def test_path_clearance_evaluates_sat_once_per_box_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearance and intersection share one separating-axis computation."""
    calls = 0
    original = geometry_module.obb_axis_separations

    def counted(left: OrientedBox, right: OrientedBox) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(geometry_module, "obb_axis_separations", counted)
    obstacle = OrientedBox(
        "far",
        _pose(1.0, 0.0, 0.0),
        np.array([0.01, 0.01, 0.01]),
    )
    clearance = path_clearance(
        (np.eye(4),),
        0.02,
        GripperGeometry(),
        (obstacle,),
    )
    assert clearance is not None
    assert calls == 3
