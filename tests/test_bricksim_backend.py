import asyncio

import pytest

from rocobrick.bricksim_backend import BrickSimTrajectoryPort, RobotBinding


class Robot:
    dof_names = ["joint1", "gripper_joint1", "joint2", "gripper_joint2"]

    def __init__(self):
        self.positions = [0.1, 0.04, 0.2, -0.04]
        self.actions = []

    def get_joint_positions(self):
        return self.positions

    def apply_action(self, action):
        self.actions.append(action)


class World:
    current_time = 1.25


class App:
    def __init__(self):
        self.steps = 0

    async def next_update_async(self):
        self.steps += 1


def factory(positions, indices):
    return positions, indices


def port(robot=None):
    robot = robot or Robot()
    backend = BrickSimTrajectoryPort(
        App(),
        World(),
        {"piper_0": RobotBinding(robot, ("joint1", "joint2"))},
        lambda robot_id: "part-a",
        lambda robot_id: True,
        action_factory=factory,
        monotonic=lambda: 10.0,
    )
    return backend, robot


def test_measured_state_uses_configured_joint_order_not_articulation_order():
    backend, _ = port()
    state = backend.measured_state("piper_0")
    assert state.joints == (0.1, 0.2)
    assert state.holding_part == "part-a"
    assert state.simulation_time_s == 1.25


def test_prefixed_moveit_names_map_to_isaac_dofs_exactly():
    backend, robot = port()
    backend.command_joint_positions(
        "piper_0", ("piper_0_joint1", "piper_0_joint2"), (0.3, 0.4)
    )
    assert robot.actions == [((0.3, 0.4), (0, 2))]


def test_wrong_joint_order_is_rejected():
    backend, _ = port()
    with pytest.raises(ValueError, match="exactly equal"):
        backend.command_joint_positions(
            "piper_0", ("piper_0_joint2", "piper_0_joint1"), (0.3, 0.4)
        )


def test_controlled_stop_holds_latest_measured_position():
    backend, robot = port()
    backend.controlled_stop("piper_0")
    assert robot.actions == [((0.1, 0.2), (0, 2))]


def test_gripper_commands_and_measures_both_fingers():
    backend, robot = port()
    assert backend.measured_gripper_opening("piper_0") == pytest.approx(0.08)
    backend.command_gripper_opening("piper_0", 0.03)
    assert robot.actions == [((0.015, -0.015), (1, 3))]


def test_loaded_gripper_contact_may_be_asymmetric():
    backend, robot = port()
    robot.positions[1] = 0.012
    robot.positions[3] = -0.004
    assert backend.measured_gripper_fingers("piper_0") == (0.012, -0.004)
    assert backend.measured_gripper_opening("piper_0") == pytest.approx(0.016)


@pytest.mark.asyncio
async def test_step_advances_injected_simulation_application():
    backend, _ = port()
    await backend.step()
    assert backend._app.steps == 1
