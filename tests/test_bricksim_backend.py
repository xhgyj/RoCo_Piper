"""Pure checks for the BrickSim robot backend's per-arm dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rocobrick.backends.bricksim import BrickSimRobotBackend


class _PinModel:
    def __init__(self) -> None:
        self.joints = {
            "joint1": SimpleNamespace(idx_q=0),
            "gripper_joint1": SimpleNamespace(idx_q=1),
        }

    def getJointId(self, name: str) -> str:  # noqa: N802 -- Pinocchio API.
        """Use the joint name as the fake model identifier.

        Returns:
            The unchanged fake joint identifier.
        """
        return name


def test_command_configuration_updates_only_assigned_arm() -> None:
    """Concurrent backends must not dispatch a stale global robot command."""
    calls = []
    pin = SimpleNamespace(
        pin_model=_PinModel(),
        home_q=np.zeros(2),
        ee_frames=("tcp",),
    )
    env = SimpleNamespace(
        robot_pins=[pin, pin],
        robot_configs=[
            {
                "Name": "piper_0",
                "Joint_Order": ["joint1", "gripper_joint1"],
            },
            {
                "Name": "piper_1",
                "Joint_Order": ["joint1", "gripper_joint1"],
            },
        ],
        robot_apply_arm_action=lambda arm_index, values: calls.append(
            (arm_index, values.copy())
        ),
    )
    backend = BrickSimRobotBackend(env, arm_index=1)

    backend.command_configuration(np.array([0.25, 0.01]))

    assert len(calls) == 1
    assert calls[0][0] == 1
    np.testing.assert_allclose(calls[0][1], [0.25, 0.01])
