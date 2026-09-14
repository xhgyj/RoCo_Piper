import pytest

from rocobrick.execution import Trajectory
from rocobrick.trajectory_execution import MeasuredState, PlaybackStatus, play_trajectory


class Port:
    def __init__(self, holding=None):
        self.time = 0.0
        self.joints = (0.0, 0.0)
        self.command = self.joints
        self.holding = holding
        self.safe = True
        self.feedback_age = 0.0
        self.stops = 0
        self.steps = 0

    def measured_state(self, robot_id):
        return MeasuredState(self.time, self.joints, self.feedback_age, self.holding, self.safe)

    def command_joint_positions(self, robot_id, joint_names, positions):
        self.command = positions

    async def step(self):
        self.time += 0.1
        self.joints = self.command
        self.steps += 1

    def controlled_stop(self, robot_id):
        self.stops += 1


def trajectory():
    return Trajectory(("j1", "j2"), ((0.0, 0.0), (0.1, -0.1)), (0.01, 0.5))


@pytest.mark.asyncio
async def test_simulator_executes_trajectory_from_measured_feedback():
    port = Port()
    result = await play_trajectory(port, "piper_0", trajectory(), expected_holding_part=None)
    assert result.status == PlaybackStatus.COMPLETED
    assert port.steps > 1
    assert port.stops == 0


@pytest.mark.asyncio
async def test_holding_slip_stops_remaining_motion():
    port = Port("part-a")

    original_step = port.step

    async def slip():
        await original_step()
        port.holding = None

    port.step = slip
    result = await play_trajectory(port, "piper_0", trajectory(), expected_holding_part="part-a")
    assert result.status == PlaybackStatus.HOLDING_FAILED
    assert port.stops == 1


@pytest.mark.asyncio
async def test_stale_feedback_stops_before_commanding():
    port = Port()
    port.feedback_age = 0.3
    result = await play_trajectory(port, "piper_0", trajectory(), expected_holding_part=None)
    assert result.status == PlaybackStatus.FEEDBACK_EXPIRED
    assert port.steps == 0
    assert port.stops == 1


@pytest.mark.asyncio
async def test_tracking_error_stops_motion():
    port = Port()
    port.joints = (0.2, 0.2)
    result = await play_trajectory(port, "piper_0", trajectory(), expected_holding_part=None)
    assert result.status == PlaybackStatus.TRACKING_FAILED
    assert port.stops == 1
