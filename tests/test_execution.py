from __future__ import annotations

from rocobrick.execution import AdmissionStatus, ExecutionKernel, PlannedMotion, Trajectory
from rocobrick.gripper import fingers_to_logical_gripper, logical_gripper_to_fingers
from rocobrick.models import ActionType, Candidate, Revisions, RobotSnapshot


class Sink:
    def __init__(self):
        self.sent = []
        self.stopped = []
        self.emergency = False

    def send_trajectory(self, robot_id, trajectory):
        self.sent.append((robot_id, trajectory))

    def controlled_stop(self, robot_id):
        self.stopped.append(robot_id)

    def emergency_stop(self):
        self.emergency = True


def motion(revisions, start=(0.0,) * 6, holding=None):
    candidate = Candidate("c1", "task", ActionType.PICK, "piper_0", revisions, frozenset({"left"}), True)
    trajectory = Trajectory(tuple(f"joint{i}" for i in range(6)), (start, (0.1,) * 6), (0.0, 1.0))
    return PlannedMotion(candidate, start, holding, trajectory, "start", "goal", "scene")


def test_start_mismatch_never_sends_trajectory():
    sink = Sink()
    kernel = ExecutionKernel({"piper_0": RobotSnapshot("piper_0", (0.0,) * 6, 1.0)}, sink)
    result = kernel.admit(motion(kernel.revisions, start=(0.2,) * 6))
    assert result.status == AdmissionStatus.START_STATE_MISMATCH
    assert sink.sent == []


def test_world_change_invalidates_candidate():
    sink = Sink()
    kernel = ExecutionKernel({"piper_0": RobotSnapshot("piper_0", (0.0,) * 6, 1.0)}, sink)
    planned = motion(kernel.revisions)
    kernel.world_changed()
    assert kernel.admit(planned).status == AdmissionStatus.STALE_WORLD
    assert sink.sent == []


def test_matching_candidate_is_sent_once():
    sink = Sink()
    kernel = ExecutionKernel({"piper_0": RobotSnapshot("piper_0", (0.0,) * 6, 1.0)}, sink)
    assert kernel.admit(motion(kernel.revisions)).status == AdmissionStatus.ACCEPTED
    assert len(sink.sent) == 1


def test_official_logical_gripper_maps_to_two_sim_fingers():
    assert logical_gripper_to_fingers(0.08) == (0.04, -0.04)
    assert fingers_to_logical_gripper(0.04, -0.04) == 0.08
