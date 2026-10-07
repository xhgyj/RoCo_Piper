"""Read-only physical monitoring while action/resource ownership is active."""

from types import SimpleNamespace

import pytest
from test_handover_joint_planning import snapshot
from test_persistent_ros_worker import TOKEN, FakeHandle, pump

from rocobrick.execution.ros_session_worker import SessionWorker
from rocobrick.execution.ros_worker_transport import RosWorkerTransport


def fixture(tmp_path, *, defect=None):
    """Allow ONLY physics observation, not planning/version or controllers.

    Returns:
        The identified, bound result; physical proof remains separately checked.
    """
    scene = snapshot(2)
    if defect == "robots":
        scene["robots"].pop("piper_1")
    session = SimpleNamespace(
        observe_all=lambda: scene,
        latest_shared_sim_time_s=lambda: 3 if defect == "stale" else 1.21,
    )
    worker = SessionWorker(tmp_path, TOKEN, ("piper_0", "piper_1"), session)
    worker.owners = {"piper_0": "active-coordinated", "piper_1": "active-coordinated"}
    worker.unsafe = True  # Monitoring must remain available during failed motion.
    transport = RosWorkerTransport(tmp_path, TOKEN, FakeHandle())
    transport.active["active-gripper"] = ("owned", None, tmp_path / "pending.json")
    transport._wait = lambda path, timeout: pump(worker, path.is_file)
    return worker, transport


def test_monitor_does_not_require_idle_and_never_changes_action_leases(tmp_path):
    """Verify test monitor does not require idle and never changes action leases."""
    """Verify test monitor rejects foreign session robot set."""
    worker, transport = fixture(tmp_path)
    owners = dict(worker.owners)
    result = transport.observe_monitor(worker.robots, 1)
    assert result["sequence"] == 2
    assert worker.owners == owners
    assert worker.seen_operations == set()
    assert "planning_scene_version" not in result
    assert transport.active


@pytest.mark.parametrize("defect", ["robots", "stale"])
def test_monitor_rejects_stale_or_incomplete_physics(tmp_path, defect):
    """Verify test monitor rejects stale or incomplete physics."""
    worker, transport = fixture(tmp_path, defect=defect)
    with pytest.raises(ValueError):
        transport.observe_monitor(worker.robots, 1)


def test_monitor_rejects_foreign_session_robot_set(tmp_path):
    """Bind the monitor to exactly the robots of its owning session."""
    worker, transport = fixture(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        transport.observe_monitor(("piper_0",), 1)
