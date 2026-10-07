"""Actual session clock-alignment logic without ROS, Isaac or motion."""

from types import SimpleNamespace

import pytest

from rocobrick.ros_moveit_runtime import RosMoveItRobotSession


def session_fixture(samples):
    """Provide controlled received clock callbacks and an injected wall clock.

    Returns:
        The actual session class with purely synthetic callback inputs.
    """
    session = RosMoveItRobotSession.__new__(RosMoveItRobotSession)
    state = SimpleNamespace(wall=0.0, index=0, spins=0)
    session._clock_fn = lambda: state.wall
    session._poll_interval_s = 0.01

    def spin():
        state.spins += 1
        state.index = min(state.index + 1, len(samples) - 1)

    def sleep(seconds):
        state.wall += seconds

    session._spin_fn = spin
    session._sleep_fn = sleep
    session.runtimes = {
        "piper_0": SimpleNamespace(
            clock_values=lambda: (int(samples[state.index] * 1e9),)
        )
    }
    return session, state


def test_snapshot_ahead_by_50ms_waits_for_received_clock_not_fabricated_time():
    """Replay the observed 64.450 versus 64.400 scene/clock race."""
    session, state = session_fixture([64.4, 64.433333, 64.466667])
    session._wait_for_snapshot_clock(64.450003361, 0.1)
    assert state.spins == 2
    assert session.latest_shared_sim_time_s() == pytest.approx(64.466667)
    assert 0 < state.wall < 0.1


def test_current_clock_needs_no_spin_or_delay():
    """Already ordered samples do not add latency."""
    session, state = session_fixture([64.5])
    session._wait_for_snapshot_clock(64.45, 0.1)
    assert state.spins == 0 and state.wall == 0


def test_stalled_clock_times_out_without_advancing_observed_time():
    """A paused or disconnected simulator is not fresh observation evidence."""
    session, state = session_fixture([64.4])
    with pytest.raises(TimeoutError, match="did not reach snapshot"):
        session._wait_for_snapshot_clock(64.45, 0.025)
    assert state.wall == pytest.approx(0.025)
    assert session.latest_shared_sim_time_s() == pytest.approx(64.4)


def test_clock_reset_fails_closed():
    """A restarted timeline cannot admit a previous scene snapshot."""
    session, _ = session_fixture([64.4, 0.01])
    with pytest.raises(RuntimeError, match="backwards"):
        session._wait_for_snapshot_clock(64.45, 0.1)


def test_alignment_does_not_relax_caller_stale_scene_gate():
    """Clock alignment cannot make an old snapshot fresh."""
    session, state = session_fixture([70.0])
    session._wait_for_snapshot_clock(64.45, 0.1)
    assert session.latest_shared_sim_time_s() - 64.45 > 0.5
    assert state.spins == 0
