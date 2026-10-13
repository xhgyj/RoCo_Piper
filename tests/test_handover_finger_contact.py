"""Contact policy/physics history tests; no controllers or simulator started."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rocobrick.adapters.isaac.pad_contacts import (
    PhysxPadContactReader,
    finger_history_document,
)
from rocobrick.adapters.moveit.handover_live import LiveSimulationHandover
from rocobrick.skills.handover_contact import (
    CONTACT_SCHEMA,
    FingerContactGuard,
    SimulationFingerContactPolicy,
)
from rocobrick.skills.handover_profile import EXECUTION_CAPABILITIES
from rocobrick.skills.handover_simulation import SimulationHandoverAuthorization
from rocobrick.skills.observation import Pose


def guard():
    """Construct a synthetic identity-bound guard.

    Returns:
        A guard with no hardware or simulator access.
    """
    return FingerContactGuard(
        session_id="sim",
        manifest_sha256="a" * 64,
        donor="piper_0",
        receiver="piper_1",
        policy=SimulationFingerContactPolicy(),
        bounds=((-0.01, -0.01, -0.005), (0.01, 0.01, 0.005)),
    )


def contact(**values):
    """Build one identified physical-contact fixture.

    Returns:
        A named pair's contact record with optional overrides.
    """
    item = dict(
        link_a="piper_0_gripper_link1",
        link_b="piper_1_gripper_link2",
        position_world_m=[0, 0, 0],
        normal_force_n=0.2,
        impulse_ns=0.0002,
        separation_m=-0.0001,
    )
    item.update(values)
    return item


def snapshot(time=1.0, *, contacts=(), frames=None, sequence=1):
    """Build a complete step history at one scene tick.

    Returns:
        An explicitly synthetic scene/history fixture.
    """
    frames = frames or [dict(sim_time_s=time, dt_s=0.01, contacts=list(contacts))]
    report = dict(
        schema=CONTACT_SCHEMA,
        source="physx_contact_report",
        session_id="sim",
        manifest_sha256="a" * 64,
        sim_time_s=time,
        sequence=sequence,
        frames=frames,
    )
    return dict(sim_time_s=time, sequence=sequence, finger_contact_history=report)


def inspect(g, raw, phase="approach", shift=0):
    """Evaluate a fixture with separately supplied actual pose observations."""
    g.inspect(
        raw,
        phase=phase,
        object_pose=Pose((0, 0, 0), (0, 0, 0, 1)),
        tcp_poses={
            r: Pose((shift, 0, 0), (0, 0, 0, 1)) for r in ("piper_0", "piper_1")
        },
    )


def test_contact_only_in_named_window_and_not_custody():
    """Collision evidence does not replace brick holding evidence."""
    g = guard()
    inspect(g, snapshot(contacts=[contact()]))
    assert g.summary["peak_pair_force_n"] == 0.2
    with pytest.raises(ValueError, match="remains"):
        g.require_clear()
    inspect(g, snapshot(1.01, sequence=2))
    g.require_clear()


@pytest.mark.parametrize(
    "phase", ["free", "receiver_open", "rendezvous", "receiver_follow"]
)
def test_free_phases_never_allow_finger_contact(phase):
    """Free-space phases keep their unconditional inter-arm collision rule."""
    with pytest.raises(ValueError, match="unpermitted"):
        inspect(guard(), snapshot(contacts=[contact()]), phase)


@pytest.mark.parametrize(
    "values,reason",
    [
        ({"link_b": "piper_1_link6"}, "unpermitted"),
        ({"link_b": "/table"}, "unpermitted"),
        ({"normal_force_n": 3.0}, "force/impulse"),
        ({"impulse_ns": 0.02}, "force/impulse"),
        ({"separation_m": -0.0006}, "penetration"),
        ({"position_world_m": [0, 0, 0.009]}, "region"),
        ({"normal_force_n": float("nan")}, "finite"),
    ],
)
def test_uncontrolled_contact_is_rejected(values, reason):
    """Reject foreign pairs, excessive forces/depth and out-of-window points."""
    with pytest.raises(ValueError, match=reason):
        inspect(guard(), snapshot(contacts=[contact(**values)]))


def test_force_is_summed_across_contact_points():
    """Several small contacts cannot evade the pair-level force cap."""
    with pytest.raises(ValueError, match="force/impulse"):
        inspect(
            guard(),
            snapshot(
                contacts=[contact(normal_force_n=1.1), contact(normal_force_n=1.1)]
            ),
        )


def test_brief_collision_between_queries_cannot_disappear():
    """Retain an excessive transient even when the latest frame is clear."""
    g = guard()
    inspect(g, snapshot())
    raw = snapshot(
        1.02,
        sequence=2,
        frames=[
            dict(sim_time_s=1.0, dt_s=0.01, contacts=[]),
            dict(sim_time_s=1.01, dt_s=0.01, contacts=[contact(normal_force_n=9)]),
            dict(sim_time_s=1.02, dt_s=0.01, contacts=[]),
        ],
    )
    with pytest.raises(ValueError, match="force/impulse"):
        inspect(g, raw)


@pytest.mark.parametrize(
    "field", ["source", "sequence", "manifest_sha256", "session_id"]
)
def test_foreign_or_stale_reports_fail_closed(field):
    """Borrowed or stale reports cannot authorize a contact motion."""
    raw = snapshot()
    raw["finger_contact_history"][field] = "wrong"
    with pytest.raises(ValueError, match="foreign"):
        inspect(guard(), raw)


def test_history_gap_and_observer_gap_are_errors():
    """Missing physics steps or delayed observers fail closed."""
    g = guard()
    inspect(g, snapshot())
    with pytest.raises(ValueError, match="lost intervening"):
        inspect(g, snapshot(1.02, sequence=2))
    with pytest.raises(ValueError, match="observer gap"):
        inspect(g, snapshot(1.3, sequence=2))


def test_free_phase_observer_gap_rebases_before_contact():
    """Slow preparation polling cannot block the first contact motion."""
    g = guard()
    inspect(g, snapshot(1.0), phase="free")
    inspect(g, snapshot(1.3, sequence=2), phase="rendezvous")
    # The first contact sample establishes a fresh continuity baseline.
    inspect(g, snapshot(1.31, sequence=3), phase="approach")


def test_contact_gap_remains_strict_after_preparation_rebase():
    """Preparation rebasing does not weaken continuity during contact."""
    g = guard()
    inspect(g, snapshot(1.0), phase="free")
    inspect(g, snapshot(1.3, sequence=2), phase="rendezvous")
    inspect(g, snapshot(1.31, sequence=3), phase="approach")
    with pytest.raises(ValueError, match="observer gap"):
        inspect(g, snapshot(1.6, sequence=4), phase="receiver_close")


@pytest.mark.parametrize("extra_render_steps", [False, True])
def test_six_step_headed_history_requires_every_physics_tick(extra_render_steps):
    """Render ticks cannot be silently missing from a purported 60 Hz history."""
    dt = 1.0 / 60.0
    time = 0.0
    frames = []
    for index in range(30):
        time += dt
        frames.append(dict(sim_time_s=time, dt_s=dt, contacts=[]))
        if extra_render_steps and index % 6 == 0:
            time += dt
    raw = snapshot(frames[-1]["sim_time_s"], frames=frames)
    if extra_render_steps:
        with pytest.raises(ValueError, match="physics history is discontinuous"):
            inspect(guard(), raw)
    else:
        inspect(guard(), raw)


def test_measured_contact_speed_is_guarded():
    """Command retiming is not a substitute for measured speed admission."""
    g = guard()
    inspect(g, snapshot())
    with pytest.raises(ValueError, match="speed"):
        inspect(g, snapshot(1.01, sequence=2), shift=0.001)


def test_sustained_contact_span_has_a_bound():
    """Persistent pushing cannot continue indefinitely."""
    g = guard()
    g.policy = replace(g.policy, max_contact_span_s=0.015)
    inspect(g, snapshot(contacts=[contact()]))
    with pytest.raises(ValueError, match="sustained"):
        inspect(g, snapshot(1.01, contacts=[contact()], sequence=2))


def test_collision_report_is_separate_from_holding_report():
    """Finger/finger events never enter pad/object holding contact reports."""
    headers = [
        SimpleNamespace(
            type="found",
            actor0="/giver",
            actor1="/receiver",
            contact_data_offset=0,
            num_contact_data=1,
        )
    ]
    points = [
        SimpleNamespace(
            position=(0, 0, 0),
            normal=(0, 1, 0),
            impulse=(0, 0.001, 0),
            separation=-0.0001,
        )
    ]
    reader = PhysxPadContactReader(
        SimpleNamespace(get_contact_report=lambda: (headers, points)),
        decode_path=str,
        pads={
            "/giver": ("piper_0", "piper_0_gripper_link1"),
            "/receiver": ("piper_1", "piper_1_gripper_link2"),
        },
        objects={"/brick": "brick"},
        contact_lost="lost",
        include_finger_contacts=True,
    )
    frame = reader.capture(sim_time_s=1, dt_s=0.01)
    assert frame.contacts == ()
    assert len(frame.finger_contacts) == 1
    doc = finger_history_document(
        [frame], sim_time_s=1, sequence=1, session_id="sim", manifest_sha256="a" * 64
    )
    raw = dict(sim_time_s=1, sequence=1, finger_contact_history=deepcopy(doc))
    inspect(guard(), raw)
    assert (
        frame.document(
            sim_time_s=1, sequence=1, session_id="sim", manifest_sha256="a" * 64
        )["contacts"]
        == []
    )


@pytest.mark.parametrize("bad", ["missing", "force", None])
def test_live_observer_actually_enforces_finger_guard(bad):
    """Exercise the concrete observer without constructing a live session."""
    from test_handover_joint_planning import planning_request
    from test_handover_joint_planning import snapshot as paired_snapshot

    request = planning_request()
    raw = paired_snapshot()
    policy = SimulationFingerContactPolicy()
    token = SimulationHandoverAuthorization(
        raw["session_id"],
        raw["manifest_sha256"],
        request.profile.sha256,
        request.profile.runtime_model_sha256,
        policy.sha256,
    )
    live = object.__new__(LiveSimulationHandover)
    live.authorization, live.finger_policy = token, policy
    live.session = SimpleNamespace(
        observe_handover_scene=lambda: raw
    )
    live.phases = {request.operation_id: "approach"}
    live.contact_clear_required = set()
    live.contact_guards = {
        request.operation_id: FingerContactGuard(
            session_id=token.session_id,
            manifest_sha256=token.manifest_sha256,
            donor=request.donor.robot_id,
            receiver=request.receiver.robot_id,
            policy=policy,
            bounds=(
                request.object_geometry.minimum_m,
                request.object_geometry.maximum_m,
            ),
        )
    }
    program = SimpleNamespace(
        request=request,
        capabilities=EXECUTION_CAPABILITIES,
        simulation_authorization=token,
        identity=SimpleNamespace(
            operation_id=request.operation_id,
            donor_robot_id=request.donor.robot_id,
            receiver_robot_id=request.receiver.robot_id,
            physical_instance_id=request.physical_instance_id,
        ),
    )
    if bad != "missing":
        raw["finger_contact_history"] = {
            "schema": CONTACT_SCHEMA,
            "source": "physx_contact_report",
            "session_id": token.session_id,
            "manifest_sha256": token.manifest_sha256,
            "sequence": raw["sequence"],
            "sim_time_s": raw["sim_time_s"],
            "frames": [
                {
                    "sim_time_s": raw["sim_time_s"],
                    "dt_s": 0.01,
                    "contacts": [
                        contact(position_world_m=[0, 0, 0.2], normal_force_n=5)
                    ]
                    if bad == "force"
                    else [],
                }
            ],
        }
    if bad is None:
        assert asyncio.run(live.observe(program)) is raw
    else:
        with pytest.raises(ValueError):
            asyncio.run(live.observe(program))


@pytest.mark.asyncio
async def test_contact_guard_failure_cancels_and_never_opens_giver():
    """A real guard exception reaches the existing physical cancel lifecycle."""
    from test_coordinated_handover_lifecycle import fixture, operation

    p, ledger, transport, backend = fixture()
    observe = transport.observe
    g = FingerContactGuard(
        session_id=p.request.scene_snapshot["session_id"],
        manifest_sha256=p.request.scene_snapshot["manifest_sha256"],
        donor=p.identity.donor_robot_id,
        receiver=p.identity.receiver_robot_id,
        policy=SimulationFingerContactPolicy(),
        bounds=(
            p.request.object_geometry.minimum_m,
            p.request.object_geometry.maximum_m,
        ),
    )

    async def guarded_observe(program):
        """Inject excessive PhysX-shaped contact into a synthetic action.

        Returns:
            The synthetic observation unless the real contact guard rejects it.
        """
        raw = await observe(program)
        if transport.phase == "approach":
            raw["finger_contact_history"] = {
                "schema": CONTACT_SCHEMA,
                "source": "physx_contact_report",
                "session_id": raw["session_id"],
                "manifest_sha256": raw["manifest_sha256"],
                "sequence": raw["sequence"],
                "sim_time_s": raw["sim_time_s"],
                "frames": [
                    {
                        "sim_time_s": raw["sim_time_s"],
                        "dt_s": 0.1,
                        "contacts": [contact(normal_force_n=20)],
                    }
                ],
            }
            g.inspect(
                raw,
                phase="approach",
                object_pose=Pose((0, 0, 0), (0, 0, 0, 1)),
                tcp_poses={r: Pose((0, 0, 0), (0, 0, 0, 1)) for r in p.participants},
            )
        return raw

    transport.observe = guarded_observe
    result = await backend.execute(operation(p))
    assert not result.passed and transport.cancelled
    assert "donor_open" not in transport.calls
    assert ledger.owner_of_object("brick-1") == p.identity.donor_robot_id
    assert ledger.observe(p.identity.receiver_robot_id, "holding:brick-1") is None
