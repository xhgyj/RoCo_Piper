"""Physics contact conversion regressions without importing Isaac or PhysX."""

from types import SimpleNamespace

import pytest

from rocobrick.adapters.isaac.pad_contacts import PhysxPadContactReader


def reader(report):
    """Return a mapped current-step reader with an injected native buffer."""
    return PhysxPadContactReader(
        SimpleNamespace(get_contact_report=lambda: report),
        decode_path=str,
        pads={"/robot/gripper_link1": ("piper_0", "piper_0_gripper_link1")},
        objects={"/brick": "brick-1"},
        contact_lost="lost",
    )


def header(*, reversed=False, kind="found", **overrides):
    """Return an actual-header-shaped record with exact actor identities."""
    actors = ("/robot/gripper_link1", "/brick")
    if reversed:
        actors = actors[::-1]
    fields = dict(
        type=kind,
        actor0=actors[0],
        actor1=actors[1],
        contact_data_offset=0,
        num_contact_data=1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def point(sign=1):
    """Return native-point-shaped SI impulse, position and unit normal."""
    return SimpleNamespace(
        position=(0.0, 0.006, 0.2),
        normal=(0.0, sign, 0.0),
        impulse=(0.0, sign * 0.002, 0.0),
    )


@pytest.mark.parametrize("reversed", [False, True])
def test_actual_impulse_and_actor_order_are_normalized(reversed):
    """Normal direction stays object-to-pad and impulse becomes force."""
    frame = reader(
        ([header(reversed=reversed)], [point(-1 if reversed else 1)])
    ).capture(
        sim_time_s=1.0,
        dt_s=0.002,
    )
    assert len(frame.contacts) == 1
    contact = frame.contacts[0]
    assert contact.normal_world == (0.0, 1.0, 0.0)
    assert contact.normal_force_n == pytest.approx(1.0)
    doc = frame.document(
        sim_time_s=1.0, sequence=3, session_id="run", manifest_sha256="a" * 64
    )
    assert doc["source"] == "physx_contact_report"
    assert doc["sequence"] == 3
    with pytest.raises(ValueError, match="different physics step"):
        frame.document(
            sim_time_s=1.002, sequence=4, session_id="run", manifest_sha256="a" * 64
        )


@pytest.mark.parametrize("kind", ["lost", "unknown-object", "zero-force", "empty"])
def test_no_cached_or_nonphysical_contact_is_emitted(kind):
    """Lost/unknown/force-free contacts cannot be promoted into holding proof."""
    h, p = header(), point()
    if kind == "lost":
        h.type = "lost"
    elif kind == "unknown-object":
        h.actor1 = "/brick/another_actor"
    elif kind == "zero-force":
        p.impulse = (0.0, 0.0, 0.0)
    report = ([], []) if kind == "empty" else ([h], [p])
    assert reader(report).capture(sim_time_s=2.0, dt_s=0.002).contacts == ()


@pytest.mark.parametrize(
    "values",
    [
        {"contact_data_offset": -1},
        {"num_contact_data": 2},
        {"num_contact_data": True},
    ],
)
def test_malformed_native_contact_range_fails_closed(values):
    """Missing or invalid buffer ranges never become an empty successful read."""
    with pytest.raises(ValueError, match="range"):
        reader(([header(**values)], [point()])).capture(sim_time_s=1.0, dt_s=0.002)
