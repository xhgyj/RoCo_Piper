"""Actual same-tick top support contacts, not stability inferred as contact."""

from collections.abc import Mapping
from dataclasses import asdict

from rocobrick.skills.handover_observation import CONTACT_SCHEMA, PadContact
from rocobrick.skills.observation import parse_pick_observation


def verify_support_contact(
    snapshot: Mapping[str, object],
    *,
    robot_id: str,
    upper_id: str,
    session_id: str,
    manifest_sha256: str,
):
    """Require an identified PhysX fingertip reaction on the upper top surface.

    Returns:
        A measured upper/TCP sample and its actual contact evidence.
    """
    sample = parse_pick_observation(
        snapshot,
        robot_id=robot_id,
        physical_instance_id=upper_id,
        session_id=session_id,
        manifest_sha256=manifest_sha256,
    )
    report = snapshot.get("pad_object_contacts")
    if (
        not isinstance(report, Mapping)
        or report.get("schema") != CONTACT_SCHEMA
        or report.get("source") != "physx_contact_report"
    ):
        raise ValueError("SUPPORT_TOP actual contact report unavailable")
    for key in ("session_id", "manifest_sha256", "sequence", "sim_time_s"):
        if report.get(key) != snapshot.get(key):
            raise ValueError("SUPPORT_TOP contact report is stale or foreign")
    contacts = []
    raw = report.get("contacts")
    if not isinstance(raw, list):
        raise ValueError("SUPPORT_TOP contact list unavailable")
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError("invalid support contact record")
        if (
            entry.get("robot_id") != robot_id
            or entry.get("physical_instance_id") != upper_id
        ):
            continue
        contact = PadContact(**entry)
        # Require a vertical reaction at the upper surface, not a side hit.
        if (
            abs(contact.normal_world[2]) >= 0.5
            and abs(contact.position_world_m[2] - sample.aabb_max[2]) <= 0.003
        ):
            contacts.append(asdict(contact))
    if not contacts:
        raise ValueError("SUPPORT_TOP has no measured top-surface contact")
    if sum(contact["normal_force_n"] for contact in contacts) > 8.0:
        raise ValueError("SUPPORT_TOP measured support force exceeded")
    return sample, {
        "confirmed": True,
        "sequence": sample.sequence,
        "sim_time_s": sample.sim_time_s,
        "contacts": contacts,
        "upper_pose": asdict(sample.object_pose),
        "tcp_pose": asdict(sample.tcp_pose),
    }
