"""Actual same-tick top support contacts, not stability inferred as contact."""

from collections.abc import Mapping
from dataclasses import asdict

import numpy as np

from rocobrick.skills.handover_observation import CONTACT_SCHEMA, PadContact
from rocobrick.skills.observation import PickObservation, parse_pick_observation
from rocobrick.skills.targets import _quat_rotate_vector

# How far the identified reaction may sit off the brick's top-face plane.
_TOP_FACE_TOLERANCE_M = 0.003
# Plausible BrickSim box sides; anything else is not a recovered part box.
_MIN_SIDE_M = 0.002
_MAX_SIDE_M = 0.300
# A recovered box must reproduce the observed AABB extents this closely.
_EXTENT_RESIDUAL_M = 5e-4
# The pose frame must sit this close to the observed AABB floor.
_ANCHOR_RESIDUAL_M = 1e-3
_LOCAL_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _top_face_offset_m(sample: PickObservation) -> float | None:
    """Measure the upper brick's top-face height above its pose origin.

    BrickSim parts anchor the pose frame at the centre of the bottom face, so
    the top face is the local-z plane at the brick's height. That height is
    recovered from the observed world AABB: a local box of sides ``d`` posed
    under rotation ``R`` has world extents ``|R| d``, and the AABB floor must
    coincide with ``pose_z - |R_xz| d_x / 2 - |R_yz| d_y / 2``. A brick that
    does not reproduce (an unexpected anchor, a degenerate rotation) returns
    None so the caller keeps the conservative AABB corner test.

    The AABB *corner* is the wrong reference for a tilted brick: its world
    AABB top is the far high corner, millimetres above the face under the pad,
    which is how a genuine press was rejected (task-c-top-offset13h: 2.49 deg
    of press tilt put the pad's contact 2.96 mm below the AABB top).

    Returns:
        The top-face height, or None when the observed AABB is unexplained.
    """
    axes = [
        _quat_rotate_vector(sample.object_pose.orientation_xyzw, axis)
        for axis in _LOCAL_AXES
    ]
    if axes[2][2] <= 0.0:
        return None
    columns = np.abs(np.asarray(axes, dtype=np.float64)).T
    extent = np.asarray(sample.aabb_max, dtype=np.float64) - np.asarray(
        sample.aabb_min, dtype=np.float64
    )
    try:
        sides = np.linalg.solve(columns, extent)
    except np.linalg.LinAlgError:
        return None
    if np.any(sides < _MIN_SIDE_M) or np.any(sides > _MAX_SIDE_M):
        return None
    if float(np.max(np.abs(columns @ sides - extent))) > _EXTENT_RESIDUAL_M:
        return None
    floor = sample.object_pose.position_xyz[2] - (
        abs(axes[0][2]) * sides[0] + abs(axes[1][2]) * sides[1]
    ) / 2.0
    if abs(float(floor) - sample.aabb_min[2]) > _ANCHOR_RESIDUAL_M:
        return None
    return float(sides[2])


def _presses_top_face(
    sample: PickObservation, contact: PadContact, top_face_m: float | None
) -> bool:
    """Return whether the reaction sits on the brick's top face.

    With a measured top-face height the reaction is compared against the
    brick's own (possibly tilted) top-face plane, keeping the same two-sided
    3 mm tolerance. Without one the AABB corner test is kept, so an unmodelled
    pose can only become more strict, never looser.
    """
    if top_face_m is None:
        return (
            abs(contact.position_world_m[2] - sample.aabb_max[2])
            <= _TOP_FACE_TOLERANCE_M
        )
    axis = _quat_rotate_vector(
        sample.object_pose.orientation_xyzw, _LOCAL_AXES[2]
    )
    distance = sum(
        axis[index]
        * (contact.position_world_m[index] - sample.object_pose.position_xyz[index])
        for index in range(3)
    )
    return abs(distance - top_face_m) <= _TOP_FACE_TOLERANCE_M


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
    top_face_m = _top_face_offset_m(sample)
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
        if abs(contact.normal_world[2]) < 0.5:
            continue
        if _presses_top_face(sample, contact, top_face_m):
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
