"""Same-tick contact and rigid following proofs for a paired handover."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from rocobrick.skills.handover_profile import HandoverProfile, finite
from rocobrick.skills.observation import (
    PickObservation,
    Pose,
    _pose,
    _vector,
    parse_pick_observation,
)
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.targets import _quat_rotate_vector

CONTACT_SCHEMA = "rocobrick-pad-object-contacts/v1"


@dataclass(frozen=True)
class PadContact:
    """Actual physics contact, not a planning-scene collision pair."""

    robot_id: str
    pad_link: str
    physical_instance_id: str
    position_world_m: tuple[float, float, float]
    normal_world: tuple[float, float, float]
    normal_force_n: float

    def __post_init__(self) -> None:
        """Require exact pad identity and finite actual force and geometry."""
        if (
            not isinstance(self.robot_id, str)
            or not self.robot_id
            or not isinstance(self.physical_instance_id, str)
            or not self.physical_instance_id
            or self.pad_link
            not in {
                f"{self.robot_id}_gripper_link1",
                f"{self.robot_id}_gripper_link2",
            }
        ):
            raise ValueError("pad contact identity is invalid")
        position = _vector(self.position_world_m, 3, "contact position")
        normal = _vector(self.normal_world, 3, "contact normal")
        if abs(math.sqrt(sum(v * v for v in normal)) - 1) > 1e-6:
            raise ValueError("contact normal must be normalized")
        finite(self.normal_force_n, "normal force", positive=True)
        object.__setattr__(self, "position_world_m", position)
        object.__setattr__(self, "normal_world", normal)


@dataclass(frozen=True)
class HandoverObservation:
    """One identified scene sample with separately identified pad contacts."""

    donor: PickObservation
    receiver: PickObservation
    contacts: tuple[PadContact, ...]

    def __post_init__(self) -> None:
        """Reject mixed objects, time, identities or simulation frames."""
        if self.donor.robot_id == self.receiver.robot_id:
            raise ValueError("handover observation requires distinct robots")
        for name in (
            "physical_instance_id",
            "session_id",
            "manifest_sha256",
            "sequence",
            "sim_time_s",
            "object_pose",
            "aabb_min",
            "aabb_max",
        ):
            if getattr(self.donor, name) != getattr(self.receiver, name):
                raise ValueError(f"handover observation has mixed {name}")
        if not isinstance(self.contacts, tuple) or any(
            not isinstance(c, PadContact)
            or c.robot_id not in {self.donor.robot_id, self.receiver.robot_id}
            or c.physical_instance_id != self.donor.physical_instance_id
            for c in self.contacts
        ):
            raise ValueError("handover contains foreign pad contact")


def parse_handover_observation(
    snapshot: Mapping[str, object],
    *,
    donor_id: str,
    receiver_id: str,
    physical_instance_id: str,
    session_id: str,
    manifest_sha256: str,
) -> HandoverObservation:
    """Require actual same-tick pad contacts in addition to ordinary scene state.

    Returns:
        An immutable paired observation; empty contacts do not prove co-hold.
    """
    parsed = [
        parse_pick_observation(
            snapshot,
            robot_id=robot_id,
            physical_instance_id=physical_instance_id,
            session_id=session_id,
            manifest_sha256=manifest_sha256,
        )
        for robot_id in (donor_id, receiver_id)
    ]
    raw = snapshot.get("pad_object_contacts")
    required = {
        "schema",
        "source",
        "session_id",
        "manifest_sha256",
        "sequence",
        "sim_time_s",
        "contacts",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("named pad contact evidence is unavailable or malformed")
    if raw["schema"] != CONTACT_SCHEMA or raw["source"] != "physx_contact_report":
        raise ValueError("pad contacts must come from actual physics reports")
    for name in ("session_id", "manifest_sha256", "sequence", "sim_time_s"):
        if type(raw[name]) is not type(snapshot[name]) or raw[name] != snapshot[name]:
            raise ValueError(f"pad contacts have foreign or stale {name}")
    entries = raw["contacts"]
    if not isinstance(entries, (list, tuple)):
        raise ValueError("pad contacts must be an explicit list")
    contacts = []
    expected = {
        "robot_id",
        "pad_link",
        "physical_instance_id",
        "position_world_m",
        "normal_world",
        "normal_force_n",
    }
    for item in entries:
        if not isinstance(item, Mapping) or set(item) != expected:
            raise ValueError("pad contact fields are incomplete or unknown")
        contact = PadContact(**dict(item))
        # The shared report can contain other robots/objects. Selection uses
        # exact identity; a foreign entry is never treated as this object's pad.
        if contact.robot_id in {donor_id, receiver_id} and (
            contact.physical_instance_id == physical_instance_id
        ):
            contacts.append(contact)
    return HandoverObservation(*parsed, tuple(contacts))


def _sequence(
    samples: Sequence[HandoverObservation],
    count: int,
    profile: HandoverProfile,
) -> None:
    if type(count) is not int or count < 3 or len(samples) < count:
        raise ValueError("handover requires the admitted follow_samples count")
    first = samples[0].donor
    previous = first
    for index, sample in enumerate(samples):
        current = sample.donor
        for name in (
            "robot_id",
            "physical_instance_id",
            "session_id",
            "manifest_sha256",
        ):
            if getattr(current, name) != getattr(first, name):
                raise ValueError("handover observation identity changed")
        if sample.receiver.robot_id != samples[0].receiver.robot_id:
            raise ValueError("receiver observation identity changed")
        if index:
            gap = current.sim_time_s - previous.sim_time_s
            if current.sequence <= previous.sequence or not 0 < gap <= (
                profile.max_observation_gap_s
            ):
                raise ValueError("handover observations are stale or discontinuous")
        previous = current
    if previous.sim_time_s - first.sim_time_s < profile.min_observation_span_s:
        raise ValueError("handover observation window is too short")


def _opening(sample: PickObservation, window: tuple[float, float]) -> None:
    lo, hi = _vector(window, 2, "opening window")
    if lo < 0 or lo > hi or not lo <= sample.opening_m <= hi:
        raise ValueError("handover gripper opening is outside its own window")


def _contact_pair(
    sample: HandoverObservation, robot: PickObservation, profile: HandoverProfile
) -> None:
    by_pad: dict[str, list[PadContact]] = {}
    for c in sample.contacts:
        if c.robot_id == robot.robot_id:
            by_pad.setdefault(c.pad_link, []).append(c)
            # Object AABB is only a necessary actual-contact consistency guard.
            if any(
                not robot.aabb_min[i] - profile.contact_position_tolerance_m
                <= c.position_world_m[i]
                <= robot.aabb_max[i] + profile.contact_position_tolerance_m
                for i in range(3)
            ):
                raise ValueError("reported pad contact is outside this object")
    names = [f"{robot.robot_id}_gripper_link{i}" for i in (1, 2)]
    if any(not by_pad.get(name) for name in names):
        raise ValueError("both identified pads must contact the held object")
    closing = _quat_rotate_vector(robot.tcp_pose.orientation_xyzw, (0, 1, 0))
    # Reports use a consistently defined object-to-pad normal. Opposite pads
    # must provide opposing normals along the physical closing axis.
    if not any(
        sum(a.normal_world[i] * closing[i] for i in range(3))
        * sum(b.normal_world[i] * closing[i] for i in range(3))
        < -0.5
        for a in by_pad[names[0]]
        for b in by_pad[names[1]]
    ):
        raise ValueError("pad contact normals do not establish opposed side contact")


def _relative_errors(
    samples: Sequence[HandoverObservation],
    role: str,
    profile: HandoverProfile,
) -> tuple[float, float]:
    first = getattr(samples[0], role)
    reference = compose(inverse(first.tcp_pose), first.object_pose)
    errors = [
        pose_error(
            compose(inverse(getattr(s, role).tcp_pose), getattr(s, role).object_pose),
            reference,
        )
        for s in samples
    ]
    maximum = max(p for p, _ in errors), max(r for _, r in errors)
    if maximum[0] > profile.relative_position_tolerance_m or (
        maximum[1] > profile.relative_orientation_tolerance_rad
    ):
        raise ValueError(f"{role} object-relative SE(3) slipped")
    return maximum


def require_giver_retention_sample(
    sample: HandoverObservation,
    *,
    profile: HandoverProfile,
    donor_window: tuple[float, float],
    expected_object_to_donor_tcp: Pose,
) -> None:
    """Guard one live tick; a tick alone never confirms custody."""
    _opening(sample.donor, donor_window)
    _contact_pair(sample, sample.donor, profile)
    error = pose_error(
        compose(inverse(sample.donor.object_pose), sample.donor.tcp_pose),
        expected_object_to_donor_tcp,
    )
    if error[0] > profile.relative_position_tolerance_m or (
        error[1] > profile.relative_orientation_tolerance_rad
    ):
        raise ValueError("giver no longer retains the admitted lift grasp")


def require_receiver_retention_sample(
    sample: HandoverObservation,
    *,
    profile: HandoverProfile,
    receiver_window: tuple[float, float],
    expected_receiver_attachment: Pose,
    donor_release_window: tuple[float, float] | None = None,
) -> None:
    """Guard receiver contact during a phase, without claiming holding-follow."""
    _opening(sample.receiver, receiver_window)
    _contact_pair(sample, sample.receiver, profile)
    error = pose_error(
        compose(inverse(sample.receiver.tcp_pose), sample.receiver.object_pose),
        expected_receiver_attachment,
    )
    if error[0] > profile.relative_position_tolerance_m or (
        error[1] > profile.relative_orientation_tolerance_rad
    ):
        raise ValueError("receiver grasp changed after co-hold admission")
    if donor_release_window is not None:
        _opening(sample.donor, donor_release_window)
        if any(c.robot_id == sample.donor.robot_id for c in sample.contacts):
            raise ValueError("giver pad contact remains after release")


def verify_giver_holding(
    samples: Sequence[HandoverObservation],
    *,
    profile: HandoverProfile,
    follow_samples: int,
    donor_window: tuple[float, float],
    expected_object_to_donor_tcp: Pose,
) -> Mapping[str, object]:
    """Confirm the giver's current opposed contacts and retained lift grasp.

    Returns:
        Current giver retention, not receiver contact or new ownership.
    """
    _sequence(samples, follow_samples, profile)
    expected = _pose(asdict(expected_object_to_donor_tcp), "retained lift grasp")
    for sample in samples:
        require_giver_retention_sample(
            sample,
            profile=profile,
            donor_window=donor_window,
            expected_object_to_donor_tcp=expected,
        )
    errors = _relative_errors(samples, "donor", profile)
    return {
        "confirmed": True,
        "proof": "opposed_giver_pads_and_retained_lift_grasp",
        "donor_robot_id": samples[0].donor.robot_id,
        "physical_instance_id": samples[0].donor.physical_instance_id,
        "session_id": samples[0].donor.session_id,
        "manifest_sha256": samples[0].donor.manifest_sha256,
        "profile_sha256": profile.sha256,
        "samples": len(samples),
        "relative_error": list(errors),
        "first_sequence": samples[0].donor.sequence,
        "first_sim_time_s": samples[0].donor.sim_time_s,
        "last_sim_time_s": samples[-1].donor.sim_time_s,
        "last_sequence": samples[-1].donor.sequence,
    }


def verify_receiver_retention(
    samples: Sequence[HandoverObservation],
    *,
    profile: HandoverProfile,
    follow_samples: int,
    receiver_window: tuple[float, float],
    expected_receiver_attachment: Pose,
    donor_release_window: tuple[float, float] | None = None,
) -> Mapping[str, object]:
    """Verify receiver custody during release/separation, without static follow.

    Returns:
        Bound retained receiver contact; verification motion is still required.
    """
    _sequence(samples, follow_samples, profile)
    for sample in samples:
        require_receiver_retention_sample(
            sample,
            profile=profile,
            receiver_window=receiver_window,
            expected_receiver_attachment=expected_receiver_attachment,
            donor_release_window=donor_release_window,
        )
    return {
        "confirmed": True,
        "proof": "retained_receiver_pad_contacts_and_se3",
        "receiver_robot_id": samples[0].receiver.robot_id,
        "physical_instance_id": samples[0].receiver.physical_instance_id,
        "samples": len(samples),
    }


def verify_co_hold(
    samples: Sequence[HandoverObservation],
    *,
    profile: HandoverProfile,
    follow_samples: int,
    donor_window: tuple[float, float],
    receiver_window: tuple[float, float],
    expected_object_to_donor_tcp: Pose,
) -> Mapping[str, object]:
    """Prove two-sided actual contact and rigid stability for each gripper.

    Returns:
        Bound co-hold evidence, never an ownership update.
    """
    _sequence(samples, follow_samples, profile)
    expected = _pose(asdict(expected_object_to_donor_tcp), "expected giver grasp")
    grasp_errors = []
    for sample in samples:
        _opening(sample.donor, donor_window)
        _opening(sample.receiver, receiver_window)
        _contact_pair(sample, sample.donor, profile)
        _contact_pair(sample, sample.receiver, profile)
        grasp_error = pose_error(
            compose(inverse(sample.donor.object_pose), sample.donor.tcp_pose),
            expected,
        )
        if grasp_error[0] > profile.relative_position_tolerance_m or (
            grasp_error[1] > profile.relative_orientation_tolerance_rad
        ):
            raise ValueError("giver grasp changed since admitted lift observation")
        grasp_errors.append(grasp_error)
    donor_error = _relative_errors(samples, "donor", profile)
    receiver_error = _relative_errors(samples, "receiver", profile)
    first, last = samples[0].donor, samples[-1].donor
    return {
        "confirmed": True,
        "proof": "same_tick_opposed_pad_contacts_and_se3",
        "physical_instance_id": first.physical_instance_id,
        "donor_robot_id": first.robot_id,
        "receiver_robot_id": samples[0].receiver.robot_id,
        "session_id": first.session_id,
        "manifest_sha256": first.manifest_sha256,
        "profile_sha256": profile.sha256,
        "samples": len(samples),
        "first_sequence": first.sequence,
        "last_sequence": last.sequence,
        "donor_relative_error": list(donor_error),
        "receiver_relative_error": list(receiver_error),
        "giver_grasp_error": [
            max(p for p, _ in grasp_errors),
            max(r for _, r in grasp_errors),
        ],
    }


def verify_receiver_follow(
    samples: Sequence[HandoverObservation],
    *,
    profile: HandoverProfile,
    follow_samples: int,
    donor_release_window: tuple[float, float],
    receiver_window: tuple[float, float],
) -> Mapping[str, object]:
    """Require genuine receiver/object motion, rigid following and donor release.

    Returns:
        Holding-follow evidence independent of planning attachment and owner.
    """
    _sequence(samples, follow_samples, profile)
    for sample in samples:
        _opening(sample.donor, donor_release_window)
        _opening(sample.receiver, receiver_window)
        _contact_pair(sample, sample.receiver, profile)
        if any(c.robot_id == sample.donor.robot_id for c in sample.contacts):
            raise ValueError("giver still contacts the object after release")
    errors = _relative_errors(samples, "receiver", profile)
    first, last = samples[0].receiver, samples[-1].receiver
    tcp_distance = math.dist(first.tcp_pose.position_xyz, last.tcp_pose.position_xyz)
    object_distance = math.dist(
        first.object_pose.position_xyz, last.object_pose.position_xyz
    )
    if min(tcp_distance, object_distance) < profile.verification_distance_m:
        raise ValueError("stationary samples cannot prove receiver holding-follow")
    return {
        "confirmed": True,
        "proof": "released_giver_and_measured_rigid_follow",
        "physical_instance_id": first.physical_instance_id,
        "receiver_robot_id": first.robot_id,
        "session_id": first.session_id,
        "manifest_sha256": first.manifest_sha256,
        "profile_sha256": profile.sha256,
        "samples": len(samples),
        "first_sequence": first.sequence,
        "last_sequence": last.sequence,
        "tcp_distance_m": tcp_distance,
        "object_distance_m": object_distance,
        "relative_error": list(errors),
    }
