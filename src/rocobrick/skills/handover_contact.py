"""Simulation-only, phase-scoped finger contact; never a custody proof.

Ordinary trajectories retain their full tracking clearance proof. During an
explicit contact phase, only the four named cross-arm finger pairs instead use
nominal penetration/region admission plus measured PhysX contact supervision.
This is not a hardware safety certificate or a force/compliance controller.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from rocobrick.skills.handover_profile import finite
from rocobrick.skills.observation import Pose, _vector
from rocobrick.skills.placement_geometry import compose, inverse

CONTACT_PHASES = frozenset(
    {"approach", "receiver_close", "donor_open", "scene_transfer", "separation"}
)
CONTACT_SCHEMA = "rocobrick-finger-contact-history/v1"


@dataclass(frozen=True)
class SimulationFingerContactPolicy:
    """Initial bounded trial limits, independent of pad/object AABB tolerance."""

    max_penetration_m: float = 0.0005
    max_pair_force_n: float = 2.0
    max_pair_impulse_ns: float = 0.01
    max_contact_span_s: float = 3.0
    region_padding_m: float = 0.003
    max_tcp_speed_mps: float = 0.0005
    max_report_gap_s: float = 0.2

    def __post_init__(self) -> None:
        """Validate finite, bounded simulation-only allowances."""
        for name, value in asdict(self).items():
            finite(value, name, positive=True)
        if self.max_penetration_m > 0.0005 or self.region_padding_m > 0.003:
            raise ValueError("finger contact geometric allowance exceeds trial bound")
        if self.max_tcp_speed_mps > 0.001 or self.max_report_gap_s > 0.2:
            raise ValueError("finger contact speed/observation allowance too large")

    @property
    def sha256(self) -> str:
        """Hash the complete simulation contact policy.

        Returns:
            Stable policy identity included in the program authorization.
        """
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def finger_pairs(donor: str, receiver: str) -> frozenset[frozenset[str]]:
    """Exact finger links, not gripper bodies, wrists or arbitrary robots.

    Returns:
        The four explicitly named cross-arm finger pairs.
    """
    if not donor or not receiver or donor == receiver:
        raise ValueError("finger contact needs distinct identified robots")
    return frozenset(
        frozenset((f"{donor}_gripper_link{i}", f"{receiver}_gripper_link{j}"))
        for i in (1, 2)
        for j in (1, 2)
    )


def region_margin(
    point: tuple[float, float, float],
    object_pose: Pose,
    bounds: tuple[tuple[float, ...], tuple[float, ...]],
    policy: SimulationFingerContactPolicy,
) -> float:
    """Distance inside the admitted object-local fingertip contact window.

    Returns:
        Minimum signed distance inside the object-local region.
    """
    local = compose(inverse(object_pose), Pose(point, (0, 0, 0, 1))).position_xyz
    lo, hi = bounds
    return min(
        min(v - a + policy.region_padding_m, b + policy.region_padding_m - v)
        for v, a, b in zip(local, lo, hi)
    )


class FingerContactGuard:
    """Consume every intervening physics step; missing history fails closed."""

    def __init__(self, *, session_id, manifest_sha256, donor, receiver, policy, bounds):
        """Bind one operation's physical identities, region and trial limits."""
        self.session_id, self.manifest_sha256 = session_id, manifest_sha256
        self.donor, self.receiver = donor, receiver
        self.policy, self.bounds = policy, bounds
        self.pairs = finger_pairs(donor, receiver)
        self.last_time: float | None = None
        self.contact_since: dict[frozenset[str], float] = {}
        self.previous_tcp: dict[str, Pose] = {}
        self.summary = {
            "samples": 0,
            "peak_pair_force_n": 0.0,
            "max_penetration_m": 0.0,
        }

    def inspect(
        self,
        snapshot: Mapping[str, object],
        *,
        phase: str,
        object_pose: Pose,
        tcp_poses: Mapping[str, Pose],
    ) -> None:
        """Reject stale, excessive or unpermitted actual physics contacts."""
        report = snapshot.get("finger_contact_history")
        if not isinstance(report, Mapping) or (
            report.get("schema") != CONTACT_SCHEMA
            or report.get("source") != "physx_contact_report"
            or report.get("session_id") != self.session_id
            or report.get("manifest_sha256") != self.manifest_sha256
            or report.get("sequence") != snapshot.get("sequence")
            or report.get("sim_time_s") != snapshot.get("sim_time_s")
        ):
            raise ValueError("finger contact history missing, stale or foreign")
        frames = report.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("finger contact history is empty")
        now = finite(snapshot["sim_time_s"], "finger observation time")
        if (
            self.last_time is not None
            and not 0 < now - self.last_time <= self.policy.max_report_gap_s
        ):
            raise ValueError("finger contact observer gap exceeded")
        allowed = phase in CONTACT_PHASES
        # A shared scene snapshot includes the actual physics TCPs. Bound both
        # arms' motion during contact, not just the receiver's command speed.
        if allowed and self.last_time is not None:
            for robot, pose in tcp_poses.items():
                previous = self.previous_tcp[robot]
                speed = math.dist(pose.position_xyz, previous.position_xyz) / (
                    now - self.last_time
                )
                if speed > self.policy.max_tcp_speed_mps * 1.1:
                    raise ValueError("finger contact measured TCP speed exceeded")
        previous_time = None
        new_frames = []
        for frame in frames:
            if not isinstance(frame, Mapping) or set(frame) != {
                "sim_time_s",
                "dt_s",
                "contacts",
            }:
                raise ValueError("invalid finger contact history frame")
            time = finite(frame["sim_time_s"], "finger frame time")
            dt = finite(frame["dt_s"], "finger physics step", positive=True)
            if time > now or (
                previous_time is not None and abs(time - previous_time - dt) > 1e-6
            ):
                raise ValueError("finger physics history is discontinuous")
            previous_time = time
            if self.last_time is None:
                if time == now:
                    new_frames.append(frame)
            elif time > self.last_time:
                if not new_frames and abs(time - self.last_time - dt) > 1e-6:
                    raise ValueError("finger contact history lost intervening steps")
                new_frames.append(frame)
        if previous_time != now or not new_frames:
            raise ValueError(
                "finger contact history does not reach current physics tick"
            )
        for frame in new_frames:
            entries = frame["contacts"]
            if not isinstance(entries, list):
                raise ValueError("finger contacts must be an explicit list")
            totals = {}
            for item in entries:
                if not isinstance(item, Mapping) or set(item) != {
                    "link_a",
                    "link_b",
                    "position_world_m",
                    "normal_force_n",
                    "impulse_ns",
                    "separation_m",
                }:
                    raise ValueError("invalid finger contact record")
                pair = frozenset((item["link_a"], item["link_b"]))
                if not allowed or pair not in self.pairs:
                    raise ValueError("unpermitted finger contact pair or phase")
                force = finite(item["normal_force_n"], "finger force")
                impulse = finite(item["impulse_ns"], "finger impulse")
                separation = finite(item["separation_m"], "finger separation")
                if force < 0 or impulse < 0:
                    raise ValueError("negative finger force/impulse")
                if separation < -self.policy.max_penetration_m:
                    raise ValueError("finger contact penetration exceeded")
                point = _vector(item["position_world_m"], 3, "finger contact point")
                if region_margin(point, object_pose, self.bounds, self.policy) < 0:
                    raise ValueError("finger contact outside admitted fingertip region")
                f, j = totals.get(pair, (0.0, 0.0))
                # Include tangential impulse, not merely normal force. Sum all
                # points conservatively before applying the pair-level limit.
                totals[pair] = (f + max(force, impulse / frame["dt_s"]), j + impulse)
                self.summary["max_penetration_m"] = max(
                    self.summary["max_penetration_m"], -separation
                )
            time = frame["sim_time_s"]
            for pair, (force, impulse) in totals.items():
                if (
                    force > self.policy.max_pair_force_n
                    or impulse > self.policy.max_pair_impulse_ns
                ):
                    raise ValueError("finger contact pair force/impulse exceeded")
                since = self.contact_since.setdefault(pair, time - frame["dt_s"])
                if time - since > self.policy.max_contact_span_s:
                    raise ValueError("finger contact sustained too long")
                self.summary["peak_pair_force_n"] = max(
                    self.summary["peak_pair_force_n"], force
                )
            self.contact_since = {
                p: t for p, t in self.contact_since.items() if p in totals
            }
            self.summary["samples"] += 1
        self.last_time, self.previous_tcp = now, dict(tcp_poses)

    def require_clear(self) -> None:
        """Require physical contact loss before leaving the contact window."""
        if self.contact_since:
            raise ValueError("finger contact remains after giver separation")
