"""Search Task C's support station and carrier pose in the arbiter geometry.

The offset12 support pre-phase jammed with piper_0's wrist-camera stand
1.5 mm from piper_1's finger in the simulator's convex-hull geometry.  Both
stations are derived by heuristics that never see the other robot: the support
jaw branch is chosen by "closest to the robot's current orientation", the
carrier yaw by "smallest tool rotation", and the surface bias is a fixed
parameter.

This tool enumerates the admissible candidates of both robots in the arbiter
geometry and reports the minimum cross-robot clearance over the critical pose
sequence -- the staged pair, the support press and the place rise -- so a
station pair can be chosen on simulator evidence instead of geometry that only
holds in MoveIt's optimistic mesh model.

Recorded results (offset12 run, 2026-10-14):

* with the vendor wrist-camera colliders enabled (the state that jammed the
  pre-phase) every candidate interpenetrates by at least ~18 mm at some stage;
* with the wrist-camera colliders disabled -- the shipped geometry, see
  rocobrick.isaac_scene._disable_camera_collisions -- the best station clears
  the staged pair, the press and the rise by 15..35 mm, but the place-up's
  commanded endpoint (the seat overshot by place_press_m) still closes to a
  couple of millimetres.

The remaining opponent is the vendor gripper collar: a 164 mm wide merged
gripper mesh whose PhysX convex hull is a solid slab, and the carrier's wrist
must raise it ~113 mm above its TCP (the calibrated flange-to-TCP offset at the
30-degree grasp tilt) through the band the supporting gripper occupies.  No
lateral shift inside the 64 mm brick can separate them, so either that collar
geometry or the choreography must change.

Usage:
    python tools/task_c_support_station_search.py --run-dir runs/<run>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rocobrick.clearance import (
    DEFAULT_FLOOR_M,
    UrdfChain,
    attachment_box,
    load_hulls,
    region_clearance,
)
from rocobrick.skills.observation import (
    Pose,
    _pose,
    parse_pick_observation,
    parse_place_observation,
)
from rocobrick.skills.placement_geometry import compose, inverse, support_top_target
from rocobrick.skills.targets import RobotBaseFrame, SkillParams, TcpProvenance

REQUEST_NAME = "pre-phase:type1-c-scaled2-piper1-support-exec-request.json"
UPPER = "brick-794d4435d29a64b5dd3e"
LOWER = "brick-b9c3d6923e8be595153b"
DISTAL_BODIES = (
    "gripper_base",
    "gripper_link1",
    "gripper_link2",
    "camera_stand_link",
    "camera_link",
)
# Sample stages: the staged pair, the support press, and the rise of the held
# brick from the pre-place toward the seated depth (millimetres above the
# pre-place), with piper_1 pressed for every stage after the staged pair.
# ``insert_end`` is the place-up's commanded endpoint -- the seat overshot by
# ``place_press_m`` (a bound the live monitor cancels) -- and is the tightest
# pose of the whole choreography.
STAGES = (
    ("staged", 0.0, False),
    ("press", 0.0, True),
    ("rise15", 15.0, True),
    ("seat", 30.0, True),
    ("insert_end", 34.5, True),
)


def world_from_pose(position, orientation) -> np.ndarray:
    """Return a world transform from a position and xyzw orientation."""
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(list(orientation)).as_matrix()
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def base_to_world(position_base, orientation_base, base: RobotBaseFrame):
    """Rotate one base-frame target back into the world frame.

    Returns:
        The world position and world xyzw orientation.
    """
    transform = world_from_pose(base.position_xyz, base.orientation_xyzw)
    position = transform[:3, :3] @ np.asarray(position_base,
        dtype=float) + transform[:3, 3]
    quaternion = (
        Rotation.from_quat(list(base.orientation_xyzw))
        * Rotation.from_quat(list(orientation_base))
    ).as_quat()
    return position, tuple(float(value) for value in quaternion)


def distal_offsets(chain: UrdfChain, robot: str, values: dict[str, float]) -> dict[str,
    np.ndarray]:
    """Return distal bodies' constant transforms in the TCP frame.

    Returns:
        Body name to 4x4 transform in the robot's TCP frame.
    """
    poses = chain.forward(values)
    inverse_tcp = np.linalg.inv(poses[f"{robot}_rocobrick_tcp"])
    return {body: inverse_tcp @ poses[f"{robot}_{body}"] for body in DISTAL_BODIES}


def support_stations(
    request: dict[str, object],
    *,
    biases: tuple[float, ...],
    tilts: tuple[float, ...],
    branches: tuple[bool, ...],
) -> list[dict[str, object]]:
    """Enumerate piper_1's support stations through the production target code.

    ``support_top_target`` is called for the approach and contact gaps; the
    second jaw branch is the same support point rolled half a turn about the
    tool axis, which is exactly the other orientation it can build.

    Returns:
        One dict per station with world approach/contact poses and tool axis.
    """
    snapshot = request["scene_snapshot"]
    context = request["target_contexts"]["piper_1"]
    base = RobotBaseFrame(
        tuple(context["base"]["position_xyz"]),
        tuple(context["base"]["orientation_xyzw"]),
    )
    provenance = TcpProvenance(**context["provenance"])
    sample = parse_pick_observation(
        snapshot,
        robot_id="piper_1",
        physical_instance_id=UPPER,
        session_id=snapshot["session_id"],
        manifest_sha256=snapshot["manifest_sha256"],
    )
    stations = []
    for bias, tilt, flip in itertools.product(biases, tilts, branches):
        params = SkillParams(
            **{
                **context["params"],
                "support_surface_bias_m": bias,
                "support_radial_tilt_deg": tilt,
            }
        )
        try:
            approach = support_top_target(sample, base, provenance, params,
                approach=True)
            contact = support_top_target(sample, base, provenance, params,
                approach=False)
        except ValueError:
            continue
        a_position, a_quat = base_to_world(approach.position, approach.orientation_xyzw,
            base)
        c_position, c_quat = base_to_world(contact.position, contact.orientation_xyzw,
            base)
        if flip:
            a_quat = tuple(
                float(value)
                for value in (
                    Rotation.from_quat(list(a_quat)) * Rotation.from_euler("z", np.pi)
                ).as_quat()
            )
            c_quat = tuple(
                float(value)
                for value in (
                    Rotation.from_quat(list(c_quat)) * Rotation.from_euler("z", np.pi)
                ).as_quat()
            )
        stations.append(
            {
                "bias_m": bias,
                "tilt_deg": tilt,
                "flipped": flip,
                "approach": np.asarray(a_position, dtype=float),
                "contact": np.asarray(c_position, dtype=float),
                "orientation_xyzw": a_quat,
                "tool_axis": Rotation.from_quat(list(a_quat)).as_matrix()
                @ np.array([0.0, 0.0, 1.0]),
            }
        )
    return stations


def carrier_candidates(
    request: dict[str, object],
) -> list[dict[str, object]]:
    """Enumerate the carrier's admissible pre-place candidates.

    Mirrors ``interface_place_up_targets``: the desired lower pose is the fixed
    upper interface rotated about the stud axis by an allowed quarter turn, the
    tool pose composes the measured grasp transform, the pre-place stands
    ``place_up_standoff_m`` below the seated depth along the stud axis, and
    only candidates whose grasp offset points away from the upper part stay
    eligible.

    Returns:
        One dict per candidate with pre-place, seat and orientation.
    """
    snapshot = request["scene_snapshot"]
    params = SkillParams(**request["target_contexts"]["piper_0"]["params"])
    sample = parse_place_observation(
        snapshot,
        robot_id="piper_0",
        upper_physical_instance_id=UPPER,
        lower_physical_instance_id=LOWER,
        session_id=snapshot["session_id"],
        manifest_sha256=snapshot["manifest_sha256"],
    )
    interface = sample.interfaces["pairs"]["|".join(sorted((UPPER, LOWER)))]
    upper_world = sample.upper.object_pose
    held = compose(inverse(sample.lower.object_pose), sample.lower.tcp_pose)
    lower_to_upper = _pose(interface["lower_to_upper"], "interface")
    nominal = compose(upper_world, inverse(lower_to_upper))
    upper_rotation = Rotation.from_quat(list(upper_world.orientation_xyzw))
    normal = upper_rotation.as_matrix() @ np.array([0.0, 0.0, 1.0])
    toward = (
        upper_world.position_xyz[0] - nominal.position_xyz[0],
        upper_world.position_xyz[1] - nominal.position_xyz[1],
    )
    candidates = []
    for quarter in interface.get("allowed_upper_yaw_quarter_turns", (0,)):
        angle = quarter * np.pi / 2.0
        candidate = compose(
            nominal,
            Pose(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, float(np.sin(angle / 2)), float(np.cos(angle / 2))),
            ),
        )
        tcp = compose(candidate, held)
        offset = (
            tcp.position_xyz[0] - candidate.position_xyz[0],
            tcp.position_xyz[1] - candidate.position_xyz[1],
        )
        if offset[0] * toward[0] + offset[1] * toward[1] > 1e-4:
            continue
        seat = np.asarray(tcp.position_xyz, dtype=float)
        candidates.append(
            {
                "quarter_turns": quarter,
                "orientation_xyzw": tcp.orientation_xyzw,
                "seat": seat,
                "preplace": seat - normal * params.place_up_standoff_m,
                "insert_axis": normal,
            }
        )
    return candidates


def held_brick_attachment(request: dict[str, object]) -> object:
    """Return the held brick as an attached box body."""
    attachment = request["physical_facts"]["piper_0"][f"holding:{LOWER}"]["attachment"]
    return attachment_box(
        "piper_0_held_brick",
        attachment["link_name"],
        attachment["minimum_m"],
        attachment["maximum_m"],
    )


def evaluate(
    chain: UrdfChain,
    request: dict[str, object],
    carrier: dict[str, object],
    station: dict[str, object],
    *,
    floor_m: float,
) -> dict[str, object]:
    """Return the worst clearance over the staged/press/rise sequence.

    Returns:
        The worst stage label, distance and binding pair.
    """
    start = request["start_state"]
    fingers = {
        f"piper_{index}_gripper_joint{side}": start[
            f"piper_{index}_gripper_joint{side}"
        ]
        for index in (0, 1)
        for side in (1, 2)
    }
    zeros = {f"piper_{index}_joint{i}": 0.0 for index in (0, 1) for i in range(1, 7)}
    offsets = {
        robot: distal_offsets(chain, robot, {**zeros, **fingers})
        for robot in ("piper_0", "piper_1")
    }
    brick = held_brick_attachment(request)
    normal = carrier["insert_axis"]
    worst = None
    for label, rise_mm, pressed in STAGES:
        tcp_0 = world_from_pose(
            np.asarray(carrier["preplace"]) + normal * (rise_mm / 1000.0),
            carrier["orientation_xyzw"],
        )
        tcp_1_target = station["contact"] if pressed else station["approach"]
        tcp_1 = world_from_pose(tcp_1_target, station["orientation_xyzw"])
        poses = {
            robot: {
                f"{robot}_{body}": tcp @ offset
                for body, offset in offsets[robot].items()
            }
            | {f"{robot}_rocobrick_tcp": tcp}
            for robot, tcp in (("piper_0", tcp_0), ("piper_1", tcp_1))
        }
        report = region_clearance(
            poses, attached={"piper_0": [brick]}, floor_m=floor_m
        )
        if report.worst is None:
            continue
        if worst is None or report.worst.distance_m < worst[1].distance_m:
            worst = (label, report.worst)
    return {
        "stage": worst[0] if worst else None,
        "distance_m": worst[1].distance_m if worst else None,
        "pair": (worst[1].body_a, worst[1].body_b) if worst else None,
    }


def main() -> int:
    """Run the search and print the ranked station table.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR_M)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    request = json.loads(
        (run / "runtime-session/coordinated" / REQUEST_NAME).read_text()
    )
    chain = UrdfChain(
        (run / "runtime-session/dual-robot-config/dual_arm/model.urdf").read_text()
    )
    hulls = load_hulls()
    carriers = carrier_candidates(request)
    stations = support_stations(
        request,
        biases=tuple(round(index * 0.002, 4) for index in range(13)),
        tilts=(0.0, 2.5, 5.0, 7.5, 10.0),
        branches=(False, True),
    )
    print(
        f"arbiter bodies: {len(hulls.link_names)}; floor "
        f"{args.floor * 1000:.1f} mm; carrier yaws "
        f"{[candidate['quarter_turns'] for candidate in carriers]}; "
        f"stations {len(stations)}"
    )
    rows = []
    for carrier, station in itertools.product(carriers, stations):
        result = evaluate(chain, request, carrier, station, floor_m=args.floor)
        rows.append((result["distance_m"], carrier, station, result))
    rows.sort(key=lambda row: (row[0] is None, row[0]))
    header = (
        f"{'yaw':>4} {'bias mm':>8} {'tilt':>5} {'flip':>5} {'worst':>11} "
        f"{'min mm':>9}  binding pair"
    )
    print("worst-first rows:")
    print(header)
    for distance, carrier, station, result in rows[:8]:
        print(
            f"{carrier['quarter_turns']:>4} {station['bias_m'] * 1000:8.1f} "
            f"{station['tilt_deg']:5.1f} {str(station['flipped']):>5} "
            f"{str(result['stage']):>8} "
            f"{(distance * 1000 if distance is not None else float('nan')):9.2f}  "
            f"{result['pair']}"
        )
    best = rows[-1]
    print(
        f"\nbest (max-min) station: bias {best[2]['bias_m'] * 1000:.1f} mm, "
        f"tilt {best[2]['tilt_deg']} deg, flipped {best[2]['flipped']}"
    )
    print("per stage at the best station:")
    for label, rise_mm, pressed in STAGES:
        tcp_0 = world_from_pose(
            np.asarray(best[1]["preplace"])
            + np.asarray(best[1]["insert_axis"]) * (rise_mm / 1000.0),
            best[1]["orientation_xyzw"],
        )
        tcp_1 = world_from_pose(
            best[2]["contact"] if pressed else best[2]["approach"],
            best[2]["orientation_xyzw"],
        )
        start = request["start_state"]
        fingers = {
            f"piper_{index}_gripper_joint{side}": start[
                f"piper_{index}_gripper_joint{side}"
            ]
            for index in (0, 1)
            for side in (1, 2)
        }
        zeros = {
            f"piper_{index}_joint{i}": 0.0
            for index in (0, 1)
            for i in range(1, 7)
        }
        offsets = {
            robot: distal_offsets(chain, robot, {**zeros, **fingers})
            for robot in ("piper_0", "piper_1")
        }
        poses = {
            robot: {
                f"{robot}_{body}": tcp @ offset
                for body, offset in offsets[robot].items()
            }
            | {f"{robot}_rocobrick_tcp": tcp}
            for robot, tcp in (("piper_0", tcp_0), ("piper_1", tcp_1))
        }
        report = region_clearance(
            poses,
            attached={"piper_0": [held_brick_attachment(request)]},
            floor_m=args.floor,
        )
        worst = report.worst
        if worst is None:
            print(f"  {label:11s} beyond the prefilter")
            continue
        print(
            f"  {label:11s} {worst.distance_m * 1000:8.2f} mm  "
            f"{worst.body_a} x {worst.body_b}"
        )
    if best[0] < args.floor:
        print(
            "no admissible station pair clears the floor: the remaining "
            "opponent is the vendor gripper collar (a 164 mm wide merged "
            "gripper mesh hulled by PhysX), which the carrier's wrist must "
            "raise through the supporting gripper's band"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
