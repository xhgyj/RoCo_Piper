#!/usr/bin/env python3
"""P0.5 bounded headless Isaac Sim 5.1 host ROS 2 bridge (Piper L).

Run headless (never a GUI) via the repository environment helper, which wires
the solved Fast DDS TCPv4 host profile:

    bash tools/isaac-ros-env.sh .venv/bin/python isaac/p05_host_bridge.py \
        --evidence-out artifacts/phase0.5/host-bridge-smoke.json

The bridge launches in two phases: the first invocation relaunches itself
through ``isaacsim.main()`` (the KitApp main loop with the full experience
and ``--exec``, the same regime as the bricksim CLI launcher — the only
regime in which BrickSim scene loading is stable); the Kit ``--exec`` child
then runs the bridge coroutine on Kit's async engine. The ros2 bridge
extension is disabled on the Kit command line and re-verified at runtime.

Architecture (fixed after the live OmniGraph bind failure): the host process
runs exactly ONE DDS participant — a single in-process rclpy node created
from the Humble rclpy bundled inside the Isaac install and wired by
``tools/isaac-ros-env.sh`` (PYTHONPATH/LD_LIBRARY_PATH). The
``isaacsim.ros2.bridge`` Kit extension is NEVER enabled: enabling it makes
the extension spawn additional in-process DDS participants, and every
participant built from the host profile tries to bind TCP listening port
17650 — the first binds and the rest abort with "TCPTransport Error binding
at port: 17650 Address already in use", breaking publishing. The in-process
rclpy node is the single-participant alternative. No OmniGraph is created
and no Kit extension startup is awaited.

The script is strictly bounded and verifies the bridge Kit extension stayed
disabled and that the imported rclpy is the bundled Humble one from
``tools/isaac-ros-env.sh`` before running the one participant.

Two modes share the same single node:

- Robot-only mode (no ``--scene-manifest``, the P0.5/P1 smokes) loads
  ``robot_assets/piper_l/USD/piper_l_gripper_d435.usd`` at ``/World/Robot_0``
  (articulation root ``/World/Robot_0/root_joint``) and keeps the pinned
  un-namespaced single-robot topics: publishes ``/clock`` and
  ``/isaac_joint_states``, subscribes ``/isaac_joint_command``.
- Scene mode (``--scene-manifest`` with a validated plan/config) loads the
  BrickSim scene through ``bootstrap_runtime_host`` and binds EVERY robot of
  the validated ``RuntimeHostInputs.robots`` mapping to its own endpoints,
  matching the frozen A3 ``RobotRuntimeProfile.from_robot_id`` namespacing:

  * one shared ``/clock`` (rosgraph_msgs/Clock, sim seconds+nanosec) for the
    whole host,
  * ``/<robot_id>/isaac_joint_states`` (sensor_msgs/JointState) per robot;
    joint names carry the ``<robot_id>_`` prefix of the A3 profile,
  * ``/<robot_id>/isaac_joint_command`` (sensor_msgs/JointState) per robot;
    only that robot's own prefixed joint names are mapped to its DOF indices
    (another robot's names are ignored, never re-routed), targets are clamped
    to the recorded official DOF limits where available and unknown joint
    names are recorded per robot,
  * ``/<robot_id>/joint_states_diag`` per robot (bridge diagnostics only; not
    part of the A3 profile contract).

  Robot ids, prim paths and joint sets come from the validated inputs alone:
  an empty robot set, a missing runtime host articulation, a duplicated or
  nested prim path, a non-ROS-safe robot id, an articulation that does not
  expose the validated joint set, or loaded robots that do not exactly match
  the validated plans all fail readiness before any endpoint is created. There
  is no implicit ``piper_0`` fallback and no second DDS participant: all
  publishers, subscriptions and the scene-state service live on the one node.

  The frozen scene snapshot binding still describes exactly ONE robot; the
  run's designated executor (``--physics-filter-robot-id``) binds it when
  given, else the only robot, else the first validated robot, and the evidence
  records which (``snapshot_binding_source``). Consumers of the artifact
  verify the robot id they need instead of assuming another robot's binding.

The single node also serves ``/p05_host_bridge/scene_state``
(std_srvs/srv/Trigger): the response message carries canonical JSON
(``rocobrick-scene-state/v1``, additive) with the shared per-brick world AABBs
and brick-pair connection flags plus one ``robots[<robot_id>]`` observation
block per loaded robot (its own gripper finger positions, logical opening and
sim time); the top-level ``gripper``/``opening_m`` block is emitted only for a
single robot. It fails closed with an error document when no BrickSim scene is
loaded or the assembly fails.

The readiness marker is printed only after every loaded articulation is
initialized AND the first ``/clock`` and per-robot joint-state messages were
published in-process. A session that removes its own container and then stops
the host with SIGINT gets ``early_stop_accepted`` in the evidence: true only
when the bridge was fully ready, stopped through the orderly SIGINT path, had
no error and completed every orderly cleanup stage. Applied controller
commands are recorded separately from skill-operation dispatch.
``hold_completed``/``passed`` stay governed by ``evaluate_passed`` alone, so
SIGINT never marks the bridge execution passed. Heavy isaacsim/omni/rclpy
imports stay inside functions so the module can be imported by offline
structural tests without booting Kit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rocobrick.gripper import (
    fingers_to_contact_opening,
    fingers_to_logical_gripper,
)
from rocobrick.moveit_contract import JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "rocobrick-host-bridge-smoke/v1"
PHASE = "P0.5"
READY_MARKER = "p0.5_host_bridge_ready"
# SIGINT early stop (operator/session initiated after its own container was
# removed): the Kit child records this exact marker only in the orderly
# KeyboardInterrupt path. The generic Kit-child cancellation uses a different
# marker and never counts as an accepted early stop.
EARLY_STOP_STOPPED = "sigint_orderly_stop"

USD_PATH = ROOT / "robot_assets" / "piper_l" / "USD" / "piper_l_gripper_d435.usd"
ROBOT_PRIM_PATH = "/World/Robot_0"
ARTICULATION_SUFFIX = "/root_joint"
ARTICULATION_PRIM_PATH = ROBOT_PRIM_PATH + ARTICULATION_SUFFIX

# The ROS 2 bridge Kit extension must stay disabled for the whole run: its
# startup spawns additional in-process DDS participants that would compete for
# the host profile TCP listening port. Only its bundled Humble rclpy tree is
# used, exposed by tools/isaac-ros-env.sh via PYTHONPATH/LD_LIBRARY_PATH.
BRIDGE_EXTENSION = "isaacsim.ros2.bridge"
# Official Isaac<->MoveIt topic contract from the pinned
# agx_arm_sim Moveit2/piper_l_gripper_moveit_config (piper_l.ros2_control.xacro,
# TopicBasedSystem): the states topic is plural, the command topic singular.
# Matching these names exactly lets P1 wire ros2_control/MoveIt with zero
# bridge-side changes.
TOPIC_CLOCK = "/clock"
TOPIC_JOINT_STATES = "/isaac_joint_states"
TOPIC_JOINT_COMMANDS = "/isaac_joint_command"
TOPIC_JOINT_STATES_DIAG = "/joint_states_diag"
JOINT_COMMAND_MESSAGE_TYPE = "sensor_msgs/JointState"

ROS_NODE_NAME = "piper_l0_host_bridge"
QOS_DEPTH = 10
NANOS_PER_SECOND = 1_000_000_000

# Scene mode runs one channel per validated robot. The robot id becomes a ROS
# namespace segment under the frozen A3 ``RobotRuntimeProfile.from_robot_id``
# contract; the legacy robot-only mode has no validated robot id at all.
ROBOT_ID_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
ROBOT_ONLY_MODE = "robot_only"
SCENE_MODE = "scene"

# Live scene-state query for the MoveIt container (same single rclpy node):
# std_srvs/srv/Trigger whose response message carries canonical JSON
# (sorted keys, no spaces) of schema rocobrick-scene-state/v1.
SCENE_STATE_SCHEMA = "rocobrick-scene-state/v1"
SCENE_STATE_SERVICE = "/p05_host_bridge/scene_state"
SCENE_STATE_ERROR_MAX_CHARS = 300
GRIPPER_JOINT_1 = "gripper_joint1"
GRIPPER_JOINT_2 = "gripper_joint2"
# Wall-clock publish rate for /clock + /isaac_joint_states. Publishing every physics
# step (~450 Hz here) starves container executors: their spin_once services
# one callback per call, and a permanently-ready /clock crowds out the
# joint-state callbacks entirely.
PUBLISH_PERIOD_S = 1.0 / 30.0

EXPECTED_DOF_COUNT = 8
MIN_READY_STEPS = 5

DEFAULT_MAX_STEPS = 1000000
DEFAULT_TIME_LIMIT_S = 300.0
DEFAULT_READY_TIMEOUT_S = 60.0
DEFAULT_HOLD_STEPS = 600
DEFAULT_HOLD_SECONDS_S = 90.0
PHYSICS_DT = 1.0 / 60.0

# Two-phase launch: the first invocation relaunches THIS bridge through
# isaacsim.main() (KitApp main loop + full experience + ``--exec`` self), the
# same regime the bricksim CLI launcher uses and the only regime in which
# BrickSim scene loading does not segfault (verified live: SimulationApp +
# private asyncio.run segfaults inside import_lego; the bricksim launcher
# regime loads the identical scene cleanly).
BRIDGE_FILE = Path(__file__).resolve()
KIT_EXPERIENCE = "isaacsim.exp.full"


def _running_inside_kit() -> bool:
    """True when this process already hosts a running Kit application.

    The probe never imports isaacsim: in a fresh process ``omni.kit.app`` is
    not importable and the probe returns False; inside a Kit ``--exec`` child
    the application object exists and the probe returns True.
    """
    try:
        from omni.kit.app import get_app

        return get_app() is not None
    except Exception:
        return False


def _argv_from_args(args: argparse.Namespace) -> list[str]:
    """Rebuild the bridge CLI argv from parsed arguments."""
    argv = [
        "--max-steps",
        str(args.max_steps),
        "--time-limit",
        str(args.time_limit),
        "--ready-timeout",
        str(args.ready_timeout),
        "--hold-steps",
        str(args.hold_steps),
        "--hold-seconds",
        str(args.hold_seconds),
        "--evidence-out",
        str(args.evidence_out),
    ]
    for flag, value in (
        ("--scene-manifest", args.scene_manifest),
        ("--scene-snapshot-out", args.scene_snapshot_out),
        ("--host-instance-id", args.host_instance_id),
        ("--plan", args.plan),
        ("--config", args.config),
        ("--structure-dir", args.structure_dir),
        ("--structure-goal", args.structure_goal),
        ("--structure-start", args.structure_start),
        ("--physics-filter-robot-id", args.physics_filter_robot_id),
    ):
        if value is not None:
            argv.extend((flag, str(value)))
    for instance_id in args.physics_filter_instance_id:
        argv.extend(("--physics-filter-instance-id", instance_id))
    return argv


def _kit_launch_argv(args: argparse.Namespace) -> list[str]:
    """Build the Kit argv of the bricksim launch regime for isaacsim.main().

    Mirrors bricksim.__main__: full experience, no empty stage on start,
    crashreporter off, and this bridge re-executed as the Kit ``--exec``
    target with the identical bridge arguments.
    """
    exec_args = [str(BRIDGE_FILE), *_argv_from_args(args)]
    argv = [
        KIT_EXPERIENCE,
        "--disable",
        "isaacsim.ros2.bridge",
        # exp.full points isaac.startup at the ros2 bridge; an empty override
        # keeps the startup sequence from re-enabling the disabled extension.
        "--/isaac/startup/ros_bridge_extension=",
        "--/app/content/emptyStageOnStart=false",
        "--/crashreporter/enabled=false",
        "--/log/outputStreamLevel=info",
        "--/log/channels/*=warn",
        "--/log/channels/bricksim=info",
        "--/log/channels/bricksim.*=info",
        "--exec",
        " ".join(shlex.quote(part) for part in exec_args),
    ]
    if os.environ.get("ROCOBRICK_HEADED") == "1":
        # Headless remains the default. This opt-in is for an operator
        # observing one bounded run from an Isaac desktop session.
        argv.insert(-2, "--/app/window/enabled=true")
    return argv


def evaluate_passed(evidence: dict) -> bool:
    """Decide pass/fail from the persisted evidence.

    Requires readiness, a completed hold, no error, orderly rclpy/world/
    timeline shutdown, and the app-close stage being reached: Kit
    fastShutdown terminates the process inside ``app.close()`` without
    returning, so ``app_close_initiated`` (flushed before the call) is the
    honest terminal marker; ``app_closed`` is only seen when close() returns.
    """
    cleanup = evidence.get("cleanup", {})
    return bool(
        evidence.get("ready")
        and evidence.get("hold_completed")
        and not evidence.get("error")
        and cleanup.get("rclpy_shutdown") is True
        and cleanup.get("world_stopped") is True
        and cleanup.get("timeline_stopped") is True
        and (
            cleanup.get("app_closed") is True
            or cleanup.get("app_close_initiated") is True
        )
    )


def evaluate_early_stop_accepted(evidence: dict) -> bool:
    """Decide whether an interrupted host run counts as an accepted early stop.

    An accepted early stop is the bounded A6 case: the session removes its own
    container and then interrupts the host, and the host must have been fully
    ready, stopped through the orderly SIGINT path (the generic Kit-child
    cancellation marker never qualifies), reported no error and completed
    every orderly cleanup stage. Controller startup may apply hold commands;
    their count remains evidence but is not a cleanup failure.

    This verdict is intentionally separate from :func:`evaluate_passed`: an
    accepted early stop leaves ``hold_completed`` and ``passed`` untouched, so
    a SIGINT never marks a bridge execution passed.

    Args:
        evidence: The persisted bridge evidence document.

    Returns:
        True only when every early-stop condition holds.
    """
    cleanup = evidence.get("cleanup")
    if not isinstance(cleanup, dict):
        return False
    return bool(
        evidence.get("ready") is True
        and evidence.get("stopped") == EARLY_STOP_STOPPED
        and evidence.get("error") is None
        and cleanup.get("rclpy_shutdown") is True
        and cleanup.get("world_stopped") is True
        and cleanup.get("timeline_stopped") is True
        and (
            cleanup.get("app_closed") is True
            or cleanup.get("app_close_initiated") is True
        )
    )


def _finite_vector(brick_id: str, field: str, values: Sequence[float]) -> list[float]:
    """Validate one AABB corner into three finite floats.

    Args:
        brick_id: Owning brick id for the error message.
        field: Corner name ("aabb_min" or "aabb_max") for the error message.
        values: The raw components.

    Returns:
        The three components as plain finite floats.

    Raises:
        ValueError: When the vector is not three finite numbers.
    """
    components = tuple(values)
    if len(components) != 3:
        raise ValueError(f"brick {brick_id} {field} must hold three components")
    vector: list[float] = []
    for index, value in enumerate(components):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"brick {brick_id} {field}[{index}] must be a number: {exc}"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(f"brick {brick_id} {field}[{index}] must be finite")
        vector.append(number)
    return vector


def scene_state_error_document(error: str) -> dict:
    """Build the fail-closed scene-state error document.

    Args:
        error: Raw error text; truncated to ``SCENE_STATE_ERROR_MAX_CHARS``.

    Returns:
        The plain error document payload (not serialized).
    """
    return {
        "schema": SCENE_STATE_SCHEMA,
        "error": str(error)[:SCENE_STATE_ERROR_MAX_CHARS],
    }


def _canonical_json(document: dict) -> str:
    """Serialize a document with sorted keys and no spaces.

    Args:
        document: The payload dict.

    Returns:
        The canonical JSON string used in Trigger response messages.
    """
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _gripper_observation(gripper_positions: Mapping[str, float]) -> dict:
    """Validate one robot's gripper finger positions into a document block.

    Args:
        gripper_positions: Exactly {"gripper_joint1", "gripper_joint2"} ->
            finite finger positions in meters.

    Returns:
        ``{"gripper": {...}, "opening_m": ...}``; a mimic-relation violation
        keeps the fingers with ``opening_m`` None plus ``opening_error``.

    Raises:
        ValueError: On any input-contract violation.
    """
    expected_gripper_keys = {GRIPPER_JOINT_1, GRIPPER_JOINT_2}
    if set(gripper_positions) != expected_gripper_keys:
        raise ValueError(
            "gripper_positions must hold exactly "
            f"{sorted(expected_gripper_keys)}, got {sorted(gripper_positions)}"
        )
    fingers: dict[str, float] = {}
    for name in (GRIPPER_JOINT_1, GRIPPER_JOINT_2):
        try:
            value = float(gripper_positions[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"gripper joint {name} must be a number: {exc}") from exc
        if not math.isfinite(value):
            raise ValueError(f"gripper joint {name} must be finite")
        fingers[name] = value

    observation: dict = {
        "gripper": {
            GRIPPER_JOINT_1: fingers[GRIPPER_JOINT_1],
            GRIPPER_JOINT_2: fingers[GRIPPER_JOINT_2],
        }
    }
    try:
        observation["opening_m"] = fingers_to_logical_gripper(
            fingers[GRIPPER_JOINT_1], fingers[GRIPPER_JOINT_2]
        )
    except ValueError as exc:
        try:
            observation["opening_m"] = fingers_to_contact_opening(
                fingers[GRIPPER_JOINT_1], fingers[GRIPPER_JOINT_2]
            )
            observation["opening_measurement"] = "physical_finger_separation"
            observation["mimic_relation_deviation"] = str(exc)
        except ValueError:
            observation["opening_m"] = None
            observation["opening_error"] = str(exc)
    return observation


def _scene_rigid_pose(raw: Mapping[str, Sequence[float]]) -> dict:
    if not isinstance(raw, Mapping) or set(raw) != {"position_xyz", "orientation_xyzw"}:
        raise ValueError("scene rigid pose must carry position and orientation")
    try:
        position = tuple(raw["position_xyz"])
    except (TypeError, KeyError) as exc:
        raise ValueError("scene rigid pose position must be a 3-vector") from exc
    if len(position) != 3 or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in position
    ):
        raise ValueError("scene rigid pose position must be finite")
    position = [float(value) for value in position]
    try:
        orientation = tuple(raw["orientation_xyzw"])
    except (TypeError, KeyError) as exc:
        raise ValueError("scene rigid pose orientation must be unit xyzw") from exc
    if (
        len(orientation) != 4
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in orientation
        )
        or abs(math.sqrt(sum(float(value) ** 2 for value in orientation)) - 1.0) > 1e-3
    ):
        raise ValueError("scene rigid pose orientation must be unit xyzw")
    return {
        "position_xyz": position,
        "orientation_xyzw": [float(value) for value in orientation],
    }


def _scene_quaternion_multiply(
    left: Sequence[float], right: Sequence[float]
) -> list[float]:
    x, y, z, w = left
    a, b, c, d = right
    return [
        w * a + x * d + y * c - z * b,
        w * b - x * c + y * d + z * a,
        w * c + x * b - y * a + z * d,
        w * d - x * a - y * b - z * c,
    ]


def _compose_scene_poses(parent: Mapping, child: Mapping) -> dict:
    parent_q = parent["orientation_xyzw"]
    child_position = child["position_xyz"]
    inverse = [-parent_q[0], -parent_q[1], -parent_q[2], parent_q[3]]
    rotated = _scene_quaternion_multiply(
        _scene_quaternion_multiply(parent_q, [*child_position, 0.0]), inverse
    )
    return _scene_rigid_pose(
        {
            "position_xyz": [
                parent["position_xyz"][index] + rotated[index] for index in range(3)
            ],
            "orientation_xyzw": _scene_quaternion_multiply(
                parent_q, child["orientation_xyzw"]
            ),
        }
    )


def scene_snapshot_digest(document: Mapping[str, object]) -> str:
    """Return a deterministic digest for one shared scene snapshot."""
    payload = {
        "schema": document.get("schema"),
        "sim_time_s": document.get("sim_time_s"),
        "sequence": document.get("sequence"),
        "manifest_sha256": document.get("manifest_sha256"),
        "bricks": document.get("bricks"),
        "connections": document.get("connections"),
        "robots": document.get("robots"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_scene_state_document(
    sim_time_s: float,
    brick_aabbs: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    connection_results: Mapping[str, bool],
    gripper_positions: Mapping[str, float] | None = None,
    *,
    robot_grippers: Mapping[str, Mapping[str, float]] | None = None,
    brick_poses: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    robot_tcp_poses: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    session_id: str | None = None,
    manifest_sha256: str | None = None,
    sequence: int | None = None,
) -> dict:
    """Assemble the scene-state response document with strict input checks.

    Args:
        sim_time_s: Current simulation time in seconds (finite, >= 0); the one
            shared /clock value observed by every robot.
        brick_aabbs: Physical instance id -> (aabb_min, aabb_max) world
            bounds, three finite components each, min <= max componentwise.
        connection_results: "<id_a>|<id_b>" -> exact bool result of
            ``bricksim.core.are_parts_connected`` for the brick pair.
        gripper_positions: Optional per-observation gripper block emitted as
            the top-level ``gripper``/``opening_m`` keys; the single robot of
            the legacy robot-only mode and of a one-robot scene. Multi-robot
            scenes pass None and observe every robot through ``robot_grippers``.
        robot_grippers: Optional ``robot_id -> gripper_positions`` mapping;
            each robot gets its own ``robots[robot_id]`` block carrying its
            ``robot_id``, the shared ``sim_time_s`` and its own gripper
            observation. The bricks/connections/clock stay top-level and
            shared, so no observation can be attributed to the wrong robot.
        brick_poses: Optional measured world poses by physical instance id.
        robot_tcp_poses: Optional measured world TCP poses by robot id.
        session_id: Scene host identity when live poses are supplied.
        manifest_sha256: Bound scene-manifest digest for pose evidence.
        sequence: Monotonic query sequence for pose evidence.

    Returns:
        The plain scene-state document dict (not serialized).

    Raises:
        ValueError: On any contract violation or when neither observation
            source is provided; callers wrap the message into an error
            document via ``scene_state_error_document``.
    """
    try:
        sim_time = float(sim_time_s)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sim_time_s must be a number: {exc}") from exc
    if not math.isfinite(sim_time):
        raise ValueError("sim_time_s must be finite")
    if sim_time < 0.0:
        raise ValueError("sim_time_s must be non-negative")

    bricks: dict[str, dict[str, list[float]]] = {}
    for brick_id, bounds in brick_aabbs.items():
        if not isinstance(brick_id, str) or not brick_id:
            raise ValueError(f"brick id must be a non-empty string: {brick_id!r}")
        bounds_tuple = tuple(bounds)
        if len(bounds_tuple) != 2:
            raise ValueError(f"brick {brick_id} bounds must be (aabb_min, aabb_max)")
        minimum = _finite_vector(brick_id, "aabb_min", bounds_tuple[0])
        maximum = _finite_vector(brick_id, "aabb_max", bounds_tuple[1])
        if any(lo > hi for lo, hi in zip(minimum, maximum)):
            raise ValueError(f"brick {brick_id} aabb_min exceeds aabb_max")
        bricks[brick_id] = {"aabb_min": minimum, "aabb_max": maximum}

    connections: dict[str, bool] = {}
    for pair_key, connected in connection_results.items():
        if not isinstance(pair_key, str) or not pair_key:
            raise ValueError(f"connection key must be a non-empty string: {pair_key!r}")
        if not isinstance(connected, bool):
            raise ValueError(f"connection {pair_key} value must be a bool")
        connections[pair_key] = connected

    document: dict = {
        "schema": SCENE_STATE_SCHEMA,
        "sim_time_s": sim_time,
        "bricks": bricks,
        "connections": connections,
    }
    if gripper_positions is not None:
        document.update(_gripper_observation(gripper_positions))

    robots_document: dict[str, dict] = {}
    if robot_grippers is not None:
        if not isinstance(robot_grippers, Mapping) or not robot_grippers:
            raise ValueError("robot_grippers must be a non-empty mapping")
        for robot_id, fingers in robot_grippers.items():
            if type(robot_id) is not str or not ROBOT_ID_PATTERN.fullmatch(robot_id):
                raise ValueError(
                    f"robot id must be a ROS-safe non-empty string: {robot_id!r}"
                )
            robots_document[robot_id] = {
                "robot_id": robot_id,
                "sim_time_s": sim_time,
                **_gripper_observation(fingers),
            }
    if brick_poses is not None or robot_tcp_poses is not None:
        if (
            brick_poses is None
            or robot_tcp_poses is None
            or set(brick_poses) != set(bricks)
            or set(robot_tcp_poses) != set(robots_document)
            or not isinstance(session_id, str)
            or not session_id
            or not isinstance(manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)
            or type(sequence) is not int
            or sequence < 0
        ):
            raise ValueError("scene pose provenance or instance set is invalid")
        document.update(
            {
                "session_id": session_id,
                "manifest_sha256": manifest_sha256,
                "sequence": sequence,
            }
        )
        for instance_id, raw in brick_poses.items():
            bricks[instance_id]["pose"] = _scene_rigid_pose(raw)
        for robot_id, raw in robot_tcp_poses.items():
            robots_document[robot_id]["tcp_pose"] = _scene_rigid_pose(raw)
            robots_document[robot_id]["sequence"] = sequence
    if gripper_positions is None and not robots_document:
        raise ValueError("gripper_positions or robot_grippers is required")
    if robots_document:
        document["robots"] = robots_document
    document["scene_digest"] = scene_snapshot_digest(document)
    return document


class RobotEndpointError(ValueError):
    """The validated robot inputs cannot be bound to unique ROS endpoints."""


@dataclass(frozen=True)
class RobotEndpoint:
    """One robot's identity and frozen ROS endpoint contract.

    Scene mode derives every field from the validated ``RuntimeHostInputs``
    (plan and config cross-checked) and the A3 namespacing: one shared
    ``/clock`` plus ``/<robot_id>/isaac_joint_states`` and
    ``/<robot_id>/isaac_joint_command`` per robot. ``robot_id`` is None only
    for the legacy robot-only mode, which has no validated robot id and keeps
    the pinned un-namespaced single-robot topics.
    """

    robot_id: str | None
    prim_path: str
    dof_names: tuple[str, ...]
    joint_states_topic: str
    joint_command_topic: str
    joint_states_diag_topic: str

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return the ROS joint names published and accepted for this robot.

        Returns:
            The validated joint names, prefixed with ``<robot_id>_`` in scene
            mode (the A3 profile contract) and unprefixed in robot-only mode.
        """
        prefix = "" if self.robot_id is None else f"{self.robot_id}_"
        return tuple(prefix + name for name in self.dof_names)


def _robot_topics(robot_id: str) -> tuple[str, str, str]:
    """Derive the namespaced per-robot topics for one robot id.

    Args:
        robot_id: The validated robot id (a ROS-safe namespace segment).

    Returns:
        The (joint states, joint commands, diagnostics) topic triple.
    """
    root = f"/{robot_id}"
    return (
        root + TOPIC_JOINT_STATES,
        root + TOPIC_JOINT_COMMANDS,
        root + TOPIC_JOINT_STATES_DIAG,
    )


def build_robot_endpoints(robots: Mapping[str, object]) -> tuple[RobotEndpoint, ...]:
    """Validate the configured robot specs and derive one endpoint per robot.

    Robot ids, prim paths and joint sets come from the validated inputs alone.
    An empty robot set, a non-ROS-safe robot id, a non-absolute prim path, a
    duplicated or nested prim path, a joint set that does not match the Piper L
    contract or an endpoint collision all fail closed here, before any ROS
    entity exists. Nothing is defaulted or reordered.

    Args:
        robots: ``robot_id -> RobotSpawnSpec`` from the validated inputs.

    Returns:
        The endpoints ordered by robot id.

    Raises:
        RobotEndpointError: On any endpoint-contract violation.
    """
    if not isinstance(robots, Mapping):
        raise RobotEndpointError("validated robot specs must be a mapping")
    if not robots:
        raise RobotEndpointError("at least one validated robot is required")
    endpoints: list[RobotEndpoint] = []
    prim_paths: dict[str, str] = {}
    topics: dict[str, str] = {}
    for robot_id in sorted(robots):
        spec = robots[robot_id]
        if type(robot_id) is not str or not ROBOT_ID_PATTERN.fullmatch(robot_id):
            raise RobotEndpointError(
                f"robot id {robot_id!r} is not a ROS-safe namespace name"
            )
        prim_path = getattr(spec, "prim_path", None)
        if (
            type(prim_path) is not str
            or not prim_path.startswith("/")
            or prim_path == "/"
        ):
            raise RobotEndpointError(
                f"robot {robot_id} prim path must be an absolute non-root path"
            )
        for other_id, other_path in prim_paths.items():
            if (
                prim_path == other_path
                or prim_path.startswith(other_path + "/")
                or other_path.startswith(prim_path + "/")
            ):
                raise RobotEndpointError(
                    f"robots {other_id} and {robot_id} share or nest prim paths: "
                    f"{other_path} / {prim_path}"
                )
        prim_paths[robot_id] = prim_path
        configured = getattr(spec, "arm_joint_names", None)
        if (
            not isinstance(configured, (list, tuple))
            or len(configured) != EXPECTED_DOF_COUNT
        ):
            raise RobotEndpointError(
                f"robot {robot_id} must configure exactly {EXPECTED_DOF_COUNT} joints"
            )
        dof_names = tuple(configured)
        if any(type(name) is not str or not name for name in dof_names):
            raise RobotEndpointError(
                f"robot {robot_id} joint names must be non-empty strings"
            )
        if len(set(dof_names)) != len(dof_names):
            raise RobotEndpointError(f"robot {robot_id} joint names must be unique")
        if set(dof_names) != set(JOINT_NAMES):
            raise RobotEndpointError(
                f"robot {robot_id} joint names do not match the Piper L joint "
                f"set: {sorted(dof_names)}"
            )
        joint_states_topic, joint_command_topic, diag_topic = _robot_topics(robot_id)
        for topic in (joint_states_topic, joint_command_topic, diag_topic):
            owner = topics.setdefault(topic, robot_id)
            if owner != robot_id:
                raise RobotEndpointError(
                    f"robots {owner} and {robot_id} share the endpoint {topic}"
                )
        endpoints.append(
            RobotEndpoint(
                robot_id=robot_id,
                prim_path=prim_path,
                dof_names=dof_names,
                joint_states_topic=joint_states_topic,
                joint_command_topic=joint_command_topic,
                joint_states_diag_topic=diag_topic,
            )
        )
    return tuple(endpoints)


def bind_robot_endpoints(
    robots: Mapping[str, object], loaded_robot_ids: object
) -> tuple[RobotEndpoint, ...]:
    """Bind the validated robot specs to the articulations the host loaded.

    The loaded articulation set must equal the validated robot set exactly: a
    robot missing from the live host or an articulation without validated
    inputs fails readiness before any ROS endpoint is created.

    Args:
        robots: ``robot_id -> RobotSpawnSpec`` from the validated inputs.
        loaded_robot_ids: The robot ids the runtime host reports as loaded.

    Returns:
        The validated endpoints ordered by robot id.

    Raises:
        RobotEndpointError: When the two robot sets differ or an endpoint is
            invalid.
    """
    endpoints = build_robot_endpoints(robots)
    configured = {endpoint.robot_id for endpoint in endpoints}
    try:
        loaded = set(loaded_robot_ids)
    except TypeError as exc:
        raise RobotEndpointError(f"loaded robot ids must be iterable: {exc}") from exc
    missing = sorted(configured - loaded)
    unexpected = sorted(loaded - configured)
    if missing or unexpected:
        raise RobotEndpointError(
            "loaded robots do not match the validated plan/config inputs: "
            f"missing={missing} unexpected={unexpected}"
        )
    return endpoints


def _channel_evidence_key(channel: "RobotChannel") -> str:
    """Evidence key of one channel (robot-only mode has no robot id).

    Args:
        channel: The channel to key.

    Returns:
        The robot id, or ``ROBOT_ONLY_MODE`` for the legacy channel.
    """
    return channel.robot_id if channel.robot_id is not None else ROBOT_ONLY_MODE


def select_snapshot_binding_endpoint(
    endpoints: Sequence[RobotEndpoint], designated_robot_id: str | None
) -> tuple[RobotEndpoint, str]:
    """Choose the robot the single-robot scene snapshot binding describes.

    The frozen ``RuntimeHostBinding`` carries exactly one robot identity while
    a scene may hold several robots. The run's designated executor
    (``--physics-filter-robot-id``) wins; a one-robot scene binds that robot;
    otherwise the first validated robot in robot-id order binds the artifact.
    The returned source string is persisted in the evidence, so the choice is
    explicit and a consumer that needs another robot fails its own identity
    check instead of silently accepting the wrong binding.

    Args:
        endpoints: The validated, robot-id-ordered endpoints.
        designated_robot_id: The run's declared executor robot, or None.

    Returns:
        The binding endpoint and the evidence source string
        (``"designated_robot"``, ``"single_robot"`` or
        ``"first_validated_robot"``).

    Raises:
        RobotEndpointError: When a designated robot is not configured.
    """
    if designated_robot_id is not None:
        for endpoint in endpoints:
            if endpoint.robot_id == designated_robot_id:
                return endpoint, "designated_robot"
        raise RobotEndpointError(
            f"designated snapshot robot is not configured: {designated_robot_id}"
        )
    if len(endpoints) == 1:
        return endpoints[0], "single_robot"
    return endpoints[0], "first_validated_robot"


class RobotChannel:
    """One robot's articulation, ROS handles, counters and command routing.

    The channel owns the joint-state publisher, its diagnostic twin and the
    command-subscription state of exactly one robot. Joint names are prefixed
    with the robot id in scene mode (the A3 profile contract) and stay
    unprefixed in the legacy robot-only mode. Commands are applied only to
    this channel's articulation: another robot's joint names are unknown here
    and are ignored, never re-routed.
    """

    def __init__(
        self,
        endpoint: RobotEndpoint,
        robot: object,
        dof_names: Sequence[str],
    ) -> None:
        """Bind one validated endpoint to one loaded articulation.

        Args:
            endpoint: The frozen per-robot endpoint contract.
            robot: The articulation this channel reads and applies actions to.
            dof_names: The articulation's DOF names in articulation array
                order (the order of ``get_joint_positions``).

        Raises:
            RobotEndpointError: When the articulation does not expose exactly
                the validated joint set.
        """
        articulation_names = tuple(str(name) for name in dof_names)
        if set(articulation_names) != set(endpoint.dof_names):
            raise RobotEndpointError(
                f"robot {endpoint.robot_id} articulation joints "
                f"{sorted(articulation_names)} do not match the validated "
                f"joints {sorted(endpoint.dof_names)}"
            )
        self.endpoint = endpoint
        self.robot_id = endpoint.robot_id
        self.prim_path = endpoint.prim_path
        self.joint_states_topic = endpoint.joint_states_topic
        self.joint_command_topic = endpoint.joint_command_topic
        self.joint_states_diag_topic = endpoint.joint_states_diag_topic
        self.joint_state_type = None
        self._robot = robot
        # Message names follow the articulation array order so positions never
        # need a permutation; each name carries the robot prefix of the A3
        # profile and was validated against the configured joint set above.
        prefix = "" if endpoint.robot_id is None else f"{endpoint.robot_id}_"
        self._dof_names = tuple(prefix + name for name in articulation_names)
        self._dof_index = {name: index for index, name in enumerate(self._dof_names)}
        self._dof_limits: dict[str, tuple[float, float]] = {}
        self.joint_state_publisher = None
        self.joint_state_diag_publisher = None
        self.command_subscription = None
        self.joint_states_published = 0
        self.joint_states_diag_published = 0
        self.positions_nan = 0
        self.commands_received = 0
        self.commands_applied = 0
        self.commands_clamped = 0
        self.unknown_joint_names: list[str] = []
        self.gravity_feedforward = False
        self.gravity_efforts = ()

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return the ROS joint names this channel publishes and accepts."""
        return self._dof_names

    @property
    def initialized(self) -> bool:
        """Report whether this channel's articulation finished initializing."""
        return bool(getattr(self._robot, "is_initialized", lambda: True)())

    def record_dof_limits(self, evidence: dict) -> None:
        """Record this articulation's official DOF limits for clamping.

        Args:
            evidence: The per-robot evidence mapping annotated with the source
                of the limits (or ``"unavailable"``).
        """
        self._dof_limits = _record_dof_limits(
            self._robot, list(self._dof_names), evidence
        )

    def joint_state_message(self, sim_time_s: float, positions, velocities):
        """Build one joint-state message for this robot.

        Args:
            sim_time_s: The shared simulation time in seconds.
            positions: Joint positions in this channel's joint-name order.
            velocities: Optional joint velocities in the same order.

        Returns:
            The sensor_msgs/JointState message (untyped here: the type comes
            from the single in-process rclpy node).
        """
        message = self.joint_state_type()
        seconds, nanoseconds = _split_seconds(sim_time_s)
        message.header.stamp.sec = seconds
        message.header.stamp.nanosec = nanoseconds
        message.name = self._dof_names
        message.position = [float(value) for value in positions]
        if velocities is not None:
            message.velocity = [float(value) for value in velocities]
        return message

    def _joint_velocities(self):
        """Read this articulation's velocities, or None when unsupported."""
        velocity_getter = getattr(self._robot, "get_joint_velocities", None)
        if not callable(velocity_getter):
            return None
        try:
            return velocity_getter()
        except Exception:
            return None

    def publish_state(self, sim_time_s: float) -> None:
        """Publish this robot's joint state and its diagnostic twin.

        Args:
            sim_time_s: The shared simulation time in seconds.
        """
        positions = self._robot.get_joint_positions()
        floats = [float(value) for value in positions]
        if any(value != value for value in floats):
            self.positions_nan += 1
        message = self.joint_state_message(sim_time_s, floats, self._joint_velocities())
        self.joint_state_publisher.publish(message)
        self.joint_states_published += 1
        # Diagnostic twin with static content isolates payload-content issues
        # from delivery issues on the same participant and QoS.
        diag = self.joint_state_type()
        diag.name = list(self._dof_names)
        diag.position = [0.0] * len(self._dof_names)
        self.joint_state_diag_publisher.publish(diag)
        self.joint_states_diag_published += 1

    def on_joint_command(self, message) -> None:
        """Apply this robot's named joint targets as a clamped action.

        Names this articulation does not expose (including another robot's
        prefixed names) are ignored and recorded, so a command can never drive
        the wrong articulation.

        Args:
            message: The sensor_msgs/JointState command message.
        """
        self.commands_received += 1
        names = list(message.name)
        positions = list(message.position)
        if not names or len(positions) != len(names):
            return
        joint_indices: list[int] = []
        joint_positions: list[float] = []
        for name, position in zip(names, positions):
            joint_name = str(name)
            index = self._dof_index.get(joint_name)
            if index is None:
                # Ignore joints this articulation does not expose.
                if joint_name not in self.unknown_joint_names:
                    self.unknown_joint_names.append(joint_name)
                continue
            target = float(position)
            limits = self._dof_limits.get(joint_name)
            if limits is not None:
                clamped = min(max(target, limits[0]), limits[1])
                if clamped != target:
                    self.commands_clamped += 1
                target = clamped
            joint_indices.append(index)
            joint_positions.append(target)
        if not joint_indices:
            return
        from isaacsim.core.utils.types import ArticulationAction

        self._robot.apply_action(
            ArticulationAction(
                joint_positions=tuple(joint_positions),
                joint_indices=tuple(joint_indices),
            )
        )
        self.commands_applied += 1

    def apply_gravity_feedforward(self) -> None:
        """Apply explicitly enabled bounded gravity forces to arm joints only."""
        if not self.gravity_feedforward:
            return
        import numpy as np

        from rocobrick.gravity_feedback import arm_gravity_efforts

        values = self._robot._articulation_view.get_generalized_gravity_forces()
        if values is None:
            raise RuntimeError("enabled gravity feedforward has no live model feedback")
        efforts = arm_gravity_efforts(self._dof_names, [float(x) for x in values[0]])
        self._robot.set_joint_efforts(np.asarray(efforts))
        self.gravity_efforts = efforts

    def dynamics_feedback(self) -> dict[str, object]:
        """Read joint efforts and gravity-model forces without changing drives.

        Returns:
            Identified finite joint vectors or explicit unavailable evidence.
        """
        try:
            efforts = self._robot.get_measured_joint_efforts()
            view = getattr(self._robot, "_articulation_view", None)
            getter = getattr(view, "get_generalized_gravity_forces", None)
            gravity = getter() if callable(getter) else None
            if efforts is None or gravity is None:
                return {"available": False, "reason": "dynamics feedback unavailable"}
            gravity_values = gravity[0]
            measured = [float(x) for x in efforts]
            model = [float(x) for x in gravity_values]
            if len(measured) != len(self._dof_names) or len(model) != len(
                self._dof_names
            ):
                raise ValueError("dynamics joint vector length mismatch")
            if not all(math.isfinite(x) for x in measured + model):
                raise ValueError("nonfinite dynamics feedback")
            return {
                "available": True,
                "robot_id": self.robot_id,
                "gravity_feedforward_enabled": self.gravity_feedforward,
                "applied_gravity_efforts": list(self.gravity_efforts),
                "joint_names": list(self._dof_names),
                "measured_efforts": measured,
                "gravity_compensation_efforts": model,
                "source": "physx_articulation",
                "tcp_wrench_calibrated": False,
            }
        except Exception as error:
            return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    def gripper_positions(self) -> dict[str, float]:
        """Read this robot's two gripper finger DOFs.

        Returns:
            {"gripper_joint1": f1, "gripper_joint2": f2} in meters.

        Raises:
            KeyError: When the articulation lacks a gripper finger DOF.
        """
        positions = self._robot.get_joint_positions()
        prefix = "" if self.robot_id is None else f"{self.robot_id}_"
        return {
            GRIPPER_JOINT_1: float(
                positions[self._dof_index[prefix + GRIPPER_JOINT_1]]
            ),
            GRIPPER_JOINT_2: float(
                positions[self._dof_index[prefix + GRIPPER_JOINT_2]]
            ),
        }


def collect_counter_evidence(
    channels: Sequence[RobotChannel], clock_published: int
) -> dict:
    """Collect the live per-robot counters into the evidence blocks.

    Used both at the end of a completed hold and from the Kit child's
    interrupted path, so an orderly SIGINT carries the same counters a normal
    run would (and ``evaluate_early_stop_accepted`` never reads stale or
    missing command counters). The per-robot breakdown is emitted whenever
    every channel carries a validated robot id (scene mode).

    Args:
        channels: The wired channels of the single in-process node.
        clock_published: Shared /clock publish count.

    Returns:
        ``{"commands": ..., "publish_totals": ...}`` evidence blocks.
    """
    commands = {
        "received": sum(channel.commands_received for channel in channels),
        "applied": sum(channel.commands_applied for channel in channels),
        "clamped_to_limits": sum(channel.commands_clamped for channel in channels),
        "unknown_joint_names_ignored": sorted(
            {name for channel in channels for name in channel.unknown_joint_names}
        ),
    }
    publish_totals = {
        "clock": clock_published,
        "joint_states": sum(channel.joint_states_published for channel in channels),
        "joint_states_diag": sum(
            channel.joint_states_diag_published for channel in channels
        ),
        "positions_nan": sum(channel.positions_nan for channel in channels),
    }
    if channels and all(channel.robot_id is not None for channel in channels):
        commands["by_robot"] = {
            channel.robot_id: {
                "received": channel.commands_received,
                "applied": channel.commands_applied,
                "clamped_to_limits": channel.commands_clamped,
                "unknown_joint_names_ignored": list(channel.unknown_joint_names),
            }
            for channel in channels
        }
        publish_totals["by_robot"] = {
            channel.robot_id: {
                "joint_states": channel.joint_states_published,
                "joint_states_diag": channel.joint_states_diag_published,
                "positions_nan": channel.positions_nan,
            }
            for channel in channels
        }
    return {"commands": commands, "publish_totals": publish_totals}


def capture_owned_window(output: Path, *, run_command=None, owner_pid=None) -> dict:
    """Capture only an Isaac window owned by this host process.

    Returns:
        Window provenance and image path, or a best-effort diagnostic error.
    """
    import subprocess

    run = subprocess.run if run_command is None else run_command
    pid = os.getpid() if owner_pid is None else owner_pid
    try:
        tree = run(
            ["xwininfo", "-root", "-tree"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout
        candidates = re.findall(
            r'^\s*(0x[0-9a-fA-F]+)\s+"([^"\n]*Isaac[^"\n]*)"',
            tree,
            re.MULTILINE | re.IGNORECASE,
        )
        for window, title in candidates[:8]:
            prop = run(
                ["xprop", "-id", window, "_NET_WM_PID"],
                capture_output=True,
                text=True,
                check=True,
                timeout=1,
            ).stdout
            match = re.search(r"=\s*(\d+)\s*$", prop.strip())
            if match is None or int(match.group(1)) != pid:
                continue
            run(
                ["import", "-window", window, str(output)],
                capture_output=True,
                text=True,
                check=True,
                timeout=4,
            )
            return {
                "image": str(output),
                "window_id": window,
                "title": title,
                "owner_pid": pid,
            }
        return {"error": "no Isaac window owned by this host process"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def sync_viewport_physics(world: object, fabric: object) -> None:
    """Copy the latest PhysX transforms to rendering without stepping Kit.

    Mirrors SimulationContext.render's Fabric refresh, deliberately excluding
    app.update: this host is already running inside the Kit async engine.
    """
    fabric.force_update(float(world.get_physics_dt()), float(world.current_time))


async def render_viewport_without_physics(
    world: object, settings: object, app: object, refresh: Callable[[], None]
) -> None:
    """Yield to Kit for rendering, without adding an unobserved physics tick.

    The bridge owns manual physics stepping and samples contacts after each
    step. Match Isaac 5.1 SimulationContext.render_async's playSimulations
    guard while retaining our explicit Fabric refresh and non-reentrant update.
    Restore the caller's setting even on cancellation or rendering failure.
    """
    key = "/app/player/playSimulations"
    previous = settings.get(key)
    if not isinstance(previous, bool):
        raise RuntimeError("Kit playSimulations setting is not a boolean")
    before = (float(world.current_time), world.current_time_step_index)
    settings.set_bool(key, False)
    try:
        refresh()
        await app.next_update_async()
        after = (float(world.current_time), world.current_time_step_index)
        if after != before:
            raise RuntimeError(
                f"render-only Kit update advanced physics: {before} -> {after}"
            )
    finally:
        settings.set_bool(key, previous)


class BridgeState:
    """Holds live Kit/rclpy objects so ``shutdown`` can clean up in order."""

    def __init__(self) -> None:
        """Start empty; live handles are assigned while the bridge runs."""
        self.app = None
        self.defer_app_shutdown = False
        self.viewport_fabric = None
        self.viewport_sync_count = 0
        self.viewport_render_only_count = 0
        self.exit_diagnostics_enabled = (
            os.environ.get("ROCOBRICK_EXIT_DIAGNOSTICS") == "1"
        )
        self.world = None
        self.ros_node = None
        self.evidence = None
        self.evidence_path = None
        # The single in-process participant, assigned when it is wired; the
        # Kit child's interrupted path reads its live channel counters before
        # the final evidence write.
        self.bridge = None
        # Scene-manifest mode only: the loaded BrickSim host and the manifest's
        # physical instance ids in ascending order. Both stay None in the
        # robot-only path, which keeps the scene-state service fail-closed.
        self.runtime_host = None
        self.scene_manifest_ids = None
        self.scene_session_id = None
        self.scene_manifest_sha256 = None
        self.tcp_transform = None
        self.stop_signal = None

    def request_stop(self, signum: int, _frame: object) -> None:
        """Record SIGINT without invoking Kit or ROS from a signal handler."""
        if signum == signal.SIGINT:
            self.stop_signal = signum

    def check_stop(self) -> None:
        """Transfer an explicit SIGINT into the coroutine cleanup boundary."""
        if self.stop_signal == signal.SIGINT:
            raise KeyboardInterrupt

    def _flush(self, status: dict) -> None:
        """Persist cleanup progress so a hang in app.close() keeps evidence."""
        if self.evidence is None or self.evidence_path is None:
            return
        self.evidence["cleanup"] = dict(status)
        self.evidence["early_stop_accepted"] = evaluate_early_stop_accepted(
            self.evidence
        )
        self.evidence["passed"] = evaluate_passed(self.evidence)
        try:
            write_evidence(self.evidence_path, self.evidence)
        except OSError:
            pass

    def refresh_viewport(self, world: object) -> None:
        """Refresh render-side transforms after manual physics-only steps."""
        if self.viewport_fabric is None:
            from omni.kit.app import get_app

            if not get_app().get_extension_manager().is_extension_enabled(
                "omni.physx.fabric"
            ):
                return
            from omni.physxfabric import get_physx_fabric_interface

            self.viewport_fabric = get_physx_fabric_interface()
        sync_viewport_physics(world, self.viewport_fabric)
        self.viewport_sync_count += 1

    async def render_viewport(self, world: object) -> None:
        """Refresh the viewport without letting Kit auto-step physics."""
        from carb.settings import get_settings
        from omni.kit.app import get_app

        await render_viewport_without_physics(
            world, get_settings(), get_app(), lambda: self.refresh_viewport(world)
        )
        self.viewport_render_only_count += 1

    def capture_exit_diagnostics(self, phase: str) -> None:
        """Record opt-in exit images and body poses without commanding motion."""
        if not self.exit_diagnostics_enabled:
            return
        if self.evidence is None or self.evidence_path is None:
            return
        report = {
            "phase": phase,
            "host_pid": os.getpid(),
            "robots": {},
            "viewport_fabric_updates": self.viewport_sync_count,
            "viewport_render_only_updates": self.viewport_render_only_count,
        }
        reports = self.evidence.setdefault("exit_diagnostics", {})
        try:
            directory = self.evidence_path.parent / "exit-diagnostics"
            directory.mkdir(parents=True, exist_ok=True)
            report["window"] = capture_owned_window(directory / f"{phase}.png")
            import carb
            import omni.physx
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            report["fabric_scene_delegate"] = carb.settings.get_settings().get(
                "/app/useFabricSceneDelegate"
            )
            physics = omni.physx.get_physx_interface()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            for channel in () if self.bridge is None else self.bridge.channels:
                robot = {"bodies": {}, "authored_joints": []}
                report["robots"][channel.robot_id] = robot
                for name in (
                    "link6",
                    "gripper_base",
                    "gripper_link1",
                    "gripper_link2",
                    "camera_stand_link",
                    "camera_link",
                ):
                    path = channel.prim_path + "/" + name
                    body = {"prim_path": path}
                    robot["bodies"][name] = body
                    try:
                        body["physx_pose"] = self.bridge._physx_pose(path, physics)
                    except Exception as exc:
                        body["physx_error"] = str(exc)
                    prim = stage.GetPrimAtPath(path)
                    if not prim.IsValid():
                        body["usd_error"] = "missing body prim"
                        continue
                    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                    body["usd_position_xyz"] = list(transform.ExtractTranslation())
                    visuals = stage.GetPrimAtPath(path + "/visuals")
                    if visuals.IsValid():
                        visual_bound = cache.ComputeWorldBound(
                            visuals
                        ).ComputeAlignedRange()
                        if not visual_bound.IsEmpty():
                            body["visual_aabb_min"] = list(visual_bound.GetMin())
                            body["visual_aabb_max"] = list(visual_bound.GetMax())
                    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    if not bounds.IsEmpty():
                        body["usd_aabb_min"] = list(bounds.GetMin())
                        body["usd_aabb_max"] = list(bounds.GetMax())
                root = stage.GetPrimAtPath(channel.prim_path)
                for prim in Usd.PrimRange(root):
                    if not prim.IsA(UsdPhysics.Joint):
                        continue
                    joint = UsdPhysics.Joint(prim)
                    bodies = [str(p) for p in joint.GetBody0Rel().GetTargets()] + [
                        str(p) for p in joint.GetBody1Rel().GetTargets()
                    ]
                    if any("gripper" in p for p in bodies):
                        robot["authored_joints"].append(
                            {
                                "prim_path": str(prim.GetPath()),
                                "bodies": bodies,
                                "enabled": joint.GetJointEnabledAttr().Get(),
                            }
                        )
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        reports[phase] = report
        try:
            (
                self.evidence_path.parent / "exit-diagnostics" / f"{phase}.json"
            ).write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            write_evidence(self.evidence_path, self.evidence)
        except (OSError, ValueError, TypeError):
            pass

    def shutdown(self) -> dict:
        """Shut down in order: rclpy node, world, timeline, Kit app.

        Returns:
            Best-effort per-stage cleanup status.
        """
        status = {}
        node = self.ros_node
        if node is not None:
            try:
                import rclpy

                node.destroy_node()
                try_shutdown = getattr(rclpy, "try_shutdown", None)
                if callable(try_shutdown):
                    try_shutdown()
                else:
                    rclpy.shutdown()
                status["rclpy_shutdown"] = True
            except Exception as exc:  # cleanup is best effort
                status["rclpy_shutdown"] = f"error: {exc}"
            self.ros_node = None
            self._flush(status)
        if self.world is not None:
            try:
                self.world.stop()
                status["world_stopped"] = True
            except Exception as exc:  # cleanup is best effort
                status["world_stopped"] = f"error: {exc}"
            self._flush(status)
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
            status["timeline_stopped"] = True
        except Exception as exc:
            status["timeline_stopped"] = f"error: {exc}"
        self._flush(status)
        if self.app is not None and not self.defer_app_shutdown:
            # Kit's supported application shutdown path is get_app().shutdown().
            # Evidence is flushed before requesting it because Kit may terminate
            # the process inside the call.
            status["app_close_initiated"] = True
            self._flush(status)
            try:
                from omni.kit.app import get_app

                get_app().shutdown()
                status["app_closed"] = True
            except Exception as exc:
                status["app_closed"] = f"error: {exc}"
        return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded-run arguments.

    Args:
        argv: Optional argument list overriding ``sys.argv``.

    Returns:
        The parsed namespace with the bounded-run options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_S)
    parser.add_argument("--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT_S)
    parser.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    parser.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS_S)
    parser.add_argument("--scene-manifest", type=Path, default=None)
    parser.add_argument("--scene-snapshot-out", type=Path, default=None)
    parser.add_argument("--host-instance-id", type=str, default=None)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--structure-dir", type=Path, default=None)
    parser.add_argument("--structure-goal", type=Path, default=None)
    parser.add_argument("--structure-start", type=Path, default=None)
    parser.add_argument("--physics-filter-robot-id", type=str, default=None)
    parser.add_argument("--physics-filter-instance-id", action="append", default=[])
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=ROOT / "artifacts" / "phase0.5" / "host-bridge-smoke.json",
    )
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if min(args.time_limit, args.ready_timeout, args.hold_seconds) <= 0.0:
        parser.error(
            "--time-limit, --ready-timeout and --hold-seconds must be positive"
        )
    if args.hold_steps <= 0:
        parser.error("--hold-steps must be positive")
    snapshot_args = (
        args.scene_manifest,
        args.scene_snapshot_out,
        args.host_instance_id,
    )
    if any(value is not None for value in snapshot_args) and not all(
        value is not None for value in snapshot_args
    ):
        parser.error(
            "--scene-manifest, --scene-snapshot-out and --host-instance-id must be provided together"
        )
    if args.scene_manifest is not None and (args.plan is None or args.config is None):
        parser.error("scene snapshot mode requires --plan and --config")
    if args.scene_manifest is None and (
        args.plan is not None or args.config is not None
    ):
        parser.error("--plan and --config require scene snapshot mode")
    if args.structure_start is not None and args.structure_goal is None:
        parser.error("--structure-start requires --structure-goal")
    if args.structure_goal is not None and args.structure_dir is not None:
        parser.error("--structure-goal cannot be combined with --structure-dir")
    if (
        args.structure_goal is not None or args.structure_start is not None
    ) and args.scene_manifest is None:
        parser.error("--structure-goal/--structure-start require scene snapshot mode")
    if any(
        type(item) is not str or not item for item in args.physics_filter_instance_id
    ):
        parser.error("--physics-filter-instance-id values must be non-empty strings")
    if len(set(args.physics_filter_instance_id)) != len(
        args.physics_filter_instance_id
    ):
        parser.error("--physics-filter-instance-id values must be unique")
    if args.physics_filter_instance_id and args.physics_filter_robot_id is None:
        parser.error("physics filter instance ids require --physics-filter-robot-id")
    return args


def initial_evidence(args: argparse.Namespace) -> dict:
    """Assemble the static evidence fields recorded before Kit starts.

    Args:
        args: The parsed bounded-run arguments.

    Returns:
        The evidence dict with the pinned single-participant contract. The
        per-robot fields are empty in scene mode until the validated inputs
        are parsed inside the Kit child; robot-only mode records its fixed
        single-robot paths and pinned topics here.
    """
    try:
        domain_id = int(os.environ.get("ROS_DOMAIN_ID", "-1"))
    except ValueError:
        domain_id = -1
    robot_only = args.scene_manifest is None
    evidence: dict = {
        "scene_snapshot": {
            "requested": args.scene_manifest is not None
            and args.scene_snapshot_out is not None
            and args.host_instance_id is not None,
            "status": "REQUESTED"
            if args.scene_manifest is not None
            else "NOT_REQUESTED",
        },
        "scene_runtime": {
            "plan_path": str(args.plan) if args.plan is not None else None,
            "config_path": str(args.config) if args.config is not None else None,
        },
        "schema": SCHEMA,
        "phase": PHASE,
        "headless": os.environ.get("ROCOBRICK_HEADED") != "1",
        "robot_mode": ROBOT_ONLY_MODE if robot_only else SCENE_MODE,
        "bounded": {
            "max_steps": args.max_steps,
            "time_limit_s": args.time_limit,
            "ready_timeout_s": args.ready_timeout,
            "hold_steps": args.hold_steps,
            "hold_seconds_s": args.hold_seconds,
            "render_steps": os.environ.get("ROCOBRICK_HEADED") == "1",
        },
        "usd": {"path": str(USD_PATH)},
        "robot_prim_path": ROBOT_PRIM_PATH if robot_only else None,
        "articulation_prim_path": ARTICULATION_PRIM_PATH if robot_only else None,
        "robots": None,
        "bridge_extension_enabled": False,
        "rclpy_source": "bundled humble rclpy via tools/isaac-ros-env.sh",
        "topics": {
            "clock": TOPIC_CLOCK,
            "joint_states": TOPIC_JOINT_STATES if robot_only else None,
            "joint_commands": TOPIC_JOINT_COMMANDS if robot_only else None,
        },
        "per_robot_topics": {
            "joint_states": f"/<robot_id>{TOPIC_JOINT_STATES}",
            "joint_commands": f"/<robot_id>{TOPIC_JOINT_COMMANDS}",
            "joint_states_diag": f"/<robot_id>{TOPIC_JOINT_STATES_DIAG}",
        },
        "joint_command_message_type": JOINT_COMMAND_MESSAGE_TYPE,
        "omnigraph_created": False,
        "in_process_rclpy": True,
        "rclpy_node_name": ROS_NODE_NAME,
        "rclpy_node_count": 1,
        "ros_domain_id": domain_id,
        "dds_profile": os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", ""),
        "readiness_marker": READY_MARKER,
        "ready": False,
        "passed": False,
        "early_stop_accepted": False,
    }
    try:
        evidence["usd"]["sha256"] = hashlib.sha256(USD_PATH.read_bytes()).hexdigest()
    except OSError as exc:
        evidence["usd"]["sha256_error"] = str(exc)
    return evidence


def _split_seconds(sim_time_s: float) -> tuple[int, int]:
    """Split sim time into (seconds, nanoseconds) for ROS time fields.

    Args:
        sim_time_s: The simulation time in seconds.

    Returns:
        The (seconds, nanoseconds) pair with nanoseconds in [0, 1e9).
    """
    seconds = int(sim_time_s)
    nanoseconds = int(round((sim_time_s - seconds) * NANOS_PER_SECOND))
    if nanoseconds >= NANOS_PER_SECOND:
        seconds += 1
        nanoseconds -= NANOS_PER_SECOND
    return seconds, nanoseconds


def _record_dof_limits(
    robot,
    dof_names: list[str],
    evidence: dict,
) -> dict[str, tuple[float, float]]:
    """Read the recorded official DOF limits from the articulation.

    Args:
        robot: The initialized articulation.
        dof_names: Ordered DOF names of the articulation.
        evidence: Evidence dict to annotate with the limit source.

    Returns:
        Mapping of DOF name to (lower, upper) limits, empty when the
        properties are unavailable in this build.
    """
    limits: dict[str, tuple[float, float]] = {}
    try:
        properties = robot.dof_properties
        lower = getattr(properties, "lower", None)
        upper = getattr(properties, "upper", None)
        if lower is None or upper is None:
            lower = properties["lower"]
            upper = properties["upper"]
        for index, name in enumerate(dof_names):
            low = float(lower[index])
            high = float(upper[index])
            if high > low:
                limits[name] = (low, high)
        evidence["dof_limits_source"] = "robot.dof_properties"
    except Exception:
        evidence["dof_limits_source"] = "unavailable"
    return limits


class InProcessBridge:
    """The single in-process rclpy participant (replaces the OmniGraph).

    One node owns the shared clock publisher, one namespaced joint-state
    publisher plus command subscription per robot and the scene-state service,
    so the host process exposes exactly one Fast DDS participant and never
    re-binds the host profile TCP listening port.
    """

    def __init__(
        self,
        state: BridgeState,
        channels: Sequence[RobotChannel],
        *,
        robot_only: bool,
    ) -> None:
        """Initialize rclpy, create the node and wire every robot channel.

        Args:
            state: Bridge state receiving the node for ordered cleanup; also
                the source of scene-mode handles for the scene-state service.
            channels: One validated channel per robot (exactly one in the
                legacy robot-only mode).
            robot_only: True for the pinned un-namespaced single-robot topics
                of the P0.5/P1 smokes; False for the validated scene mode with
                A3 namespaced per-robot endpoints.

        Raises:
            RuntimeError: When the channels do not match the mode.
        """
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from std_srvs.srv import Trigger

        if robot_only and len(channels) != 1:
            raise RuntimeError("robot-only mode requires exactly one robot channel")
        if not robot_only and any(channel.robot_id is None for channel in channels):
            raise RuntimeError("scene mode requires identified robot channels")
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._rclpy = rclpy
        self._clock_type = Clock
        self._scene_state_type = Trigger
        self.node = rclpy.create_node(ROS_NODE_NAME)
        state.ros_node = self.node
        self._state = state
        self._channels = tuple(channels)
        self._scene_sequence = 0
        self._pad_contact_reader = None
        self._pad_contact_frame = None
        self._pad_contact_error = None
        self._finger_contact_history = []
        if not robot_only:
            try:
                import omni.usd

                from rocobrick.adapters.isaac.pad_contacts import PhysxPadContactReader

                self._pad_contact_reader = PhysxPadContactReader.from_stage(
                    omni.usd.get_context().get_stage(),
                    robot_paths={c.robot_id: c.prim_path for c in self._channels},
                    object_paths=state.runtime_host.physical_instance_paths,
                    include_finger_contacts=os.environ.get(
                        "ROCOBRICK_HANDOVER_SIMULATION"
                    )
                    == "1",
                )
            except Exception as error:
                # Missing contact sensing must not invent holding evidence or
                # break unrelated PICK-only sessions. HANDOVER admission reads
                # its explicit capability/error before dispatch.
                self._pad_contact_error = f"{type(error).__name__}: {error}"
        for channel in self._channels:
            channel.joint_state_type = JointState
        self.clock_publisher = self.node.create_publisher(Clock, TOPIC_CLOCK, QOS_DEPTH)
        self.clock_published = 0
        if robot_only:
            # Pinned legacy single-robot topics of the P0.5/P1 smokes.
            channel = self._channels[0]
            channel.joint_state_publisher = self.node.create_publisher(
                JointState, TOPIC_JOINT_STATES, QOS_DEPTH
            )
            channel.joint_state_diag_publisher = self.node.create_publisher(
                JointState, TOPIC_JOINT_STATES_DIAG, QOS_DEPTH
            )
            channel.command_subscription = self.node.create_subscription(
                JointState, TOPIC_JOINT_COMMANDS, channel.on_joint_command, QOS_DEPTH
            )
        else:
            for channel in self._channels:
                topic = channel.joint_states_topic
                diag_topic = channel.joint_states_diag_topic
                command_topic = channel.joint_command_topic
                channel.joint_state_publisher = self.node.create_publisher(
                    JointState, topic, QOS_DEPTH
                )
                channel.joint_state_diag_publisher = self.node.create_publisher(
                    JointState, diag_topic, QOS_DEPTH
                )
                channel.command_subscription = self.node.create_subscription(
                    JointState, command_topic, channel.on_joint_command, QOS_DEPTH
                )
        # Same single node, no second DDS participant: the scene-state query
        # service the MoveIt container calls during skill confirmation.
        self.scene_state_service_created = False
        self.node.create_service(Trigger, SCENE_STATE_SERVICE, self._on_scene_state)
        self.scene_state_service_created = True

    @property
    def channels(self) -> tuple[RobotChannel, ...]:
        """Return the wired per-robot channels in robot-id order."""
        return self._channels

    @property
    def clock_and_joint_states_published(self) -> bool:
        """Report whether the clock and every robot's first state went out.

        Returns:
            True once /clock and every robot's joint-state topic were each
            published in-process.
        """
        return self.clock_published > 0 and all(
            channel.joint_states_published > 0 for channel in self._channels
        )

    @property
    def robots_ready(self) -> bool:
        """Report whether every loaded articulation finished initializing."""
        return all(channel.initialized for channel in self._channels)

    def _clock_message(self, sim_time_s: float):
        message = self._clock_type()
        seconds, nanoseconds = _split_seconds(sim_time_s)
        message.clock.sec = seconds
        message.clock.nanosec = nanoseconds
        return message

    def publish_state(self, world) -> None:
        """Publish /clock once and every robot's joint state from the sim state."""
        sim_time_s = float(world.current_time)
        self.clock_publisher.publish(self._clock_message(sim_time_s))
        self.clock_published += 1
        for channel in self._channels:
            channel.apply_gravity_feedforward()
            channel.publish_state(sim_time_s)

    def _robot_gripper_observations(self) -> dict:
        """Read every loaded robot's gripper DOFs for the scene-state document.

        Returns:
            ``robot_id -> {gripper_joint1, gripper_joint2}``; the legacy
            robot-only channel contributes the None key (no validated id).
        """
        return {
            channel.robot_id: channel.gripper_positions() for channel in self._channels
        }

    def _scene_state_ready(self) -> bool:
        """Report whether a validated BrickSim scene and manifest are loaded."""
        return bool(
            getattr(self._state, "runtime_host", None) is not None
            and getattr(self._state, "scene_manifest_ids", None)
        )

    def capture_pad_contacts(self, world: object) -> None:
        """Capture after each physics step, never reusing the preceding buffer."""
        self._pad_contact_frame = None
        if self._pad_contact_reader is None:
            return
        try:
            self._pad_contact_frame = self._pad_contact_reader.capture(
                sim_time_s=float(world.current_time),
                dt_s=float(world.get_physics_dt()),
            )
            self._pad_contact_error = None
            self._finger_contact_history.append(self._pad_contact_frame)
            # Bound memory by simulation time and count; the handler rejects
            # a query gap that has fallen outside this complete step history.
            now = float(world.current_time)
            self._finger_contact_history = [
                f for f in self._finger_contact_history if now - f.sim_time_s <= 0.5
            ][-512:]
        except Exception as error:
            self._finger_contact_history.clear()
            self._pad_contact_error = f"{type(error).__name__}: {error}"

    def _brick_aabbs(self) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
        """Compute world AABBs for every manifest brick via a BBoxCache.

        Same UsdGeom.BBoxCache approach as
        ``rocobrick.runtime_host.collect_world_aabbs``, but returning raw
        (min, max) corner tuples instead of planning boxes.

        Returns:
            Mapping of physical instance id to (aabb_min, aabb_max) tuples.

        Raises:
            RuntimeError: When a manifest prim is missing from the live stage.
        """
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        aabbs: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        for (
            instance_id,
            prim_path,
        ) in self._state.runtime_host.physical_instance_paths.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"missing runtime prim: {instance_id}")
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            aabbs[instance_id] = (
                tuple(float(value) for value in aligned.GetMin()),
                tuple(float(value) for value in aligned.GetMax()),
            )
        return aabbs

    def _handover_contact_geometry(self) -> dict[str, object]:
        """Export authored body collider bounds in the object's LOCAL frame.

        Never invert a moving world AABB or infer dimensions from a grasp.
        This additive capability is separate from PICK scene observations.
        """
        import hashlib
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
            raise ValueError("HANDOVER contact geometry requires SI stage units")
        result = {}
        for physical, path in self._state.runtime_host.physical_instance_paths.items():
            prim = stage.GetPrimAtPath(path)
            body = prim.GetChild("BodyCollider")
            if not body.IsValid() or not body.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError(f"identified body collider unavailable: {physical}")
            if not body.IsA(UsdGeom.Cube):
                raise ValueError("HANDOVER requires an authored Cube body collider")
            # Never invert a moving world transform: it changes the geometry
            # fingerprint through floating-point roundoff on every scene tick.
            half = float(UsdGeom.Cube(body).GetSizeAttr().Get()) / 2.0
            local = UsdGeom.Xformable(body).GetLocalTransformation()
            if UsdGeom.Xformable(body).GetResetXformStack():
                raise ValueError("body collider must inherit its object frame")
            bounds = Gf.BBox3d(
                Gf.Range3d(Gf.Vec3d(-half), Gf.Vec3d(half)), local
            ).ComputeAlignedRange()
            descriptor = {
                "physical_instance_id": physical,
                "prim_path": str(body.GetPath()),
                "source": "usd_object_local_body_collider",
                "manifest_sha256": self._state.scene_manifest_sha256,
                "minimum_m": [float(v) for v in bounds.GetMin()],
                "maximum_m": [float(v) for v in bounds.GetMax()],
                "collider_type": str(body.GetTypeName()),
            }
            descriptor["source_sha256"] = hashlib.sha256(
                _canonical_json(descriptor).encode()
            ).hexdigest()
            result[physical] = descriptor
        return result

    def _environment_colliders(self) -> dict[str, object]:
        """Measure leaf collision geometry, never aggregate room bounds.

        Returns:
            Leaf collider AABBs in the shared assembly-world frame.

        Raises:
            RuntimeError: If an enabled collision aggregate has no measurable
                leaf geometry or a leaf has an empty bound.
        """
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        excluded = tuple(
            self._state.runtime_host.physical_instance_paths.values()
        ) + tuple(channel.prim_path for channel in self._channels)

        def collision_enabled(prim: object) -> bool:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                return False
            attribute = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
            value = attribute.Get()
            return value is not False

        def subtree_geometries(prim: object) -> list[object]:
            """Return the prim's strict-subtree Gprims via ``Usd.PrimRange``.

            ``Usd.Prim`` has no ``GetDescendants``; the subtree walk must use
            ``Usd.PrimRange(prim)``, which yields the prim itself first.
            """
            return [
                descendant
                for descendant in Usd.PrimRange(prim)
                if descendant != prim and descendant.IsA(UsdGeom.Gprim)
            ]

        def is_geometry_leaf(prim: object, path: str) -> bool:
            return (
                "/colliders/" in path
                and prim.IsA(UsdGeom.Gprim)
                and not subtree_geometries(prim)
            )

        def has_geometry_descendant(prim: object) -> bool:
            return bool(subtree_geometries(prim))

        result: dict[str, object] = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if any(path == root or path.startswith(root + "/") for root in excluded):
                continue
            if not is_geometry_leaf(prim, path) and (
                not collision_enabled(prim) or has_geometry_descendant(prim)
            ):
                continue
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bounds.IsEmpty():
                raise RuntimeError(f"leaf environment collider has no bounds: {path}")
            minimum = tuple(float(value) for value in bounds.GetMin())
            maximum = tuple(float(value) for value in bounds.GetMax())
            if any(high <= low for low, high in zip(minimum, maximum)):
                raise RuntimeError(f"leaf environment collider is degenerate: {path}")
            result[path] = {"aabb_min": list(minimum), "aabb_max": list(maximum)}
        if not result:
            raise RuntimeError("live environment has no leaf collision envelopes")
        return result

    def _physx_pose(self, path: str, physics: object) -> dict:
        """Read one current rigid-body transform; missing physics fails closed.

        Returns:
            The validated world-frame rigid pose.
        """
        result = physics.get_rigidbody_transformation(path)
        if not isinstance(result, Mapping) or result.get("ret_val") is not True:
            raise RuntimeError(f"no live PhysX rigid transform at {path}")
        return _scene_rigid_pose(
            {
                "position_xyz": result.get("position"),
                "orientation_xyzw": result.get("rotation"),
            }
        )

    def _live_rigid_poses(self) -> tuple[dict, dict]:
        """Sample mapped bricks and calibrated TCPs in one service callback.

        Returns:
            Per-instance brick and per-robot TCP pose mappings.
        """
        import omni.physx

        physics = omni.physx.get_physx_interface()
        bricks = {
            instance_id: self._physx_pose(path, physics)
            for instance_id, path in self._state.runtime_host.physical_instance_paths.items()
        }
        transform = self._state.tcp_transform
        if transform is None:
            raise RuntimeError("calibrated TCP transform is unavailable")
        tcp_poses = {}
        for channel in self._channels:
            if channel.robot_id is None:
                continue
            link6 = self._physx_pose(channel.prim_path + "/link6", physics)
            tcp_poses[channel.robot_id] = _compose_scene_poses(link6, transform)
        return bricks, tcp_poses

    def _brick_connections(self) -> dict[str, bool]:
        """Query bricksim for every unordered manifest brick pair.

        Pair keys order the ids by the manifest's ascending physical
        instance id ("<id_a>|<id_b>" with id_a < id_b).

        Returns:
            Mapping of pair key to the live ``are_parts_connected`` result.
        """
        from bricksim.core import are_parts_connected

        instance_ids = list(self._state.scene_manifest_ids)
        paths = self._state.runtime_host.physical_instance_paths
        connections: dict[str, bool] = {}
        for index, id_a in enumerate(instance_ids):
            for id_b in instance_ids[index + 1 :]:
                connections[f"{id_a}|{id_b}"] = bool(
                    are_parts_connected(paths[id_a], paths[id_b])
                )
        return connections

    def _placement_interfaces(self) -> dict[str, object]:
        """Return immutable stud-hole geometry, without creating connections."""
        from bricksim.core import (
            compute_connection_transform,
            get_assembly_debug_infos,
            get_assembly_thresholds,
            lookup_physics_connection,
        )

        paths = self._state.runtime_host.physical_instance_paths
        ids = getattr(self._state, "instance_by_topology_id", {})
        result = {}
        for connection in getattr(self._state, "goal_connections", ()):
            lower = ids.get(connection["stud_id"])
            upper = ids.get(connection["hole_id"])
            if lower not in paths or upper not in paths:
                continue
            rotation, position = compute_connection_transform(
                stud_path=paths[lower],
                stud_if=connection["stud_iface"],
                hole_path=paths[upper],
                hole_if=connection["hole_iface"],
                offset=tuple(connection["offset"]),
                yaw=connection["yaw"],
            )
            # BrickSim returns WXYZ; scene rigid poses use XYZW.
            part = getattr(self._state, "topology_parts", {}).get(
                connection["hole_id"], {}
            )
            payload = part.get("payload", {})
            symmetries = [0]
            if part.get("type") == "brick" and all(
                type(payload.get(name)) is int and payload[name] > 0
                for name in ("L", "W", "H")
            ):
                symmetries = [0, 1, 2, 3] if payload["L"] == payload["W"] else [0, 2]
            # Regular brick prototypes use a centred local origin; only their
            # verified C4 shape symmetries may substitute for the nominal yaw.
            live = lookup_physics_connection(
                paths[lower],
                connection["stud_iface"],
                paths[upper],
                connection["hole_iface"],
            )
            actual = None
            if live is not None:
                actual_rotation, actual_position = compute_connection_transform(
                    paths[lower],
                    live.stud_ifid,
                    paths[upper],
                    live.hole_ifid,
                    tuple(live.offset),
                    live.yaw,
                )
                actual = {
                    "stud_interface": live.stud_ifid,
                    "hole_interface": live.hole_ifid,
                    "offset": list(live.offset),
                    "yaw": live.yaw,
                    "lower_to_upper": {
                        "position_xyz": list(actual_position),
                        "orientation_xyzw": [
                            actual_rotation[1],
                            actual_rotation[2],
                            actual_rotation[3],
                            actual_rotation[0],
                        ],
                    },
                }
            result["|".join(sorted((lower, upper)))] = {
                "connection_id": connection.get("connection_id", connection.get("id")),
                "actual_connection": actual,
                "contact_force_n": None,
                "stud_path": paths[lower],
                "hole_path": paths[upper],
                "allowed_upper_yaw_quarter_turns": symmetries,
                "upper_shape": {"type": part.get("type"), "payload": payload},
                "lower_physical_instance_id": lower,
                "upper_physical_instance_id": upper,
                "lower_to_upper": {
                    "position_xyz": list(position),
                    "orientation_xyzw": [
                        rotation[1],
                        rotation[2],
                        rotation[3],
                        rotation[0],
                    ],
                },
                "stud_interface": connection["stud_iface"],
                "hole_interface": connection["hole_iface"],
                "offset": list(connection["offset"]),
                "yaw": connection["yaw"],
            }
        thresholds = get_assembly_thresholds()
        diagnostics = []
        for info in get_assembly_debug_infos():
            if info.stud_path in paths.values() and info.hole_path in paths.values():
                for pair in result.values():
                    if (
                        info.stud_path == pair["stud_path"]
                        and info.hole_path == pair["hole_path"]
                    ):
                        pair["contact_force_n"] = float(info.projected_force)
                diagnostics.append(
                    {
                        name: getattr(info, name)
                        for name in (
                            "accepted",
                            "relative_distance",
                            "tilt",
                            "projected_force",
                            "yaw_error",
                            "position_error",
                            "grid_pos",
                            "grid_pos_snapped",
                            "stud_path",
                            "hole_path",
                            "stud_interface",
                            "hole_interface",
                        )
                    }
                )
        return {
            "pairs": result,
            "diagnostics": diagnostics,
            "thresholds": {
                name: getattr(thresholds, name)
                for name in (
                    "enabled",
                    "distance_tolerance",
                    "max_penetration",
                    "z_angle_tolerance",
                    "required_force",
                    "yaw_tolerance",
                    "position_tolerance",
                )
            },
        }

    def _on_scene_state(self, request, response):
        """Fill the Trigger response with canonical per-robot scene-state JSON.

        Never raises: a missing scene or any assembly failure fails closed
        with an error document in ``response.message``. The document carries
        the shared bricks/connections/clock plus one ``robots[robot_id]``
        observation block per loaded robot; the top-level gripper block is
        emitted only for a single robot (legacy robot-only mode or a one-robot
        scene), so an observation is never attributed to another robot.

        Args:
            request: The empty Trigger request (unused).
            response: The Trigger response to fill.

        Returns:
            The filled response.
        """
        if not self._scene_state_ready():
            response.success = False
            response.message = _canonical_json(
                scene_state_error_document("scene_not_loaded")
            )
            return response
        try:
            observations = self._robot_gripper_observations()
            identified = {
                robot_id: fingers
                for robot_id, fingers in observations.items()
                if robot_id is not None
            }
            sim_time_s = float(self._state.world.current_time)
            brick_aabbs = self._brick_aabbs()
            connections = self._brick_connections()
            brick_poses, robot_tcp_poses = self._live_rigid_poses()
            self._scene_sequence += 1
            if identified:
                document = build_scene_state_document(
                    sim_time_s,
                    brick_aabbs,
                    connections,
                    (next(iter(identified.values())) if len(identified) == 1 else None),
                    robot_grippers=identified,
                    brick_poses=brick_poses,
                    robot_tcp_poses=robot_tcp_poses,
                    session_id=self._state.scene_session_id,
                    manifest_sha256=self._state.scene_manifest_sha256,
                    sequence=self._scene_sequence,
                )
            else:
                document = build_scene_state_document(
                    sim_time_s,
                    brick_aabbs,
                    connections,
                    next(iter(observations.values())),
                )
        except Exception as exc:  # fail closed; never raise out of a callback
            response.success = False
            response.message = _canonical_json(
                scene_state_error_document(f"{type(exc).__name__}: {exc}")
            )
            return response
        for channel in self._channels:
            if channel.robot_id in document.get("robots", {}):
                document["robots"][channel.robot_id]["dynamics"] = (
                    channel.dynamics_feedback()
                )
        if getattr(self._state, "goal_connections", None) is not None:
            document["placement_interfaces"] = self._placement_interfaces()
        if self._scene_state_ready():
            try:
                document["tcp_provenance"] = dict(self._state.tcp_provenance)
                document["environment_colliders"] = self._environment_colliders()
                if os.environ.get("ROCOBRICK_HANDOVER_SIMULATION") == "1":
                    try:
                        document["handover_contact_geometry"] = (
                            self._handover_contact_geometry()
                        )
                    except Exception as geometry_error:
                        document["handover_contact_geometry_error"] = str(
                            geometry_error
                        )
                document["handover_capabilities"] = {"named_pad_contacts": False}
                frame = getattr(self, "_pad_contact_frame", None)
                if frame is not None and frame.sim_time_s == sim_time_s:
                    document["pad_object_contacts"] = dict(
                        frame.document(
                            sim_time_s=sim_time_s,
                            sequence=self._scene_sequence,
                            session_id=self._state.scene_session_id,
                            manifest_sha256=self._state.scene_manifest_sha256,
                        )
                    )
                    document["handover_capabilities"]["named_pad_contacts"] = True
                    if os.environ.get("ROCOBRICK_HANDOVER_SIMULATION") == "1":
                        from rocobrick.adapters.isaac.pad_contacts import (
                            finger_history_document,
                        )

                        document["finger_contact_history"] = dict(
                            finger_history_document(
                                self._finger_contact_history,
                                sim_time_s=sim_time_s,
                                sequence=self._scene_sequence,
                                session_id=self._state.scene_session_id,
                                manifest_sha256=self._state.scene_manifest_sha256,
                            )
                        )
                error = getattr(self, "_pad_contact_error", None)
                if error is not None:
                    document["handover_capabilities"]["pad_contacts_error"] = error
            except Exception as error:
                response.success = False
                response.message = _canonical_json(
                    scene_state_error_document(str(error))
                )
                return response
        payload = {
            key: value for key, value in document.items() if key != "scene_digest"
        }
        document["scene_digest"] = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        response.success = True
        response.message = _canonical_json(document)
        return response

    def spin_once(self) -> None:
        """Drain one callback round without blocking the stepping loop.

        Service requests (the scene-state query) are answered at this same
        bounded per-step point; timeout 0 never blocks the physics loop.
        """
        self._rclpy.spin_once(self.node, timeout_sec=0.0)


async def run_bridge(
    args: argparse.Namespace, state: BridgeState, evidence: dict
) -> None:
    """Boot headless Kit, verify the bridge extension stayed disabled, then run.

    The isaacsim.ros2.bridge Kit extension is never enabled; the bundled
    Humble rclpy wired by tools/isaac-ros-env.sh is imported directly and its
    provenance is pinned in the evidence before the single participant starts.
    """
    if not _running_inside_kit():
        raise RuntimeError(
            "run_bridge must run inside the Kit child regime (isaacsim.main "
            "+ --exec); launch the bridge through main() so it relaunches "
            "itself first"
        )

    import omni.kit.app
    import omni.timeline
    import omni.usd

    # The Kit extension must stay disabled for the whole run: enabling it
    # would let the extension spawn its own in-process DDS participants and
    # re-bind the host profile TCP listening port. Verify the disabled state,
    # then import the bundled Humble rclpy exposed by tools/isaac-ros-env.sh
    # (PYTHONPATH/LD_LIBRARY_PATH). The import alone creates no participant;
    # the single one is created later by InProcessBridge.
    manager = omni.kit.app.get_app().get_extension_manager()
    if manager.is_extension_enabled(BRIDGE_EXTENSION):
        raise RuntimeError(
            f"{BRIDGE_EXTENSION} must stay disabled; enabling it would "
            "create additional DDS participants in this process"
        )
    evidence["bridge_extension_enabled"] = False

    if args.scene_manifest is None:
        # The original P0.5 robot-only path remains available.
        from isaacsim.core.api import PhysicsContext, World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import create_prim

        PhysicsContext()
        create_prim(prim_path=ROBOT_PRIM_PATH, usd_path=USD_PATH.as_posix())
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(ROBOT_PRIM_PATH)
        if not prim.IsValid():
            raise RuntimeError(f"robot prim missing after load: {ROBOT_PRIM_PATH}")
        world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
        state.world = world
        robot = world.scene.add(
            SingleArticulation(prim_path=ARTICULATION_PRIM_PATH, name="piper_l")
        )
        world.reset()
        dof_names = [str(name) for name in robot.dof_names]
        channels = [
            RobotChannel(
                RobotEndpoint(
                    robot_id=None,
                    prim_path=ROBOT_PRIM_PATH,
                    dof_names=tuple(dof_names),
                    joint_states_topic=TOPIC_JOINT_STATES,
                    joint_command_topic=TOPIC_JOINT_COMMANDS,
                    joint_states_diag_topic=TOPIC_JOINT_STATES_DIAG,
                ),
                robot,
                dof_names,
            )
        ]
        timeline = omni.timeline.get_timeline_interface()
        world.play()
    else:
        from rocobrick.isaac_snapshot_host_adapter import (
            build_host_binding,
            build_scene_manifest_from_runtime_host,
            collect_stage_snapshot,
            load_scene_snapshot_manifest,
            write_scene_manifest_atomic,
            write_snapshot_artifact_atomic,
        )
        from rocobrick.robot_operation_plan import parse_robot_operation_plan
        from rocobrick.runtime_host import bootstrap_runtime_host
        from rocobrick.runtime_host_inputs import build_runtime_host_inputs
        from rocobrick.stap_contract import parse_stap_plan
        from rocobrick.structure_scene import (
            build_structure_scene_plan,
            build_structure_scene_plan_from_files,
        )

        try:
            plan_bytes = args.plan.read_bytes()
            evidence["scene_runtime"]["plan_sha256"] = hashlib.sha256(
                plan_bytes
            ).hexdigest()
            plan_document = json.loads(plan_bytes)
            if args.structure_dir is not None:
                plan, structure_evidence = build_structure_scene_plan(args.structure_dir, plan_document)
                evidence["scene_runtime"]["structure"] = structure_evidence
            elif args.structure_goal is not None:
                # Additive explicit-files mode (scene loading only): the
                # external robot-operation-plan/v1 is parsed strictly first,
                # then the explicit goal (and optional start; None means EMPTY
                # initial structure - no directory default) build the internal
                # topology carrier. This dispatches no operation and
                # authorizes no motion; any parse or binding failure raises
                # here, before bootstrap_runtime_host runs.
                parse_robot_operation_plan(plan_document)
                plan, structure_evidence = build_structure_scene_plan_from_files(
                    args.structure_start,
                    args.structure_goal,
                    plan_document,
                    manifest=(
                        args.scene_manifest
                        if args.scene_manifest is not None
                        and args.scene_manifest.is_file()
                        else None
                    ),
                )
                evidence["scene_runtime"]["structure"] = structure_evidence
            else:
                plan = parse_stap_plan(plan_document)
            config = json.loads(args.config.read_text(encoding="utf-8"))
            inputs = build_runtime_host_inputs(
                plan,
                config,
                config_dir=args.config.resolve().parent,
                collision_filter_robot_id=args.physics_filter_robot_id,
                collision_filter_instance_ids=tuple(args.physics_filter_instance_id),
            )
            evidence["scene_runtime"]["initial_part_allocation"] = {
                instance_id: dict(reason)
                for instance_id, reason in inputs.scene_config.initial_part_allocation.items()
            }
            runtime_host = await bootstrap_runtime_host(
                plan, inputs.topology, inputs.robots, inputs.scene_config
            )
            state.runtime_host = runtime_host
            state.goal_connections = inputs.topology.expected_goal_connections
            state.topology_parts = {
                part["id"]: part for part in inputs.topology.goal["parts"]
            }
            state.instance_by_topology_id = inputs.topology.instance_by_topology_id
            state.app = runtime_host.app
            world = runtime_host.world
            state.world = world

            # Every validated robot gets its own namespaced endpoints (A3
            # profile contract); the loaded articulation set must match the
            # validated one exactly. Robot ids and prim paths come from the
            # validated inputs, never from a hard-coded piper_0.
            endpoints = bind_robot_endpoints(inputs.robots, runtime_host.robots)
            stage = omni.usd.get_context().get_stage()
            channels = []
            for endpoint in endpoints:
                robot = runtime_host.robots[endpoint.robot_id]
                if not stage.GetPrimAtPath(endpoint.prim_path).IsValid():
                    raise RuntimeError(
                        f"robot {endpoint.robot_id} prim missing after load: "
                        f"{endpoint.prim_path}"
                    )
                channels.append(
                    RobotChannel(
                        endpoint,
                        robot,
                        tuple(str(name) for name in robot.dof_names),
                    )
                )
            evidence["robots"] = {
                channel.robot_id: {
                    "robot_id": channel.robot_id,
                    "prim_path": channel.prim_path,
                    "joint_states_topic": channel.joint_states_topic,
                    "joint_command_topic": channel.joint_command_topic,
                    "joint_states_diag_topic": channel.joint_states_diag_topic,
                    "joint_names": list(channel.joint_names),
                }
                for channel in channels
            }
            # The frozen single-robot snapshot binding describes one robot: the
            # run's designated executor when given, else the only robot, else
            # the first validated robot. The choice is persisted below.
            binding_endpoint, binding_source = select_snapshot_binding_endpoint(
                endpoints, args.physics_filter_robot_id
            )
            binding_channel = next(
                channel
                for channel in channels
                if channel.robot_id == binding_endpoint.robot_id
            )
            timeline = omni.timeline.get_timeline_interface()
            evidence["scene_runtime"].update(
                {
                    "robot_id": binding_endpoint.robot_id,
                    "robot_prim_path": binding_endpoint.prim_path,
                    "articulation_prim_path": (
                        binding_endpoint.prim_path + ARTICULATION_SUFFIX
                    ),
                    "snapshot_binding_source": binding_source,
                    "snapshot_binding_note": (
                        "the frozen scene snapshot binding describes one robot; "
                        "pass --physics-filter-robot-id to designate it"
                    ),
                    "robots": {
                        channel.robot_id: {
                            "prim_path": channel.prim_path,
                            "joint_states_topic": channel.joint_states_topic,
                            "joint_command_topic": channel.joint_command_topic,
                        }
                        for channel in channels
                    },
                    "part_mapping": dict(runtime_host.physical_instance_paths),
                    "collision_filter_pairs": list(runtime_host.collision_filter_pairs),
                }
            )

            manifest_generated = False
            if args.scene_manifest.is_file():
                manifest = load_scene_snapshot_manifest(args.scene_manifest)
            else:
                manifest = build_scene_manifest_from_runtime_host(
                    runtime_host,
                    scene_id=args.host_instance_id,
                    runtime_usd_sha256=evidence["usd"]["sha256"],
                )
                write_scene_manifest_atomic(manifest, args.scene_manifest)
                manifest_generated = True
            evidence["scene_runtime"]["scene_manifest"] = {
                "path": str(args.scene_manifest),
                "source": ("isaac_runtime_stage" if manifest_generated else "provided"),
                "manifest_sha256": manifest.manifest_sha256,
            }
            manifest_map = {
                item.physical_instance_id: item.prim_path for item in manifest.objects
            }
            if manifest_map != dict(runtime_host.physical_instance_paths):
                raise RuntimeError(
                    "manifest does not match loaded BrickSim instance mapping"
                )
            for item in manifest.objects:
                actual_brick_id = runtime_host.physical_instance_brick_ids.get(
                    item.physical_instance_id
                )
                if actual_brick_id != item.brick_id:
                    raise RuntimeError(
                        f"manifest brick id mismatch: {item.physical_instance_id}"
                    )
            # Ascending manifest order drives the scene-state pair keys.
            state.scene_manifest_ids = sorted(manifest_map)
            state.scene_session_id = args.host_instance_id
            state.scene_manifest_sha256 = manifest.manifest_sha256
            calibration_path = (
                ROOT / "artifacts/model-audit/piper_l_tcp_calibration.json"
            )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            transform = calibration.get("transform")
            if (
                calibration.get("passed") is not True
                or not isinstance(transform, dict)
                or transform.get("parent_frame") != "link6"
                or transform.get("child_frame") != "rocobrick_tcp"
            ):
                raise RuntimeError("project TCP calibration is not resolved")
            state.tcp_provenance = {
                "runtime_model_sha256": "a781d52f9d55291fca9c8f2695abf9e311640d996314d272f47639d5323e2375",
                "source_sha256": hashlib.sha256(
                    calibration_path.read_bytes()
                ).hexdigest(),
            }
            state.tcp_transform = _scene_rigid_pose(
                {
                    "position_xyz": transform.get("translation_m"),
                    "orientation_xyzw": transform.get("rotation_xyzw"),
                }
            )

            profile_path = Path(evidence["dds_profile"])
            profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
            host_binding = build_host_binding(
                host_instance_id=args.host_instance_id,
                ros_domain_id=int(evidence["ros_domain_id"]),
                host_profile_sha256=profile_sha,
                host_bridge_schema=SCHEMA,
                runtime_usd_path=str(USD_PATH),
                runtime_usd_sha256=evidence["usd"]["sha256"],
                robot_id=binding_endpoint.robot_id,
                robot_prim_path=binding_endpoint.prim_path,
                articulation_prim_path=(
                    binding_endpoint.prim_path + ARTICULATION_SUFFIX
                ),
                joint_dofs=tuple(binding_channel.joint_names),
                participant_count=1,
            )
            snapshot = collect_stage_snapshot(stage, manifest, host_binding)
            write_snapshot_artifact_atomic(snapshot, args.scene_snapshot_out)
            evidence["scene_snapshot"].update(
                {
                    "status": "CAPTURED",
                    "path": str(args.scene_snapshot_out),
                    "snapshot_sha256": snapshot.snapshot_sha256,
                    "object_ids": [
                        obj.physical_instance_id for obj in snapshot.objects
                    ],
                    "host_instance_id": args.host_instance_id,
                    "provenance": "runtime_host_bricksim_scene_pre_rclpy",
                }
            )
        except Exception as exc:
            cause = exc.__cause__
            evidence["scene_snapshot"].update(
                {
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "cause": (
                        f"{type(cause).__name__}: {cause}"
                        if cause is not None
                        else None
                    ),
                    "provenance": "runtime_host_bricksim_scene_pre_rclpy",
                }
            )
            raise

    if args.scene_manifest is None:
        dof_names = list(channels[0].joint_names)
        evidence["joint_dofs"] = dof_names
        if len(dof_names) != EXPECTED_DOF_COUNT:
            raise RuntimeError(
                f"expected {EXPECTED_DOF_COUNT} DOFs, "
                f"articulation reported {len(dof_names)}"
            )
    else:
        # load_verified_scene already reset and started the shared world.
        world.play()
    evidence["timeline_playing"] = bool(timeline.is_playing())

    # Import rclpy only after scene loading and snapshot admission. Importing
    # it earlier can initialize ROS resources before the Kit/BrickSim scene.
    import rclpy

    rclpy_path = Path(getattr(rclpy, "__file__", "")).as_posix()
    evidence["rclpy_module"] = rclpy_path
    evidence["rclpy_bundled_humble"] = "humble/rclpy" in rclpy_path
    if not evidence["rclpy_bundled_humble"]:
        raise RuntimeError(
            "bundled Humble rclpy not on sys.path; run through "
            f"tools/isaac-ros-env.sh (got: {rclpy_path or 'none'})"
        )

    # The ONLY DDS participant: one rclpy node owning the shared clock
    # publisher, one namespaced publisher/subscription pair per robot and the
    # scene-state service (the validated in-process alternative).
    # Gravity feed-forward is the controller-layer compensation this arm's
    # position drives need to behave like a real motor controller (live
    # evidence 2026-10-02: placement press transmits reliably with it and
    # mostly stalls without it). Default ON; ROCOBRICK_GRAVITY_FEEDFORWARD=0
    # disables it for comparison runs. Read before any DDS resource exists.
    gravity_enabled = os.environ.get("ROCOBRICK_GRAVITY_FEEDFORWARD", "1") != "0"
    evidence["gravity_feedforward"] = {
        "enabled": gravity_enabled,
        "arm_limit_nm": 10.0,
        "finger_efforts": 0.0,
    }
    for channel in channels:
        channel.gravity_feedforward = gravity_enabled
        channel.record_dof_limits(
            evidence
            if args.scene_manifest is None
            else evidence["robots"][channel.robot_id]
        )
    evidence["scene_state_service"] = {
        "name": SCENE_STATE_SERVICE,
        "created": False,
    }
    render_enabled = os.environ.get("ROCOBRICK_HEADED") == "1"
    bridge = InProcessBridge(state, channels, robot_only=args.scene_manifest is None)
    # The Kit child's interrupted path reads the live channel counters from
    # here before its final evidence write.
    state.bridge = bridge
    evidence["scene_state_service"]["created"] = bool(
        bridge.scene_state_service_created
    )
    evidence["rclpy_started"] = True
    signal.signal(signal.SIGINT, state.request_stop)

    render_stride = 6
    loop_started = time.monotonic()
    deadline = loop_started + min(args.ready_timeout, args.time_limit)
    hard_deadline = loop_started + args.time_limit
    steps = 0
    last_publish = 0.0  # publish on the first readiness step
    while not (
        bridge.robots_ready
        and bridge.clock_and_joint_states_published
        and steps >= MIN_READY_STEPS
        and world.current_time > 0.0
    ):
        if steps >= args.max_steps:
            raise TimeoutError(f"readiness not reached within {args.max_steps} steps")
        if time.monotonic() >= deadline or time.monotonic() >= hard_deadline:
            raise TimeoutError("readiness not reached before the bounded deadline")
        state.check_stop()
        world.step(render=False)
        bridge.capture_pad_contacts(world)
        if render_enabled and steps % render_stride == 0:
            await state.render_viewport(world)
        steps += 1
        now = time.monotonic()
        if now - last_publish >= PUBLISH_PERIOD_S:
            bridge.publish_state(world)
            last_publish = now
        bridge.spin_once()

    evidence["ready"] = True
    evidence["steps_to_ready"] = steps
    evidence["sim_time_at_ready_s"] = round(float(world.current_time), 6)
    evidence["clock_published_at_ready"] = bridge.clock_published
    evidence["joint_states_published_at_ready"] = sum(
        channel.joint_states_published for channel in bridge.channels
    )
    evidence["readiness"] = {
        "all_robots_initialized": bridge.robots_ready,
        "clock_published": bridge.clock_published,
        "robots": {
            _channel_evidence_key(channel): {
                "joint_states_published": channel.joint_states_published,
                "initialized": channel.initialized,
            }
            for channel in bridge.channels
        },
    }
    ready_document: dict = {
        "schema": SCHEMA,
        "stage": "ready",
        "marker": READY_MARKER,
        "steps": steps,
        "elapsed_s": round(time.monotonic() - loop_started, 3),
    }
    if args.scene_manifest is None:
        ready_document["joint_dofs"] = list(bridge.channels[0].joint_names)
    else:
        ready_document["robots"] = {
            channel.robot_id: list(channel.joint_names) for channel in bridge.channels
        }
    print(json.dumps(ready_document, sort_keys=True), flush=True)

    # Hold phase: keep stepping so the external container probe can subscribe
    # /clock and /isaac_joint_states, publish /isaac_joint_command and verify movement.
    hold_started = time.monotonic()
    hold_steps = 0
    hold_samples: list[dict] = []
    while (
        hold_steps < args.hold_steps
        or time.monotonic() - hold_started < args.hold_seconds
    ):
        if steps >= args.max_steps:
            raise TimeoutError(f"hold exceeded {args.max_steps} total steps")
        if time.monotonic() >= hard_deadline:
            raise TimeoutError("hold exceeded the bounded time limit")
        state.check_stop()
        world.step(render=False)
        bridge.capture_pad_contacts(world)
        if render_enabled and steps % render_stride == 0:
            await state.render_viewport(world)
        steps += 1
        hold_steps += 1
        now = time.monotonic()
        if now - last_publish >= PUBLISH_PERIOD_S:
            bridge.publish_state(world)
            last_publish = now
        bridge.spin_once()
        if hold_steps % 120 == 0 and len(hold_samples) < 10:
            hold_samples.append(
                {
                    "step": steps,
                    "wall_s": round(time.monotonic() - hold_started, 3),
                    "sim_time_s": round(float(world.current_time), 3),
                    "playing": bool(timeline.is_playing()),
                }
            )
    evidence["hold_samples"] = hold_samples
    evidence.update(collect_counter_evidence(bridge.channels, bridge.clock_published))

    evidence["hold_completed"] = True
    evidence["hold"] = {
        "steps": hold_steps,
        "seconds_wall": round(time.monotonic() - hold_started, 3),
        "sim_time_final_s": round(float(world.current_time), 6),
    }
    evidence["steps_total"] = steps
    if args.scene_manifest is not None:
        for channel in bridge.channels:
            evidence["robots"][channel.robot_id].update(
                {
                    "initialized": channel.initialized,
                    "joint_states_published": channel.joint_states_published,
                    "joint_states_diag_published": channel.joint_states_diag_published,
                    "positions_nan": channel.positions_nan,
                    "commands_received": channel.commands_received,
                    "commands_applied": channel.commands_applied,
                    "commands_clamped": channel.commands_clamped,
                    "unknown_joint_names_ignored": list(channel.unknown_joint_names),
                }
            )
    # Persist before teardown: a native abort inside app.close() would
    # otherwise skip the finally-block write entirely.
    try:
        write_evidence(args.evidence_out, evidence)
    except OSError as exc:
        evidence["evidence_write_error"] = str(exc)


def write_evidence(path: Path, evidence: dict) -> None:
    """Persist the evidence JSON next to a printable copy on stdout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the bounded bridge inside the bricksim Kit launch regime.

    Phase 1 (fresh process): relaunch THIS bridge through ``isaacsim.main()``
    with the full experience and ``--exec`` self, then translate the Kit exit
    through the persisted evidence verdict. Phase 2 (Kit ``--exec`` child,
    same process): bootstrap bricksim exactly like ``bricksim._entry``, run
    the bridge as one coroutine on Kit's async engine, always write evidence,
    and shut the application down.

    Args:
        argv: Optional argument list overriding ``sys.argv``.

    Returns:
        0 when the persisted evidence records a passed run, else 1.
    """
    args = parse_args(argv)
    sys.argv = [sys.argv[0]]  # keep Kit from parsing our arguments
    started = time.monotonic()
    state = BridgeState()
    evidence = initial_evidence(args)
    state.evidence = evidence
    state.evidence_path = args.evidence_out

    if not _running_inside_kit():
        evidence["kit_relaunch"] = {
            "experience": KIT_EXPERIENCE,
            "exec_target": str(BRIDGE_FILE),
        }
        write_evidence(args.evidence_out, evidence)
        kit_argv = _kit_launch_argv(args)
        sys.argv = [sys.argv[0], *kit_argv]
        import isaacsim

        try:
            isaacsim.main()
        except SystemExit as kit_exit:
            kit_code = kit_exit.code
        else:  # pragma: no cover - isaacsim.main always raises SystemExit
            kit_code = 0
        try:
            final = json.loads(args.evidence_out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return kit_code if isinstance(kit_code, int) else 1
        return 0 if final.get("passed") is True else 1

    # Kit child: bootstrap the bricksim extension the same way the bricksim
    # launcher's _entry does, then schedule the bridge on Kit's async engine.
    import asyncio

    import bricksim  # noqa: F401
    import omni.kit.async_engine as async_engine

    async def _child_bridge() -> None:
        try:
            await run_bridge(args, state, evidence)
        except KeyboardInterrupt:
            # The A6 early-stop path: the session removes its own container and
            # then SIGINTs the host. Only this marker can be accepted by
            # evaluate_early_stop_accepted.
            evidence["stopped"] = EARLY_STOP_STOPPED
        except SystemExit as exc:
            # Kit can raise SystemExit from the async child before readiness.
            # Preserve the exact code instead of letting the parent report only
            # the generic "exited before p0.5" bootstrap failure.
            evidence["error"] = f"SystemExit from Kit child: code={exc.code!r}"
        except asyncio.CancelledError:
            # Generic Kit-child cancellation: never an accepted early stop.
            evidence["stopped"] = "kit_child_cancelled_orderly_stop"
            raise
        except Exception as exc:
            evidence["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            # Write evidence before requesting the Kit shutdown: the process
            # may not survive past get_app().shutdown().
            state.evidence = evidence
            state.evidence_path = args.evidence_out
            # An interrupted run must still carry the live per-robot command
            # counters, so the early-stop verdict below reads the channels'
            # exact state instead of stale or missing evidence.
            bridge = state.bridge
            if bridge is not None:
                evidence.update(
                    collect_counter_evidence(bridge.channels, bridge.clock_published)
                )
            evidence["elapsed_s"] = round(time.monotonic() - started, 3)
            try:
                write_evidence(args.evidence_out, evidence)
            except OSError as exc:
                evidence["evidence_write_error"] = str(exc)
            if state.exit_diagnostics_enabled and state.world is not None:
                try:
                    await asyncio.wait_for(
                        state.render_viewport(state.world), timeout=2
                    )
                except Exception as exc:
                    evidence["exit_pre_render_error"] = f"{type(exc).__name__}: {exc}"
            state.capture_exit_diagnostics("before-stop")
            # Defer app close until the stopped world has rendered its final
            # diagnostic frame. ROS/world/timeline teardown order is unchanged.
            state.defer_app_shutdown = state.exit_diagnostics_enabled
            evidence["cleanup"] = state.shutdown()
            if state.exit_diagnostics_enabled:
                try:
                    from omni.kit.app import get_app

                    for _ in range(2):
                        await asyncio.wait_for(get_app().next_update_async(), timeout=2)
                except Exception as exc:
                    evidence["exit_render_error"] = f"{type(exc).__name__}: {exc}"
                state.capture_exit_diagnostics("after-stop")
            # The Kit application itself is shut down right after this write;
            # there is no SimulationApp handle to close in the child regime.
            evidence["cleanup"]["app_close_initiated"] = True
            evidence["elapsed_s"] = round(time.monotonic() - started, 3)
            # An accepted early stop needs readiness, the orderly SIGINT
            # marker, no error and complete cleanup. Controller hold commands
            # remain in the counters but are not skill-operation dispatches.
            evidence["early_stop_accepted"] = evaluate_early_stop_accepted(evidence)
            evidence["passed"] = evaluate_passed(evidence)
            try:
                write_evidence(args.evidence_out, evidence)
            except OSError as exc:
                evidence["evidence_write_error"] = str(exc)
            print(json.dumps(evidence, sort_keys=True), flush=True)
            from omni.kit.app import get_app

            get_app().shutdown()

    async_engine.run_coroutine(_child_bridge())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
