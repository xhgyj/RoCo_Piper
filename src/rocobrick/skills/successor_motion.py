"""Shared skill boundary movement; no skill-specific scheduler or TRANSIT."""

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Protocol

from rocobrick.coordinated_session import (
    FreeSpaceMotionEvidence,
    FreeSpaceMotionRequest,
)
from rocobrick.planning_scene import PlanningSceneUpdate, box_from_aabb, scene_digest
from rocobrick.skills.observation import _pose
from rocobrick.skills.pre_phase_targets import serialize_target


class BoundaryTransport(Protocol):
    """Existing facade's shared observation and combined execution seams."""

    @property
    def robot_ids(self) -> tuple[str, ...]:
        """Return all robots owned by this persistent session."""
        ...

    async def free_space_context(
        self, participants: tuple[str, ...]
    ) -> tuple[Mapping[str, object], int, Mapping[str, float]]:
        """Read the shared snapshot, planning version and complete joints."""
        ...

    async def execute_free_space(
        self, request: FreeSpaceMotionRequest
    ) -> FreeSpaceMotionEvidence:
        """Execute one combined plan and report physical terminal evidence."""
        ...


def contact_scene_payload(shared, *, scene_id: str, attached_ids=(), allowed_pairs=()):
    """Preserve the whole measured world while enabling only named contact pairs.

    Returns:
        A digest-bound planning scene diff, leaving robot collisions enabled.
    """
    boxes = [
        box_from_aabb(
            object_id=identity,
            frame_id="dual_world",
            minimum_m=brick["aabb_min"],
            maximum_m=brick["aabb_max"],
        )
        for identity, brick in shared["bricks"].items()
        if identity not in attached_ids
    ]
    environment = shared.get("environment_colliders")
    if isinstance(environment, Mapping) and environment:
        boxes.extend(
            box_from_aabb(
                object_id=f"environment:{identity}",
                frame_id="dual_world",
                minimum_m=brick["aabb_min"],
                maximum_m=brick["aabb_max"],
            )
            for identity, brick in environment.items()
        )
    else:
        ground = shared["ground"]
        boxes.append(
            box_from_aabb(
                object_id="unified_ground",
                frame_id="dual_world",
                minimum_m=ground["aabb_min"],
                maximum_m=ground["aabb_max"],
            )
        )
    pairs = tuple(sorted({tuple(sorted(pair)) for pair in allowed_pairs}))
    update = PlanningSceneUpdate(
        scene_id=scene_id,
        boxes=tuple(boxes),
        expected_scene_digest="0" * 64,
        allowed_collision_pairs=pairs,
    )
    return asdict(replace(update, expected_scene_digest=scene_digest(update)))


async def move_boundary(
    transport: BoundaryTransport,
    operation,
    context,
    target,
    *,
    phase: str,
    physical_facts=None,
    active_requests=None,
) -> FreeSpaceMotionEvidence:
    """Move the executor while all other participants keep measured joints.

    Returns:
        The combined request's physical arrival and terminal proof.
    """
    values = operation.target.values if operation.target else {}
    peers = [values.get("placement_robot_id"), values.get("support_robot_id")]
    participants = tuple(
        dict.fromkeys((*operation.participants, *(robot for robot in peers if robot)))
    )
    scene, version, start = await transport.free_space_context(participants)
    targets = {
        operation.executor_robot_id: serialize_target(
            target, operation.executor_robot_id
        )
    }
    for robot in participants:
        if robot in targets:
            continue
        # A maintain_pose target is solved from this robot's measured joints;
        # its TCP payload is informational, never a new pose constraint.
        pose = _pose(scene["robots"][robot]["tcp_pose"], "held participant TCP")
        targets[robot] = {
            "maintain_pose": True,
            "tcp_pose": {
                "position": pose.position_xyz,
                "orientation_xyzw": pose.orientation_xyzw,
                "frame_id": "dual_world",
                "tip_link": f"{robot}_rocobrick_tcp",
                "runtime_model_sha256": target.runtime_model_sha256,
                "source_sha256": target.source_sha256,
                "position_tolerance_m": target.position_tolerance_m,
                "orientation_tolerance_rad": target.orientation_tolerance_rad,
            },
        }
    request = FreeSpaceMotionRequest(
        motion_id=f"{operation.operation_id}:{phase}",
        session_robots=transport.robot_ids,
        participants=participants,
        scene_snapshot=scene,
        scene_version=version,
        targets=targets,
        start_state=start,
        timeout_s=120.0,
        physical_facts=physical_facts or {},
    )
    if active_requests is not None:
        active_requests[operation.operation_id] = request.motion_id
    try:
        return await transport.execute_free_space(request)
    finally:
        if active_requests is not None:
            active_requests.pop(operation.operation_id, None)
