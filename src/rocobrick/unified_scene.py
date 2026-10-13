"""Unified collision-scene readback binding for combined plan-only requests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, replace

from rocobrick.coordinated_session import CoordinatedSessionRequest
from rocobrick.planning_scene import PlanningSceneUpdate, box_from_aabb, scene_digest


def readback_digest(objects: Mapping[str, object]) -> str:
    """Hash complete read-back box geometry, rejecting unsupported collision forms.

    Returns:
        Canonical digest of the world collision objects.
    """
    document = {}
    for identity, item in objects.items():
        if (
            len(item.primitives) != 1
            or len(item.primitive_poses) != 1
            or item.meshes
            or item.planes
        ):
            raise ValueError("unified world readback contains unsupported geometry")
        primitive = item.primitives[0]
        pose = item.primitive_poses[0]
        document[identity] = {
            "frame": item.header.frame_id,
            "type": primitive.type,
            "dimensions": list(primitive.dimensions),
            "position": [pose.position.x, pose.position.y, pose.position.z],
            "orientation": [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
        }
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class UnifiedSceneBinding:
    """Own serialized scene application, full readback and monotonic version."""

    def __init__(self, transport: object):
        """Bind one combined-scene service transport."""
        self.transport = transport
        self.version = 0
        self.digest = None

    def sync(self, request: CoordinatedSessionRequest) -> Mapping[str, object]:
        """Apply the complete measured world, then bind the actual readback.

        Returns:
            Versioned planning-scene evidence distinct from the snapshot digest.
        """
        snapshot = request.scene_snapshot
        bricks = snapshot.get("bricks")
        if not isinstance(bricks, Mapping) or not bricks:
            raise ValueError("unified snapshot lacks measured world objects")
        if snapshot.get("holding") or snapshot.get("supporting"):
            raise ValueError(
                "attached/support scene admission requires explicit physical facts"
            )
        attachments = {}
        contact_pairs = set()
        for robot, facts in request.physical_facts.items():
            for key, proof in facts.items():
                if key.startswith("holding:"):
                    identity = key.removeprefix("holding:")
                    if (
                        identity not in bricks
                        or identity in attachments
                        or not isinstance(proof, Mapping)
                    ):
                        raise ValueError("holding attachment ownership is invalid")
                    payload = proof.get("attachment")
                    if (
                        not isinstance(payload, Mapping)
                        or payload.get("object_id") != identity
                        or payload.get("link_name") != f"{robot}_rocobrick_tcp"
                        or payload.get("frame_id") != f"{robot}_rocobrick_tcp"
                    ):
                        raise ValueError(
                            "holding requires measured robot-scoped attachment geometry"
                        )
                    attached_result = self.transport.apply_attached_planning_scene(
                        payload, request.timeout_s
                    )
                    if attached_result.get("passed") is not True:
                        raise ValueError("combined attachment readback failed")
                    attachments[identity] = dict(payload)
                elif key.startswith("supporting:"):
                    if (
                        not isinstance(proof, Mapping)
                        or not proof
                        or request.targets[robot].get("maintain_pose") is not True
                    ):
                        raise ValueError(
                            "supporting participant requires proven support "
                            "and continuous pose constraint"
                        )
                    identity = key.removeprefix("supporting:")
                    for index in (1, 2):
                        contact_pairs.add(
                            tuple(sorted((identity, f"{robot}_gripper_link{index}")))
                        )
        boxes = []
        for identity, brick in bricks.items():
            if identity in attachments:
                continue
            if not isinstance(brick, Mapping):
                raise ValueError("unified brick observation is invalid")
            boxes.append(
                box_from_aabb(
                    object_id=str(identity),
                    frame_id="dual_world",
                    minimum_m=brick["aabb_min"],
                    maximum_m=brick["aabb_max"],
                )
            )
        environment = snapshot.get("environment_colliders")
        if isinstance(environment, Mapping) and environment:
            for identity, envelope in environment.items():
                boxes.append(
                    box_from_aabb(
                        object_id=f"environment:{identity}",
                        frame_id="dual_world",
                        minimum_m=envelope["aabb_min"],
                        maximum_m=envelope["aabb_max"],
                    )
                )
        else:
            ground = snapshot.get("ground")
            if not isinstance(ground, Mapping):
                raise ValueError(
                    "unified scene requires a measured ground/environment envelope"
                )
            boxes.append(
                box_from_aabb(
                    object_id="unified_ground",
                    frame_id="dual_world",
                    minimum_m=ground["aabb_min"],
                    maximum_m=ground["aabb_max"],
                )
            )
        update = PlanningSceneUpdate(
            scene_id=request.coordination_id,
            boxes=tuple(boxes),
            expected_scene_digest="0" * 64,
            allowed_collision_pairs=tuple(sorted(contact_pairs)),
        )
        update = replace(update, expected_scene_digest=scene_digest(update))
        result = self.transport.apply_planning_scene(update, request.timeout_s)
        if result.details.get("readback_confirmed") is not True:
            raise ValueError(f"unified scene application failed: {result.details}")
        expected_ids = {box.object_id for box in boxes}
        applied_ids = set(getattr(result, "applied_object_ids", expected_ids))
        if applied_ids != expected_ids:
            raise ValueError(
                "unified scene application object set mismatch: "
                f"expected={sorted(expected_ids)} observed={sorted(applied_ids)}"
            )
        digest = self.transport.unified_scene_fingerprint(request.timeout_s)
        if digest != self.digest:
            self.version += 1
            self.digest = digest
        return {
            "snapshot_digest": request.scene_digest,
            "scene_digest": digest,
            "scene_version": self.version,
            "applied": asdict(update),
            "readback_confirmed": True,
            "attachments": attachments,
        }

    def refresh_version(self, timeout_s: float = 5.0) -> int:
        """Refresh the monotonic scene version from current MoveIt readback.

        Returns:
            The current version, advanced only by a changed actual readback.
        """
        digest = self.transport.unified_scene_fingerprint(timeout_s)
        if digest != self.digest:
            self.version += 1
            self.digest = digest
        return self.version

    def validate(self, proof: Mapping[str, object]) -> bool:
        """Read back the actual scene after planning, without accepting stale versions.

        Returns:
            Whether the complete world still matches the admitted version/digest.
        """
        objects, _ = self.transport._read_world_collision_objects(
            5.0, self.transport._resolve_factories()
        )
        return (
            objects is not None
            and proof.get("scene_version") == self.version
            and proof.get("scene_digest")
            == self.digest
            == self.transport.unified_scene_fingerprint(5.0)
        )
