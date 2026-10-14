"""Shared elevated feeding platforms and deterministic loose-brick placement."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FeedingPlatform:
    """One axis-aligned physical platform in assembly-world coordinates."""

    robot_id: str
    center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    top_z_m: float
    thickness_m: float

    @property
    def prim_path(self) -> str:
        """Return the stable environment-collider path."""
        return f"/World/feeding_platforms/{self.robot_id}"

    @property
    def minimum_m(self) -> tuple[float, float, float]:
        """Return the platform's lower collision-box corner."""
        return (
            self.center_xy_m[0] - self.size_xy_m[0] / 2,
            self.center_xy_m[1] - self.size_xy_m[1] / 2,
            self.top_z_m - self.thickness_m,
        )

    @property
    def maximum_m(self) -> tuple[float, float, float]:
        """Return the platform's upper collision-box corner."""
        return (
            self.center_xy_m[0] + self.size_xy_m[0] / 2,
            self.center_xy_m[1] + self.size_xy_m[1] / 2,
            self.top_z_m,
        )


@dataclass(frozen=True)
class FeedingSlot:
    """An original loose brick's slot, with no duplication or task rewriting."""

    position_xy_m: tuple[float, float]
    platform: FeedingPlatform
    preferred_robot_id: str
    distance_to_robot_base_m: float

    def evidence(self) -> dict[str, object]:
        """Return JSON-safe initialization provenance."""
        return {
            "allocator": "feeding-platforms/v1",
            "preferred_robot_id": self.preferred_robot_id,
            "platform_robot_id": self.platform.robot_id,
            "platform_prim_path": self.platform.prim_path,
            "position_xy_m": list(self.position_xy_m),
            "support_top_z_m": self.platform.top_z_m,
            "distance_to_robot_base_m": self.distance_to_robot_base_m,
        }


def _number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two numbers")
    return _number(value[0], name), _number(value[1], name)


def parse_feeding_platforms(
    document: object, robot_ids: Sequence[str]
) -> tuple[FeedingPlatform, ...]:
    """Parse explicit platforms; missing configuration preserves ground slots.

    Returns:
        Validated platforms, with one region per configured robot.
    """
    if document is None:
        return ()
    if not isinstance(document, list) or not document:
        raise ValueError("Feeding_Platforms must be a nonempty list")
    platforms = []
    owners = set()
    required = {"Robot_ID", "Center_XY", "Size_XY", "Top_Z", "Thickness"}
    for raw in document:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("Feeding_Platforms entry keys are invalid")
        owner = raw["Robot_ID"]
        if not isinstance(owner, str) or owner not in robot_ids or owner in owners:
            raise ValueError("Feeding_Platforms robot must be configured and unique")
        if not owner.replace("_", "").isalnum():
            raise ValueError("Feeding_Platforms robot is not a valid prim name")
        center = _pair(raw["Center_XY"], "platform center")
        size = _pair(raw["Size_XY"], "platform size")
        top = _number(raw["Top_Z"], "platform top")
        thickness = _number(raw["Thickness"], "platform thickness")
        if min(size) <= 0 or top <= 0 or thickness <= 0 or thickness > top:
            raise ValueError("platform sizes/heights must be positive and above z=0")
        platform = FeedingPlatform(owner, center, size, top, thickness)
        for previous in platforms:
            if all(
                abs(center[i] - previous.center_xy_m[i])
                < (size[i] + previous.size_xy_m[i]) / 2
                for i in (0, 1)
            ):
                raise ValueError("feeding platforms must not overlap")
        platforms.append(platform)
        owners.add(owner)
    if owners != set(robot_ids):
        raise ValueError("Feeding_Platforms must cover every configured robot")
    return tuple(sorted(platforms, key=lambda platform: platform.robot_id))


def allocate_feeding_slots(
    physical_instance_ids: Sequence[str],
    *,
    platforms: Sequence[FeedingPlatform],
    footprints_xy_m: Mapping[str, tuple[float, float]],
    preferred_robots: Mapping[str, str],
    robot_base_xy: Mapping[str, tuple[float, float]],
    board_center_xy: tuple[float, float],
    board_half_extent_xy: tuple[float, float],
    minimum_spacing_m: float = 0.12,
    edge_clearance_m: float = 0.01,
    minimum_robot_distance_m: float = 0.25,
) -> Mapping[str, FeedingSlot]:
    """Balance original loose bricks across whole platforms without prescribing IK.

    Returns:
        Fully supported, non-overlapping slots or an explicit capacity failure.
    """
    if not platforms or len(set(physical_instance_ids)) != len(physical_instance_ids):
        raise ValueError("feeding platforms and unique loose identities are required")
    if min(minimum_spacing_m, edge_clearance_m, minimum_robot_distance_m) <= 0:
        raise ValueError("feeding clearances must be positive")
    for platform in platforms:
        if platform.robot_id not in robot_base_xy:
            raise ValueError("feeding platform robot is not configured")
        if all(
            abs(platform.center_xy_m[i] - board_center_xy[i])
            < platform.size_xy_m[i] / 2 + board_half_extent_xy[i] + edge_clearance_m
            for i in (0, 1)
        ):
            raise ValueError("feeding platform overlaps the baseplate clearance")
        if any(
            all(
                abs(base[i] - platform.center_xy_m[i])
                < platform.size_xy_m[i] / 2 + 0.09
                for i in (0, 1)
            )
            for base in robot_base_xy.values()
        ):
            raise ValueError("feeding platform overlaps a robot base clearance")
    slots: dict[str, FeedingSlot] = {}
    occupied: list[tuple[tuple[float, float], tuple[float, float]]] = []
    counts = {platform.robot_id: 0 for platform in platforms}
    for identity in sorted(physical_instance_ids):
        if identity not in footprints_xy_m or identity not in preferred_robots:
            raise ValueError("feeding brick footprint/preferred robot is missing")
        footprint = _pair(footprints_xy_m[identity], "brick footprint")
        if min(footprint) <= 0 or preferred_robots[identity] not in robot_base_xy:
            raise ValueError("feeding footprint or preferred robot is invalid")
        selected = None
        for platform in sorted(
            platforms,
            key=lambda item: (
                counts[item.robot_id],
                item.robot_id != preferred_robots[identity],
                item.robot_id,
            ),
        ):
            bounds = tuple(
                (platform.size_xy_m[i] - footprint[i]) / 2 - edge_clearance_m
                for i in (0, 1)
            )
            if min(bounds) < 0:
                continue
            candidates = [
                (
                    platform.center_xy_m[0] + x,
                    platform.center_xy_m[1] + y,
                )
                for x in (0.0, -bounds[0], bounds[0])
                for y in (0.0, -bounds[1], bounds[1])
            ]
            candidates.sort(
                key=lambda point: (math.dist(point, platform.center_xy_m), point)
            )
            for point in candidates:
                distance = math.dist(point, robot_base_xy[platform.robot_id])
                if distance < minimum_robot_distance_m:
                    continue
                if any(
                    math.dist(point, previous) < minimum_spacing_m
                    or all(
                        abs(point[i] - previous[i])
                        < (footprint[i] + previous_size[i]) / 2 + edge_clearance_m
                        for i in (0, 1)
                    )
                    for previous, previous_size in occupied
                ):
                    continue
                selected = FeedingSlot(
                    point, platform, preferred_robots[identity], distance
                )
                break
            if selected is not None:
                break
        if selected is None:
            raise ValueError(
                f"no elevated feeding slot for {identity}: platform capacity"
            )
        slots[identity] = selected
        occupied.append((selected.position_xy_m, footprint))
        counts[selected.platform.robot_id] += 1
    return slots
