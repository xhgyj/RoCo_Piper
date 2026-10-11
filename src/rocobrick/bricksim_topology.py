"""Convert structural STAP facts to BrickSim topology without motion geometry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from rocobrick.stap_contract import StapPlan, StructuralPart
from rocobrick.expanded_bricks import expanded_lego_library


class TopologyError(ValueError):
    pass


@dataclass(frozen=True)
class BaseplateSpec:
    size_studs: tuple[int, int]
    color: str


@dataclass(frozen=True)
class BrickSimTopologyBundle:
    goal: Mapping[str, object]
    initial: Mapping[str, object]
    loose: Mapping[str, object]
    topology_id_by_instance: Mapping[str, int]
    instance_by_topology_id: Mapping[int, str]
    expected_goal_connections: tuple[Mapping[str, object], ...]
    expected_initial_connections: tuple[Mapping[str, object], ...]


TopologyConverter = Callable[..., Mapping[str, object]]
ColorParser = Callable[[str], tuple[int, int, int]]


def _default_converter() -> tuple[TopologyConverter, ColorParser]:
    from bricksim.colors import parse_color
    from bricksim.topology.legolization import legolization_json_to_topology_json, load_default_lego_library

    library = expanded_lego_library(load_default_lego_library())

    def converter(structure, **kwargs):
        return legolization_json_to_topology_json(structure, lego_library=library, **kwargs)

    return converter, parse_color


def _grid_record(part: StructuralPart) -> dict[str, object]:
    return {
        "brick_id": part.brick_id,
        "x": part.x,
        "y": part.y,
        "z": part.z,
        "ori": part.orientation_quarter_turns,
        "color": part.color,
    }


def _subset(topology: Mapping[str, object], keep: set[int], *, connections: bool) -> dict[str, object]:
    result = dict(topology)
    result["parts"] = [part for part in topology["parts"] if part["id"] in keep]
    result["connections"] = [
        connection for connection in topology["connections"]
        if connections and connection["stud_id"] in keep and connection["hole_id"] in keep
    ]
    result["pose_hints"] = [
        hint for hint in topology.get("pose_hints", []) if hint["part"] in keep
    ] if connections else []
    return result


def build_bricksim_topologies(
    plan: StapPlan,
    baseplate: BaseplateSpec,
    *,
    converter: TopologyConverter | None = None,
    parse_color: ColorParser | None = None,
) -> BrickSimTopologyBundle:
    """Build full, initially connected, and loose-part topology documents.

    BrickSim part id 0 is reserved for the baseplate.  Physical instance IDs
    are deterministically sorted and assigned 1..N, independent of source keys.
    """
    if min(baseplate.size_studs) <= 0:
        raise TopologyError("baseplate dimensions must be positive")
    if converter is None or parse_color is None:
        default_converter, default_color = _default_converter()
        converter = converter or default_converter
        parse_color = parse_color or default_color

    ordered = sorted(plan.goal_parts.values(), key=lambda part: part.physical_instance_id)
    topology_id_by_instance = {
        part.physical_instance_id: index for index, part in enumerate(ordered, start=1)
    }
    source = {
        str(topology_id_by_instance[part.physical_instance_id]): _grid_record(part)
        for part in ordered
    }
    colors = [parse_color(source[str(index)]["color"]) for index in range(1, len(source) + 1)]
    goal = dict(converter(
        source,
        color=colors,
        include_base_plate=True,
        base_plate_size=list(baseplate.size_studs),
        base_plate_color=parse_color(baseplate.color),
    ))
    returned_ids = {part["id"] for part in goal.get("parts", [])}
    expected_ids = set(range(len(ordered) + 1))
    if returned_ids != expected_ids:
        raise TopologyError(
            f"BrickSim converter returned part ids {sorted(returned_ids)}, expected {sorted(expected_ids)}"
        )

    initial_ids = {0} | {
        topology_id_by_instance[instance] for instance in plan.initially_assembled
    }
    # The runtime scene contains every goal part that is not already present
    # in the initial structure.  This is important for partial robot plans
    # (for example TRANSIT -> PICK without PLACE_DOWN), where the carrier has
    # no structural task yet but the target still must be a loose brick.
    initial_instance_ids = set(plan.initially_assembled)
    loose_ids = {
        topology_id
        for instance_id, topology_id in topology_id_by_instance.items()
        if instance_id not in initial_instance_ids
    }
    initial = _subset(goal, initial_ids, connections=True)
    loose = _subset(goal, loose_ids, connections=False)
    goal_connections = tuple(dict(connection) for connection in goal["connections"])
    initial_connections = tuple(dict(connection) for connection in initial["connections"])
    return BrickSimTopologyBundle(
        goal=goal,
        initial=initial,
        loose=loose,
        topology_id_by_instance=topology_id_by_instance,
        instance_by_topology_id={value: key for key, value in topology_id_by_instance.items()},
        expected_goal_connections=goal_connections,
        expected_initial_connections=initial_connections,
    )
