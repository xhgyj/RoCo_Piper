from rocobrick.bricksim_topology import BaseplateSpec, build_bricksim_topologies
from rocobrick.stap_contract import parse_stap_plan
from test_stap_contract import payload


def _converter(source, **kwargs):
    assert list(source) == ["1", "2"]
    return {
        "schema": "bricksim/lego_topology@2",
        "parts": [
            {"id": 0, "type": "brick", "payload": {}},
            {"id": 1, "type": "brick", "payload": {}},
            {"id": 2, "type": "brick", "payload": {}},
        ],
        "connections": [
            {"id": 0, "stud_id": 0, "hole_id": 2},
            {"id": 1, "stud_id": 2, "hole_id": 1},
        ],
        "pose_hints": [{"part": 0}],
    }


def test_topology_separates_initial_connections_and_loose_parts():
    plan = parse_stap_plan(payload())
    bundle = build_bricksim_topologies(
        plan, BaseplateSpec((32, 32), "Gray"), converter=_converter,
        parse_color=lambda unused: (1, 2, 3),
    )
    old_id = bundle.topology_id_by_instance["brick-old"]
    new_id = bundle.topology_id_by_instance["brick-new"]
    assert {part["id"] for part in bundle.initial["parts"]} == {0, old_id}
    assert {part["id"] for part in bundle.loose["parts"]} == {new_id}
    assert bundle.loose["connections"] == []
    assert len(bundle.initial["connections"]) == 1


def test_mapping_is_independent_of_source_goal_key():
    first = payload()
    first_plan = parse_stap_plan(first)
    first_map = build_bricksim_topologies(
        first_plan, BaseplateSpec((32, 32), "Gray"), converter=_converter,
        parse_color=lambda unused: (1, 2, 3),
    ).topology_id_by_instance
    assert first_map == {"brick-new": 1, "brick-old": 2}
