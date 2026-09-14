import copy
from hashlib import sha256
import json

import pytest

from rocobrick.stap_contract import StapContractError, parse_stap_plan


def payload():
    old = {
        "physical_instance_id": "brick-old", "source_goal_key": "8",
        "source_start_key": "7", "brick_id": 12, "x": 0, "y": 0,
        "z": 0, "ori": 0, "color": "Blue",
    }
    new = {
        "physical_instance_id": "brick-new", "source_goal_key": "9",
        "brick_id": 2, "x": 0, "y": 0, "z": 1, "ori": 0, "color": "Red",
    }
    value = {
        "schema": "assembly-plan/v2",
        "plan_id": "pending",
        "source": {"goal_sha256": "a" * 64, "start_sha256": "b" * 64},
        "planner": {"name": "hierarchical-ltl-stap", "core": "Hierarchical-LTL-STAP"},
        "robots": [{"robot_id": "piper_0"}, {"robot_id": "piper_1"}],
        "structural_semantics": {
            "coordinate_system": "grid", "position_unit": "stud/layer grid units",
            "orientation": {"encoding": "quarter_turn", "values": [0, 1], "one_means": "axis swap"},
            "motion_geometry": "omitted",
        },
        "goal_structure": {"instances": [old, new]},
        "initial_structure": {"facts": [{
            "fact_id": "assembled:brick-old", "predicate": "assembled",
            "physical_instance_id": "brick-old", "source_start_key": "7",
        }]},
        "parts": [{
            **new, "task_id": "assemble-new", "skill_type": "PLACE_DOWN",
            "primary_robot": "piper_1", "supporting_robot": None,
        }],
        "dependencies": [],
        "initial_dependencies": [["brick-old", "assemble-new"]],
        "allocation": {"piper_0": [], "piper_1": ["assemble-new"]},
        "execution_layers": [["assemble-new"]],
        "planned_schedule": [[{"robot": "piper_1", "task": "assemble-new"}]],
        "specification": {
            "type": "hierarchical-ltl", "root_formula": "<> p1a",
            "hierarchy": [{"p0": "<> p1a"}, {"p1a": "<> assemble1"}],
        },
        "diagnostics": {},
    }
    material = {
        "source": value["source"], "robots": value["robots"], "parts": value["parts"],
        "dependencies": value["dependencies"],
        "initial_dependencies": value["initial_dependencies"],
        "allocation": value["allocation"], "execution_layers": value["execution_layers"],
        "planned_schedule": value["planned_schedule"],
        "hierarchy": value["specification"]["hierarchy"],
    }
    value["plan_id"] = "assembly-plan-" + sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()[:24]
    return value


def test_preserves_hltl_and_separates_initial_structure_from_tasks():
    plan = parse_stap_plan(payload())
    assert plan.hltl_root_formula == "<> p1a"
    assert plan.hltl_hierarchy[1]["p1a"] == "<> assemble1"
    assert plan.initially_assembled == {"brick-old"}
    assert plan.tasks["assemble-new"].initial_supports == {"brick-old"}


def test_rejects_task_that_does_not_match_goal_instance():
    value = payload()
    value["parts"][0]["color"] = "Green"
    with pytest.raises(StapContractError, match="plan_id"):
        parse_stap_plan(value)


def test_rejects_hltl_root_rewrite():
    value = copy.deepcopy(payload())
    value["specification"]["root_formula"] = "<> changed"
    with pytest.raises(StapContractError, match="root_formula"):
        parse_stap_plan(value)


def test_rejects_initial_and_moving_partition_overlap():
    value = payload()
    value["parts"][0]["physical_instance_id"] = "brick-old"
    with pytest.raises(StapContractError):
        parse_stap_plan(value)
