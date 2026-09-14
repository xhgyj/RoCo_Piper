from __future__ import annotations

import copy

import pytest

from rocobrick.contracts import ContractError, parse_plan, parse_scene, validate_binding


def test_arbitrary_task_ids_and_explicit_instances_are_accepted(valid_payloads):
    plan_payload, scene_payload = valid_payloads
    plan, scene = parse_plan(plan_payload), parse_scene(scene_payload)
    validate_binding(plan, scene)
    assert set(plan.tasks) == {"foundation:left", "cap@right", "joined/top"}


def test_v1_is_rejected_without_conversion(valid_payloads):
    plan_payload, _ = valid_payloads
    plan_payload = copy.deepcopy(plan_payload)
    plan_payload["schema"] = "assembly-plan/v1"
    with pytest.raises(ContractError, match="automatic conversion is disabled"):
        parse_plan(plan_payload)


def test_unimplemented_skill_is_rejected_before_motion(valid_payloads):
    plan_payload, _ = valid_payloads
    plan_payload = copy.deepcopy(plan_payload)
    plan_payload["tasks"][0]["skill_type"] = "HANDOVER"
    with pytest.raises(ContractError, match="CAPABILITY_UNAVAILABLE"):
        parse_plan(plan_payload)


def test_scene_binding_does_not_guess_by_brick_type(valid_payloads):
    plan_payload, scene_payload = copy.deepcopy(valid_payloads)
    scene_payload["parts"][0]["part_instance_id"] = "another-brick-of-same-type"
    with pytest.raises(ContractError, match="missing part_instance_id physical-brick-44"):
        validate_binding(parse_plan(plan_payload), parse_scene(scene_payload))


def test_dependency_cycle_is_rejected(valid_payloads):
    plan_payload, _ = valid_payloads
    plan_payload = copy.deepcopy(plan_payload)
    plan_payload["tasks"][0]["predecessors"] = ["joined/top"]
    plan_payload["tasks"][0]["level"] = 2
    plan_payload["tasks"][0]["schedule_slot"] = 2
    with pytest.raises(ContractError, match="cycle"):
        parse_plan(plan_payload)
