"""Geometry-derived pre-contact targets from one shared live observation."""

from __future__ import annotations

from collections.abc import Mapping

from rocobrick.adapters.moveit.transit import TransitContext
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.observation import (
    _pose,
    parse_pick_observation,
    parse_place_observation,
)
from rocobrick.skills.placement_geometry import (
    interface_place_targets,
    interface_place_up_targets,
    support_top_target,
)
from rocobrick.skills.targets import (
    _tcp_pose_target,
    pick_targets,
    transit_target,
    world_orientation_to_base,
    world_position_to_base,
)


def serialize_target(target, robot):
    """Serialize exact calibrated target provenance and tolerance.

    Returns:
        A robot-scoped combined planner target.
    """
    return {
        "tcp_pose": {
            "position": target.position,
            "orientation_xyzw": target.orientation_xyzw,
            "frame_id": f"{robot}_base_link",
            "tip_link": f"{robot}_rocobrick_tcp",
            "runtime_model_sha256": target.runtime_model_sha256,
            "source_sha256": target.source_sha256,
            "position_tolerance_m": target.position_tolerance_m,
            "orientation_tolerance_rad": target.orientation_tolerance_rad,
        }
    }


def derive_wire_targets(operation_document, shared, context_documents):
    """Decode calibrated execution context separately from immutable operations.

    A wire document may carry cohort_operations. Their executor targets
    are unioned into one combined request; non-executors retain the shared
    hold-pose target from the primary operation.

    Returns:
        All participants' calibrated targets for the single combined request.
    """
    cohort = operation_document.get("cohort_operations")
    if isinstance(cohort, list) and cohort:
        combined = {}
        primary = dict(operation_document)
        primary.pop("cohort_operations", None)
        documents = [primary, *cohort]
        for document in documents:
            if not isinstance(document, Mapping):
                raise ValueError("cohort operation must be a mapping")
            single = dict(document)
            single.pop("cohort_operations", None)
            derived = derive_wire_targets(single, shared, context_documents)
            robot = single.get("executor_robot_id")
            if not isinstance(robot, str) or robot not in derived:
                raise ValueError("cohort operation executor target is missing")
            combined[robot] = derived[robot]
        primary_targets = derive_wire_targets(primary, shared, context_documents)
        for robot, target in primary_targets.items():
            combined.setdefault(robot, target)
        if not combined:
            raise ValueError("cohort target derivation produced no targets")
        return combined
    from rocobrick.robot_operation_plan import OperationTarget
    from rocobrick.skills.targets import RobotBaseFrame, SkillParams, TcpProvenance

    operation = RobotOperation(
        operation_id=operation_document["operation_id"],
        source_task_id=operation_document["source_task_id"],
        skill=operation_document["skill"],
        executor_robot_id=operation_document["executor_robot_id"],
        participants=tuple(operation_document["participants"]),
        role_bindings=operation_document["role_bindings"],
        target=OperationTarget(operation_document["target"])
        if operation_document["target"]
        else None,
        parameters=operation_document["parameters"],
        preconditions=tuple(operation_document["preconditions"]),
        effects=tuple(operation_document["effects"]),
        resources=tuple(operation_document["resources"]),
        evidence_requirements=tuple(operation_document["evidence_requirements"]),
    )
    contexts = {}
    for robot, document in context_documents.items():
        base = document["base"]
        contexts[robot] = TransitContext(
            RobotBaseFrame(
                tuple(base["position_xyz"]), tuple(base["orientation_xyzw"])
            ),
            TcpProvenance(**document["provenance"]),
            SkillParams(**document["params"]),
        )
    return derive_pre_phase_targets(operation, shared, contexts)


def derive_pre_phase_targets(
    operation: RobotOperation,
    shared: Mapping[str, object],
    contexts: Mapping[str, TransitContext],
):
    """Derive skill boundaries while keeping every non-moving participant fixed.

    Returns:
        Targets for the complete combined arm union.
    """
    robot = operation.executor_robot_id
    context = contexts[robot]
    values = operation.target.values if operation.target else {}
    session = shared["session_id"]
    manifest = shared["manifest_sha256"]
    connection_transit = (
        operation.skill == "TRANSIT"
        and isinstance(values.get("to"), Mapping)
        and values["to"].get("kind") == "connection"
    )
    if operation.skill in {"PICK", "TRANSIT"} and not connection_transit:
        destination = values.get("to")
        physical = (
            destination.get("physical_instance_id")
            if isinstance(destination, Mapping)
            else values.get("physical_instance_id")
        )
        if not physical:
            raise ValueError("connection transport requires explicit interface binding")
        sample = parse_pick_observation(
            shared,
            robot_id=robot,
            physical_instance_id=physical,
            session_id=session,
            manifest_sha256=manifest,
        )
        target = (
            pick_targets(
                sample.aabb_min,
                sample.aabb_max,
                context.base,
                context.provenance,
                context.params,
            ).pregrasp
            if operation.skill == "PICK"
            or (
                operation.skill == "TRANSIT"
                and not values.get("carrying_physical_instance_id")
            )
            else transit_target(
                sample.aabb_min,
                sample.aabb_max,
                context.base,
                context.provenance,
                context.params,
            )
        )
    elif operation.skill == "SUPPORT_TOP":
        sample = parse_pick_observation(
            shared,
            robot_id=robot,
            physical_instance_id=values["upper_physical_instance_id"],
            session_id=session,
            manifest_sha256=manifest,
        )
        target = support_top_target(
            sample, context.base, context.provenance, context.params, approach=True
        )
    elif operation.skill in {"PLACE_DOWN", "PLACE_UP"} or (
        operation.skill == "TRANSIT"
        and isinstance(values.get("to"), Mapping)
        and values["to"].get("kind") == "connection"
    ):
        if connection_transit:
            pairs = shared.get("placement_interfaces", {}).get("pairs", {})
            matches = [
                pair
                for pair in pairs.values()
                if pair.get("connection_id") == values["to"]["connection_id"]
            ]
            if len(matches) != 1:
                raise ValueError(
                    "connection transport lacks unique live interface binding"
                )
            values = {**matches[0], **values}
        upper, lower = (
            values["upper_physical_instance_id"],
            values["lower_physical_instance_id"],
        )
        sample = parse_place_observation(
            shared,
            robot_id=robot,
            upper_physical_instance_id=upper,
            lower_physical_instance_id=lower,
            session_id=session,
            manifest_sha256=manifest,
        )
        interface = sample.interfaces["pairs"]["|".join(sorted((upper, lower)))]
        connection = _pose(interface["lower_to_upper"], "interface transform")
        symmetries = tuple(interface.get("allowed_upper_yaw_quarter_turns", (0,)))
        upward = operation.skill == "PLACE_UP" or (
            connection_transit and values.get("carrying_physical_instance_id") == lower
        )
        if upward:
            targets, _, _ = interface_place_up_targets(
                sample.upper.object_pose,
                connection,
                sample.lower.object_pose,
                sample.lower.tcp_pose,
                sample.lower.opening_m,
                context.base,
                context.provenance,
                context.params,
                symmetries,
            )
        else:
            targets, _, _ = interface_place_targets(
                sample.lower.object_pose,
                connection,
                sample.upper.object_pose,
                sample.upper.tcp_pose,
                sample.upper.opening_m,
                context.base,
                context.provenance,
                context.params,
                symmetries,
            )
        target = targets.preplace
    else:
        raise ValueError(f"{operation.skill} does not define a motion pre-phase")
    result = {robot: serialize_target(target, robot)}
    if (
        operation.skill == "SUPPORT_TOP"
        and values.get("placement_robot_id") in contexts
    ):
        carrier = values["placement_robot_id"]
        upper, lower = (
            values["upper_physical_instance_id"],
            values["lower_physical_instance_id"],
        )
        carried = parse_place_observation(
            shared,
            robot_id=carrier,
            upper_physical_instance_id=upper,
            lower_physical_instance_id=lower,
            session_id=session,
            manifest_sha256=manifest,
        )
        interface = carried.interfaces["pairs"]["|".join(sorted((upper, lower)))]
        carrier_context = contexts[carrier]
        placement, _, _ = interface_place_up_targets(
            carried.upper.object_pose,
            _pose(interface["lower_to_upper"], "interface transform"),
            carried.lower.object_pose,
            carried.lower.tcp_pose,
            carried.lower.opening_m,
            carrier_context.base,
            carrier_context.provenance,
            carrier_context.params,
            tuple(interface.get("allowed_upper_yaw_quarter_turns", (0,))),
        )
        result[carrier] = serialize_target(placement.preplace, carrier)
    for participant, participant_context in contexts.items():
        if participant in result:
            continue
        pose = _pose(shared["robots"][participant]["tcp_pose"], "participant TCP")
        hold = _tcp_pose_target(
            world_position_to_base(pose.position_xyz, participant_context.base),
            world_orientation_to_base(pose.orientation_xyzw, participant_context.base),
            participant_context.provenance,
            participant_context.params,
        )
        result[participant] = {
            **serialize_target(hold, participant),
            "maintain_pose": True,
        }
    return result
