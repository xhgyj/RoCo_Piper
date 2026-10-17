"""Strict ingress contract for external multi-robot operation plans.

Wire shape ``robot-operation-plan/v1`` (frozen):

* plan keys: ``schema``, ``plan_id``, ``source_plan_id``, ``robots``,
  ``operations``, ``dependencies``, ``execution_layers``,
  ``planned_schedule``, ``evidence_requirements``;
* operation keys: ``operation_id``, ``source_task_id``, ``skill``,
  ``executor_robot_id``, ``participants``, ``role_bindings``, ``target``,
  ``parameters``, ``preconditions``, ``effects``, ``resources``,
  ``evidence_requirements``;
* skills: ``PICK``, ``HANDOVER``, ``WAIT``, ``PLACE_DOWN``, ``PLACE_UP``,
  ``SUPPORT_TOP``;
* target shapes: ``PICK`` is ``{physical_instance_id}``; ``HANDOVER`` is
  ``{physical_instance_id, donor_robot_id, receiver_robot_id}``;
  ``PLACE_DOWN`` is ``{connection_id, upper_physical_instance_id,
  lower_physical_instance_id}``; ``PLACE_UP`` is ``{connection_id,
  upper_physical_instance_id, lower_physical_instance_id, support_robot_id}``
  -- the executor grips the LOWER part and inserts it upward into the UPPER
  part while ``support_robot_id`` stabilizes the upper part; ``SUPPORT_TOP``
  is ``{connection_id, upper_physical_instance_id,
  lower_physical_instance_id, placement_robot_id}`` -- the executor
  stabilizes the UPPER part for the named placement robot's upward
  insertion; ``WAIT`` has a ``null`` target;
* resource vocabulary: ``robot:<robot_id>``, ``part:<physical_instance_id>``,
  ``connection:<connection_id>``.

Nothing is repaired, reordered, defaulted, or filled in: any deviation from
the frozen shape, any ungrounded reference, or any internal inconsistency is
rejected as-is.

The scheduler contract also reads this parse result: ``execution_layers``
determines real concurrency and barriers (a same-layer resource overlap is
rejected here), while ``planned_schedule`` only fixes evidence display order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

ROBOT_OPERATION_PLAN_SCHEMA = "robot-operation-plan/v1"
SKILLS = frozenset(
    {"PICK", "HANDOVER", "WAIT", "PLACE_DOWN", "PLACE_UP", "SUPPORT_TOP"}
)

_TOP = frozenset(
    {
        "schema",
        "plan_id",
        "source_plan_id",
        "robots",
        "operations",
        "dependencies",
        "execution_layers",
        "planned_schedule",
        "evidence_requirements",
    }
)
_OP = frozenset(
    {
        "operation_id",
        "source_task_id",
        "skill",
        "executor_robot_id",
        "participants",
        "role_bindings",
        "target",
        "parameters",
        "preconditions",
        "effects",
        "resources",
        "evidence_requirements",
    }
)
_REF = frozenset({"kind", "physical_instance_id", "connection_id", "robot_id"})
_DEPENDENCY = frozenset(
    {
        "predecessor_operation_id",
        "successor_operation_id",
        "reason",
    }
)
_SCHEDULE_ENTRY = frozenset({"operation_id", "robot_id"})
_EVIDENCE_ENTRY = frozenset({"operation_id", "evidence"})
_REF_KINDS = frozenset({"robot_current", "robot_home", "part", "connection"})
_RESOURCE_KINDS = frozenset({"robot", "part", "connection"})


class RobotOperationPlanError(ValueError):
    pass


def _obj(value: object, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise RobotOperationPlanError(f"{path} must be an object")
    return value


def _arr(value: object, path: str) -> list:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RobotOperationPlanError(f"{path} must be an array")
    return list(value)


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise RobotOperationPlanError(f"{path} must be a non-empty string")
    return value


def _opt(value: object, path: str) -> str | None:
    return None if value is None else _text(value, path)


def _keys(value: Mapping, expected: frozenset[str], path: str) -> None:
    actual = frozenset(str(key) for key in value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RobotOperationPlanError(
            f"{path} keys mismatch: missing {missing}, unexpected {unknown}"
        )


def _texts(value: object, path: str) -> tuple[str, ...]:
    items = tuple(_text(item, path + "[]") for item in _arr(value, path))
    if len(items) != len(set(items)):
        raise RobotOperationPlanError(f"{path} duplicates")
    return items


def _ref(value: object, path: str) -> Mapping:
    """Validate one target reference and return it as a mapping."""
    item = _obj(value, path)
    _keys(item, _REF, path)
    kind = _text(item["kind"], path + ".kind")
    if kind not in _REF_KINDS:
        raise RobotOperationPlanError(f"{path}.kind {kind!r} is unsupported")
    physical = _opt(item["physical_instance_id"], path + ".physical_instance_id")
    connection = _opt(item["connection_id"], path + ".connection_id")
    robot = _opt(item["robot_id"], path + ".robot_id")
    valid = {
        "robot_current": connection is None and physical is None,
        "robot_home": robot is not None and physical is None and connection is None,
        "part": physical is not None and connection is None and robot is None,
        "connection": connection is not None and physical is None and robot is None,
    }
    if not valid[kind]:
        raise RobotOperationPlanError(f"{path} {kind} reference mismatch")
    return item


def _handover_parameters(value: object, path: str) -> Mapping:
    """Validate immutable HANDOVER strategy parameters."""
    item = _obj(value, path)
    expected = frozenset({"grasp_profile_id", "co_hold_timeout_s", "follow_samples"})
    _keys(item, expected, path)
    _text(item["grasp_profile_id"], path + ".grasp_profile_id")
    timeout = item["co_hold_timeout_s"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise RobotOperationPlanError(
            f"{path}.co_hold_timeout_s must be a positive number"
        )
    samples = item["follow_samples"]
    if type(samples) is not int or samples < 3:
        raise RobotOperationPlanError(
            f"{path}.follow_samples must be an integer >= 3"
        )
    return item


def _target(
    skill: str, value: object, path: str, executor: str
) -> "OperationTarget | None":
    if skill == "WAIT":
        if value is not None:
            raise RobotOperationPlanError(f"{path}.target must be null for WAIT")
        return None
    item = _obj(value, f"{path}.target")
    if skill == "PICK":
        _keys(item, frozenset({"physical_instance_id"}), f"{path}.target")
        _text(item["physical_instance_id"], f"{path}.target.physical_instance_id")
    elif skill == "HANDOVER":
        expected = frozenset(
            {"physical_instance_id", "donor_robot_id", "receiver_robot_id"}
        )
        _keys(item, expected, f"{path}.target")
        for key in sorted(expected):
            _text(item[key], f"{path}.target.{key}")
        if item["donor_robot_id"] == item["receiver_robot_id"]:
            raise RobotOperationPlanError(
                f"{path}.target donor and receiver must be different robots"
            )
    elif skill == "PLACE_DOWN":
        expected = frozenset(
            {
                "connection_id",
                "upper_physical_instance_id",
                "lower_physical_instance_id",
            }
        )
        _keys(item, expected, f"{path}.target")
        for key in sorted(expected):
            _text(item[key], f"{path}.target.{key}")
    elif skill == "PLACE_UP":
        expected = frozenset(
            {
                "connection_id",
                "upper_physical_instance_id",
                "lower_physical_instance_id",
                "support_robot_id",
            }
        )
        _keys(item, expected, f"{path}.target")
        for key in sorted(expected):
            _text(item[key], f"{path}.target.{key}")
    else:
        expected = frozenset(
            {
                "connection_id",
                "upper_physical_instance_id",
                "lower_physical_instance_id",
                "placement_robot_id",
            }
        )
        _keys(item, expected, f"{path}.target")
        for key in sorted(expected):
            _text(item[key], f"{path}.target.{key}")
    return OperationTarget(dict(item))


def _resources(
    skill: str,
    participants: tuple[str, ...],
    target: "OperationTarget | None",
    value: object,
    path: str,
) -> tuple[str, ...]:
    """Validate the resource list is exactly grounded in participants/target."""
    items = _texts(value, f"{path}.resources")
    required = {f"robot:{robot}" for robot in participants}
    parts: set[str] = set()
    connections: set[str] = set()
    if target is not None:
        values = target.values
        if skill == "PICK":
            part = values["physical_instance_id"]
            parts.add(part)
            required.add(f"part:{part}")
        elif skill == "HANDOVER":
            part = values["physical_instance_id"]
            parts.add(part)
            required.add(f"part:{part}")
            if set(participants) != {
                values["donor_robot_id"],
                values["receiver_robot_id"],
            }:
                raise RobotOperationPlanError(
                    f"{path}.participants must exactly cover HANDOVER "
                    "donor and receiver"
                )
        else:
            connection = values["connection_id"]
            connections.add(connection)
            parts.update(
                (
                    values["upper_physical_instance_id"],
                    values["lower_physical_instance_id"],
                )
            )
            required.update(
                {
                    f"connection:{connection}",
                    f"part:{values['upper_physical_instance_id']}",
                    f"part:{values['lower_physical_instance_id']}",
                }
            )
            if skill == "PLACE_UP":
                partner = values["support_robot_id"]
                if set(participants) != {participants[0], partner}:
                    raise RobotOperationPlanError(
                        f"{path}.participants must be exactly the executor "
                        f"{participants[0]!r} and support robot {partner!r}"
                    )
            elif skill == "SUPPORT_TOP":
                partner = values["placement_robot_id"]
                parts.discard(values["lower_physical_instance_id"])
                required.discard(f"part:{values['lower_physical_instance_id']}")
                if set(participants) != {participants[0]}:
                    raise RobotOperationPlanError(
                        f"{path}.participants must contain only support executor "
                        f"{participants[0]!r}; placement robot {partner!r} "
                        "is target metadata"
                    )
    for item in items:
        kind, separator, name = item.partition(":")
        if not separator or kind not in _RESOURCE_KINDS or not name:
            raise RobotOperationPlanError(
                f"{path}.resources contains invalid resource {item!r}"
            )
        if kind == "part" and name not in parts:
            raise RobotOperationPlanError(
                f"{path}.resources contains part {name!r} not referenced by its target"
            )
        if kind == "connection" and name not in connections:
            raise RobotOperationPlanError(
                f"{path}.resources contains connection {name!r} not referenced "
                "by its target"
            )
        if kind == "robot" and name not in participants:
            raise RobotOperationPlanError(
                f"{path}.resources contains robot {name!r} outside participants"
            )
    missing = sorted(required - set(items))
    if missing:
        raise RobotOperationPlanError(f"{path}.resources missing {missing}")
    return items


@dataclass(frozen=True)
class OperationTarget:
    values: Mapping[str, object]


@dataclass(frozen=True)
class RobotOperation:
    operation_id: str
    source_task_id: str
    skill: str
    executor_robot_id: str
    participants: tuple[str, ...]
    role_bindings: Mapping[str, str]
    target: OperationTarget | None
    parameters: Mapping[str, object]
    preconditions: tuple[str, ...]
    effects: tuple[str, ...]
    resources: tuple[str, ...]
    evidence_requirements: tuple[str, ...]


@dataclass(frozen=True)
class OperationDependency:
    predecessor_operation_id: str
    successor_operation_id: str
    reason: str


@dataclass(frozen=True)
class ScheduledOperation:
    slot: int
    operation_id: str
    robot_id: str


@dataclass(frozen=True)
class RobotOperationPlan:
    schema: str
    plan_id: str
    source_plan_id: str
    robots: tuple[str, ...]
    operations: Mapping[str, RobotOperation]
    dependencies: tuple[OperationDependency, ...]
    execution_layers: tuple[tuple[str, ...], ...]
    planned_schedule: tuple[tuple[ScheduledOperation, ...], ...]
    evidence_requirements: tuple[tuple[str, str], ...]


def _operation_connections(operation: RobotOperation) -> tuple[str, ...]:
    if operation.target is None:
        return ()
    values = operation.target.values
    if operation.skill in {"PLACE_DOWN", "PLACE_UP", "SUPPORT_TOP"}:
        return (values["connection_id"],)
    return ()


def _bind_connections(operations: Mapping[str, RobotOperation]) -> None:
    """Ground every connection id in plan-local placement pair structure."""
    pairs: dict[str, tuple[str, str]] = {}
    labels: dict[tuple[str, str], str] = {}
    for operation in operations.values():
        if (
            operation.skill not in {"PLACE_DOWN", "PLACE_UP", "SUPPORT_TOP"}
            or operation.target is None
        ):
            continue
        values = operation.target.values
        pair = (
            values["upper_physical_instance_id"],
            values["lower_physical_instance_id"],
        )
        connection = values["connection_id"]
        known = pairs.get(connection)
        if known is not None and known != pair:
            raise RobotOperationPlanError(
                f"connection {connection!r} is bound to both {known!r} and {pair!r}"
            )
        pairs[connection] = pair
        labelled = labels.get(pair)
        if labelled is not None and labelled != connection:
            raise RobotOperationPlanError(
                f"connection pair {pair!r} is labelled both {labelled!r} "
                f"and {connection!r}"
            )
        labels[pair] = connection
    for operation in operations.values():
        for connection in _operation_connections(operation):
            if connection not in pairs:
                raise RobotOperationPlanError(
                    f"operation {operation.operation_id!r} references connection "
                    f"{connection!r} that no placement target grounds"
                )


def parse_robot_operation_plan(payload):
    """Validate and freeze one ``robot-operation-plan/v1`` document.

    The formal ingress contract is self-contained: ``source_plan_id`` is a
    required, validated field of the wire shape and is never cross-checked
    against a second plan document. The structure-carrier binding is done by
    the caller against the exact task structure bytes.

    Args:
        payload: Decoded plan document; it is never mutated.

    Returns:
        The immutable ``RobotOperationPlan``, preserving input order.

    Raises:
        RobotOperationPlanError: If any frozen-shape, grounding, coverage,
            ordering, allocation, or resource constraint is violated.
    """
    root = _obj(payload, "plan")
    _keys(root, _TOP, "plan")
    if root["schema"] != ROBOT_OPERATION_PLAN_SCHEMA:
        raise RobotOperationPlanError("unsupported schema")
    pid = _text(root["plan_id"], "plan_id")
    sid = _text(root["source_plan_id"], "source_plan_id")
    robots = _texts(root["robots"], "robots")
    if not robots:
        raise RobotOperationPlanError("robots must not be empty")
    operations: dict[str, RobotOperation] = {}
    for index, value in enumerate(_arr(root["operations"], "operations")):
        path = f"operations[{index}]"
        item = _obj(value, path)
        _keys(item, _OP, path)
        operation_id = _text(item["operation_id"], path + ".operation_id")
        if operation_id in operations:
            raise RobotOperationPlanError(f"duplicate operation_id {operation_id!r}")
        source_task_id = _text(item["source_task_id"], path + ".source_task_id")
        skill = _text(item["skill"], path + ".skill")
        if skill not in SKILLS:
            raise RobotOperationPlanError(f"{path}.skill {skill!r} is unsupported")
        executor = _text(item["executor_robot_id"], path + ".executor_robot_id")
        if executor not in robots:
            raise RobotOperationPlanError(
                f"{path}.executor_robot_id {executor!r} is not a plan robot"
            )
        participants = _texts(item["participants"], path + ".participants")
        if executor not in participants:
            raise RobotOperationPlanError(
                f"{path}.participants must include the executor"
            )
        outside = sorted(set(participants) - set(robots))
        if outside:
            raise RobotOperationPlanError(
                f"{path}.participants are not plan robots: {outside}"
            )
        raw_bindings = _obj(item["role_bindings"], path + ".role_bindings")
        bindings = {
            _text(key, path + ".role_bindings key"): _text(
                bound, path + ".role_bindings." + str(key)
            )
            for key, bound in raw_bindings.items()
        }
        if (
            not bindings
            or bindings.get("executor") != executor
            or set(bindings.values()) != set(participants)
        ):
            raise RobotOperationPlanError(
                f"{path}.role_bindings must bind the executor and every participant"
            )
        target = _target(skill, item["target"], path, executor)
        parameters = dict(_obj(item["parameters"], path + ".parameters"))
        if skill == "HANDOVER":
            _handover_parameters(parameters, path + ".parameters")
            bindings_expected = {"executor", "donor", "receiver"}
            if set(bindings) != bindings_expected:
                raise RobotOperationPlanError(
                    f"{path}.role_bindings must contain executor, donor and receiver"
                )
            handover_target = target.values if target is not None else {}
            if (
                bindings["donor"] != handover_target.get("donor_robot_id")
                or bindings["receiver"] != handover_target.get("receiver_robot_id")
                or bindings["executor"] != handover_target.get("receiver_robot_id")
            ):
                raise RobotOperationPlanError(
                    f"{path}.role_bindings do not match HANDOVER target roles"
                )
        if target is not None and skill in {"PLACE_UP", "SUPPORT_TOP"}:
            partner_key = (
                "support_robot_id" if skill == "PLACE_UP" else "placement_robot_id"
            )
            partner = target.values[partner_key]
            if partner == executor:
                raise RobotOperationPlanError(
                    f"{path}.target.{partner_key} must be the collaborating robot, "
                    f"not the executor {executor!r}"
                )
            if partner not in robots:
                raise RobotOperationPlanError(
                    f"{path}.target.{partner_key} {partner!r} is not a plan robot"
                )
            if skill == "PLACE_UP":
                if set(participants) != {executor, partner}:
                    raise RobotOperationPlanError(
                        f"{path}.participants must be exactly the executor "
                        f"{executor!r} and support robot {partner!r}"
                    )
            else:
                if set(participants) != {executor}:
                    raise RobotOperationPlanError(
                        f"{path}.participants must contain only support executor "
                        f"{executor!r}; placement robot {partner!r} is target metadata"
                    )
        operations[operation_id] = RobotOperation(
            operation_id,
            source_task_id,
            skill,
            executor,
            participants,
            bindings,
            target,
            parameters,
            _texts(item["preconditions"], path + ".preconditions"),
            _texts(item["effects"], path + ".effects"),
            _resources(skill, participants, target, item["resources"], path),
            _texts(item["evidence_requirements"], path + ".evidence_requirements"),
        )

    dependencies: list[OperationDependency] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, value in enumerate(_arr(root["dependencies"], "dependencies")):
        path = f"dependencies[{index}]"
        item = _obj(value, path)
        _keys(item, _DEPENDENCY, path)
        predecessor = _text(
            item["predecessor_operation_id"], path + ".predecessor_operation_id"
        )
        successor = _text(
            item["successor_operation_id"], path + ".successor_operation_id"
        )
        if (
            predecessor == successor
            or predecessor not in operations
            or successor not in operations
            or (predecessor, successor) in seen_pairs
        ):
            raise RobotOperationPlanError(f"{path} is not a valid operation dependency")
        seen_pairs.add((predecessor, successor))
        dependencies.append(
            OperationDependency(
                predecessor, successor, _text(item["reason"], path + ".reason")
            )
        )

    layer_of: dict[str, int] = {}
    layers: list[tuple[str, ...]] = []
    for index, value in enumerate(_arr(root["execution_layers"], "execution_layers")):
        layer = _texts(value, f"execution_layers[{index}]")
        for operation_id in layer:
            if operation_id not in operations or operation_id in layer_of:
                raise RobotOperationPlanError(
                    f"execution_layers[{index}] operation {operation_id!r} is "
                    "unknown or duplicated"
                )
            layer_of[operation_id] = index
        layers.append(layer)
    if set(layer_of) != set(operations):
        raise RobotOperationPlanError(
            "execution_layers must cover every operation exactly once"
        )

    slot_of: dict[str, int] = {}
    schedule: list[tuple[ScheduledOperation, ...]] = []
    for slot, value in enumerate(_arr(root["planned_schedule"], "planned_schedule")):
        used: set[str] = set()
        entries: list[ScheduledOperation] = []
        for entry_index, entry in enumerate(_arr(value, f"planned_schedule[{slot}]")):
            path = f"planned_schedule[{slot}][{entry_index}]"
            item = _obj(entry, path)
            _keys(item, _SCHEDULE_ENTRY, path)
            operation_id = _text(item["operation_id"], path + ".operation_id")
            robot = _text(item["robot_id"], path + ".robot_id")
            if operation_id not in operations or operation_id in slot_of:
                raise RobotOperationPlanError(
                    f"{path} operation is unknown or duplicated"
                )
            if robot != operations[operation_id].executor_robot_id or robot in used:
                raise RobotOperationPlanError(
                    f"{path} robot does not exclusively match the operation executor"
                )
            used.add(robot)
            slot_of[operation_id] = slot
            entries.append(ScheduledOperation(slot, operation_id, robot))
        schedule.append(tuple(entries))
    if set(slot_of) != set(operations):
        raise RobotOperationPlanError(
            "planned_schedule must cover every operation exactly once"
        )

    for dependency in dependencies:
        before = dependency.predecessor_operation_id
        after = dependency.successor_operation_id
        if layer_of[before] >= layer_of[after] or slot_of[before] >= slot_of[after]:
            raise RobotOperationPlanError(
                f"dependency {before!r} -> {after!r} does not respect layer and "
                "schedule order"
            )

    for index, layer in enumerate(layers):
        used: set[str] = set()
        for operation_id in layer:
            conflict = used & set(operations[operation_id].resources)
            if conflict:
                raise RobotOperationPlanError(
                    f"execution_layers[{index}] has a resource conflict on "
                    f"{sorted(conflict)}"
                )
            used.update(operations[operation_id].resources)

    evidence: list[tuple[str, str]] = []
    seen_evidence: set[str] = set()
    for index, value in enumerate(
        _arr(root["evidence_requirements"], "evidence_requirements")
    ):
        path = f"evidence_requirements[{index}]"
        item = _obj(value, path)
        _keys(item, _EVIDENCE_ENTRY, path)
        operation_id = _text(item["operation_id"], path + ".operation_id")
        label = _text(item["evidence"], path + ".evidence")
        if operation_id not in operations or operation_id in seen_evidence:
            raise RobotOperationPlanError(f"{path} operation is unknown or duplicated")
        if label not in operations[operation_id].evidence_requirements:
            raise RobotOperationPlanError(
                f"{path} evidence {label!r} is not declared by operation "
                f"{operation_id!r}"
            )
        seen_evidence.add(operation_id)
        evidence.append((operation_id, label))

    _bind_connections(operations)

    return RobotOperationPlan(
        ROBOT_OPERATION_PLAN_SCHEMA,
        pid,
        sid,
        robots,
        operations,
        tuple(dependencies),
        tuple(layers),
        tuple(schedule),
        tuple(evidence),
    )
