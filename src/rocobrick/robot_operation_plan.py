"""Strict ingress contract for external multi-robot operation plans.

Wire shape ``robot-operation-plan/v1`` (frozen):

* plan keys: ``schema``, ``plan_id``, ``source_plan_id``, ``robots``,
  ``operations``, ``dependencies``, ``execution_layers``,
  ``planned_schedule``, ``evidence_requirements``;
* operation keys: ``operation_id``, ``source_task_id``, ``skill``,
  ``executor_robot_id``, ``participants``, ``role_bindings``, ``target``,
  ``parameters``, ``preconditions``, ``effects``, ``resources``,
  ``evidence_requirements``;
* skills: ``PICK``, ``TRANSIT``, ``WAIT``, ``PLACE_DOWN``, ``PLACE_UP``,
  ``SUPPORT_TOP``;
* target shapes: ``PICK`` is ``{physical_instance_id}``; ``TRANSIT`` is
  ``{from, to, carrying_physical_instance_id}`` where ``from``/``to`` are
  references ``{kind, physical_instance_id, connection_id, robot_id}``;
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

``source_plan``
---------------

When the caller holds the authoritative ``assembly-plan/v2`` structural plan
(a validated :class:`~rocobrick.stap_contract.StapPlan`, or its raw payload
which is validated here), it must pass it. The operation plan is then
additionally bound to it:

* ``source_plan_id`` equals the StapPlan ``plan_id`` and ``robots`` is the
  same set;
* every ``source_task_id`` exists; a motion skill (PICK/TRANSIT/PLACE_DOWN)
  must run on the task's allocated ``primary_robot`` with exactly the
  primary/supporting participants bound to roles, while a ``WAIT`` is a
  passive single-robot observation that may run on any declared robot;
* every PICK or carried part is the source task's own part; TRANSIT part
  references are the source task's part or one of its structural supports
  (initial supports and predecessor-task parts);
* a PLACE_DOWN target is grounded in the documented grid structure: the upper
  part is the source task's part and the lower part is the directly beneath
  goal part (same x/y, z-1) that is also an initial support or a predecessor
  task's part;
* connection ids are opaque labels, never derived: the STAP grammar carries
  none. A connection id is accepted only when some PLACE_DOWN target in the
  same plan binds it to exactly one ``(upper, lower)`` pair, that pair is
  labelled by exactly one connection id, and every TRANSIT connection
  reference is placed by its own source task.

The scheduler contract also reads this parse result: ``execution_layers``
determines real concurrency and barriers (a same-layer resource overlap is
rejected here), while ``planned_schedule`` only fixes evidence display order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rocobrick.stap_contract import StapContractError, StapPlan, parse_stap_plan

ROBOT_OPERATION_PLAN_SCHEMA = "robot-operation-plan/v1"
SKILLS = frozenset({"PICK", "TRANSIT", "WAIT", "PLACE_DOWN", "PLACE_UP", "SUPPORT_TOP"})

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


def _authoritative_stap(source_plan: object) -> StapPlan | None:
    """Return the authoritative StapPlan, validating raw payloads once."""
    if source_plan is None:
        return None
    if isinstance(source_plan, StapPlan):
        return source_plan
    if isinstance(source_plan, Mapping):
        try:
            return parse_stap_plan(source_plan)
        except StapContractError as error:
            raise RobotOperationPlanError(
                f"source_plan is not a valid assembly-plan/v2: {error}"
            ) from error
    raise RobotOperationPlanError(
        "source_plan must be a validated StapPlan or an assembly-plan/v2 object"
    )


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
    elif skill == "TRANSIT":
        _keys(
            item,
            frozenset({"from", "to", "carrying_physical_instance_id"}),
            f"{path}.target",
        )
        for key in ("from", "to"):
            reference = _ref(item[key], f"{path}.target.{key}")
            if reference["robot_id"] is not None and reference["robot_id"] != executor:
                raise RobotOperationPlanError(
                    f"{path}.target.{key}.robot_id must be the executor robot"
                )
        _opt(
            item["carrying_physical_instance_id"],
            f"{path}.target.carrying_physical_instance_id",
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
        elif skill == "TRANSIT":
            for key in ("from", "to"):
                reference = values[key]
                if reference["kind"] == "part":
                    parts.add(reference["physical_instance_id"])
                elif reference["kind"] == "connection":
                    connections.add(reference["connection_id"])
            carrying = values["carrying_physical_instance_id"]
            if carrying is not None:
                parts.add(carrying)
                required.add(f"part:{carrying}")
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
                        f"{participants[0]!r}; placement robot {partner!r} is target metadata"
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
    if operation.skill == "TRANSIT":
        return tuple(
            values[key]["connection_id"]
            for key in ("from", "to")
            if values[key]["kind"] == "connection"
        )
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


def _supporting_instances(plan: StapPlan, task: object) -> set[str]:
    supports = set(task.initial_supports)
    for predecessor in task.predecessors:
        predecessor_task = plan.tasks.get(predecessor)
        if predecessor_task is not None:
            supports.add(predecessor_task.physical_instance_id)
    return supports


def _bind_place_down(
    operation: RobotOperation,
    plan: StapPlan,
    task: object,
    supports: set[str],
) -> str:
    values = operation.target.values
    upper = values["upper_physical_instance_id"]
    lower = values["lower_physical_instance_id"]
    if upper != task.physical_instance_id:
        raise RobotOperationPlanError(
            f"operation {operation.operation_id!r} PLACE_DOWN upper part {upper!r} "
            f"is not source task part {task.physical_instance_id!r}"
        )
    if lower not in supports:
        raise RobotOperationPlanError(
            f"operation {operation.operation_id!r} PLACE_DOWN lower part {lower!r} "
            "is not an initial support or predecessor part of the source task"
        )
    upper_part = plan.goal_parts.get(upper)
    lower_part = plan.goal_parts.get(lower)
    if upper_part is None or lower_part is None:
        raise RobotOperationPlanError(
            f"operation {operation.operation_id!r} PLACE_DOWN references an "
            "unknown goal part"
        )
    if lower_part.z != upper_part.z - 1 or (lower_part.x, lower_part.y) != (
        upper_part.x,
        upper_part.y,
    ):
        raise RobotOperationPlanError(
            f"operation {operation.operation_id!r} PLACE_DOWN lower part {lower!r} "
            f"is not the goal part directly beneath {upper!r}"
        )
    return values["connection_id"]


def _bind_to_stap(operations: Mapping[str, RobotOperation], plan: StapPlan) -> None:
    placed: dict[str, set[str]] = {}
    referenced: dict[str, set[str]] = {}
    for operation in operations.values():
        task = plan.tasks.get(operation.source_task_id)
        if task is None:
            raise RobotOperationPlanError(
                f"operation {operation.operation_id!r} source_task_id "
                f"{operation.source_task_id!r} is not a source plan task"
            )
        if operation.skill == "WAIT":
            # WAIT is a passive single-robot observation: it names an existing
            # source task but never claims that task's motion allocation, so
            # any declared plan robot may observe it.
            if set(operation.participants) != {operation.executor_robot_id}:
                raise RobotOperationPlanError(
                    f"operation {operation.operation_id!r} WAIT participants "
                    f"{list(operation.participants)} must be exactly its executor"
                )
        else:
            if operation.executor_robot_id != task.primary_robot:
                raise RobotOperationPlanError(
                    f"operation {operation.operation_id!r} executor "
                    f"{operation.executor_robot_id!r} does not match source task "
                    f"allocation {task.primary_robot!r}"
                )
            expected_participants = {task.primary_robot}
            if task.supporting_robot is not None:
                expected_participants.add(task.supporting_robot)
            if set(operation.participants) != expected_participants:
                raise RobotOperationPlanError(
                    f"operation {operation.operation_id!r} participants "
                    f"{list(operation.participants)} do not match source task "
                    f"allocation {sorted(expected_participants)}"
                )
        supports = _supporting_instances(plan, task)
        if operation.skill == "PICK":
            picked = operation.target.values["physical_instance_id"]
            if picked != task.physical_instance_id:
                raise RobotOperationPlanError(
                    f"operation {operation.operation_id!r} PICK part {picked!r} "
                    f"is not source task part {task.physical_instance_id!r}"
                )
        elif operation.skill == "TRANSIT":
            values = operation.target.values
            allowed = {task.physical_instance_id} | supports
            for key in ("from", "to"):
                reference = values[key]
                if (
                    reference["kind"] == "part"
                    and reference["physical_instance_id"] not in allowed
                ):
                    raise RobotOperationPlanError(
                        f"operation {operation.operation_id!r} TRANSIT "
                        f"{key}.physical_instance_id "
                        f"{reference['physical_instance_id']!r} is neither the "
                        "source task part nor one of its supports"
                    )
            carrying = values["carrying_physical_instance_id"]
            if carrying is not None and carrying != task.physical_instance_id:
                raise RobotOperationPlanError(
                    f"operation {operation.operation_id!r} carries {carrying!r}, "
                    f"not source task part {task.physical_instance_id!r}"
                )
        elif operation.skill == "PLACE_DOWN":
            connection = _bind_place_down(operation, plan, task, supports)
            placed.setdefault(task.task_id, set()).add(connection)
        referenced.setdefault(task.task_id, set()).update(
            _operation_connections(operation)
        )
    for task_id, connections in placed.items():
        if len(connections) > 1:
            raise RobotOperationPlanError(
                f"source task {task_id!r} establishes multiple connections "
                f"{sorted(connections)}"
            )
    for task_id, connections in referenced.items():
        ungrounded = sorted(connections - placed.get(task_id, set()))
        if ungrounded:
            raise RobotOperationPlanError(
                f"source task {task_id!r} references connections not placed by "
                f"it: {ungrounded}"
            )


def parse_robot_operation_plan(payload, *, source_plan=None):
    """Validate and freeze one ``robot-operation-plan/v1`` document.

    Args:
        payload: Decoded plan document; it is never mutated.
        source_plan: Authoritative ``StapPlan`` (or raw ``assembly-plan/v2``
            payload) this plan must be bound to, or ``None`` for the
            structural-only contract.

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
    stap = _authoritative_stap(source_plan)
    if stap is not None:
        if sid != stap.plan_id:
            raise RobotOperationPlanError(
                f"source_plan_id {sid!r} does not equal source plan_id {stap.plan_id!r}"
            )
        if set(robots) != set(stap.robots):
            raise RobotOperationPlanError(
                f"robots {list(robots)} do not equal source plan robots "
                f"{list(stap.robots)}"
            )
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
            dict(_obj(item["parameters"], path + ".parameters")),
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
    if stap is not None:
        _bind_to_stap(operations, stap)

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
