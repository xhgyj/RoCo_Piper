# Pick / Assemble Unified Architecture

## Boundary

The upper planner owns task allocation. It decides the action, object, goal,
and robot before calling the manipulation layer:

```python
result = await manipulation_executor.execute(action)
```

`ManipulationExecutor` never searches for another robot and never reassigns an
action. Failure on the assigned robot is returned as `PLANNING_FAILED` or
`EXECUTION_FAILED`.

The current implemented scope is:

- `PICK`
- single-step `PLACE_DOWN`
- one or more target connections in that step

Place-Up, support, handover, assembly-sequence planning, dual-arm coordination,
and compatibility with the previous dense1 demonstrations are outside this
refactor. Their enum values remain vocabulary reserved for later work, but the
registry and executor do not advertise them as available.

## Data flow

```text
upper planner
  ManipulationAction(action_id, robot_ids, skill_type, object_id, goal_id)
        |
        v
ManipulationExecutor.execute(action)
  validate exact assigned robot and held-object ownership
        |
        v
ActionGrounder
  symbolic IDs -> SceneGeometry + optional AssemblyGoal
        |
        v
GraspPlanner / AssemblyPlanner
  geometry + assigned RobotBackend -> complete continuous IK plan
        |
        v
PickSkill / AssembleSkill + shared primitives
  execute only the selected robot's plan
        |
        v
ManipulationResult
```

Grounding is domain-specific. `BrickSimActionGrounder` is the BrickSim adapter;
the planners and skills do not import BrickSim or Isaac Sim.

## Contracts

`ManipulationAction` is the only public command. `robot_ids` is an input
constraint, not a candidate set. `action_id` correlates the result with the
upper planner. For BrickSim `PLACE_DOWN`, `goal_id` is an exact target-scoped
identifier (`bricksim:place_down:<target-object-id>`), rather than a flag that
merely says a goal exists. Grounding rejects a stale or mismatched identifier.

`AssemblyGoal` is a robot-independent geometric/semantic target. It contains
the target object pose, world insertion direction, all exact connection
conditions, allowed contact object IDs, and the live semantic success check.
It does not contain an arm choice or joint values.

`GraspPlan` is a robot-specific result. It contains the fixed
object-to-TCP transform, grasp width/axis, clearance and stability scores, and
the complete sampled pregrasp, approach, and lift IK branches.

`AssemblyPlan` is also robot-specific. It contains the held-object transport IK
branch, preassembly and goal TCP poses, and geometry-derived insertion, retreat,
and alignment limits.

`HeldObjectState` persists between calls. It keeps the acquisition transform,
the once-settled transform after Pick, the current transform, and cumulative
translation/rotation drift. Later waypoints never reset this baseline.

`ManipulationResult` always reports one of `SUCCESS`, `INVALID_ACTION`,
`PLANNING_FAILED`, or `EXECUTION_FAILED`, plus a stable failure code, stage, and
detail when unsuccessful. Its metrics distinguish planned/executed waypoints,
controller iterations, and physics simulation steps; `steps` is retained only
as a compatibility view of `simulation_steps`.

## Geometry and generalization

No brick family or exact `1x2` case is used by the new planners.

- Each target has separate object, collision, and grasp-region frames. Asset
  calibration therefore does not leak into the symbolic object pose.
- The BrickSim adapter reads asset-local USD bounds for collision centers and
  extents, with symbolic dimensions only as a non-simulator fallback.
- Collision bodies and obstacles are oriented bounding boxes (OBBs); a target
  may expose multiple grasp regions and allowed local jaw axes.
- The gripper is represented by two finger OBBs and a palm OBB.
- Collision uses the 3D separating-axis test.
- Swept paths are sampled at at most 1 mm translation and 2 degrees rotation.
- Pick considers both object-local jaw axes, both tool signs, and multiple
  centers along the perpendicular footprint axis.
- Candidates are filtered by gripper opening, swept-volume clearance, safe
  continuous IK on the assigned robot, and the full downstream held-object
  route when a placement goal is supplied.
- Free-space pregrasp motion is a bounded joint-space path whose actual TCP
  sweep is recovered with forward kinematics and collision-checked. Precision
  approach, lift, and held-object motion use task-space samples plus adaptive
  IK subdivision when a joint-space jump exceeds the configured bound.
- Collision sampling remains independently dense, so IK resolution is not
  coupled to the 1 mm swept-volume check.
- Ranking is deterministic: clearance, grasp leverage/stability, joint margin,
  path length, then candidate index.
- Assembly clearance, insertion travel, and rotation step are continuous
  functions of object geometry and grasp stability.
- Only the requested mating references may be whitelisted for intentional
  contact during insertion. Free-space transport has no contact whitelist.

The current collision scope covers the gripper and held-object swept volumes.
Continuous collision for every arm link remains future work; backend
`configuration_is_safe` is still required for every sampled IK configuration.

## Execution invariants

Planning completes before the first command. Therefore a planning failure sends
no robot command. Execution consumes the planned IK samples rather than asking
the skill to select another candidate.

Pick is the only phase allowed to establish a settled grasp transform. During
transport and assembly, stability checks compare against the immutable
acquisition reference and cumulative drift remains observable. Assemble never
rebases the grasp after alignment.

Release is allowed only after the injected semantic success check verifies all
requested connections. The same condition is checked again after retreat.

## Verification

Pure tests cover action ownership, no fallback to another robot, zero commands
on planning failure, exact goal identity, independent collision/grasp frames,
metric semantics, OBB invariance under world rigid transforms and renaming,
cumulative drift, exact connection checks, Pick/Assemble primitive behavior,
and both insertion-direction controller mechanics. The isolated BrickSim GPU
smoke run also exercises a goal-aware Pick assigned to `piper_0`.

Run the isolated Pick smoke test with an explicitly assigned robot:

```bash
uv run bricksim ./run/smoke_pick_executor.py --arm-index 0
```

This entry point does not invoke the legacy preparation, dense1 sequence, or
assembly demo workflows.

Run the complete unified Pick -> PlaceDown path for one Task-1 directory:

```bash
uv run bricksim ./run/test_pick_assemble.py \
  tasks/type1/example1 --arm-index 0
```

Test several directories in one Isaac Sim process and write a JSON report:

```bash
uv run bricksim ./run/test_pick_assemble.py \
  tasks/type1/example1 /path/to/another/task \
  --arm-index 0 --output /tmp/roco-pick-assemble.json
```

Recursively discover every valid Task-1 pair under a directory, or explicitly
test the same tasks with both upper-planner arm assignments:

```bash
uv run bricksim ./run/test_pick_assemble.py \
  --tasks-root tasks/type1 --arm-index 0

uv run bricksim ./run/test_pick_assemble.py \
  tasks/type1/example1 --arm-index 0 --arm-index 1
```

The runner creates temporary user configuration outside the repository. It
does not modify `config/user_config.json`. Each task must contain exactly one
new target between `structure_start.json` and `structure_goal.json`; invalid
tasks and unsuitable arm assignments are returned as structured failures.
