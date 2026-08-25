# Manipulation Skills Refactor Plan

## Goal

Refactor RoCo_Piper into a backend-independent manipulation stack inspired by
APEX-MR. One upper-level planner dispatches six standard manipulation skills:

- `Pick`
- `Handover`
- `Place-Down`
- `Support-Bottom`
- `Place-Up`
- `Support-Top`

The six planner-visible names map to parameterized skill templates rather than
six independent control implementations:

```text
Pick                   -> PickSkill
Place-Down / Place-Up  -> AssembleSkill(direction)
Support-Bottom / Top   -> SupportSkill(direction)
Handover               -> CoordinatedHandoverSkill
```

`Up`, `Down`, `Top`, and `Bottom` are task semantics. Grounding converts the
corresponding interface-local normal into a world-frame unit direction before
execution.

## Architecture

The dependency direction is fixed:

```text
Planner
  -> SkillExecutor / SkillRegistry
  -> Task Skills
  -> Shared Primitives
  -> Controllers
  -> RobotBackend / WorldModel
```

Safety checks monitor primitives and controllers but are not planner skills.
Skills, primitives, controllers, and safety utilities must not import Isaac
Sim or BrickSim or access `Env` private fields.

### Planner and execution

Define one skill type enum with the six standard values. A planner action
contains the assigned robot IDs, skill type, object ID, and symbolic goal ID.
Single-arm actions have one robot; Handover has a giver and receiver.

`SkillRegistry` records the implementation template, request type, required
backend capabilities, resource cardinality, and current availability for each
standard skill. Unavailable skills are rejected before plan execution rather
than represented by failing placeholders.

Each `SkillExecutor` owns one robot context. A
`MultiRobotSkillCoordinator` owns only cross-robot synchronization for
Handover and Support lifetimes; task skills never select robots themselves.

### Backend boundary

Define two protocols:

- `RobotBackend`: robot identity, state feedback, FK/IK, joint commands,
  gripper commands, and home configuration.
- `WorldModel`: control-cycle advancement, object-pose queries, scene state,
  and object tracking.

BrickSim adapters translate object IDs to USD prims, wrap the current `Env`
and Pinocchio objects, ground reference/offset/yaw into target poses, and
implement connection verification. The initial refactor implements these
protocols, test fakes, and the BrickSim backend only; it does not add an empty
ROS backend.

## Shared primitives

All primitives expose an asynchronous `execute(context, request)` operation
and return a structured result containing success, termination reason, steps,
final error, and peak force. They share a common failure enum.

- `Move`: free-space motion to a pre-action pose, with optional held-object
  tracking.
- `Approach`: straight motion along a supplied world-frame unit direction,
  bounded by target pose and maximum travel.
- `Align`: align selected Cartesian degrees of freedom while suppressing
  motion along the insertion direction.
- `Grasp`: close the gripper and verify grasp formation and object following.
- `Release`: open the gripper only after the caller's success condition is
  satisfied.
- `InsertPress`: contact motion along an arbitrary unit direction, bounded by
  distance, speed, force, and success criteria. Insert and press remain one
  primitive.
- `Hold`: maintain a pose or directional support force until a completion or
  cancellation token arrives.
- `Twist`: perform a bounded small-angle rotation about an arbitrary axis;
  deferred until the fourth phase.
- `Retreat`: leave the operation region along a supplied safe direction.

`Approach` performs pre-contact positioning. `InsertPress` owns contact-phase
motion and must not treat reaching the nominal geometric pose alone as
assembly success.

## Controllers and safety

Controllers contain no task semantics:

- `IKController` solves IK, preserves seed continuity, checks joint limits,
  and verifies solutions with FK.
- `CartesianController` performs incremental pose servoing and bounds command
  lead and tracking error.
- `TrajectoryController` executes joint targets and waypoint sequences with
  settling and timeout rules.

Independent safety utilities are:

- `ForceGuard`: total and direction-projected force/torque limits.
- `ContactCheck`: contact detection from directional force, travel, and
  velocity.
- `CollisionCheck`: a single mandatory checking interface. Its first BrickSim
  implementation integrates existing joint-limit, IK/self-collision, and
  grasp-clearance checks. Full continuous scene and inter-arm collision
  checking is a later milestone.
- `SuccessCheck`: an injected semantic verification protocol. The BrickSim
  implementation verifies every requested reference connection and rejects
  incorrect offset/yaw.

## Skill composition

### PickSkill

```text
Move(pregrasp)
-> Approach(grasp direction)
-> Grasp
-> Retreat(lift direction)
```

Pick finishes above the object's initial position and returns a `HeldObject`
containing object ID, robot ID, measured object-to-TCP transform, grasp axis,
and gripper width. Grounding generates ordered grasp candidates using both
pickup feasibility and the known downstream placement constraints; Pick does
not receive an assembly goal directly.

### AssembleSkill

The same template implements Place-Down and Place-Up:

```text
Move(preassembly)
-> Align(goal pose, excluding insertion axis)
-> Approach(insertion axis)
-> InsertPress(insertion axis)
-> SuccessCheck
-> Release
-> Retreat(opposite insertion axis)
```

Alignment must complete before insertion-axis motion. Release is forbidden
until all requested connections are verified. Direction, target pose, and
retreat direction are grounded parameters rather than implementation branches.

### SupportSkill

The same template implements Support-Bottom and Support-Top:

```text
Move(pre-support)
-> Approach(contact direction)
-> Hold(target pose or force, completion token)
-> Retreat(opposite contact direction)
```

Hold remains active until the paired placement completes or the coordinator
cancels it. A force or controller failure is propagated to all paired actions.

### CoordinatedHandoverSkill

```text
both robots Move
-> both Align
-> giver Hold
-> receiver Approach + Grasp
-> transfer verification
-> giver Release
-> both Retreat
```

The giver must not release before the receiver's grasp is verified. Any
failure must leave at least one robot retaining the object.

## Migration from the current expert

Migrate behavior in small, testable slices:

- Split the pregrasp, approach, close, and lift portion of
  `prepare_safe_start` into PickSkill.
- Move carry-height, transport, and safe-start behavior into AssembleSkill's
  Move stage.
- Move joint/cartesian waypoint execution, FK/IK, gripper actuation, and state
  access into controllers and shared primitives.
- Split `GTAssemblyRuntime` feedback, wrench, and bounded-action logic among
  the backend, ForceGuard, and CartesianController.
- Peel ALIGN, APPROACH, PRESS, and HOLD behavior out of
  `GTAssemblyExpert.act()` one phase at a time. Do not retain the monolithic
  control loop after parity is achieved.
- Move BrickSim connection and conflict checks behind SuccessCheck.
- Retain `prepare_safe_start`, `run_gt_assembly_expert`, and
  `release_and_return_home` as thin compatibility adapters until demos and
  validation entry points migrate.

Structural migration must not also tune current gripper-opening, slip, or
continuous-yaw behavior. Those failures are fixed separately after their
owning skill interface is stable.

## Delivery phases

1. **Pick foundation**: add backend/controller/safety contracts; implement
   Move, Approach, Grasp, and Retreat; migrate and stabilize Pick. Only Pick is
   available in the registry.
2. **Assembly**: implement Align, InsertPress, Release, and SuccessCheck;
   migrate transport and mating behavior; enable Place-Down and Place-Up.
3. **Support**: implement Hold and coordinator completion/cancellation;
   enable Support-Bottom and Support-Top.
4. **Collaboration and recovery**: implement Twist and Handover; then build
   Recover and Regrasp by composing existing primitives rather than copying
   controllers.

## Verification and acceptance

- Skills, primitives, controllers, and safety modules import and run against a
  fake backend without Isaac or BrickSim installed.
- Primitive tests cover success, timeout, IK failure, force guard, collision
  rejection, cancellation, and postconditions.
- Pick tests cover candidate fallback, grasp feedback, lift following, slip,
  and robot ownership of `HeldObject`.
- Place-Up and Place-Down share the same test matrix with opposite grounded
  directions; tests enforce alignment before insertion and verification before
  release.
- Support tests cover blocking Hold, completion, cancellation, excessive
  force, and paired-action failure propagation.
- Handover tests prove that the giver cannot release before receiver grasp
  verification and that failures avoid a double-release state.
- Preserve existing grounding, multi-reference, continuous-yaw, IK, and
  connection-conflict tests.
- At every phase run `uv run pytest`, `uv run ruff check src tests`, and the
  corresponding BrickSim smoke workflow.
- Fixed-task and fixed-seed validation must not regress from the recorded
  pre-refactor baseline unless a separate behavior-fix change explicitly
  documents the expected difference.
