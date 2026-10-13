# Coordinated bottom HANDOVER implementation status

Status: post-lift admission and persistent transport milestone, **not a
completed physical HANDOVER implementation**. Working base:
`3b391823d2572ad9c038022d7234b78ffaf01aa9`.
This change does not launch Isaac, ROS, MoveIt, a controller, or robot motion.

## Implemented

- `skills/handover_profile.py` and
  `skills/profiles/handover-bottom-coordinated-v1.json`: exact-key versioned
  profile, model binding, bounded candidate grids and evaluation budget,
  evidence tolerances, explicit capability requirements. Numerical values are
  draft planning policy, not experimentally qualified safety parameters.
- `skills/handover_planning.py`: fresh same-scene measured start state,
  explicitly identified giver holding, object-local geometry and separate
  calibrated robot contexts. Generates relocated/rotated object goals and
  bottom/oblique receiver goals while preserving the measured object-to-giver
  TCP transform. Candidate SHA binds all paired geometry. The search reports
  missing gates and failure categories; budget exhaustion does not prove
  physical impossibility outside the configured candidate family.
- `skills/handover_joint_backend.py`: read-only provider protocol and admission
  pipeline for both IK goals, combined geometry, gripper sweeps, approach and
  separation, one combined trajectory, continuous collision validation and
  scene readback. Reuses `CoordinatedOperation`. Requires both full six-joint
  arms, measured starts, matching paired ends, identical time grids and the
  configured synchronization budget. It has no execution/scene-write method.
- `skills/handover_observation.py`: same-tick
  `rocobrick-pad-object-contacts/v1` reports from `physx_contact_report`, exact
  pad/object identity, opposed pad normals, separate opening windows, advancing
  sample count/span, full relative SE(3), preserved lift-time giver grasp,
  released giver and actual receiver/brick displacement for follow proof.
  Assembly stud/hole force and MoveIt collision data are not substitutes.
- Existing handler safety fixes: honor formal `follow_samples`; mark holding
  uncertain on a failed attempted scene transfer or release, including missing
  post-dispatch feedback. Ledger ownership is not adopted on these failures.
- Formal CLI admission resolves and audits profile IDs before creating a
  runtime or executing a preceding PICK. Old `brick-side-grasp-v1` is not
  silently reinterpreted. The new profile is planning-only. Changing its
  qualification flag alone cannot activate the legacy PICK-target handler.
- `adapters/moveit/handover_geometry.py`: isolated Pinocchio/Coal scene using
  the actual combined exported URDF collision geometry, unchanged model joint
  limits, calibrated bases/TCP FK, full measured independent joint states and
  conservative measured brick/environment envelopes. Model/SRDF and all mesh
  assets are SHA-bound. No mesh, gripper state or environment fallback exists.
  Only SRDF robot-vs-robot self-exclusions are inherited; cross-robot exclusions
  are rejected. Attached-object collision against the wrist is retained even
  when they share a parent joint or the TCP link has SRDF exclusions.
- Native finite-budget multi-seed IK; whole joint-linear segments are certified
  by conservative clearance/displacement bounds, not endpoint sampling. The
  Cartesian certificate additionally bounds the complete position/orientation
  tube. Depth/query/time exhaustion is rejection. These certificates do not
  apply to arbitrary spline interpolation or unverified controller tracking.
- `adapters/moveit/handover_scene_planner.py`: concrete planning provider with
  explicit calibrated complete finger configurations, receiving-finger opening,
  approach, closing, giver opening, separation and receiver verification-motion
  paths. Giver joints remain unchanged during final approach. A planned receiver
  attachment is geometry-only, not physical ownership. Finger/path certificates
  are immutable and cannot be reused with changed paired goals/calibration.
- Read-only adapter to the existing `CoordinatedMoveItPlanRuntime`: one
  twelve-joint `dual_arm` plan-only goal, identical returned time grids, profile
  endpoint tolerance, no controller call and no shared-scene apply/sync method.
  Requires exact captured scene identity readback before planning and checks
  every returned segment using the native complete scene.
- `coordinated_ros_runtime.py`: measured `maintain_pose` joints now enter both
  goal and real path constraints. Every returned waypoint is checked for hold
  drift. Missing measured state, different hold goal, bool/invalid tolerance and
  endpoint-only hold proofs are rejected. General goal tolerance remains 0.01
  unless explicitly narrowed; HANDOVER uses the profile's stricter tolerance.
- `adapters/isaac/pad_contacts.py` and `isaac/p05_host_bridge.py`: native current
  PhysX-step contact buffer reader, exact pad/rigid-object actor bindings,
  contact-report API setup, normal direction normalization, positive normal
  impulse/dt force, LOST/unmapped/zero-force rejection. Both host loops capture
  immediately after physics stepping. The scene query emits the report only
  for that exact captured step, otherwise advertises missing capability; stale
  contacts are never reused. Non-SI stages fail this capability explicitly.
- `skills/post_lift_handover.py`: explicit continuation from the original PICK's
  completed lift. Requires confirmed exclusive giver ownership, complete PICK
  lift/gripper/following evidence and a later same-session snapshot; captures
  full measured start state only after these gates. Retains the entire PICK
  proof. It does not change PICK parameters, add LIFT/TRANSIT, reset or home.
- `execution/prepared_motion.py`: strict SHA-bound prepared path envelope,
  exact two-arm/common-grid/start binding, and semantic live admission. Query
  time/sequence/digest may advance while geometry stays static; actual object,
  TCP, environment, model, version, arm or finger drift is rejected. Mutated
  nested admitted input cannot be silently rehashed. A hash is not collision
  certification or custody proof.
- Prepared execution is wired through `runtime_session`, the existing owned
  worker, `RosMoveItRobotSession` and both arm participants. No replanning or
  scene apply occurs. Both actual controller goals receive the same future
  simulation-clock header timestamp; strict path/goal tolerances are set.
  Controller phase feedback is cross-checked against newly stamped Isaac joint
  samples and the admitted joint-linear interpolation, including first/end
  coverage and bounded gaps. Goal acceptance time is not synchronization proof.
  Final arrival additionally requires a fresh post-result physics sample.
- Coordinated failure observes whichever participant finishes first, cancels
  unfinished peers and waits for all execution lanes before final cancellation
  proof. An unowned/uncertain dispatch cannot become safe through an idle cancel.
  Worker replay, token, robot ownership and terminal-safe gates remain active.
- `adapters/moveit/readonly_combined_plan.py` and `SessionCombinedPlanRuntime`:
  native host planning can query the container's existing combined MoveGroup
  through a dedicated read-only worker lane, without `sync_unified_scene`.
  Scene version, full joint goals, starts/ends, common time grid and input
  mutation are checked. Planning identity does not consume execution identity;
  planning does not release leases or certify native collision geometry.

## Not implemented / must remain blocked

The concrete provider, persistent read-only planning lane and prepared-motion
execution lane now exist. The formal CLI has **not instantiated the native
provider and new HANDOVER phase handler yet**.
Native tests exercise real Pinocchio/Coal IK/collision on explicitly synthetic
primitive URDF scenes; ROS transport and PhysX contact buffers are test doubles.
No live two-Piper scene plan, actual PhysX contact capture or synchronization
evidence has been produced. `plan_handover` remains an API, not a new CLI entry.

The production path still needs:

1. Bind the concrete provider inside the existing persistent ROS session:
   capture the actual combined exported model/mesh paths, exact refreshed scene
   identity, full measured finger state and calibrated opening/squeeze states.
   Keep candidate queries read-only; verify the existing shared MoveIt scene is
   consistent with the captured scene rather than applying candidate diffs.
   Native no-solution/certification-budget rejection needs phase/collision-pair
   diagnostics in the formal run evidence; never infer global infeasibility.
2. The prepared-arm execution lane is implemented, but still needs production
   phase-handler integration, actual controller/physics evidence, calibrated
   tracking/clearance budgets and whole-phase giver hold during final approach.
   Arm arrival does not establish contact, custody or gripper-sweep execution.
3. Verify the new PhysX bridge reader against actual loaded Piper rigid-body
   actors, exact object actor paths, SI stage units and physics/ROS clock stamps.
   Bridge source integration is implemented, but no live contact report has
   been observed. Contact capability alone does not mean any pad is touching.
4. Receiver verification motion and phase-scoped provisional custody before
   final ledger adoption; a new handler integrated with the formal executor.
   The scheduler currently expects a physical successor pre-phase for retained
   PICK leases; HANDOVER target derivation explicitly rejects that pre-phase.
   Direct lift continuation needs a truthful stationary custody/lease adoption
   boundary or genuinely executed rendezvous pre-phase, not fabricated
   planned/prepared/started/arrival booleans or bypassed leases.
5. Model/geometry-backed positive and negative planning examples, calibrated
   profile parameters, then an explicitly authorized controlled Isaac run in
   a fresh `runs/` directory. No joint-limit expansion, blanket cross-robot
   collision exemption, or immutable input rewrite is permitted.

Until these exist, a plan containing HANDOVER rejects before PICK or runtime
creation. PICK-only plans keep their existing dispatch path. No new STAP or
TRANSIT requirement was introduced. Historical input files are unchanged.

## Verification

Remote locked environment, ordinary offline suite:

```bash
PYTHONPATH=src uv run --no-sync python -m pytest -q -p no:cacheprovider tests/
```

Previous milestone: **1984 passed, 2 skipped**, 14.53 seconds. All new modules
and the focused HANDOVER/CLI preflight test files pass Ruff. The single-line import
allowlist addition in `tests/test_piper_run.py` introduces no lint diagnostics:
that historical file has 54 diagnostics both at HEAD and after the change.

Continuation verification: **2016 passed, 2 skipped** in the remote locked
environment, 14.91 seconds. All newly added modules/tests and the modified production combined
planner pass Ruff. Historical test-file diagnostics are not represented as a
clean repository-wide lint result.

Tests cover stale/future scene admission, giver identity, grid budgets, frozen
request/candidate bindings, downward/degenerate approach rejection, retained
giver grasp, both-arm trajectory bindings and timing, sweep/path refusals,
missing/non-physics/mixed contacts, false static follow, SE(3) slip, release and
transfer uncertainty, and no runtime creation for unqualified HANDOVER even
with a preceding PICK. Ordinary tests do not constitute physical acceptance.

This continuation additionally covers actual native wrist/object collision,
cross-robot SRDF refusal, clear-endpoint/interior-collision rejection, native
bounded IK, continuous Cartesian-tube refusal, actual phase geometry with one
combined transport fixture, changed scene/finger bindings, missing mesh state,
stale PhysX buffers and maintained-joint interior drift. Two older hold-goal
fixtures were updated to provide their measured starting joints and check real
path constraints; the missing-start refusal was not weakened.

## Original PICK/lift continuation input (2026-10-07)

The user's original input is
`runs/stage5-pick-real-20261005T123925Z/pick-only-robot-operation-plan.json`.
It assigns PICK to **piper_0**. The new independent draft is
`runs/pick-lift-handover-input-20261007-prepared-v1/robot-operation-plan.json`:
original PICK, including its internal lift, followed by HANDOVER **piper_0 ->
piper_1**, in two dependency-linked layers of the same session. No home,
restart, new STAP or separate LIFT operation is introduced. The original PICK
operation is structurally identical; the original input is not overwritten.
`input-provenance.json` records both input SHA values and the blocking items.
The new input parses formally but is **NOT executable HANDOVER yet**: the
qualification and new-provider admission gates remain intact before PICK.

Final remote continuation suite: **2074 passed, 2 skipped**, 15.01 seconds,
including production timestamp-reader and late/unknown-dispatch cancellation
regressions. New production modules and new test
files pass Ruff; historical lint in older large modules/tests is not claimed
to be clean.

The user explicitly owns real simulation execution. This implementation task
does not launch Isaac/ROS/MoveIt or dispatch motion. Test feedback is scripted,
not evidence of a real synchronized dual-Piper motion or a physical HANDOVER.
The new read-only/native constructor wiring, verified local contact geometry,
lease adoption, new handler/custody and parameter qualification still require
completion before handing over a runnable motion command.

Controller header start semantics and positions-only linear interpolation are
documented by [ROS 2 Control Humble trajectory representation](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html).
The phase monitor uses controller feedback plus independent stamped physics
tracking; it does not claim to observe the exact first physical movement tick.

No motion, qualification-flag change, commit or push was performed.

## Complete transfer lifecycle continuation (2026-10-07)

The new `skills/coordinated_handover.py` owns the entire transfer lifecycle:
receiver opening, combined rendezvous, bottom approach, receiver close,
measured co-hold, provisional admission, giver release, retained receiver
contact, attachment transfer/readback, giver separation, and measured receiver
follow before atomic owner adoption. It monitors fresh pad/SE(3) samples while
actions run and includes before/during/after verification-motion samples.
Prepared-path, tracking/arrival, gripper-sweep, and scene-readback receipts are
separate gates, bound to the immutable full program. Phase paths must contain
the full measured joint state, continuous boundaries and a common twelve-arm-
joint time grid. Constrained phases may not move the held arm.

The ledger now represents provisional co-hold separately from confirmed
receiver holding. It revokes confirmed giver holding BEFORE a releasing send.
The giver remains protective owner until release, separation, physical follow
and readback have passed. Failure/cancellation does not adopt the receiver;
safe controller stop alone does not release an unresolved physical lease.
Missing initial giver contact also invalidates stale confirmation, without
sending any motion. Operation-specific cancellation settles both action lanes.

`skills/handover_continuation.py` admits a stationary, no-motion scheduler
transition from the original PICK/lift. It checks fresh retained giver contact,
SE(3), static TCP/object poses, full joint state, clock and scene version,
preserves the complete original PICK evidence, and revalidates a one-shot
certificate against actual physics before scheduler lease adoption. It does
not create a TRANSIT or claim a prepared/started/arrived motion.

`adapters/moveit/handover_program.py` compiles all native paths before dispatch,
using explicit post-lift capture/native-provider callbacks and calibrated
opening windows. Native compilation runs real Pinocchio/Coal in primitive-
model tests; MoveGroup transport is synthetic.

The persistent session now exposes the read-only
`rocobrick-scene-monitor/v1` mailbox for current physics during owned actions.
It neither requires idle action ownership nor calls GetPlanningScene/its
planning lock, changes leases, or grants physical admission. Stale and missing
participant samples fail closed.

The formal CLI's old receiver-only PICK-target HANDOVER builder/observation
implementation has been removed. Legacy handler code remains only for
compatibility/offline tests, not as a production fallback. PICK itself is
unchanged. Original piper_0 PICK/internal lift -> piper_0 gives to piper_1
remains the intended same-session workflow, with no new LIFT/STAP/home/reset.

Final locked remote offline suite: **2119 passed, 2 skipped, 16.82 seconds**;
45 tests added this continuation. Fifteen related modules/test files pass
Ruff; repository-wide historical lint cleanliness is not claimed.

This is NOT a runnable physical HANDOVER yet. A concrete live phase transport,
actual CLI constructors, verified local contact-geometry/calibration inputs,
and all instance bindings are still missing. Nominal collision certificates
also do NOT cover execution tracking error automatically: the admitted joint
tracking budget needs geometric inflation/certification, and actual co-hold
finger/attachment deviations require recertification of the fixed paths,
never relaxed limits or silent replanning. Actual contact/synchronization/
release/separation/follow qualification remains user-operated simulation work.
The installed profile stays execution_qualified=false and both production
admission gates remain before runtime/PICK. No simulation, robot command,
input/provenance overwrite, commit or push occurred.

### PhysX API references (unchanged)

- [OmniPhysics contact reports](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html)
- [PhysX contact point normal and impulse](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_contact_pair_point.html)

The installed IsaacLab schema code was also inspected for the pinned
`CreateThresholdAttr` API. These references support API usage, not physical
HANDOVER qualification.

## Authorized live simulation baseline (2026-10-07)

The user subsequently authorized the agent to launch simulation verification.
The original headed `python -m rocobrick.piper_run --robot-operation-plan ...`
entry and structure/manifest/config flags were retained. No STAP, PICK parameter
change, profile qualification override, or legacy HANDOVER fallback was added.

Two independent evidence directories were created:

- `runs/pick-handover-validation-20261007T015645Z-42c5a9/`: the prepared
  PICK -> HANDOVER input parsed correctly, then exited 7 with
  `HANDOVER_PROFILE_NOT_QUALIFIED`. No runtime or motion was created. The
  concrete live phase transport and formal CLI constructors remain missing.
- `runs/pick-lift-sim-validation-20261007T015706Z-32e00f/`: the original
  PICK-only input completed the actual headed Isaac/ROS execution with
  `EXECUTED`, `passed=true`, `PICK_CONFIRMED`, lift/gripper/holding evidence,
  `execution_safe=true`, and `cleanup_passed=true`. Four holding samples
  measured 0.10333908488973975 m lift, maximum relative position change
  1.9773827082039706e-05 m and orientation change 0.001043580844504289 rad.
  Receiver approach and HANDOVER were not dispatched.

The worker shut down orderly, the host stopped world/timeline/rclpy orderly,
and the owned container was removed with no session leftovers. The separate
host smoke `passed=false` reflects intentional early stop before its 300 s
hold, not failure of the formal run. Input immutability checks passed; the
original PICK and draft plan SHA values are unchanged. No production source
edit, commit, or push was performed in this verification step.

### Actual contact guard mismatch

The shared sample at sequence 60 / simulation time 60.03333646431565 s includes
ten actual PhysX contacts from both named **giver piper_0** pads to the brick.
The formal paired observation parser accepts its identity/time binding.
This is a post-close, PRE-LIFT capture, not a post-lift retention window or a
receiver/co-hold proof.

Offline replay through `require_giver_retention_sample` rejects it with
`reported pad contact is outside this object`: a reported contact exceeds the
exported USD world AABB by 0.00021723174353210917 m (0.217 mm), beyond the
current hardcoded 0.1 mm consistency tolerance. Contact/rest offsets, actual
collision versus display geometry, and physics-step/USD update timing have
not yet been disambiguated. Capture collision-local bounds, margins, and
coherent timing before changing this check. Do not merely widen tolerance or
discard failing contacts to manufacture retention evidence.

The actual generated combined model also passes read-only provenance capture:
11 unique collision mesh files resolve through the configured
`robot_assets/piper_l` package root. Native Pinocchio/Coal loading succeeds
with nq=18, nv=18, and 22 collision geometry objects. This is model-loading
verification only, not an isolated-scene/planning/tracking certificate.

There is still no actual HANDOVER trajectory/attempt in this validation.
The evidence does not establish bottom-approach infeasibility, an incorrect
giver pose, or a gripper/brick dimensional impossibility. Formal integration,
contact guard geometry/timing, tracking envelopes, and transfer qualification
remain outstanding.

## Explicit Isaac trial integration and actual diagnosis (2026-10-07)

The user subsequently authorized a bounded contact relaxation and agent-run
simulation. The formal entry remains the same, with explicit
`ROCOBRICK_HANDOVER_SIMULATION=1` added to the existing headed PICK/lift command
and the immutable two-operation plan. The installed profile remains unqualified;
the opt-in authorizes one exact Isaac session/manifest/model/profile, not hardware.

Contact positions have a profile-bound 0.5 mm trial allowance (default 0.1 mm).
Named physical pads, same-frame identity/time, opposing normals, measured
retention/cohold/follow and terminal proofs are unchanged. Authored Cube geometry
comes directly from BodyCollider and its local transform, not a moving world AABB.

The live adapter now binds the same original PICK/lift ledger and persistent
session to native full-model planning, prepared paired trajectories, actual
gripper feedback, atomic attachment transfer/readback and measured custody.
There is no independent LIFT, home/reset, STAP or receiver-PICK fallback.

Actual attempts:

- `runs/pick-handover-sim-trial-20261007T023325Z-5578bd`: PICK/lift confirmed,
  HANDOVER 128 candidates rejected (giver IK 82 / receiver IK 46).
- `runs/pick-handover-sim-trial-20261007T024339Z-v2-88a2f1`: PICK/lift confirmed,
  HANDOVER 128 rejected (giver 78 / receiver 50); HANDOVER motion_sent=false.
  Owned host/worker stopped orderly, container removed, no owned leftovers.

Later changes have not yet been validated by a subsequent full simulation:

- Bound-aware trust-region IK with finite residual/seed/deadline limits,
  reproducible seeds, and bounded repeated-target caching.
- Explicit world-fixed base-link/tableTopActor installation contacts only:
  no moving-link, cross-arm or gripper collision exemptions.
- Capture found piper_1_joint2=-4.9665e-10 rad at its zero lower bound. Only
  <=1e-8 numeric roundoff is snapped into unchanged bounds in geometry;
  raw feedback remains unchanged and real excursions remain rejected.
- Trial search 1024 candidates / 120 s. Existing model limits and workspace
  remain unchanged.
- Conservative per-shape downstream chain/mesh extents/full finger stroke
  bound relative motion. Common rigid ancestors cancel; 0.0005 rad tracking
  uncertainty remains. Mesh local bounds are cached. Disjoint padded world
  AABBs supply LOWER clearance bounds; overlapping pairs still use full native
  mesh collision/distance. No point sampling alone certifies a path.

The full captured grid replay checked 945 candidates: giver IK rejected 576,
receiver IK rejected 365, 4 paired endpoints were valid, but all 4 complete
phase paths were rejected. One approach has nominal inter-gripper distance
about 1.401 mm versus about 6.800 mm conservative tracking bound. Smaller
receiver opening alone also did not certify that case. Neither endpoint IK
nor an attachment ledger write is physical handover proof.

Latest full offline tests: 2131 passed / 2 skipped; Ruff passed. Finger-stroke
radius completion followed that test run and must be rechecked. Larger bounded
giver tilt candidates are being investigated read-only. Complete HANDOVER has
NOT succeeded in simulation; no hardware qualification, commit or push.

## Simulation-only guarded finger contact — 2026-10-07

The user approved a phase-scoped contact handler after pausing tuning to
review the failure. The original PICK/lift, operation input, combined free
rendezvous, controller cancellation and custody ledger remain unchanged.
There is no new simulation launch or physical motion in this implementation
turn. Installed profiles remain hardware-unqualified.

`ROCOBRICK_HANDOVER_SIMULATION=1` now binds a finger contact policy SHA into
the simulation authorization/program and records its values in run evidence.
Only approach, receiver_close, donor_open, scene_transfer and separation may
admit the four exact cross-arm gripper_link1/2 pairs, with contact positions
restricted to the provenance-bound object-local box plus 3 mm. Wrist,
gripper-base, arm-link, environment and free-space collisions remain forbidden.
Receiver opening, combined MoveGroup rendezvous and receiver follow keep
their full clearance contract. No global SRDF/ACM or PhysX response is disabled.

The native geometry checks nominal mesh contact depth and region; it does not
simply deactivate the permitted pairs. Only these pairs use a different
contract: nominal swept geometry plus measured contact supervision rather
than guaranteed no-contact throughout the arm tracking envelope. All other
pairs and the 0.0005 rad controller tracking bound stay unchanged. This is
NOT a hardware safety certificate or a force/compliance controller.

Initial trial limits are 0.5 mm penetration, 2 N summed per pair including
impulse-derived total-force bounds, 0.01 N s impulse, 3 s continuous contact,
0.5 mm/s contact TCP translation and 0.2 s maximum observer gap. Contact
trajectory timing uses actual FK distances; actual observed speed is guarded.
The 0.5 mm pad/object position-consistency tolerance is a separate policy.

PhysX finger-contact reports never enter pad/object custody reports. The host
retains every physics step, including empty frames, for up to 0.5 s/512 frames.
Missing/discontinuous/stale/foreign history fails closed; transient force
peaks cannot disappear merely because the current frame is empty. Excessive
force/depth/region/contact duration raises into the existing cancellation
lifecycle. Separation must end with contact loss. Original physical co-hold,
retention and receiver-follow gates still precede final ownership adoption.

Full locked CPU suite: **2176 passed / 2 skipped**, 33 new regressions
since 2143. Eight touched module/test files pass Ruff. Source and host bridge
compile checks pass. Native Coal box contact penetration is negative in this
build; the guard uses its magnitude and deep-contact rejection is tested.
The concrete live observer and cancellation/no-giver-release path are tested
with explicitly synthetic physics/action fixtures, not live robot motion.

Read-only replay of the previously failing actual PICK/lift snapshot and full
URDF/meshes no longer stops on the original cross-arm finger clearance pair.
That candidate still fails approach with SEGMENT_DEPTH_BUDGET: the limiting
pair is piper_1_link2_0 / piper_1_gripper_base_0, approximately 1.038427 mm
conservative distance against 1.038417 mm relative motion/tracking bound,
without the required numerical margin. This is outside the permitted finger
pairs and stays strict. It is not an observed physical collision.

Nominal zero-clearance crossing now includes the penetration budget, with a
whole-region proof from the intersection of object-relative swept full-shape
AABBs. A central contact witness cannot exempt an oversized contact patch;
zero-motion segments cannot bypass the region proof either.

The full original bounded grid still gives 945 evaluated: giver IK 576,
receiver IK 365, endpoint-valid 4, phase-valid 0. Independent replay of those
four endpoints confirms approach rejects candidates 56/716 because complete
contact localization cannot be certified. Candidate 323 is limited by
piper_0_gripper_link1_0 / piper_1_gripper_base_0, approximately 4.505 mm versus
6.462 mm relative tracking/motion envelope. Candidate 425 has the link2/base
limitation above. `contact_region_pair` diagnostics distinguish localization
failure from ordinary clearance failure. Neither the whitelist nor the region
was expanded to admit these candidates.

Complete HANDOVER has NOT succeeded in simulation. Free-space combined
rendezvous, actual contact forces and stop latency remain unvalidated for this
trial. No hardware qualification, commit or push.

## 2026-10-07: exact-distance refinement and complementary receiver grasps

This update supersedes the current-blocker conclusion above while preserving
the historical run record. Changes are deployed; no new Isaac/ROS session,
controller dispatch, commit or push occurred in this implementation turn.

### Re-established cause

At the four formerly rejected approach states, the actual mesh distances of
the limiting pairs were approximately 27.889 mm (finger/finger), 95.786 mm
(giver finger/receiver base), and 162.978 mm (receiver link2/its gripper base).
The previously reported small distances were conservative AABB lower bounds,
not observed collisions. Refining only those distances, with tracking/contact
limits unchanged, lets the original four approach paths pass. Their original
receiver closing paths then reveal mesh collisions: maximum reported depths
about 2.035 mm for three candidates and 5.180/7.220 mm for the fourth, partly
outside the allowed localized contact region. Those original relative grasps
are not acceptable shallow contact; this does not prove all grasps infeasible.

### Implementation and unchanged safety boundary

- Ambiguous AABB clearances are refined against native meshes before rejection.
  The exact result uses the SAME relative motion/tracking certificate and is
  cached only for the currently checked state. Nonfinite or contradictory
  negative refined distances fail closed. No pair, tracking envelope, depth
  limit or contact region is removed or relaxed.
- Optional profile fields `receiver_lateral_offsets_m` and
  `receiver_depth_offsets_m` default to `(0.0,)`; old profile JSON still parses.
  Fields enter the complete policy SHA and generation bound. Values must be
  finite/unique/nonempty, with absolute lateral <=10 mm and depth in [0,8 mm].
  The explicit simulation profile uses lateral (0,+4,-4,+8,-8) mm and depth
  (0,2,4) mm. Sideways shifts use tool X, extra depth tool -Z, not closing Y;
  the original projected closing span remains. Sideways centers beyond the
  object's projected tangent half-width minus observation margin are rejected.
- Coarse workspace/orientation coverage precedes refinement. `plan_handover`
  prioritizes `grasp_variants` only after both IK and paired endpoint admission,
  preserving the object/giver targets. All candidates are deduplicated and use
  the SAME 1024 evaluation / 120 s trial budget, not a larger blind search.
- Closed-receiver simultaneous holding geometry is checked FIRST in
  `_prepare_paths`, before approach/sweeps. New `cohold_endpoint` diagnostics
  include exact rejected collision pairs. This nominal precheck never replaces
  swept validation or runtime physical pad/object cohold evidence. Successful
  candidates no longer retain a predecessor's path failure diagnostics.
- PICK/lift, formal CLI, contact handler, giver-release gates and custody
  adoption are unchanged. Contact-phase continuous two-arm co-optimization is
  NOT implemented here: existing Cartesian approach/separation prepared paths
  still move one arm at a time. Future co-motion must separately propagate the
  shared timeline and held-object constraints through planning and execution.

### Latest no-motion validation

Replay uses the actual sequence-57 snapshot from
`runs/pick-handover-sim-trial-20261007T023325Z-5578bd` and its complete exported
URDF/SRDF/meshes, not a newly executing scene. A geometry-only holding stub
admits this historical replay; it is not fresh custody authority.

An initial diagnostic found a valid +4 mm tangent / +2 mm depth combination.
The formal hierarchical search, using the actual `HandoverJointPlanningBackend`,
then reaches fully validated contact-phase geometry at evaluation **929** in
approximately **106.56 s**, within the unchanged budget. Prior rejections:
giver IK 554, receiver IK 339, phase paths 35.

Selected candidate:
`d4f9f45bb16095cb60075d00f8b8d59649a013b6eab7c782774a5b0d71716a06`.
Object target approximately (0,-0.05,0.32) m; receiver tangent 0 and extra depth
4 mm. Receiver-open/approach/receiver-close/donor-open/separation/receiver-follow
contain respectively 2/22/2/2/22/8 complete states. Continuous full-scene native
checks pass. Both nominal receiver finger/object pairs report five contact
points at the original squeeze target. These are expected model contacts,
NOT measured PhysX pad forces or physical retention proof.

The diagnostic stops BEFORE the real combined MoveGroup query. No fabricated
combined plan, shared scene/attachment write, motion, or custody transfer is
used as evidence. Combined rendezvous, live session timing, physical contacts,
stop latency, retention/follow and complete simulated HANDOVER remain unproven.

Independent final replay of the selected grasp passes again. Trial profile SHA:
`5255be81fa0c5ac6b9c8a4dd31625f04698ef41bb86fc3877e49ef1587bafde8`;
request binding:
`af943ee9a5a2a1c2a17f43c90a0f30701e8bce57e9e249b31b3944bc40948578`.
Captured URDF/SRDF SHA respectively:
`0a96d11ae907cdf02293edb210b08c258843f415c1b1f98d95a472e9f3d26391`,
`4ef22ba68b6d386c3628249a5976c21e003ce6cc3a4dd823cab97390bc1321c3`.
Original PICK and prepared complete-handover input plan SHA remain unchanged.

Locked CPU regression: **2195 passed / 2 skipped**, 19 more than 2176.
All eight touched source/test files pass Ruff and formatting checks. New tests
cover exact clearance refinement with unchanged tracking, profile compatibility
and bounds, closing-axis-preserving offsets, prioritized search, deduplication,
budget/foreign binding, preserving valid grasps after an invalid first offset,
cohold-before-approach and cleared diagnostics. Scoped `git diff --check` passes.
Existing
deep contact, interior collision, cancellation and no-premature-release tests
remain passing.

Next: verify the actual combined rendezvous and PhysX handover through the same
successful formal PICK/lift -> HANDOVER entry. No STAP, independent LIFT, new
PICK or alternative startup path is needed. Hardware qualification stays false.

## 20. Snapshot clock alignment and headed exit diagnostics (2026-10-07)

The previous headed run `pick-handover-sim-validation-20261007045300Z-afe1c7`
completed PICK/lift but rejected HANDOVER before planning: the service snapshot
was 50–67 ms ahead of the last received ROS /clock sample. No receiver motion.
The user authorized fixing this ordering and inspecting headed exit frames.

`RosMoveItRobotSession.observe_all` now pumps real clock callbacks until they
reach the validated snapshot time, within the original observation timeout.
Stalled clocks time out; a clock moving backwards while waiting fails closed.
The original digest/provenance and freshness gates remain. No fabricated time,
collision/contact relaxation, new PICK, independent LIFT or STAP is introduced.

Opt-in `ROCOBRICK_EXIT_DIAGNOSTICS=1` records before-stop and after-stop images
before closing Kit, together with PhysX/USD gripper body poses, bounds and
authored joint relationships. Native screenshots require an Isaac window whose
`_NET_WM_PID` is exactly this host PID; no full desktop screenshot. The stopped
world gets bounded asynchronous Kit updates, not reentrant app.update or added
motion. Failures are best effort and never prevent normal teardown. Default
runs retain their original shutdown behavior.

Locked CPU suite: **2205 passed / 2 skipped**. The new clock test file passes
Ruff; baseline-versus-current lint comparison of the three existing modified
files reports no new diagnostics. Existing whole-file lint debt remains; this
is not an all-files Ruff-pass claim. Scoped whitespace checks pass.

Formal run `runs/pick-handover-exit-diagnostics-20261007T051100Z-f385c5`, runtime
`20261007T051013.525027Z_2c7d03da-efee-4c59-9ee5-93259d160968`:

- Original PICK/lift confirmed, lift approximately **109.02 mm**.
- Unified observations now have positive scene ages **50.00 / 16.67 ms**;
  snapshot/clock admission succeeds and reaches the read-only combined query.
- HANDOVER fails with `read-only combined planning failed: 'int' object is not
  callable`. Read-only source inspection identifies an interface mismatch:
  `unified_planning_scene_version` is an integer `@property`, but the read-only
  adapter invokes it as a method. Not a no-solution/collision proof. This turn
  does not repair/retry this newly exposed blocker.
- HANDOVER motion_sent=false; receiver commands=0, giver commands=3680 include
  PICK/lift and holding. CLI exit=10; host orderly exit=0, cleanup_passed=true,
  no session-owned process/container leftovers. Immutable plan SHA unchanged:
  `3d0db58db2dc8114c0599d7336c6929f93bec89af939c773742c8ad819cfebca`.

All three original frames (during planning, before stop, after stop) were
visually inspected. A ground-level gripper-like shape is already visible while
physics is running, not just after exit. However, before-stop PhysX/USD poses
agree for the recorded gripper bodies. Giver fingers z=110.26/110.16 mm, base
and link6 z=246.01 mm; receiver fingers z=239.43 mm, base z=227.59 mm. Their
recorded bounds are not on the ground. Enabled authored joints alone are NOT
proof of runtime constraint integrity. No HANDOVER inter-arm collision occurred
because no HANDOVER motion was sent. The ground shape's specific prim remains
unidentified; do not declare all accessories intact or a visual-only defect.
Next inspect camera/stand/accessory and mesh/body binding and initial state.
After timeline stop, unavailable PhysX transforms and reset USD authored poses
cannot be interpreted as in-motion detachment evidence.

Remote original frames/JSON are under runtime-session/exit-diagnostics; the
local workspace has copies and formal results under
`evidence/pick-handover-exit-diagnostics-20261007T051100Z-f385c5/`.
Complete simulated handover remains unproven. No hardware motion, commit/push.

## 21. Current implementation checkpoint (2026-10-07)

This checkpoint collects the post-lift coordinated HANDOVER work since commit
`3b39182`. It preserves the formal `piper_run --robot-operation-plan` entry,
original PICK with its internal lift, shared Isaac/ROS session, immutable plan
and existing model assets. It does not introduce STAP, another PICK, or a
standalone LIFT requirement.

- The dedicated backend captures post-lift giver custody and calibrated
  object/pad geometry, searches paired giver/receiver poses with reception
  from below, proves full-scene phase geometry, and requests one twelve-joint
  `dual_arm` rendezvous plan. Independent arm plans are not concatenated.
- Complete prepared phase paths, a common simulation-time barrier, measured
  controller/physics synchronization, tracking feedback, peer cancellation,
  and physical contact/retention/follow observations are wired into the
  persistent worker/session interfaces. The protective giver owner is not
  changed until release, separation and receiver follow have been proven;
  interrupted release/transfer produces uncertain custody.
- Stationary continuation reuses the original confirmed PICK/lift holding
  fact. HANDOVER has a bounded layer deadline without changing other skills'
  deadlines or rewriting the operation plan.
- The explicit Isaac-only trial requires `ROCOBRICK_HANDOVER_SIMULATION=1`.
  Installed `execution_qualified=false` and hardware admission remain closed.
  Trial pad/object consistency tolerance is 0.5 mm. Shallow finger-to-finger
  contact is permitted only in bound contact phases/regions, guarded by depth,
  force, impulse, duration and speed limits. Wrist/base/arm/environment and
  free-space collisions remain rejected; neither SRDF/ACM nor PhysX collision
  response is globally disabled.
- Snapshot admission waits for real ROS clock callbacks. The read-only
  planning adapter now reads the integer scene-version property correctly.
  After bounded native search, production refreshes the observation envelope
  only after the full static-binding guard verifies unchanged geometry/state;
  it does not rebind the candidate or loosen freshness gates.
- Headed diagnostics record owned-window exit frames and physical/USD body
  data. Visual instance expansion is confined to physics-free `visuals`
  subtrees in the USD session layer. Original assets/colliders are unchanged.
  The user subsequently confirmed that the apparent ground-level gripper
  problem was a viewing-angle issue; section 20 is historical, not a current
  detachment diagnosis. The global USD scene-delegate experiment was reverted.
  The Fabric refresh hook records zero updates when that extension is disabled;
  it is not evidence of a verified rendering repair.
- Observer defaults now use focal length 28 mm, eye `(0, 0.60, 0.42)` m,
  target `(0, -0.10, 0.16)` m. These settings affect only the viewport, not
  robot sensors, calibration, collision geometry or motion.

Current locked CPU regression: **2223 passed / 2 skipped**. All newly added
Python modules/tests pass Ruff. Existing modified files retain historical lint
debt; this is not a whole-repository Ruff-pass claim. Generated runs, vendor
snapshots, backup files, model-audit edits, lockfile URL churn and unrelated
deletions/build configuration are excluded from the implementation commit.
No new simulation or hardware motion was launched for this checkpoint.

## 22. Latest run and confirmed limits mismatch (2026-10-07)

Latest completed record inspected: `runs/pick-handover-close-view-20261007T062922Z-8d412c`,
runtime `20261007T062934.506882Z_1a40ab57-048e-403c-88f5-5e4249042c21`.
The recorded run starts at 14:29 and terminates at 14:32 China standard time.

- PICK/lift is CONFIRMED; physical holding evidence records **97.865 mm** lift.
- HANDOVER reaches the actual combined MoveGroup query. The final planning
  log says the motion plan was computed successfully, but read-only admission
  rejects it with `read-only path does not match captured start/paired goal`.
- HANDOVER `motion_sent=false`; receiver received/applied commands=0. The
  plan-level `motion_sent=true` refers to PICK/lift, not a successful handover.
  The giver's 3122 applied messages include PICK/lift/holding, not proof of
  HANDOVER dispatch. No physical receiver close/giver release/custody transfer
  has been demonstrated.
- CLI exit=10, host orderly exit=0, `cleanup_passed=true`, and session-owned
  process/container leftovers are empty. The original operation-plan SHA is
  still `3d0db58db2dc8114c0599d7336c6929f93bec89af939c773742c8ad819cfebca`.

Read-only analysis of the saved combined request and captured model identifies
a concrete planner-contract mismatch:

| Joint | Requested target (rad) | URDF limit (rad) | MoveIt YAML limit (rad) | Target outside YAML (rad) |
| --- | ---: | --- | --- | ---: |
| piper_0_joint5 | +1.243157140 | +/-1.5620696 | +/-1.2217304 | 0.021426740 |
| piper_1_joint5 | -1.256762190 | +/-1.5620696 | +/-1.2217304 | 0.035031790 |

Native Pinocchio IK currently reads position bounds from the captured URDF
model only (about +/-89.5 degrees). MoveIt additionally applies the captured
`joint_limits.yaml` (about +/-70 degrees for joint5; joint4 is also tighter
than URDF). The requested joint5 targets are +71.228 and -72.007 degrees.
The same final combined-plan log explicitly warns that piper_0_joint5's goal
is above maximum and piper_1_joint5's goal is below minimum, then clamps both
constraints. These required changes exceed the requested/admitted **0.002 rad**
goal tolerance by approximately 10.7 and 17.5 times. Thus the candidate accepted
by native search is not an admissible exact goal for the production planner.
This is a confirmed limits mismatch and a strong explanation for the endpoint
rejection, not a proof that handover geometry is physically impossible.

The current failed result discards the raw trajectory and combines start/end
checks into one message, so it cannot establish which joint/check first raised.
The log also reports a tiny negative piper_1_joint2 start being brought within
bounds (captured value approximately -4.97e-10 rad); that value alone is far
below the 0.002 rad start threshold and is not a demonstrated blocker.
Clock/aggregator warnings remain worth inspecting separately, but did not
prevent this query from returning a plan.

Next work, not implemented in this checkpoint: bind native IK, complete phase
and continuous-path checks to the effective intersection of captured URDF and
MoveIt YAML bounds, with provenance for both. Reject/resample out-of-range
candidates before querying MoveGroup, rather than increasing limits, clipping
targets, or loosening the captured start/goal gate. Add per-joint start/end
deviation diagnostics and preserve rejected raw planning evidence, then verify
through the same PICK/lift -> HANDOVER entry. Actual synchronized rendezvous,
guarded contacts, release, separation and final receiver custody are still
unproven in end-to-end simulation; hardware remains unqualified.

## 23. Effective joint limits bound before MoveGroup queries (2026-10-07)

Section 22's confirmed limits mismatch is now closed for planning admission.
No limit was widened, no target is clipped, and no admission gate was
loosened.

- `HandoverModelFiles` optionally captures the combined MoveIt
  `joint_limits.yaml` with its own SHA-256 provenance; the live trial
  factory in `handover_live.py` now passes the session-generated dual-arm
  file. A changed file on disk fails `validate()` like every other
  captured input.
- `IsolatedHandoverGeometry` computes the EFFECTIVE bounds as the
  intersection of the exported URDF limits and the captured MoveIt
  overrides. An entry without `has_position_limits: true` never narrows,
  mirroring MoveIt semantics; an empty intersection, foreign joint name,
  inverted range or malformed entry fails closed. IK solve bounds,
  configuration snapping, `MODEL_JOINT_LIMIT_VIOLATION` and the prismatic
  motion-radius bound all use the intersection. A joint5 goal beyond
  +/-1.2217304 rad is now rejected inside native candidate search, before
  any MoveGroup query can silently clamp it.
- `perform_readonly_combined_plan` endpoint rejection now reports per-joint
  start/goal deviation dictionaries and preserves the complete returned raw
  trajectory in the failed result envelope, instead of one summarized
  message with the evidence discarded.

Locked CPU suite after the change: **2227 passed / 2 skipped** (2223
baseline plus four new regressions: YAML narrowing admission,
no-flag/inverted/foreign fail-closed behavior, capture provenance, and
per-joint deviation diagnostics with the preserved raw plan). All touched
source/test files pass Ruff. One pre-existing order-dependent flake in
`test_production_imports_no_tools_or_smoke_modules` was observed once and
reproduced neither in isolation nor in the full-suite rerun; it is
unrelated to this change, which adds no top-level imports.

No simulation, MoveGroup query or hardware motion was launched. The next
authorized trial must re-verify the same PICK/lift -> HANDOVER entry end
to end: candidates now either solve within the effective limits or the
bounded search reports the IK rejection honestly.

## 24. MoveIt-first dual TCP rendezvous (2026-10-07)

The user's run `runs/pick-handover-limits-fix-20261007T083748Z` confirmed
the effective limit fix: PICK/lift succeeded, but all 1024 native candidate
evaluations failed before any MoveGroup request (839 giver IK failures,
185 receiver IK failures). This proves the former native prefilter exhausted
its budget; it does NOT prove that MoveIt cannot find a joint solution, that
all physically valid handovers are impossible, or that collisions caused
this particular failure. The user subsequently approved changing the
planning order, not another native seed/grid increase.

### Implemented production path

- `NativeCoordinatedHandoverFactory` explicitly selects pose-first mode.
  For each existing object/grasp candidate, the host sends both calibrated
  giver TCP and receiver precontact TCP goals to one `dual_arm` MoveGroup
  query BEFORE native rendezvous IK/contact phase gates. Both arms may move
  during rendezvous; no fixed-giver path constraint is added. This remains
  a common twelve-joint trajectory, not two independent plans.
- The persistent read-only lane now accepts either both exact joint goals
  (offline compatibility) or both complete calibrated TCP goals. Mixed
  goals, wrong frame/tip/provenance, invalid quaternion/vectors and widened
  handover tolerances fail closed. Pose replies explicitly state that
  native FK validation is still required; plan-only success never permits
  release or asserts physical execution/custody.
- Each pose query gets at most 10 seconds, bounded by the remaining shared
  profile budget; the MoveGroup message also respects the request timeout.
  A recorded unsuccessful query rejects the candidate with
  `MOVEIT_POSE_PLAN_FAILED` and retains its reason as `last_moveit_failure`,
  so bounded search can continue. Malformed or foreign successful replies
  remain errors. There is no production fallback to native joint-goal
  planning after a pose query fails.
- The native provider checks the returned start and calibrated endpoint
  FK, then admits the actual returned joint endpoints into the existing
  full-pair/phase/swept-geometry checks under the effective URDF/YAML limits.
  A certified measured-start bridge is retained; there is no end bridge
  back to a native IK proposal and no endpoint clipping/rounding.
- The short insertion starts at actual receiver FK with its actual tool
  direction and constant orientation. Its endpoint is checked relative to
  the object derived from actual giver FK; individually tolerated TCPs do
  not imply a correctly aligned simultaneous grasp. The exact rendezvous
  endpoint joins the phase chain without resetting arm joints.

### Preserved boundaries and verification

The original formal PICK/internal lift -> HANDOVER launch/session/input
remain unchanged. This change does NOT expand the object/grasp candidate
grid, add object roll, remove the bottom-receive constraint, or implement
joint two-arm contact insertion. The last short bottom approach still holds
the giver fixed. Native IK is still used for that guarded contact handler,
separation and follow, not as the pre-MoveIt rendezvous filter.

Effective joint limits, full geometry/environment collision checks and the
existing phase-bound finger-contact policy are unchanged. Opening/closing,
cohold, release, separation, receiver retention/follow and final custody
proofs remain mandatory. No model/config/plan inputs were changed.

Locked CPU command `uv run --no-sync python -m pytest -q tests` passed:
**2259 passed / 2 skipped**, including 32 new regressions. They cover
pose-first ordering/no fallback, actual endpoint and phase continuity,
calibration/tolerance/start admission, relative contact alignment,
cohold/interior collisions, scene/deadline guards and dual-TCP message
construction. The nine touched Python files without historical lint
issues pass Ruff. `tests/test_coordinated_ros_runtime.py` retains exactly
its 13 baseline docstring diagnostics with no additions. The initial
unscoped pytest collection failed on unrelated ROS/vendor dependencies and
historical `runs` backup test-name collisions; the formal `tests/` suite is
the reported passing scope.

These CPU tests use an explicitly synthetic MoveGroup transport and do NOT
validate the installed ROS dual-TCP constraint sampler, real planning
success, PhysX contacts or complete handover. No Isaac/ROS motion trial,
hardware motion, Git commit or push was performed in this change. The next
step is a user-operated trial through the same headed PICK/lift -> HANDOVER
entry: inspect both TCP targets in the read-only requests, actual MoveGroup
outcome, native endpoint/path/phase rejections and (only when all planning
proofs pass) the existing physical phase evidence. Hardware remains
unqualified; complete simulated handover remains unproven.

## 25. Real Humble dual-TCP sampler allocation fixed (2026-10-07)

The user-operated run `runs/pick-handover-moveit-first-20261007T093401Z`
confirmed PICK/lift but sent no HANDOVER motion (receiver piper_1 received
zero commands). Of 248 recorded read-only requests, 247 returned
`ABORTED / error_code=99999` and one failed freshness admission. The
retained MoveGroup log reports `Unable to construct goal representation`.
The final report's 249 evaluations / exhausted budget was therefore NOT
evidence that all targets or collision-free paths were unreachable.

### Confirmed cause and scoped fix

The combined kinematics YAML registered both `piper_*_arm` and
`piper_*_arm_tcp` KDL solvers. Humble's compound IK allocator greedily
consumes disjoint joint sets in allocator-name order. The ordinary arm
groups won their shared six joints before the TCP groups; their link sets
exclude `piper_*_rocobrick_tcp`. Consequently the dual-arm constraint sampler
could not associate either TCP pose constraint with its allocated subgroups.

`combine_kinematics` now emits only each robot's explicitly configured
`*_arm_tcp` solver for the COMBINED model. It fails closed on missing,
malformed or duplicated TCP solver entries. Per-robot input YAMLs and all
SRDF groups are retained; ordinary arm groups remain available for joint
goals. Single-chain KDL is not assigned to the branched dual_arm group.
The original PICK pose path already uses arm_tcp. The installed combined
launch helper is a symlink to the edited source, so the original launch
command loads the fix without rebuilding the image.

The read-only worker also preserves the actual MoveIt error code in failure
lifecycle evidence. Internal code 99999 now propagates immediately as
`MOVEIT_INTERNAL_FAILURE`, rather than repeatedly rejecting geometric
candidates until the budget expires. Normal failed planning/IK codes remain
candidate rejections. Reaching the deadline before another query no longer
clears the preceding `last_moveit_failure` diagnostic.

### Installed-library A/B verification, not motion

An isolated `--network none` / private-IPC / no-controller container used
the installed MoveIt 2.5.10/KDL libraries, the captured URDF/SRDF, the
original mesh package (22 collision-bearing links loaded), and FK-derived
known-reachable goals expressed in each robot's base_link with the same
1 mm / 0.015 rad tolerances as production.

- Original four-solver YAML: dual_arm allocated the two ordinary arm
  subgroups; neither contained a TCP link. `SAMPLER=0`, `SAMPLED=0`.
- YAML produced by the FIXED PRODUCTION GENERATOR from the same per-robot
  captured inputs: dual_arm allocated both arm_tcp groups.
  `SAMPLER=1`, `SAMPLED=1`.

The captured URDF/SRDF and meshes were not rewritten. The probe only created
and sampled an in-memory planning state; it did not invoke MoveGroup actions,
controllers, Isaac or hardware. Initial probe attempts had missing mesh
resource indexing, and one incorrect ROS environment expansion; these were
diagnostic harness issues, corrected before the reported full-model A/B
verification. Probe source/build/reports are isolated under
`/tmp/piper-tcp-sampler-rTzGKA`, not committed or added to production.

Final locked CPU suite: **2272 passed / 2 skipped**, including 13 new tests
for TCP-only selection/input preservation/fail-closed admission, typed ROS
error propagation/internal failure classification and retained deadline
diagnostics. Six touched Python files without historical lint issues pass
Ruff. Generator diagnostics decreased from 6 to 5; the existing ROS message
test retains its 13 baseline docstring diagnostics. No new Ruff diagnostic
remains in the touched files.

This closes the reproduced sampler-construction failure, NOT actual
handover reachability or a complete physical simulation trial. Candidate
positions/orientations, effective limits, collision/contact policy and the
formal PICK/lift -> HANDOVER session remain unchanged. The next user-run
headed trial should now reach genuine pose sampling/path search, or report
its actual next rejection. Full simulated transfer and hardware qualification
remain unproven. No Git commit or push was performed.

## 26. Clean-reference sampler and ordinary MoveIt failure classification

The subsequent user trial `runs/pick-handover-moveit-first-20261007T100556Z`
reproduced a different Humble UnionConstraintSampler issue: OMPL aliases the
sample output and reference; output randomization made the first IK sampler's
reference link transforms dirty. A scoped plugin copies/updates the reference,
then delegates to the original subgroup samplers. Known-reachable full Piper
model goals passed the real plugin/OMPL pipeline (16 waypoints). This was an
incremental build of our configuration package, NOT a MoveIt/OMPL reinstall.

Section 25's numeric-99999 classification is superseded: MoveIt FAILURE=99999
also represents ordinary multi-attempt planning exhaustion. It no longer
automatically aborts all search as an internal integration defect. The actual
status/code remain in read-only evidence. A replay of the actual first fixed
TCP targets still failed IK after fixing dirty references; known-reachable
SDK fixtures are not proof of actual handover feasibility.

## 27. Coupled continuous preparation region (2026-10-07)

User scope: preserve original PICK/internal lift -> HANDOVER command, replace
fixed preparation points with a region, permit receiver angles independent of
world-up while retaining an exposed brick-local stud face, then run ONE headed
simulation and stop for instructions. No hardware motion or Git commit.

Production compilation now queries a separate `tcp_region` read-only contract.
The giver's measured post-lift TCP anchors a 0.12 m translation neighborhood
and 1.0 rad axis-wise orientation neighborhood; there is no mandatory extra
vertical lift or fixed-height grid. The measured giver/object transform and
authored object-local contact box are immutable. A MoveIt goal-sampler plugin
samples the giver region, derives the carried brick from that solution, then
samples receiver side-pinch orientation/continuous oblique angle and bounded
4 mm lateral/depth offsets relative to that same brick. MoveIt's own KDL IK
and OMPL combined path search own preparation; host point enumeration/native
rendezvous IK are no longer called by the production factory.

The stud-exposure preparation policy uses brick-local +Z, side-face closing
axes, and the receiver-palm-back hemisphere. This is a bounded geometric grasp
family, NOT proof that every arbitrary angle or grasp is physically feasible.
There is no positive-world-Z approach predicate. Native FK verifies the actual
returned endpoint lies in the coupled family; it is never rounded/replaced.
Each repeated region query has its own consumed-once mailbox identity. Exact
pose/joint read-only contracts retain their prior tolerance restrictions.

Contact/sweep/continuous-path/limit/scene gates remain mandatory before dispatch.
The receiver opens for the larger of the two eligible side spans, so a sampler
choosing the other side does not invalidate the prepared finger state. At
actual arrival the handler reconstructs and recertifies only the short insertion
from measured FK and measured object-pose agreement; it does not replan the
free-space route or move the giver during insertion. Fresh co-hold, release,
attachment, separation and receiver-follow proofs remain required; planning
success does not transfer custody.

Pre-trial verification: focused Python regression 112 passed; clean-reference
and region-rejection C++ cases 8 passed; full CPU regression 2288 passed / 2
skipped before the final world-direction/unique-query tests. Real installed
Humble/KDL/OMPL with the previous measured Piper start and 22 collision-bearing
links selected `RoCoCoupledHandoverRegionSampler` and returned a combined
region-goal path (error_code=1, 19 waypoints). That read-only SDK scene does not
include physical contact or all live world/attachment constraints, and is not
handover execution evidence. The SDK's separate raw sampling call failed from
the unadjusted epsilon-out-of-bounds reference; the planning adapter handled
that start before the successful pipeline query. No bounds were enlarged.

The one authorized headed trial and final regression results are recorded below
after completion; no automatic parameter tuning or second trial is authorized.

### One authorized headed trial: failed before region MoveGroup dispatch

Final Python regression: **2290 passed / 2 skipped**; touched Python files pass
Ruff. C++ gtest cases: **8 passed**. Unit message factories are permissive and
did not enforce generated ROS message field types; the direct C++ SDK probe
also bypasses the production Python ROS-message construction seam.

Run: `runs/pick-handover-region-20261007T115837Z`.
Runtime: `20261007T115858.744194Z_42bab913-99bd-4cd1-acbe-e82f21de2ea5`.
Terminal: **EXECUTION_FAILED**, process exit 10. PICK/lift is again physically
confirmed (`PICK_CONFIRMED`, motion_sent=true, measured lift 0.10592 m).
HANDOVER motion_sent=false; receiver piper_1 received/applied **0 commands**.

The sole read-only result records **The 'x' field must be of type 'float'**.
The new region message builder uses receiver identity quaternion integer
components `(0, 0, 0, 1)`, while generated ROS geometry_msgs requires Python
float fields. Construction fails before send_goal_async; the live region
MoveGroup action is therefore NOT queried. This is an implementation/type
integration defect, not evidence of impossible IK, excessive collision
constraints or a failed physical contact transfer.

The exception is AssertionError rather than one of the read-only boundary's
caught exception types. The outer worker then returns a generic lifecycle
without read_only=true; the host displays **read-only combined result
identity/capability mismatch**, which masks the retained lower-level type
error. Both the scalar conversion defect and this diagnostic masking remain
unfixed pending the user's next instruction. No code was retuned and no second
trial was started after this failure.

Session cleanup passed: own container removed, ROS worker terminal/orderly,
host exit 0 with sigint_orderly_stop, no session-owned residual processes or
containers. External operation-plan SHA remains
`3d0db58db2dc8114c0599d7336c6929f93bec89af939c773742c8ad819cfebca`.
Before-stop/after-stop headed screenshots and raw results are retained under
this run's exit-diagnostics directory and copied to the collaborating local
workspace's evidence directory. Full simulated handover remains unproven.

## 28. ROS region scalar types and diagnostic boundary fixed (2026-10-07)

Scope: repair the generated ROS double-field failure from section 27, then
run ONE headed PICK/lift -> HANDOVER trial and report without retuning.

`_append_region_constraints` now explicitly converts position components,
quaternion components, region radius and orientation tolerances to Python
float. The receiver identity quaternion also uses floating literals. Valid
integer-valued input metadata remains legal, unchanged input; the conversion
happens at the generated ROS-message boundary only. No TCP calibration,
external plan, region bounds, effective joint limits, collision allowances,
sampler or controller parameters were changed.

The read-only boundary now retains generated-message AssertionError as a
terminal no-motion failure carrying read_only=true, physical_execution=false
and the original error. It does not invent trajectory success, permit release
or hide the failure behind the worker's generic identity/capability wrapper.

Regression: float-only message setter doubles reproduce the precise original
integer assignment failure; both captured geometry and valid integer vectors
pass the fixed builder without mutating input. The read-only assertion test
checks the host still sees the original failure under its strict identity
envelope. Focused tests: **59 passed**. Full locked CPU suite: **2294 passed /
2 skipped**; four touched Python files pass Ruff.

A separate --network none, read-only-mount container used the existing ROS
Humble image and real production `_build_goal` plus real generated MoveGroup
and geometry_msgs factories. Quaternion(x=0) reproduced the prior failure;
the original captured failing region request and legal integer-input variant
then constructed, CDR serialized and deserialized successfully (2180 / 2044
bytes). No rclpy node, action client, MoveGroup query, controller or simulator
was created by that check; no ROS/MoveIt dependencies were reinstalled.

The authorized headed trial uses
`runs/pick-handover-region-float-fix-20261007T124823Z`, the unchanged prepared-v1
external plan, and the original ROCOBRICK_HEADED/HANDOVER_SIMULATION piper_run
entry. Final execution and cleanup evidence follow below when it terminates.

### Authorized single headed trial result

Runtime: `20261007T124834.144733Z_1279f0d8-189b-4692-9628-90b52ae2396a`.
Terminal: **EXECUTION_FAILED**, process exit 10. PICK/lift again reports
PICK_CONFIRMED, motion_sent=true. HANDOVER reports motion_sent=false; receiver
piper_1 received/applied 0 commands. No region read-only request/result was
created in this trial.

New blocker: **candidate precontact does not follow its tool approach axis**.
The preparation seed constructs contact/precontact from measured post-lift
geometry and then the HandoverCandidate constructor rejects its direction /
collinearity / identical-orientation invariant before dispatching the region
query. This is a pre-query geometry-consistency failure, not a MoveIt IK or
collision search result. The constructor emits a shared diagnostic for several
numeric conditions, and the failing seed's component residuals were not
persisted; the specific numerical cause is not yet established. In particular,
raw-quaternion angle comparison in pose_error is a follow-up diagnostic target,
not an asserted proven cause for this run.

The requested ROS scalar bug is verified fixed with real message construction
and CDR replay; this live trial stopped even earlier and did not exercise that
region message dispatch. It does not prove complete handover readiness. No
additional code fix, parameter change or second headed trial was performed
after the new failure. The next decision remains with the user.

Session cleanup passed: own container removed, host orderly exit 0, ROS worker
cleanup passed, no session-owned process/container leftovers. Original external
plan SHA remains
`3d0db58db2dc8114c0599d7336c6929f93bec89af939c773742c8ad819cfebca`.
Terminal results, command counters and before-/after-stop screenshots are
retained under this run and in the local workspace's evidence directory.
No Git commit/push or hardware motion was performed.

## 29. Preparation seed quaternion error localized and repaired (2026-10-07)

The section 28 blocker is now numerically reproduced using saved ROS worker
shared-scene responses from the same run, rather than inferred from screenshots:
sequences 73 and 74 contain raw object quaternion norm-squared values
0.9999999938174134 and 0.9999999982605685. Contact/precontact quaternion components
are identical, tool-axis error is zero, approach distance is 0.04 m, and
collinearity residual is below 1.4e-13 m. Nevertheless the old raw-dot/acos
pose_error reports false angles 0.0002223975999906021 and
0.00011796377370156664 rad, exceeding the unchanged 1e-7 rad candidate gate.
Sequence 75 has norm-squared slightly above one and the old clamp reports zero:
acceptance incorrectly depends on the sign of sensor norm roundoff.

Both failing snapshots reproduce the same constructor error in pure geometry
replay. The diagnostic uses actual saved TCP/object poses and local collider
descriptors with a minimal read-only request adapter, not a complete runtime
replay or a MoveGroup query. Normalizing only the comparison in that diagnostic
admits the seeds, while a genuine 1e-5 rad rotation remains measurable.

The production shared pose_error now validates both poses through the existing
strict pose contract, normalizes only local comparison copies, aligns quaternion
signs, and computes the shortest angle from quaternion difference/sum norms
using atan2. This avoids acos loss of precision near identical orientations.
Raw observations, hashes, immutable inputs and grasp transforms are not changed.
Zero, nonfinite, malformed or grossly nonunit quaternions fail closed. No region
bound, contact allowance, collision policy, joint limit, controller, sampler or
execution authorization was relaxed.

Regressions cover recorded raw rotations on either side of unit norm, q/-q
equivalence, genuine rotations down to 1e-10 rad, shortest-arc behavior, illegal
quaternions on either input, and production-formula HANDOVER seed construction.
The seed test also rejects a genuine precontact-only rotation at the unchanged
axis/orientation gate. Focused CPU regression: **180 passed**. Full locked CPU
suite: **2363 passed / 2 skipped** (18.97 s). The three touched Python files
pass Ruff. Replaying saved shared poses 73/74/75 against the repaired production
functions now admits all three candidates with zero false orientation error,
without any diagnostic function replacement.

No Isaac, ROS node, MoveGroup query, gripper or arm motion has been started for
this repair. Complete HANDOVER remains unverified; notify the user before any
headed execution. This repair does not establish that later IK, collision,
local insertion, co-hold or release gates will pass.

## 30. Quaternion repair: authorized single headed trial (2026-10-07)

Run: `runs/pick-handover-quaternion-fix-20261007T140404Z`.
Runtime: `20261007T140503.696532Z_3f745a01-f6c5-4221-91e8-af9739ca8482`.
Used the original headed/simulation/exit-diagnostics piper_run entry and unchanged
prepared-v1 external plan. Terminal **EXECUTION_FAILED**, exit 10.

PICK/lift: PICK_CONFIRMED, motion_sent=true, measured lift
0.09989526169374585 m. HANDOVER: motion_sent=false; piper_1 received/applied
zero commands, and no HANDOVER execution request exists. The prior preparation
seed quaternion rejection is gone; region queries reached actual MoveGroup.

There were 61 read-only region results: 60 returned combined trajectories and
one failed with ABORTED/error_code=-2. All 60 returned endpoints were rejected
by `moveit_region_endpoint:STUD_FACE_OR_SIDE_PINCH_CONSTRAINT`. Final:
`NO_FEASIBLE_HANDOVER_IN_PREPARATION_REGION; evaluated=61`.
No collision pair was reported at this gate. This does not prove later contact
or collision certification would pass; those phases were not reached.

Current blocker is agreement between MoveIt's region endpoint and the native
brick-local stud/side-pinch invariant, not the previous numerical seed error or
a complete absence of MoveIt trajectories. The shared rejection covers three
predicate components; their per-endpoint numerical values are not yet audited.
Next diagnostic should replay saved endpoint joints and compare sampler frame
semantics, calibrated TCP/object transforms and native FK without sending motion.

Cleanup passed: own container removed, host orderly exit 0, worker orderly,
no session-owned process/container leftovers, host port 17650 released.
All five original input SHA values are unchanged; external plan remains
`3d0db58db2dc8114c0599d7336c6929f93bec89af939c773742c8ad819cfebca`.
Before/after-stop screenshots and raw execution evidence are retained remotely
and copied to the collaborating local evidence directory. No tuning, second
trial, additional production fix, commit or push followed this result.

## 31. Paired preparation separated from local contact (2026-10-07)

User approved relaxing preparation-only side-axis requirements, keeping giver
and receiver jointly adjustable, adding local receiver correction, then ONE
headed simulated PICK/lift -> HANDOVER trial with advance notification.

HandoverPreparationEndpoint now represents actual paired MoveIt arrival with
the original binding and preserved giver attachment. Preparation checks the
existing giver neighborhood and receiver 0.15 m proximity; it does not require
final side-pinch/stud orientation. Both six-joint arm endpoints remain exactly
those returned by MoveIt. The donor is not locked at the original lift pose.

The local contact geometry projects the arrival's closing direction onto the
brick side plane continuously, rather than demanding X/Y-only pinch. Final
stud half-space and side-plane gates remain, as do actual pad, cohold, collision,
gripper sweep, joint-limit and custody proofs. The receiver correction is
bounded by 0.08 m and pi/2 rad; degeneracy or excess correction fails closed.
This geometric projection is a handler target, not host global IK/search.

Before dispatch, the provider certifies receiver-only alignment with strict
no-touch geometry, then insertion and the remaining complete contact/release/
separation/follow paths. Giver arm joints stay fixed only during local receiving,
not combined preparation. Local rigid-pose interpolation includes rotation;
the Cartesian tube proof uses rotational midpoints and bounds desired angular
motion as well as joint motion. The native collision/limit policy is unchanged.

At actual arrival, the runtime rebases the chosen contact target using measured
giver FK and the preserved object attachment, verifies measured brick drift,
then recertifies alignment/insertion from measured joints. It never executes a
nominal preparation endpoint or silently replaces MoveIt's returned joint goal.
Validation and the authorized headed trial outcome will be recorded below.

### 31.1 Validation and ONE authorized headed trial

Focused regressions: 220 passed. New preparation module: 24 passed. Final full
locked CPU suite: 2387 passed / 2 skipped (19.33 s). Six modified Python files
pass Ruff; git diff --check passes. Original inputs remained immutable.

Run: runs/pick-handover-preparation-local-20261007T145134Z.
Runtime: 20261007T145205.409477Z_badf84aa-3716-4d7c-8141-c643c85325b4.
Original headed/simulation/exit-diagnostics entry and prepared-v1 plan were used.
Terminal EXECUTION_FAILED (exit 10). PICK_CONFIRMED, motion_sent=true, measured
object lift 0.10827102558687329 m. HANDOVER motion_sent=false; receiver received
and applied zero commands; no preparation/contact motion was dispatched.

52 read-only region queries: 48 trajectories, 3 ABORTED/error_code=-2, final
query timed out. Terminal exception: read-only combined result identity/capability
mismatch. Identity fields match, but the timeout result omits lifecycle.read_only
and terminal. Capability checking precedes ordinary planning failure handling,
so this incomplete failure result masked the normal bounded-search diagnostic.
This is not evidence that physical HANDOVER or collision avoidance is impossible.

Offline FK and current production preparation/contact geometry replay (no IK,
ROS, simulation or motion) on the 48 successful endpoints: 41 rejected with
LOCAL_ALIGNMENT_BUDGET_EXCEEDED, 5 with COUPLED_PRECONTACT_REGION_ESCAPE, and
2 passed geometric admission only. No geometry-admitted candidate acquired
complete path/execution permission. The final exception does not persist their
path rejection counters; do not assert IK, collision or limit failure without
further read-only evidence. Broad preparation and bounded local-handler reach
still need alignment; merely removing the old side-axis gate is insufficient.

Cleanup passed: own container removed, host orderly exit 0, worker orderly exit
0, no owned leftovers, port 17650 released. All five original input hashes are
unchanged. Raw result, host evidence, screenshots and endpoint replay are retained
in the local evidence/pick-handover-preparation-local-20261007T145134Z directory.
Before-stop camera is occluded by the table edge; no contact conclusion is drawn
from that screenshot. No camera adjustment, tuning, second trial, additional
production fix, commit or push followed this result. Next proposed work: repair
timeout lifecycle/failure classification, retain phase rejection evidence, then
diagnose the two geometrically admitted endpoints before changing motion policy.

## 32. Execute paired preparation before contact admission (2026-10-07)

User explicitly approved separating safe preparation movement from final
HANDOVER feasibility and requested a headed verification of simultaneous
giver/receiver movement. Live simulation now opts into staged_preparation.
Hardware remains behind its existing qualification/full-program boundary.

CoordinatedHandoverPreparation binds the captured PICK/lift grasp, calibrated
receiver opening and returned combined MoveIt trajectory. The native provider
certifies the opening sweep and carrying path (both complete arms, unchanged
joint limits, grippers, carried brick, environment, tracking and scene binding).
It does NOT test local correction budgets, stud/side contact geometry, contact
IK, cohold, receiver closure, donor release or separation before preparation.
Both arm endpoints are still MoveIt's, on one shared time grid and synchronized
execution barrier. No extra lift or artificial demonstration movement is added.

Only after receiver opening and physical paired arrival does the factory capture
fresh measured joints/TCPs/object pose and compile all final contact paths. The
continuation has rendezvous=None and binds the executed preparation hash; it
cannot repeat receiver opening/preparation. Contact failure is reported as
HANDOVER_AFTER_PREPARATION, retains arrival evidence, stops both participants
and never opens the donor or publishes receiver custody. The operation remains
FAILED even if preparation succeeded. Measured joint changes and runtime start
barrier evidence are retained separately from command receipts.

TimeoutError now preserves the read-only/no-physical-execution envelope instead
of falling into the generic worker result. It retains terminal=false rather
than claiming that an unproven planning cancellation settled. No identity or
capability check is bypassed. Verification and the single headed outcome follow.

### 32.1 CPU regression and initial headed integration result

Focused tests: 96 passed; full locked suite: 2400 passed / 2 skipped (19.90 s).
Six changed Python files pass Ruff; git diff --check passes. Original inputs
are unchanged. Initial headed run: pick-handover-staged-preparation-20261007T152512Z,
runtime 20261007T152536.104798Z_018a7d73-61fd-4d8c-8841-03bb5fa43edd.
PICK_CONFIRMED, lift 0.11323465825989842 m. Two read-only queries were recorded;
the preparation program compiled and obtained its bound program hash. It failed
at first live observation, BEFORE receiver opening or arm dispatch, with
KeyError: scene_snapshot. Receiver commands remain zero; both-arm motion was
NOT verified. Cleanup passed, own host/worker exited 0, no own leftovers.

The concrete observe_handover_scene/observe_monitor contract returns the physics
snapshot DIRECTLY, not the idle observe_shared_scene planning wrapper. The live
observer incorrectly unwrapped it again; the finger-guard unit double had copied
that incorrect wrapper and hidden this latent integration defect. The consumer
now uses the direct snapshot, and the existing live-guard regression uses the
real contract. No provenance, monitor guard, contact policy or safety check was
relaxed. User is notified before an additional headed validation of this repair.

### 32.2 Second headed trial: missing physics-history ticks

Run: pick-handover-staged-monitor-fix-20261007T153241Z, runtime
20261007T153316.504904Z_a845431b-1a54-4f79-bf5c-756e510f65be.
PICK_CONFIRMED, lift 0.10391346924006939 m. Preparation compiled after one
read-only query, but the first live monitor rejected the rolling contact history
with "finger physics history is discontinuous", before HANDOVER dispatch.
Receiver commands were zero; simultaneous preparation motion was not verified.
The monitor contained 25 frames at sim_time_s=94.56667159870267: every sixth
sample interval was 33.333 ms instead of the declared 16.667 ms. Cleanup passed;
own host/worker exited 0, own container removed, no own leftovers and all five
original input hashes unchanged.

## 33. Render-only Kit updates preserve contact sampling (2026-10-07)

User explicitly requested repair of the periodic missing physics/contact tick.
Installed Isaac 5.1 SimulationContext.render_async guards its Kit update with
/app/player/playSimulations=false. The bridge's direct next_update_async lacked
that guard: the playing Kit timeline could step physics on each sixth-step
viewport refresh, after the bridge had already captured its contact report.

Readiness, operation hold and pre-stop diagnostic rendering now share a guarded
render-only coroutine. It refreshes Fabric explicitly, disables automatic Kit
physics for the awaited update, and restores the original boolean setting in
finally (including cancellation). Simulation time AND physics-step index must
remain unchanged across rendering, otherwise the bridge fails explicitly.
Manual world.step(render=False) and its immediate native contact capture remain
the sole intended step/sample pair. No timeline reset, synthetic history,
threshold relaxation, contact-policy change or dependency installation is used.
After-stop diagnostic updates remain after world/timeline teardown.

Offline regressions cover the six-step auto-tick reproduction, continuous
capture, settings restoration, cancellation, unexpected time/index advance and
the unchanged contact guard's rejection of histories with missing ticks.
CPU and authorized headed verification results follow after execution.

### 33.1 Regression and ONE headed verification outcome

Focused bridge/contact regressions: 157 passed. Full locked CPU suite:
2413 passed / 2 skipped (20.02 s). Existing Fabric-order regression now verifies
the shared guarded renderer, with refresh still preceding the non-reentrant Kit
yield. Ruff reports no new diagnostics relative to the captured baselines
(bridge had 13, its older test file had 2); modified contact/Fabric tests pass
Ruff outright. git diff --check passes. Unrelated dirty changes are preserved.

Run: runs/pick-handover-render-tick-fix-20261007155144Z.
Runtime: 20261007T155149.123047Z_4231616f-841d-48e5-b633-65d4c2bfe472.
Original headed/simulation/exit-diagnostics piper-run entry and prepared-v1
plan were used, with no input changes. 1926 render-only updates completed
without time/index advance errors. All nine saved native scene/contact
snapshots contain 30 consecutive frames each; the maximum interval deviation
from the declared 1/60 s is 8.692344032057431e-10 s. The earlier periodic 33.333 ms
interval is absent. No interpolation or continuity-policy relaxation was used.

Terminal EXECUTION_FAILED (exit 10), with a different downstream monitor
failure: "finger contact observer gap exceeded". PICK_CONFIRMED, motion_sent=true,
lift 0.1096079908311367 m. The first HANDOVER monitoring snapshots have sequences
76/77/78 at sim_time_s=89.48333800025284 / 89.73333801329136 /
89.96667135879397: the first interval is 0.25000001303852 s, exceeding the
unchanged 0.2 s observer limit despite each rolling physics history being
complete. HANDOVER motion_sent=false; receiver received/applied zero commands.
Neither simultaneous arm preparation nor a completed handover is established.
Preparation compiled, but initial repeated monitoring failed before dispatch.
The next issue is observation scheduling/latency versus advancing simulation
time, not missing native physics samples or demonstrated geometric infeasibility.
No observer-gap threshold tuning or further trial followed this result.

Cleanup passed: own host/worker exited 0, own container removed, no own process
or container leftovers, port 17650 released. All five original input hashes
remain unchanged. Result, operation run, host evidence, three HANDOVER native
monitor snapshots and before/after-stop screenshots are retained in local
evidence/pick-handover-render-tick-fix-20261007155144Z. No commit or push was made.
