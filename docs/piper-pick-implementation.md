# Piper PICK Implementation Status

## Scope

The verified path is a simulated PICK on the original Piper L robot assets,
using Isaac Sim 5.1, BrickSim and the namespaced ROS 2 Humble/MoveIt session.
The runtime URDF, finger meshes and USD physics layers remain identical to the
pinned vendor assets. No reversed finger installation is required.

The only operation dispatch entry is `piper-run --robot-operation-plan PATH`.
Direct skill plans bind to explicit structure inputs. `--preflight-only` checks
the same inputs without launching the runtime. STAP-backed assembly runs still
require assembly effects; a standalone PICK does not claim completed assembly.
PLACE_DOWN now has independent connection, release and retreat evidence gates.
The full example1 PICK/PLACE_DOWN sequence has been verified in simulation.
This is not a hardware validation or a guarantee for every object, grasp
orientation or robot configuration.

## Calibration And Geometry

The old selector used STL facets 2 through 8 on a proximal metal mounting region,
not the distal contact surface. The corrected selector uses facets 545 through
548 on each original-mounted finger. These four connected inward-facing facets
match a separately materialized visual surface in the DAE asset.

Fifteen measured pose/opening samples confirm:

- `link6 -> rocobrick_tcp` translation z: approximately 0.130589587 m;
- TCP orientation: identity relative to link6;
- physical distal-pad separation: logical opening, with unit slope and a
  numerically negligible intercept (no additional 8 mm compensation);
- finger-tip extension beyond the TCP centroid: approximately 5.211 mm.

Accepted provenance is stored in `config/calibration/piper_l_tcp_source.json`
and `artifacts/model-audit/piper_l_tcp_{selector_discovery,calibration,binding_contract}.json`.
`isaac/calibrate_piper_l_tcp.py` performs bounded selector discovery and measured
calibration. `tools/publish_piper_l_tcp_calibration.py --template PATH` validates
the passed calibration and its inputs, then publishes measured values and hashes
into the binding contract and generated TCP xacro. Calibration is a maintenance
operation, not something to run concurrently with a production session.

The planning checkout has a flange spacer and different finger origin depths.
`dual_robot_config.py` aligns those origins with the original runtime model
without changing finger orientations, meshes, axes or limits.

## Execution And Evidence

PICK executes separate planning-scene, opening, pregrasp, approach, closing,
planning attachment, lift and following-verification phases.

- Approach height includes the measured finger-tip extension and floor margin.
  The ground remains a hard planning constraint.
- Isaac retains physical gripper/brick collisions. MoveIt allowed contact pairs
  permit planning contact; they do not disable PhysX contact.
- Gripper results require fresh measured feedback. A contact-limited canceled
  close is accepted only with safe terminal cancellation, settled finite
  feedback and an opening inside the expected window. Closing alone never
  proves holding. The post-action settle carries the in-loop no-net-progress
  anchor: stall time already served while the action ran is not re-waited,
  but the carry is accepted only after one fresh post-result finger state
  stays within the stall drift of the anchor, otherwise the full window
  restarts. The arm's post-action at-rest dwell (8 live samples at 0.2 s
  cadence) is unchanged: the arm converges at the action result, so that
  window is the physical proof itself, not paperwork.
- The planning attachment uses measured brick geometry in the TCP frame. It is
  a MoveIt collision representation, not an Isaac physical attachment.
- Following verification compares pre-lift grasp evidence with advancing
  post-lift observations. It checks actual object rise and relative pose
  stability before confirming holding.
- Goal sampling uses a tighter fraction of the requested tolerance, while the
  final live FK gate still applies the actual requested tolerance.
- A bounded direct-controller endpoint-completion path is retained for the
  container execution pipeline; final FK evidence remains required.

`config/skill_params.json` currently uses a 4.0 s gripper trajectory, 0.002 m
per-finger goal tolerance, 0.08 m pregrasp standoff, 0.03 m lift increment and
0.35 rad PICK holding orientation tolerance and 0.05 rad resting-stack stability
tolerance. PICK still requires measured object rise and positional following;
the relaxed angular bound accommodates rotation while grasped and does not
prove a universal pad-escape angle. The lift target includes the
standoff; the increment is not the total observed object rise. These settings
are verified for the example1 supply geometry and remain configurable.

## Placement

`skills/place_down.py` runs transport, alignment, insertion, connection check,
release, planning-scene detach/world restoration, retreat and final verification.
The external plan supplies these operations; `piper-run` never appends a placement
to a PICK-only input. Both operations run within the same Isaac/ROS session.

The host publishes exact stud/hole interface geometry from the validated goal
connection, using BrickSim's read-only `compute_connection_transform`. It does
not call `create_connection` during execution. Placement targets are computed as
`T_world_lower * T_lower_upper * T_upper_tcp`, where the grasp transform is measured
live after lift. This handles interface offsets, heights, yaw and off-centre
holding without using AABB tops (which include studs) as insertion surfaces.
The simulator's assembly thresholds remain unchanged, and its native connection
detector supplies the connection verdict.

Regular BrickSim `brick` prototype dimensions provide explicit shape symmetries:
rectangles allow half turns and squares allow quarter turns. Other or unknown
asset types retain only their nominal pose. These candidates do not change the
external plan's geometry or operation order. General geometry is covered by
varying heights, offsets, rotations and base orientations in offline tests;
physical end-to-end verification currently covers example1 only. Non-upward
stud axes and unreachable targets fail closed rather than implying universal
shape or workspace support.

The earlier three 1.5 mm position corrections were removed after a user run
showed roughly 10 degrees of support deflection and no failure retreat.
After the microstep era, contact insertion was structurally simplified to the
reference policy's shape: ONE continuous, speed-capped, collision-checked
descent per placement and no retries. Pilz LIN plans once to a commanded
bound BELOW the seated depth (`place_press_m`, 4.5 mm — a bound, not a press
to hold: the in-flight guard monitor cancels the action the moment the exact
connection forms or any physical guard fires, so the deep bound absorbs
gravity droop and grip compliance). The plan is retimed to the measured
Cartesian distance at the 1 mm/s contact cap, executed as one
FollowJointTrajectory, and monitored live: support tilt (3 degrees; position
2 mm), held-object slip (3 mm / 0.15 rad), native interface force (8 N when
available), wrong interface identity or stale feedback stop the motion.
Controller tolerances for contact moves are deliberately generous
(goal 0.03 rad / path 0.05 rad): contact loads and uncompensated droop are
physical, and the scene monitor — not tracking tightness — is the authority.
A non-connection stop re-inspects once (a cancel can race the connection
forming during the settle), then withdraws once and fails closed; there are
no press retries, no replanning during contact, and no re-alignment under
load. `place_interface_align_tolerance_m` (2.5 mm) is the descent gate:
measured studs still self-centre within roughly this offset, and larger
offsets get at most two measured free-space corrections before failing
closed. Native projected interface force is diagnostic, not a calibrated TCP
wrench; missing diagnostics remain explicitly unknown.

`ROCOBRICK_PLACE_CONTACT_PROBE=1` selects the validation variant: guard mode
`probe` stops the descent at the FIRST measured interface contact (laterally
overlapping AABBs within 0.5 mm along the support axis), after which the
operation fails closed following a verified withdrawal. It never releases,
never publishes a placed state, and is how controlled contact plus
controlled unloading are validated before insertion is trusted.

A safely stopped unconnected contact enters ONE supervised withdrawal (guard
mode `unload`): a single guarded ascent past the clearance bound, cancelled
in flight if the connection appears or the held relation slips. Recovery
requires measured 6 mm clearance, retained holding and restored support; an
unknown action terminal or an active connection prohibits closed-finger
pulling. One unload/re-align/reapproach retry is allowed without resetting
the original holding reference; it also covers a controller tracking abort
at a rim hit (live evidence: a spike past the 0.01 rad path tolerance at
-1.0 mm engagement with zero support deflection is assembly physics, not
divergence — the physical guards stay authoritative). The action lifecycle
remains separate from physical safety: a terminal action alone does not
prove successful unloading or placement. The host natural-exit grace was
reduced from 90 s to 2 s so failed pressing setpoints do not remain active
during a long process-cleanup wait.

Connection must persist over advancing observations with exact interface,
offset/yaw-equivalent rigid geometry evidence. After opening the stack must
return to its target and original support pose, then remain stable. Retreat
requires actual TCP rise with both bricks stable and connected. Only then does
the ledger revoke holding and publish placed state.

On 2026-10-02 the continuous guarded path completed example1 twice in a row
with gravity feed-forward enabled. `runs/place-continuous-20261002T140628Z/`
returned `EXECUTED` exit 0: one 7.2 s seated descent (max joint tracking
error 0.00013 rad) plus two 0.5 mm press increments, the guard cancelling the
third in flight on the exact connection (support tilt peaked at 2.73 degrees,
under the 3 degree bound), connection/release/retreat each confirmed over
advancing samples, TCP retreat 50.4 mm, cleanup passed. The repeat at
`runs/place-continuous2-20261002T140834Z/` connected after one press with
1.00 degree maximum support tilt and 49.5 mm retreat. The probe variant was
validated first at `runs/place-probe2-20261002T140433Z/`: the descent stopped
at the first measured contact (0.47 mm gap, action cancelled safely) and one
continuous withdrawal verified 7.98 mm clearance with support and holding
restored. An earlier 5 mm/s probe attempt (`runs/place-probe-20261002T140146Z/`)
aborted at 0.024 rad viscous tracking lag, which set the 1 mm/s cap.

Interface alignment correction and the single-descent design were validated
on 2026-10-02. Without gravity feed-forward the same simple flow succeeded
once (lateral 0.55 mm) and failed safely four times (lateral 0.93-1.68 mm:
the un-compensated arm both scatters its endpoint and transmits a weaker
press, so studs ride rims until the support guard stops the descent; every
failure withdrew once with verified clearance). The decisive comparison:
lateral 1.64 mm failed without compensation while lateral 1.68 mm connected
with it — the discriminator is press effectiveness, not alignment. With
`ROCOBRICK_GRAVITY_FEEDFORWARD=1` (now the default) the single-descent flow
connected in
flight (`runs/place-simple-g-20261002T153845Z/`) with connection/release/
retreat evidence and clean cleanup; gravity feed-forward is a controller-layer
compensation this simulated arm physically needs for reliable placement, not
part of the placement strategy.

On 2026-10-02 the complete example1 run at
`runs/place-complete-20261002T093024Z/` returned `EXECUTED`, exit 0, with
`PICK_CONFIRMED` and `PLACE_DOWN_CONFIRMED`. Object lift was 105.77 mm; TCP retreat
was 52.95 mm. Connection, release and retreat each carried three advancing
samples. Worker, container and host cleanup passed with no session leftovers.
Earlier attempts failed closed at unreachable orientation, unconnected insertion,
or release verification and remain in their own evidence directories.

That earlier successful run is NOT acceptance evidence for the newer guarded
contact implementation. Guarded tests at `runs/place-guarded-20261002T104942Z/`
and `runs/place-guarded-20261002T105229Z/` stopped failed insertion and verified
6.02 mm / 6.18 mm physical unloading, restored support and retained holding.
The final guarded test at `runs/place-guarded-final-20261002T105955Z/` stopped at
approximately 3.06 degrees of support deflection, but could not prove unloading
because held-object drift exceeded the recovery envelope. It returned
`EXECUTION_FAILED` with an unsafe operation terminal, despite orderly process
cleanup. The guarded path has not yet completed example1 successfully and must
not be presented as a reliable assembly implementation. Small MoveIt LIN actions
still exhibit tracking drift; they are supervised collision-planned actions,
not the old simulator's high-rate servo loop or a calibrated force controller.


## Precise Contact Execution

Guarded contact now separates collision planning from controller execution.
MoveIt plans a LIN path from explicitly measured joint state; every trajectory
point passes identity/limit admission before the complete path is submitted to
that robot's FollowJointTrajectory controller. Short paths are retimed to at
least 0.4 s without changing geometry; goal tolerance is 0.0003 rad, path
tolerance 0.01 rad, start-state drift limit 0.002 rad. Post-result fresh joint
feedback and live TCP FK are required; the final TCP position error must be
within 0.5 mm unless an identified contact stop already ended the step.
Controller aborts retain their ABORTED action status and may only authorize
recovery, never a successful insertion or assembled state.

The immutable USD uses finite-force position drives. Slow strict contact tests
still showed static joint2/joint3 errors of about 0.003 / 0.0016 rad. Joint
force observations matched model gravity (approximately 5.3 / 4.5 Nm), while
TopicBasedSystem's verified 1e-5 default deadband did not explain those errors.
Arm-joint gravity feed-forward applies live articulation
model gravity forces to arm joints only, bounded to 10 Nm per joint; finger
feed-forward is zero, assets/gains/limits stay unchanged. Missing/nonfinite or
out-of-bound model feedback raises rather than silently continuing. Dynamics
observations record efforts, model forces and whether compensation was enabled;
this does not claim a calibrated TCP wrench.

With compensation, measured short-path TCP errors fell to roughly 0.017-0.032 mm
and arm joint errors to 1e-5-1e-4 rad in the 2026-10-02 tests, and the
single-descent placement presses reliably enough for the studs to self-centre.
It is therefore ON by default (same controller-layer role as a real motor
controller's gravity compensation); `ROCOBRICK_GRAVITY_FEEDFORWARD=0`
explicitly disables it for comparison runs, and every run's evidence records
which mode was active.

The final compensated run `runs/place-contact-budget-20261002T122340Z/`
executed 20 insertion microsteps with final TCP errors 0.015-0.036 mm and
maximum sampled support tilt 0.265 degrees. At the local contact budget it
returned `EXECUTION_FAILED` with confirmed safe unloading: actual TCP rise
6.48 mm, holding position error 0.033 mm, restored support and passing cleanup.
This is verified precise tracking and failure recovery, not successful assembly.

## Session And Cleanup

All plan robots share one Isaac bridge, DDS participant and clock, with separate
namespaced ROS action/state endpoints and per-robot TF topics. The configured
host hold budget remains active through operation and observation phases.

Since 2026-10-02 the production session keeps ONE persistent container ROS
worker (`src/rocobrick/execution/ros_session_worker.py` +
`ros_worker_transport.py`) instead of one `docker exec` child per phase. Every
phase envelope keeps its own request id, fresh all-robot readiness admission
(one report reused as gate evidence instead of re-probing), per-robot operation
ownership, strictly validated atomic mailbox results and operation-targeted
cancellation; replay, stale results and worker death fail closed, and session
cleanup additionally requires the worker's own orderly-shutdown attestation
(`cleanup["ros_worker"]`). One-shot child mode remains available for tests via
`persistent_ros_client: false`. This removed roughly 3 s of per-phase process
restart overhead without touching the settling/verification evidence itself.

Kit can handle SIGINT outside the bridge coroutine. To prevent that exit from
skipping final evidence, the bridge handler records only stop intent. The
physics loop detects it and enters the existing cleanup boundary, recording
ROS shutdown, world stop, timeline stop and application close initiation.
`early_stop_accepted` requires the existing strict attestation, corroborated by
the session's SIGINT exit record. Killing a process or finding no leftovers
alone does not prove successful cleanup.

## Verification

On 2026-10-02, a headed run using the original assets and corrected calibration
completed with `EXECUTED`, `passed=true`, exit code 0 and `PICK_CONFIRMED`.
Measured object rise was 104.09 mm; maximum relative position change was
0.62 mm. Cleanup passed through the accepted SIGINT path, with no session-owned
container or process leftovers. PhysX contact-point locations were not sampled,
so these measurements prove physical lift/following, not every contact triangle.

The local evidence directory is
`runs/pick-cleanup-fixed-20261002T062022Z/`; run evidence, vendor snapshots and
ROS build/install/log trees are intentionally not committed.

Run offline tests without launching Isaac or ROS:

```bash
PYTHONPATH="$PWD" uv run --no-sync python -m pytest -q -p no:cacheprovider tests/
```

A headed runtime check requires the GPU, Docker image, display session and an
unused DDS listener port. Supply an existing external operation plan and use a
fresh evidence directory:

```bash
RUN_DIR="runs/pick-headed-$(date -u +%Y%m%dT%H%M%SZ)"
ROCOBRICK_HEADED=1 PYTHONPATH="$PWD" .venv/bin/piper-run \
  --robot-operation-plan /absolute/path/to/pick-only.json \
  --structure-goal tasks/type1/example1/structure_goal.json \
  --structure-start tasks/type1/example1/structure_start.json \
  --run-dir "$RUN_DIR"
```
