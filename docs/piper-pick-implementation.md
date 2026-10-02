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
PLACE_DOWN remains fail-closed until connection, release and retreat evidence
are implemented. This verification is not a hardware validation or a guarantee
for every object, grasp orientation or robot configuration.

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
  proves holding.
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
0.05 rad holding-follow orientation tolerance. The lift target includes the
standoff; the increment is not the total observed object rise. These settings
are verified for the example1 supply geometry and remain configurable.

## Session And Cleanup

All plan robots share one Isaac bridge, DDS participant and clock, with separate
namespaced ROS action/state endpoints and per-robot TF topics. The configured
host hold budget remains active through operation and observation phases.

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
