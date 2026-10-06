# HANDOVER implementation checkpoint — 2026-10-06

This checkpoint includes the preceding agent's uncommitted unified dual-arm
runtime work and the continuation fixes. It is not a claim that physical
HANDOVER has succeeded. The live checks below use Isaac simulation, not hardware.

## Formal operation and execution boundaries

- `robot-operation-plan/v1` accepts PICK, HANDOVER, WAIT, PLACE_DOWN,
  PLACE_UP and SUPPORT_TOP. TRANSIT is rejected at the formal plan boundary.
- HANDOVER identifies one physical instance, a distinct donor and receiver,
  both participants and their robot/object resources. The executor is the
  receiver. PICK is an optional predecessor; execution still requires confirmed
  donor holding in the state ledger.
- The sole dispatch entry remains `python -m rocobrick.piper_run
  --robot-operation-plan ...`. The structure-bound direct entry does not require
  a STAP source plan. TCP, plan, structure and live readiness gates still apply.
- One Isaac bridge and one ROS worker serve the session. Shared planning uses
  the combined dual-arm MoveIt owner; controller/action identities remain
  robot-specific. Skill-owned free-space requests use the shared scene and
  namespaced TCP targets.

HANDOVER proceeds through receiver free-space approach, receiver gripper
feedback, co-hold observations, attached-scene transfer, donor release,
receiver-follow observations and atomic ledger adoption. Failed or missing
evidence cannot authorize owner adoption; uncertain execution retains resources.

## Continuation fixes

1. Added HANDOVER to direct dispatch, removed TRANSIT from that admission list,
   and repaired the missing lazy gripper-payload import.
2. Replaced the receiver's idle/current pose with the measured object boundary.
   Pose numbers are transformed into the receiver's base and labeled with
   `<receiver>_base_link` and `<receiver>_rocobrick_tcp`.
3. Reused successful PICK's calibrated 3.5 mm pad-midpoint clearance and
   floor/finger-tip margin. No collision checks or joint limits were disabled.
4. Added necessary receiver contact-geometry checks: a distant stationary
   object and plausible opening alone cannot establish co-hold. Observation
   session identity is bound to the original request.
5. Recomputed receiver-relative attachment geometry and its digest from the
   post-approach observation before transferring the planning-scene attachment.
6. Distinguished failures before entering the physical executor from uncertain
   failures after entering it. Known no-dispatch planning failures remain failed
   operations but carry terminal-safe proof, allowing orderly worker shutdown.
   After-dispatch exceptions remain unsafe. A failed operation blocks further
   worker admission even when teardown can be orderly.
7. Corrected HANDOVER motion reporting for explicit no-dispatch failures.

Submission cleanup also restores PLACE_UP and SUPPORT_TOP's independent
`preflight` methods, which the previous free-space refactor had inadvertently
placed after another method's return. Regression tests cover this boundary.
Static example1 and type1-C operation-plan fixtures now live under
`tests/fixtures/robot_operation_plans/`; original historical inputs are untouched.
Legacy TRANSIT tests were migrated without re-enabling the removed operation.

## Verification and evidence

The continuation's focused offline regression passed 469 tests. After submission
cleanup, the complete `tests/` suite passed 1922 tests with 2 skipped, and the
ROS dual-arm configuration suite passed 4 tests. Ruff passed
for the continuation's production fixes and new HANDOVER/preflight-boundary
tests. These checks send no robot motion; this is not a claim that the entire
inherited codebase is Ruff-clean.

```bash
PYTHONPATH="$PWD" uv run --no-sync python -m pytest -q -p no:cacheprovider tests/
PYTHONPATH="$PWD:$PWD/ros_ws/src" uv run --no-sync python -m pytest -q -p no:cacheprovider ros_ws/src/rocobrick_piper_l_moveit_config/test/
```

The direct PICK/HANDOVER structure and TCP preflight completed with exit code 0,
`runtime_created=false` and `motion_sent=false`. A preflight is not execution
success; its top-level `passed=false` deliberately does not certify assembly.

Untracked historical evidence retained on the development machine:

| Run | Observed result |
| --- | --- |
| `runs/pick-handover-headed-20261006T105444Z-792ad6` | PICK confirmed, approximately 10.4 cm lift. HANDOVER receiver IK failed after 24 attempts with code -21; logs also reported missing `base_link` TF. Worker safe-shutdown evidence failed. |
| `runs/handover-target-fix-preflight-20261006T111211Z` | Structure/TCP preflight completed without creating runtime or sending motion. |
| `runs/pick-handover-headed-retry-20261006T111512Z-VgcQEM` | PICK completed; HANDOVER receiver IK again failed after 24 attempts, now with code -31. The receiver executor was not entered. Worker shutdown was orderly with no retained robots. |

The latest run completed built-in teardown; no session container, launcher,
or listener remained. Historical run artifacts and immutable input files are
not included in this source checkpoint.

## Remaining limitations

- The latest receiver target has no solution among the 24 attempted IK seeds.
  The failure does not by itself prove that every possible grasp is unreachable.
  Inspect receiver grasp orientation, target position, joint limits and seed
  coverage before changing geometry or performing another live check.
- Co-hold, planning-scene transfer, donor release and receiver owner adoption
  have not yet been validated by a successful live HANDOVER. PLACE_UP is outside
  the live checks performed here.
- Strategy fields are validated in the formal plan, but the present physical
  path still uses the generic grasp geometry and three-sample observation loop.
  Profile-specific behavior and arbitrary `follow_samples` are not implemented.
- Geometric proximity is a necessary guard, not independent proof of grasp
  contact or sustained receiver holding.

This checkpoint excludes generated runs, ROS build/install/log directories,
vendor snapshots, probe tools, experimental stage-4 fake-controller installation,
the unrelated model-audit date change, lockfile churn and unrelated file
deletions. Those local changes are preserved.
