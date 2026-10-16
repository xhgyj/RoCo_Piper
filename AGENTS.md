# RoCo Piper Agent Instructions

## Project

This repository runs the RoCo BrickAssembly Piper L fork on Linux x86_64 with Python 3.11, Isaac Sim 5.1, BrickSim, and ROS 2 Humble/MoveIt. Main areas:

- `src/rocobrick/`: contracts, `piper-run`, multi-robot scheduler/session, ROS/MoveIt adapters, scene and skill logic.
- `isaac/`: Isaac/BrickSim host bridge and simulator-side tools.
- `ros_ws/src/rocobrick_piper_l_moveit_config/`: namespaced dual-Piper MoveIt package and launch/config generator.
- `config/`, `tasks/`, `robot_assets/`, `artifacts/model-audit/`: fixed runtime inputs and provenance.
- `tests/`: contract, offline integration, and runtime-adapter tests.
- `runs/`, `ros_ws/build/`, `ros_ws/install/`, `ros_ws/log/`, `.zcode/`: generated/local state; do not commit them by default.

Read `README.md` and `config/README.md` when changing repository-wide setup or input formats.

## Commands

Use the repository Python environment and keep the locked dependency set:

```bash
uv sync --locked
PYTHONPATH="$PWD" uv run --no-sync python -m pytest -q -p no:cacheprovider tests/
uv run ruff check
```

For focused work, run the smallest relevant test files first. Runtime-contract tests can be run without launching Isaac/ROS. A real runtime check requires an NVIDIA GPU, Docker, the `roco-piper-ros:humble` image, an unused DDS host listener port, and a fresh evidence directory.

Do not launch real Isaac/ROS/MoveIt or send robot motion from ordinary unit-test work. Real runs must use unique `runs/` paths, preserve input files, and leave session-owned containers/processes cleaned up. Never stop or delete resources belonging to another run.

## Execution Architecture

The production path is:

```text
assembly-plan/v2
  -> STAP parse and structural binding
  -> TCP gate/authorization
  -> external robot-operation-plan/v1
  -> multi-robot readiness
  -> layer scheduler
  -> RobotOperationExecutor
  -> operation/cleanup/evidence aggregation
```

`piper-run --robot-operation-plan PATH` is the only operation dispatch entry. Do not reintroduce the old `--operation` or `--skill-request/v1` dispatch paths. `piper-run` must read and validate the external plan; it must not reorder, repair, or synthesize operations. Failed STAP, binding, TCP, plan, readiness, or cleanup gates must prevent motion and must produce an auditable failure.

The external plan is immutable input. Preserve its original bytes and SHA-256 in the run directory, rehash all original inputs after execution, and keep `robot-operation-plan-run.json` JSON-safe. Final success requires all required assembly evidence and cleanup, not merely a command being sent.

The runtime topology is one Isaac host bridge plus one ROS/MoveIt container session for all plan robots. The host bridge has one DDS participant and one shared `/clock`; container ROS processes share the configured DDS domain. Robot endpoints are namespaced (`/piper_0/...`, `/piper_1/...`) and must never fall back to canonical `piper_0` names for another robot. Scene state is served globally at `/p05_host_bridge/scene_state` and is sliced by `robots[robot_id]` with no cross-robot or snapshot fallback.

The scheduler treats `execution_layers` as the dispatch/barrier unit. Same-layer operations may run concurrently; `planned_schedule` is evidence/log order only. Resources are acquired before dispatch and released only after confirmed terminal-safe evidence. A failed operation cancels unfinished peers; later layers become blocked. `asyncio.Task.cancel()` is not proof that physical motion stopped. Unknown or unsafe terminal state fails closed and retains resources.

`OperationExecution` and `CancelEvidence` are strict contracts. Do not accept truthy non-booleans, missing terminal evidence, or legacy results in production. Every robot operation must carry robot identity, terminal state, motion status, and JSON-safe evidence.

## Skill and Directory Boundary

The production skill path is split by responsibility:

```text
src/rocobrick/
├── piper_run.py                         # formal CLI/run orchestration
├── multi_robot_scheduler.py             # layers, dependencies, resources, cancel
├── execution/
│   ├── robot_operation_executor.py     # operation dispatch and result evidence
│   ├── operation_state_ledger.py       # confirmed per-robot/physical state
│   └── core.py                          # execution primitives
├── skills/
│   ├── programs.py                      # phase definitions
│   ├── targets.py                       # target geometry and TCP provenance
│   ├── wait.py                          # dedicated WAIT handler
│   ├── transit.py                       # dedicated TRANSIT handler
│   ├── pick.py                          # dedicated PICK handler
│   ├── place_down.py                    # dedicated PLACE_DOWN handler
│   ├── observation.py                   # identity/freshness/following checks
│   └── contracts.py                     # skill context/result protocols
├── adapters/moveit/
│   ├── transit.py                       # current TRANSIT target/action adapter
│   ├── motion.py                        # planned shared action lifecycle adapter
│   ├── gripper.py                       # planned gripper action/feedback adapter
│   └── planning_scene.py                # planned scene synchronization adapter
├── runtime_session.py                   # Isaac/ROS/MoveIt session lifecycle
├── ros_session_entry.py                 # ROS session and robot namespaces
├── ros_moveit_runtime.py                # action/controller/terminal lifecycle
└── moveit_execution.py                  # MoveIt execution contracts
```

The old top-level compatibility forwarding modules and same-named shadow files
have been retired. Import the classified owning packages directly. The root
package and skills package no longer eagerly re-export legacy APIs.

The old gRPC planner, pregrasp/playback, phase backend, isolated single-session
and legacy launch/fake-controller paths are retired. The sole dispatch entry
remains piper-run with an external robot operation plan. Standalone validate/demo entries are retired; archived maintenance tools under maintenance/archived_tools/ are not production dispatch dependencies.

Current capability boundary:

- `WAIT` confirms only from the executing robot's live observation: its own
  simulation time must advance and the requested measured object must remain
  stable.
- Empty TRANSIT to a measured part target is implemented through the MoveIt
  adapter and final TCP feedback gates.
- Keep all formal skill branches: WAIT, TRANSIT, PICK, PLACE_DOWN, HANDOVER,
  SUPPORT_TOP and PLACE_UP. Their implementation and fail-closed evidence gates
  define capability; a Task C trajectory is not proof that other skills are dead.
  A carrying request must be rejected before motion unless the ledger confirms
  the requested holding state. Runtime success requires fresh physical evidence;
  the cleanup tests do not certify that every live skill succeeds.
- `PICK` must keep pregrasp, approach, gripper feedback, grasp, lift, and
  holding evidence separate. Gripper opening or brick height change alone does
  not prove holding.
- `PLACE_DOWN` must separately prove connection, release, and retreat before
  publishing an assembled state.
- Same-layer operations may run concurrently for different robots. Each
  operation needs its own phase context, evidence, and action handle; each
  robot needs an independent goal/result/cancel channel. A robot, object,
  connection, or shared planning-scene resource cannot be concurrently owned.
- Do not replace action feedback with `sleep`, command publication, or a
  successful request return. Do not release a resource or mark an assembly
  complete without terminal, connection, release, and retreat evidence.

## Editing Rules

- Preserve unrelated dirty worktree changes; do not reset, checkout, or delete them.
- Keep changes within the owning module boundary. Update focused tests with contract changes.
- Prefer existing dataclasses, protocols, and JSON schemas over new parallel abstractions.
- Do not commit simulator/vendor snapshots, generated build/install/log trees, run evidence, probe scripts, or temporary files without an explicit request.
- Maintain Python 3.11 compatibility and the repository's Ruff rules. Avoid `typing.Any` and blanket `typing.cast`; use concrete types, `Protocol`, or `object` as appropriate.
- Treat calibration, plan, task, scene manifest, and model-audit artifacts as immutable provenance inputs. Never modify them in place during a run.
