# RoCo Piper

RoCo Piper is the formal multi-robot operation-plan runtime for the Piper L Isaac Sim, ROS 2, and MoveIt environment.

This README describes the current production workflow. Older challenge examples, standalone policy demos, validation entrypoints, historical runs, and diagnostic probes are not supported production interfaces.

## Current Scope

The supported production workflow starts from an external robot-operation-plan/v1 document and executes it through the multi-robot runtime:

    external robot-operation-plan/v1
      -> multi-robot readiness
      -> execution layers and scheduler
      -> RobotOperationExecutor
      -> operation, cleanup, and evidence aggregation

The formal execution surface does not use the former assembly-plan/v2, STAP, or TCP-gate workflow as its production entry path. Those names may still occur in historical artifacts or offline compatibility code; they are not instructions for starting a current production run.

The formal skill surface is:

- WAIT
- PICK
- PLACE_DOWN
- HANDOVER
- SUPPORT_TOP
- PLACE_UP

A valid plan must preserve its declared operation order and layer semantics. The runtime does not invent, reorder, or silently repair operations.

## Offline Architecture Migration

`src/rocobrick/task_plan/`, `src/rocobrick/contracts/`, and
`src/rocobrick/scheduling/` contain the first offline event-driven foundation.
The new `robot-task-plan/v2` parser is **not a production CLI input** yet.
Production dispatch and motion behavior are unchanged. See
[implementation status and remaining safety gates](docs/event-driven-foundation-status.md).

## Requirements

The supported environment is:

- Linux x86_64
- Python 3.11
- Isaac Sim 5.1
- ROS 2 Humble and MoveIt
- NVIDIA GPU and compatible drivers
- Docker
- The project ROS image, normally roco-piper-ros:humble
- A configured DDS environment and an unused host listener port

Offline contract tests do not require Isaac Sim, ROS, Docker, or robot hardware. A real runtime check requires all of those runtime dependencies and a clean, isolated evidence directory.

## Installation

The dependency set is locked with uv:

    uv sync --locked

Use the repository environment for commands:

    PYTHONPATH="$PWD" uv run --no-sync piper-run --help

The project requires Python >=3.11,<3.12. Do not update the lockfile or replace the Isaac/ROS dependency set as part of a runtime run.

## Production Entry Point

The only production dispatch command is:

    piper-run \
      --robot-operation-plan PATH_TO_ROBOT_OPERATION_PLAN_JSON \
      --structure-goal PATH_TO_GOAL_JSON \
      --run-dir PATH_TO_NEW_EVIDENCE_DIRECTORY

Use an explicit initial structure when the operation requires one:

    piper-run \
      --robot-operation-plan PATH_TO_ROBOT_OPERATION_PLAN_JSON \
      --structure-start PATH_TO_START_JSON \
      --structure-goal PATH_TO_GOAL_JSON \
      --config PATH_TO_USER_CONFIG_JSON \
      --run-dir PATH_TO_NEW_EVIDENCE_DIRECTORY

Useful optional inputs include:

- --scene-manifest PATH
- --scene-snapshot-out PATH
- --host-instance-id ID
- --robot-id ID
- --config USER_CONFIG_JSON

The run directory must be new. The runtime records the input plan and execution evidence there and must never overwrite an existing run directory or mutate the source plan.

Run help from the locked environment to see the exact options available in the checked-out revision:

    PYTHONPATH="$PWD" uv run --no-sync piper-run --help

The retired rocobrick-validate, run/demo.py, run/main.py, and run/validate_plan.py commands are not supported and must not be restored as compatibility wrappers.

## Runtime Topology

A production run uses one Isaac host bridge and one ROS/MoveIt container session for the plan's robots.

- The combined dual-arm MoveIt launch is the supported ROS launch path.
- Robot topics and action channels are namespaced per robot, including /piper_0 and /piper_1 where applicable.
- The Isaac host uses one DDS participant and one shared clock source.
- Container ROS processes use the configured DDS domain and project Fast DDS profiles.
- Persistent ROS worker channels remain owned by the runtime session and are cleaned up with the run.
- Scene state is selected by robot identity; a robot must not fall back to another robot's state or to an old snapshot.

The production implementation lives primarily in:

- src/rocobrick/runtime_session.py
- src/rocobrick/ros_session_entry.py
- src/rocobrick/multi_robot_scheduler.py
- src/rocobrick/execution/
- src/rocobrick/skills/
- src/rocobrick/adapters/moveit/
- isaac/p05_host_bridge.py
- ros_ws/src/rocobrick_piper_l_moveit_config/

## Execution and Evidence

The scheduler treats execution layers as dispatch and barrier units. Operations in the same layer may run concurrently when their resources do not conflict. Later layers remain blocked when a required operation fails.

Resources are acquired before dispatch and released only after confirmed terminal-safe evidence. Cancellation is not proof that physical motion stopped. Unknown or unsafe terminal state fails closed and retains affected resources.

A successful run requires:

- Readiness for every participating robot
- Valid operation identity and resource ownership
- Terminal action evidence
- Skill-specific physical evidence
- Cleanup and session shutdown evidence
- A JSON-safe run summary

Primary run artifacts include:

- robot-operation-plan.json - captured external input bytes
- robot-operation-plan-run.json - execution, readiness, cleanup, and result evidence
- Session, scheduler, and skill evidence files produced during the run

Rehash original inputs after execution. Preserve the original plan and configuration bytes as provenance.

## Formal Skills

### WAIT

Confirms fresh observation from the executing robot. Simulation time must advance and the requested measured object must remain stable. A stale or cross-robot observation is not valid evidence.

### PICK

Keeps pregrasp, approach, gripper feedback, grasp, lift, and holding evidence separate. A gripper opening or object-height change alone does not prove that the robot is holding the object.

### PLACE_DOWN

Separately proves connection, release, and retreat before publishing an assembled state. A command or successful action result alone does not establish placement.

### HANDOVER

Coordinates donor and receiver ownership, observation freshness, local geometry, attachment state, and cleanup. The ledger must confirm the requested holding state before carrying or release actions are admitted.

### SUPPORT_TOP

Uses the support-top operation contract and its dedicated readiness, planning, execution, and evidence gates. Do not replace those gates with generic motion success.

### PLACE_UP

Uses the place-up operation contract and requires its release, retreat, and final evidence conditions. Do not mark completion from a command publication alone.

All formal skills must retain fail-closed behavior, independent action handles, robot identity, terminal state, motion status, and JSON-safe evidence.

## Repository Layout

    src/rocobrick/                         Runtime, scheduler, skills, contracts
    isaac/                                  Isaac host bridge and simulator-side tools
    ros_ws/src/rocobrick_piper_l_moveit_config/
                                            Dual-arm MoveIt package and sampler/plugin
    config/                                 Runtime configuration and user inputs
    tasks/                                  Task and structure data
    robot_assets/                           Robot models and scene assets
    artifacts/model-audit/                  Model and TCP provenance artifacts
    artifacts/phase1/                       DDS prerequisite evidence
    tests/                                  Offline and runtime-contract tests
    maintenance/archived_tools/             Retired diagnostic and maintenance tools
    maintenance/archived_tests/             Tests for archived tools
    runs/                                   Local generated evidence; do not commit by default
    ros_ws/build/                           Generated ROS build output
    ros_ws/install/                         Generated ROS install output
    ros_ws/log/                             Generated ROS logs

The formal production closure includes the runtime session, scheduler, executor, all six formal skills, dual-arm MoveIt package, sampler/plugin, Docker self-check, DDS gate evidence, and immutable model/TCP/calibration/scene inputs.

## Required Production Inputs

Treat these inputs as immutable provenance during a run:

- External robot-operation-plan/v1 JSON
- Structure start and goal data when required by the selected plan
- User/runtime configuration
- Scene manifest and scene assets
- Robot model and calibration data
- Model-audit and TCP evidence artifacts
- DDS prerequisite report at artifacts/phase1/dds-multiprocess-gate.json

Formal maintenance tools supporting this boundary include:

- tools/isaac-ros-env.sh
- tools/generate_dds_profiles.py
- tools/smoke_p1_dds_multiprocess.sh
- tools/p1_host_dds_probe.py
- tools/p1_container_dds_probe.py
- tools/piper_l_model_audit.py
- tools/sim_collision_hulls.py
- tools/validate_piper_assets.py

These tools support validation or prerequisite generation; they are not alternate production dispatch entries.

## Testing

Run the repository's offline test suite with the locked environment:

    PYTHONPATH="$PWD" uv run --no-sync pytest -q -p no:cacheprovider tests/

Run focused contract tests when changing the corresponding boundary:

    PYTHONPATH="$PWD" uv run --no-sync pytest -q -p no:cacheprovider \
      tests/test_piper_run.py \
      tests/test_tcp_structural_binding.py \
      tests/test_p1_dds_multiprocess_gate.py \
      tests/test_dual_robot_profile_launch.py \
      tests/test_host_bridge_p05.py \
      tests/test_piper_l_model_contract.py

Run formatting and lint checks when applicable:

    uv run ruff check
    uv run ruff format --check

Tests that require ROS package dependencies may need to run inside the ROS build environment. Offline tests must not launch Isaac Sim, ROS, MoveIt, Docker, or robot motion.

## Real Runtime Safety

Before a real runtime check:

1. Confirm NVIDIA, Isaac Sim, Docker, ROS, MoveIt, and DDS prerequisites.
2. Use a unique new run directory.
3. Confirm that plan, configuration, model, calibration, and scene inputs are the intended immutable files.
4. Confirm that the ROS image and required host ports are available.
5. Start only the session owned by this run.
6. Preserve logs and evidence.
7. Confirm orderly container, process, worker, and session cleanup.

Never send motion from a unit-test command. Never stop or delete another run's containers or processes. Never reuse an existing evidence directory for a new execution.

## Archived and Historical Material

The following are intentionally outside the production dispatch path:

- Retired validation and demo entrypoints
- Old single-session and fake-controller launch paths
- P1/R2/P5 live probes and smoke runners
- Legacy DDS diagnostic scripts under the archived maintenance tree
- Historical run outputs
- The original upstream challenge README and submission instructions

Archived tools are retained for historical inspection or separately controlled maintenance work. They are not supported deployment commands and are not evidence that the current production runtime is healthy.

## Known Limits

- A clean offline test run does not prove that Isaac Sim, ROS, MoveIt, DDS, or robot hardware is available.
- ROS package tests may require a sourced ROS environment and a built workspace.
- Runtime success requires physical and terminal evidence, not merely a successful process return code.
- The HANDOVER implementation has known issues tracked by the existing tests; this README does not change or waive those contracts.
- Generated run evidence, ROS build/install/log trees, vendor state, and simulator outputs are local state unless explicitly requested for archival.

## Further Documentation

- AGENTS.md - engineering rules, production architecture, and safety boundaries
- config/README.md - configuration and input descriptions
- docs/cleanup-session-handoff.md - cleanup history and current boundary
- docs/live-code-cleanup-analysis.md - live-code classification and rationale
- artifacts/model-audit/ - model and TCP provenance

When documentation and runtime behavior disagree, verify the checked-out piper-run --help, the operation-plan schema, and the owning production module before changing an interface.
