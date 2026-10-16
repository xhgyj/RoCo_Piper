# Dependency recovery baseline

Read-only verification on 2026-10-15 in /home/mzictt/RoCo_Piper. This is an observed dependency map, not proof that a fresh checkout builds or runs. No upstream fetch, checkout, submodule initialization, dependency sync, or simulator launch was performed.

## Verified local vendor identities

| Local directory | Observed HEAD | State |
| --- | --- | --- |
| isaac/vendor/agx_arm_sim | f8cd8b147c75d59e14f90fb0646770eefa268ed0 | No tracked modifications; untracked Python caches present |
| ros_ws/src/agx_arm_ros | b9ad14de2a6eb818a2d206fbd1adba48343662f7 | No tracked modifications |
| ros_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf | f6642ce0d7872c686f29c99e9e10cd23d1d49313 | Initialized; no tracked or untracked modifications observed |

These HEADs match third_party/manifest.lock.yaml. sha256sum also matched both locked license sources:

- isaac/vendor/agx_arm_sim/README.md: b91918f2972101eee621cba9badbc90e6057b9c5269c0bb7ae00f84195d31399
- ros_ws/src/agx_arm_ros/LICENSE: 247944c63de1573653722ba784f96123da267105b7b48577e0c8826a24b809f6

The sim README MIT declaration does not resolve USD/mesh licensing; manifest still explicitly marks assets_license_confirmed=false. No public redistribution or blanket vendor ignore is justified by this check.

## Important incomplete submodule

The sim parent tree records agx_arm_description/agx_arm_urdf at 1ee5659d02b33b9379fd647a2e5647800d67f0f4. Local git submodule status --recursive reports a leading minus, and the directory is empty: it is NOT initialized. Its commit differs from the ROS vendor URDF submodule. Do not substitute one for the other or claim both trees are fully restored.

Recovery must follow the existing third_party/README.md exact-commit manual process, initialize recursively, and verify the two distinct gitlinks. Do not create a floating-branch updater or mutate existing vendor trees as part of cleanup. Top-level Git has no vendor gitlinks, so cloning only this repository does not restore these dependencies.

## Files a clean checkout currently misses

The untracked list includes actual production and test dependencies, not only disposable output:

- src/rocobrick/clearance/*.py and piper_l_sim_hulls.json; the latter is consumed by runtime clearance logic.
- Four new test modules for clearance, Isaac camera colliders, upward feedback, and viewport capture.
- ros_ws/src/rocobrick_piper_l_moveit_config/urdf/piper_l.camera_stand.xacro and meshes/realsense_mid_stand.dae.
- third_party/README.md and manifest.lock.yaml, plus vendor checkout recovery.
- Probe/smoke tools directly loaded by existing tracked tests.
- Stable assembly plan fixtures added during cleanup.
- Historical JSON contract reports under artifacts/phase0, phase1, phase1-real, phase2-adapter, phase5. Tests refer to these directly; some are mandatory, others conditionally skip. Do NOT blanket ignore or delete phase directories. Their logs and reports need separate classification.

Examples of the last dependency: test_isaac_sim_runtime_contract.py and test_ros_runtime_contract.py use phase0 runtime reports; test_p1_dds_multiprocess_gate.py uses phase1/dds-multiprocess-gate.json; test_p1_real_joint_probe.py and test_p1_real_pose_probe.py use target reports; test_r2_live_adapter.py uses phase2-adapter/live-adapter.json; test_r5_tcp_moveit.py uses phase5/tcp-pose-planning-offline.json. Reproducing the current 2557-pass result requires accounting for these, not simply ignoring generated artifacts.

The old robot_assets/piper_l models are still referenced by configs and calibration. docker/planner.Dockerfile still builds planner_ws legacy packages; docker/ros.Dockerfile prepares the newer ROS runtime environment. Neither legacy models nor planner_ws should be removed until the active runtime references and image build paths are isolated and tested.

## Required next verification

1. Preserve and classify each effective untracked source/input before making a candidate baseline commit. Keep logs, source, immutable contract fixtures, and external dependencies separate.
2. Resolve whether frozen phase reports belong in tests/fixtures or in a reviewed provenance area; preserve bytes and update all references together.
3. Verify a candidate in an isolated new checkout, restoring exact vendor commits and required assets, without relying on ignored build/install state.
4. Compare focused and full offline results with the known baseline; do not hide the existing 27 failures with skips or relaxed checks.
5. Only then archive tracked historical evidence and perform a separately authorized fresh unique-directory runtime acceptance.
