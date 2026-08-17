# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
uv sync --locked         # install dependencies (Isaac Sim 5.1, BrickSim, Pinocchio, etc.)
uv run bricksim ./run/demo.py   # run example policy (single-task demo)
uv run bricksim ./run/main.py   # evaluate custom policy (competition entry point)
uv run pytest            # run tests
uv run ruff check        # lint (src/, tests/)
```

## Architecture

This is the **RoCo-BrickAssembly** simulation environment for the IROS26 RoCo Challenge. It runs on **Isaac Sim 5.1** with BrickSim for LEGO-style brick assembly tasks.

### Key Fork: DexMate → Piper L Dual-Arm

The competition's official robot is **DexMate Vega U** (23D action space). This fork **replaces it with dual Piper L arms** from agx_arm_sim:

| Aspect | Original (DexMate) | This Fork (Piper L) |
|--------|-------------------|---------------------|
| Arms | 1 robot, 2 EE hands (tip_l/tip_r) | 2 independent arms (piper_0, piper_1) |
| Action space | 23D flat | 16D global concat (2×8D: j1..j6 + gripper_j1, gripper_j2) |
| EE frame | tip_l / tip_r | link6 (no tool0 frame in Piper URDF) |
| IK | Jacobian pseudoinverse | Jacobian + scipy L-BFGS-B fallback (`method="auto"`) |
| Gripper control | Hardcoded q[9]/q[18] indices | Pinocchio `getJointId` by name |

Backward compatibility aliases exist in Env: `self.robot_pin` → `self.robot_pins[0]`, `self.robot` → `self.robots[0]`.

### Action / Observation Flow

```
Policy.get_action(obs) → 16D global q_cmd → Env.robot_apply_action(q_cmd)
                                                    ↓
                            Per-arm dispatch via arm_joint_slices[name] → (start, length)
                            Maps config Joint_Order names → USD articulation dof indices
                                                    ↓
                            Isaac Sim PD controller (stiffness=200, damping=5, maxForce=50)
```

The **global joint order** is built by concatenating config `Joint_Order` entries with arm name prefixes:
```
["piper_0_joint1", ..., "piper_0_gripper_joint2", "piper_1_joint1", ..., "piper_1_gripper_joint2"]
```

Observations (`get_observations`) return the same 16D structure plus per-arm camera images keyed as `{arm_name}_{cam_key}_rgb/depth`.

### Key Modules

- **`Env.py`** — `setup_robots()` loads N arms from `Robot_Config.Robots[]`. Auto-configures USD joint drives when `Joints_Physics` is empty. Maintains `global_joint_order`, `arm_joint_slices`, dof maps.
- **`Robot.py`** — `Robot_Pin` wraps Pinocchio model. IK has three modes: `"jacobian"` (fast gradient), `"optimize"` (scipy L-BFGS-B with random restarts, position-only cost), `"auto"` (jacobian first, falls back to optimize comparing position error). `lock_joints()` removes joints from `controllable_joints` list.
- **`NaivePolicy.py`** — Original single-arm state machine (home→pregrasp→grasp→close_gripper→...→home). Uses `self.cur_q` (commanded position) for `np.array_equal` comparison — immune to PD steady-state error. `_to_global_action()` wraps single-arm q into 16D global vector.
- **`Policy.py`** — User-implemented policy stub (competition entry point in `run/main.py`).

### Config (`user_config.json`)

- `Robot_Config.Robots[]` — list of arm configs. Each has: Name, URDF/USD paths, prim path, EE_Frames, Joint_Order, Joint_Home_Position (8D), Robot_Base_Frame (Position + Orientation quaternion [w,x,y,z]), Joints_Physics, Camera_Config, Gripper_Config.
- `Robot_Base_Frame.Position[2]` (z) set to 0.35 to prevent arm links from passing through ground plane when reaching bricks at z≈0.
- `Joints_Physics: {}` triggers auto-drive configuration in Env (stiffness=200, damping=5, maxForce=50 for revolute).
- Backward compat: if `"Robots"` key is absent, legacy single-robot config is auto-wrapped as a list.

### PD Controller Behavior

Joints are driven by Isaac Sim's `UsdPhysics.RevoluteJoint` / `UsdPhysics.PrismaticJoint` with:
- Revolute: stiffness=200, damping=5, maxForce=50
- Prismatic (gripper): stiffness=50, damping=2, maxForce=10

Steady-state error ≈ τ_gravity / stiffness. No gravity compensation or feedforward torque exists. Higher stiffness causes underdamped oscillation → physics instability (exit 137/139).

### Current State

- **Working**: dual-arm loading, joint drive auto-config, IK convergence (scipy fallback), DOF name mapping
- **Known issue**: state machine races through all states instantly because `np.array_equal(goal_q, self.cur_q)` matches every step when IK returns solutions close to the seed. The arms move (PD tracking works) but may collide with environment due to incorrect IK targets or reachability issues with the elevated base (z=0.35).
- **Piper L kinematics**: 6 revolute arm joints + 2 prismatic gripper = 8 DOF. Shoulder (joint1) at z=+0.123 above base_link. Total reach ~0.75m. Arm base at world z=0.35, bricks at z≈0, vertical offset ~0.46m (within reachable workspace).
