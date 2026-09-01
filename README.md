# RoCo-BrickAssembly

Welcome to the official simulation repository for the **Brick Assembly** track of the [RoCo Challenge @ IROS26](https://rocochallenge.github.io/RoCo-IROS2026/). 

---

## 💻 Prerequisites

Before you begin, please make sure you have confirmed your participation in this track with the organizers.

Ensure your system meets the following requirements:
* **OS:** Ubuntu 22.04 or higher
* **Hardware:** NVIDIA GPU

## 🚀 Getting Started

Follow these steps to install dependencies and run the example demo:

```bash
# 1. Install uv (Python package installer and resolver)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# 2. Setup Environment
uv sync --locked

# 3. Execute Example Demo
uv run bricksim ./run/demo.py
```

## 📂 Repository Structure

```text
RoCo-BrickAssembly/
├── README.md                      # This file: repository instructions
├── uv.lock                        # venv configuration
├── config/                        # Configuration files
│   ├── README.md                  # Configuration descriptions
│   ├── lego_library.json          # Definitions of `brick_id`
│   ├── system_config.json         # Configuration of BrickSim
│   └── user_config.json           # Your configurations (TODOs)
├── robot_assets/                  # Robot models (URDF, USD, visual/collision meshes)
├── tasks/                         # Brick assembly tasks
│   ├── README.md                  # Task descriptions
│   ├── type1/                     # Type-1 tasks
│   └── type2/                     # Type-2 tasks
├── src/                       
│   └── rocobrick/                 # RoCo-BrickAssembly source code
│       ├── env/                   # BrickSim environment
│       ├── policy/
│       │   ├── gt_assembly.py     # GT pickup, local assembly, and cleanup
│       │   ├── assembly_control.py# Cartesian phases, feedback, and safety
│       │   └── assembly_runtime.py# Isaac/Piper runtime adapter
│       ├── robot/                 # Pinocchio robot model (FK, IK, etc.)
│       ├── task_config/           # Task loader and symbolic generator
│       └── utils.py               # Helper functions
└── run/
    ├── demo_gt_assembly.py        # Run a saved task from the safe pose
    ├── demo_symbolic_assembly.py  # Inspect a structure, then run the expert
    └── main.py                    # Current GT expert evaluation entry point
```

---

## 🛠️ Build Your Policy

The current repository intentionally contains only the privileged GT expert
needed to validate and later collect the local alignment/insertion phase. The
learned ACT/Diffusion Policy implementation will be added only after GT
grounding, goal-mask validation, and demonstration collection are ready.
`run/main.py` currently evaluates this expert and is not a competition-ready
vision-only inference policy.

### GT single-step assembly expert

The sole scripted expert uses simulator GT to resolve one target-centric
assembly step. An unrecorded preparation phase physically picks the loose
target outside the plate, lifts it with a Cartesian path, and transports it to
a verified pose with its center 60 mm directly above the goal. Preparation
preserves the pickup yaw, so the separately bounded expert
trajectory contains real alignment as well as approach, contact, press, and hold.
Alignment is completed at clearance height before descent. Persistent alignment
loss after contact triggers a full unload-and-realign cycle instead of repeated
sub-millimeter contact retries.
High-clearance alignment uses a separate 4-degree rotation step and 6-degree
tracking lead. The controller restores the conservative 2-degree limit within
20 mm of the assembly surface. A 1x2 target uses a lower 2 mm grasp and a
1-degree high-clearance yaw step to prevent physical tipping. Candidate grasps
are rejected before pickup unless the complete high-clearance Cartesian yaw
path has a verified continuous IK branch.
After BrickSim verifies every requested connection, an unrecorded cleanup phase
opens the gripper, retreats vertically, and returns the selected arm home.

The preparation phase never teleports the target. It measures the actual
brick-to-TCP transform after the initial lift and checks every remaining loaded
segment against that single baseline so cumulative slip cannot be hidden. Grasp
planning evaluates both local x/y axes and rotates the gripper 90 degrees when
neighboring bricks block the default axis. The jaw opening is tailored to the
selected brick width rather than always using full travel. High-aspect-ratio
bricks prefer a wider long-axis grasp for yaw stability, while compact bricks
prefer the shorter axis. Pickup uses the lower sidewall grasp plane and checks
directional slip throughout transport. Free-space execution raises to a safe
carry height, translates in the pickup orientation, then descends without yaw
alignment. It uses a ramped Cartesian command lead and runs faster than
the contact-sensitive local assembly trajectory. Every Env-based command requests an uncancellable Isaac Kit
shutdown from `finally`, on both success and error, so completed demos do not
leave simulator processes running:

```bash
uv run bricksim ./run/demo.py
```

Symbolic one-step tasks can be generated without launching Isaac Sim:

```bash
uv run python ./run/generate_symbolic_tasks.py \
  --output tasks/type1 \
  --seed 7 \
  --count-per-family 10
```

Use the visual demo to inspect a preplaced structure and run the same complete
workflow:

```bash
uv run bricksim ./run/demo_symbolic_assembly.py --family basic --seed 7
uv run bricksim ./run/demo_symbolic_assembly.py --family bridge --seed 7

# Inspect dense/1 and save the global-camera assembly video in this directory
uv run bricksim ./run/demo_symbolic_assembly.py \
  --task-dir tasks/type1/dense/1 \
  --initial-yaw-deg -135 \
  --inspect-seconds 5 \
  --save-video \
  --video-view assembly-close \
  --video-output ./symbolic_dense1.mp4
```

Run a complete planner-driven episode through the single BrickSim entry point:

```bash
uv run bricksim ./run/main.py \
  --episode config/episodes/task_d/episode.json \
  --output /tmp/task_d_result.json
```

The episode loader builds a topology problem, calls the reference Python task
planner, expands its immutable arm assignments into manipulation actions, and
runs them with per-arm and shared-assembly resource synchronization. Task D
retains the calibrated dual-Piper bases, pickup points, and parking slots in
its episode configuration.

Generated tasks are grouped as `tasks/type1/<family>/<sequence>/`, and every
sequence directory contains only `structure_start.json` and
`structure_goal.json`.

The demo shows a translucent green goal preview during its initial inspection
pause, removes it before execution, and holds the verified final structure for
review. Pass `--task-dir tasks/type1/basic/1` to inspect a saved task.

Set an exact loose-target yaw, or sample one reproducibly outside the plate:

```bash
uv run bricksim ./run/demo_symbolic_assembly.py \
  --task-dir tasks/type1/dense/1 \
  --initial-yaw-deg -135

uv run bricksim ./run/demo_symbolic_assembly.py \
  --task-dir tasks/type1/dense/1 \
  --random-initial-yaw \
  --yaw-seed 7
```

These options preserve the sampled pickup yaw through preparation. Geometric
symmetry equivalence is intentionally not applied yet: the expert still tracks
the exact BrickSim connection yaw.

To validate only pickup and transport to the pose directly above the goal,
without running local alignment/insertion, use:

```bash
uv run bricksim ./run/demo_symbolic_assembly.py \
  --task-dir tasks/type1/multilevel/7 \
  --prepare-only \
  --final-hold-seconds 10
```

Validate every saved Type-1 task in isolated headless Isaac processes:

```bash
uv run python ./run/validate_expert_tasks.py --tasks-root tasks/type1
```

For deterministic continuous-yaw coverage and a visible HTML report:

```bash
uv run python ./run/validate_expert_tasks.py \
  --tasks-root tasks/type1 \
  --yaw-samples 3 \
  --yaw-seed 7 \
  --output validation_reports/yaw_seed7

xdg-open validation_reports/yaw_seed7/report.html
```

The command displays terminal progress and continuously updates an ignored
`validation_reports/<timestamp>/report.html` report containing PASS/FAIL,
duration, failure reasons, and links to each complete simulator log. Restrict a
run with repeated `--family` options or use `--limit` for a smoke test. Reuse
the same `--output` with `--resume` to skip earlier PASS results whose detailed
logs are still present; failed, interrupted, and log-less tasks are rerun.

Run one saved task directly without the generated preview stage. The printed
`ExpertResult.steps` counts only the local assembly trajectory; preparation and
cleanup are intentionally excluded:

```bash
uv run bricksim ./run/demo_gt_assembly.py \
  --task-dir tasks/type1/basic/1 \
  --safe-height-mm 60
```

### ⚠️ Rules & Restrictions

> **What You Can Do:**
> * **Modify specific files:** You are only allowed to modify files explicitly labeled with **(TODOs)** or add new files that support your TODO implementations. If you believe modifying other core files is necessary, you must confirm with the committee first.
> * **Create custom tests:** Feel free to design additional assembly structures to thoroughly test and evaluate your policy.
> * **Collect data freely:** You may use any method for data collection and training, including the use of privileged information, teleoperation, and synthetic generation. 
> * **Be creative:** We do not restrict the underlying method, architecture, or algorithm you choose. Build the best brick builder possible!
> 
> **What You Cannot Do:**
> * **Do NOT use privileged information during inference:** During inference/runtime, your policy may *only* rely on realistic observations (e.g., camera feeds and robot proprioceptive feedback). The GT assembly expert in this repository is a teacher/validation tool; it is not the final learned inference policy.
> * **Do NOT use human intervention:** Fully autonomous execution is required. No human intervention or teleoperation is allowed during inference runtime.

---

## 🏆 Scoring & Leaderboard

* **Scoring Metric:** Evaluation follows the official metric outlined in the [Task Description](./tasks/README.md). Note that your score will be evaluated against both the released tasks **and** an unreleased, hidden set of tasks. The tasks used during the onsite competition will be drawn from this complete set.
* **Leaderboard:** We maintain an active [Leaderboard](https://rocochallenge.github.io/RoCo-IROS2026/brick_assembly_overview.html) on our website. Rankings are always based on your **latest submission**.

## 📤 Submission Instructions

When you are ready to submit, please compile the following files into a single zip file named `TeamName.zip`:
1.  `./config/user_config.json`
2.  `./src/rocobrick/policy/Policy.py`
3.  Additional files if needed.
4.  A `README` detailing the instructions to run your policy, e.g., where to put the files, additional commands needed to run.

**Submit your file here:** [Official Submission Portal](https://forms.gle/AaxqVhRHTegXYSab7)

> **Note:** Each team is allowed **one submission per week**. Please test thoroughly before submitting!

---

## 🧱 Tasks
You can download the tasks at this link: [brick assembly tasks download](https://drive.google.com/file/d/1GREvEKAZtgpM3r-xb9umneN6-EzCg3IU/view?usp=sharing). Modify the task path in `./config/user_config.json` to assemble different structures.

---

## 🤖 Hardware

The Brick Assembly track will use the **[DexMate Vega U](https://www.dexmate.ai/product/vega-u)** as the official robot platform. Simulation assets are provided in `robot_assets/DexMate` to support your team's development and preparation.

We are excited to give participants hands-on access to state-of-the-art robotic embodiments through meaningful and challenging assembly tasks, advancing the future of embodied AI!

---

## 💬 Contact & Support

If you have any questions, run into issues, or just want to chat with fellow competitors, feel free to drop into our [Discord Channel](https://discord.gg/BvxEN5vAh3) or reach out to the organizers:

* **RoCo Committee:** [Challenge Website](https://rocochallenge.github.io/RoCo-IROS2026/)
* **Ruixuan Liu:** ruixuanl@andrew.cmu.edu
* **Haowei Wen:** haoweiw@andrew.cmu.edu
