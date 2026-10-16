# RoCo_Piper — formal operation-plan runtime

This fork runs the Piper_L multi-robot Isaac/ROS/MoveIt operation-plan platform.
The upstream challenge documentation below is retained as historical project
context, not as the current deployment or evaluation interface.

## Current entry and cleanup boundary

- Use `uv sync --locked` for the existing dependency set and `uv run --no-sync piper-run --help` for explicit immutable input flags.
- `piper-run --robot-operation-plan PATH` is the only production dispatch entry; provide the corresponding structure/STAP, config and authorization inputs. Do not synthesize or reorder the external plan.
- The live topology is one Isaac host, one ROS container session, the combined `dual_arm_move_group.launch.py`, per-robot namespaced channels, and the persistent ROS worker.
- Keep all formal skill branches and cancel/terminal/cleanup gates, not just Task C. Static tests and clean builds are not live execution evidence.
- The old gRPC planner, pregrasp/playback, phase backend, single-session and isolated/fake launch paths have been retired from this candidate. Import classified package owners directly rather than old compatibility forwarding modules.
- Historical runs are outputs, not deployed startup inputs. Offline fixtures carry exact historical plan bytes; TCP provenance byte objects are test-only audit inputs. The production DDS prerequisite report remains required.
- Original uncommitted functional changes and immutable model/calibration/plan bytes must be preserved. See `AGENTS.md` for safety and test rules.
- The old standalone validate/demo entries are retired. Non-production maintenance and diagnostic tools are preserved under maintenance/archived_tools/ and are not supported deployment commands.

## Historical upstream challenge README (not the current launch contract)

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
# Historical demo command retired; use piper-run for the supported runtime.
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
│       │   ├── NaivePolicy.py     # Example rule-based policy (uses privileged info)
│       │   └── Policy.py          # Your robot policy implementation (TODOs)
│       ├── robot/                 # Pinocchio robot model (FK, IK, etc.)
│       ├── task_config/           # Task loader and parser
│       └── utils.py               # Helper functions
└── run/
    ├── demo.py                    # Example script using NaivePolicy 
    └── main.py                    # Main script to evaluate your policy
```

---

## 🛠️ Build Your Policy

Your objective is to build an intelligent robot policy capable of successfully constructing as many assembly structures as possible. 

* **Implementation:** Your custom policy code must be written in `RoCo-BrickAssembly/src/rocobrick/policy/Policy.py`. 
  * **Input**: Observations, including camera and robot feedback.
  * **Output**: robot joint positions, a 23-dimensional array. *Format*: [Lift, torso_flip, L_arm_j1 to L_arm_j7, L_gripper_joint, Unused, R_arm_j1 to R_arm_j7, R_gripper_joint, Unused, head_j1, head_j2, head_j3].
* **Evaluation:** The historical standalone policy entry was retired; production uses the operation-plan interface above.

### Example Policy
We have provided a naive example policy in `src/rocobrick/policy/NaivePolicy.py` to demonstrate how to interact with BrickSim. You can test it by running the `demo.py` script.

### ⚠️ Rules & Restrictions

> **What You Can Do:**
> * **Modify specific files:** You are only allowed to modify files explicitly labeled with **(TODOs)** or add new files that support your TODO implementations. If you believe modifying other core files is necessary, you must confirm with the committee first.
> * **Create custom tests:** Feel free to design additional assembly structures to thoroughly test and evaluate your policy.
> * **Collect data freely:** You may use any method for data collection and training, including the use of privileged information, teleoperation, and synthetic generation. 
> * **Be creative:** We do not restrict the underlying method, architecture, or algorithm you choose. Build the best brick builder possible!
> 
> **What You Cannot Do:**
> * **Do NOT use privileged information during inference:** During inference/runtime, your policy may *only* rely on realistic observations (e.g., camera feeds and robot proprioceptive feedback). The `NaivePolicy` demo uses privileged information (like exact ground-truth brick states) for illustration purposes, but relying on this for your final evaluation is **strictly forbidden**.
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