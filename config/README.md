# RoCo-BrickAssembly Configuration Guide

### 1. `lego_library.json`
Defines the physical dimensions of the available bricks. Maps a unique `brick_id` to its corresponding `width` and `height` properties (e.g., a `brick_id` of `"2"` defines a 2x4 brick).

### 2. `system_config.json`
⚠️ **Do not modify this file.**
This establishes the default configuration for the BrickSim environment.
### 3. `user_config.json`
✅ **Modify this configuration as you see fit.**
This file contains the customizable variables for your specific assembly task, robotic hardware, and spatial environment. 

### 4. `episodes/<name>/episode.json`
Defines one planner-driven simulation run. An episode references the shared
scene configurations and task structures, declares the robot arms available to
the upper planner, and supplies static simulator-only settings such as robot
base overrides and staging locations. The planner receives the resulting
topology and returns the assembly DAG in memory; it does not need to know
about pickup points, safety resources, or BrickSim control actions.

For a planned episode, `staging.arms.<arm>.parking_slots` are the loose
bricks' initial pickup locations. During scene construction the runtime maps
the planner's arm assignments to these slots; the corresponding arm later
picks each brick directly from its original slot, without a second restaging
teleport.

`execution.allow_prefetch` controls whether the next task may pick while the
previous task is still placing. Keep it `false` for layouts such as Task D,
where a waiting arm or held brick can obstruct the shared assembly approach.
Enable it only for episodes with independently validated safe pickup and
holding regions.

Run an episode through the single entry point:

```bash
uv run bricksim ./run/main.py --episode config/episodes/task_d/episode.json
```

#### Task Configuration (`Task_Config`)
Defines the objectives and foundational setup of the current assembly task.
* **`Task_Path`**: File path pointing to the specific task folder.
* **`Task_Type`**: Classifies the task as Type-1 or Type-2. (See [Task Descriptions](../tasks/README.md) for detailed definitions).
* **`Base_Plate`**: Properties of the baseplate, including its `Dimension` (e.g., [32, 32]), `Position`, `Orientation`, and `Color`.

#### Robot Configuration (`Robot_Config`)
Defines the robotic asset, its initial state, kinematic properties, and sensory payload.
* **Asset Paths**:
    * `Robot_Package_Dir`: Directory containing the robot's models and meshes.
    * `Robot_URDF_Path`: Path to the robot's URDF file.
    * `Robot_USD_Path`: Path to the robot's USD file.
* **Positioning & State**:
    * `Robot_Base_Frame`: The spawn `Position` and `Orientation` for the robot's base.
    * `Joint_Home_Position`: A 23-dimensional array establishing the robot's initial resting pose. 
        * *Array Format:* `[Lift, torso_flip, L_arm_j1 to L_arm_j7, L_gripper_joint, Unused, R_arm_j1 to R_arm_j7, R_gripper_joint, Unused, head_j1, head_j2, head_j3]`
* **Physics & Hardware**:
    * `Joints_Physics`: Allows localized overrides for joint dynamics.
        * *Format:* `"joint_name": {"Max_Force": float, "Damping": float, "Stiffness": float}`
    * `Camera_Config`: Defines the vision sensors attached to the robot. (Note: These cameras must be pre-allocated in the robot's USD file).
        * *Format:* `"camera_name": {"FPS": int, "Resolution": [width, height], "Prim_Path": "prim_name_in_usd"}`
    * `Gripper_Config`: Configures the physical interaction properties of the end-effectors. Specifies a custom `Material` (`Static_Friction`, `Dynamic_Friction`, `Restitution`) and an array of `Link_Instance_Names` to dictate which structural links inherit these properties.

#### Environment Configuration (`Env_Config`)
Defines the surrounding spatial setup and external objects.
* **`Storage_Config`**: Configures the staging area where unassembled bricks are spawned before manipulation. It is defined as a bounding cuboid with specific `Size`, `Position` (centroid), and `Orientation` variables.
