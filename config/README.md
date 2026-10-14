# RoCo-BrickAssembly Configuration Guide

### 1. `lego_library.json`
Defines the physical dimensions of the available bricks. Maps a unique `brick_id` to its corresponding `width` and `height` properties (e.g., a `brick_id` of `"2"` defines a 2x4 brick).

### 2. `system_config.json`
⚠️ **Do not modify this file.**
This establishes the default configuration for the BrickSim environment.
### 3. `user_config.json`
✅ **Modify this configuration as you see fit.**
This file contains the customizable variables for your specific assembly task, robotic hardware, and spatial environment. 

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

#### Shared Elevated Feeding Platforms

The Piper runtime optionally accepts `Env_Config.Feeding_Platforms`. When
present, it initializes every loose goal brick on one of the shared platforms,
independent of its successor skill. Existing assembled structure parts stay at
their original poses. Omitting the field retains the previous ground allocator.

Each entry has exactly these fields, in assembly-world coordinates and metres:

```json
{
  "Robot_ID": "piper_0",
  "Center_XY": [0.35, 0.06],
  "Size_XY": [0.24, 0.24],
  "Top_Z": 0.08,
  "Thickness": 0.08
}
```

Configure one non-overlapping platform per session robot. These are whole static
collision boxes, not individual brick pillars. Both platform surfaces in the
scaled2 Task C configuration are at 80 mm, outside the baseplate and farther from
the robot bases than the old loose-brick slots. Their measured collision bounds
are included in the host's environment snapshot and the unified MoveIt scene.

The allocator balances loose bricks between regions and uses the first task's
robot preference as a tie-breaker. It does not modify robot operation allocation,
copy bricks, move assembled parts, or change PICK logic. A partial plan with only
one loose brick creates both platforms but initializes that brick only once.
Slot placement accounts for brick footprint and quarter-turn orientation, leaves
10 mm platform-edge clearance and at least 120 mm between brick centres, and
rejects insufficient capacity rather than placing bricks back on the ground.

A broad elevated surface supports ordinary overhead picking. Raising the whole
surface does not by itself permit horizontal fingers to pass below that surface;
side-grasp feasibility must still be established separately before changing PICK.

#### Vertical Offset PICK Trial

Set `ROCOBRICK_TOP_OFFSET_PICK=1` to try downward picking on the exposed end of
bricks whose external PICK operation declares `successor_skill: PLACE_UP`.
Without the switch, the existing grasp-region path remains unchanged.

The trial reads the captured goal structure and the explicit PLACE_UP successor
to identify a free rectangular strip. For scaled2 Task C's original 4x4 brick,
only half overlaps the upper brick and the contact centre is offset 8 mm toward
the free end in brick-local coordinates. The brick size, source JSON and
operation allocation are not rewritten.

The PICK lane submits exactly one Cartesian TCP goal: the offset contact pose
with the tool pointing down and the closing axis horizontal. The host performs
no inverse kinematics, adds no pregrasp waypoint, path corridor or trajectory
retiming; the combined dual-arm MoveGroup request carries the pose goal plus the
full collision scene, and MoveIt selects the IK solution and the whole approach
path. Because the two jaw orientations about the tool axis are physically
equivalent and a pinned wrist roll can be kinematically unreachable (live
evidence: no IK branch at the Task C feeding platform for one of the two
families), the tilt tolerances stay tight while the roll about the tool axis is
left free; MoveIt picks whichever symmetric wrist branch is reachable. The peer
arm is held at its measured joints over every waypoint. Measured arrival is
gated on the brick not moving, the TCP within 4 mm of the offset contact, the
tool pointing down and the closing axis aligned with the brick side faces — not
on a pinned quaternion. The phase order is scene synchronization, open,
offset-pose approach, measured arrival, finger-contact permission, close,
measured attachment and lift/follow verification. Full collision geometry,
including the feeding platforms, stays enabled until measured arrival
authorizes the two finger contacts.

Recorded `successor_grasp` evidence includes the free strip, overlap fraction,
planned contact and `place_up_compatibility_confirmed: false`. Successful PICK
alone does not certify upward assembly or clearance for the whole gripper. The
known discrepancy between the source JSON's half overlap and the previous live
interface transform has not been repaired by this trial; actual PLACE_UP remains
subject to its existing planning and physical evidence gates.
