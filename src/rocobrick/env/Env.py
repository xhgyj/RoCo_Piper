import omni.kit.app
from isaacsim.core.api.world import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim, SingleGeometryPrim
from isaacsim.core.utils.stage import open_stage_async, add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from pxr import Gf, Usd, UsdGeom, Sdf

from bricksim.assets import DEFAULT_STAGE_PATH
from bricksim.core import (
    arrange_parts_in_workspace,
    AssemblyThresholds,
    import_lego,
    set_assembly_thresholds,
)

from rocobrick.utils import *
from rocobrick.robot.Robot import *
from rocobrick.task_config.Task import *

class Env():
    def __init__(self, root_dir, user_config_path, system_config_path):
        self.root_dir = root_dir
        self.user_config_path = os.path.join(self.root_dir, user_config_path)
        self.system_config_path = os.path.join(self.root_dir, system_config_path)
        self.brick_unit_height = 0.0096 # Unit height of a standard lego brick, m.

    async def reset(self):
        """
        Resets the simulation environment by loading the configuration, setting up the simulation, robot, and task, and resetting the world.
        This should be called at the beginning of each episode.
        """
        # Load configuration json files
        self.config = self.load_config()
        self.cameras = {}

        # Parse robot configs: support "Robots" list (multi-arm) or legacy single-robot
        robot_cfg = self.config.get("Robot_Config", {})
        if "Robots" in robot_cfg:
            self.robot_configs = robot_cfg["Robots"]
        else:
            # Backward compat: wrap legacy single-robot config as a list
            self.robot_configs = [robot_cfg]

        # Setup simulation, robot, and task
        self.task_config = TaskConfig(self.config)
        self.app, self.world = await self.setup_bricksim(self.config)
        self.robots, self.robot_pins = await self.setup_robots(self.world)
        # Legacy aliases (first robot)
        self.robot = self.robots[0] if self.robots else None
        self.robot_pin = self.robot_pins[0] if self.robot_pins else None
        self.topology, self.pre_placed_parts, self.to_place_placed = self.setup_task(self.task_config)
        await self.world.reset_async()
        self.apply_pose_offset_world("/World/Cube", 0, 0.3)
        await self.step()

    def load_config(self):
        """
        Loads the configuration from JSON files.
        """
        user_config = load_json(self.user_config_path)
        system_config = load_json(self.system_config_path)
        config = deep_merge(user_config, system_config)
        return config
    
    def setup_task(self, task_config):
        """
        Loads the task.
        Setup the task environment by placing the pre-placed parts onto the plate and arranging the to-be-placed parts in the storage area. 
        
        Returns:
            topology: the full structure topology.
            pre_placed_parts: the parts that are already placed on the base plate.
            to_place_placed: the parts that need to be placed in the storage area.
        """
        # Load structure to assemble (including base plate)
        topology = deepcopy(task_config.topology)
        pre_placed_topology = deepcopy(task_config.pre_placed_topology)
        to_place_topology = deepcopy(task_config.to_placed_topology)
        baseplate_pose = task_config.baseplate_pose

        # Place pre-placed parts
        pre_placed_parts, pre_placed_conns = import_lego(
            json=pre_placed_topology,
            env_id=-1,
            ref_pos=baseplate_pose[0],
            ref_rot=baseplate_pose[1],
        )
        parts_not_placed = set(part['id'] for part in pre_placed_topology['parts']) - set(pre_placed_parts.keys())
        conns_not_placed = set(conn['id'] for conn in pre_placed_topology['connections']) - set(pre_placed_conns.keys())
        if len(parts_not_placed) > 0 or len(conns_not_placed) > 0:
            raise RuntimeError(f"Failed to place pre-placed parts/connections; not placed parts: {parts_not_placed}, not placed connections: {conns_not_placed}")

        # Spawn unplaced parts on table for assembly
        to_place_placed, _ = import_lego(
            json=to_place_topology,
            env_id=-1
        )
        to_place_not_placed = set(part['id'] for part in to_place_topology['parts']) - set(to_place_placed.keys())
        if len(to_place_not_placed) > 0:
            raise RuntimeError(f"Failed to place parts for assembly; not placed: {to_place_not_placed}")
        arranged, not_arranged = arrange_parts_in_workspace(
            workspace_path="/World/LegoWorkspace",
            parts_to_arrange=[path for id, path in to_place_placed.items()],
        )
        if len(not_arranged) > 0:
            raise RuntimeError(f"Failed to arrange all parts in workspace; not arranged: {not_arranged}")
        return topology, pre_placed_parts, to_place_placed

    async def setup_bricksim(self, config):
        """
        Sets up the BrickSim simulation environment.
        """
        # Initialize simulation
        app = omni.kit.app.get_app()
        if World._world_initialized:
            World.clear_instance()
        time.sleep(0.5)
        await open_stage_async(str(DEFAULT_STAGE_PATH))
        world: World = World(
            backend="numpy",
            device="cpu",
            physics_prim_path="/physicsScene"
        ) 
        await world.initialize_simulation_context_async()

        # Set simulation fps
        physics_context = world.get_physics_context()
        physics_context.set_physics_dt(1.0 / config["BrickSim_Physics"]["FPS"])

        # Set physics material for tabletop
        table_material = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials/Tabletop",
            static_friction=config["BrickSim_Physics"]["Table_Material"]["Static_Friction"],
            dynamic_friction=config["BrickSim_Physics"]["Table_Material"]["Dynamic_Friction"],
            restitution=config["BrickSim_Physics"]["Table_Material"]["Restitution"],
        )
        SingleGeometryPrim(prim_path="/World/scene/roomScene/colliders/table/tableTopActor").apply_physics_material(table_material)

        # Set assembly thresholds
        thresholds = AssemblyThresholds()
        thresholds.distance_tolerance = config["BrickSim_Physics"]["Assembly_Config"]["Distance_Tolerance"]
        thresholds.max_penetration = config["BrickSim_Physics"]["Assembly_Config"]["Max_Penetration"]
        thresholds.z_angle_tolerance = config["BrickSim_Physics"]["Assembly_Config"]["Z_Angle_Tolerance"] * (math.pi / 180.0)
        thresholds.required_force = config["BrickSim_Physics"]["Assembly_Config"]["Required_Force"]
        thresholds.yaw_tolerance = config["BrickSim_Physics"]["Assembly_Config"]["Yaw_Tolerance"] * (math.pi / 180.0)
        thresholds.position_tolerance = config["BrickSim_Physics"]["Assembly_Config"]["Position_Tolerance"]
        set_assembly_thresholds(thresholds)

        # Setup workspace
        storage_size = config["Env_Config"]["Storage_Config"]["Size"]
        storage_pos = config["Env_Config"]["Storage_Config"]["Position"]
        storage_ori = config["Env_Config"]["Storage_Config"]["Orientation"]
        workspace_prim = get_current_stage().GetPrimAtPath("/World/LegoWorkspace")
        workspace_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(storage_size[0], storage_size[1], storage_size[2]))
        workspace_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(storage_pos[0], storage_pos[1], storage_pos[2]))
        workspace_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(storage_ori[0], storage_ori[1], storage_ori[2], storage_ori[3]))
        # Register robot bases as workspace obstacles (so parts aren't placed under robots)
        for rc in self.robot_configs:
            base_path = f"{rc['Robot_Prim_Path']}/base_link"
            workspace_prim.GetRelationship("lego:workspace_obstacles").AddTarget(base_path)
        
        set_camera_view(
            eye=np.array([-0.40, 1.0, 0.60]),
            target=np.array([0.0, -0.20, 0.30]),
            camera_prim_path="/OmniverseKit_Persp",
        )
        return app, world

    async def setup_robots(self, world):
        """
        Sets up multiple robots in the simulation by spawning each from its config,
        creating articulations, and configuring joints, grippers, and cameras.
        """
        robots = []
        robot_pins = []
        stage = get_current_stage()

        # Per-arm dof_name → dof index maps (built lazily after world is ready)
        self._arm_dof_maps = None

        for i, rc in enumerate(self.robot_configs):
            name = rc.get("Name", f"robot_{i}")
            prim_path = rc["Robot_Prim_Path"]

            # --- Pinocchio model ---
            ee_frames = rc.get("EE_Frames", ("tip_l", "tip_r"))
            rp = Robot_Pin(
                os.path.join(self.root_dir, rc["Robot_URDF_Path"]),
                os.path.join(self.root_dir, rc["Robot_Package_Dir"]),
                ee_frames=ee_frames,
            )
            rp.home_q = np.array(rc["Joint_Home_Position"])

            # Base transform
            pos = rc["Robot_Base_Frame"]["Position"]
            ori = rc["Robot_Base_Frame"]["Orientation"]
            base_T = np.eye(4)
            base_T[:3, 3] = pos
            base_T[:3, :3] = R.from_quat(ori[1:] + [ori[0]]).as_matrix()
            rp.BASE_T = base_T

            # --- USD reference ---
            add_reference_to_stage(
                usd_path=os.path.join(self.root_dir, rc["Robot_USD_Path"]),
                prim_path=prim_path,
            )

            # --- Set base pose ---
            robot_xf = SingleXFormPrim(prim_path=prim_path, name=name)
            robot_xf.set_world_pose(position=pos, orientation=ori)

            # --- Articulation ---
            robot = SingleArticulation(prim_path=prim_path, name=name)
            world.scene.add(robot)

            # Solver iterations
            stage.GetPrimAtPath(prim_path).CreateAttribute(
                "physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int
            ).Set(64)

            # --- Joint physics ---
            if rc.get("Joints_Physics"):
                for joint_path, jp in rc["Joints_Physics"].items():
                    joint_prim = stage.GetPrimAtPath(joint_path)
                    if joint_prim.IsValid():
                        joint_prim.GetAttribute("drive:angular:physics:maxForce").Set(jp["Max_Force"])
                        joint_prim.GetAttribute("drive:angular:physics:damping").Set(jp["Damping"])
                        joint_prim.GetAttribute("drive:angular:physics:stiffness").Set(jp["Stiffness"])
            else:
                # Auto-configure drives for all joints by traversing the prim hierarchy
                from pxr import UsdPhysics
                root_prim = stage.GetPrimAtPath(prim_path)
                if root_prim.IsValid():
                    for prim in Usd.PrimRange(root_prim):
                        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                            try:
                                if prim.IsA(UsdPhysics.RevoluteJoint):
                                    prim.GetAttribute("drive:angular:physics:maxForce").Set(50.0)
                                    prim.GetAttribute("drive:angular:physics:damping").Set(5.0)
                                    prim.GetAttribute("drive:angular:physics:stiffness").Set(200.0)
                                    print(f"[setup] {name}: configured revolute drive for {prim.GetPath()}", flush=True)
                                else:
                                    prim.GetAttribute("drive:linear:physics:maxForce").Set(10.0)
                                    prim.GetAttribute("drive:linear:physics:damping").Set(2.0)
                                    prim.GetAttribute("drive:linear:physics:stiffness").Set(50.0)
                                    print(f"[setup] {name}: configured prismatic drive for {prim.GetPath()}", flush=True)
                            except Exception as e:
                                print(f"[setup] WARNING: failed to configure drive for {prim.GetPath()}: {e}", flush=True)

            # --- Gripper physics material ---
            gc = rc.get("Gripper_Config", {})
            if gc.get("Material"):
                mat = gc["Material"]
                pad_material = PhysicsMaterial(
                    prim_path=f"/World/PhysicsMaterials/FingerPad_{name}",
                    static_friction=mat["Static_Friction"],
                    dynamic_friction=mat["Dynamic_Friction"],
                    restitution=mat["Restitution"],
                )
                for link_path in mat.get("Link_Instance_Names", []):
                    try:
                        stage.GetPrimAtPath(link_path).SetInstanceable(False)
                        SingleGeometryPrim(prim_path=link_path).apply_physics_material(pad_material)
                    except Exception as e:
                        print(f"[setup] WARNING: Gripper material failed for {link_path}: {e}")

            # --- Cameras ---
            cc = rc.get("Camera_Config", {})
            for cam_key, cam_cfg in cc.items():
                cam_path = cam_cfg["Prim_Path"]
                cam_name_key = f"{name}_{cam_key}"
                prim = stage.GetPrimAtPath(cam_path) if stage else None
                if not prim or not prim.IsValid():
                    print(f"[setup] WARNING: {cam_name_key} not found at {cam_path!r}. Skipping.")
                    continue
                cam = Camera(
                    prim_path=cam_path,
                    name=cam_name_key,
                    resolution=(cam_cfg["Resolution"][0], cam_cfg["Resolution"][1]),
                    frequency=cam_cfg["FPS"],
                )
                cam.initialize()
                cam.add_distance_to_image_plane_to_frame()
                self.cameras[cam_name_key] = cam

            # --- Initial state ---
            robot.set_joint_positions(rp.home_q)
            robot.set_joint_velocities(np.zeros(rp.nq))

            robots.append(robot)
            robot_pins.append(rp)

        # --- Global joint bookkeeping ---
        self.global_joint_order = []
        self.arm_joint_slices = {}   # arm_name -> (start, length)
        self._arm_configs = self.robot_configs

        for i, rc in enumerate(self.robot_configs):
            name = rc.get("Name", f"robot_{i}")
            order = rc.get("Joint_Order", robot_pins[i].controllable_joints)
            start = len(self.global_joint_order)
            for jname in order:
                self.global_joint_order.append(f"{name}_{jname}")
            self.arm_joint_slices[name] = (start, len(order))

        await self.step()
        return robots, robot_pins

    def _ensure_dof_maps(self):
        """Build per-arm dof_name → dof index maps (deferred until world is ready)."""
        if self._arm_dof_maps is not None:
            return
        self._arm_dof_maps = []
        for i, robot in enumerate(self.robots):
            if robot.dof_names is None:
                raise RuntimeError(f"robot.dof_names is None — articulation not ready. "
                                   "Make sure the world has been reset/played.")
            dof_map = {dn: idx for idx, dn in enumerate(robot.dof_names)}
            name = self.robot_configs[i].get("Name", f"robot_{i}")
            print(f"[Env] arm={name} dof_names={robot.dof_names}", flush=True)
            print(f"[Env] arm={name} dof_map={dof_map}", flush=True)
            self._arm_dof_maps.append(dof_map)

    def robot_apply_action(self, q_cmd, input_joint_orders=None):
        """
        Applies the given joint position command to all robots. The input q_cmd
        is expected to be in the order of input_joint_orders (defaults to
        self.global_joint_order), and is dispatched to each arm's articulation.
        """
        self._ensure_dof_maps()
        if input_joint_orders is None:
            input_joint_orders = self.global_joint_order

        for i, robot in enumerate(self.robots):
            rc = self.robot_configs[i]
            name = rc.get("Name", f"robot_{i}")
            joint_order = rc.get("Joint_Order", self.robot_pins[i].controllable_joints)
            dof_map = self._arm_dof_maps[i]

            q_i = np.zeros(self.robot_pins[i].nq)
            for jname in joint_order:
                logical = f"{name}_{jname}"
                if logical in input_joint_orders and jname in dof_map:
                    q_i[dof_map[jname]] = q_cmd[input_joint_orders.index(logical)]

            robot.apply_action(ArticulationAction(joint_positions=q_i))
        
    async def step(self):
        """
        Advances the simulation by one step. 
        This should be called after applying actions to the robot to progress the simulation.
        """
        await self.app.next_update_async()

    async def play(self):
        """
        Starts the simulation. This should be called after reset() to begin the simulation loop.
        """
        await self.world.play_async()
        await self.step()

    async def pause(self):
        """
        Pauses the simulation.
        """
        await self.world.pause_async()

    async def get_robot_ready(self):
        """
        Moves all robots to their home positions. Each arm ramps from its
        current pose to home_q over ~240 steps.
        """
        # Ramp each arm toward home
        for _ in range(240):
            q_cmd = np.zeros(len(self.global_joint_order), dtype=np.float32)
            for i, (rp, rc) in enumerate(zip(self.robot_pins, self.robot_configs)):
                name = rc.get("Name", f"robot_{i}")
                start, length = self.arm_joint_slices[name]
                joint_order = rc.get("Joint_Order", rp.controllable_joints)
                home = rp.home_q[:len(joint_order)]
                q_cmd[start:start + length] = home
            self.robot_apply_action(q_cmd)
            await self.step()

    def apply_pose_offset_world(self, prim_path, x_offset, y_offset):
        """
        Applies a translation offset in the world frame to the specified prim. 
        The prim is identified by its path in the USD stage, and the offset is given by x_offset and y_offset in meters.
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = world_tf.ExtractTranslation()
        
        t[0] = t[0] + x_offset
        t[1] = t[1] + y_offset
        parent = prim.GetParent()
        if parent:
            parent_xform = UsdGeom.Xformable(parent)
            parent_world = parent_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            parent_world_inv = parent_world.GetInverse()
            local_t = parent_world_inv.Transform(t)
        else:
            local_t = t

        # --- reuse existing translate op if present ---
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if translate_op is None:
            translate_op = xformable.AddTranslateOp()

        translate_op.Set(Gf.Vec3d(float(local_t[0]), float(local_t[1]), float(local_t[2])))

    def get_prim_world_T(self, prim_path):
        """
        Retrieves the world transform of the specified prim as a 4x4 homogeneous transformation matrix. 
        The prim is identified by its path in the USD stage.
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        local_to_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        r = local_to_world.ExtractRotationMatrix()
        t = local_to_world.ExtractTranslation()
        T = np.eye(4)
        T[:3, :3] = np.array(r).transpose()
        T[:3, 3] = np.array(t)
        return T

    def get_prim_robot_T(self, prim_path):
        """
        Retrieves the transform of the specified prim in the first robot's base frame.
        (Legacy — use get_prim_arm_T for multi-arm.)
        """
        return self.get_prim_arm_T(prim_path, arm_idx=0)

    def get_prim_arm_T(self, prim_path, arm_idx=0):
        """
        Retrieves the transform of the specified prim in the specified arm's base frame.
        arm_idx: index into self.robot_pins
        """
        T = self.get_prim_world_T(prim_path)
        T = world_T_to_robot_T(T, self.robot_pins[arm_idx].BASE_T)
        return T
    
    def get_observations(self):
        """
        Retrieves the current observations from the environment, including
        all robots' joint positions (concatenated in global_joint_order) and
        per-arm camera images.
        """
        self._ensure_dof_maps()
        obs = {}

        # Joint positions: concatenate all arms in global_joint_order
        q_all = np.zeros(len(self.global_joint_order), dtype=np.float32)
        for i, robot in enumerate(self.robots):
            rc = self.robot_configs[i]
            name = rc.get("Name", f"robot_{i}")
            start, length = self.arm_joint_slices[name]
            joint_order = rc.get("Joint_Order", self.robot_pins[i].controllable_joints)
            dof_map = self._arm_dof_maps[i]
            q_raw = robot.get_joint_positions()
            for k, jname in enumerate(joint_order):
                if jname in dof_map:
                    q_all[start + k] = q_raw[dof_map[jname]]
        obs["joint_positions"] = q_all

        # Images: per-arm keys
        obs["images"] = {}
        for cam_key, cam in self.cameras.items():
            obs["images"][f"{cam_key}_rgb"] = cam.get_rgb()
            obs["images"][f"{cam_key}_depth"] = cam.get_depth()

        return obs