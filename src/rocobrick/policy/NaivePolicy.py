from bricksim.topology.ordering import bfs_sort_connections
from rocobrick.utils import *

class Policy():
    def __init__(self, env):
        self.task_id = 0
        self.state_dicts = {}
        self.state = "home"
        self.env = env
        self.plan = self.plan_assembly_sequence(env.topology, env.pre_placed_parts, env.to_place_placed)
        self.cur_q = env.robot_pin.home_q.copy()

        # Lock gripper joints when computing IK (Piper: gripper_joint1, gripper_joint2)
        gripper_joints = [j for j in env.robot_pin.controllable_joints if "gripper" in j]
        env.robot_pin.lock_joints(gripper_joints)
        print("Controllable joints:", self.env.robot_pin.controllable_joints)
        self.plan_goal_sequence(self.plan, self.task_id)

    def get_action(self, obs, max_step=0.005):
        """
        A naive open-loop policy that executes the planned waypoints with simple interpolation.
        Returns a global flat joint vector (all arms concatenated).
        """
        joint_positions = obs["joint_positions"]
        images = obs["images"]

        if self.task_id >= len(self.plan):
            self.cur_q = self.env.robot_pin.home_q.copy()
            return self._to_global_action(self.cur_q)

        state_dict = self.state_dicts[self.task_id]

        goal_q = state_dict[self.state]
        ee_name = state_dict["ee_name"]

        # A naive open-loop state machine policy.
        if(np.array_equal(goal_q, self.cur_q)):
            if(self.state == "home"):
                self.state = "pregrasp"
            elif(self.state == "pregrasp"):
                self.state = "grasp"
            elif(self.state == "grasp"):
                self.state = "close_gripper"
            elif(self.state == "close_gripper"):
                self.state = "retreat_grasp"
            elif(self.state == "retreat_grasp"):
                self.state = "preplace"
            elif(self.state == "preplace"):
                self.state = "place"
            elif(self.state == "place"):
                self.state = "open_gripper"
            elif(self.state == "open_gripper"):
                self.state = "retreat_place"
            elif(self.state == "retreat_place"):
                self.state = "back_home"
            elif(self.state == "back_home"):
                self.state = "home"
                self.task_id += 1
                if self.task_id < len(self.plan):
                    self.plan_goal_sequence(self.plan, self.task_id)
            print("Now executing:", self.state)
        else:
            # Interpolate towards the goal with a maximum step size
            if(np.max(abs(goal_q - self.cur_q)) < max_step):
                step = goal_q - self.cur_q
            else:
                step = (goal_q - self.cur_q) / np.max(abs(goal_q - self.cur_q)) * max_step
            self.cur_q = self.cur_q + step

        return self._to_global_action(self.cur_q)

    def _to_global_action(self, q):
        """Wrap single-arm joint positions into the global multi-arm action vector."""
        q_global = np.zeros(len(self.env.global_joint_order), dtype=np.float32)
        for i, rc in enumerate(self.env.robot_configs):
            name = rc.get("Name", f"robot_{i}")
            start, length = self.env.arm_joint_slices[name]
            if i == 0:
                q_global[start:start + length] = q[:length]
            else:
                # Hold inactive arms at their home position
                q_global[start:start + length] = self.env.robot_pins[i].home_q[:length]
        return q_global

    def plan_goal_sequence(self, plan, task_id):
        """
        For each sub-task in the assembly plan, we compute the key waypoints (pre-grasp, grasp,
        pre-place, place, etc.) using inverse kinematics and store them in a state dictionary.
        The policy will then execute these waypoints in sequence for each sub-task.
        """
        task = plan[task_id]
        to_brick = task['stud_path']
        grab_brick = task['hole_path']
        to_brick_T = self.env.get_prim_robot_T(to_brick)
        grab_brick_T = self.env.get_prim_robot_T(grab_brick)

        # Use the configured EE frame (single EE for Piper)
        pin = self.env.robot_pin
        ee_name = pin.ee_frames[0]

        state_dict = {"ee_name": ee_name,
                      "to_brick": to_brick, "grab_brick": grab_brick,
                      "home": pin.home_q.copy(),
                      "pregrasp": None, "grasp": None, "close_gripper": None, "retreat_grasp": None,
                      "preplace": None, "place": None, "open_gripper": None, "retreat_place": None,
                      "back_home": pin.home_q.copy()}

        # Preset waypoints for each sub-task.
        unit_height = self.env.brick_unit_height
        pre_grasp_z_offset = 4 * unit_height
        grasp_z_offset = 0.97 * unit_height
        preplace_z_offset = 4 * unit_height
        if("Part_0" in to_brick):
            place_z_offset = 0.0015
        else:
            place_z_offset = unit_height * 1.85
        pre_q = state_dict["home"].copy()

        pregrasp_T = grab_brick_T @ trans_z(pre_grasp_z_offset)
        state_dict["pregrasp"], ik_log = pin.IK({ee_name: pregrasp_T}, pre_q, pin.controllable_joints)
        pre_q = state_dict["pregrasp"].copy()

        grasp_T = grab_brick_T @ trans_z(grasp_z_offset)
        state_dict["grasp"], ik_log = pin.IK({ee_name: grasp_T}, pre_q, pin.controllable_joints)
        pre_q = state_dict["grasp"].copy()

        state_dict["close_gripper"] = self.close_gripper(state_dict["grasp"])
        pre_q = state_dict["close_gripper"].copy()

        state_dict["retreat_grasp"], ik_log = pin.IK({ee_name: pregrasp_T}, pre_q, pin.controllable_joints)
        pre_q = state_dict["retreat_grasp"].copy()

        preplace_T = to_brick_T @ trans_z(preplace_z_offset)
        state_dict["preplace"], ik_log = pin.IK({ee_name: preplace_T}, pre_q, pin.controllable_joints)
        pre_q = state_dict["preplace"].copy()

        place_T = to_brick_T @ trans_z(place_z_offset)
        state_dict["place"], ik_log = pin.IK({ee_name: place_T}, pre_q, pin.controllable_joints)
        pre_q = state_dict["place"].copy()

        state_dict["open_gripper"] = self.open_gripper(state_dict["place"])
        pre_q = state_dict["open_gripper"].copy()

        state_dict["retreat_place"], ik_log = pin.IK({ee_name: preplace_T}, pre_q, pin.controllable_joints)

        self.state_dicts[task_id] = state_dict


    def close_gripper(self, q):
        """
        Computing the joint configuration for closing the gripper.
        """
        q_close = q.copy()
        pin = self.env.robot_pin
        for jname in pin.controllable_joints:
            # Gripper joints were locked (removed from controllable), so check original
            pass
        # Set gripper joints by name via Pinocchio model
        for jname in self.env.robot_configs[0].get("Joint_Order", []):
            if "gripper" in jname:
                jid = pin.pin_model.getJointId(jname)
                idx_q = pin.pin_model.joints[jid].idx_q
                q_close[idx_q] = 0.0
        return q_close

    def open_gripper(self, q):
        """
        Computing the joint configuration for opening the gripper.
        """
        q_open = q.copy()
        pin = self.env.robot_pin
        for jname in self.env.robot_configs[0].get("Joint_Order", []):
            if "gripper" in jname:
                jid = pin.pin_model.getJointId(jname)
                idx_q = pin.pin_model.joints[jid].idx_q
                q_open[idx_q] = 0.02
        return q_open

    def plan_assembly_sequence(self, topology, pre_placed_parts, to_place_placed):
        """
        Plan the assembly sequence based on the task topology.
        """
        # Generate a naive assembly plan
        sorted_topology = bfs_sort_connections(topology)
        def part_id_to_path(id):
            if id in pre_placed_parts:
                return pre_placed_parts[id]
            else:
                return to_place_placed[id]
        def format_part(id):
            if id in pre_placed_parts:
                return f"{pre_placed_parts[id]} (pre-placed)"
            else:
                return f"{to_place_placed[id]}"
        print("Assembly Order:")
        plan = []
        for conn in sorted_topology['connections']:
            skip = conn['stud_id'] in pre_placed_parts.keys() and conn['hole_id'] in pre_placed_parts.keys()
            if not skip:
                plan.append({
                    'stud_path': part_id_to_path(conn['stud_id']),
                    'stud_iface': conn['stud_iface'],
                    'hole_path': part_id_to_path(conn['hole_id']),
                    'hole_iface': conn['hole_iface'],
                    'offset': conn['offset'],
                    'yaw': conn['yaw'],
                })
            print(f" {'SKIP' if skip else '    '} #{conn['id']}: stud = {format_part(conn['stud_id'])} % {conn['stud_iface']}; hole = {format_part(conn['hole_id'])} % {conn['hole_iface']}; offset = {conn['offset']}, yaw = {conn['yaw']}")
        return plan
