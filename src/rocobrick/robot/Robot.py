import pinocchio as pin
import numpy as np
import time
from scipy.optimize import minimize

class Robot_Pin():
    """
    A wrapper class for Pinocchio robot model, providing convenient methods for IK, FK, etc.
    """
    def __init__(self, urdf_fname, package_dir, ee_frames=("tip_l", "tip_r")):
        """
        Load the robot model from URDF and initialize Pinocchio data structures.

        Args:
            urdf_fname: path to the URDF file
            package_dir: directory to resolve package:// URIs
            ee_frames: tuple of end-effector frame names (e.g., ("link6",) for Piper, ("tip_l","tip_r") for DexMate)
        """
        # Pinocchio
        self.pin_model, self.pin_collision_model, self.pin_visual_model = pin.buildModelsFromUrdf(urdf_fname, package_dirs=[package_dir])
        self.pin_data = self.pin_model.createData()
        self.nq = self.pin_model.nq
        self.ee_frames = list(ee_frames)
        self.EE_FRAME_LEFT = self.ee_frames[0] if len(self.ee_frames) > 0 else "tip_l"
        self.EE_FRAME_RIGHT = self.ee_frames[1] if len(self.ee_frames) > 1 else self.ee_frames[0] if len(self.ee_frames) > 0 else "tip_r"
        self.BASE_T = np.eye(4)
        self.home_q = np.zeros(self.nq)

        self.controllable_joints = []
        for j in range(1, self.pin_model.njoints):  # skip universe joint (0)
            joint = self.pin_model.joints[j]
            if joint.nq > 0:   # or joint.nv > 0
                self.controllable_joints.append(self.pin_model.names[j])

    def lock_joints(self, joint_names):
        """
        Lock the specified joints by removing them from the list of controllable joints.
        """
        for joint_name in joint_names:
            if joint_name in self.controllable_joints:
                self.controllable_joints.remove(joint_name)

    def unlock_joints(self, joint_names):
        """
        Unlock the specified joints by adding them back to the list of controllable joints.
        """
        for joint_name in joint_names:
            if joint_name not in self.controllable_joints:
                self.controllable_joints.append(joint_name)

    def partial_q_to_full_q(self, q, joint_names): 
        """
        Convert a partial joint configuration (only for controllable joints) to a full joint configuration for the entire robot.
        The non-controllable joints will be filled with the home configuration values.
        """
        full_q = self.home_q.copy()
        for i in range(len(joint_names)):
            joint_name = joint_names[i]
            joint_id = self.pin_model.getJointId(joint_name)
            idx_q = self.pin_model.joints[joint_id].idx_q
            full_q[idx_q] = q[i]
        return full_q
    
    def full_q_to_partial_q(self, full_q, joint_names):
        """
        Convert a full joint configuration to a partial joint configuration for the controllable joints.
        """
        q = np.zeros(len(joint_names))
        for i in range(len(joint_names)):
            joint_name = joint_names[i]
            joint_id = self.pin_model.getJointId(joint_name)
            idx_q = self.pin_model.joints[joint_id].idx_q
            q[i] = full_q[idx_q]
        return q

    def IK(self, targets, init_full_q, moving_joints, ROT_WEIGHT=1.0, STEP_SIZE=0.1, TOL=1e-3, MAX_ITERS=10000,
           method="auto", random_restarts=20):
        """
        A generic IK solver that can handle multiple end-effectors with different target poses.

        Args:
            targets: a dictionary of {frame_name: target_T} specifying the desired target pose for each end-effector frame.
            init_full_q: the initial full joint configuration for the IK solver to start from.
            moving_joints: a list of joint names that are allowed to move during IK. The IK solution will only be computed for these joints, and the rest will be fixed at their initial values
            method: "jacobian" (fast gradient-based), "optimize" (scipy L-BFGS-B, robust), or "auto" (try jacobian first, fall back to optimize)

        Returns:
            q: the resulting full joint configuration that achieves the desired end-effector poses (or is close to them within the specified tolerance).
            ik_log: a dictionary containing information about the IK solving process, such as success status, number of iterations, final error, and solve time.
        """
        ts = time.time()
        ee_ids = []
        target_poses = []
        for frame_name in targets.keys():
            cart_T = targets[frame_name]
            ee_id = self.pin_model.getFrameId(frame_name)
            ee_ids.append(ee_id)
            target_pose = pin.SE3(cart_T[:3, :3], cart_T[:3, 3])
            target_poses.append(target_pose)

        unlock_joint_ids = []
        for joint_name in moving_joints:
            joint_id = self.pin_model.getJointId(joint_name)
            idx_q = self.pin_model.joints[joint_id].idx_q
            unlock_joint_ids.append(idx_q)
        lower_bounds = self.pin_model.lowerPositionLimit[unlock_joint_ids]
        upper_bounds = self.pin_model.upperPositionLimit[unlock_joint_ids]

        arm_joint_names = [j for j in moving_joints if "gripper" not in j]
        arm_joint_idx = []
        for jname in arm_joint_names:
            jid = self.pin_model.getJointId(jname)
            arm_joint_idx.append(self.pin_model.joints[jid].idx_q)
        arm_lb = self.pin_model.lowerPositionLimit[arm_joint_idx]
        arm_ub = self.pin_model.upperPositionLimit[arm_joint_idx]

        # --- Helper: run jacobian IK from a given seed ---
        def _jacobian_ik(q_seed, max_iters, tol):
            q0 = q_seed[unlock_joint_ids].copy()
            for i in range(max_iters):
                full_q0 = self.partial_q_to_full_q(q0, moving_joints)
                pin.forwardKinematics(self.pin_model, self.pin_data, full_q0)
                pin.updateFramePlacements(self.pin_model, self.pin_data)
                error = np.zeros(6 * len(ee_ids))
                unscaled_error = np.zeros(6 * len(ee_ids))
                Js = np.zeros((6 * len(ee_ids), len(unlock_joint_ids)))
                idx = 0
                for ee_id, target_pose in zip(ee_ids, target_poses):
                    current_pose = self.pin_data.oMf[ee_id]
                    err = pin.log6(current_pose.inverse() * target_pose).vector
                    unscaled_error[6*idx:6*(idx+1)] = err.copy()
                    err[3:] *= ROT_WEIGHT
                    J = pin.computeFrameJacobian(self.pin_model, self.pin_data, full_q0, ee_id, pin.LOCAL)
                    J = J[:, unlock_joint_ids]
                    J[3:, :] *= ROT_WEIGHT
                    Js[6*idx:6*(idx+1), :] = J
                    error[6*idx:6*(idx+1)] = err
                    idx += 1
                if np.linalg.norm(error) < tol:
                    return full_q0, True, i+1, unscaled_error
                J_inv = np.linalg.pinv(Js)
                dq = STEP_SIZE * J_inv @ error
                q0 = q0 + dq
                q0 = np.minimum(np.maximum(q0, lower_bounds), upper_bounds)
            full_q0 = self.partial_q_to_full_q(q0, moving_joints)
            pin.forwardKinematics(self.pin_model, self.pin_data, full_q0)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            unscaled_error = np.zeros(6 * len(ee_ids))
            idx = 0
            for ee_id, target_pose in zip(ee_ids, target_poses):
                current_pose = self.pin_data.oMf[ee_id]
                err = pin.log6(current_pose.inverse() * target_pose).vector
                unscaled_error[6*idx:6*(idx+1)] = err
                idx += 1
            return full_q0, False, max_iters, unscaled_error

        # --- Helper: optimization-based IK with random restarts ---
        def _optimize_ik():
            best_cost = 1e9
            best_x = None
            for restart in range(random_restarts):
                if restart == 0:
                    x0 = init_full_q[arm_joint_idx].copy()
                else:
                    x0 = np.random.uniform(arm_lb, arm_ub)

                def cost(x):
                    fq = init_full_q.copy()
                    for k, idx_q in enumerate(arm_joint_idx):
                        fq[idx_q] = x[k]
                    pin.forwardKinematics(self.pin_model, self.pin_data, fq)
                    pin.updateFramePlacements(self.pin_model, self.pin_data)
                    total = 0.0
                    for ee_id, target_pose in zip(ee_ids, target_poses):
                        current_pose = self.pin_data.oMf[ee_id]
                        err = pin.log6(current_pose.inverse() * target_pose).vector
                        # A Piper parallel gripper must approach with its tool
                        # axis pointing into the brick.  Ignoring orientation
                        # can put the wrist at the right point with the fingers
                        # facing away from the workpiece.
                        total += np.sum(err[:3]**2)
                        total += (ROT_WEIGHT ** 2) * np.sum(err[3:]**2)
                    return total

                res = minimize(cost, x0, method='L-BFGS-B',
                              bounds=list(zip(arm_lb, arm_ub)),
                              options={'maxiter': 2000, 'ftol': 1e-12})
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_x = res.x

            # Final FK to compute error
            fq = init_full_q.copy()
            for k, idx_q in enumerate(arm_joint_idx):
                fq[idx_q] = best_x[k]
            pin.forwardKinematics(self.pin_model, self.pin_data, fq)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            unscaled_error = np.zeros(6 * len(ee_ids))
            idx = 0
            for ee_id, target_pose in zip(ee_ids, target_poses):
                current_pose = self.pin_data.oMf[ee_id]
                err = pin.log6(current_pose.inverse() * target_pose).vector
                unscaled_error[6*idx:6*(idx+1)] = err
                idx += 1
            err_norm = np.linalg.norm(unscaled_error)
            # Scale check with ROT_WEIGHT
            scaled = unscaled_error.copy()
            scaled[3:] *= ROT_WEIGHT
            success = np.linalg.norm(scaled) < TOL
            return fq, success, 0, unscaled_error

        # --- Dispatch ---
        if method == "jacobian":
            q, status, iters, unscaled_error = _jacobian_ik(init_full_q, MAX_ITERS, TOL)
        elif method == "optimize":
            q, status, iters, unscaled_error = _optimize_ik()
        else:  # auto
            # Try fast jacobian first
            q, status, iters, unscaled_error = _jacobian_ik(init_full_q, MAX_ITERS, TOL)
            if not status:
                # Fall back to optimization with random restarts
                q_opt, status_opt, _, unscaled_error_opt = _optimize_ik()
                scaled_jac = unscaled_error.copy()
                scaled_opt = unscaled_error_opt.copy()
                scaled_jac[3:] *= ROT_WEIGHT
                scaled_opt[3:] *= ROT_WEIGHT
                if status_opt or np.linalg.norm(scaled_opt) < np.linalg.norm(scaled_jac):
                    q, status, unscaled_error = q_opt, status_opt, unscaled_error_opt

        return q, {"success": status, "iterations": iters if method != "optimize" else -1,
                   "final_error": unscaled_error,
                   "final_error_norm": np.linalg.norm(unscaled_error),
                   "solve_time": time.time() - ts}

    def FK(self, q, frame_names):
        """
        Compute the forward kinematics for the specified end-effector frames given the full joint configuration q.
        
        Args:
            q: the full joint configuration for the entire robot.
            frame_names: a list of end-effector frame names for which to compute the forward kinematics.
        
        Returns:
            Ts: a dictionary mapping each frame name to its corresponding homogeneous transformation matrix.
        """
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        Ts = dict()
        for frame_name in frame_names:
            ee_id = self.pin_model.getFrameId(frame_name)
            T = self.pin_data.oMf[ee_id].homogeneous
            Ts[frame_name] = T
        return Ts


if __name__ == "__main__":
    # Example Usage and Test
    robot = Robot_Pin("./robot_assets/DexMate/vega_1u_gripper.urdf", "./robot_assets/DexMate")
    ee_names = ["tip_l"]
    init_q = np.zeros(robot.nq)

    # Get arbitrary goal T
    q = init_q.copy()
    Ts = robot.FK(q, ee_names)
    T = Ts[ee_names[0]]

    # Test IK
    target_q, ik_log = robot.IK({"tip_l": T}, init_q, robot.controllable_joints, ROT_WEIGHT=1.0)
    print("IK log:", ik_log)

    # Verify FK( IK(T) ) = T
    Ts = robot.FK(target_q, ee_names)
    print(Ts[ee_names[0]])
    print(T)
    assert(np.allclose(Ts[ee_names[0]], T, atol=1e-3))
