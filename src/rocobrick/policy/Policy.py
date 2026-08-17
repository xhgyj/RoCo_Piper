# noqa: N999 — module name follows repo convention (Policy.py, NaivePolicy.py)
"""DemoPolicy: Robust dual-arm assembly policy for Piper L.

Improvements over NaivePolicy:
  - State transitions gate on *actual* joint positions from obs.
  - Per-arm state machines with independent task assignment.
  - Tasks assigned to the arm whose base is closer to the pick brick.
  - Settle detection: N consecutive steps within tolerance.
  - Per-state timeout with graceful skip.
  - Gripper open/close with joint-limit awareness.
"""
import numpy as np
from bricksim.topology.ordering import bfs_sort_connections

from rocobrick.utils import trans_z

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
POSITION_TOLERANCE = 0.30       # rad (~17 deg) — PD steady-state error τ_g/k≈0.25
SETTLE_WINDOW = 15              # rolling window size for settle detection
MAX_STEPS_PER_STATE = 600       # ~10 s at 60 FPS; timeout triggers recovery
MAX_STEP = 0.005                # rad/step interpolation cap
GRIPPER_WAIT_STEPS = 40         # 0.67 s for gripper close/open
GRIPPER_OPEN_J1 = 0.04          # m; gripper_joint1 open width (max 0.05)
GRIPPER_OPEN_J2 = 0.0           # m; gripper_joint2 open (0.0 = URDF upper limit)
GRIPPER_CLOSE_J1 = 0.0          # m; closed
GRIPPER_CLOSE_J2 = 0.0          # m


class Policy:
    """Dual-arm brick-assembly policy with closed-loop state transitions."""

    def __init__(self, env):
        """Initialize the policy with the simulation environment.

        Builds assembly plan, locks gripper joints for IK, distributes
        tasks across arms, and pre-plans waypoints for each arm's first task.
        """
        self.env = env
        self.num_arms = len(env.robot_configs)

        # -- Build assembly plan (reuse NaivePolicy BFS logic) --
        self.plan = self._build_plan(
            env.topology, env.pre_placed_parts, env.to_place_placed
        )

        # -- Lock gripper joints on every arm for IK --
        for rp in env.robot_pins:
            gripper_joints = [j for j in rp.controllable_joints if "gripper" in j]
            if gripper_joints:
                rp.lock_joints(gripper_joints)
            cj = rp.controllable_joints
            print(f"[DemoPolicy] locked grippers, controllable: {cj}")

        # -- Per-arm state --
        self.arm_state = ["home"] * self.num_arms
        # commanded (interpolated) joint positions
        self.arm_cmd = [rp.home_q.copy() for rp in env.robot_pins]
        self.arm_task_idx = [0] * self.num_arms
        # waypoints for current task
        self.arm_waypoints = [{} for _ in range(self.num_arms)]
        self.arm_step_cnt = [0] * self.num_arms
        self.arm_err_hist = [[] for _ in range(self.num_arms)]  # rolling error window
        self.arm_done = [False] * self.num_arms
        # per-arm task assignment lists
        self.arm_tasks = [[] for _ in range(self.num_arms)]
        # retry / fail tracking
        self.arm_retries = [0] * self.num_arms
        self.arm_task_done = [False] * self.num_arms

        # -- Distribute tasks across arms --
        self._distribute_tasks()

        # -- Plan first task for each arm --
        for arm_idx in range(self.num_arms):
            self._load_next_task(arm_idx)

        if not self.plan:
            print("[DemoPolicy] empty plan — nothing to assemble.")
            for i in range(self.num_arms):
                self.arm_done[i] = True

    # ==================================================================
    # Public API
    # ==================================================================

    def get_action(self, obs):
        """Return the 16D global joint-position command."""
        n = len(self.env.global_joint_order)
        q_global = np.zeros(n, dtype=np.float32)

        for arm_idx in range(self.num_arms):
            rc = self.env.robot_configs[arm_idx]
            name = rc.get("Name", f"piper_{arm_idx}")
            start, length = self.env.arm_joint_slices[name]
            q_arm = self._step_arm(obs, arm_idx)
            q_global[start:start + length] = q_arm.astype(np.float32)

        return q_global

    def is_done(self):
        """Return True when every arm has finished and returned home."""
        return all(self.arm_done)

    # ==================================================================
    # Assembly planning
    # ==================================================================

    def _build_plan(self, topology, pre_placed, to_place):
        """Build BFS-sorted assembly plan, skipping fully pre-placed connections.

        Returns:
            List of task dicts with stud_path, hole_path, offset, yaw, etc.
        """
        sorted_topo = bfs_sort_connections(topology)

        def path_for(pid):
            if pid in pre_placed:
                return pre_placed[pid]
            return to_place[pid]

        def label_for(pid):
            if pid in pre_placed:
                return f"{pre_placed[pid]} (pre-placed)"
            return f"{to_place[pid]}"

        print("[DemoPolicy] Assembly Order:")
        plan = []
        for conn in sorted_topo["connections"]:
            skip = (conn["stud_id"] in pre_placed
                    and conn["hole_id"] in pre_placed)
            if not skip:
                plan.append({
                    "stud_path": path_for(conn["stud_id"]),
                    "stud_iface": conn["stud_iface"],
                    "hole_path": path_for(conn["hole_id"]),
                    "hole_iface": conn["hole_iface"],
                    "offset": conn["offset"],
                    "yaw": conn["yaw"],
                })
            tag = "SKIP" if skip else "   "
            msg = (f" {tag} #{conn['id']}:"
                   f" stud={label_for(conn['stud_id'])}"
                   f" % {conn['stud_iface']};"
                   f" hole={label_for(conn['hole_id'])}"
                   f" % {conn['hole_iface']};"
                   f" offset={conn['offset']}, yaw={conn['yaw']}")
            print(msg)
        return plan

    def _distribute_tasks(self):
        """Assign each plan task to the nearest arm; check both arms' IK."""
        for task_idx, task in enumerate(self.plan):
            grab_path = task["hole_path"]
            # Print distances to both arms
            try:
                gpos = self.env.get_prim_world_T(grab_path)[:3, 3]
            except Exception:
                gpos = np.zeros(3)
            for i, rp in enumerate(self.env.robot_pins):
                d = np.linalg.norm(gpos - rp.BASE_T[:3, 3])
                print(f"[DemoPolicy] Arm {i} dist to grab brick: {d:.3f} m"
                      f" (base={rp.BASE_T[:3, 3]})")
            arm = self._nearest_arm(grab_path)
            self.arm_tasks[arm].append(task_idx)

        for arm_idx in range(self.num_arms):
            assigned = self.arm_tasks[arm_idx]
            print(f"[DemoPolicy] Arm {arm_idx} tasks: {assigned}"
                  f" ({len(assigned)} total)")

    def _nearest_arm(self, prim_path):
        """Return the arm index whose base is closest to *prim_path*."""
        try:
            pos = self.env.get_prim_world_T(prim_path)[:3, 3]
        except Exception:
            return 0
        best = 0
        best_d = float("inf")
        for i, rp in enumerate(self.env.robot_pins):
            d = np.linalg.norm(pos - rp.BASE_T[:3, 3])
            if d < best_d:
                best_d = d
                best = i
        return best

    # ==================================================================
    # Per-arm stepping
    # ==================================================================

    def _step_arm(self, obs, arm_idx):
        """Advance one arm's state machine.

        Returns:
            8D numpy array of commanded joint positions for this arm.
        """
        rc = self.env.robot_configs[arm_idx]
        name = rc.get("Name", f"piper_{arm_idx}")
        start, length = self.env.arm_joint_slices[name]
        actual_q = obs["joint_positions"][start:start + length].copy()

        # Already done — hold position
        if self.arm_done[arm_idx]:
            return self.arm_cmd[arm_idx]

        state = self.arm_state[arm_idx]
        wp = self.arm_waypoints[arm_idx]

        if not wp:
            # No task — drift toward home
            goal = self.env.robot_pins[arm_idx].home_q.copy()
            self.arm_cmd[arm_idx] = self._interp(self.arm_cmd[arm_idx], goal)
            return self.arm_cmd[arm_idx]

        goal_q = wp.get(state)
        if goal_q is None:
            return self.arm_cmd[arm_idx]

        # Tick step counter (timeout guard)
        self.arm_step_cnt[arm_idx] += 1

        # ==============================================================
        # State transition logic (gated on *actual* joint positions)
        # ==============================================================
        if state == "home":
            if self.arm_step_cnt[arm_idx] >= 30:
                next_t = self._next_task(arm_idx)
                if next_t is not None:
                    self.arm_task_idx[arm_idx] = next_t
                    self._plan_waypoints(next_t, arm_idx)
                    self._enter(arm_idx, "pregrasp")
                else:
                    if self._settled(actual_q, goal_q, arm_idx):
                        self.arm_done[arm_idx] = True
                        print(f"[DemoPolicy] Arm {arm_idx}: all done.")
            else:
                self._settled(actual_q, goal_q, arm_idx)

        elif state == "pregrasp":
            if self._settled(actual_q, goal_q, arm_idx):
                self._enter(arm_idx, "grasp")

        elif state == "grasp":
            if self._settled(actual_q, goal_q, arm_idx):
                self._enter(arm_idx, "close_gripper")

        elif state == "close_gripper":
            if self.arm_step_cnt[arm_idx] >= GRIPPER_WAIT_STEPS:
                err = np.max(np.abs(actual_q[6:8] - goal_q[6:8]))
                if err > 0.01:
                    print(f"[DemoPolicy] Arm {arm_idx}: gripper may not"
                          f" have closed (err={err:.4f}) — continuing")
                self._enter(arm_idx, "retreat_grasp")

        elif state == "retreat_grasp":
            if self._settled(actual_q, goal_q, arm_idx):
                self._enter(arm_idx, "preplace")

        elif state == "preplace":
            if self._settled(actual_q, goal_q, arm_idx):
                self._enter(arm_idx, "place")

        elif state == "place":
            if self._settled(actual_q, goal_q, arm_idx):
                self._enter(arm_idx, "open_gripper")

        elif state == "open_gripper":
            if self.arm_step_cnt[arm_idx] >= GRIPPER_WAIT_STEPS:
                self._enter(arm_idx, "retreat_place")

        elif state == "retreat_place":
            if self._settled(actual_q, goal_q, arm_idx):
                self._enter(arm_idx, "back_home")

        elif state == "back_home":
            if self._settled(actual_q, goal_q, arm_idx):
                self._enter(arm_idx, "home")

        else:
            print(f"[DemoPolicy] WARNING: unknown state '{state}'"
                  f" for arm {arm_idx}")
            self._enter(arm_idx, "home")

        # ==============================================================
        # Timeout guard
        # ==============================================================
        if self.arm_step_cnt[arm_idx] > MAX_STEPS_PER_STATE:
            self.arm_retries[arm_idx] += 1
            print(f"[DemoPolicy] WARNING: Arm {arm_idx} state '{state}'"
                  f" timed out (retry {self.arm_retries[arm_idx]}/2)")
            if self.arm_retries[arm_idx] >= 2:
                self._skip_current_task(arm_idx)
            else:
                self._enter(arm_idx, "back_home")

        # ==============================================================
        # Interpolate command toward current goal
        # ==============================================================
        cur_goal = self.arm_waypoints[arm_idx].get(
            self.arm_state[arm_idx], goal_q)
        if cur_goal is not None:
            self.arm_cmd[arm_idx] = self._interp(
                self.arm_cmd[arm_idx], cur_goal)

        return self.arm_cmd[arm_idx]

    # ==================================================================
    # State helpers
    # ==================================================================

    def _settled(self, actual_q, goal_q, arm_idx):
        """Check if arm joints (indices 0-5) have settled.

        Uses a rolling maximum over the last SETTLE_WINDOW steps.
        This is robust to PD oscillation: if the peak error within
        the window stays below the threshold, the arm has settled.

        Returns:
            True if settled.
        """
        err = np.max(np.abs(goal_q[:6] - actual_q[:6]))
        hist = self.arm_err_hist[arm_idx]
        hist.append(err)
        if len(hist) > SETTLE_WINDOW:
            hist.pop(0)
        # Periodic debug: show tracking error
        if self.arm_step_cnt[arm_idx] % 60 == 0:
            window_max = max(hist) if hist else err
            print(f"[DemoPolicy] Arm {arm_idx} {self.arm_state[arm_idx]}"
                  f" step={self.arm_step_cnt[arm_idx]}"
                  f" max_err={err:.4f} rad"
                  f" window_max={window_max:.4f} rad"
                  f" tol={POSITION_TOLERANCE:.2f}")
        if len(hist) < SETTLE_WINDOW:
            return False
        return max(hist) < POSITION_TOLERANCE

    def _enter(self, arm_idx, new_state):
        """Transition arm to a new state; reset counters and error history."""
        old = self.arm_state[arm_idx]
        self.arm_state[arm_idx] = new_state
        self.arm_step_cnt[arm_idx] = 0
        self.arm_err_hist[arm_idx] = []
        if new_state != old:
            wp = self.arm_waypoints[arm_idx]
            goal = wp.get(new_state)
            if goal is not None:
                print(f"[DemoPolicy] Arm {arm_idx}: {old} -> {new_state}"
                      f" goal[:6]={goal[:6].round(3)}"
                      f" cmd[:6]={self.arm_cmd[arm_idx][:6].round(3)}")
            else:
                print(f"[DemoPolicy] Arm {arm_idx}: {old} -> {new_state}")

    def _interp(self, cur, goal, max_step=MAX_STEP):
        """Bounded-velocity interpolation toward goal.

        Returns:
            Next commanded joint vector, stepped by at most *max_step*.
        """
        cur = np.asarray(cur, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        delta = goal - cur
        m = np.max(np.abs(delta))
        if m <= max_step:
            return goal.copy()
        return cur + delta / m * max_step

    # ==================================================================
    # Task queue
    # ==================================================================

    def _next_task(self, arm_idx):
        """Return the next plan index for this arm, or None.

        Returns:
            Plan index (int) or None.
        """
        tasks = self.arm_tasks[arm_idx]
        idx = self.arm_task_idx[arm_idx]
        if idx < len(tasks):
            return tasks[idx]
        return None

    def _load_next_task(self, arm_idx):
        """Load the next task for this arm and plan its waypoints."""
        next_t = self._next_task(arm_idx)
        if next_t is not None:
            self.arm_retries[arm_idx] = 0
            self._plan_waypoints(next_t, arm_idx)
            self._enter(arm_idx, "home")
        else:
            self.arm_done[arm_idx] = True

    def _skip_current_task(self, arm_idx):
        """Skip the current task after repeated failure."""
        self.arm_task_idx[arm_idx] += 1
        self.arm_retries[arm_idx] = 0
        next_t = self._next_task(arm_idx)
        if next_t is not None:
            print(f"[DemoPolicy] Arm {arm_idx}: skipping to next task"
                  f" #{next_t}")
            self._plan_waypoints(next_t, arm_idx)
            self._enter(arm_idx, "home")
        else:
            print(f"[DemoPolicy] Arm {arm_idx}: no more tasks — marking done")
            self.arm_done[arm_idx] = True

    # ==================================================================
    # IK waypoint planning (per-arm, per-task)
    # ==================================================================

    def _plan_waypoints(self, task_idx, arm_idx):
        """Compute IK waypoints for *task_idx* on *arm_idx*."""
        task = self.plan[task_idx]
        to_path = task["stud_path"]      # target / placed brick
        grab_path = task["hole_path"]    # brick to pick up

        pin = self.env.robot_pins[arm_idx]
        ee = pin.ee_frames[0]  # "link6"

        # --- Diagnostics: world-frame positions ---
        grab_world = self.env.get_prim_world_T(grab_path)
        to_world = self.env.get_prim_world_T(to_path)
        base_pos = pin.BASE_T[:3, 3]
        print(f"[DemoPolicy] Arm {arm_idx} base world pos: {base_pos}")
        print(f"[DemoPolicy] Arm {arm_idx} grab brick world pos:"
              f" {grab_world[:3, 3]}")
        print(f"[DemoPolicy] Arm {arm_idx} to   brick world pos:"
              f" {to_world[:3, 3]}")
        print(f"[DemoPolicy] Arm {arm_idx} grab dist from base:"
              f" {np.linalg.norm(grab_world[:3, 3] - base_pos):.3f} m")

        # --- FK at home for reference ---
        home_fk = pin.FK(pin.home_q, [ee])
        home_ee_world = pin.BASE_T @ home_fk[ee]
        print(f"[DemoPolicy] Arm {arm_idx} EE home world pos:"
              f" {home_ee_world[:3, 3]}")

        to_t = self.env.get_prim_arm_T(to_path, arm_idx=arm_idx)
        grab_t = self.env.get_prim_arm_T(grab_path, arm_idx=arm_idx)
        # Normalize: ensure rotation is valid SO(3)
        for label, t in [("to_t", to_t), ("grab_t", grab_t)]:
            det = np.linalg.det(t[:3, :3])
            if abs(det - 1.0) > 0.1:
                print(f"[DemoPolicy] WARNING: {label} det(R)={det:.4f}"
                      f" — rotation may be invalid!")
        print(f"[DemoPolicy] Arm {arm_idx} grab_t (in base frame):")
        print(f"  pos: {grab_t[:3, 3]}")
        print(f"  z-axis: {grab_t[:3, 2]}")

        uh = self.env.brick_unit_height
        pre_grasp_z = 4 * uh
        grasp_z = 0.97 * uh
        preplace_z = 4 * uh
        place_z = 0.0015 if "Part_0" in to_path else uh * 1.85

        wp = {
            "ee_name": ee,
            "to_brick": to_path,
            "grab_brick": grab_path,
            "home": pin.home_q.copy(),
        }

        seed = pin.home_q.copy()

        # -- pregrasp --
        pregrasp_t = grab_t @ trans_z(pre_grasp_z)
        q, log = pin.IK({ee: pregrasp_t}, seed, pin.controllable_joints)
        self._warn_ik(arm_idx, "pregrasp", log, pin, ee, pregrasp_t, q)
        wp["pregrasp"] = q.copy()
        seed = q.copy()

        # -- grasp --
        grasp_t = grab_t @ trans_z(grasp_z)
        q, log = pin.IK({ee: grasp_t}, seed, pin.controllable_joints)
        self._warn_ik(arm_idx, "grasp", log, pin, ee, grasp_t, q)
        wp["grasp"] = q.copy()
        seed = q.copy()

        # -- close_gripper --
        wp["close_gripper"] = self._gripper(q, arm_idx, close=True)
        seed = wp["close_gripper"].copy()

        # -- retreat_grasp --
        wp["retreat_grasp"] = wp["pregrasp"].copy()
        seed = wp["retreat_grasp"].copy()

        # -- preplace --
        preplace_t = to_t @ trans_z(preplace_z)
        q, log = pin.IK({ee: preplace_t}, seed, pin.controllable_joints)
        self._warn_ik(arm_idx, "preplace", log, pin, ee, preplace_t, q)
        wp["preplace"] = q.copy()
        seed = q.copy()

        # -- place --
        place_t = to_t @ trans_z(place_z)
        q, log = pin.IK({ee: place_t}, seed, pin.controllable_joints)
        self._warn_ik(arm_idx, "place", log, pin, ee, place_t, q)
        wp["place"] = q.copy()

        # -- open_gripper --
        wp["open_gripper"] = self._gripper(q, arm_idx, close=False)

        # -- retreat_place --
        wp["retreat_place"] = wp["preplace"].copy()

        # -- back_home --
        wp["back_home"] = pin.home_q.copy()

        self.arm_waypoints[arm_idx] = wp
        print(f"[DemoPolicy] Arm {arm_idx} waypoints for task"
              f" #{task_idx}: grab={grab_path} -> place on {to_path}")

    def _gripper(self, q, arm_idx, close):
        """Return copy of *q* with gripper joints set to open/close."""
        q_new = q.copy()
        rc = self.env.robot_configs[arm_idx]
        pin = self.env.robot_pins[arm_idx]

        for jname in rc.get("Joint_Order", []):
            if "gripper_joint1" in jname:
                try:
                    jid = pin.pin_model.getJointId(jname)
                    idx = pin.pin_model.joints[jid].idx_q
                    q_new[idx] = GRIPPER_CLOSE_J1 if close else GRIPPER_OPEN_J1
                except Exception:
                    pass
            elif "gripper_joint2" in jname:
                try:
                    jid = pin.pin_model.getJointId(jname)
                    idx = pin.pin_model.joints[jid].idx_q
                    q_new[idx] = GRIPPER_CLOSE_J2 if close else GRIPPER_OPEN_J2
                except Exception:
                    pass
        return q_new

    @staticmethod
    def _warn_ik(arm_idx, stage, log, pin=None, ee=None,
                 target_t=None, q=None):
        """Print warning if IK position error is large (FK-verified)."""
        if pin is not None and ee is not None and target_t is not None \
                and q is not None:
            fk_t = pin.FK(q, [ee])[ee]
            pos_err = np.linalg.norm(fk_t[:3, 3] - target_t[:3, 3])
            if pos_err > 0.01:
                print(f"[DemoPolicy] WARNING: Arm {arm_idx} {stage}"
                      f" IK pos_err={pos_err:.4f} m")
        elif not log["success"] and log["final_error_norm"] > 0.03:
            print(f"[DemoPolicy] WARNING: Arm {arm_idx} {stage}"
                  f" IK final_error_norm={log['final_error_norm']:.4f}")
