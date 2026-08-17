# noqa: N999 -- module name follows the public competition convention.
"""Closed-loop scripted expert for dual Piper brick assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from bricksim.core import compute_connection_transform, lookup_physics_connection
from bricksim.topology.ordering import bfs_sort_connections
from scipy.spatial.transform import Rotation

PREGRASP_HEIGHT = 0.10
PREPLACE_HEIGHT = 0.06
PRESS_DEPTH = 0.003
FINGERTIP_CLEARANCE = 0.002
MAX_JOINT_STEP = 0.006
MAX_GRIPPER_STEP = 0.0005
MAX_STATE_STEPS = 600
SETTLE_STEPS = 12
GRIPPER_WAIT_STEPS = 45
POSITION_TOLERANCE = 0.012
ROTATION_TOLERANCE = 0.30
IK_ROTATION_WEIGHT = 0.05
HOME_TOLERANCE = 0.08
TRACKING_COMPENSATION = 0.35
MAX_TRACKING_COMPENSATION = 0.15
GRIPPER_CLOSE_J1 = 0.0
GRIPPER_J2 = 0.0


@dataclass
class EpisodeResult:
    """Terminal and progress information exposed to data collectors."""

    done: bool = False
    success: bool = False
    failure_stage: str | None = None
    failure_reason: str | None = None
    completed_tasks: int = 0
    total_tasks: int = 0


class Policy:
    """Sequential, privileged expert policy for collecting demonstrations."""

    def __init__(self, env, strict_demo: bool = False):
        self.env = env
        self.strict_demo = strict_demo
        self.num_arms = len(env.robot_configs)
        self.plan = self._build_plan()
        self.result = EpisodeResult(total_tasks=len(self.plan))
        self.commands = [rp.home_q.copy() for rp in env.robot_pins]
        self.previous_actual = [rp.home_q.copy() for rp in env.robot_pins]
        self.active_arm = 0
        self.task_index = 0
        self.state = "done" if not self.plan else "home"
        self.state_steps = 0
        self.settle_count = 0
        self.waypoints: dict[str, np.ndarray] = {}
        self.targets: dict[str, np.ndarray] = {}
        self.grasp_start_T = None
        self.brick_to_tcp = None

        for rp in env.robot_pins:
            gripper = [j for j in rp.controllable_joints if "gripper" in j]
            rp.lock_joints(gripper)

        if not self.plan:
            self.result.done = True
            self.result.success = True
        else:
            self._prepare_task()

    def get_action(self, obs):
        """Advance the expert and return the global joint-position command."""
        actual = self._split_actual(obs["joint_positions"])
        if not self.result.done:
            self._advance(actual)

        output = np.zeros(len(self.env.global_joint_order), dtype=np.float32)
        for arm_idx, command in enumerate(self.commands):
            name = self.env.robot_configs[arm_idx].get("Name", f"piper_{arm_idx}")
            start, length = self.env.arm_joint_slices[name]
            output[start:start + length] = command[:length]
            self.previous_actual[arm_idx] = actual[arm_idx]
        return output

    def is_done(self):
        """Return whether the episode has reached a terminal state."""
        return self.result.done

    def succeeded(self):
        """Return true only after every requested connection was verified."""
        return self.result.done and self.result.success

    def episode_result(self):
        """Return a JSON-friendly snapshot of expert progress."""
        return {
            "done": self.result.done,
            "success": self.result.success,
            "failure_stage": self.result.failure_stage,
            "failure_reason": self.result.failure_reason,
            "completed_tasks": self.result.completed_tasks,
            "total_tasks": self.result.total_tasks,
        }

    def _build_plan(self):
        parts = {part["id"]: part for part in self.env.topology["parts"]}
        ordered = bfs_sort_connections(self.env.topology)

        def path_for(part_id):
            if part_id in self.env.pre_placed_parts:
                return self.env.pre_placed_parts[part_id]
            return self.env.to_place_placed[part_id]

        plan = []
        for connection in ordered["connections"]:
            stud_id = connection["stud_id"]
            hole_id = connection["hole_id"]
            if (
                stud_id in self.env.pre_placed_parts
                and hole_id in self.env.pre_placed_parts
            ):
                continue
            # A valid BFS assembly step adds the hole-side brick to an
            # already-built stud-side component.
            if hole_id not in self.env.to_place_placed:
                continue
            plan.append({
                "stud_path": path_for(stud_id),
                "stud_iface": connection["stud_iface"],
                "hole_path": path_for(hole_id),
                "hole_iface": connection["hole_iface"],
                "offset": tuple(connection["offset"]),
                "yaw": connection["yaw"],
                "dimensions": parts[hole_id]["payload"],
            })
        return plan

    def _split_actual(self, global_q):
        result = []
        for arm_idx, rp in enumerate(self.env.robot_pins):
            name = self.env.robot_configs[arm_idx].get("Name", f"piper_{arm_idx}")
            start, length = self.env.arm_joint_slices[name]
            q = rp.home_q.copy()
            q[:length] = np.asarray(global_q[start:start + length])
            result.append(q)
        return result

    def _prepare_task(self):
        task = self.plan[self.task_index]
        self.active_arm = self._select_arm(task["hole_path"])
        self.grasp_start_T = self.env.get_prim_world_T(task["hole_path"])
        grasp_world, pregrasp_world = self._grasp_targets(task, self.active_arm)
        self.targets = {
            "grasp": grasp_world,
            "pregrasp": pregrasp_world,
        }
        self.waypoints = {}
        self.waypoints["home"] = self.env.robot_pins[self.active_arm].home_q.copy()
        seed = self.waypoints["home"]
        self.waypoints["pregrasp"] = self._with_gripper(
            self._solve(pregrasp_world, seed, "plan_pregrasp"),
            close=False,
            task=task,
        )
        self.waypoints["grasp"] = self._with_gripper(
            self._solve(
                grasp_world, self.waypoints["pregrasp"], "plan_grasp"
            ),
            close=False,
            task=task,
        )
        self.waypoints["close_gripper"] = self._with_gripper(
            self.waypoints["grasp"], close=True, task=task
        )
        self.waypoints["lift"] = self._with_gripper(
            self._solve(
                pregrasp_world, self.waypoints["grasp"], "plan_lift"
            ),
            close=True,
            task=task,
        )
        self._enter("home")
        print(
            f"[Expert] task {self.task_index + 1}/{len(self.plan)} "
            f"uses arm {self.active_arm}"
        )

    def _select_arm(self, brick_path):
        brick_pos = self.env.get_prim_world_T(brick_path)[:3, 3]
        return min(
            range(self.num_arms),
            key=lambda i: np.linalg.norm(
                brick_pos - self.env.robot_pins[i].BASE_T[:3, 3]
            ),
        )

    def _grasp_targets(self, task, arm_idx):
        brick = self.env.get_prim_world_T(task["hole_path"])
        dims = task["dimensions"]
        short_axis = 0 if dims["L"] <= dims["W"] else 1
        x_axis = brick[:3, short_axis].copy()
        z_axis = -brick[:3, 2].copy()
        y_axis = np.cross(z_axis, x_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)
        candidates = []
        for sign in (1.0, -1.0):
            target = np.eye(4)
            target[:3, :3] = np.column_stack((sign * x_axis, sign * y_axis, z_axis))
            target[:3, 3] = brick[:3, 3] + brick[:3, 2] * FINGERTIP_CLEARANCE
            target_arm = np.linalg.inv(self.env.robot_pins[arm_idx].BASE_T) @ target
            q, log = self.env.robot_pins[arm_idx].IK(
                {self.env.robot_pins[arm_idx].ee_frames[0]: target_arm},
                self.env.robot_pins[arm_idx].home_q,
                self.env.robot_pins[arm_idx].controllable_joints,
                ROT_WEIGHT=IK_ROTATION_WEIGHT,
            )
            cost = np.linalg.norm(q[:6] - self.env.robot_pins[arm_idx].home_q[:6])
            cost += 10.0 * np.linalg.norm(log["final_error"][:3])
            cost += np.linalg.norm(log["final_error"][3:])
            candidates.append((cost, target))
        grasp = min(candidates, key=lambda item: item[0])[1]
        pregrasp = grasp.copy()
        pregrasp[:3, 3] += brick[:3, 2] * PREGRASP_HEIGHT
        return grasp, pregrasp

    def _solve(self, world_target, seed, planning_stage="planning"):
        rp = self.env.robot_pins[self.active_arm]
        arm_target = np.linalg.inv(rp.BASE_T) @ world_target
        q, log = rp.IK(
            {rp.ee_frames[0]: arm_target}, seed, rp.controllable_joints,
            ROT_WEIGHT=IK_ROTATION_WEIGHT,
        )
        # ``log6`` translation is coupled to its rotation component.  It is
        # therefore not a Cartesian distance when the target orientation has
        # not converged, and previously rejected usable Piper solutions before
        # the first command was sent.  Validate the returned solution with FK.
        ee = rp.ee_frames[0]
        solved = rp.FK(q, [ee])[ee]
        pos_error = np.linalg.norm(solved[:3, 3] - arm_target[:3, 3])
        rot_error = np.linalg.norm(
            Rotation.from_matrix(
                solved[:3, :3].T @ arm_target[:3, :3]
            ).as_rotvec()
        )
        print(
            f"[Expert] {planning_stage}: IK position={pos_error:.4f} m "
            f"rotation={rot_error:.3f} rad"
        )
        if pos_error > 0.025:
            self._fail(
                f"IK position error {pos_error:.4f} m",
                stage=planning_stage,
            )
        return q

    def _advance(self, actual):
        self.state_steps += 1
        arm_q = actual[self.active_arm]
        if self.state_steps > MAX_STATE_STEPS:
            reason = f"state timeout after {MAX_STATE_STEPS} steps"
            target = self.targets.get(self.state)
            if target is not None:
                pos_error, rot_error = self._pose_errors(arm_q, target)
                reason += (
                    f" (position={pos_error:.4f} m, "
                    f"rotation={rot_error:.3f} rad)"
                )
            self._fail(reason)
            return

        if self.state == "home" and self._joint_settled(arm_q, self.waypoints["home"]):
            self._enter("pregrasp")
        elif self.state == "pregrasp" and self._pose_settled(
            arm_q, self.targets["pregrasp"]
        ):
            self._enter("grasp")
        elif self.state == "grasp" and self._pose_settled(arm_q, self.targets["grasp"]):
            self._enter("close_gripper")
        elif self.state == "close_gripper" and self.state_steps >= GRIPPER_WAIT_STEPS:
            self._enter("lift")
        elif self.state == "lift" and self._pose_settled(
            arm_q, self.targets["pregrasp"]
        ):
            if not self._verify_grasp(arm_q):
                self._fail("brick did not follow the gripper during lift")
                return
            self._plan_place(arm_q)
            self._enter("preplace")
        elif self.state == "preplace" and self._pose_settled(
            arm_q, self.targets["preplace"]
        ):
            self._enter("place")
        elif self.state == "place" and self._pose_settled(arm_q, self.targets["place"]):
            self._enter("press")
        elif self.state == "press" and self.state_steps >= GRIPPER_WAIT_STEPS:
            if not self._verify_connection():
                self._fail("BrickSim did not accept the requested connection")
                return
            self._enter("open_gripper")
        elif self.state == "open_gripper" and self.state_steps >= GRIPPER_WAIT_STEPS:
            self._enter("retreat")
        elif self.state == "retreat" and self._pose_settled(
            arm_q, self.targets["preplace"]
        ):
            self._enter("back_home")
        elif self.state == "back_home" and self._joint_settled(
            arm_q, self.waypoints["back_home"]
        ):
            self.result.completed_tasks += 1
            self.task_index += 1
            if self.task_index >= len(self.plan):
                self.result.done = True
                self.result.success = True
                self.state = "done"
                print("[Expert] all requested connections verified")
            else:
                self._prepare_task()

        if not self.result.done:
            goal = self.waypoints.get(self.state)
            if goal is not None:
                compensated = goal.copy()
                if self.state not in {"close_gripper", "open_gripper"}:
                    error = goal[:6] - arm_q[:6]
                    compensated[:6] += np.clip(
                        TRACKING_COMPENSATION * error,
                        -MAX_TRACKING_COMPENSATION,
                        MAX_TRACKING_COMPENSATION,
                    )
                self.commands[self.active_arm] = self._interp(
                    self.commands[self.active_arm], compensated
                )

    def _plan_place(self, actual_q):
        task = self.plan[self.task_index]
        brick_world = self.env.get_prim_world_T(task["hole_path"])
        tcp_world = self._tcp_world(actual_q)
        self.brick_to_tcp = np.linalg.inv(brick_world) @ tcp_world
        quat, pos = compute_connection_transform(
            stud_path=task["stud_path"],
            stud_if=task["stud_iface"],
            hole_path=task["hole_path"],
            hole_if=task["hole_iface"],
            offset=task["offset"],
            yaw=task["yaw"],
        )
        stud_to_hole = np.eye(4)
        stud_to_hole[:3, :3] = Rotation.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        ).as_matrix()
        stud_to_hole[:3, 3] = pos
        desired_brick = self.env.get_prim_world_T(task["stud_path"]) @ stud_to_hole
        place = desired_brick @ self.brick_to_tcp
        preplace = place.copy()
        preplace[:3, 3] += desired_brick[:3, 2] * PREPLACE_HEIGHT
        press = place.copy()
        press[:3, 3] -= desired_brick[:3, 2] * PRESS_DEPTH
        self.targets.update({"preplace": preplace, "place": place, "press": press})
        self.waypoints["preplace"] = self._with_gripper(
            self._solve(preplace, actual_q, "plan_preplace"), close=True, task=task
        )
        self.waypoints["place"] = self._with_gripper(
            self._solve(
                place, self.waypoints["preplace"], "plan_place"
            ), close=True, task=task
        )
        self.waypoints["press"] = self._with_gripper(
            self._solve(
                press, self.waypoints["place"], "plan_press"
            ), close=True, task=task
        )
        self.waypoints["open_gripper"] = self._with_gripper(
            self.waypoints["press"], close=False, task=task
        )
        self.waypoints["retreat"] = self._with_gripper(
            self._solve(
                preplace, self.waypoints["press"], "plan_retreat"
            ), close=False, task=task
        )
        self.waypoints["back_home"] = self.env.robot_pins[self.active_arm].home_q.copy()

    def _verify_grasp(self, actual_q):
        now = self.env.get_prim_world_T(self.plan[self.task_index]["hole_path"])
        lift = now[2, 3] - self.grasp_start_T[2, 3]
        tcp_distance = np.linalg.norm(now[:3, 3] - self._tcp_world(actual_q)[:3, 3])
        return lift > 0.035 and tcp_distance < 0.16

    def _verify_connection(self):
        task = self.plan[self.task_index]
        info = lookup_physics_connection(
            stud_path=task["stud_path"], stud_if=task["stud_iface"],
            hole_path=task["hole_path"], hole_if=task["hole_iface"],
        )
        return (
            info is not None
            and tuple(info.offset) == task["offset"]
            and info.yaw == task["yaw"]
        )

    def _tcp_world(self, q):
        rp = self.env.robot_pins[self.active_arm]
        ee = rp.ee_frames[0]
        return rp.BASE_T @ rp.FK(q, [ee])[ee]

    def _pose_settled(self, q, target_world):
        pos_error, rot_error = self._pose_errors(q, target_world)
        return self._stable(
            pos_error < POSITION_TOLERANCE
            and rot_error < ROTATION_TOLERANCE
        )

    def _pose_errors(self, q, target_world):
        actual = self._tcp_world(q)
        pos_error = np.linalg.norm(actual[:3, 3] - target_world[:3, 3])
        rot_error = np.linalg.norm(
            Rotation.from_matrix(
                actual[:3, :3].T @ target_world[:3, :3]
            ).as_rotvec()
        )
        return pos_error, rot_error

    def _joint_settled(self, q, goal):
        return self._stable(np.max(np.abs(q[:6] - goal[:6])) < HOME_TOLERANCE)

    def _stable(self, condition):
        self.settle_count = self.settle_count + 1 if condition else 0
        return self.settle_count >= SETTLE_STEPS

    def _with_gripper(self, q, close, task):
        result = q.copy()
        dims = task["dimensions"]
        width = min(dims["L"], dims["W"]) * 0.008
        open_width = min(width + 0.012, 0.045)
        rp = self.env.robot_pins[self.active_arm]
        for name in self.env.robot_configs[self.active_arm]["Joint_Order"]:
            if name == "gripper_joint1":
                idx = rp.pin_model.joints[rp.pin_model.getJointId(name)].idx_q
                result[idx] = GRIPPER_CLOSE_J1 if close else open_width
            elif name == "gripper_joint2":
                idx = rp.pin_model.joints[rp.pin_model.getJointId(name)].idx_q
                result[idx] = GRIPPER_J2 if close else -open_width
        return result

    def _enter(self, state):
        if state != self.state:
            print(f"[Expert] {self.state} -> {state}")
        self.state = state
        self.state_steps = 0
        self.settle_count = 0

    def _fail(self, reason, stage=None):
        if self.result.done:
            return
        self.result.done = True
        self.result.success = False
        self.result.failure_stage = stage or self.state
        self.result.failure_reason = reason
        print(f"[Expert] FAILED in {self.result.failure_stage}: {reason}")

    @staticmethod
    def _interp(current, goal):
        delta = np.asarray(goal) - np.asarray(current)
        limits = np.full(delta.shape, MAX_JOINT_STEP, dtype=float)
        if delta.size > 6:
            limits[6:] = MAX_GRIPPER_STEP
        scale = np.max(np.abs(delta) / limits)
        if scale <= 1.0:
            return np.asarray(goal).copy()
        return np.asarray(current) + delta / scale
