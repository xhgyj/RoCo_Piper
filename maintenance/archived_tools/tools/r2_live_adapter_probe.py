#!/usr/bin/env python3
"""R2 LIVE adapter probe: formal R1 library + R2 adapter vs the proven stack.

Runs INSIDE the rootless ROS Humble container (started by
``tools/smoke_r2_live_adapter.sh`` together with the ``move_group`` stack) and
live-verifies the accepted formal modules end to end against the proven
P1.A/B/C host Isaac bridge topology. This script is a THIN COMPOSITION LAYER:
every request build, admission gate, readiness predicate, waypoint check,
feedback metric and success verdict comes from the accepted library/adapter
modules (rocobrick.moveit_contract, moveit_requests, admission, readiness,
feedback, moveit_execution, ros_moveit_runtime, ros_moveit_executor). No
quaternion, joint-limit, trajectory, error-metric or readiness logic is
re-defined here and no execution limit is hardcoded.

Phases, all wall-clock bounded:

1. READINESS: register the runtime subscriptions/clients on this node and
   ``wait_until_ready``; the aggregated ``ReadinessSnapshot`` must pass every
   component (clock, both complete 8-DOF states, exact broadcaster match,
   three active controllers, arm action, MoveGroup action, /compute_fk).
2. OUTBOUND JOINT GOAL: capture the live baseline from the runtime, build the
   target with ``build_joint_target(baseline, {"joint1": 0.10, "joint4":
   0.08})`` (strict caller-side admission with sign fallback, never clamped),
   execute it through ``RosMoveItExecutor.execute_joint`` and demand
   ``is_success``, >= 1 observed /isaac_joint_command message and final arm
   errors inside the adapter contract (0.04 rad, gripper drift 0.01).
3. RETURN TO BASELINE: a ``JointTarget`` of the exact captured baseline
   through the same executor; ``is_success`` required.
4. BASELINE LINK6 POSE DERIVATION: the live ``/compute_fk`` service through
   the minimal adapter hook ``RosMoveItRuntime.compute_link6_fk`` (no
   hardcoded pose anywhere); ``build_link6_pose_target`` turns the derived
   pose into an admitted ``Link6PoseTarget`` (goal constraints tighter than
   the contract maxima, per the live P1.C evidence) and the executor runs it;
   ``is_success`` plus FK-verified final errors <= 0.01 m / 0.03 rad are
   required. This exact-FK pose may be physically nontrivial or minimal; it
   is adapter parity only and carries NO TCP semantics (the TCP stays
   unresolved, the tip frame is link6 in base_link).
5. FINAL RETURN: the exact baseline ``JointTarget`` again through the
   executor; ``is_success`` required, then the arm settles at rest.

The probe prints exactly one machine-readable JSON result line (schema
``rocobrick-r2-live-adapter-probe/v1``) on stdout and exits 0 only when
readiness, outbound joint, joint return, link6 pose and final baseline return
all passed.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict

import rclpy
from rclpy.executors import SingleThreadedExecutor

from rocobrick.feedback import state_is_stable
from rocobrick.moveit_contract import (
    ARM_JOINT_NAMES,
    BASE_FRAME,
    JOINT_NAMES,
    MAX_ORIENTATION_TOLERANCE_RAD,
    MAX_POSITION_TOLERANCE_M,
    SERVICE_COMPUTE_FK,
    JointTarget,
)
from rocobrick.moveit_execution import is_success
from rocobrick.moveit_requests import build_joint_target, build_link6_pose_target
from rocobrick.ros_moveit_executor import RosMoveItExecutor
from rocobrick.ros_moveit_runtime import (
    FINAL_ARM_ERROR_MAX,
    GRIPPER_DRIFT_MAX,
    RosMoveItRuntime,
)

SCHEMA = "rocobrick-r2-live-adapter-probe/v1"
PHASE = "R2"

# Outbound signed deltas applied to the live baseline (rad); admission and
# sign fallback are entirely the library's (build_joint_target).
OUTBOUND_DELTAS = {"joint1": 0.10, "joint4": 0.08}

# Requested goal tolerances for the link6 pose target (m / rad). TIGHTER than
# the contract maxima the runtime verifies against (0.01 m / 0.03 rad), per
# the live P1.C evidence (a constraint at the verification boundary left no
# execution margin). Never looser than the contract.
GOAL_POSITION_TOLERANCE_M = 0.005
GOAL_ORIENTATION_TOLERANCE_RAD = 0.015

# At-rest settle discipline between goals (mirrors the live P1 evidence:
# move_group aborts a plan whose start point deviates from the live state).
SETTLE_STABLE_SAMPLES = 8
SETTLE_STABLE_DELTA_MAX = 5.0e-4
SETTLE_POLL_INTERVAL_S = 0.2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded probe arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="readiness phase bound (s)"
    )
    parser.add_argument(
        "--goal-timeout",
        type=float,
        default=90.0,
        help="per MoveGroup action wall deadline (s); an overrun is cancelled",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=60.0,
        help="per-goal at-rest settle bound (s)",
    )
    parser.add_argument(
        "--fk-timeout",
        type=float,
        default=10.0,
        help="per /compute_fk service call bound (s)",
    )
    args = parser.parse_args(argv)
    if min(args.timeout, args.goal_timeout, args.settle_timeout, args.fk_timeout) <= 0.0:
        parser.error(
            "--timeout, --goal-timeout, --settle-timeout and --fk-timeout "
            "must be positive"
        )
    return args


def _make_fk_hook(args: argparse.Namespace, holder: dict):
    """Build the runtime fk_verifier hook over the minimal adapter FK method.

    The hook closes over ``holder`` because the runtime instance is created
    after the hook; at call time it delegates to the live /compute_fk through
    ``RosMoveItRuntime.compute_link6_fk`` and never invents a pose.
    """

    def fk_hook(state):
        adapter = holder.get("runtime")
        if adapter is None:
            return None
        return adapter.compute_link6_fk(state, timeout_s=args.fk_timeout)

    return fk_hook


def _settle_at_rest(runtime: RosMoveItRuntime, settle_timeout_s: float):
    """Bounded at-rest wait over live samples; returns (state, streak).

    Stability across consecutive runtime samples is the library predicate
    ``rocobrick.feedback.state_is_stable``; no motion is ever sent here.
    """
    previous = runtime.latest_isaac_state()
    streak = 0
    if previous is None:
        return None, streak
    deadline = time.monotonic() + settle_timeout_s
    while time.monotonic() < deadline:
        time.sleep(SETTLE_POLL_INTERVAL_S)
        current = runtime.latest_isaac_state()
        if current is None:
            streak = 0
            continue
        streak = (
            streak + 1
            if state_is_stable(previous, current, max_step=SETTLE_STABLE_DELTA_MAX)
            else 0
        )
        previous = current
        if streak >= SETTLE_STABLE_SAMPLES:
            break
    return runtime.latest_isaac_state(), streak


def _record_step(label: str, outcome) -> dict:
    """JSON-safe evidence record of one ExecutionResult (no verdict here)."""
    return {
        "label": label,
        "status": str(outcome.status.value),
        "moveit_error_code": outcome.moveit_error_code,
        "planned_waypoints": outcome.planned_waypoints,
        "command_messages": outcome.command_messages,
        "final": {
            name: round(float(value), 9)
            for name, value in outcome.final_errors.items()
        },
        "is_success": is_success(outcome),
        "passed": False,
    }


def _joint_record(label: str, outcome) -> dict:
    """Step record with the adapter-contract gates for a joint goal."""
    record = _record_step(label, outcome)
    drift = outcome.details.get("gripper_drift")
    within = bool(
        record["final"]
        and all(value <= FINAL_ARM_ERROR_MAX for value in record["final"].values())
        and drift is not None
        and float(drift) <= GRIPPER_DRIFT_MAX
    )
    record["gripper_drift"] = (
        round(float(drift), 9) if drift is not None else None
    )
    record["final_arm_error_max"] = FINAL_ARM_ERROR_MAX
    record["gripper_drift_max"] = GRIPPER_DRIFT_MAX
    record["within_adapter_contract"] = within
    record["passed"] = bool(
        record["is_success"] and record["command_messages"] >= 1 and within
    )
    return record


def _pose_record(label: str, outcome) -> dict:
    """Step record with the contract FK gates for the link6 pose goal."""
    record = _record_step(label, outcome)
    position = record["final"].get("position_error_m")
    orientation = record["final"].get("orientation_error_rad")
    within = bool(
        position is not None
        and orientation is not None
        and position <= MAX_POSITION_TOLERANCE_M
        and orientation <= MAX_ORIENTATION_TOLERANCE_RAD
    )
    record["position_tolerance_m"] = MAX_POSITION_TOLERANCE_M
    record["orientation_tolerance_rad"] = MAX_ORIENTATION_TOLERANCE_RAD
    record["within_adapter_contract"] = within
    record["passed"] = bool(record["is_success"] and within)
    return record


def _decision_record(decision) -> dict:
    """JSON-safe record of one AdmissionDecision (evidence only)."""
    return {
        "accepted": bool(decision.accepted),
        "reason": decision.reason,
        "details": dict(decision.details),
    }


def run_probe(args: argparse.Namespace) -> dict:
    """Run the bounded probe; return the result dict (never raises)."""
    result: dict = {
        "schema": SCHEMA,
        "phase": PHASE,
        "library": {
            "contract": "rocobrick.moveit_contract",
            "requests": "rocobrick.moveit_requests",
            "execution": "rocobrick.moveit_execution",
            "runtime": "rocobrick.ros_moveit_runtime.RosMoveItRuntime",
            "executor": "rocobrick.ros_moveit_executor.RosMoveItExecutor",
        },
        "bounds": {
            "readiness_timeout_s": float(args.timeout),
            "goal_timeout_s": float(args.goal_timeout),
            "settle_timeout_s": float(args.settle_timeout),
            "fk_timeout_s": float(args.fk_timeout),
            "settle_stable_samples": SETTLE_STABLE_SAMPLES,
            "settle_stable_delta_max": SETTLE_STABLE_DELTA_MAX,
        },
        "request": {
            "outbound_deltas": dict(OUTBOUND_DELTAS),
            "goal_position_tolerance_m": GOAL_POSITION_TOLERANCE_M,
            "goal_orientation_tolerance_rad": GOAL_ORIENTATION_TOLERANCE_RAD,
            "contract_position_tolerance_m": MAX_POSITION_TOLERANCE_M,
            "contract_orientation_tolerance_rad": MAX_ORIENTATION_TOLERANCE_RAD,
            "final_arm_error_max": FINAL_ARM_ERROR_MAX,
            "gripper_drift_max": GRIPPER_DRIFT_MAX,
            "compute_fk_service": SERVICE_COMPUTE_FK,
        },
        "tip_link_frame": BASE_FRAME,
        "readiness": {"waited_ok": False, "passed": False, "snapshot": {}},
        "baseline": None,
        "steps": {},
        "command_messages_total": 0,
        "passed": False,
    }

    rclpy.init()
    node = rclpy.create_node("r2_live_adapter_probe")
    bg_executor = SingleThreadedExecutor()
    bg_executor.add_node(node)
    spin_thread = threading.Thread(target=bg_executor.spin, daemon=True)
    spin_thread.start()
    holder: dict = {}
    runtime = RosMoveItRuntime(node, fk_verifier=_make_fk_hook(args, holder))
    holder["runtime"] = runtime
    moveit_executor = RosMoveItExecutor(runtime)
    try:
        # ------------------------------------- registration + readiness phase
        runtime.register_subscriptions()
        runtime.register_clients()
        ready = runtime.wait_until_ready(timeout_s=args.timeout)
        snapshot = runtime.readiness_snapshot()
        result["readiness"] = {
            "waited_ok": bool(ready),
            "passed": bool(snapshot.passed),
            "snapshot": asdict(snapshot),
        }
        if not (ready and snapshot.passed):
            result["error"] = (
                "readiness phase failed: the adapter snapshot did not pass "
                "every component"
            )
            return result

        # -------------------------------------------------- baseline capture
        baseline, streak = _settle_at_rest(runtime, args.settle_timeout)
        result["baseline_settle_streak"] = streak
        if baseline is None:
            result["error"] = "no complete live baseline state was observed"
            return result
        result["baseline"] = {
            name: round(float(baseline[name]), 9) for name in JOINT_NAMES
        }

        # -------------------------------------------------- outbound joint goal
        target, decision = build_joint_target(baseline, OUTBOUND_DELTAS)
        result["steps"]["outbound_admission"] = _decision_record(decision)
        if not decision.accepted or target is None:
            result["error"] = f"outbound joint target rejected: {decision.reason}"
            return result
        outbound = moveit_executor.execute_joint(target, timeout_s=args.goal_timeout)
        record = _joint_record("outbound_joint", outbound)
        result["steps"]["outbound_joint"] = record
        if not record["passed"]:
            result["error"] = (
                "outbound joint goal failed the adapter contract gates"
            )
            return result

        # ------------------------------------------------ return to baseline
        _, streak = _settle_at_rest(runtime, args.settle_timeout)
        result["steps"]["pre_return_settle_streak"] = streak
        return_target = JointTarget(
            positions={name: float(baseline[name]) for name in ARM_JOINT_NAMES}
        )
        returned = moveit_executor.execute_joint(
            return_target, timeout_s=args.goal_timeout
        )
        record = _joint_record("return_joint", returned)
        result["steps"]["return_joint"] = record
        if not record["passed"]:
            result["error"] = "return to the baseline joints failed the gates"
            return result

        # ------------------------------------- live link6 baseline pose (FK)
        _, streak = _settle_at_rest(runtime, args.settle_timeout)
        result["steps"]["pre_pose_settle_streak"] = streak
        derived = runtime.compute_link6_fk(baseline, timeout_s=args.fk_timeout)
        if derived is None:
            result["error"] = (
                "the live /compute_fk baseline derivation through the adapter "
                "hook returned no pose"
            )
            return result
        position, quaternion = derived
        result["steps"]["link6_fk"] = {
            "service": runtime.services.compute_fk,
            "source_state": {
                name: round(float(baseline[name]), 9) for name in JOINT_NAMES
            },
            "position": [round(float(value), 9) for value in position],
            "orientation_xyzw": [round(float(value), 12) for value in quaternion],
        }

        # ------------------------------------------------ link6 pose execution
        pose_target, pose_decision = build_link6_pose_target(
            frame_id=BASE_FRAME,
            position=position,
            orientation_xyzw=quaternion,
            position_tolerance_m=GOAL_POSITION_TOLERANCE_M,
            orientation_tolerance_rad=GOAL_ORIENTATION_TOLERANCE_RAD,
        )
        result["steps"]["pose_admission"] = _decision_record(pose_decision)
        if not pose_decision.accepted or pose_target is None:
            result["error"] = f"link6 pose target rejected: {pose_decision.reason}"
            return result
        pose_outcome = moveit_executor.execute_link6_pose(
            pose_target, timeout_s=args.goal_timeout
        )
        record = _pose_record("link6_pose", pose_outcome)
        result["steps"]["link6_pose"] = record
        if not record["passed"]:
            result["error"] = (
                "link6 pose goal failed the executor/FK parity gates"
            )
            return result

        # ------------------------------------------------ final baseline return
        _, streak = _settle_at_rest(runtime, args.settle_timeout)
        result["steps"]["pre_final_settle_streak"] = streak
        final_outcome = moveit_executor.execute_joint(
            return_target, timeout_s=args.goal_timeout
        )
        record = _joint_record("final_return", final_outcome)
        result["steps"]["final_return"] = record
        if not record["passed"]:
            result["error"] = (
                "the final baseline return failed the adapter contract gates"
            )
            return result

        result["command_messages_total"] = sum(
            step.get("command_messages", 0)
            for key, step in result["steps"].items()
            if isinstance(step, dict) and "command_messages" in step
        )
        result["passed"] = bool(
            result["readiness"]["passed"]
            and record["passed"]
            and result["steps"]["outbound_joint"]["passed"]
            and result["steps"]["return_joint"]["passed"]
            and result["steps"]["link6_pose"]["passed"]
            and result["steps"]["final_return"]["passed"]
        )
        return result
    except Exception as exc:  # noqa: BLE001 - the probe must always print one line
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        # Order matters: quiesce and join the spinning thread BEFORE the node
        # is destroyed, otherwise the rmw teardown races the daemon executor
        # thread and can abort the process after the result was printed.
        runtime.shutdown()
        bg_executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    """Run the bounded live adapter probe; print one JSON result line."""
    args = parse_args(argv)
    result = run_probe(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
