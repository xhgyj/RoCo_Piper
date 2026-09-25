"""Physical phase backend executing one skill program over a READY session.

Implements :class:`rocobrick.skill_execution.PhysicalPhaseBackend` for the
formal R4.2 runtime session: every phase of the TRANSIT / WAIT / PICK
programs is executed as one or more strictly admitted
:class:`rocobrick.isaac_backend.RuntimeOperation` requests against one
already-READY session. Request and result documents are written into the
run directory (atomic, one pair per operation) and every runtime/transport
failure is synthesized into a failed operation result — this module never
raises through :meth:`execute_phase`.

Phase → operation table (CONFIRMED only when every operation of the phase
passes):

====================================== =========================================
Phase (skill)                          Operations in order
====================================== =========================================
plan_grasp (PICK)                      APPLY_PLANNING_SCENE (frozen update doc)
reach_pregrasp (PICK)                  1. GRIPPER_TARGET reach opening/window
                                       2. TCP_POSE pregrasp target
approach (PICK)                        TCP_POSE approach target
close_gripper (PICK)                   GRIPPER_TARGET close opening/window
confirm_grasp (PICK)                   SCENE_QUERY (brick stillness vs capture)
lift (PICK)                            TCP_POSE lift target
confirm_holding (PICK)                 SCENE_QUERY (follow of lift_height_m)
reach_transit (TRANSIT)                TCP_POSE transit target
hold (WAIT)                            SCENE_QUERY sim_time_s, then poll
                                       (sleep(scene_query_poll_s) + SCENE_QUERY)
                                       until sim_time advanced at least
                                       wait_duration_s, bounded by
                                       operation_timeout_s
====================================== =========================================

On the first motion phase of TRANSIT/PICK one SCENE_QUERY (label ``scene``)
captures the brick AABB for the step's physical instance id; a missing brick
or failed query fails the phase with STALE_WORLD. TRANSIT/PICK targets are
derived from that captured AABB through :mod:`rocobrick.skill_targets`; WAIT
scopes its scene queries to the planning-scene object ids.

Failure-reason mapping (first failure is terminal for the phase):

====================================== =========================================
Condition                              PhaseResult
====================================== =========================================
unknown phase_id                       FAILED CAPABILITY_UNAVAILABLE
brick AABB capture missing/failed      FAILED STALE_WORLD
plan_grasp with no planning-scene doc  FAILED CAPABILITY_UNAVAILABLE
plan_grasp APPLY_PLANNING_SCENE fail   FAILED PLANNING_FAILED
reach_pregrasp GRIPPER_TARGET fail     FAILED TRACKING_FAILED
TCP_POSE execution status REJECTED     FAILED COLLISION_BLOCKED
any other TCP_POSE fail                FAILED TRACKING_FAILED
close_gripper GRIPPER_TARGET fail      FAILED TRACKING_FAILED
confirm_grasp brick moved / query miss FAILED HOLDING_FAILED / STALE_WORLD
lift TCP_POSE fail                     FAILED TRACKING_FAILED / COLLISION_BLOCKED
confirm_holding follow / query miss    FAILED HOLDING_FAILED / STALE_WORLD
hold deadline exceeded                 FAILED TEMPORARILY_UNREACHABLE
hold initial scene query failed        FAILED STALE_WORLD
====================================== =========================================
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from rocobrick.isaac_backend import (
    OperationKind,
    RuntimeOperation,
    RuntimeOperationResult,
    gripper_operation,
    planning_scene_operation,
    scene_query_operation,
    tcp_pose_operation,
)
from rocobrick.models import FailureReason, SkillType
from rocobrick.moveit_contract import GripperTarget
from rocobrick.runtime_session import RuntimeSessionError
from rocobrick.skill_execution import PhaseResult, PhaseStatus
from rocobrick.skills import SkillPhase
from rocobrick.skill_targets import (
    PickTargets,
    RobotBaseFrame,
    SkillParams,
    TcpProvenance,
    pick_targets,
    transit_target,
)

_MOTION_PHASES: Mapping[SkillType, frozenset[str]] = {
    SkillType.TRANSIT: frozenset({"reach_transit"}),
    SkillType.PICK: frozenset({"reach_pregrasp", "approach", "lift"}),
}


@dataclass(frozen=True)
class SkillStepSpec:
    """Identity and numeric knobs of one skill step inside a run."""

    skill: SkillType  # TRANSIT | WAIT | PICK
    physical_instance_id: str | None  # required for TRANSIT/PICK, None for WAIT
    params: SkillParams


def _is_finite_number(value: object) -> bool:
    """True for a finite non-boolean real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _brick_center(brick: Mapping[str, object]) -> tuple[float, float, float] | None:
    """Extract one brick's world-frame AABB center from a scene-state entry."""
    minimum = brick.get("aabb_min")
    maximum = brick.get("aabb_max")
    if not isinstance(minimum, (list, tuple)) or not isinstance(maximum, (list, tuple)):
        return None
    if len(minimum) != 3 or len(maximum) != 3:
        return None
    if not all(_is_finite_number(value) for value in (*minimum, *maximum)):
        return None
    return tuple(  # type: ignore[return-value]
        (float(low) + float(high)) / 2.0 for low, high in zip(minimum, maximum)
    )


class SessionPhaseBackend:
    """Executes skill phases as runtime operations over one READY session."""

    def __init__(
        self,
        *,
        session: object,
        run_dir: Path,
        base: RobotBaseFrame,
        provenance: TcpProvenance,
        spec: SkillStepSpec,
        planning_scene_update: Mapping[str, object] | None,
        request_prefix: str,
        operation_timeout_s: float,
    ) -> None:
        """Bind the backend to one session and one skill step.

        Args:
            session: A READY session exposing ``run_operation(request_path,
                result_path) -> Mapping`` (the real
                :class:`rocobrick.runtime_session.IsaacMoveItSession` or a
                fake with the same seam).
            run_dir: Directory receiving one atomic request/result document
                pair per operation (must be inside the session directory for
                the real session).
            base: The robot base pose in the world frame.
            provenance: Mandatory calibrated-TCP provenance digests.
            spec: Skill identity, physical instance id, and numeric params.
            planning_scene_update: The frozen planning-scene update payload
                applied by ``plan_grasp``; also scopes WAIT scene queries.
            request_prefix: Filename prefix for every operation document.
            operation_timeout_s: Timeout carried by every operation request
                and the bound of the WAIT hold poll loop.
        """
        self._session = session
        self._run_dir = Path(run_dir)
        self._base = base
        self._provenance = provenance
        self._spec = spec
        self._params = spec.params
        self._planning_scene_update = planning_scene_update
        self._request_prefix = request_prefix
        self._operation_timeout_s = float(operation_timeout_s)
        self._phase_id = "<none>"
        self._captured_aabb: (
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        ) = None
        self._pick_targets: PickTargets | None = None

    # ------------------------------------------------------------ plumbing

    async def execute_phase(
        self,
        task: object,
        phase: SkillPhase,
        intent_id: str,
        attempt_id: str,
    ) -> PhaseResult:
        """Execute one skill phase; never raises, never changes the phase."""
        self._phase_id = phase.phase_id
        handler = self._handlers().get(phase.phase_id)
        if handler is None:
            return PhaseResult(
                phase, PhaseStatus.FAILED, FailureReason.CAPABILITY_UNAVAILABLE
            )
        return handler(phase)

    def _handlers(self) -> Mapping[str, Callable[[SkillPhase], PhaseResult]]:
        """The phase_id → handler table for the bound skill."""
        if self._spec.skill is SkillType.TRANSIT:
            return {"reach_transit": self._reach_transit}
        if self._spec.skill is SkillType.WAIT:
            return {"hold": self._hold}
        return {
            "plan_grasp": self._plan_grasp,
            "reach_pregrasp": self._reach_pregrasp,
            "approach": self._approach,
            "close_gripper": self._close_gripper,
            "confirm_grasp": self._confirm_grasp,
            "lift": self._lift,
            "confirm_holding": self._confirm_holding,
        }

    @staticmethod
    def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
        """Write one JSON document atomically (temporary file, then replace)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    def _run_op(
        self, operation: RuntimeOperation, label: str
    ) -> RuntimeOperationResult:
        """Dispatch one operation; synthesize a failed result, never raise."""
        stem = f"{self._request_prefix}-{self._phase_id}-{label}"
        request_path = self._run_dir / f"{stem}-request.json"
        result_path = self._run_dir / f"{stem}-result.json"
        try:
            self._atomic_write_json(request_path, operation.serialize())
            document = self._session.run_operation(request_path, result_path)
            return RuntimeOperationResult.parse(document)
        except (RuntimeSessionError, ValueError, OSError) as error:
            return RuntimeOperationResult(
                request_id=operation.request_id,
                kind=operation.kind,
                passed=False,
                execution={},
                evidence={},
                error=(
                    f"runtime operation {str(operation.kind.value)} {label} "
                    f"failed: {error}"
                ),
            )

    def _run_operation(
        self,
        phase: SkillPhase,
        label: str,
        kind: OperationKind,
        build: Callable[[str], RuntimeOperation],
    ) -> RuntimeOperationResult:
        """Build (strictly admitted) then dispatch one operation of a phase."""
        request_id = (
            f"{self._request_prefix}-{phase.phase_id}-{label}-{uuid4().hex[:8]}"
        )
        try:
            operation = build(request_id)
        except ValueError as error:  # admission/contract rejects before dispatch
            return RuntimeOperationResult(
                request_id=request_id,
                kind=kind,
                passed=False,
                execution={},
                evidence={},
                error=(
                    f"the {str(kind.value)} operation was rejected before "
                    f"dispatch: {error}"
                ),
            )
        return self._run_op(operation, label)

    # ------------------------------------------------------- scene capture

    def _scene_query_brick_ids(self) -> list[str]:
        """Brick ids scoping the scene queries of the bound skill."""
        if self._spec.physical_instance_id is not None:
            return [self._spec.physical_instance_id]
        document = self._planning_scene_update
        if not isinstance(document, Mapping):
            return []
        boxes = document.get("boxes")
        if not isinstance(boxes, (list, tuple)):
            return []
        return sorted(
            str(box["object_id"])
            for box in boxes
            if isinstance(box, Mapping) and "object_id" in box
        )

    def _query_scene_state(
        self, phase: SkillPhase, label: str
    ) -> Mapping[str, object] | None:
        """Run one SCENE_QUERY and return its scene-state document or None."""
        brick_ids = self._scene_query_brick_ids()
        if not brick_ids:
            return None
        result = self._run_operation(
            phase,
            label,
            OperationKind.SCENE_QUERY,
            lambda request_id: scene_query_operation(
                brick_ids,
                [],
                include_gripper=False,
                timeout_s=self._operation_timeout_s,
                request_id=request_id,
            ),
        )
        if not result.passed:
            return None
        execution = result.execution
        state = execution.get("scene_state") if isinstance(execution, Mapping) else None
        return state if isinstance(state, Mapping) else None

    def _brick_center_now(
        self, state: Mapping[str, object]
    ) -> tuple[float, float, float] | None:
        """Center of the step's brick in one scene-state document."""
        if self._spec.physical_instance_id is None:
            return None
        bricks = state.get("bricks")
        if not isinstance(bricks, Mapping):
            return None
        brick = bricks.get(self._spec.physical_instance_id)
        if not isinstance(brick, Mapping):
            return None
        return _brick_center(brick)

    def _sim_time_s(self, state: Mapping[str, object]) -> float | None:
        """The scene-state simulation time, or None when missing/invalid."""
        value = state.get("sim_time_s")
        return float(value) if _is_finite_number(value) else None

    def _ensure_captured(self, phase: SkillPhase) -> PhaseResult | None:
        """Capture the brick AABB on the first motion phase; None on success."""
        motion_phases = _MOTION_PHASES.get(self._spec.skill, frozenset())
        if phase.phase_id not in motion_phases or self._captured_aabb is not None:
            return None
        state = self._query_scene_state(phase, "scene")
        if state is None:
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.STALE_WORLD)
        if self._spec.physical_instance_id is None:
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.STALE_WORLD)
        bricks = state.get("bricks")
        brick = (
            bricks.get(self._spec.physical_instance_id)
            if isinstance(bricks, Mapping)
            else None
        )
        if not isinstance(brick, Mapping):
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.STALE_WORLD)
        minimum = brick.get("aabb_min")
        maximum = brick.get("aabb_max")
        if (
            not isinstance(minimum, (list, tuple))
            or not isinstance(maximum, (list, tuple))
            or len(minimum) != 3
            or len(maximum) != 3
            or not all(_is_finite_number(value) for value in (*minimum, *maximum))
        ):
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.STALE_WORLD)
        self._captured_aabb = (
            (float(minimum[0]), float(minimum[1]), float(minimum[2])),
            (float(maximum[0]), float(maximum[1]), float(maximum[2])),
        )
        return None

    def _targets(self) -> PickTargets:
        """The derived PICK targets of the captured brick AABB."""
        if self._pick_targets is None:
            assert self._captured_aabb is not None  # noqa: S101 - capture precedes motion
            self._pick_targets = pick_targets(
                self._captured_aabb[0],
                self._captured_aabb[1],
                self._base,
                self._provenance,
                self._params,
            )
        return self._pick_targets

    # --------------------------------------------------------- phase bodies

    def _plan_grasp(self, phase: SkillPhase) -> PhaseResult:
        """plan_grasp: apply the frozen planning-scene update document."""
        if self._planning_scene_update is None:
            return PhaseResult(
                phase, PhaseStatus.FAILED, FailureReason.CAPABILITY_UNAVAILABLE
            )
        result = self._run_operation(
            phase,
            "planning-scene",
            OperationKind.APPLY_PLANNING_SCENE,
            lambda request_id: planning_scene_operation(
                self._planning_scene_update,  # type: ignore[arg-type]
                self._operation_timeout_s,
                request_id,
            ),
        )
        if not result.passed:
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.PLANNING_FAILED)
        return PhaseResult(phase, PhaseStatus.CONFIRMED)

    def _gripper_phase(
        self,
        phase: SkillPhase,
        *,
        opening_m: float,
        expected_window_m: tuple[float, float],
        label: str,
    ) -> PhaseResult:
        """One GRIPPER_TARGET phase; any failure is TRACKING_FAILED."""
        result = self._run_operation(
            phase,
            label,
            OperationKind.GRIPPER_TARGET,
            lambda request_id: gripper_operation(
                GripperTarget(
                    opening_m=opening_m,
                    duration_s=self._params.gripper_duration_s,
                    goal_tolerance=self._params.gripper_goal_tolerance,
                    expected_opening_m=expected_window_m,
                ),
                self._operation_timeout_s,
                request_id,
            ),
        )
        if not result.passed:
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.TRACKING_FAILED)
        return PhaseResult(phase, PhaseStatus.CONFIRMED)

    def _pose_phase(
        self, phase: SkillPhase, target: object, label: str
    ) -> PhaseResult:
        """One TCP_POSE phase: REJECTED is collision, otherwise tracking."""
        result = self._run_operation(
            phase,
            label,
            OperationKind.TCP_POSE,
            lambda request_id: tcp_pose_operation(
                target,  # type: ignore[arg-type]
                self._operation_timeout_s,
                request_id,
            ),
        )
        if not result.passed:
            status = (
                result.execution.get("status")
                if isinstance(result.execution, Mapping)
                else None
            )
            reason = (
                FailureReason.COLLISION_BLOCKED
                if status == "REJECTED"
                else FailureReason.TRACKING_FAILED
            )
            return PhaseResult(phase, PhaseStatus.FAILED, reason)
        return PhaseResult(phase, PhaseStatus.CONFIRMED)

    def _reach_pregrasp(self, phase: SkillPhase) -> PhaseResult:
        """reach_pregrasp: open the gripper, then move to the pregrasp pose."""
        capture_failure = self._ensure_captured(phase)
        if capture_failure is not None:
            return capture_failure
        targets = pick_targets(
            self._captured_aabb[0],
            self._captured_aabb[1],
            self._base,
            self._provenance,
            self._params,
        )
        self._pick_targets = targets
        gripper = self._gripper_phase(
            phase,
            opening_m=targets.reach_opening_m,
            expected_window_m=targets.reach_expected_window_m,
            label="gripper-open",
        )
        if gripper.status is not PhaseStatus.CONFIRMED:
            return gripper
        return self._pose_phase(phase, targets.pregrasp, "pose")

    def _approach(self, phase: SkillPhase) -> PhaseResult:
        """approach: move the TCP onto the brick contact mid-height."""
        capture_failure = self._ensure_captured(phase)
        if capture_failure is not None:
            return capture_failure
        return self._pose_phase(phase, self._targets().approach, "pose")

    def _close_gripper(self, phase: SkillPhase) -> PhaseResult:
        """close_gripper: close onto the brick with the contact window."""
        return self._gripper_phase(
            phase,
            opening_m=self._targets().close_opening_m,
            expected_window_m=self._targets().close_expected_window_m,
            label="gripper-close",
        )

    def _confirm_grasp(self, phase: SkillPhase) -> PhaseResult:
        """confirm_grasp: the brick must still sit at its captured center."""
        state = self._query_scene_state(phase, "scene")
        if state is None:
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.STALE_WORLD)
        center = self._brick_center_now(state)
        if center is None or self._captured_aabb is None:
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.STALE_WORLD)
        captured = self._captured_aabb
        captured_center = tuple(
            (low + high) / 2.0 for low, high in zip(captured[0], captured[1])
        )
        displacement = math.sqrt(
            sum((now - then) ** 2 for now, then in zip(center, captured_center))
        )
        if displacement > self._params.still_tolerance_m:
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.HOLDING_FAILED)
        return PhaseResult(phase, PhaseStatus.CONFIRMED)

    def _lift(self, phase: SkillPhase) -> PhaseResult:
        """lift: move the TCP up by the pregrasp standoff plus lift height."""
        capture_failure = self._ensure_captured(phase)
        if capture_failure is not None:
            return capture_failure
        return self._pose_phase(phase, self._targets().lift, "pose")

    def _confirm_holding(self, phase: SkillPhase) -> PhaseResult:
        """confirm_holding: the brick followed the commanded lift height."""
        state = self._query_scene_state(phase, "scene")
        if state is None:
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.STALE_WORLD)
        center = self._brick_center_now(state)
        if center is None or self._captured_aabb is None:
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.STALE_WORLD)
        captured = self._captured_aabb
        captured_center_z = (captured[0][2] + captured[1][2]) / 2.0
        displacement = center[2] - captured_center_z
        if (
            abs(displacement - self._params.lift_height_m)
            > self._params.hold_follow_tolerance_m
        ):
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.HOLDING_FAILED)
        return PhaseResult(phase, PhaseStatus.CONFIRMED)

    def _reach_transit(self, phase: SkillPhase) -> PhaseResult:
        """reach_transit: move above the brick at the transit apex height."""
        capture_failure = self._ensure_captured(phase)
        if capture_failure is not None:
            return capture_failure
        target = transit_target(
            self._captured_aabb[0],
            self._captured_aabb[1],
            self._base,
            self._provenance,
            self._params,
        )
        return self._pose_phase(phase, target, "pose")

    def _hold(self, phase: SkillPhase) -> PhaseResult:
        """hold: bounded poll until simulation time advanced wait_duration_s."""
        params = self._params
        deadline = time.monotonic() + self._operation_timeout_s
        state = self._query_scene_state(phase, "scene")
        start = self._sim_time_s(state) if state is not None else None
        if start is None:
            return PhaseResult(phase, PhaseStatus.FAILED, FailureReason.STALE_WORLD)
        polls = 0
        while True:
            state = self._query_scene_state(phase, f"scene-poll-{polls}")
            polls += 1
            now = self._sim_time_s(state) if state is not None else None
            if now is not None and now - start >= params.wait_duration_s:
                return PhaseResult(phase, PhaseStatus.CONFIRMED)
            if time.monotonic() >= deadline:
                return PhaseResult(
                    phase,
                    PhaseStatus.FAILED,
                    FailureReason.TEMPORARILY_UNREACHABLE,
                )
            time.sleep(params.scene_query_poll_s)
