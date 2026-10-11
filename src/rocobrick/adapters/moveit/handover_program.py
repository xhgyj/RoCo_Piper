"""Compile carrying preparation and post-arrival contact permissions separately.

The request callback captures the existing PICK/lift, and the planner callback
constructs the native full-model provider. Neither may expose controllers.
This compiler does not qualify the installed draft execution profile.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Callable

from rocobrick.adapters.moveit.handover_scene_planner import NativeHandoverScenePlanner
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.coordinated_handover import (
    CoordinatedHandoverPreparation,
    CoordinatedHandoverProgram,
)
from rocobrick.skills.handover_joint_backend import HandoverJointPlanningBackend
from rocobrick.skills.handover_planning import HandoverPlanningRequest
from rocobrick.skills.handover_profile import finite, require_handover_capabilities
from rocobrick.skills.handover_simulation import SimulationHandoverAuthorization
from rocobrick.skills.placement_geometry import compose, inverse, pose_error


class NativeCoordinatedHandoverFactory:
    """Capture post-lift input; hardware retains the qualified full-program gate."""

    def __init__(
        self,
        request_factory: Callable[[RobotOperation], HandoverPlanningRequest],
        planner_factory: Callable[
            [HandoverPlanningRequest], NativeHandoverScenePlanner
        ],
        opening_windows: Callable[
            [RobotOperation],
            tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        ],
        *,
        execution_capabilities: frozenset[str],
        phase_timeout_s: float,
        simulation_authorization: SimulationHandoverAuthorization | None = None,
        staged_preparation: bool = False,
    ) -> None:
        """Bind captured inputs, calibrated windows and session capabilities."""
        self.request_factory, self.planner_factory = request_factory, planner_factory
        self.opening_windows = opening_windows
        self.execution_capabilities = execution_capabilities
        self.simulation_authorization = simulation_authorization
        if type(staged_preparation) is not bool or (
            staged_preparation and simulation_authorization is None
        ):
            raise ValueError(
                "staged preparation is explicitly authorized simulation only"
            )
        self.staged_preparation = staged_preparation
        self.phase_timeout_s = finite(
            phase_timeout_s, "motion phase budget", positive=True
        )

    async def prepare(
        self, operation: RobotOperation
    ) -> CoordinatedHandoverProgram | CoordinatedHandoverPreparation:
        """Compile before any gripper opening or controller creation.

        Returns:
            Safe simulation preparation or a fully admitted legacy program.
        """
        return await asyncio.to_thread(self._compile, operation)

    def _compile(
        self, operation: RobotOperation
    ) -> CoordinatedHandoverProgram | CoordinatedHandoverPreparation:
        request = self.request_factory(operation)
        if request.operation_id != operation.operation_id:
            raise ValueError("captured HANDOVER request belongs to another operation")
        if self.simulation_authorization is None:
            require_handover_capabilities(
                request.profile, self.execution_capabilities, physical=True
            )
        else:
            self.simulation_authorization.validate(
                request.profile, request.scene_snapshot, self.execution_capabilities
            )
        binding = request.binding_sha256
        provider = self.planner_factory(request)
        if not isinstance(provider, NativeHandoverScenePlanner):
            raise ValueError(
                "HANDOVER requires the native full-model planning provider"
            )
        if self.staged_preparation:
            return self._compile_preparation(operation, request, provider)
        backend = HandoverJointPlanningBackend(
            provider, use_moveit_pose_goals=True, clock=provider.clock
        )
        deadline = provider.clock() + request.profile.planning_timeout_s
        rejections = Counter()
        candidate = evaluation = None
        evaluated = 0
        while (
            evaluated < request.profile.max_evaluations and provider.clock() < deadline
        ):
            evaluated += 1
            prepared = provider.plan_region(request, deadline_s=deadline)
            if prepared is None:
                rejections["MOVEIT_REGION_PLAN_OR_ENDPOINT_FAILED"] += 1
                continue
            candidate, path = prepared
            evaluation = backend.evaluate_candidate(
                request, candidate, deadline_s=deadline, preplanned=path
            )
            if evaluation.passed:
                break
            rejections[evaluation.reason] += 1
        if (
            request.binding_sha256 != binding
            or evaluation is None
            or not evaluation.passed
        ):
            raise ValueError(
                f"NO_FEASIBLE_HANDOVER_IN_PREPARATION_REGION; evaluated={evaluated}; "
                f"rejections={dict(rejections)}; "
                f"phase_rejections={provider.phase_rejections}; "
                f"last_path_failure={provider.last_path_failure}; "
                f"last_moveit_failure={provider.planner.last_planning_failure}"
            )
        if candidate is None or evaluation is None or evaluation.joint_plan is None:
            raise ValueError("selected HANDOVER lacks its admitted combined trajectory")
        windows = self.opening_windows(operation)
        return CoordinatedHandoverProgram(
            request,
            candidate,
            evaluation.joint_plan,
            provider.phase_paths(candidate),
            self.execution_capabilities,
            *windows,
            self.phase_timeout_s,
            self.simulation_authorization,
        )

    def _compile_preparation(self, operation, request, provider):
        """Do not demand a future grasp to authorize safe carrying motion.

        Returns:
            An independently admitted paired preparation program.
        """
        deadline = provider.clock() + request.profile.planning_timeout_s
        evaluated = 0
        while (
            evaluated < request.profile.max_evaluations and provider.clock() < deadline
        ):
            evaluated += 1
            prepared = provider.plan_preparation(request, deadline_s=deadline)
            if prepared is not None:
                plan, receiver_open = prepared
                return CoordinatedHandoverPreparation(
                    request,
                    plan,
                    receiver_open,
                    self.execution_capabilities,
                    self.opening_windows(operation)[0],
                    self.phase_timeout_s,
                    self.simulation_authorization,
                )
        raise ValueError(
            f"NO_SAFE_PAIRED_PREPARATION; evaluated={evaluated}; "
            f"phase_rejections={provider.phase_rejections}; "
            f"last_path_failure={provider.last_path_failure}; "
            f"last_moveit_failure={provider.planner.last_planning_failure}"
        )

    async def complete_preparation(self, operation, preparation):
        """Observe actual paired arrival, then certify all final contact phases.

        Returns:
            The contact program bound to the executed preparation.
        """
        return await asyncio.to_thread(
            self._complete_preparation, operation, preparation
        )

    def _complete_preparation(self, operation, preparation):
        if (
            not self.staged_preparation
            or preparation.current_binding() != preparation.binding_sha256
        ):
            raise ValueError("unbound or disabled staged HANDOVER continuation")
        original = preparation.request
        last_fk_error = None
        for attempt in range(3):
            request = self.request_factory(operation)
            if (
                request.operation_id != original.operation_id
                or request.physical_instance_id != original.physical_instance_id
                or request.donor != original.donor
                or request.receiver != original.receiver
                or request.profile != original.profile
            ):
                raise ValueError("arrival capture changed transfer identity or policy")
            self.simulation_authorization.validate(
                request.profile, request.scene_snapshot, self.execution_capabilities
            )
            observed = request.observation(request.donor.robot_id)
            # The measured arrival is the new physical baseline. A change in
            # object-to-TCP pose is diagnostic evidence, not a drop by itself:
            # the contact observer below must still prove that both giver pads
            # retain this exact object. Rebinding the candidate to the measured
            # giver/receiver TCPs lets the handover continue when the brick is
            # still visibly and physically held after tracking drift.
            try:
                provider = self.planner_factory(request)
            except ValueError as error:
                if "captured model FK does not match measured runtime TCP" not in str(error):
                    raise
                last_fk_error = error
                if attempt < 2:
                    time.sleep(0.1)
                    continue
                raise
            if not isinstance(provider, NativeHandoverScenePlanner):
                raise ValueError("arrival requires the native full-model provider")
            candidate = provider.prepare_contact_from_arrival(
                request, deadline_s=provider.clock() + request.profile.planning_timeout_s
            )
            if candidate is None:
                raise ValueError(
                    "HANDOVER_AFTER_PREPARATION: "
                    f"phase_rejections={provider.phase_rejections}; "
                    f"last_path_failure={provider.last_path_failure}"
                )
            return CoordinatedHandoverProgram(
                request,
                candidate,
                None,
                provider.phase_paths(candidate),
                self.execution_capabilities,
                *self.opening_windows(operation),
                self.phase_timeout_s,
                self.simulation_authorization,
                preparation_binding_sha256=preparation.binding_sha256,
            )
        raise ValueError(f"arrival FK did not settle: {last_fk_error}")
