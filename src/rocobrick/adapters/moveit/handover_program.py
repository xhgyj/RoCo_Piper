"""Compile a full coordinated HANDOVER program before physical dispatch.

The request callback captures the existing PICK/lift, and the planner callback
constructs the native full-model provider. Neither may expose controllers.
This compiler does not qualify the installed draft execution profile.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable

from rocobrick.adapters.moveit.handover_scene_planner import NativeHandoverScenePlanner
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.coordinated_handover import CoordinatedHandoverProgram
from rocobrick.skills.handover_joint_backend import HandoverJointPlanningBackend
from rocobrick.skills.handover_planning import HandoverPlanningRequest
from rocobrick.skills.handover_profile import finite, require_handover_capabilities
from rocobrick.skills.handover_simulation import SimulationHandoverAuthorization


class NativeCoordinatedHandoverFactory:
    """Capture post-lift input and compile all paths in one plan-only provider."""

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
    ) -> None:
        """Bind captured inputs, calibrated windows and session capabilities."""
        self.request_factory, self.planner_factory = request_factory, planner_factory
        self.opening_windows = opening_windows
        self.execution_capabilities = execution_capabilities
        self.simulation_authorization = simulation_authorization
        self.phase_timeout_s = finite(
            phase_timeout_s, "motion phase budget", positive=True
        )

    async def prepare(self, operation: RobotOperation) -> CoordinatedHandoverProgram:
        """Compile before any gripper opening or controller creation.

        Returns:
            The bound paired program, including all gripper/separation paths.
        """
        return await asyncio.to_thread(self._compile, operation)

    def _compile(self, operation: RobotOperation) -> CoordinatedHandoverProgram:
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
