"""Plan-only coordinator seam for the combined dual-arm MoveIt group.

This module deliberately stops before controller execution. It converts a
strict combined MoveIt trajectory response into a validated
:class:`CoordinatedOperation`; sending the two arm controller goals is a later
phase with its own prepare/start/cancel protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rocobrick.coordinated_moveit import CoordinatedOperation
from rocobrick.coordinated_moveit_adapter import (
    coordinated_operation_from_joint_trajectory,
)
from rocobrick.coordinated_session import CoordinatedSessionRequest
from rocobrick.dual_arm_readiness import DualArmReadiness, require_group_readiness


class DualArmPlanOnlyCoordinator:
    """Admit one combined planning result without sending motion."""

    def __init__(self, expected_joint_binding: Mapping[str, Sequence[str]]) -> None:
        """Bind exact robot-to-joint ownership for returned trajectories."""
        if not expected_joint_binding:
            raise ValueError("expected_joint_binding must not be empty")
        self._joint_binding = {
            str(robot): tuple(str(joint) for joint in joints)
            for robot, joints in expected_joint_binding.items()
        }

    def admit_readiness(
        self,
        document: Mapping[str, object],
        request: CoordinatedSessionRequest,
    ) -> DualArmReadiness:
        """Validate readiness against the exact coordinated request.

        Returns:
            Frozen group readiness evidence.
        """
        return require_group_readiness(
            document,
            coordination_id=request.coordination_id,
            participants=request.participants,
            scene_digest=request.scene_digest,
        )

    def admit_plan(
        self,
        request: CoordinatedSessionRequest,
        *,
        joint_names: Sequence[str],
        points: Sequence[Mapping[str, object]],
    ) -> CoordinatedOperation:
        """Validate a combined MoveIt plan and return a no-motion operation.

        Returns:
            The strict coordinated operation whose trajectories are ready for a
            later controller coordinator.
        """
        return coordinated_operation_from_joint_trajectory(
            coordination_id=request.coordination_id,
            scene_digest=request.scene_digest,
            participants=request.participants,
            joint_names=joint_names,
            points=points,
            joint_binding=self._joint_binding,
            start_at_sim_time_s=request.targets[request.participants[0]].get(
                "start_at_sim_time_s", 0.0
            ),
            synchronization_tolerance_s=request.targets[request.participants[0]].get(
                "synchronization_tolerance_s", 0.02
            ),
        )
