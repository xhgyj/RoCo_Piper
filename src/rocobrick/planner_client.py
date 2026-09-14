"""Typed host client for the isolated planning-only MoveIt service."""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass

import grpc

from rocobrick.execution import Trajectory
from rocobrick.generated import planner_pb2, planner_pb2_grpc
from rocobrick.planner_contract import PlanningRequest, PlanningResponse, PlanningStatus


@dataclass(frozen=True)
class PlannerHealth:
    ready: bool
    model_name: str
    trajectory_execution_enabled: bool


class GrpcPlannerClient:
    def __init__(self, address: str = "127.0.0.1:50051") -> None:
        self._channel = grpc.insecure_channel(address)
        self._stub = planner_pb2_grpc.PlannerStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def health(self, timeout_s: float = 2.0) -> PlannerHealth:
        response = self._stub.Health(planner_pb2.HealthRequest(), timeout=timeout_s)
        return PlannerHealth(response.ready, response.model_name, response.trajectory_execution_enabled)

    def plan(self, request: PlanningRequest, timeout_s: float = 30.0) -> PlanningResponse:
        response = self._stub.Plan(_encode_request(request), timeout=timeout_s)
        status = PlanningStatus(planner_pb2.PlanResponse.Status.Name(response.status))
        trajectory = None
        if response.HasField("trajectory") and response.trajectory.points:
            trajectory = Trajectory(
                tuple(response.trajectory.joint_names),
                tuple(tuple(point.positions) for point in response.trajectory.points),
                tuple(point.time_from_start_s for point in response.trajectory.points),
            )
        return PlanningResponse(
            response.request_id,
            status,
            type(request.revisions)(response.revisions.world, response.revisions.robot_state, response.revisions.motion),
            trajectory,
            frozenset(response.required_resources),
            response.start_digest,
            response.goal_digest,
            response.scene_digest,
            tuple(response.collision_report),
        )

    def plan_async(self, request: PlanningRequest, executor: Executor, timeout_s: float = 30.0):
        return executor.submit(self.plan, request, timeout_s)


def _pose(pose):
    return planner_pb2.Pose(frame_id=pose.frame_id, position_m=pose.position, orientation_xyzw=pose.orientation_xyzw)


def _trajectory(trajectory: Trajectory | None):
    if trajectory is None:
        return planner_pb2.Trajectory()
    return planner_pb2.Trajectory(
        joint_names=trajectory.joint_names,
        points=[planner_pb2.TrajectoryPoint(positions=positions, time_from_start_s=time_s) for positions, time_s in zip(trajectory.points, trajectory.times_s)],
    )


def _encode_request(request: PlanningRequest):
    objects = [
        planner_pb2.CollisionObject(
            object_id=item.object_id,
            pose=_pose(item.pose),
            shape=planner_pb2.CollisionObject.Shape.Value(item.shape.value),
            dimensions_m=item.dimensions_m,
            attached_link=item.attached_to or "",
        )
        for item in request.objects
    ]
    allowances = [
        planner_pb2.ContactAllowance(
            first=item.first,
            second=item.second,
            max_force_n=item.max_force_n,
            max_penetration_m=item.max_penetration_m,
            max_duration_s=item.max_duration_s,
        )
        for item in request.allowed_contacts
    ]
    return planner_pb2.PlanRequest(
        request_id=request.request_id,
        action=request.action.value,
        robot_id=request.robot_id,
        revisions=planner_pb2.Revisions(world=request.revisions.world, robot_state=request.revisions.robot_state, motion=request.revisions.motion),
        start=planner_pb2.JointState(
            names=tuple(f"{request.robot_id}_joint{index}" for index in range(1, len(request.start.joints) + 1)),
            positions=request.start.joints,
        ),
        robots=[
            planner_pb2.RobotState(
                robot_id=robot.robot_id,
                joints=planner_pb2.JointState(
                    names=tuple(
                        f"{robot.robot_id}_joint{index}"
                        for index in range(1, len(robot.joints) + 1)
                    ),
                    positions=robot.joints,
                ),
                holding_part=robot.holding_part or "",
            )
            for robot in request.robots
        ],
        target=_pose(request.target),
        objects=objects,
        other_robot_remaining_motion=_trajectory(request.other_robot_remaining_motion),
        allowed_contacts=allowances,
    )
