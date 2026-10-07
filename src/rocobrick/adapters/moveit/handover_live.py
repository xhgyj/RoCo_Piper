"""Live Isaac HANDOVER trial on the original persistent PICK/lift session.

No new simulator, independent arm planner, home/reset, or synthetic custody.
All phases use admitted paths and actual gripper/action/physics feedback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from rocobrick.adapters.moveit.handover_geometry import (
    HandoverModelFiles,
    IsolatedHandoverGeometry,
)
from rocobrick.adapters.moveit.handover_program import NativeCoordinatedHandoverFactory
from rocobrick.adapters.moveit.handover_scene_planner import (
    HandoverGripperConfigurations,
    NativeHandoverScenePlanner,
    ReadOnlyCombinedHandoverPlanner,
    SessionCombinedPlanRuntime,
)
from rocobrick.adapters.moveit.transit import TransitContext
from rocobrick.coordinated_moveit import (
    CoordinatedOperation,
    CoordinatedRobotTrajectory,
    CoordinatedTrajectoryPoint,
)
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.execution.prepared_motion import (
    PreparedMotionRequest,
    SynchronizationPolicy,
)
from rocobrick.gripper import (
    logical_gripper_to_fingers,
    pad_separation_to_logical_opening,
)
from rocobrick.isaac_backend import (
    gripper_operation,
    transfer_attached_planning_scene_operation,
)
from rocobrick.moveit_contract import GripperTarget
from rocobrick.multi_robot_scheduler import to_json_value
from rocobrick.planning_scene import box_geometry_digest
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.runtime_session import MultiRobotIsaacMoveItSession
from rocobrick.skills.coordinated_handover import (
    CoordinatedHandoverBackend,
    HandoverPhaseReceipt,
)
from rocobrick.skills.handover_contact import (
    CONTACT_PHASES,
    FingerContactGuard,
    SimulationFingerContactPolicy,
)
from rocobrick.skills.handover_continuation import StationaryHandoverContinuation
from rocobrick.skills.handover_planning import (
    HandoverCandidate,
    HandoverPlanningRequest,
    HandoverRobotContext,
    ObjectGraspGeometry,
)
from rocobrick.skills.handover_profile import EXECUTION_CAPABILITIES, finite
from rocobrick.skills.handover_simulation import (
    SimulationHandoverAuthorization,
    simulation_profile,
)
from rocobrick.skills.observation import Pose, _pose, parse_pick_observation
from rocobrick.skills.pick import PickHandler, _tcp_relative_aabb
from rocobrick.skills.placement_geometry import pose_error
from rocobrick.skills.post_lift_handover import capture_post_lift_handover

TRACKING_ERROR_RAD = 0.0005


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _roles(operation: RobotOperation) -> tuple[str, str, str]:
    target = operation.target.values if operation.target is not None else {}
    values = tuple(
        target.get(k)
        for k in ("donor_robot_id", "receiver_robot_id", "physical_instance_id")
    )
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError("HANDOVER trial needs exact roles/object")
    return values


def contact_geometry(
    snapshot: Mapping[str, object], physical: str
) -> ObjectGraspGeometry:
    """Validate the captured local collider descriptor, not a world AABB.

    Returns:
        Hash-bound object-local contact dimensions.
    """
    geometries = snapshot.get("handover_contact_geometry")
    raw = geometries.get(physical) if isinstance(geometries, Mapping) else None
    if not isinstance(raw, Mapping):
        raise ValueError(
            "local collider unavailable: "
            + str(snapshot.get("handover_contact_geometry_error"))
        )
    descriptor = dict(raw)
    checksum = descriptor.pop("source_sha256", None)
    if (
        checksum != _sha(descriptor)
        or descriptor.get("physical_instance_id") != physical
        or descriptor.get("manifest_sha256") != snapshot.get("manifest_sha256")
        or descriptor.get("source") != "usd_object_local_body_collider"
        or descriptor.get("collider_type") != "Cube"
    ):
        raise ValueError("HANDOVER contact box must be an identified authored cube")
    return ObjectGraspGeometry(
        tuple(raw["minimum_m"]), tuple(raw["maximum_m"]), checksum
    )


def require_static_binding(request: HandoverPlanningRequest, shared) -> None:
    """Compare live semantic state while clocks/contact buffers legitimately advance."""
    snapshot, joints = shared["scene_snapshot"], shared["start_state"]
    if shared["planning_scene_version"] != request.scene_version:
        raise ValueError("planning scene changed during HANDOVER planning")
    for key in (
        "session_id",
        "manifest_sha256",
        "tcp_provenance",
        "handover_contact_geometry",
    ):
        if snapshot.get(key) != request.scene_snapshot.get(key):
            raise ValueError(f"HANDOVER input binding changed: {key}")
    if set(joints) != set(request.start_state) or any(
        abs(joints[n] - v) > request.profile.joint_hold_tolerance_rad
        for n, v in request.start_state.items()
    ):
        raise ValueError("arms/fingers moved during HANDOVER planning")
    for key in ("bricks", "robots"):
        if set(snapshot[key]) != set(request.scene_snapshot[key]):
            raise ValueError("HANDOVER scene coverage changed")
        for name, before in request.scene_snapshot[key].items():
            after = snapshot[key][name]
            pose_key = "pose" if key == "bricks" else "tcp_pose"
            error = pose_error(
                _pose(before[pose_key], pose_key), _pose(after[pose_key], pose_key)
            )
            if (
                error[0] > request.profile.position_tolerance_m
                or error[1] > request.profile.orientation_tolerance_rad
            ):
                raise ValueError("physical scene moved during HANDOVER planning")
    old_env, new_env = (
        request.scene_snapshot["environment_colliders"],
        snapshot["environment_colliders"],
    )
    if set(old_env) != set(new_env) or any(
        math.dist(old_env[n][k], new_env[n][k]) > 1e-6
        for n in old_env
        for k in ("aabb_min", "aabb_max")
    ):
        raise ValueError("environment changed during HANDOVER planning")


class LiveSimulationHandover:
    """Own the concrete planner, phase transport, and stationary continuation."""

    def __init__(
        self,
        session: MultiRobotIsaacMoveItSession,
        ledger: OperationStateLedger,
        contexts: Mapping[str, TransitContext],
        *,
        session_id: str,
        manifest_sha256: str,
        package_dirs: tuple[Path, ...],
        profile_id: str,
    ) -> None:
        """Bind one real Isaac session; constructing this service sends no motion."""
        if not isinstance(
            session, MultiRobotIsaacMoveItSession
        ) or session.config.robot_ids != ("piper_0", "piper_1"):
            raise ValueError("simulation HANDOVER needs the actual dual Isaac session")
        self.session, self.ledger, self.contexts = session, ledger, contexts
        self.profile = simulation_profile(profile_id)
        self.finger_policy = SimulationFingerContactPolicy()
        self.authorization = SimulationHandoverAuthorization(
            session_id,
            manifest_sha256,
            self.profile.sha256,
            self.profile.runtime_model_sha256,
            self.finger_policy.sha256,
        )
        self.package_dirs = package_dirs
        self.providers: dict[str, NativeHandoverScenePlanner] = {}
        self.contact_guards: dict[str, FingerContactGuard] = {}
        self.phases: dict[str, str] = {}
        self.contact_clear_required: set[str] = set()
        self.selected_windows = {}
        self.active: dict[str, tuple[str, str, str]] = {}
        self.stopped: set[str] = set()
        self.completed: dict[str, dict[str, dict[str, object]]] = {}
        self.factory = NativeCoordinatedHandoverFactory(
            self.capture,
            self.planner,
            lambda op: self.selected_windows[op.operation_id],
            execution_capabilities=EXECUTION_CAPABILITIES,
            phase_timeout_s=90.0,
            simulation_authorization=self.authorization,
        )
        self.backend = CoordinatedHandoverBackend(self.factory, self, ledger)
        self.continuation = StationaryHandoverContinuation(session, ledger, self)

    def profile_for(self, operation):
        """Resolve only the policy bound to this explicitly opted-in trial.

        Returns:
            The unchanged session-bound simulation policy.
        """
        if operation.parameters.get("grasp_profile_id") != self.profile.profile_id:
            raise ValueError("simulation profile differs from external operation")
        return self.profile

    def donor_window_for(self, operation):
        """Reuse giver's actual original feedback window, not receiver defaults.

        Returns:
            The original measured giver opening acceptance window.
        """
        donor, _, physical = _roles(operation)
        fact = self.ledger.observe(donor, f"holding:{physical}")
        if fact is None:
            raise ValueError("original PICK holding fact unavailable")
        feedback = fact.evidence.get("gripper_close_feedback_confirmed")
        window = (
            feedback.get("expected_opening_m")
            if isinstance(feedback, Mapping)
            else None
        )
        if not isinstance(window, (tuple, list)) or len(window) != 2:
            raise ValueError("original giver calibrated opening window unavailable")
        return tuple(finite(v, "giver window") for v in window)

    def capture(self, operation):
        """Capture fresh post-lift geometry from the same running host.

        Returns:
            Current paired planning inputs with original PICK evidence.
        """
        _, _, physical = _roles(operation)
        shared = self.session.observe_shared_scene()
        snapshot = shared["scene_snapshot"]
        self.authorization.validate(
            self.profile_for(operation), snapshot, EXECUTION_CAPABILITIES
        )
        geometry = contact_geometry(snapshot, physical)
        # The existing capture boundary re-observes after ledger admission;
        # local geometry is stationary authored data, not a pose estimate.
        return capture_post_lift_handover(
            operation,
            ledger=self.ledger,
            session=self.session,
            contexts={
                r: HandoverRobotContext(r, c.base, c.provenance)
                for r, c in self.contexts.items()
            },
            object_geometry=geometry,
            profile=self.profile,
        )

    def planner(self, request):
        """Instantiate actual full URDF/mesh geometry and one combined MoveGroup.

        Returns:
            The isolated native provider on the existing session.
        """
        directory = (
            self.session.session_dir
            / self.session.config.config_output_subdir
            / "dual_arm"
        )
        files = HandoverModelFiles.capture(
            directory / "model.urdf",
            directory / "model.srdf",
            package_dirs=self.package_dirs,
            runtime_model_sha256=self.profile.runtime_model_sha256,
            joint_limits_path=directory / "joint_limits.yaml",
        )
        geometry = IsolatedHandoverGeometry(
            files,
            request,
            tracking_error_rad=TRACKING_ERROR_RAD,
            finger_contact_policy=self.finger_policy,
            simulation_authorization=self.authorization,
            fixed_mount_contacts=tuple(
                (
                    r + "_base_link",
                    "world::environment::/World/scene/roomScene/colliders/table/tableTopActor",
                )
                for r in (request.donor.robot_id, request.receiver.robot_id)
            ),
        )
        calibration_sha = _sha(
            {
                "urdf_sha256": files.urdf_sha256,
                "tcp": asdict(request.donor.provenance),
                "receiver_tcp": asdict(request.receiver.provenance),
                "gripper_source_sha256": hashlib.sha256(
                    Path(__file__).parents[2].joinpath("gripper.py").read_bytes()
                ).hexdigest(),
            }
        )

        def fingers(candidate: HandoverCandidate):
            receiver, donor = request.receiver.robot_id, request.donor.robot_id
            opening = pad_separation_to_logical_opening(
                max(
                    b - a
                    for a, b in zip(
                        request.object_geometry.minimum_m[:2],
                        request.object_geometry.maximum_m[:2],
                    )
                )
                + 0.02
            )
            closing = pad_separation_to_logical_opening(
                max(0.0, candidate.receiver_span_m - 0.002)
            )
            donor_opening = 0.08

            def values(robot, target):
                raw = {
                    n: request.start_state[n]
                    for n in geometry.model.names
                    if n.startswith(robot + "_")
                    and n not in {f"{robot}_joint{i}" for i in range(1, 7)}
                }
                a, b = logical_gripper_to_fingers(target)
                raw.update({f"{robot}_gripper_joint1": a, f"{robot}_gripper_joint2": b})
                geometry.configuration({**request.start_state, **raw})
                return raw

            # Candidate spans are object-local validated dimensions. Keep the
            # giver window from its own original measured PICK feedback.
            op_window = request.giver_holding_evidence["pick_evidence"][
                "gripper_close_feedback_confirmed"
            ]["expected_opening_m"]
            self.selected_windows[request.operation_id] = (
                tuple(op_window),
                (
                    max(0, candidate.receiver_span_m - 0.003),
                    candidate.receiver_span_m + 0.003,
                ),
                (0.075, 0.085),
            )
            return HandoverGripperConfigurations(
                candidate.candidate_id,
                calibration_sha,
                values(receiver, opening),
                values(receiver, closing),
                values(donor, donor_opening),
            )

        def read_static_scene(captured):
            if captured.binding_sha256 != request.binding_sha256:
                raise ValueError(
                    "fresh observation belongs to another planning request"
                )
            shared = self.session.observe_shared_scene()
            require_static_binding(captured, shared)
            return shared

        def read_binding():
            read_static_scene(request)
            return request.binding_sha256

        provider = NativeHandoverScenePlanner(
            geometry,
            ReadOnlyCombinedHandoverPlanner(
                SessionCombinedPlanRuntime(self.session),
                read_static_scene=read_static_scene,
            ),
            grippers=fingers,
            read_binding=read_binding,
        )
        self.providers[request.operation_id] = provider
        self.contact_guards[request.operation_id] = FingerContactGuard(
            session_id=self.authorization.session_id,
            manifest_sha256=self.authorization.manifest_sha256,
            donor=request.donor.robot_id,
            receiver=request.receiver.robot_id,
            policy=self.finger_policy,
            bounds=(
                request.object_geometry.minimum_m,
                request.object_geometry.maximum_m,
            ),
        )
        self.phases[request.operation_id] = "free"
        return provider

    async def observe(self, program):
        """Read real same-tick physics without taking the planning-scene lock.

        Returns:
            A full current physical scene and actual contact report.
        """
        self._validate_contact_binding(program)
        self.authorization.validate(
            program.request.profile,
            program.request.scene_snapshot,
            program.capabilities,
        )
        shared = await asyncio.to_thread(self.session.observe_handover_scene)
        snapshot = shared["scene_snapshot"]
        donor, receiver, physical = (
            program.identity.donor_robot_id,
            program.identity.receiver_robot_id,
            program.identity.physical_instance_id,
        )
        # Keep finger/finger collision reports completely separate from the
        # named pad/object contacts used by the custody state machine.
        observations = {
            r: parse_pick_observation(
                snapshot,
                physical_instance_id=physical,
                robot_id=r,
                session_id=self.authorization.session_id,
                manifest_sha256=self.authorization.manifest_sha256,
            )
            for r in (donor, receiver)
        }
        self.contact_guards[program.identity.operation_id].inspect(
            snapshot,
            phase=self.phases[program.identity.operation_id],
            object_pose=observations[donor].object_pose,
            tcp_poses={r: o.tcp_pose for r, o in observations.items()},
        )
        if program.identity.operation_id in self.contact_clear_required:
            self.contact_guards[program.identity.operation_id].require_clear()
            self.contact_clear_required.remove(program.identity.operation_id)
        return snapshot

    def _validate_contact_binding(self, program) -> None:
        """Keep the compiled, observed and executed contact policies identical."""
        if (
            program.simulation_authorization != self.authorization
            or self.authorization.finger_contact_policy_sha256
            != self.finger_policy.sha256
        ):
            raise ValueError("HANDOVER finger contact policy binding changed")

    def _proofs(self, program):
        return self.completed.setdefault(
            program.identity.operation_id,
            {
                r: {
                    "terminal": True,
                    "safe_to_release": True,
                    "motion_sent": False,
                    "basis": "no HANDOVER action dispatched to this participant",
                }
                for r in program.participants
            },
        )

    def _receipt(self, p, phase, passed, terminal, safe, evidence):
        return HandoverPhaseReceipt(
            p.identity.operation_id,
            p.binding_sha256,
            phase,
            passed,
            terminal,
            safe,
            to_json_value(evidence),
        )

    async def execute_phase(self, program, phase, receiver_attachment):
        """Recertify measured fingers/attachment then execute without replanning.

        Returns:
            A phase receipt bound to actual terminal/action feedback.
        """
        op_id = program.identity.operation_id
        self._validate_contact_binding(program)
        if op_id in self.stopped or program.current_binding() != program.binding_sha256:
            raise ValueError("HANDOVER canceled or mutated before phase admission")
        shared = await asyncio.to_thread(self.session.observe_shared_scene)
        self.phases[op_id] = phase
        provider = self.providers[op_id]
        provider.geometry.files.validate()
        if phase == "scene_transfer":
            return await self._transfer(program, phase, shared)
        if phase in {"receiver_open", "receiver_close", "donor_open"}:
            return await self._gripper(program, phase, shared, provider)
        receipt = await self._motion(
            program, phase, shared, provider, receiver_attachment
        )
        if receipt.passed and phase == "separation":
            # The existing lifecycle observes once more after the receipt.
            # Do not race its single physics observer with a second reader.
            self.contact_clear_required.add(op_id)
        return receipt

    async def _gripper(self, p, phase, shared, provider):
        robot = (
            p.identity.donor_robot_id
            if phase == "donor_open"
            else p.identity.receiver_robot_id
        )
        initial = dict(shared["start_state"])
        target = dict(initial)
        for i in (1, 2):
            name = f"{robot}_gripper_joint{i}"
            target[name] = getattr(p.paths, phase)[-1][name]
        geometry = provider.geometry
        if not geometry.certify_segment(
            initial,
            target,
            deadline_s=time.monotonic() + 15,
            clock=time.monotonic,
            holder=p.identity.donor_robot_id,
            receiver_touch=phase in {"receiver_close", "donor_open"},
            finger_contact=phase in CONTACT_PHASES,
        ):
            raise ValueError(f"{phase}: measured gripper sweep not certified")
        opening = target[f"{robot}_gripper_joint1"] - target[f"{robot}_gripper_joint2"]
        window = (
            p.receiver_holding_window
            if phase == "receiver_close"
            else (
                p.donor_release_window
                if phase == "donor_open"
                else (max(0, opening - 0.003), opening + 0.003)
            )
        )
        request_id = f"{p.identity.operation_id}:{phase}"
        operation = gripper_operation(
            GripperTarget(opening, 4.0, 0.002, window), p.phase_timeout_s, request_id
        )
        self.active[p.identity.operation_id] = ("gripper", request_id, robot)
        proofs = self._proofs(p)
        proofs[robot] = {"terminal": False, "safe_to_release": False}
        try:
            record = await asyncio.to_thread(
                self.session.run_motion_operation, robot, operation
            )
            raw = record.get("result", {})
            execution = raw.get("execution", {})
            terminal, safe = (
                execution.get("terminal") is True,
                execution.get("safe_to_release") is True,
            )
            proofs[robot] = {
                "terminal": terminal,
                "safe_to_release": safe,
                "request_id": request_id,
            }
            feedback = PickHandler._close_feedback({"runtime": to_json_value(raw)})
            passed = raw.get("passed") is True and terminal and safe
            return self._receipt(
                p,
                phase,
                passed,
                terminal,
                safe,
                {
                    "gripper_feedback_confirmed": passed,
                    "sweep_certified": True,
                    "feedback": dict(feedback),
                    "runtime": to_json_value(raw),
                    "simulation_only": True,
                    "finger_contact_policy_sha256": self.finger_policy.sha256,
                    "finger_contact_guard": dict(
                        self.contact_guards[p.identity.operation_id].summary
                    ),
                },
            )
        finally:
            self.active.pop(p.identity.operation_id, None)

    async def _motion(self, p, phase, shared, provider, attachment):
        geometry = provider.geometry
        current = dict(shared["start_state"])
        if phase == "approach":
            from rocobrick.skills.handover_region import candidate_from_endpoint

            # Rebase only the short handler insertion on the measured arrival.
            # No MoveGroup query, nominal endpoint replacement, or giver motion.
            actual = candidate_from_endpoint(
                p.request,
                geometry.tcp_pose(p.identity.donor_robot_id, current),
                geometry.tcp_pose(p.identity.receiver_robot_id, current),
            )
            observed_object = shared["scene_snapshot"]["bricks"][
                p.identity.physical_instance_id
            ]["pose"]
            observed_pose = Pose(
                tuple(observed_object["position_xyz"]),
                tuple(observed_object["orientation_xyzw"]),
            )
            error = pose_error(actual.object_goal, observed_pose)
            if (
                error[0] > p.request.profile.relative_position_tolerance_m
                or error[1] > p.request.profile.relative_orientation_tolerance_rad
            ):
                raise ValueError("approach: measured carried-object transform drift")
            states = provider._cartesian(
                p.request,
                p.identity.receiver_robot_id,
                current,
                actual.receiver_precontact,
                actual.receiver_contact,
                deadline_s=time.monotonic() + 20,
                holder=p.identity.donor_robot_id,
                receiver_touch=True,
                finger_contact=True,
            )
            if states is None:
                raise ValueError(
                    "approach: measured Cartesian insertion failed: "
                    f"{provider._cartesian_failure}"
                )
            states = list(states)
        elif phase == "rendezvous":
            states = []
            for index in range(len(p.rendezvous.trajectories[0].points)):
                state = dict(current)
                for trajectory in p.rendezvous.trajectories:
                    state.update(
                        zip(trajectory.joint_names, trajectory.points[index].positions)
                    )
                states.append(state)
        else:
            states = [
                {
                    **current,
                    **{
                        n: v
                        for n, v in s.items()
                        if n
                        in {
                            f"{r}_joint{i}" for r in p.participants for i in range(1, 7)
                        }
                    },
                }
                for s in getattr(p.paths, phase)
            ]
        if any(
            abs(current[n] - states[0][n]) > p.request.profile.joint_hold_tolerance_rad
            for r in p.participants
            for n in (f"{r}_joint{i}" for i in range(1, 7))
        ):
            raise ValueError(f"{phase}: live arm start exceeds admitted drift")
        states.insert(0, current)
        holder = (
            p.identity.receiver_robot_id
            if phase in {"separation", "receiver_follow"}
            else p.identity.donor_robot_id
        )
        deadline = time.monotonic() + 20
        for a, b in zip(states, states[1:]):
            if not geometry.certify_segment(
                a,
                b,
                deadline_s=deadline,
                clock=time.monotonic,
                holder=holder,
                receiver_touch=phase == "approach",
                receiver_attachment=attachment,
                finger_contact=phase in CONTACT_PHASES,
            ):
                raise ValueError(
                    f"{phase}: measured geometry/tracking envelope not certified"
                )
        times = [0.0]
        for a, b in zip(states, states[1:]):
            contact_duration = 0.0
            if phase in CONTACT_PHASES:
                contact_duration = max(
                    math.dist(
                        geometry.tcp_pose(r, a).position_xyz,
                        geometry.tcp_pose(r, b).position_xyz,
                    )
                    / self.finger_policy.max_tcp_speed_mps
                    for r in p.participants
                )
            times.append(
                times[-1]
                + max(
                    0.2,
                    contact_duration,
                    max(
                        abs(b[f"{r}_joint{i}"] - a[f"{r}_joint{i}"]) / 0.08
                        for r in p.participants
                        for i in range(1, 7)
                    ),
                )
            )
        if times[-1] >= p.phase_timeout_s - 5:
            raise ValueError("slow certified trajectory exceeds phase timeout")
        request_id = f"{p.identity.operation_id}:{phase}"
        trajectories = tuple(
            CoordinatedRobotTrajectory(
                r,
                tuple(f"{r}_joint{i}" for i in range(1, 7)),
                tuple(
                    CoordinatedTrajectoryPoint(
                        t, tuple(s[f"{r}_joint{i}"] for i in range(1, 7))
                    )
                    for t, s in zip(times, states)
                ),
            )
            for r in p.participants
        )
        operation = CoordinatedOperation(
            request_id,
            shared["scene_snapshot"]["scene_digest"],
            p.participants,
            trajectories,
            0.0,
            p.request.profile.max_sync_skew_s,
        )
        request = PreparedMotionRequest(
            operation,
            p.request.session_robots,
            shared["scene_snapshot"],
            shared["planning_scene_version"],
            current,
            SynchronizationPolicy(0.02, TRACKING_ERROR_RAD, 0.2),
            p.phase_timeout_s,
        )
        self.active[p.identity.operation_id] = ("motion", request_id, "")
        proofs = self._proofs(p)
        for r in p.participants:
            proofs[r] = {"terminal": False, "safe_to_release": False}
        try:
            result = await asyncio.to_thread(
                self.session.execute_prepared_motion, request
            )
            lifecycle = result.lifecycle
            participant = lifecycle.get("participants", {})
            for r in p.participants:
                raw = participant.get(r, {})
                proofs[r] = {
                    "terminal": raw.get("terminal") is True,
                    "safe_to_release": raw.get("safe_to_release") is True,
                    "request_id": request_id,
                }
            passed = (
                result.passed
                and lifecycle.get("prepared_binding_sha256") == request.binding_sha256
                and lifecycle.get("replanned") is False
            )
            return self._receipt(
                p,
                phase,
                bool(passed),
                lifecycle.get("terminal") is True,
                lifecycle.get("safe_to_release") is True,
                {
                    "prepared_path_executed": bool(passed),
                    "arrival_confirmed": bool(passed),
                    "tracking_certified": bool(passed),
                    "replanned": False,
                    "tracking_error_rad": TRACKING_ERROR_RAD,
                    "measured_geometry_recertified": True,
                    "handler_cartesian_rebased": phase == "approach",
                    "prepared_binding_sha256": request.binding_sha256,
                    "runtime": to_json_value(lifecycle),
                    "simulation_only": True,
                    "collision_semantics": "nominal_mesh_and_measured_contact_guard"
                    if phase in CONTACT_PHASES
                    else "full_tracking_clearance",
                    "finger_contact_policy_sha256": self.finger_policy.sha256,
                    "finger_contact_guard": dict(
                        self.contact_guards[p.identity.operation_id].summary
                    ),
                },
            )
        finally:
            self.active.pop(p.identity.operation_id, None)

    async def _transfer(self, p, phase, shared):
        snapshot = shared["scene_snapshot"]
        sample = parse_pick_observation(
            snapshot,
            robot_id=p.identity.receiver_robot_id,
            physical_instance_id=p.identity.physical_instance_id,
            session_id=self.authorization.session_id,
            manifest_sha256=self.authorization.manifest_sha256,
        )
        lo, hi = _tcp_relative_aabb(sample)
        receiver, donor = p.identity.receiver_robot_id, p.identity.donor_robot_id
        request_id = f"{p.identity.operation_id}:scene_transfer"
        frame = f"{receiver}_rocobrick_tcp"
        payload = {
            "transfer_id": request_id,
            "object_id": p.identity.physical_instance_id,
            "donor_robot_id": donor,
            "donor_link_name": f"{donor}_rocobrick_tcp",
            "receiver_robot_id": receiver,
            "receiver_link_name": frame,
            "frame_id": frame,
            "minimum_m": list(lo),
            "maximum_m": list(hi),
            "geometry_digest": box_geometry_digest(frame, lo, hi),
            "touch_links": [f"{receiver}_gripper_link{i}" for i in (1, 2)],
            "expected_before": {"donor_attached": True, "receiver_attached": False},
        }
        record = await asyncio.to_thread(
            self.session.run_motion_operation,
            receiver,
            transfer_attached_planning_scene_operation(
                payload, p.phase_timeout_s, request_id
            ),
        )
        raw = record.get("result", {})
        execution = raw.get("execution", {})
        committed = (
            raw.get("passed") is True
            and execution.get("transfer_status") == "COMMITTED"
        )
        readback = (
            execution.get("donor_after") == []
            and isinstance(execution.get("receiver_after"), list)
            and len(execution["receiver_after"]) == 1
        )
        return self._receipt(
            p,
            phase,
            committed and readback,
            execution.get("terminal") is True,
            execution.get("safe_to_release") is True,
            {
                "transfer_status": execution.get("transfer_status"),
                "readback_confirmed": bool(committed and readback),
                "runtime": to_json_value(raw),
                "simulation_only": True,
            },
        )

    async def cancel(self, program):
        """Cancel owned physical lanes; unknown dispatch never becomes safe idle.

        Returns:
            Both participants' independently established stop evidence.
        """
        op_id = program.identity.operation_id
        self.stopped.add(op_id)
        entry = self.active.get(op_id)
        proofs = self._proofs(program)
        if entry is not None:
            kind, request_id, robot = entry
            method = (
                self.session.cancel_free_space
                if kind == "motion"
                else self.session.cancel_motion_operation
            )
            result = await asyncio.to_thread(method, request_id)
            if kind == "motion":
                participants = result.get("participants", {})
                for r in program.participants:
                    proof = participants.get(r, {})
                    proofs[r] = {
                        "terminal": proof.get("terminal") is True,
                        "safe_to_release": proof.get("safe_to_release") is True,
                    }
            else:
                proofs[robot] = {
                    "terminal": result.get("terminal") is True,
                    "safe_to_release": result.get("safe") is True,
                }
        return {"operation_id": op_id, "participants": dict(proofs)}
