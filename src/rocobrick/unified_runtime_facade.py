"""Host transport facade for bound successor pre-phase requests."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Protocol

from rocobrick.coordinated_session import (
    CoordinatedSessionRequest,
    CoordinatedSessionResult,
    FreeSpaceMotionEvidence,
    FreeSpaceMotionRequest,
)
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.planning_scene import box_from_aabb
from rocobrick.skills.observation import _pose, parse_pick_observation
from rocobrick.skills.pick import _tcp_relative_aabb
from rocobrick.skills.placement_geometry import compose, inverse, pose_error
from rocobrick.skills.support_evidence import verify_support_contact
from rocobrick.unified_dual_arm_runtime import (
    SharedSceneSnapshot,
    SuccessorPrePhaseEvidence,
    SuccessorPrePhaseRequest,
    TerminalPrePhasePolicy,
    failed_pre_phase,
)

SHARED_SNAPSHOT_MAX_AGE_S = 0.5


class HandoverContinuationBoundary(Protocol):
    """Physical stationary admission and revalidation, with no motion API."""

    async def admit_stationary(
        self,
        request: SuccessorPrePhaseRequest,
    ) -> SuccessorPrePhaseEvidence:
        """Prove retained giver custody for stationary lease adoption."""
        ...

    async def validate_stationary(
        self,
        request: SuccessorPrePhaseRequest,
        evidence: SuccessorPrePhaseEvidence,
    ) -> bool:
        """Recheck exact issued evidence against current physical custody."""
        ...


class UnifiedDualArmRuntimeFacade:
    """Dispatch one combined plan and require measured physical completion."""

    def __init__(
        self,
        session: object,
        contexts: Mapping[str, object] | None = None,
        state_ledger: OperationStateLedger | None = None,
    ) -> None:
        """Bind the host mailbox transport, calibrated context and ledger."""
        self._session = session
        self._contexts = dict(contexts or {})
        self._state_ledger = state_ledger
        self._handover_continuation: HandoverContinuationBoundary | None = None

    def bind_handover_continuation(
        self,
        continuation: HandoverContinuationBoundary,
    ) -> None:
        """Bind explicit stationary custody admission without substituting TRANSIT."""
        if self._handover_continuation is not None:
            raise ValueError("HANDOVER continuation already bound")
        for name in ("admit_stationary", "validate_stationary"):
            if not callable(getattr(continuation, name, None)):
                raise ValueError(
                    "HANDOVER continuation lacks physical custody admission"
                )
        self._handover_continuation = continuation

    @property
    def robot_ids(self) -> tuple[str, ...]:
        """Return the complete persistent session robot identity."""
        return tuple(sorted(self._contexts))

    async def observe_all(self) -> Mapping[str, object]:
        """Fetch one shared scene through the persistent child.

        Returns:
            Complete scene snapshot, not a merge of per-robot observations.
        """
        return await asyncio.to_thread(self._session.observe_all)

    async def observe_pre_phase(
        self, participants: tuple[str, ...]
    ) -> SharedSceneSnapshot:
        """Read one admitted shared snapshot through the persistent worker.

        Returns:
            One admitted observation covering every participant, bound to the
            session provenance, planning-scene version, shared-clock freshness
            and confirmed physical facts of exactly this run.
        """
        snapshot, version, observed = await asyncio.to_thread(self._read_shared)
        return self._admit(snapshot, version, observed, tuple(participants))

    async def plan_only(self, operation, physical_facts):
        """Derive targets on ROS side and admit a no-motion combined trajectory.

        Returns:
            Plan-only evidence that cannot satisfy physical pre-phase completion.
        """
        snapshot = await self.observe_all()
        request = CoordinatedSessionRequest(
            coordination_id=f"plan-only:{operation.operation_id}",
            robots=tuple(sorted(self._contexts)),
            participants=tuple(operation.participants),
            scene_digest=snapshot["scene_digest"],
            planning_group="dual_arm",
            plan_only=True,
            timeout_s=60.0,
            targets={},
            scene_snapshot=snapshot,
            physical_facts=physical_facts,
            operation=asdict(operation),
            target_contexts={
                robot: asdict(context) for robot, context in self._contexts.items()
            },
        )
        return await asyncio.to_thread(
            self._session.execute_successor_pre_phase, request
        )

    async def free_space_context(
        self, participants: tuple[str, ...]
    ) -> tuple[Mapping[str, object], int, Mapping[str, float]]:
        """Read shared scene/version and the complete combined start state.

        Returns:
            The current identified snapshot, scene version and measured joints.
        """
        shared = await asyncio.to_thread(self._session.observe_shared_scene)
        if not isinstance(shared, Mapping):
            raise RuntimeError("free-space shared observation is invalid")
        snapshot = shared.get("scene_snapshot")
        version = shared.get("planning_scene_version")
        if not isinstance(snapshot, Mapping):
            raise RuntimeError("free-space scene snapshot is unavailable")
        if type(version) is not int or version < 0:
            raise RuntimeError("free-space scene version is invalid")
        if not set(participants) <= set(snapshot.get("robots", {})):
            raise RuntimeError("free-space context does not cover participants")
        start_state = shared.get("start_state")
        if not isinstance(start_state, Mapping):
            raise RuntimeError("free-space start state is unavailable")
        return (
            snapshot,
            version,
            {str(name): float(value) for name, value in start_state.items()},
        )

    def _free_space_request(
        self, request: FreeSpaceMotionRequest, *, plan_only: bool
    ) -> CoordinatedSessionRequest:
        """Use the same measured collision facts for planning and execution.

        Returns:
            A request bound to this snapshot, joints and physical facts.
        """
        coordinated = request.to_coordinated_request(plan_only=plan_only)
        robots = request.scene_snapshot.get("robots", {})
        facts = {
            robot: dict((robots.get(robot) or {}).get("facts") or {})
            for robot in request.participants
        }
        for robot, carried in request.physical_facts.items():
            facts[robot].update(dict(carried))
        if self._state_ledger is not None:
            for robot in request.participants:
                facts[robot].update(
                    {
                        key: dict(state.evidence)
                        for key, state in self._state_ledger.snapshot(robot).items()
                        if key.startswith(("holding:", "supporting:"))
                    }
                )
        return replace(
            coordinated,
            physical_facts=self._refresh_physical_facts(request.scene_snapshot, facts),
        )

    async def plan_free_space(
        self, request: FreeSpaceMotionRequest
    ) -> CoordinatedSessionResult:
        """Evaluate a candidate in the existing plan-only worker lane.

        This may refresh the measured MoveIt world, but never prepares or
        sends controller goals. Selected targets are revalidated/replanned
        from fresh measured state by normal execute_free_space afterward.

        Returns:
            Strict no-motion planning evidence, not arrival or custody.
        """
        if not isinstance(request, FreeSpaceMotionRequest):
            raise TypeError("plan_free_space requires FreeSpaceMotionRequest")
        coordinated = self._free_space_request(request, plan_only=True)
        plan = getattr(self._session, "execute_successor_pre_phase", None)
        if not callable(plan):
            raise RuntimeError("combined plan-only transport is unavailable")
        result = await asyncio.to_thread(plan, coordinated)
        if not isinstance(result, CoordinatedSessionResult) or (
            result.coordination_id != coordinated.coordination_id
            or result.robots != coordinated.robots
            or result.participants != coordinated.participants
            or result.lifecycle.get("plan_only") is not True
            or result.lifecycle.get("physical_execution") is not False
            or result.lifecycle.get("started") is not False
            or result.lifecycle.get("prepared") is not False
            or result.lifecycle.get("terminal") is not True
        ):
            raise RuntimeError(
                "candidate plan returned unbound or physical execution evidence"
            )
        return result

    async def execute_free_space(
        self, request: FreeSpaceMotionRequest
    ) -> FreeSpaceMotionEvidence:
        """Execute one skill-neutral physical free-space request.

        Returns:
            Strict bound physical completion, or explicit failed admission.
        """
        if not isinstance(request, FreeSpaceMotionRequest):
            raise TypeError("execute_free_space requires FreeSpaceMotionRequest")
        try:
            coordinated = self._free_space_request(request, plan_only=False)
            execute = getattr(self._session, "execute_coordinated_pre_phase", None)
            if not callable(execute):
                raise RuntimeError("combined free-space transport is unavailable")
            result = await asyncio.to_thread(execute, coordinated)
            lifecycle = result.lifecycle
            raw_participants = lifecycle.get("participants")
            participant_evidence = {
                robot: (
                    dict(raw_participants[robot])
                    if isinstance(raw_participants, Mapping)
                    and isinstance(raw_participants.get(robot), Mapping)
                    else {}
                )
                for robot in request.participants
            }
            missing_participants = [
                robot for robot, item in participant_evidence.items() if not item
            ]
            passed = bool(
                result.passed
                and lifecycle.get("prepared") is True
                and lifecycle.get("started") is True
                and lifecycle.get("terminal") is True
                and lifecycle.get("safe_to_release") is True
                and set(participant_evidence) == set(request.participants)
                and all(
                    item.get("arrival_confirmed") is True
                    for item in participant_evidence.values()
                )
            )
            return FreeSpaceMotionEvidence(
                motion_id=request.motion_id,
                participants=request.participants,
                planned=bool(result.trajectory),
                prepared=lifecycle.get("prepared") is True,
                started=lifecycle.get("started") is True,
                arrival_confirmed=passed,
                terminal=lifecycle.get("terminal") is True,
                safe_to_release=lifecycle.get("safe_to_release") is True,
                participant_evidence=participant_evidence,
                evidence={
                    "lifecycle": dict(lifecycle),
                    "trajectory": dict(result.trajectory),
                },
                reason=(
                    None
                    if passed
                    else str(
                        result.error
                        or lifecycle.get("error")
                        or (
                            f"missing participant evidence: {missing_participants}"
                            if missing_participants
                            else "free-space lifecycle did not pass"
                        )
                    )
                ),
            )
        except Exception as error:
            return FreeSpaceMotionEvidence(
                motion_id=request.motion_id,
                participants=request.participants,
                planned=False,
                prepared=False,
                started=False,
                arrival_confirmed=False,
                terminal=False,
                safe_to_release=False,
                participant_evidence={robot: {} for robot in request.participants},
                evidence={},
                reason=f"free-space dispatch failed: {error}",
            )

    async def cancel_free_space(self, motion_id: str) -> Mapping[str, object]:
        """Cancel one free-space request only through an explicit session seam.

        Returns:
            Independent physical cancellation proof or an unsafe refusal.
        """
        cancel = getattr(self._session, "cancel_free_space", None)
        if not callable(cancel):
            return {
                "motion_id": motion_id,
                "attempted": False,
                "terminal": False,
                "safe": False,
                "error": "free-space cancellation boundary is unavailable",
            }
        return await asyncio.to_thread(cancel, motion_id)

    async def execute_pre_phase(
        self, request: SuccessorPrePhaseRequest
    ) -> SuccessorPrePhaseEvidence:
        """Dispatch the successor's free-space pre-phase as combined motion.

        Returns:
            Evidence that authorizes lease adoption only after every
            participant proved prepared/started/terminal-safe arrival.
        """
        if (
            request.next_operation is not None
            and request.next_operation.skill == "HANDOVER"
        ):
            if self._handover_continuation is None:
                return failed_pre_phase(
                    request, "stationary HANDOVER custody admission unavailable"
                )
            return await self._handover_continuation.admit_stationary(request)
        if request.terminal_cleanup:
            return failed_pre_phase(
                request, "terminal cleanup execution is not implemented"
            )
        if not self._contexts or not request.successor_operations:
            return failed_pre_phase(
                request, "pre-phase requires a motion successor and contexts"
            )
        try:
            shared = await asyncio.to_thread(self._session.observe_shared_scene)
            snapshot = shared["scene_snapshot"]
            if "scene_digest" not in snapshot:
                snapshot = await asyncio.to_thread(self._session.observe_all)
            start_state = shared.get("start_state", {})
            document = request.scene_snapshot.document
            robots = document.get("robots")
            facts = (
                {
                    robot: dict(block.get("facts") or {})
                    for robot, block in (robots or {}).items()
                    if robot in request.participants
                }
                if isinstance(robots, Mapping)
                else {}
            )
            facts = self._refresh_physical_facts(snapshot, facts)
            facts = {robot: facts.get(robot, {}) for robot in request.participants}
            operations = request.successor_operations
            operation = operations[0]
            operation_document = asdict(operation)
            if len(operations) > 1:
                operation_document["cohort_operations"] = [
                    asdict(item) for item in operations[1:]
                ]
            coordinated = CoordinatedSessionRequest(
                coordination_id=request.transition_id,
                robots=tuple(sorted(self._contexts)),
                participants=request.participants,
                scene_digest=str(snapshot["scene_digest"]),
                planning_group="dual_arm",
                plan_only=False,
                timeout_s=60.0,
                targets={},
                scene_snapshot=snapshot,
                scene_version=shared["planning_scene_version"],
                start_state=start_state,
                physical_facts=facts or {robot: {} for robot in request.participants},
                operation=operation_document,
                target_contexts={
                    robot: asdict(context)
                    for robot, context in self._contexts.items()
                    if robot in request.participants
                },
            )
            execute = getattr(self._session, "execute_coordinated_pre_phase", None)
            if not callable(execute):
                return failed_pre_phase(
                    request,
                    "combined physical pre-phase transport is unavailable",
                )
            result = await asyncio.to_thread(execute, coordinated)
            if (
                not isinstance(result, CoordinatedSessionResult)
                or result.coordination_id != request.transition_id
                or result.participants != request.participants
                or result.robots != coordinated.robots
            ):
                raise RuntimeError("combined pre-phase result identity mismatch")
        except Exception as error:
            return failed_pre_phase(
                request, f"combined pre-phase dispatch failed: {error}"
            )
        lifecycle = result.lifecycle
        participants = lifecycle.get("participants") or {}
        arrival = set(participants) == set(request.participants) and all(
            isinstance(item, Mapping) and item.get("arrival_confirmed") is True
            for item in participants.values()
        )
        passed = bool(
            result.passed
            and lifecycle.get("prepared") is True
            and lifecycle.get("started") is True
            and lifecycle.get("terminal") is True
            and lifecycle.get("safe_to_release") is True
        )
        final_facts = {
            robot: {
                key: proof
                for key, proof in (facts.get(robot) or {}).items()
                if key.startswith(("holding:", "supporting:"))
            }
            for robot in request.participants
        }
        continuity_error = None
        if passed and arrival:
            try:
                final_snapshot = await asyncio.to_thread(self._session.observe_all)
                final_facts = self._refresh_physical_facts(final_snapshot, final_facts)
            except (ValueError, RuntimeError, KeyError) as error:
                continuity_error = str(error)
                passed = False
        return SuccessorPrePhaseEvidence(
            transition_id=request.transition_id,
            predecessor_operation_ids=request.predecessor_operation_ids,
            successor_operation_ids=request.successor_operation_ids,
            participants=request.participants,
            resources=request.resources,
            lease_owners=request.lease_owners,
            scene_digest=request.scene_digest,
            scene_version=request.scene_version,
            planned=bool(result.trajectory),
            prepared=lifecycle.get("prepared") is True,
            started=lifecycle.get("started") is True,
            arrival_confirmed=passed and arrival,
            terminal=lifecycle.get("terminal") is True,
            safe=lifecycle.get("safe_to_release") is True,
            adoption_ready=passed and arrival,
            final_facts=final_facts,
            evidence={
                "lifecycle": dict(lifecycle),
                "trajectory": dict(result.trajectory),
                "combined_execution_passed": result.passed,
            },
            terminal_cleanup=False,
            reason=None
            if passed and arrival
            else str(
                continuity_error
                or result.error
                or lifecycle.get("error")
                or "participant arrival not confirmed"
            ),
        )

    async def validate_pre_phase(
        self, request: SuccessorPrePhaseRequest, evidence: SuccessorPrePhaseEvidence
    ) -> bool:
        """Confirm the live shared binding still matches the admitted request.

        Returns:
            True only for passed bound evidence while one fresh admitted
            observation keeps the same session/manifest provenance and the
            admitted planning-scene version. A fresher scene digest is
            expected (shared time advances) and is not a mismatch; the
            admitted digest stays bound to the request.
        """
        if not evidence.matches(request) or not evidence.passed:
            return False
        if evidence.transition_mode == "stationary_handover":
            if self._handover_continuation is None:
                return False
            return await self._handover_continuation.validate_stationary(
                request, evidence
            )
        try:
            snapshot, version, observed = await asyncio.to_thread(self._read_shared)
            self._admit(snapshot, version, observed, request.participants)
        except Exception:
            return False
        document = request.scene_snapshot.document
        trajectory = evidence.evidence.get("trajectory", {})
        planning = (
            trajectory.get("planning_scene", {})
            if isinstance(trajectory, Mapping)
            else {}
        )
        expected_version = (
            planning.get("scene_version", request.scene_version)
            if isinstance(planning, Mapping)
            else request.scene_version
        )
        if isinstance(trajectory, Mapping):
            expected_version = trajectory.get(
                "planning_scene_version", expected_version
            )
        return (
            snapshot.get("session_id") == document.get("session_id")
            and snapshot.get("manifest_sha256") == document.get("manifest_sha256")
            and version == expected_version
        )

    async def cancel_pre_phase(self, transition_id: str) -> Mapping[str, object]:
        """Cancel physical participants through the existing session boundary.

        Returns:
            The actual session's independent cancellation/terminal proof.
        """
        result = await self.cancel_free_space(transition_id)
        return {**result, "transition_id": transition_id}

    def _refresh_physical_facts(self, snapshot, facts):
        """Carry measured attachments and recheck custody/support across motion.

        Returns:
            A fresh copy of the retained facts, never newly inferred custody.
        """
        refreshed = {robot: dict(block) for robot, block in facts.items()}
        for robot, block in refreshed.items():
            for key, proof in tuple(block.items()):
                if key.startswith("holding:"):
                    identity = key.removeprefix("holding:")
                    sample = parse_pick_observation(
                        snapshot,
                        robot_id=robot,
                        physical_instance_id=identity,
                        session_id=snapshot["session_id"],
                        manifest_sha256=snapshot["manifest_sha256"],
                    )
                    reference = proof.get("continuation_observation") or proof.get(
                        "lift_start_observation"
                    )
                    if isinstance(reference, Mapping):
                        old = compose(
                            inverse(_pose(reference["object_pose"], "held reference")),
                            _pose(reference["tcp_pose"], "held TCP reference"),
                        )
                        live = compose(inverse(sample.object_pose), sample.tcp_pose)
                        position, angle = pose_error(live, old)
                        context = self._contexts.get(robot)
                        if context is not None and (
                            position > context.params.hold_follow_tolerance_m
                            or angle > context.params.holding_orientation_tolerance_rad
                        ):
                            raise ValueError(
                                f"{robot} lost measured holding continuity"
                            )
                    minimum, maximum = _tcp_relative_aabb(sample)
                    payload = dict(proof.get("attachment") or {})
                    payload.update(
                        {
                            "object_id": identity,
                            "frame_id": f"{robot}_rocobrick_tcp",
                            "link_name": f"{robot}_rocobrick_tcp",
                            "minimum_m": list(minimum),
                            "maximum_m": list(maximum),
                            "geometry_digest": box_from_aabb(
                                object_id=identity,
                                frame_id=f"{robot}_rocobrick_tcp",
                                minimum_m=minimum,
                                maximum_m=maximum,
                            ).geometry_digest,
                            "touch_links": [
                                f"{robot}_gripper_link1",
                                f"{robot}_gripper_link2",
                            ],
                        }
                    )
                    block[key] = {
                        **proof,
                        "attachment": payload,
                        "continuation_observation": asdict(sample),
                    }
                elif key.startswith("supporting:"):
                    identity = key.removeprefix("supporting:")
                    sample, contact = verify_support_contact(
                        snapshot,
                        robot_id=robot,
                        upper_id=identity,
                        session_id=snapshot["session_id"],
                        manifest_sha256=snapshot["manifest_sha256"],
                    )
                    if isinstance(proof.get("upper_pose"), Mapping):
                        position, angle = pose_error(
                            sample.object_pose,
                            _pose(proof["upper_pose"], "support reference"),
                        )
                        if position > 0.002 or angle > 0.05236:
                            raise ValueError(
                                "supported upper deflected before continuation"
                            )
                    block[key] = {**proof, "support_contact_confirmed": contact}
        return refreshed

    def terminal_pre_phase_policy(self, operations) -> TerminalPrePhasePolicy | None:
        """Return the explicit terminal departure policy, never an inferred one.

        Returns:
            None until an authorized external policy source exists; the
            scheduler fails closed without it.
        """
        return None

    def _read_shared(self) -> tuple[Mapping[str, object], int, float]:
        """Return the raw shared snapshot, planning-scene version and clock."""
        shared = self._session.observe_shared_scene()
        if not isinstance(shared, Mapping):
            raise RuntimeError("shared observation answer is invalid")
        snapshot = shared.get("scene_snapshot")
        version = shared.get("planning_scene_version")
        observed = shared.get("observed_sim_time_s")
        if not isinstance(snapshot, Mapping):
            raise RuntimeError("shared observation returned no scene snapshot")
        if type(version) is not int or version < 0:
            raise RuntimeError("shared observation planning-scene version is invalid")
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise RuntimeError("shared observation carries no simulation-time sample")
        return snapshot, version, float(observed)

    def _admit(
        self,
        snapshot: Mapping[str, object],
        version: int,
        observed: float,
        participants: tuple[str, ...],
    ) -> SharedSceneSnapshot:
        """Bind one raw shared observation to authorized admitted robot state.

        Returns:
            The admitted observation with exact current physical facts.
        """
        if self._state_ledger is None:
            raise RuntimeError("shared snapshot admission requires the state ledger")
        robots = snapshot.get("robots")
        if not isinstance(robots, Mapping) or not robots:
            raise RuntimeError("shared observation carries no robot states")
        if not set(participants) <= set(robots):
            raise RuntimeError("shared observation does not cover all participants")
        readiness = getattr(self._session, "readiness", None)
        reports = getattr(readiness, "readiness_by_robot", None)
        authorized = (
            getattr(readiness, "passed", False) is True
            and isinstance(reports, Mapping)
            and set(reports) >= set(robots)
            and all(
                isinstance(reports.get(robot), Mapping)
                and reports[robot].get("passed") is True
                for robot in robots
            )
        )
        if not authorized:
            raise RuntimeError(
                "shared snapshot admission requires every robot's readiness "
                "authorization"
            )
        admitted_robots: dict[str, Mapping[str, object]] = {}
        for robot, block in robots.items():
            if not isinstance(block, Mapping):
                raise RuntimeError(f"shared observation block for {robot!r} is invalid")
            uncertain = self._state_ledger.uncertain_keys(robot)
            if uncertain:
                raise RuntimeError(
                    f"robot {robot!r} has uncertain physical state: {sorted(uncertain)}"
                )
            facts = {
                key: dict(state.evidence)
                for key, state in self._state_ledger.snapshot(robot).items()
            }
            admitted_robots[robot] = {
                **block,
                "authorized": True,
                # The worker serves a shared observation only for an idle,
                # ownership-free participant group, so no motion is in flight.
                "terminal_safe": True,
                "holding_known": True,
                "supporting_known": True,
                "facts": facts,
            }
        document = {**snapshot, "scene_version": version, "robots": admitted_robots}
        return SharedSceneSnapshot(document, observed, SHARED_SNAPSHOT_MAX_AGE_S)


__all__ = [
    "SHARED_SNAPSHOT_MAX_AGE_S",
    "UnifiedDualArmRuntimeFacade",
]
