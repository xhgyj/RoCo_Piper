"""Host transport facade for bound successor pre-phase requests."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict

from rocobrick.coordinated_session import (
    CoordinatedSessionRequest,
    FreeSpaceMotionEvidence,
    FreeSpaceMotionRequest,
)
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.unified_dual_arm_runtime import (
    SharedSceneSnapshot,
    SuccessorPrePhaseEvidence,
    SuccessorPrePhaseRequest,
    TerminalPrePhasePolicy,
    failed_pre_phase,
)

SHARED_SNAPSHOT_MAX_AGE_S = 0.5


class UnifiedDualArmRuntimeFacade:
    """Dispatch combined planning without claiming physical execution."""

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
        """Read shared scene/version and the complete combined start state."""
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
        return snapshot, version, {
            str(name): float(value) for name, value in start_state.items()
        }

    async def execute_free_space(
        self, request: FreeSpaceMotionRequest
    ) -> FreeSpaceMotionEvidence:
        """Execute one skill-neutral physical free-space request."""
        if not isinstance(request, FreeSpaceMotionRequest):
            raise TypeError("execute_free_space requires FreeSpaceMotionRequest")
        try:
            coordinated = request.to_coordinated_request(plan_only=False)
            robots = request.scene_snapshot.get("robots")
            if isinstance(robots, Mapping):
                facts = {
                    robot: dict((robots.get(robot) or {}).get("facts") or {})
                    for robot in request.participants
                    if isinstance(robots.get(robot), Mapping)
                }
            else:
                facts = {}
            for robot, carried in request.physical_facts.items():
                facts.setdefault(robot, {}).update(dict(carried))
            if self._state_ledger is not None:
                for robot in request.participants:
                    confirmed = {
                        key: dict(state.evidence)
                        for key, state in self._state_ledger.snapshot(robot).items()
                        if key.startswith(("holding:", "supporting:"))
                    }
                    if confirmed:
                        facts.setdefault(robot, {}).update(confirmed)
            coordinated = CoordinatedSessionRequest(
                coordination_id=coordinated.coordination_id,
                robots=coordinated.robots,
                participants=coordinated.participants,
                scene_digest=coordinated.scene_digest,
                planning_group=coordinated.planning_group,
                plan_only=False,
                timeout_s=coordinated.timeout_s,
                targets=coordinated.targets,
                scene_snapshot=coordinated.scene_snapshot,
                scene_version=coordinated.scene_version,
                physical_facts=facts,
                start_state=coordinated.start_state,
            )
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
        """Cancel one free-space request only through an explicit session seam."""
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
        if request.terminal_cleanup:
            return failed_pre_phase(
                request, "terminal cleanup execution is not implemented"
            )
        if not self._contexts or not request.successor_operations:
            return failed_pre_phase(
                request, "pre-phase requires a motion successor and contexts"
            )
        try:
            snapshot = await asyncio.to_thread(self._session.observe_all)
            document = request.scene_snapshot.document
            robots = document.get("robots")
            facts = {
                robot: dict(block.get("facts") or {})
                for robot, block in (robots or {}).items()
            } if isinstance(robots, Mapping) else {}
            operation = request.successor_operations[0]
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
                physical_facts=facts or {
                    robot: {} for robot in request.participants
                },
                operation=asdict(operation),
                target_contexts={
                    robot: asdict(context)
                    for robot, context in self._contexts.items()
                    if robot in request.participants
                },
            )
            execute = getattr(
                self._session, "execute_coordinated_pre_phase", None
            )
            if not callable(execute):
                return failed_pre_phase(
                    request,
                    "combined physical pre-phase transport is unavailable",
                )
            result = await asyncio.to_thread(execute, coordinated)
        except Exception as error:
            return failed_pre_phase(
                request, f"combined pre-phase dispatch failed: {error}"
            )
        lifecycle = result.lifecycle
        participants = lifecycle.get("participants") or {}
        arrival = bool(participants) and all(
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
                "plan_only_admission": result.passed,
            },
            terminal_cleanup=False,
            reason=None if passed else str(result.error or lifecycle.get("error")),
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
        try:
            snapshot, version, observed = await asyncio.to_thread(self._read_shared)
            self._admit(snapshot, version, observed, request.participants)
        except Exception:
            return False
        document = request.scene_snapshot.document
        return (
            snapshot.get("session_id") == document.get("session_id")
            and snapshot.get("manifest_sha256") == document.get("manifest_sha256")
            and version == request.scene_version
        )

    async def cancel_pre_phase(self, transition_id: str) -> Mapping[str, object]:
        """Report that this plan-only facade has not dispatched controller motion.

        Returns:
            Explicit no-motion cancellation evidence.
        """
        return {
            "transition_id": transition_id,
            "attempted": False,
            "terminal": True,
            "safe": True,
            "motion_sent": False,
        }

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
        """Bind one raw shared observation to authorized admitted robot state."""
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
