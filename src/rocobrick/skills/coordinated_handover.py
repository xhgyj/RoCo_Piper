"""Complete bottom-HANDOVER lifecycle, independent of legacy PICK targeting.

Each dispatched phase is admitted before dispatch. Transport receipts bind certified
execution, but custody comes only from fresh measured pad/SE(3) observations.
The production profile remains fail-closed until its hardware qualification.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Protocol

from rocobrick.adapters.moveit.handover_scene_planner import HandoverPhasePaths
from rocobrick.coordinated_moveit import CoordinatedOperation
from rocobrick.execution.operation_state_ledger import OperationStateLedger
from rocobrick.multi_robot_scheduler import (
    JSONValue,
    OperationExecution,
    OperationStatus,
    to_json_value,
)
from rocobrick.robot_operation_plan import RobotOperation
from rocobrick.skills.contracts import HandoverEvidence, HandoverTransferIdentity
from rocobrick.skills.handover_contact import CONTACT_PHASES
from rocobrick.skills.handover_observation import (
    HandoverObservation,
    parse_handover_observation,
    require_giver_retention_sample,
    require_receiver_retention_sample,
    verify_co_hold,
    verify_giver_holding,
    verify_receiver_follow,
    verify_receiver_retention,
)
from rocobrick.skills.handover_planning import (
    HandoverCandidate,
    HandoverPlanningRequest,
)
from rocobrick.skills.handover_profile import (
    digest,
    finite,
    require_handover_capabilities,
)
from rocobrick.skills.handover_simulation import SimulationHandoverAuthorization
from rocobrick.skills.observation import Pose
from rocobrick.skills.placement_geometry import compose, inverse, pose_error

PHASES = (
    "receiver_open",
    "rendezvous",
    "approach",
    "receiver_close",
    "donor_open",
    "scene_transfer",
    "separation",
    "receiver_follow",
)


@dataclass(frozen=True)
class CoordinatedHandoverPreparation:
    """Simulation-only paired preparation, not final-contact permission."""

    request: HandoverPlanningRequest
    rendezvous: CoordinatedOperation
    receiver_open: tuple[Mapping[str, float], Mapping[str, float]]
    capabilities: frozenset[str]
    donor_holding_window: tuple[float, float]
    phase_timeout_s: float
    simulation_authorization: SimulationHandoverAuthorization
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Bind only safe opening and the paired carrying trajectory."""
        if not isinstance(
            self.simulation_authorization, SimulationHandoverAuthorization
        ):
            raise ValueError(
                "staged preparation requires explicit simulation authorization"
            )
        self.simulation_authorization.validate(
            self.request.profile, self.request.scene_snapshot, self.capabilities
        )
        finite(self.phase_timeout_s, "preparation timeout", positive=True)
        if (
            not isinstance(self.donor_holding_window, tuple)
            or len(self.donor_holding_window) != 2
        ):
            raise ValueError("preparation requires a calibrated giver window")
        lo, hi = self.donor_holding_window
        if (
            not 0
            <= finite(lo, "giver opening minimum")
            <= finite(hi, "giver opening maximum")
        ):
            raise ValueError("invalid preparation giver window")
        if len(self.receiver_open) != 2 or dict(self.receiver_open[0]) != dict(
            self.request.start_state
        ):
            raise ValueError(
                "preparation opening must start at captured measured state"
            )
        for state in self.receiver_open:
            if set(state) != set(self.request.start_state):
                raise ValueError("preparation requires the complete measured state")
            for name, value in state.items():
                finite(value, f"preparation joint {name}")
                if (
                    not name.startswith(self.request.receiver.robot_id + "_gripper_")
                    and value != self.request.start_state[name]
                ):
                    raise ValueError(
                        "preparation opening changes an arm or giver finger"
                    )
        object.__setattr__(
            self,
            "receiver_open",
            tuple(MappingProxyType(dict(s)) for s in self.receiver_open),
        )
        plan = self.rendezvous
        if (
            plan.participants != self.participants
            or plan.scene_digest != self.request.scene_snapshot["scene_digest"]
            or len(plan.trajectories) != 2
        ):
            raise ValueError("preparation trajectory identity mismatch")
        if plan.synchronization_tolerance_s > self.request.profile.max_sync_skew_s:
            raise ValueError("preparation synchronization tolerance exceeds policy")
        grids = []
        for trajectory in plan.trajectories:
            if trajectory.robot_id not in self.participants or set(
                trajectory.joint_names
            ) != {f"{trajectory.robot_id}_joint{i}" for i in range(1, 7)}:
                raise ValueError("preparation omits one complete arm")
            grid = tuple(p.time_from_start_s for p in trajectory.points)
            if (
                len(grid) < 2
                or grid[0] != 0
                or grid[-1] >= self.phase_timeout_s
                or any(b <= a for a, b in zip(grid, grid[1:]))
            ):
                raise ValueError("preparation lacks a bounded shared time grid")
            grids.append(grid)
            if trajectory.points[0].positions != tuple(
                self.request.start_state[n] for n in trajectory.joint_names
            ):
                raise ValueError("preparation changed captured arm start")
        if {t.robot_id for t in plan.trajectories} != set(self.participants) or grids[
            0
        ] != grids[1]:
            raise ValueError("preparation requires both arms on one time grid")
        object.__setattr__(self, "binding_sha256", self.current_binding())

    @property
    def participants(self) -> tuple[str, str]:
        """Return the exact giver and receiver."""
        return self.request.donor.robot_id, self.request.receiver.robot_id

    @property
    def identity(self) -> HandoverTransferIdentity:
        """Preparation never changes physical ownership."""
        return HandoverTransferIdentity(
            self.request.operation_id,
            *self.participants,
            self.request.physical_instance_id,
        )

    @property
    def object_to_donor_tcp(self) -> Pose:
        """Keep the measured original grasp throughout paired preparation."""
        observed = self.request.observation(self.request.donor.robot_id)
        return compose(inverse(observed.object_pose), observed.tcp_pose)

    def current_binding(self) -> str:
        """Detect mutation without inventing a final contact candidate.

        Returns:
            The current immutable-input and carrying-trajectory binding.
        """
        raw = {
            "request": self.request.binding_sha256,
            "rendezvous": asdict(self.rendezvous),
            "receiver_open": [dict(s) for s in self.receiver_open],
            "donor_window": self.donor_holding_window,
            "timeout": self.phase_timeout_s,
            "capabilities": sorted(self.capabilities),
            "simulation_authorization": asdict(self.simulation_authorization),
        }
        return hashlib.sha256(
            json.dumps(raw, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


@dataclass(frozen=True)
class CoordinatedHandoverProgram:
    """A completely certified paired program plus calibrated opening windows."""

    request: HandoverPlanningRequest
    candidate: HandoverCandidate
    rendezvous: CoordinatedOperation | None
    paths: HandoverPhasePaths
    capabilities: frozenset[str]
    donor_holding_window: tuple[float, float]
    receiver_holding_window: tuple[float, float]
    donor_release_window: tuple[float, float]
    phase_timeout_s: float
    simulation_authorization: SimulationHandoverAuthorization | None = None
    preparation_binding_sha256: str | None = None
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Bind every path and policy, without enabling an unqualified profile."""
        if self.simulation_authorization is None:
            require_handover_capabilities(
                self.request.profile, self.capabilities, physical=True
            )
        else:
            self.simulation_authorization.validate(
                self.request.profile, self.request.scene_snapshot, self.capabilities
            )
        if self.candidate.binding_sha256 != self.request.binding_sha256 or (
            self.paths.candidate_id != self.candidate.candidate_id
            or (
                self.rendezvous is not None
                and (
                    self.rendezvous.coordination_id != self.candidate.candidate_id
                    or set(self.rendezvous.participants) != set(self.participants)
                    or self.rendezvous.scene_digest
                    != self.request.scene_snapshot["scene_digest"]
                )
            )
        ):
            raise ValueError("HANDOVER program/candidate/path identity mismatch")
        if self.rendezvous is None:
            if self.simulation_authorization is None:
                raise ValueError(
                    "arrival continuation requires simulation authorization"
                )
            digest(self.preparation_binding_sha256, "executed preparation binding")
        elif self.preparation_binding_sha256 is not None:
            raise ValueError("arrival continuation must not redispatch preparation")
        self._validate_phase_chain()
        finite(self.phase_timeout_s, "phase timeout", positive=True)
        for window in (
            self.donor_holding_window,
            self.receiver_holding_window,
            self.donor_release_window,
        ):
            if (
                not isinstance(window, tuple)
                or len(window) != 2
                or (
                    not 0
                    <= finite(window[0], "opening minimum")
                    <= finite(window[1], "opening maximum")
                )
            ):
                raise ValueError("HANDOVER needs calibrated opening windows")
        object.__setattr__(self, "binding_sha256", self.current_binding())

    def _validate_phase_chain(self) -> None:
        """Require a common twelve-joint grid and continuous phase boundaries."""
        paths = self.paths
        names = tuple(f"{r}_joint{i}" for r in self.participants for i in range(1, 7))
        for phase in PHASES:
            if phase in {"rendezvous", "scene_transfer"}:
                continue
            states = getattr(paths, phase)
            for state in states:
                if set(state) != set(self.request.start_state):
                    raise ValueError(
                        "HANDOVER phase omits measured full-state variables"
                    )
                for name, value in state.items():
                    finite(value, f"phase joint {name}")
        if dict(paths.receiver_open[0]) != dict(self.request.start_state):
            raise ValueError("receiver opening does not start at captured lift state")
        grid = None
        goals = {}
        starts = {}
        for trajectory in (
            self.rendezvous.trajectories if self.rendezvous is not None else ()
        ):
            expected = {f"{trajectory.robot_id}_joint{i}" for i in range(1, 7)}
            if (
                set(trajectory.joint_names) != expected
                or len(trajectory.joint_names) != 6
            ):
                raise ValueError("rendezvous omits a complete identified arm")
            times = tuple(p.time_from_start_s for p in trajectory.points)
            if (
                len(times) < 2
                or times[0] != 0
                or (
                    any(b <= a for a, b in zip(times, times[1:]))
                    or times[-1] >= self.phase_timeout_s
                    or grid not in (None, times)
                )
            ):
                raise ValueError("rendezvous lacks one bounded common time grid")
            grid = times
            starts.update(zip(trajectory.joint_names, trajectory.points[0].positions))
            goals.update(zip(trajectory.joint_names, trajectory.points[-1].positions))
        if self.rendezvous is None:
            starts = {n: self.request.start_state[n] for n in names}
            goals = dict(starts)
        for name in names:
            if starts[name] != self.request.start_state[name] or (
                goals[name] != paths.approach[0][name]
                or paths.receiver_open[1][name] != self.request.start_state[name]
            ):
                raise ValueError(
                    "rendezvous changed the admitted start/approach binding"
                )
        for first, second in (
            (paths.approach, paths.receiver_close),
            (paths.receiver_close, paths.donor_open),
            (paths.donor_open, paths.separation),
            (paths.separation, paths.receiver_follow),
        ):
            if dict(first[-1]) != dict(second[0]):
                raise ValueError("HANDOVER phase chain has a discontinuous full state")
        for states, held_robot in (
            (paths.approach, self.request.donor.robot_id),
            (paths.separation, self.request.receiver.robot_id),
            (paths.receiver_follow, self.request.donor.robot_id),
        ):
            for state in states:
                if any(
                    state[f"{held_robot}_joint{i}"]
                    != states[0][f"{held_robot}_joint{i}"]
                    for i in range(1, 7)
                ):
                    raise ValueError("HANDOVER constrained phase moves the held arm")

    @property
    def object_to_donor_tcp(self) -> Pose:
        """Expose the same grasp guard for complete and preparation programs."""
        return self.candidate.object_to_donor_tcp

    @property
    def participants(self) -> tuple[str, str]:
        """Return exact giver and receiver, never canonical defaults.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return self.request.donor.robot_id, self.request.receiver.robot_id

    @property
    def identity(self) -> HandoverTransferIdentity:
        """Return a physical transfer identity without a fake legacy request.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        return HandoverTransferIdentity(
            self.request.operation_id,
            *self.participants,
            self.request.physical_instance_id,
        )

    def current_binding(self) -> str:
        """Detect nested input/trajectory mutation throughout execution.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        paths = {
            name: [dict(state) for state in getattr(self.paths, name)]
            for name in PHASES
            if name not in {"rendezvous", "scene_transfer"}
        }
        raw = {
            "request": self.request.binding_sha256,
            "candidate": self.candidate.candidate_id,
            "paths": paths,
            "rendezvous": asdict(self.rendezvous)
            if self.rendezvous is not None
            else None,
            "preparation_binding_sha256": self.preparation_binding_sha256,
            "attachment": {
                "position": self.paths.planned_receiver_attachment.position_xyz,
                "orientation": self.paths.planned_receiver_attachment.orientation_xyzw,
            },
            "windows": [
                self.donor_holding_window,
                self.receiver_holding_window,
                self.donor_release_window,
            ],
            "capabilities": sorted(self.capabilities),
            "timeout": self.phase_timeout_s,
            "simulation_authorization": (
                asdict(self.simulation_authorization)
                if self.simulation_authorization is not None
                else None
            ),
        }
        return hashlib.sha256(
            json.dumps(raw, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


@dataclass(frozen=True)
class HandoverPhaseReceipt:
    """Strict receipt from a certified path/gripper/scene adapter."""

    operation_id: str
    program_binding_sha256: str
    phase: str
    passed: bool
    terminal: bool
    safe_to_release: bool
    evidence: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        """Never equate goal acceptance with physical completion."""
        digest(self.program_binding_sha256, "phase program")
        if not self.operation_id or self.phase not in PHASES:
            raise ValueError("invalid HANDOVER phase identity")
        if any(
            type(v) is not bool
            for v in (self.passed, self.terminal, self.safe_to_release)
        ):
            raise ValueError("phase lifecycle must contain strict booleans")
        object.__setattr__(self, "evidence", to_json_value(self.evidence))


class CoordinatedHandoverTransport(Protocol):
    """Already-running shared session; implementations never replan phases."""

    async def observe(
        self, program: CoordinatedHandoverProgram | CoordinatedHandoverPreparation
    ) -> Mapping[str, object]:
        """Read current same-tick physics even while both actions are active."""
        ...

    async def execute_phase(
        self,
        program: CoordinatedHandoverProgram | CoordinatedHandoverPreparation,
        phase: str,
        receiver_attachment: Pose | None,
    ) -> HandoverPhaseReceipt:
        """Execute the admitted phase, recertifying measured finger deviations."""
        ...

    async def cancel(
        self, program: CoordinatedHandoverProgram | CoordinatedHandoverPreparation
    ) -> Mapping[str, object]:
        """Stop BOTH participants and return independently terminal-safe proof."""
        ...


class CoordinatedHandoverFactory(Protocol):
    """Admit each stage before its dispatch, using measured preparation arrival."""

    async def prepare(
        self, operation: RobotOperation
    ) -> CoordinatedHandoverProgram | CoordinatedHandoverPreparation:
        """Return independently admitted preparation or a complete program."""
        ...

    async def complete_preparation(
        self, operation: RobotOperation, preparation: CoordinatedHandoverPreparation
    ) -> CoordinatedHandoverProgram:
        """Certify contact only AFTER physical paired preparation has arrived."""
        ...


class CoordinatedHandoverBackend:
    """Execute/monitor all transfer phases and atomically commit final custody."""

    def __init__(
        self,
        factory: CoordinatedHandoverFactory,
        transport: CoordinatedHandoverTransport,
        ledger: OperationStateLedger,
    ) -> None:
        """Bind owned services; constructing the backend never starts a runtime."""
        self.factory, self.transport, self.ledger = factory, transport, ledger
        self._active: dict[str, _Execution] = {}

    async def execute(self, operation: RobotOperation) -> OperationExecution:
        """Run an exact external HANDOVER with no synthesized PICK or lift.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        if operation.operation_id in self._active:
            raise ValueError("HANDOVER already has an owned execution")
        state = _Execution(operation, self.transport, self.ledger, self.factory)
        self._active[operation.operation_id] = state
        try:
            state.program = await self.factory.prepare(operation)
            state.validate_operation()
            return await state.execute()
        except (Exception, asyncio.CancelledError) as error:
            return await state.fail(error)
        finally:
            if state.inflight is None or state.inflight.done():
                self._active.pop(operation.operation_id, None)

    async def cancel(self, operation_id: str) -> Mapping[str, object]:
        """Request physical cancellation, never cancel an asyncio motion task.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        state = self._active.get(operation_id)
        if state is None:
            return {
                "operation_id": operation_id,
                "terminal": False,
                "safe_to_release": False,
                "reason": "no owned HANDOVER",
            }
        state.cancel_requested = True
        proof = dict(await state.stop())
        if (
            state.motion_attempted or state.custody_monitored
        ) and state.program is not None:
            state.ledger.mark_handover_uncertain(
                state.program.identity,
                "HANDOVER_CANCELLED",
                to_json_value(proof),
            )
            proof["safe_to_release"] = False
            proof["custody_unknown"] = True
        return proof

    async def execute_after_preparation(
        self,
        operation: RobotOperation,
        preparation: CoordinatedHandoverPreparation,
        completed: OperationExecution,
    ) -> OperationExecution:
        """Continue an executed paired arrival without redispatching rendezvous.

        Returns:
            The formal continuation result, including custody evidence.
        """
        if operation.operation_id in self._active:
            raise ValueError("HANDOVER already has an owned execution")
        state = _Execution(operation, self.transport, self.ledger, self.factory)
        self._active[operation.operation_id] = state
        try:
            if (
                not isinstance(preparation, CoordinatedHandoverPreparation)
                or completed.operation_id != operation.operation_id
                or completed.robot_id != operation.executor_robot_id
                or completed.status != OperationStatus.CONFIRMED
                or completed.passed is not True
                or completed.terminal is not True
                or completed.motion_sent is not True
            ):
                raise ValueError("continuation requires confirmed physical preparation")
            proof = completed.evidence.get("preparation", {})
            lifecycle = proof.get("lifecycle", {})
            participants = lifecycle.get("participants", {})
            if (
                lifecycle.get("terminal") is not True
                or lifecycle.get("safe_to_release") is not True
                or set(participants) != set(preparation.participants)
                or any(
                    v.get("arrival_confirmed") is not True
                    or v.get("terminal") is not True
                    or v.get("safe") is not True
                    for v in participants.values()
                )
            ):
                raise ValueError("continuation lacks both terminal-safe arm arrivals")
            state.program = preparation
            state.motion_attempted = True
            state.validate_operation()
            state.details.update(completed.evidence)
            state.details.update(
                preparation_arrived=True,
                preparation_only=False,
                handover_complete=False,
                completion_scope="full_handover_with_cohold_and_owner_transfer",
                preparation_binding_sha256=preparation.binding_sha256,
            )
            complete = await self.factory.complete_preparation(operation, preparation)
            if (
                not isinstance(complete, CoordinatedHandoverProgram)
                or complete.rendezvous is not None
                or complete.preparation_binding_sha256 != preparation.binding_sha256
            ):
                raise ValueError("contact program is not bound to executed preparation")
            state.program = complete
            state.preparation_already_arrived = True
            state.validate_operation()
            return await state.execute()
        except (Exception, asyncio.CancelledError) as error:
            return await state.fail(error)
        finally:
            if state.inflight is None or state.inflight.done():
                self._active.pop(operation.operation_id, None)


class _Execution:
    """Private state owned by one operation and its two physical action lanes."""

    def __init__(
        self,
        operation: RobotOperation,
        transport: CoordinatedHandoverTransport,
        ledger: OperationStateLedger,
        factory: CoordinatedHandoverFactory,
    ) -> None:
        """Initialize the explicitly bound fixture state."""
        self.operation, self.transport, self.ledger = operation, transport, ledger
        self.factory = factory
        self.program: (
            CoordinatedHandoverProgram | CoordinatedHandoverPreparation | None
        ) = None
        self.inflight: asyncio.Task[HandoverPhaseReceipt] | None = None
        self.cancel_requested = False
        self.motion_attempted = False
        self.custody_monitored = False
        self.last: HandoverObservation | None = None
        self.last_phase: str | None = None
        self.current_phase: str | None = None
        self.attachment: Pose | None = None
        self.preparation_already_arrived = False
        self.details: dict[str, JSONValue] = {}
        self.stop_lock = asyncio.Lock()

    def bound(self) -> CoordinatedHandoverProgram | CoordinatedHandoverPreparation:
        """Require unchanged admitted input before every observation/dispatch.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        if (
            self.program is None
            or self.program.current_binding() != self.program.binding_sha256
        ):
            raise ValueError("HANDOVER admitted program changed or is unavailable")
        return self.program

    def validate_operation(self) -> None:
        """Reject mismatched roles, object, operation, profile or giver ledger."""
        """Read the next identified same-tick physics sample.

        Returns:
            The bound result of this read-only test boundary."""
        """Verify giver guard."""
        """Verify receiver guard."""
        """Verify window.

        Returns:
            The bound result of this read-only test boundary."""
        """Verify phase.

        Returns:
            The bound result of this read-only test boundary."""
        """Verify execute.

        Returns:
            The bound result of this read-only test boundary."""
        p = self.bound()
        op = self.operation
        target = op.target.values if op.target else {}
        if (
            op.skill != "HANDOVER"
            or op.operation_id != p.identity.operation_id
            or (
                op.executor_robot_id != p.identity.receiver_robot_id
                or set(op.participants) != set(p.participants)
                or dict(op.role_bindings)
                != {
                    "executor": p.identity.receiver_robot_id,
                    "receiver": p.identity.receiver_robot_id,
                    "donor": p.identity.donor_robot_id,
                }
                or any(
                    target.get(n) != getattr(p.identity, n)
                    for n in (
                        "donor_robot_id",
                        "receiver_robot_id",
                        "physical_instance_id",
                    )
                )
                or op.parameters.get("grasp_profile_id") != p.request.profile.profile_id
                or self.ledger.owner_of_object(p.identity.physical_instance_id)
                != p.identity.donor_robot_id
                or self.ledger.observe(
                    p.identity.donor_robot_id,
                    f"holding:{p.identity.physical_instance_id}",
                )
                is None
                or any(self.ledger.uncertain_keys(r) for r in p.participants)
            )
        ):
            raise ValueError(
                "HANDOVER program differs from operation or confirmed giver"
            )
        self.details["planning_binding_sha256"] = p.request.binding_sha256
        self.details["program_binding_sha256"] = p.binding_sha256

    async def observe(self) -> HandoverObservation:
        p = self.bound()
        if self.cancel_requested:
            raise RuntimeError("HANDOVER cancellation requested")
        self.custody_monitored = True
        raw = await asyncio.wait_for(
            self.transport.observe(p), p.request.co_hold_timeout_s
        )
        sample = parse_handover_observation(
            raw,
            donor_id=p.identity.donor_robot_id,
            receiver_id=p.identity.receiver_robot_id,
            physical_instance_id=p.identity.physical_instance_id,
            session_id=p.request.scene_snapshot["session_id"],
            manifest_sha256=p.request.scene_snapshot["manifest_sha256"],
        )
        previous = (
            self.last.donor
            if self.last
            else p.request.observation(p.identity.donor_robot_id)
        )
        gap = sample.donor.sim_time_s - previous.sim_time_s
        if sample.donor.sequence <= previous.sequence or gap <= 0:
            raise ValueError(
                "HANDOVER monitor returned repeated/stale/discontinuous physics"
            )
        # Isaac may advance substantially while a preparation observation or
        # phase handoff completes.  That interval is outside the contact proof:
        # retain freshness/sequence checks, then establish a new baseline when
        # the first contact phase begins.  Once contact monitoring is active,
        # every subsequent sample remains subject to the strict profile gap.
        if (
            self.last is not None
            and self.current_phase in CONTACT_PHASES
            and self.last_phase in CONTACT_PHASES
            and gap > p.request.profile.max_observation_gap_s
        ):
            raise ValueError(
                "HANDOVER monitor returned repeated/stale/discontinuous physics"
            )
        self.last, self.last_phase = sample, self.current_phase
        return sample

    def giver_guard(self, sample: HandoverObservation) -> None:
        p = self.bound()
        require_giver_retention_sample(
            sample,
            profile=p.request.profile,
            donor_window=p.donor_holding_window,
            expected_object_to_donor_tcp=p.object_to_donor_tcp,
        )

    def receiver_guard(self, sample: HandoverObservation, *, released: bool) -> None:
        p = self.bound()
        if self.attachment is None:
            raise ValueError("receiver contact has no measured co-hold attachment")
        require_receiver_retention_sample(
            sample,
            profile=p.request.profile,
            receiver_window=p.receiver_holding_window,
            expected_receiver_attachment=self.attachment,
            donor_release_window=p.donor_release_window if released else None,
        )

    async def window(
        self,
        guard: Callable[[HandoverObservation], None],
    ) -> tuple[HandoverObservation, ...]:
        p = self.bound()
        samples: list[HandoverObservation] = []
        deadline = asyncio.get_running_loop().time() + p.request.co_hold_timeout_s
        while len(samples) < p.request.follow_samples or (
            samples[-1].donor.sim_time_s - samples[0].donor.sim_time_s
            < p.request.profile.min_observation_span_s
        ):
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("HANDOVER physical observation window timed out")
            sample = await self.observe()
            guard(sample)
            samples.append(sample)
        return tuple(samples)

    async def phase(
        self,
        phase: str,
        guard: Callable[[HandoverObservation], None],
    ) -> tuple[HandoverObservation, ...]:
        p = self.bound()
        if self.cancel_requested:
            raise RuntimeError("HANDOVER cancellation requested before dispatch")
        self.motion_attempted = (
            True  # BEFORE dispatch; send errors may have moved hardware.
        )
        self.current_phase = phase
        self.inflight = asyncio.create_task(
            self.transport.execute_phase(p, phase, self.attachment)
        )
        samples: list[HandoverObservation] = []
        deadline = asyncio.get_running_loop().time() + p.phase_timeout_s
        while not self.inflight.done():
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"HANDOVER {phase} physical completion timed out")
            sample = await self.observe()
            guard(sample)
            samples.append(sample)
        receipt = self.inflight.result()
        self.inflight = None
        if not isinstance(receipt, HandoverPhaseReceipt) or (
            receipt.operation_id != p.identity.operation_id
            or receipt.phase != phase
            or receipt.program_binding_sha256 != p.binding_sha256
            or not all((receipt.passed, receipt.terminal, receipt.safe_to_release))
        ):
            raise ValueError(f"HANDOVER {phase} missing bound terminal-safe completion")
        if phase in {"rendezvous", "approach", "separation", "receiver_follow"}:
            if (
                any(
                    receipt.evidence.get(k) is not True
                    for k in (
                        "prepared_path_executed",
                        "arrival_confirmed",
                        "tracking_certified",
                    )
                )
                or receipt.evidence.get("replanned") is not False
            ):
                raise ValueError(
                    f"HANDOVER {phase} lacks certified prepared-path feedback"
                )
        elif phase == "scene_transfer":
            if receipt.evidence.get("transfer_status") != "COMMITTED" or (
                receipt.evidence.get("readback_confirmed") is not True
            ):
                raise ValueError("HANDOVER scene transfer lacks committed readback")
        elif any(
            receipt.evidence.get(k) is not True
            for k in (
                "gripper_feedback_confirmed",
                "sweep_certified",
            )
        ):
            raise ValueError(
                f"HANDOVER {phase} lacks physical gripper/sweep completion"
            )
        self.details[phase] = dict(receipt.evidence)
        sample = await self.observe()
        guard(sample)
        samples.append(sample)
        return tuple(samples)

    async def execute(self) -> OperationExecution:
        p = self.bound()
        samples = await self.window(self.giver_guard)
        self.details["giver_retention_proof"] = to_json_value(
            verify_giver_holding(
                samples,
                profile=p.request.profile,
                follow_samples=p.request.follow_samples,
                donor_window=p.donor_holding_window,
                expected_object_to_donor_tcp=p.object_to_donor_tcp,
            )
        )
        if isinstance(p, CoordinatedHandoverPreparation):
            for phase in ("receiver_open", "rendezvous"):
                await self.phase(phase, self.giver_guard)
            self.details["preparation_arrived"] = True
            self.details["preparation_binding_sha256"] = p.binding_sha256
            if self.last is not None:
                self.details["preparation_arrival"] = to_json_value(
                    {
                        "sequence": self.last.donor.sequence,
                        "sim_time_s": self.last.donor.sim_time_s,
                        "object_pose": asdict(self.last.donor.object_pose),
                        "tcp_poses": {
                            p.identity.donor_robot_id: asdict(self.last.donor.tcp_pose),
                            p.identity.receiver_robot_id: asdict(
                                self.last.receiver.tcp_pose
                            ),
                        },
                    }
                )
            complete = await self.factory.complete_preparation(self.operation, p)
            if (
                not isinstance(complete, CoordinatedHandoverProgram)
                or complete.preparation_binding_sha256 != p.binding_sha256
            ):
                raise ValueError("contact program is not bound to executed preparation")
            self.program = complete
            self.validate_operation()
            p = complete
            remaining = ("approach", "receiver_close")
        elif self.preparation_already_arrived:
            remaining = ("approach", "receiver_close")
        else:
            remaining = (
                ("receiver_open", "approach", "receiver_close")
                if p.rendezvous is None
                else ("receiver_open", "rendezvous", "approach", "receiver_close")
            )
        for phase in remaining:
            await self.phase(phase, self.giver_guard)
        samples = await self.window(self.giver_guard)
        co_hold = verify_co_hold(
            samples,
            profile=p.request.profile,
            follow_samples=p.request.follow_samples,
            donor_window=p.donor_holding_window,
            receiver_window=p.receiver_holding_window,
            expected_object_to_donor_tcp=p.candidate.object_to_donor_tcp,
        )
        self.attachment = compose(
            inverse(samples[-1].receiver.tcp_pose), samples[-1].receiver.object_pose
        )
        errors = pose_error(self.attachment, p.paths.planned_receiver_attachment)
        if errors[0] > p.request.profile.relative_position_tolerance_m or (
            errors[1] > p.request.profile.relative_orientation_tolerance_rad
        ):
            raise ValueError(
                "measured receiver attachment exceeds certified co-hold binding"
            )
        self.details["co_hold_proof"] = to_json_value(co_hold)
        self.ledger.begin_provisional_handover(
            p.identity, p.request.binding_sha256, to_json_value(co_hold)
        )
        # Confirmation is revoked BEFORE the first potentially releasing send.
        self.ledger.mark_provisional_release(p.identity, p.request.binding_sha256)
        await self.phase("donor_open", lambda s: self.receiver_guard(s, released=False))

        def released_guard(sample: HandoverObservation) -> None:
            self.receiver_guard(sample, released=True)

        samples = await self.window(released_guard)
        self.details["release_retention_proof"] = to_json_value(
            verify_receiver_retention(
                samples,
                profile=p.request.profile,
                follow_samples=p.request.follow_samples,
                receiver_window=p.receiver_holding_window,
                expected_receiver_attachment=self.attachment,
                donor_release_window=p.donor_release_window,
            )
        )
        await self.phase("scene_transfer", released_guard)
        await self.phase("separation", released_guard)
        self.details["giver_separated"] = True
        # Include the BEFORE tick and live moving ticks, not just settled endpoints.
        before = await self.observe()
        released_guard(before)
        moving = await self.phase("receiver_follow", released_guard)
        after = await self.window(released_guard)
        follow = verify_receiver_follow(
            (before, *moving, *after),
            profile=p.request.profile,
            follow_samples=p.request.follow_samples,
            donor_release_window=p.donor_release_window,
            receiver_window=p.receiver_holding_window,
        )
        self.details["receiver_follow_proof"] = to_json_value(follow)
        evidence = HandoverEvidence(
            p.identity.operation_id,
            *p.participants,
            p.identity.physical_instance_id,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            self.details,
        )
        self.ledger.commit_provisional_handover(
            p.identity, p.request.binding_sha256, evidence
        )
        self.details.update(
            {
                name: True
                for name in (
                    "receiver_free_space",
                    "receiver_grasp_feedback",
                    "co_hold_confirmed",
                    "scene_transfer_committed",
                    "donor_release_feedback",
                    "receiver_holding_follow",
                    "terminal",
                    "safe_to_release",
                    "owner_transfer_committed",
                )
            }
        )
        self.details["handover_complete"] = True
        return OperationExecution(
            self.operation.operation_id,
            self.operation.executor_robot_id,
            OperationStatus.CONFIRMED,
            True,
            True,
            True,
            self.details,
        )

    async def stop(self) -> Mapping[str, object]:
        """Stop both physical lanes, then wait for the motion coroutine to settle.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        async with self.stop_lock:
            return await self._stop()

    async def _stop(self) -> Mapping[str, object]:
        """Serialize owned cancellation and physical task settlement.

        Returns:
            The identified, bound result; physical proof remains separately checked.
        """
        if self.program is None or not self.motion_attempted:
            return {
                "operation_id": self.operation.operation_id,
                "terminal": True,
                "safe_to_release": True,
                "motion_sent": False,
            }
        p = self.program
        try:
            proof = await asyncio.wait_for(
                self.transport.cancel(p),
                p.phase_timeout_s,
            )
            settled = True
            if self.inflight is not None:
                done, _ = await asyncio.wait({self.inflight}, timeout=p.phase_timeout_s)
                settled = bool(done)
                if settled:
                    # Retrieve failures; task cancellation is not stop proof.
                    if not self.inflight.cancelled():
                        self.inflight.exception()
                    self.inflight = None
            participants = proof.get("participants")
            safe = (
                settled
                and proof.get("operation_id") == p.identity.operation_id
                and (
                    isinstance(participants, Mapping)
                    and set(participants) == set(p.participants)
                    and all(
                        isinstance(v, Mapping)
                        and v.get("terminal") is True
                        and v.get("safe_to_release") is True
                        for v in participants.values()
                    )
                )
            )
            return {
                "operation_id": p.identity.operation_id,
                "terminal": bool(safe),
                "safe_to_release": bool(safe),
                "motion_sent": True,
                "physical_cancel": to_json_value(proof),
            }
        except (Exception, asyncio.CancelledError) as error:
            return {
                "operation_id": p.identity.operation_id,
                "terminal": False,
                "safe_to_release": False,
                "motion_sent": True,
                "reason": str(error),
            }

    async def fail(self, error: BaseException) -> OperationExecution:
        """Verify fail.

        Returns:
            The bound result of this read-only test boundary.
        """
        reason = f"COORDINATED_HANDOVER_FAILED: {type(error).__name__}: {error}"
        stop = await self.stop()
        self.details["stop"] = to_json_value(stop)
        safe = stop.get("terminal") is True and stop.get("safe_to_release") is True
        if (
            self.motion_attempted or self.custody_monitored
        ) and self.program is not None:
            self.ledger.mark_handover_uncertain(
                self.program.identity, reason, self.details
            )
            # Physical stop does not authorize releasing an object lease.
            safe = False
        self.details["safe_to_release"] = safe
        return OperationExecution(
            self.operation.operation_id,
            self.operation.executor_robot_id,
            OperationStatus.FAILED,
            False,
            stop.get("terminal") is True,
            self.motion_attempted,
            self.details,
            reason,
        )
