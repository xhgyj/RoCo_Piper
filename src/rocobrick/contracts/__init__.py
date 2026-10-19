"""Pure, strict cross-module contracts for the multi-arm execution platform.

The package is a leaf: it imports only the standard library and its own
submodules, never skills, schedulers, adapters, ROS, MoveIt or frameworks.

Public contracts:

* :class:`~rocobrick.contracts.identity.PhaseIdentity` -- plan/operation/phase/
  attempt coordinate; retries are new identities.
* :class:`~rocobrick.contracts.evidence.PhysicalEvidence` -- validated physical
  observation; planner, controller and dispatch results are not evidence.
* :class:`~rocobrick.contracts.phases.PhaseIntent` -- declarative phase
  requirements, goal/constraint/contact references and per-arm permissions.
* :class:`~rocobrick.contracts.motion.MotionCommit` -- validated, lease-bound,
  version-pinned motion commitment for exactly the bound participants.
* :class:`~rocobrick.contracts.failures.FailureReport` -- closed failure
  taxonomy that cannot assert physical success.

Every contract is an immutable frozen dataclass with a strict ``from_dict``
parser, a JSON-safe ``to_dict`` serializer and a full JSON roundtrip.
"""

from rocobrick.contracts._validation import ContractViolationError
from rocobrick.contracts.evidence import (
    PHYSICAL_EVIDENCE_SCHEMA,
    EvidencePolarity,
    EvidenceProvenance,
    ObservationChannel,
    PhysicalEvidence,
    PhysicalFact,
    PhysicalFactKind,
    ValidationCheck,
    ValidationProof,
)
from rocobrick.contracts.failures import (
    FAILURE_REPORT_SCHEMA,
    FailureCategory,
    FailureReport,
)
from rocobrick.contracts.identity import (
    PHASE_IDENTITY_SCHEMA,
    PhaseIdentity,
    unique_identities,
)
from rocobrick.contracts.motion import (
    MOTION_COMMIT_SCHEMA,
    JointSample,
    JointTrajectory,
    MotionCommit,
)
from rocobrick.contracts.phases import (
    PHASE_INTENT_SCHEMA,
    ContactDeclaration,
    ContactMode,
    GoalDeclaration,
    GoalKind,
    MotionPermission,
    PermissionKind,
    PhaseIntent,
    PhysicalRequirement,
    ResourceKind,
    ResourceRequirement,
)

__all__ = [
    "FAILURE_REPORT_SCHEMA",
    "MOTION_COMMIT_SCHEMA",
    "PHASE_IDENTITY_SCHEMA",
    "PHASE_INTENT_SCHEMA",
    "PHYSICAL_EVIDENCE_SCHEMA",
    "ContactDeclaration",
    "ContactMode",
    "ContractViolationError",
    "EvidencePolarity",
    "EvidenceProvenance",
    "FailureCategory",
    "FailureReport",
    "GoalDeclaration",
    "GoalKind",
    "JointSample",
    "JointTrajectory",
    "MotionCommit",
    "MotionPermission",
    "ObservationChannel",
    "PermissionKind",
    "PhaseIdentity",
    "PhaseIntent",
    "PhysicalEvidence",
    "PhysicalFact",
    "PhysicalFactKind",
    "PhysicalRequirement",
    "ResourceKind",
    "ResourceRequirement",
    "ValidationCheck",
    "ValidationProof",
    "unique_identities",
]
