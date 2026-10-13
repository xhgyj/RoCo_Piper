"""Isolated full-model collision checks and conservative swept-path proofs.

The combined exported URDF supplies actual robot collision geometry and limits.
Measured brick/environment bounds are conservative collision envelopes, never
grasp dimensions. Default cross-robot checks remain strict. Explicit Isaac
contact phases may instead admit shallow, localized finger contact under a
separately hash-bound nominal/observed-contact contract. Imports are lazy.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from rocobrick.skills.handover_contact import (
    SimulationFingerContactPolicy,
    finger_pairs,
    region_margin,
)
from rocobrick.skills.handover_planning import HandoverPlanningRequest
from rocobrick.skills.handover_profile import EXECUTION_CAPABILITIES, digest, finite
from rocobrick.skills.handover_simulation import SimulationHandoverAuthorization
from rocobrick.skills.observation import Pose, _vector
from rocobrick.skills.placement_geometry import (
    compose,
    interpolate_pose,
    inverse,
    pose_error,
)


@dataclass(frozen=True)
class HandoverModelFiles:
    """Captured combined MoveIt model, never a guessed canonical-arm fallback."""

    urdf_path: Path
    srdf_path: Path
    package_dirs: tuple[Path, ...]
    urdf_sha256: str
    srdf_sha256: str
    runtime_model_sha256: str
    mesh_assets: tuple[tuple[Path, str], ...] = ()
    joint_limits_path: Path | None = None
    joint_limits_sha256: str | None = None

    @classmethod
    def capture(
        cls,
        urdf_path: Path,
        srdf_path: Path,
        *,
        package_dirs: tuple[Path, ...],
        runtime_model_sha256: str,
        joint_limits_path: Path | None = None,
    ) -> HandoverModelFiles:
        """Capture model files, meshes and the effective MoveIt joint limits.

        Returns:
            An exact hash-bound model bundle. Ambiguous/missing package assets
            fail rather than loading a different arm or dropping geometry.
        """
        urdf_bytes, srdf_bytes = urdf_path.read_bytes(), srdf_path.read_bytes()
        meshes = set()
        for mesh in ET.fromstring(urdf_bytes).findall("link/collision/geometry/mesh"):
            name = mesh.get("filename", "")
            if name.startswith("package://"):
                package, separator, relative = name[len("package://") :].partition("/")
                if not separator or not package or not relative:
                    raise ValueError("collision mesh package URI is incomplete")
                candidates = {root / package / relative for root in package_dirs}
                candidates |= {
                    root / relative for root in package_dirs if root.name == package
                }
            else:
                if "://" in name and not name.startswith("file://"):
                    raise ValueError("unsupported collision mesh URI")
                path = Path(name.removeprefix("file://"))
                candidates = {path if path.is_absolute() else urdf_path.parent / path}
            matches = {p.resolve() for p in candidates if p.is_file()}
            if len(matches) != 1:
                raise ValueError("collision mesh asset is missing or ambiguous")
            meshes.update(matches)
        joint_limits_sha256 = None
        if joint_limits_path is not None:
            joint_limits_sha256 = hashlib.sha256(
                joint_limits_path.read_bytes()
            ).hexdigest()
        value = cls(
            urdf_path,
            srdf_path,
            package_dirs,
            hashlib.sha256(urdf_bytes).hexdigest(),
            hashlib.sha256(srdf_bytes).hexdigest(),
            runtime_model_sha256,
            tuple(
                (p, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(meshes)
            ),
            joint_limits_path,
            joint_limits_sha256,
        )
        value.validate()
        return value

    def validate(self) -> None:
        """Verify both captured model files and the admitted runtime identity."""
        for path, expected in (
            (self.urdf_path, self.urdf_sha256),
            (self.srdf_path, self.srdf_sha256),
        ):
            digest(expected, "captured model")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("captured HANDOVER model changed on disk")
        digest(self.runtime_model_sha256, "runtime model")
        if (self.joint_limits_path is None) != (self.joint_limits_sha256 is None):
            raise ValueError("captured MoveIt joint-limit provenance is incomplete")
        if self.joint_limits_path is not None and self.joint_limits_sha256 is not None:
            digest(self.joint_limits_sha256, "MoveIt joint limits")
            if hashlib.sha256(self.joint_limits_path.read_bytes()).hexdigest() != (
                self.joint_limits_sha256
            ):
                raise ValueError("captured MoveIt joint limits changed on disk")
        if not isinstance(self.mesh_assets, tuple) or len(
            {p for p, _ in self.mesh_assets}
        ) != len(self.mesh_assets):
            raise ValueError("collision mesh provenance must be unique and immutable")
        for path, expected in self.mesh_assets:
            digest(expected, "collision mesh")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("captured collision mesh changed on disk")
        if not isinstance(self.package_dirs, tuple) or any(
            not isinstance(p, Path) or not p.is_dir() for p in self.package_dirs
        ):
            raise ValueError("collision mesh package directories are unavailable")


def _moveit_position_limits(
    path: Path, variables: Mapping[str, tuple[int, int]]
) -> dict[str, tuple[float, float]]:
    """Read the combined MoveIt joint_limits.yaml position overrides.

    Follows MoveIt semantics: only entries with ``has_position_limits: true``
    narrow planning bounds; an absent flag never narrows, mirroring the
    planner that ignores min/max without the flag. Every malformed entry,
    inverted range or foreign joint name fails closed.

    Returns:
        Per-joint ``(minimum, maximum)`` overrides for model joints only.
    """
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or not isinstance(
        document.get("joint_limits"), Mapping
    ):
        raise ValueError("MoveIt joint limits document is malformed")
    result: dict[str, tuple[float, float]] = {}
    for joint, raw in document["joint_limits"].items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"MoveIt joint limits entry {joint!r} is malformed")
        if raw.get("has_position_limits") is not True:
            continue
        bounds = []
        for key in ("min_position", "max_position"):
            value = raw.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"MoveIt joint limits {joint!r} lack finite {key}")
            bounds.append(float(value))
        if bounds[0] >= bounds[1]:
            raise ValueError(f"MoveIt joint limits {joint!r} are inverted")
        if joint not in variables:
            raise ValueError(f"MoveIt joint limits name a foreign joint {joint!r}")
        result[str(joint)] = (bounds[0], bounds[1])
    return result


@dataclass(frozen=True)
class CollisionVerdict:
    """Bounds and full-geometry clearance in the isolated scene."""

    valid: bool
    clearance_m: float
    collision_pairs: tuple[tuple[str, str], ...]
    reason: str | None = None


class IsolatedHandoverGeometry:
    """Read-only Pinocchio/Coal mirror of one combined planning scene."""

    def __init__(
        self,
        files: HandoverModelFiles,
        request: HandoverPlanningRequest,
        *,
        tracking_error_rad: float = 0.0,
        fixed_mount_contacts: tuple[tuple[str, str], ...] = (),
        finger_contact_policy: SimulationFingerContactPolicy | None = None,
        simulation_authorization: SimulationHandoverAuthorization | None = None,
    ):
        """Load the full model, SRDF exclusions and measured collision envelopes."""
        files.validate()
        if files.runtime_model_sha256 != request.profile.runtime_model_sha256:
            raise ValueError("collision model does not match HANDOVER profile")
        import coal
        import numpy as np
        import pinocchio as pin

        self.pin, self.np, self.coal = pin, np, coal
        self.files = files
        self.tracking_error_rad = finite(tracking_error_rad, "geometry tracking budget")
        if not 0 <= self.tracking_error_rad <= 0.01:
            raise ValueError("geometry tracking budget is out of bounds")
        self.request = request
        if finger_contact_policy is not None:
            if (
                simulation_authorization is None
                or simulation_authorization.finger_contact_policy_sha256
                != finger_contact_policy.sha256
            ):
                raise ValueError(
                    "finger contact needs exact simulation policy authorization"
                )
            simulation_authorization.validate(
                request.profile, request.scene_snapshot, EXECUTION_CAPABILITIES
            )
        self.finger_contact_policy = finger_contact_policy
        self._finger_pairs = finger_pairs(
            request.donor.robot_id, request.receiver.robot_id
        )
        self._guarded_pairs = set()
        self._finger_margins = {}
        self.binding = request.binding_sha256
        self.model = pin.buildModelFromUrdf(str(files.urdf_path))
        self.geometry = pin.buildGeomFromUrdf(
            self.model,
            str(files.urdf_path),
            pin.GeometryType.COLLISION,
            package_dirs=[str(p) for p in files.package_dirs],
        )
        if not self.geometry.geometryObjects:
            raise ValueError("combined model contains no collision geometry")
        captured_meshes = {p.resolve() for p, _ in files.mesh_assets}
        loaded_meshes = {
            Path(g.meshPath).resolve()
            for g in self.geometry.geometryObjects
            if not isinstance(
                g.geometry, (coal.Box, coal.Sphere, coal.Cylinder, coal.Capsule)
            )
        }
        if loaded_meshes != captured_meshes:
            raise ValueError(
                "every loaded collision mesh requires exact file provenance"
            )
        files.validate()
        self._variables = {}
        for index in range(1, self.model.njoints):
            joint = self.model.joints[index]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError("HANDOVER supports only bounded one-DOF Piper joints")
            self._variables[self.model.names[index]] = (joint.idx_q, joint.idx_v)
        # Native admission uses the EFFECTIVE bounds: the intersection of the
        # exported URDF limits with the combined MoveIt joint_limits.yaml the
        # production planner enforces. A candidate outside this intersection
        # is rejected before any MoveGroup query, never clamped downstream.
        self._lower_limits = self.np.array(self.model.lowerPositionLimit, dtype=float)
        self._upper_limits = self.np.array(self.model.upperPositionLimit, dtype=float)
        if files.joint_limits_path is not None:
            for name, (minimum, maximum) in _moveit_position_limits(
                files.joint_limits_path, self._variables
            ).items():
                index = self._variables[name][0]
                self._lower_limits[index] = max(self._lower_limits[index], minimum)
                self._upper_limits[index] = min(self._upper_limits[index], maximum)
                if self._lower_limits[index] > self._upper_limits[index]:
                    raise ValueError(
                        f"MoveIt joint limits for {name} exclude the URDF range"
                    )
        for ctx in (request.donor, request.receiver):
            expected = {f"{ctx.robot_id}_joint{i}" for i in range(1, 7)}
            if not expected <= set(self._variables):
                raise ValueError("combined model omits one complete arm")
        self._base_q = self.configuration(request.start_state)
        self._data = self.model.createData()
        self._robot_objects = len(self.geometry.geometryObjects)
        self._object_name = "held::" + request.physical_instance_id
        donor = request.observation(request.donor.robot_id)
        self._object_to_tcp = compose(inverse(donor.tcp_pose), donor.object_pose)
        corners = self._local_box(donor.aabb_min, donor.aabb_max, donor.object_pose)
        self._object_bounds = corners
        self._object_index = self._add_box(
            self._object_name,
            *corners,
            pose=self._object_to_tcp,
            frame_id=f"{request.donor.robot_id}_rocobrick_tcp",
        )
        for physical, raw in request.scene_snapshot["bricks"].items():
            if physical != request.physical_instance_id:
                self._add_world_box("world::brick::" + physical, raw)
        environment = request.scene_snapshot.get("environment_colliders")
        if not isinstance(environment, Mapping) or not environment:
            raise ValueError(
                "complete measured environment collision envelopes are required"
            )
        for name, raw in environment.items():
            self._add_world_box("world::environment::" + name, raw)
        self._link_names = [
            self.model.frames[g.parentFrame].name
            if g.parentFrame < len(self.model.frames)
            else ""
            for g in self.geometry.geometryObjects
        ]
        self.fixed_mount_contacts = frozenset(
            frozenset(pair) for pair in fixed_mount_contacts
        )
        # An explicit simulation installation contact may only join a captured
        # immobile base to an identified world collider, never a moving link.
        for link, world_name in fixed_mount_contacts:
            if link not in {
                request.donor.robot_id + "_base_link",
                request.receiver.robot_id + "_base_link",
            } or not world_name.startswith("world::environment::"):
                raise ValueError("invalid fixed installation contact")
            bases = [
                g
                for i, g in enumerate(self.geometry.geometryObjects)
                if self._link_names[i] == link and i < self._robot_objects
            ]
            worlds = [g for g in self.geometry.geometryObjects if g.name == world_name]
            if not bases or len(worlds) != 1 or any(g.parentJoint != 0 for g in bases):
                raise ValueError(
                    "installation contact must have two world-fixed bodies"
                )
        exclusions = self._srdf_exclusions(files.srdf_path)
        # Pinocchio's addAllCollisionPairs omits equal-parent-joint pairs.
        # Explicit construction is required so an attachment on joint6 is
        # still checked against the wrist and rigidly mounted gripper body.
        for a, b in itertools.combinations(
            range(len(self.geometry.geometryObjects)), 2
        ):
            if self._link_names[a] == self._link_names[b] and (
                a < self._robot_objects and b < self._robot_objects
            ):
                continue
            ga, gb = self.geometry.geometryObjects[a], self.geometry.geometryObjects[b]
            if (
                frozenset((self._link_names[a], gb.name)) in self.fixed_mount_contacts
            ) or (
                frozenset((self._link_names[b], ga.name)) in self.fixed_mount_contacts
            ):
                continue
            if (ga.name.startswith("world::") and gb.name.startswith("world::")) or (
                a < self._robot_objects
                and b < self._robot_objects
                and frozenset((self._link_names[a], self._link_names[b])) in exclusions
            ):
                continue
            self.geometry.addCollisionPair(pin.CollisionPair(a, b))
        self._geometry_data = pin.GeometryData(self.geometry)
        local_boxes = []
        for shape in self.geometry.geometryObjects:
            shape.geometry.computeLocalAABB()
            box = shape.geometry.aabb_local
            minimum, maximum = np.array(box.min_), np.array(box.max_)
            local_boxes.append(((minimum + maximum) / 2, (maximum - minimum) / 2))
        self._local_collision_boxes = tuple(local_boxes)
        self._pair_clearances = [0.0] * len(self.geometry.collisionPairs)
        self._exact_pair_distances: set[int] = set()
        self.last_certificate_failure: str | None = None
        self.last_clearance_failure: tuple[str, str, float, float] | None = None
        self.last_contact_region_failure: tuple[str, str] | None = None
        self._relative_layout_key = None
        self._relative_radii = ()
        self._arm_variables = frozenset(
            f"{r}_joint{i}" for r in request.session_robots for i in range(1, 7)
        )
        self._reach_bound = self._motion_radius_bound()
        self._assert_base_frames()

    def solve_tcp_ik(
        self,
        robot: str,
        world_goal: Pose,
        seed: Mapping[str, float],
        *,
        deadline_s: float,
        clock: Callable[[], float],
        attempts: int = 24,
    ) -> Mapping[str, float] | None:
        """Solve calibrated FK numerically within unchanged exported limits.

        Returns:
            One six-joint solution, or None after the finite seed/time budget.
            Collision admission remains a separate complete-pair gate.
        """
        if robot not in (self.request.donor.robot_id, self.request.receiver.robot_id):
            raise ValueError("IK robot does not belong to this isolated scene")
        if type(attempts) is not int or not 1 <= attempts <= 128:
            raise ValueError("IK seed budget must be an integer in [1, 128]")
        finite(deadline_s, "IK deadline")
        frame = self._frame(robot + "_rocobrick_tcp")
        names = tuple(f"{robot}_joint{i}" for i in range(1, 7))
        q_indices = [self._variables[name][0] for name in names]
        target = self._se3(world_goal)
        generator = self.np.random.default_rng(int(self.binding[:16], 16))
        base = self.configuration(seed)
        lo = self._lower_limits[q_indices]
        hi = self._upper_limits[q_indices]
        if not self.np.isfinite(lo).all() or not self.np.isfinite(hi).all():
            raise ValueError("IK requires finite captured arm joint limits")
        from scipy.optimize import least_squares

        class DeadlineExpiredError(Exception):
            pass

        def residual(values):
            if clock() >= deadline_s:
                raise DeadlineExpiredError
            q = base.copy()
            q[q_indices] = values
            self.pin.framesForwardKinematics(self.model, self._data, q)
            return self.pin.log6(self._data.oMf[frame].actInv(target)).vector.copy()

        for attempt in range(attempts):
            q = base.copy()
            if attempt == 1:
                q[q_indices] = (lo + hi) / 2.0
            elif 2 <= attempt <= 5:
                # Reproducible global seeds, independent of clock/session IDs.
                q[q_indices] = self.np.random.default_rng(attempt - 2).uniform(lo, hi)
            elif attempt:
                q[q_indices] = generator.uniform(lo, hi)
            try:
                # Bound-aware trust-region steps avoid the old DLS/clipping
                # trap at Piper limits. Every residual checks the wall budget.
                solved = least_squares(
                    residual,
                    self.np.clip(q[q_indices], lo, hi),
                    bounds=(lo, hi),
                    max_nfev=160,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                error = residual(solved.x)
            except DeadlineExpiredError:
                return None
            if self.np.linalg.norm(error[:3]) <= (
                self.request.profile.position_tolerance_m * 0.1
            ) and self.np.linalg.norm(error[3:]) <= (
                self.request.profile.orientation_tolerance_rad * 0.1
            ):
                return {name: float(value) for name, value in zip(names, solved.x)}
        return None

    def configuration(self, joints: Mapping[str, float]):
        """Require every exported independent joint, including both grippers.

        Returns:
            A complete native configuration with no neutral/zero fallback.
        """
        q = self.np.empty(self.model.nq)
        for name, (index, _) in self._variables.items():
            q[index] = finite(joints.get(name), f"measured joint {name}")
        # Snap only nanoscopic solver/readback roundoff to the unchanged
        # bounds. Real excursions remain outside and fail check(); raw action
        # feedback is not rewritten. Arm tracking uncertainty is much larger.
        clipped = self.np.clip(q, self._lower_limits, self._upper_limits)
        near = self.np.abs(clipped - q) <= 1e-8
        q[near] = clipped[near]
        return q

    def tcp_pose(self, robot: str, joints: Mapping[str, float]) -> Pose:
        """Return actual captured-model FK, without modifying the shared scene."""
        if robot not in (self.request.donor.robot_id, self.request.receiver.robot_id):
            raise ValueError("FK robot does not belong to this isolated scene")
        self.pin.framesForwardKinematics(
            self.model, self._data, self.configuration(joints)
        )
        pose = self._data.oMf[self._frame(robot + "_rocobrick_tcp")]
        return Pose(
            tuple(float(x) for x in pose.translation),
            tuple(float(x) for x in self.pin.Quaternion(pose.rotation).coeffs()),
        )

    def certify_cartesian_segment(
        self,
        initial: Mapping[str, float],
        final: Mapping[str, float],
        *,
        robot: str,
        first_pose: Pose,
        last_pose: Pose,
        deadline_s: float,
        clock: Callable[[], float],
        receiver_touch: bool = False,
        holder: str | None = None,
        receiver_attachment: Pose | None = None,
        finger_contact: bool = False,
    ) -> bool:
        """Certify collision and the position/shortest-rotation tube throughout.

        Returns:
            False when joint-linear interpolation cannot be proved to remain
            within profile position/rotation tolerances; endpoint IK is not proof.
        """
        if not self.certify_segment(
            initial,
            final,
            deadline_s=deadline_s,
            clock=clock,
            receiver_touch=receiver_touch,
            holder=holder,
            receiver_attachment=receiver_attachment,
            finger_contact=finger_contact,
        ):
            return False
        queue = [(dict(initial), dict(final), first_pose, last_pose, 0)]
        for _ in range(4096):
            if not queue:
                return True
            if clock() >= deadline_s:
                self.last_certificate_failure = "CARTESIAN_TIME_BUDGET"
                return False
            first, last, a, b, depth = queue.pop()
            middle = {name: (first[name] + last[name]) / 2 for name in first}
            target = interpolate_pose(a, b, 0.5)
            position, rotation = pose_error(self.tcp_pose(robot, middle), target)
            if position > self.request.profile.position_tolerance_m or (
                rotation > self.request.profile.orientation_tolerance_rad
            ):
                self.last_certificate_failure = "CARTESIAN_TUBE_EXCEEDED"
                return False
            # Bound FK and desired-line movement from the interval midpoint.
            displacement = self.motion_bound(first, last) / 4
            desired = math.dist(a.position_xyz, b.position_xyz) / 2
            desired_angle = pose_error(a, b)[1] / 2
            angle = sum(
                abs(last[n] - first[n]) / 2
                for n in self._variables
                if n.startswith(robot + "_")
                and self.model.joints[self.model.getJointId(n)]
                .shortname()
                .startswith("JointModelR")
            )
            if (
                position + displacement + desired
                <= self.request.profile.position_tolerance_m
                and (
                    rotation + angle + desired_angle
                    <= self.request.profile.orientation_tolerance_rad
                )
            ):
                continue
            if depth >= 24:
                self.last_certificate_failure = "CARTESIAN_DEPTH_BUDGET"
                return False
            queue.extend(
                (
                    (first, middle, a, target, depth + 1),
                    (middle, last, target, b, depth + 1),
                )
            )
        self.last_certificate_failure = "CARTESIAN_CHECK_BUDGET"
        return False

    def _se3(self, pose: Pose):
        q = self.pin.Quaternion(self.np.array(pose.orientation_xyzw))
        return self.pin.SE3(q.matrix(), self.np.array(pose.position_xyz))

    def _frame(self, name: str) -> int:
        index = self.model.getFrameId(name)
        if index >= len(self.model.frames):
            raise ValueError(f"captured combined model has no frame {name}")
        return index

    def _local_box(self, lo, hi, object_pose):
        transform = inverse(object_pose)
        points = [
            compose(transform, Pose(tuple(p), (0, 0, 0, 1))).position_xyz
            for p in itertools.product(*zip(lo, hi))
        ]
        return (
            tuple(min(p[i] for p in points) for i in range(3)),
            tuple(max(p[i] for p in points) for i in range(3)),
        )

    def _add_world_box(self, name: str, raw: object) -> None:
        if not isinstance(raw, Mapping):
            raise ValueError("measured collision envelope is malformed")
        self._add_box(
            name,
            raw.get("aabb_min"),
            raw.get("aabb_max"),
            pose=Pose((0, 0, 0), (0, 0, 0, 1)),
        )

    def _add_box(self, name, minimum, maximum, *, pose, frame_id=None):
        lo = _vector(minimum, 3, "collision minimum")
        hi = _vector(maximum, 3, "collision maximum")
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError("collision envelope is degenerate")
        center = tuple((a + b) / 2 for a, b in zip(lo, hi))
        shape = self.coal.Box(*[b - a for a, b in zip(lo, hi)])
        frame = self._frame(frame_id) if frame_id else 0
        parent = self.model.frames[frame].parentJoint if frame_id else 0
        placement = self._se3(compose(pose, Pose(center, (0, 0, 0, 1))))
        if frame_id:
            placement = self.model.frames[frame].placement * placement
        obj = self.pin.GeometryObject(name, parent, frame, placement, shape)
        return self.geometry.addGeometryObject(obj)

    def _srdf_exclusions(self, path: Path) -> set[frozenset[str]]:
        robots = (self.request.donor.robot_id, self.request.receiver.robot_id)
        result = set()
        for element in ET.fromstring(path.read_bytes()).findall("disable_collisions"):
            links = (element.get("link1", ""), element.get("link2", ""))
            owners = [
                next((r for r in robots if link.startswith(r + "_")), None)
                for link in links
            ]
            if (
                owners[0] is not None
                and owners[1] is not None
                and owners[0] != owners[1]
            ):
                raise ValueError("cross-robot collision exemptions are forbidden")
            result.add(frozenset(links))
        return result

    def _assert_base_frames(self) -> None:
        self.pin.framesForwardKinematics(self.model, self._data, self._base_q)
        for context in (self.request.donor, self.request.receiver):
            actual = self._data.oMf[self._frame(context.robot_id + "_base_link")]
            expected = self._se3(
                Pose(context.base.position_xyz, context.base.orientation_xyzw)
            )
            if (
                self.np.linalg.norm(self.pin.log6(actual.inverse() * expected).vector)
                > 1e-6
            ):
                raise ValueError(
                    "model base pose differs from admitted runtime calibration"
                )
            tcp = self._data.oMf[self._frame(context.robot_id + "_rocobrick_tcp")]
            observed = self._se3(self.request.observation(context.robot_id).tcp_pose)
            delta = self.pin.log6(tcp.actInv(observed)).vector
            if self.np.linalg.norm(
                delta[:3]
            ) > self.request.profile.position_tolerance_m or (
                self.np.linalg.norm(delta[3:])
                > self.request.profile.orientation_tolerance_rad
            ):
                raise ValueError(
                    "captured model FK does not match measured runtime TCP "
                    f"for {context.robot_id}: position_error_m="
                    f"{self.np.linalg.norm(delta[:3]):.6f}, "
                    f"orientation_error_rad={self.np.linalg.norm(delta[3:]):.6f}, "
                    f"tolerance_m={self.request.profile.position_tolerance_m:.6f}, "
                    f"tolerance_rad={self.request.profile.orientation_tolerance_rad:.6f}"
                )

    def _motion_radius_bound(self) -> float:
        # Every collision shape must supply its native conservative local AABB.
        # Summing ALL kinematic translations is conservative for every serial
        # link chain; include shape placements and attached-object offsets.
        chain = sum(
            float(self.np.linalg.norm(p.translation))
            for p in self.model.jointPlacements
        )
        for joint in self.model.joints[1:]:
            if joint.shortname().startswith("JointModelP"):
                chain += max(
                    abs(float(self._lower_limits[joint.idx_q])),
                    abs(float(self._upper_limits[joint.idx_q])),
                )
        maximum = 0.0
        for obj, (center, half) in zip(
            self.geometry.geometryObjects, self._local_collision_boxes
        ):
            if obj.name.startswith("world::"):
                continue
            extent = max(
                float(self.np.linalg.norm(p)) for p in (center - half, center + half)
            )
            maximum = max(
                maximum, extent + float(self.np.linalg.norm(obj.placement.translation))
            )
        radius = chain + maximum
        return finite(radius, "collision motion radius", positive=True)

    def check(
        self,
        joints: Mapping[str, float],
        *,
        holder: str,
        receiver_touch: bool = False,
        receiver_attachment: Pose | None = None,
        finger_contact: bool = False,
    ) -> CollisionVerdict:
        """Check all model bounds and active full-scene collision pairs.

        Returns:
            Collision pairs and minimum clearance, not a guessed true marker.
        """
        if self.request.binding_sha256 != self.binding:
            raise ValueError("isolated scene binding changed")
        if holder not in (self.request.donor.robot_id, self.request.receiver.robot_id):
            raise ValueError("unbound attachment holder")
        if type(receiver_touch) is not bool:
            raise ValueError("contact permission must be an explicit bool")
        if type(finger_contact) is not bool or (
            finger_contact and self.finger_contact_policy is None
        ):
            raise ValueError("finger contact requires a bound simulation policy")
        self._guarded_pairs.clear()
        self._finger_margins.clear()
        self._exact_pair_distances.clear()
        self._bind_holder(holder, receiver_attachment)
        q = self.configuration(joints)
        if self.np.any(q < self._lower_limits) or (self.np.any(q > self._upper_limits)):
            return CollisionVerdict(False, 0.0, (), "MODEL_JOINT_LIMIT_VIOLATION")
        allowed = {f"{holder}_gripper_link{i}" for i in (1, 2)}
        if receiver_touch:
            allowed |= {
                f"{self.request.receiver.robot_id}_gripper_link{i}" for i in (1, 2)
            }
        for index, pair in enumerate(self.geometry.collisionPairs):
            other = pair.second if pair.first == self._object_index else pair.first
            permitted = self._object_index in (pair.first, pair.second) and (
                self._link_names[other] in allowed
            )
            if permitted:
                self._geometry_data.deactivateCollisionPair(index)
            else:
                self._geometry_data.activateCollisionPair(index)
        self.pin.updateGeometryPlacements(
            self.model, self._data, self.geometry, self._geometry_data, q
        )
        boxes = []
        for pose, (center, half) in zip(
            self._geometry_data.oMg, self._local_collision_boxes
        ):
            world_center = pose.rotation @ center + pose.translation
            extent = self.np.abs(pose.rotation) @ half
            # Outward padding makes numerical roundoff conservative.
            boxes.append((world_center - extent - 1e-8, world_center + extent + 1e-8))
        collided = []
        attachment = (
            self._object_to_tcp
            if holder == self.request.donor.robot_id
            else receiver_attachment
        )
        object_pose = compose(self.tcp_pose(holder, joints), attachment)
        self._contact_object_pose = object_pose
        for i, pair in enumerate(self.geometry.collisionPairs):
            if not self._geometry_data.activeCollisionPairs[i]:
                continue
            amin, amax = boxes[pair.first]
            bmin, bmax = boxes[pair.second]
            gap = self.np.maximum(self.np.maximum(amin - bmax, bmin - amax), 0)
            lower = float(self.np.linalg.norm(gap))
            guarded = (
                finger_contact
                and frozenset(
                    (self._link_names[pair.first], self._link_names[pair.second])
                )
                in self._finger_pairs
            )
            if guarded:
                self._guarded_pairs.add(i)
            if lower > 0:
                # Mesh-derived boxes are supersets. Disjoint boxes prove
                # noncollision and this distance is a conservative LOWER bound,
                # not a substituted mesh or an optimistic exact clearance.
                self._pair_clearances[i] = lower
                continue
            if self.pin.computeCollision(self.geometry, self._geometry_data, i):
                permitted = False
                if guarded:
                    # Re-query with contacts enabled, not just a binary ACM
                    # exemption. A saturated/invalid contact set fails closed.
                    request = self._geometry_data.collisionRequests[i]
                    request.enable_contact, request.num_max_contacts = True, 128
                    self.pin.computeCollision(self.geometry, self._geometry_data, i)
                    result = self._geometry_data.collisionResults[i]
                    contacts = result.getContacts()
                    if contacts and len(contacts) < 128:
                        margins = []
                        bounds = (
                            self.request.object_geometry.minimum_m,
                            self.request.object_geometry.maximum_m,
                        )
                        for contact in contacts:
                            # This Coal build reports overlapping box contact
                            # depths NEGATIVELY; use the penetration magnitude.
                            depth = abs(
                                finite(
                                    float(contact.penetration_depth),
                                    "native finger penetration",
                                )
                            )
                            point = tuple(float(v) for v in contact.pos)
                            margins.append(
                                min(
                                    self.finger_contact_policy.max_penetration_m
                                    - max(0.0, depth),
                                    region_margin(
                                        point,
                                        object_pose,
                                        bounds,
                                        self.finger_contact_policy,
                                    ),
                                )
                            )
                        self._finger_margins[i] = min(margins)
                        permitted = self._finger_margins[i] >= 0
                if not permitted:
                    collided.append(
                        (
                            self.geometry.geometryObjects[pair.first].name,
                            self.geometry.geometryObjects[pair.second].name,
                        )
                    )
            self._pair_clearances[i] = float(
                self.pin.computeDistance(
                    self.geometry, self._geometry_data, i
                ).min_distance
            )
            self._exact_pair_distances.add(i)
        if collided:
            return CollisionVerdict(
                False, 0.0, tuple(collided), "FULL_GEOMETRY_COLLISION"
            )
        distances = [
            max(0.0, self._pair_clearances[i])
            if i in self._guarded_pairs
            else self._pair_clearances[i]
            for i in range(len(self.geometry.collisionPairs))
            if self._geometry_data.activeCollisionPairs[i]
        ]
        clearance = min(distances) if distances else 0.0
        if not math.isfinite(clearance) or clearance < 0:
            raise ValueError("native collision distance is unavailable or invalid")
        return CollisionVerdict(True, clearance, ())

    def _bind_holder(self, holder: str, receiver_attachment: Pose | None) -> None:
        # Before transfer the brick follows the measured giver grasp. After
        # transfer a caller must explicitly supply the measured receiver grasp;
        # an arbitrary receiver-relative attachment is never inferred here.
        if holder != self.request.donor.robot_id and receiver_attachment is None:
            raise ValueError(
                "receiver attachment requires an explicit verified transform"
            )
        relative = (
            self._object_to_tcp
            if holder == self.request.donor.robot_id
            else receiver_attachment
        )
        frame = self._frame(holder + "_rocobrick_tcp")
        lo, hi = self._object_bounds
        center = tuple((a + b) / 2 for a, b in zip(lo, hi))
        obj = self.geometry.geometryObjects[self._object_index]
        obj.parentJoint = self.model.frames[frame].parentJoint
        obj.parentFrame = frame
        obj.placement = self.model.frames[frame].placement * self._se3(
            compose(relative, Pose(center, (0, 0, 0, 1))),
        )
        self._reach_bound = max(self._reach_bound, self._motion_radius_bound())

    def motion_bound(
        self, initial: Mapping[str, float], final: Mapping[str, float]
    ) -> float:
        """Return a conservative maximum relative collision-shape displacement."""
        q0, q1 = self.configuration(initial), self.configuration(final)
        motion = 0.0
        for index in range(1, self.model.njoints):
            joint = self.model.joints[index]
            delta = abs(float(q1[joint.idx_q] - q0[joint.idx_q]))
            name = joint.shortname()
            if name.startswith("JointModelR"):
                motion += self._reach_bound * delta
            elif name.startswith("JointModelP"):
                motion += delta
            else:
                raise ValueError("unsupported joint type in swept-path proof")
        return 2.0 * motion

    def certify_segment(
        self,
        initial: Mapping[str, float],
        final: Mapping[str, float],
        *,
        deadline_s: float,
        clock: Callable[[], float],
        receiver_touch: bool = False,
        max_checks: int = 4096,
        holder: str | None = None,
        receiver_attachment: Pose | None = None,
        finger_contact: bool = False,
    ) -> bool:
        """Prove a linearly interpolated joint segment by clearance bounds.

        Returns:
            True only when every subsegment is clearance-certified. An
            exhausted depth/query/time budget is failure, not sampled success.
        """
        finite(deadline_s, "swept-path deadline")
        if (
            set(initial) != set(final)
            or type(max_checks) is not int
            or not 1 <= max_checks <= 65536
        ):
            raise ValueError("swept path joint binding/budget is invalid")
        queue = [(dict(initial), dict(final), 0)]
        self.last_certificate_failure = None
        self.last_clearance_failure = None
        self.last_contact_region_failure = None
        owner = self.request.donor.robot_id if holder is None else holder
        checks = 0
        while queue:
            if clock() >= deadline_s:
                self.last_certificate_failure = "SEGMENT_TIME_BUDGET"
                return False
            if checks >= max_checks:
                self.last_certificate_failure = "SEGMENT_CHECK_BUDGET"
                return False
            first, last, depth = queue.pop()
            checks += 1
            start = self.check(
                first,
                holder=owner,
                receiver_touch=receiver_touch,
                receiver_attachment=receiver_attachment,
                finger_contact=finger_contact,
            )
            if not start.valid:
                self.last_certificate_failure = start.reason
                return False
            bound = self.motion_bound(first, last)
            if self._pairwise_clearance_certified(first, last):
                continue
            if bound == 0.0 and self.tracking_error_rad == 0.0 and not finger_contact:
                # Identical valid geometry remains valid; zero clearance alone
                # does not authorize moving even infinitesimally into contact.
                continue
            if depth >= 24:
                self.last_certificate_failure = "SEGMENT_DEPTH_BUDGET"
                return False
            mid = {name: (first[name] + last[name]) / 2 for name in first}
            queue.extend(((first, mid, depth + 1), (mid, last, depth + 1)))
        final_verdict = self.check(
            final,
            holder=owner,
            receiver_touch=receiver_touch,
            receiver_attachment=receiver_attachment,
            finger_contact=finger_contact,
        )
        if not final_verdict.valid:
            self.last_certificate_failure = final_verdict.reason
        return final_verdict.valid

    def _pairwise_clearance_certified(self, first, last) -> bool:
        """Bound relative shape motion and tracking; common rigid ancestors cancel.

        Distances are from the preceding full-state check. Tracking applies to
        the twelve arm joints, not to fictitious rotary finger errors. Fixed
        relative geometry has no arm-tracking uncertainty.

        Returns:
            Whether every active pair clears its relative tracking/motion bound.
        """
        self.last_contact_region_failure = None
        attached = self.geometry.geometryObjects[self._object_index]
        key = (attached.parentJoint, tuple(attached.placement.translation))
        if self._relative_layout_key != key:
            radii = []
            for shape, (center, half) in zip(
                self.geometry.geometryObjects, self._local_collision_boxes
            ):
                radius = max(
                    float(self.np.linalg.norm(p))
                    for p in (center - half, center + half)
                ) + float(self.np.linalg.norm(shape.placement.translation))
                chain = {}
                joint = shape.parentJoint
                while joint:
                    chain[joint] = radius
                    radius += float(
                        self.np.linalg.norm(
                            self.model.jointPlacements[joint].translation
                        )
                    )
                    variable = self.model.joints[joint]
                    if variable.shortname().startswith("JointModelP"):
                        radius += max(
                            abs(float(self._lower_limits[variable.idx_q])),
                            abs(float(self._upper_limits[variable.idx_q])),
                        )
                    joint = int(self.model.parents[joint])
                radii.append(chain)
            self._relative_radii = tuple(radii)
            self._relative_layout_key = key
        for index, pair in enumerate(self.geometry.collisionPairs):
            if not self._geometry_data.activeCollisionPairs[index]:
                continue
            motion = 0.0
            first_radii = self._relative_radii[pair.first]
            second_radii = self._relative_radii[pair.second]
            for joint_id in first_radii.keys() ^ second_radii.keys():
                joint = self.model.joints[joint_id]
                name = self.model.names[joint_id]
                delta = abs(last[name] - first[name])
                if name in self._arm_variables and index not in self._guarded_pairs:
                    delta += 2 * self.tracking_error_rad
                if joint.shortname().startswith("JointModelR"):
                    # Each XOR joint moves exactly one shape. Its lever arm is
                    # the complete downstream chain + actual collision extent,
                    # not the radius of both robots plus their mount offsets.
                    motion += (
                        first_radii.get(joint_id, 0) + second_radii.get(joint_id, 0)
                    ) * delta
                elif joint.shortname().startswith("JointModelP"):
                    motion += delta
                else:
                    raise ValueError("unsupported joint type in tracking certificate")
            distance = self._pair_clearances[index]
            if distance <= motion + 1e-8 and index not in self._exact_pair_distances:
                # An AABB lower bound is sufficient for acceptance, never for
                # rejection. Refine ambiguous pairs against the actual meshes
                # at this checked state; keep the SAME relative motion/tracking
                # bound and all localized-contact conditions below.
                distance = finite(
                    float(
                        self.pin.computeDistance(
                            self.geometry, self._geometry_data, index
                        ).min_distance
                    ),
                    "native pair distance",
                )
                if distance < 0:
                    raise ValueError(
                        "native pair distance contradicts disjoint collision AABBs"
                    )
                self._pair_clearances[index] = distance
                self._exact_pair_distances.add(index)
            if index in self._guarded_pairs:
                # This pair has a DIFFERENT safety contract: nominal mesh
                # penetration/region bound + actual contact guard. Keep arm
                # tracking limits for controllers and all nonpermitted pairs.
                distance = self._finger_margins.get(
                    index,
                    max(0.0, distance) + self.finger_contact_policy.max_penetration_m,
                )
                # Crossing zero nominal clearance is permitted only while the
                # complete possible swept contact region remains localized.
                # Checking the current closest point alone would miss a new
                # contact appearing elsewhere on the same full finger mesh.
                if self._pair_clearances[
                    index
                ] <= motion + 1e-8 and not self._finger_region_certified(
                    pair, first, last
                ):
                    self.last_contact_region_failure = (
                        self.geometry.geometryObjects[pair.first].name,
                        self.geometry.geometryObjects[pair.second].name,
                    )
                    self.last_clearance_failure = (
                        self.geometry.geometryObjects[pair.first].name,
                        self.geometry.geometryObjects[pair.second].name,
                        distance,
                        motion,
                    )
                    return False
            if motion > 0 and distance <= motion + 1e-8:
                self.last_clearance_failure = (
                    self.geometry.geometryObjects[pair.first].name,
                    self.geometry.geometryObjects[pair.second].name,
                    distance,
                    motion,
                )
                return False
        return True

    def _finger_region_certified(self, pair, first, last) -> bool:
        """Bound ALL possible nominal contact points, not selected witnesses.

        Returns:
            Whether the intersection of object-relative swept shape AABBs is
            empty or wholly inside the allowed fingertip region.
        """
        object_frame = self._se3(self._contact_object_pose)
        boxes = []
        for index in (pair.first, pair.second):
            relative = object_frame.actInv(self._geometry_data.oMg[index])
            center, half = self._local_collision_boxes[index]
            local_center = relative.rotation @ center + relative.translation
            extent = self.np.abs(relative.rotation) @ half
            # All motion is bounded relative to the held object, so the
            # region proof stays valid when a giver/receiver moves in world.
            radii = self._relative_radii[index]
            object_radii = self._relative_radii[self._object_index]
            movement = 0.0
            for joint_id in radii.keys() ^ object_radii.keys():
                joint = self.model.joints[joint_id]
                name = self.model.names[joint_id]
                delta = abs(last[name] - first[name])
                if joint.shortname().startswith("JointModelR"):
                    movement += (
                        radii.get(joint_id, 0) + object_radii.get(joint_id, 0)
                    ) * delta
                elif joint.shortname().startswith("JointModelP"):
                    movement += delta
                else:
                    raise ValueError("unsupported joint type in contact region proof")
            boxes.append(
                (
                    local_center - extent - movement - 1e-8,
                    local_center + extent + movement + 1e-8,
                )
            )
        lo = self.np.maximum(boxes[0][0], boxes[1][0])
        hi = self.np.minimum(boxes[0][1], boxes[1][1])
        if self.np.any(lo > hi):
            return True
        minimum = (
            self.np.array(self.request.object_geometry.minimum_m)
            - self.finger_contact_policy.region_padding_m
        )
        maximum = (
            self.np.array(self.request.object_geometry.maximum_m)
            + self.finger_contact_policy.region_padding_m
        )
        return bool(self.np.all(lo >= minimum) and self.np.all(hi <= maximum))
