"""Read only the current PhysX step's identified finger-pad/object contacts.

This is separate from BrickSim's stud/hole assembly forces. No persistent
contact cache, pose-derived contact, or planning collision is a substitute
for a contact impulse reported in this step by the actual physics engine.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from rocobrick.skills.handover_contact import CONTACT_SCHEMA as FINGER_CONTACT_SCHEMA
from rocobrick.skills.handover_observation import CONTACT_SCHEMA, PadContact
from rocobrick.skills.handover_profile import digest, finite


def _triple(raw: object, label: str) -> tuple[float, float, float]:
    try:
        values = tuple(raw)
    except TypeError as error:
        raise ValueError(f"{label} is not a vector") from error
    if len(values) != 3:
        raise ValueError(f"{label} must have three components")
    return tuple(finite(v, label) for v in values)


@dataclass(frozen=True)
class PadContactFrame:
    """One captured physics step, never a last-known-contact fallback."""

    sim_time_s: float
    contacts: tuple[PadContact, ...]
    finger_contacts: tuple[Mapping[str, object], ...] = ()
    dt_s: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid times and untyped or mutable contact buffers."""
        if finite(self.sim_time_s, "physics time") < 0 or (
            not isinstance(self.contacts, tuple)
            or any(not isinstance(c, PadContact) for c in self.contacts)
        ):
            raise ValueError("invalid physics contact frame")

    def document(
        self,
        *,
        sim_time_s: float,
        sequence: int,
        session_id: str,
        manifest_sha256: str,
    ) -> Mapping[str, object]:
        """Bind the captured step to the exact shared scene query.

        Returns:
            The strict additive pad-contact report.
        """
        if finite(sim_time_s, "scene time") != self.sim_time_s:
            raise ValueError("pad contact report is from a different physics step")
        if (
            type(sequence) is not int
            or sequence < 1
            or (not isinstance(session_id, str) or not session_id)
        ):
            raise ValueError("pad contact frame identity is invalid")
        digest(manifest_sha256, "manifest")
        return {
            "schema": CONTACT_SCHEMA,
            "source": "physx_contact_report",
            "session_id": session_id,
            "manifest_sha256": manifest_sha256,
            "sequence": sequence,
            "sim_time_s": sim_time_s,
            "contacts": [
                {
                    "robot_id": c.robot_id,
                    "pad_link": c.pad_link,
                    "physical_instance_id": c.physical_instance_id,
                    "position_world_m": list(c.position_world_m),
                    "normal_world": list(c.normal_world),
                    "normal_force_n": c.normal_force_n,
                }
                for c in self.contacts
            ],
        }

    def finger_document(self) -> Mapping[str, object]:
        """Separate collision evidence from pad/object custody evidence.

        Returns:
            This step's independent finger collision document.
        """
        finite(self.dt_s, "finger physics step", positive=True)
        return {
            "sim_time_s": self.sim_time_s,
            "dt_s": self.dt_s,
            "contacts": [dict(c) for c in self.finger_contacts],
        }


def finger_history_document(
    frames, *, sim_time_s, sequence, session_id, manifest_sha256
):
    """Include each intervening step, including empty frames, not a peak cache.

    Returns:
        A scene-bound, complete recent physics-step history.
    """
    if not frames or frames[-1].sim_time_s != sim_time_s:
        raise ValueError("finger history does not match current physics tick")
    return {
        "schema": FINGER_CONTACT_SCHEMA,
        "source": "physx_contact_report",
        "sim_time_s": sim_time_s,
        "sequence": sequence,
        "session_id": session_id,
        "manifest_sha256": manifest_sha256,
        "frames": [dict(f.finger_document()) for f in frames],
    }


class PhysxPadContactReader:
    """Map actual actor identities and impulses to named pad contact reports."""

    def __init__(
        self,
        interface: object,
        *,
        decode_path: Callable[[object], str],
        pads: Mapping[str, tuple[str, str]],
        objects: Mapping[str, str],
        contact_lost: object,
        include_finger_contacts: bool = False,
    ) -> None:
        """Bind exact USD rigid actors, not substring or collider guesses."""
        if not pads or not objects or set(pads) & set(objects):
            raise ValueError("pad/object actor maps must be nonempty and disjoint")
        if len(set(objects.values())) != len(objects):
            raise ValueError("each physical object needs one exact rigid actor")
        for path, (robot, pad) in pads.items():
            if not path.startswith("/") or pad not in {
                f"{robot}_gripper_link1",
                f"{robot}_gripper_link2",
            }:
                raise ValueError("invalid pad actor binding")
        if not callable(getattr(interface, "get_contact_report", None)):
            raise ValueError("PhysX current-step contact reporting is unavailable")
        self._interface = interface
        self._decode_path = decode_path
        self._pads = dict(pads)
        self._objects = dict(objects)
        self._lost = contact_lost
        if type(include_finger_contacts) is not bool:
            raise ValueError("finger reporting opt-in must be an explicit bool")
        self._include_finger_contacts = include_finger_contacts

    @classmethod
    def from_stage(
        cls,
        stage: object,
        *,
        robot_paths: Mapping[str, str],
        object_paths: Mapping[str, str],
        include_finger_contacts: bool = False,
    ) -> PhysxPadContactReader:
        """Enable real per-pad reports lazily within the existing Isaac host.

        Returns:
            A reader on the existing physics interface, with no new DDS node.
        """
        import omni.physx
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics

        # Contact positions/impulses are in stage units. Reject unsupported
        # scaling instead of labelling centimetres or non-SI impulses as SI.
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or (
            UsdPhysics.GetStageKilogramsPerUnit(stage) != 1.0
        ):
            raise ValueError("pad contact reports require verified SI stage units")
        pads = {}
        for robot, root in robot_paths.items():
            for index in (1, 2):
                path = f"{root}/gripper_link{index}"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(f"pad rigid actor unavailable: {path}")
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                api.CreateThresholdAttr().Set(0.0)
                pads[path] = (robot, f"{robot}_gripper_link{index}")
        objects = {path: physical for physical, path in object_paths.items()}
        for path in objects:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"object rigid actor unavailable: {path}")
        return cls(
            omni.physx.get_physx_simulation_interface(),
            decode_path=lambda raw: str(PhysicsSchemaTools.intToSdfPath(raw)),
            pads=pads,
            objects=objects,
            contact_lost=ContactEventType.CONTACT_LOST,
            include_finger_contacts=include_finger_contacts,
        )

    def capture(self, *, sim_time_s: float, dt_s: float) -> PadContactFrame:
        """Read the current contact buffer once, immediately after world.step.

        Returns:
            Identified actual contacts for this step only, including empty sets.
        """
        time = finite(sim_time_s, "physics time")
        dt = finite(dt_s, "physics dt", positive=True)
        read = getattr(self._interface, "get_contact_report")
        report = read()
        if not isinstance(report, (tuple, list)) or len(report) not in (2, 3):
            raise ValueError("PhysX contact buffer response is malformed")
        headers, data = report[:2]
        contacts = []
        finger_contacts = []
        for header in headers:
            if getattr(header, "type") == self._lost:
                continue
            actor0 = self._decode_path(getattr(header, "actor0"))
            actor1 = self._decode_path(getattr(header, "actor1"))
            if actor0 in self._pads and actor1 in self._objects:
                pad_actor, object_actor, sign = actor0, actor1, 1.0
            elif actor1 in self._pads and actor0 in self._objects:
                pad_actor, object_actor, sign = actor1, actor0, -1.0
            elif self._include_finger_contacts and (
                actor0 in self._pads or actor1 in self._pads
            ):
                # Retain unexpected pad/wrist/environment contacts too, so the
                # handler can reject them instead of treating them as silence.
                pad_actor = actor0 if actor0 in self._pads else actor1
                object_actor, sign = None, 1.0
            else:
                continue
            offset = getattr(header, "contact_data_offset")
            count = getattr(header, "num_contact_data")
            if (
                type(offset) is not int
                or type(count) is not int
                or (offset < 0 or count < 0 or offset + count > len(data))
            ):
                raise ValueError("PhysX contact range is invalid")
            robot, pad = self._pads[pad_actor]
            for point in data[offset : offset + count]:
                normal = _triple(getattr(point, "normal"), "contact normal")
                norm = math.sqrt(sum(v * v for v in normal))
                if abs(norm - 1.0) > 1e-5:
                    raise ValueError("PhysX contact normal is not a unit vector")
                normal = tuple(v / norm for v in normal)
                impulse = _triple(getattr(point, "impulse"), "contact impulse")
                # PhysX normals point shape1 -> shape0. Make all emitted
                # normals object -> pad irrespective of actor ordering.
                force = abs(sum(a * b for a, b in zip(impulse, normal))) / dt
                if object_actor is None:
                    finger_contacts.append(
                        {
                            "link_a": self._pads[actor0][1]
                            if actor0 in self._pads
                            else actor0,
                            "link_b": self._pads[actor1][1]
                            if actor1 in self._pads
                            else actor1,
                            "position_world_m": list(
                                _triple(getattr(point, "position"), "contact position")
                            ),
                            "normal_force_n": force,
                            "impulse_ns": math.sqrt(sum(v * v for v in impulse)),
                            "separation_m": finite(
                                getattr(point, "separation"), "finger separation"
                            ),
                        }
                    )
                    continue
                if force <= 0:
                    continue
                contacts.append(
                    PadContact(
                        robot,
                        pad,
                        self._objects[object_actor],
                        _triple(getattr(point, "position"), "contact position"),
                        tuple(sign * v for v in normal),
                        force,
                    )
                )
        return PadContactFrame(time, tuple(contacts), tuple(finger_contacts), dt)
