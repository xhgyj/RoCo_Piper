"""Per-robot ROS/MoveIt runtime naming contract (A3).

One container session hosts one namespaced stack per robot id
(``robot_state_publisher`` + ``controller_manager`` + ``move_group``). Every
per-robot name is derived from the robot id, so two profiles never share a
topic, action, service, joint, frame, or planning group:

* endpoints live under the robot namespace ``/<robot_id>``; ``/clock`` stays
  global because sim time is shared by the whole session/DDS domain;
* joints, links, and MoveIt groups/controllers are prefixed with
  ``<robot_id>_`` in the generated model (the vendor description cannot be
  renamed in place), matching the ``TopicBasedSystem`` JointState names
  ``<robot_id>_<joint>`` frozen with the host bridge;
* ``tip_link`` is the project-calibrated zero-geometry ``rocobrick_tcp``
  frame, not the raw ``link6`` flange; ``planning_group`` is the calibrated
  ``<robot_id>_arm_tcp`` SRDF chain group. ``link6_link`` and
  ``link6_planning_group`` carry the legacy raw-flange chain names for the
  backward-compatible ``Link6PoseTarget`` path, so no caller ever has to
  reconstruct the ``<robot_id>_`` prefix itself. Both are defaulted and
  derived from the robot id/namespace when omitted, so constructing a
  profile with the pre-A3 field set stays valid.

``ros_ws/src/rocobrick_piper_l_moveit_config/launch/dual_robot_config.py``
derives the same names for the generated model/config files and
``tests/test_dual_robot_profile_launch.py`` pins both sides equal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rocobrick.moveit_contract import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    JOINT_NAMES,
    TCP_LINK,
)

# Same ROS-safe identifier rule as the config generator
# (ros_ws/.../launch/dual_robot_config.py): the robot id becomes a namespace,
# a model-name prefix, and a file-system segment of the generated configs.
_ROBOT_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _robot_id(value: str) -> str:
    if not isinstance(value, str) or not _ROBOT_ID_PATTERN.match(value):
        raise ValueError("robot_id must be a ROS-safe identifier")
    return value


def _ns(value: str) -> str:
    if not isinstance(value, str) or not value or not value.startswith("/"):
        raise ValueError("namespace must be an absolute non-empty ROS namespace")
    if value != "/" and value.endswith("/"):
        value = value.rstrip("/")
    if "//" in value or any(not (c.isalnum() or c in "_/-") for c in value):
        raise ValueError("namespace contains invalid ROS name characters")
    return value


@dataclass(frozen=True)
class RobotRuntimeProfile:
    """Frozen per-robot naming/routing contract of one container-side stack.

    Built through :meth:`from_robot_id` (namespaced dual-robot model) or
    :meth:`canonical` (pinned single-robot model); direct construction stays
    valid for the pre-A3 field set because the legacy ``link6_*`` fields are
    derived when omitted.
    """

    robot_id: str
    namespace: str
    joint_names: tuple[str, ...]
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    clock: str
    isaac_joint_states: str
    joint_states: str
    isaac_joint_command: str
    move_group_action: str
    arm_follow: str
    gripper_follow: str
    list_controllers: str
    plan: str
    compute_fk: str
    base_frame: str
    tip_link: str
    planning_group: str
    # Legacy raw-flange chain names. Defaulted and derived in __post_init__
    # from the robot id/namespace so a caller that builds a profile with the
    # pre-A3 field set keeps working (the explicit factories always pass the
    # values they freeze).
    link6_link: str = ""
    link6_planning_group: str = ""

    def __post_init__(self) -> None:
        """Validate the contract and derive omitted legacy-chain names."""
        _robot_id(self.robot_id)
        if (
            len(self.joint_names) != 8
            or len(self.arm_joint_names) != 6
            or len(self.gripper_joint_names) != 2
        ):
            raise ValueError("Piper profile must contain 6 arm and 2 gripper joints")
        if self.joint_names != self.arm_joint_names + self.gripper_joint_names:
            raise ValueError("joint slices must reconstruct joint_names")
        for name in (
            self.clock, self.isaac_joint_states, self.joint_states,
            self.isaac_joint_command, self.move_group_action, self.arm_follow,
            self.gripper_follow, self.list_controllers, self.plan, self.compute_fk,
        ):
            if not name.startswith("/"):
                raise ValueError("runtime names must be absolute")
        # Same derivation as the classmethods: a root-namespace profile uses
        # the un-prefixed canonical chain, a namespaced one the robot prefix.
        prefix = "" if self.namespace in ("", "/") else self.robot_id + "_"
        if not self.link6_link:
            object.__setattr__(self, "link6_link", prefix + "link6")
        if not self.link6_planning_group:
            object.__setattr__(self, "link6_planning_group", prefix + "arm")
        for name in (
            self.base_frame, self.tip_link, self.planning_group,
            self.link6_link, self.link6_planning_group,
        ):
            if not name or name.startswith("/"):
                raise ValueError(
                    "frames and planning groups must be relative ROS names"
                )

    @classmethod
    def from_robot_id(
        cls, robot_id: str, namespace: str | None = None
    ) -> "RobotRuntimeProfile":
        """Namespaced profile of the ``<robot_id>_``-prefixed generated model.

        Returns:
            The frozen per-robot contract under ``namespace`` (default
            ``/<robot_id>``) with prefixed joints, frames, and groups.
        """
        _robot_id(robot_id)
        ns = _ns(namespace if namespace is not None else "/" + robot_id)
        prefix = robot_id + "_"
        joints = tuple(prefix + name for name in JOINT_NAMES)
        root = ns
        return cls(
            robot_id=robot_id,
            namespace=ns,
            joint_names=joints,
            arm_joint_names=joints[:6],
            gripper_joint_names=joints[6:],
            clock="/clock",
            isaac_joint_states=root + "/isaac_joint_states",
            joint_states=root + "/joint_states",
            isaac_joint_command=root + "/isaac_joint_command",
            move_group_action=root + "/move_action",
            arm_follow=root + "/arm_controller/follow_joint_trajectory",
            gripper_follow=root + "/gripper_controller/follow_joint_trajectory",
            list_controllers=root + "/controller_manager/list_controllers",
            plan=root + "/plan_kinematic_path",
            compute_fk=root + "/compute_fk",
            base_frame=prefix + "base_link",
            # Calibrated skill TCP (R5.A): the generated
            # link6 -> rocobrick_tcp overlay is renamed with the same prefix.
            tip_link=prefix + TCP_LINK,
            planning_group=prefix + "arm_tcp",
            link6_link=prefix + "link6",
            link6_planning_group=prefix + "arm",
        )

    @classmethod
    def canonical(cls, robot_id: str = "piper_0") -> "RobotRuntimeProfile":
        """Single-robot (root namespace) profile of the pinned P1 stack.

        The pinned model carries the generated ``rocobrick_tcp`` overlay, so
        the canonical profile exposes the same calibrated skill TCP as the
        namespaced profiles; the raw ``link6`` flange is not a skill endpoint.

        Returns:
            The root-namespace contract of the pinned single-robot stack.
        """
        return cls(
            robot_id=robot_id,
            namespace="",
            joint_names=JOINT_NAMES,
            arm_joint_names=ARM_JOINT_NAMES,
            gripper_joint_names=GRIPPER_JOINT_NAMES,
            clock="/clock",
            isaac_joint_states="/isaac_joint_states",
            joint_states="/joint_states",
            isaac_joint_command="/isaac_joint_command",
            move_group_action="/move_action",
            arm_follow="/arm_controller/follow_joint_trajectory",
            gripper_follow="/gripper_controller/follow_joint_trajectory",
            list_controllers="/controller_manager/list_controllers",
            plan="/plan_kinematic_path",
            compute_fk="/compute_fk",
            base_frame="base_link",
            tip_link=TCP_LINK,
            planning_group="arm_tcp",
            link6_link="link6",
            link6_planning_group="arm",
        )


def assert_profiles_disjoint(profiles: tuple[RobotRuntimeProfile, ...]) -> None:
    """Fail if two profiles share any routing, joint, frame, or group name.

    Endpoint names are unique because the namespaces are distinct; joint,
    link/frame, and planning-group names are unique because the generated
    models carry the ``<robot_id>_`` prefix. A collision here means a goal or
    a state update of one robot could resolve against the other robot's
    stack, so the caller must fail closed instead of launching.
    """
    names: dict[str, str] = {}
    owners: set[str] = set()
    shared: set[str] = set()
    for profile in profiles:
        if profile.robot_id in owners:
            raise ValueError(f"duplicate robot id {profile.robot_id!r} in profiles")
        owners.add(profile.robot_id)
        if not shared:
            shared.add(profile.clock)
        elif profile.clock not in shared:
            raise ValueError("profiles disagree on the shared session clock")
        checked = (
            profile.isaac_joint_states, profile.joint_states,
            profile.isaac_joint_command, profile.move_group_action,
            profile.arm_follow, profile.gripper_follow,
            profile.list_controllers, profile.plan, profile.compute_fk,
            profile.base_frame, profile.tip_link, profile.planning_group,
            profile.link6_link, profile.link6_planning_group,
            *profile.joint_names,
        )
        for name in checked:
            owner = names.setdefault(name, profile.robot_id)
            if owner != profile.robot_id:
                raise ValueError(f"profiles share runtime name {name!r}")
