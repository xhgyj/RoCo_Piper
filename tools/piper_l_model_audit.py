"""P0.4 minimal Piper_L model contract audit (offline, deterministic).

Reads only the pinned official checkouts recorded in
``third_party/manifest.lock.yaml`` plus the runtime robot assets already in
the worktree, and writes ``artifacts/model-audit/piper_l_model_contract.json``.

No network access. USD facts are extracted with the pxr USD python modules
shipped inside the pinned ``isaacsim==5.1.0`` wheel when those are importable
in this venv; when they are not, the USD section records hashes only and no
composed-USD value is claimed. Runtime articulation DOF order is never read
from the USD because USD does not store it; the expected order is derived
from evidence (URDF movable-joint declaration order, config ``Joint_Order``,
and the bootstrap that verifies ``articulation.dof_names`` at runtime).

Usage: ``uv run python tools/piper_l_model_audit.py [--output PATH]``
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "third_party/manifest.lock.yaml"
AGX_ARM_ROS_DIR = ROOT / "ros_ws/src/agx_arm_ros"
AGX_ARM_SIM_DIR = ROOT / "isaac/vendor/agx_arm_sim"
AGX_ARM_URDF_DIR = AGX_ARM_ROS_DIR / "src/agx_arm_description/agx_arm_urdf"

PIPER_L_URDF = AGX_ARM_URDF_DIR / "piper_l/urdf/piper_l_description.urdf"
PIPER_L_GRIPPER_XACRO = (
    AGX_ARM_URDF_DIR / "piper_l/urdf/piper_l_with_gripper_description.xacro"
)
COMPOSED_URDF = ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf"
USER_CONFIG = ROOT / "config/user_config.json"
MOVEIT_CONFIG_DIR = AGX_ARM_SIM_DIR / "Moveit2/piper_l_gripper_moveit_config/config"
USD_TOP = ROOT / "robot_assets/piper_l/USD/piper_l_gripper_d435.usd"
USD_CONFIGURATION_DIR = ROOT / "robot_assets/piper_l/USD/configuration"

# The pinned agx_arm_sim files the runtime robot_assets assets must stay
# byte-identical to. This repository's URDF/USD pair is the execution source
# of record: it is internally consistent, while the agx_arm_urdf submodule
# carries wider joint4/joint5 limits that no official USD confirms.
AGX_ARM_SIM_URDF = (
    AGX_ARM_SIM_DIR / "agx_arm_description/urdf/piper_l_gripper_d435.urdf"
)
AGX_ARM_SIM_USD_TOP = (
    AGX_ARM_SIM_DIR
    / "agx_arm_description/urdf/USD/piper_l_gripper_d435"
    / "piper_l_gripper_d435.usd"
)

# The agx_arm_urdf submodule pin required for the model contract. Recorded
# explicitly so a moved pin fails the audit instead of silently re-auditing
# different model data.
AGX_ARM_URDF_PIN = "f6642ce0d7872c686f29c99e9e10cd23d1d49313"

ARM_JOINT_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["gripper_joint1", "gripper_joint2"]
EXPECTED_DOF_ORDER = ARM_JOINT_ORDER + GRIPPER_JOINTS

HASHED_FILES = [
    PIPER_L_URDF,
    PIPER_L_GRIPPER_XACRO,
    AGX_ARM_SIM_URDF,
    COMPOSED_URDF,
    USER_CONFIG,
    USD_TOP,
    AGX_ARM_SIM_USD_TOP,
    *sorted(USD_CONFIGURATION_DIR.glob("*.usd")),
    MOVEIT_CONFIG_DIR / "piper_l.srdf",
    MOVEIT_CONFIG_DIR / "piper_l.ros2_control.xacro",
    MOVEIT_CONFIG_DIR / "joint_limits.yaml",
    MOVEIT_CONFIG_DIR / "moveit_controllers.yaml",
    MOVEIT_CONFIG_DIR / "ros2_controllers.yaml",
    MOVEIT_CONFIG_DIR / "kinematics.yaml",
    MOVEIT_CONFIG_DIR / "initial_positions.yaml",
    MOVEIT_CONFIG_DIR / "pilz_cartesian_limits.yaml",
    AGX_ARM_URDF_DIR / "LICENSE",
    AGX_ARM_ROS_DIR / "LICENSE",
    AGX_ARM_SIM_DIR / "README.md",
]


def sha256(path: Path) -> str:
    """Return the hex sha256 digest of ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str, cwd: Path) -> str:
    """Run a local git command inside ``cwd``.

    Returns:
        Stripped stdout of the git command.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sim_submodule_dir() -> Path:
    """Return the (possibly empty) agx_arm_urdf path inside agx_arm_sim."""
    return AGX_ARM_SIM_DIR / "agx_arm_description/agx_arm_urdf"


def _sim_submodule_initialized() -> bool:
    """Report whether the agx_arm_sim submodule checkout is populated.

    Returns:
        True only if the submodule directory exists and is non-empty.
    """
    directory = _sim_submodule_dir()
    return directory.is_dir() and any(directory.iterdir())


def _origin(path: Path) -> dict:
    return {
        "path": _relative(path),
        "sha256": sha256(path),
    }


def parse_joint(element: ET.Element) -> dict:
    """Flatten one URDF ``<joint>`` element into a plain dict.

    Returns:
        The joint name, type, and any present axis/origin/limit/mimic data.
    """
    axis_element = element.find("axis")
    limit_element = element.find("limit")
    origin_element = element.find("origin")
    parent_element = element.find("parent")
    child_element = element.find("child")
    mimic_element = element.find("mimic")
    joint: dict = {
        "name": element.attrib["name"],
        "type": element.attrib["type"],
    }
    if axis_element is not None:
        joint["axis"] = [
            float(v) for v in axis_element.attrib["xyz"].split()
        ]
    if origin_element is not None:
        joint["origin_xyz"] = [
            float(v) for v in origin_element.attrib.get("xyz", "0 0 0").split()
        ]
        joint["origin_rpy"] = [
            float(v) for v in origin_element.attrib.get("rpy", "0 0 0").split()
        ]
    if parent_element is not None:
        joint["parent_link"] = parent_element.attrib["link"]
    if child_element is not None:
        joint["child_link"] = child_element.attrib["link"]
    if limit_element is not None:
        joint["limit"] = {
            key: float(limit_element.attrib[key])
            for key in ("lower", "upper", "effort", "velocity")
            if key in limit_element.attrib
        }
    if mimic_element is not None:
        joint["mimic"] = {
            "joint": mimic_element.attrib["joint"],
            "multiplier": float(mimic_element.attrib["multiplier"]),
            "offset": float(mimic_element.attrib.get("offset", "0.0")),
        }
    return joint


def parse_urdf_joints(path: Path) -> dict[str, dict]:
    """Parse a URDF/xacro file into a name-to-joint mapping.

    Returns:
        Every ``<joint>`` element keyed by joint name.
    """
    root = ET.parse(path).getroot()
    return {
        joint["name"]: joint
        for joint in (parse_joint(element) for element in root.findall("joint"))
    }


def extract_usd_facts() -> dict:
    """Extract composed-USD facts with the pinned isaacsim pxr modules.

    Returns:
        A dict with an ``extraction`` entry describing what happened. Fails
        soft: any import or stage error downgrades the section to hashes-only
        evidence instead of fabricating values.
    """
    pxr_root_candidates = sorted(
        (ROOT / ".venv/lib").glob(
            "python*/site-packages/isaacsim/extscache/omni.usd.libs-*"
        )
    )
    extraction: dict = {
        "method": "pxr_usd_python",
        "source": "isaacsim==5.1.0 extscache omni.usd.libs (pinned wheel, offline)",
        "requires_runtime": False,
    }
    if not pxr_root_candidates:
        return {"extraction": {"status": "unavailable", **extraction}}
    pxr_root = pxr_root_candidates[-1]
    bin_dir = pxr_root / "bin"
    ld_paths = [str(bin_dir)] if bin_dir.is_dir() else []
    # libpython3.11.so.1.0 sits next to the interpreter prefix, not on the
    # default loader path for uv-managed CPython installs.
    for prefix in {
        Path(sys.base_prefix).resolve(),
        Path(sys.executable).resolve().parent.parent,
    }:
        lib_dir = prefix / "lib"
        if lib_dir.is_dir():
            ld_paths.append(str(lib_dir))
    for existing in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if existing:
            ld_paths.append(existing)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(pxr_root)!r})\n"
        "from pxr import Usd, UsdGeom, UsdPhysics\n"
        f"stage = Usd.Stage.Open({str(USD_TOP)!r})\n"
        "facts = {}\n"
        "root = stage.GetDefaultPrim()\n"
        "facts['default_prim'] = root.GetPath().pathString\n"
        "facts['up_axis'] = UsdGeom.GetStageUpAxis(stage)\n"
        "facts['meters_per_unit'] = float(UsdGeom.GetStageMetersPerUnit(stage))\n"
        "facts['used_layers'] = sorted(\n"
        "    layer.realPath for layer in stage.GetUsedLayers() if layer.realPath\n"
        ")\n"
        "facts['movable_joints'] = []\n"
        "facts['articulation_root_prims'] = []\n"
        "movable = ('PhysicsRevoluteJoint', 'PhysicsPrismaticJoint')\n"
        "for prim in stage.Traverse():\n"
        "    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):\n"
        "        facts['articulation_root_prims'].append(prim.GetPath().pathString)\n"
        "    if prim.GetTypeName() not in movable:\n"
        "        continue\n"
        "    def get(name):\n"
        "        attr = prim.GetAttribute(name)\n"
        "        if not attr.HasValue():\n"
        "            return None\n"
        "        return attr.Get()\n"
        "    drive = (\n"
        "        'angular'\n"
        "        if prim.GetTypeName() == 'PhysicsRevoluteJoint'\n"
        "        else 'linear'\n"
        "    )\n"
        "    facts['movable_joints'].append({\n"
        "        'name': prim.GetPath().name,\n"
        "        'usd_path': prim.GetPath().pathString,\n"
        "        'usd_type': prim.GetTypeName(),\n"
        "        'axis': get('physics:axis'),\n"
        "        'lower_limit': get('physics:lowerLimit'),\n"
        "        'upper_limit': get('physics:upperLimit'),\n"
        "        'units': 'deg' if drive == 'angular' else 'm',\n"
        "        'drive': {\n"
        "            'stiffness': get(f'drive:{drive}:physics:stiffness'),\n"
        "            'damping': get(f'drive:{drive}:physics:damping'),\n"
        "            'max_force': get(f'drive:{drive}:physics:maxForce'),\n"
        "        },\n"
        "    })\n"
        "print(json.dumps(facts))\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(code)
        script_path = handle.name
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        reason = stderr.splitlines()[-1] if stderr else "unknown error"
        return {
            "extraction": {"status": "unavailable", "reason": reason, **extraction}
        }
    except (subprocess.SubprocessError, OSError) as error:
        return {
            "extraction": {"status": "unavailable", "reason": str(error), **extraction}
        }
    finally:
        Path(script_path).unlink(missing_ok=True)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return {"extraction": {"status": "ok", **extraction}, **payload}


def build_artifact() -> dict:
    """Collect the P0.4 model contract facts from the pinned sources.

    Returns:
        The artifact dict ready for JSON serialization.
    """
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    repositories = manifest["repositories"]
    agx_arm_ros_commit = repositories["agx_arm_ros"]["commit"]
    agx_arm_sim_commit = repositories["agx_arm_sim"]["commit"]

    # --- provenance: pinned checkouts and submodules -----------------------
    agx_arm_ros_head = git("rev-parse", "HEAD", cwd=AGX_ARM_ROS_DIR)
    agx_arm_sim_head = git("rev-parse", "HEAD", cwd=AGX_ARM_SIM_DIR)
    urdf_checkout_head = git("rev-parse", "HEAD", cwd=AGX_ARM_URDF_DIR)
    ros_gitlink = git(
        "ls-tree", "HEAD", "src/agx_arm_description/agx_arm_urdf", cwd=AGX_ARM_ROS_DIR
    ).split()[2]
    sim_gitlink_line = git(
        "ls-tree", "HEAD", "agx_arm_description/agx_arm_urdf", cwd=AGX_ARM_SIM_DIR
    )
    sim_gitlink = sim_gitlink_line.split()[2] if sim_gitlink_line else None
    if agx_arm_ros_head != agx_arm_ros_commit:
        raise SystemExit(
            f"agx_arm_ros checkout {agx_arm_ros_head} != pinned {agx_arm_ros_commit}"
        )
    if agx_arm_sim_head != agx_arm_sim_commit:
        raise SystemExit(
            f"agx_arm_sim checkout {agx_arm_sim_head} != pinned {agx_arm_sim_commit}"
        )
    if urdf_checkout_head != AGX_ARM_URDF_PIN or ros_gitlink != AGX_ARM_URDF_PIN:
        raise SystemExit(
            f"agx_arm_urdf pin drift: checkout {urdf_checkout_head}, "
            f"gitlink {ros_gitlink}, required {AGX_ARM_URDF_PIN}"
        )

    file_sha256 = {_relative(path): sha256(path) for path in HASHED_FILES}

    # --- official arm joints from the pinned agx_arm_urdf submodule --------
    arm_joints = []
    for name in ARM_JOINT_ORDER:
        joint = parse_urdf_joints(PIPER_L_URDF)[name]
        arm_joints.append(
            {
                "name": joint["name"],
                "type": joint["type"],
                "axis": joint["axis"],
                "limit": joint["limit"],
                "parent_link": joint["parent_link"],
                "child_link": joint["child_link"],
                "origin_xyz": joint["origin_xyz"],
                "origin_rpy": joint["origin_rpy"],
                "source": _origin(PIPER_L_URDF),
            }
        )

    # --- joint4/5 limit conflict between the two official repositories ----
    # agx_arm_urdf (input of the official MoveIt xacro) carries wider
    # joint4/5 limits than the agx_arm_sim URDF/USD pair. Record the conflict
    # and pin the execution limits to the intersection (the agx_arm_sim
    # values, the narrower set).
    sim_urdf_joints = parse_urdf_joints(AGX_ARM_SIM_URDF)
    submodule_by_name = {joint["name"]: joint for joint in arm_joints}
    limit_conflicts = []
    for name in ARM_JOINT_ORDER:
        submodule_limit = submodule_by_name[name]["limit"]
        sim_limit = sim_urdf_joints[name]["limit"]
        same = math.isclose(
            submodule_limit["lower"], sim_limit["lower"], abs_tol=1e-9
        ) and math.isclose(
            submodule_limit["upper"], sim_limit["upper"], abs_tol=1e-9
        )
        if not same:
            limit_conflicts.append(
                {
                    "joint": name,
                    "agx_arm_urdf_rad": [
                        submodule_limit["lower"],
                        submodule_limit["upper"],
                    ],
                    "agx_arm_sim_rad": [
                        sim_limit["lower"],
                        sim_limit["upper"],
                    ],
                }
            )

    # --- runtime robot_assets vs pinned agx_arm_sim byte identity ---------
    def _identity(runtime_path: Path, vendor_path: Path) -> dict:
        runtime_hash = sha256(runtime_path)
        vendor_hash = sha256(vendor_path)
        return {
            "runtime_path": _relative(runtime_path),
            "vendor_path": _relative(vendor_path),
            "runtime_sha256": runtime_hash,
            "vendor_sha256": vendor_hash,
            "byte_identical": runtime_hash == vendor_hash,
        }

    vendor_configuration_dir = AGX_ARM_SIM_USD_TOP.parent / "configuration"
    vendor_files = [
        AGX_ARM_SIM_URDF,
        AGX_ARM_SIM_USD_TOP,
        *sorted(vendor_configuration_dir.glob("*.usd")),
    ]
    for path in vendor_files:
        relative = path.relative_to(AGX_ARM_SIM_DIR).as_posix()
        checked_out_blob = git("hash-object", str(path), cwd=AGX_ARM_SIM_DIR)
        pinned_blob = git("rev-parse", f"HEAD:{relative}", cwd=AGX_ARM_SIM_DIR)
        if checked_out_blob != pinned_blob:
            raise SystemExit(f"pinned agx_arm_sim asset was modified: {relative}")
    runtime_layers = sorted(USD_CONFIGURATION_DIR.glob("*.usd"))
    vendor_layers = sorted(vendor_configuration_dir.glob("*.usd"))
    if {path.name for path in runtime_layers} != {path.name for path in vendor_layers}:
        raise SystemExit("runtime USD configuration layers differ from pinned agx_arm_sim")
    vendor_identity = {
        "urdf": _identity(COMPOSED_URDF, AGX_ARM_SIM_URDF),
        "usd_top": _identity(USD_TOP, AGX_ARM_SIM_USD_TOP),
        "usd_configuration_layers": {
            path.name: _identity(path, vendor_configuration_dir / path.name)
            for path in runtime_layers
        },
    }
    if not all(
        item["byte_identical"]
        for item in (
            vendor_identity["urdf"],
            vendor_identity["usd_top"],
            *vendor_identity["usd_configuration_layers"].values(),
        )
    ):
        raise SystemExit("runtime Piper URDF/USD differs from pinned agx_arm_sim")

    # --- official gripper from the pinned agx_arm_urdf submodule ----------
    gripper_joints = parse_urdf_joints(PIPER_L_GRIPPER_XACRO)
    master = gripper_joints["gripper"]
    fingers = []
    for name in GRIPPER_JOINTS:
        joint = gripper_joints[name]
        fingers.append(
            {
                "name": joint["name"],
                "type": joint["type"],
                "axis": joint["axis"],
                "limit": joint["limit"],
                "mimic": joint["mimic"],
                "parent_link": joint["parent_link"],
                "child_link": joint["child_link"],
                "origin_xyz": joint["origin_xyz"],
                "source": _origin(PIPER_L_GRIPPER_XACRO),
            }
        )

    # --- composed robot_assets URDF (input of the USD conversion) ---------
    composed_joints = parse_urdf_joints(COMPOSED_URDF)
    movable_types = {"revolute", "continuous", "prismatic"}
    composed_movable = [
        joint
        for joint in composed_joints.values()
        if joint["type"] in movable_types
    ]
    composed_movable_names = [joint["name"] for joint in composed_movable]
    composed_has_mimic = any("mimic" in joint for joint in composed_movable)
    composed_root = ET.parse(COMPOSED_URDF).getroot()
    composed_flange_present = any(
        element.attrib.get("name") == "flange_link"
        for element in composed_root.findall("link")
    )
    urdf_snapshot_differs = (
        composed_movable_names != EXPECTED_DOF_ORDER
        or composed_has_mimic
        or not composed_flange_present
    )
    submodule_url = git(
        "config",
        "--file",
        ".gitmodules",
        "--get",
        "submodule.src/agx_arm_description/agx_arm_urdf.url",
        cwd=AGX_ARM_ROS_DIR,
    )

    # --- planning groups and controllers ----------------------------------
    srdf_path = MOVEIT_CONFIG_DIR / "piper_l.srdf"
    srdf_root = ET.parse(srdf_path).getroot()
    # Official Isaac<->MoveIt topic contract from the TopicBasedSystem
    # hardware block of the pinned piper_l.ros2_control.xacro.
    ros2_control_xacro = ET.parse(
        MOVEIT_CONFIG_DIR / "piper_l.ros2_control.xacro"
    ).getroot()
    topic_params = {
        param.attrib["name"]: (param.text or "").strip()
        for param in ros2_control_xacro.iter("param")
        if param.attrib.get("name", "").endswith("_topic")
    }
    isaac_bridge_topics = {
        "hardware_plugin": "topic_based_ros2_control/TopicBasedSystem",
        "joint_states": topic_params["joint_states_topic"],
        "joint_commands": topic_params["joint_commands_topic"],
        "message_type": "sensor_msgs/JointState",
        "source": _origin(MOVEIT_CONFIG_DIR / "piper_l.ros2_control.xacro"),
    }
    groups: dict[str, dict] = {}
    for group in srdf_root.findall("group"):
        groups[group.attrib["name"]] = {
            "joints": [joint.attrib["name"] for joint in group.findall("joint")],
            "chains": [
                {
                    "base_link": chain.attrib["base_link"],
                    "tip_link": chain.attrib["tip_link"],
                }
                for chain in group.findall("chain")
            ],
        }
    group_states: dict[str, dict] = {}
    for state in srdf_root.findall("group_state"):
        group_states.setdefault(state.attrib["group"], {})[state.attrib["name"]] = {
            joint.attrib["name"]: float(joint.attrib["value"])
            for joint in state.findall("joint")
        }
    moveit_controllers = yaml.safe_load(
        (MOVEIT_CONFIG_DIR / "moveit_controllers.yaml").read_text(encoding="utf-8")
    )
    ros2_controllers = yaml.safe_load(
        (MOVEIT_CONFIG_DIR / "ros2_controllers.yaml").read_text(encoding="utf-8")
    )
    joint_limits = yaml.safe_load(
        (MOVEIT_CONFIG_DIR / "joint_limits.yaml").read_text(encoding="utf-8")
    )
    kinematics = yaml.safe_load(
        (MOVEIT_CONFIG_DIR / "kinematics.yaml").read_text(encoding="utf-8")
    )
    initial_positions = yaml.safe_load(
        (MOVEIT_CONFIG_DIR / "initial_positions.yaml").read_text(encoding="utf-8")
    )

    # --- expected USD DOF order (evidence-based, never read from USD) -----
    user_config = json.loads(USER_CONFIG.read_text(encoding="utf-8"))
    config_joint_orders = {
        robot["Name"]: robot["Joint_Order"]
        for robot in user_config["Robot_Config"]["Robots"]
    }
    for name, order in config_joint_orders.items():
        if order != EXPECTED_DOF_ORDER:
            raise SystemExit(
                f"config Joint_Order for {name} {order} != expected "
                f"{EXPECTED_DOF_ORDER}"
            )

    # --- composed USD facts (only what pxr actually returned) -------------
    usd_facts = extract_usd_facts()
    usd_movable = usd_facts.get("movable_joints", [])
    usd_by_name = {joint["name"]: joint for joint in usd_movable}
    official_fingers = {finger["name"]: finger for finger in fingers}
    for name, joint in usd_by_name.items():
        if name in submodule_by_name:
            # Execution-limit reference: the pinned agx_arm_sim URDF, which
            # its own USD physics layer matches. The wider agx_arm_urdf
            # values are recorded alongside for the conflicted joints.
            joint["execution_urdf"] = _origin(AGX_ARM_SIM_URDF)
            joint["lower_limit_rad"] = math.radians(joint["lower_limit"])
            joint["upper_limit_rad"] = math.radians(joint["upper_limit"])
            joint["matches_execution_urdf_limits"] = (
                math.isclose(
                    joint["lower_limit_rad"],
                    sim_urdf_joints[name]["limit"]["lower"],
                    abs_tol=1e-6,
                )
                and math.isclose(
                    joint["upper_limit_rad"],
                    sim_urdf_joints[name]["limit"]["upper"],
                    abs_tol=1e-6,
                )
            )
            submodule_limit = submodule_by_name[name]["limit"]
            joint["planning_urdf_limits_rad"] = [
                submodule_limit["lower"],
                submodule_limit["upper"],
            ]
            joint["within_planning_urdf_limits"] = (
                joint["lower_limit_rad"] >= submodule_limit["lower"] - 1e-6
                and joint["upper_limit_rad"] <= submodule_limit["upper"] + 1e-6
            )
        elif name in official_fingers:
            joint["official_urdf"] = _origin(PIPER_L_GRIPPER_XACRO)
            joint["matches_official_urdf_limits"] = (
                math.isclose(
                    joint["lower_limit"],
                    official_fingers[name]["limit"]["lower"],
                    abs_tol=1e-6,
                )
                and math.isclose(
                    joint["upper_limit"],
                    official_fingers[name]["limit"]["upper"],
                    abs_tol=1e-6,
                )
            )

    now = datetime.datetime.now(datetime.timezone.utc)
    artifact = {
        "schema": "rocobrick-piper-l-model-contract/v1",
        "phase": "P0.4",
        "model": "agilex_piper_l",
        "collected_date_utc": now.strftime("%Y-%m-%d"),
        "offline": True,
        "provenance": {
            "pinned_repositories": {
                "agx_arm_ros": {
                    "url": repositories["agx_arm_ros"]["url"],
                    "branch": repositories["agx_arm_ros"]["branch"],
                    "commit": agx_arm_ros_commit,
                    "checkout_path": _relative(AGX_ARM_ROS_DIR),
                    "head_verified": agx_arm_ros_head == agx_arm_ros_commit,
                },
                "agx_arm_sim": {
                    "url": repositories["agx_arm_sim"]["url"],
                    "branch": repositories["agx_arm_sim"]["branch"],
                    "commit": agx_arm_sim_commit,
                    "checkout_path": _relative(AGX_ARM_SIM_DIR),
                    "head_verified": agx_arm_sim_head == agx_arm_sim_commit,
                },
            },
            "submodules": {
                "agx_arm_urdf_in_agx_arm_ros": {
                    "url": submodule_url,
                    "recorded_gitlink": ros_gitlink,
                    "checked_out_head": urdf_checkout_head,
                    "pin": AGX_ARM_URDF_PIN,
                    "pin_verified": (
                        urdf_checkout_head == ros_gitlink == AGX_ARM_URDF_PIN
                    ),
                    "path": _relative(AGX_ARM_URDF_DIR),
                    "note": (
                        "Official planning model source; joint4 and joint5 "
                        "limits differ from the pinned agx_arm_sim URDF/USD."
                    ),
                },
                "agx_arm_urdf_in_agx_arm_sim": {
                    "recorded_gitlink": sim_gitlink,
                    "initialized": _sim_submodule_initialized(),
                    "note": (
                        "The agx_arm_sim checkout records gitlink "
                        f"{sim_gitlink} for its agx_arm_urdf submodule but it is "
                        "not checked out and is not used for this audit."
                    ),
                },
            },
            "file_sha256": file_sha256,
            "runtime_vendor_identity": vendor_identity,
            "robot_assets_snapshot": {
                "path": _relative(ROOT / "robot_assets/piper_l"),
                "version_controlled": True,
                "movable_joint_names": composed_movable_names,
                "has_mimic_declarations": composed_has_mimic,
                "has_flange_link": composed_flange_present,
                "differs_from_pinned_submodule": urdf_snapshot_differs,
                "note": (
                    "Version-controlled local copy of the agx_arm_sim URDF/USD "
                    "used at runtime; the audited URDF and USD layers are "
                    "byte-identical to that pinned vendor checkout. The "
                    "separate agx_arm_urdf planning model has flange_link, "
                    "mimic declarations, different finger origins, and wider "
                    "joint4/joint5 limits. Its gripper mimic relation remains "
                    "the logical gripper reference."
                ),
            },
        },
        "arm_joints": {
            "count": len(arm_joints),
            "order": ARM_JOINT_ORDER,
            "joints": arm_joints,
            "source_of_record": _origin(PIPER_L_URDF),
        },
        "limit_conflict": {
            "planning_urdf": _origin(PIPER_L_URDF),
            "execution_urdf": _origin(AGX_ARM_SIM_URDF),
            "execution_usd": _origin(AGX_ARM_SIM_USD_TOP),
            "conflicts": limit_conflicts,
            "execution_limits_rad": {
                name: [
                    max(
                        submodule_by_name[name]["limit"]["lower"],
                        sim_urdf_joints[name]["limit"]["lower"],
                    ),
                    min(
                        submodule_by_name[name]["limit"]["upper"],
                        sim_urdf_joints[name]["limit"]["upper"],
                    ),
                ]
                for name in ARM_JOINT_ORDER
            },
            "policy": "intersection_of_pinned_urdf_limits",
            "planning_enforcement": (
                "Pending MoveIt integration: apply the execution limits to "
                "MoveIt before enabling trajectory execution. The official "
                "MoveIt xacro uses agx_arm_urdf, whose joint4/joint5 position "
                "limits are wider; its joint_limits.yaml only changes dynamics."
            ),
        },
        "gripper": {
            "logical_master_joint": {
                "name": master["name"],
                "type": master["type"],
                "axis": master["axis"],
                "opening_range_m": [master["limit"]["lower"], master["limit"]["upper"]],
                "child_link": master["child_link"],
                "child_link_has_geometry": False,
                "present_in_pinned_submodule": True,
                "present_in_composed_urdf": "gripper" in composed_joints,
                "source": _origin(PIPER_L_GRIPPER_XACRO),
            },
            "fingers": fingers,
            "logical_to_finger_mapping": {
                "rule": "finger1_m = +0.5 * opening_m ; finger2_m = -0.5 * opening_m",
                "opening_range_m": [0.0, 0.1],
                "finger_ranges_m": {
                    "gripper_joint1": [
                        fingers[0]["limit"]["lower"],
                        fingers[0]["limit"]["upper"],
                    ],
                    "gripper_joint2": [
                        fingers[1]["limit"]["lower"],
                        fingers[1]["limit"]["upper"],
                    ],
                },
                "implementation": "rocobrick.gripper.logical_gripper_to_fingers",
                "inverse": "rocobrick.gripper.fingers_to_logical_gripper",
            },
        },
        "usd": {
            "stage_file": _origin(USD_TOP),
            "configuration_layers": [
                _origin(path) for path in sorted(USD_CONFIGURATION_DIR.glob("*.usd"))
            ],
            "extraction": usd_facts.get("extraction"),
            "composed_stage_facts": (
                {
                    "default_prim": usd_facts.get("default_prim"),
                    "up_axis": usd_facts.get("up_axis"),
                    "meters_per_unit": usd_facts.get("meters_per_unit"),
                    "used_layers": usd_facts.get("used_layers"),
                    "articulation_root_prims": usd_facts.get(
                        "articulation_root_prims"
                    ),
                }
                if usd_facts.get("extraction", {}).get("status") == "ok"
                else None
            ),
            "evidenced_movable_joints": usd_movable,
            "expected_dof_order": EXPECTED_DOF_ORDER,
            "expected_dof_order_evidence": {
                "urdf_movable_declaration_order": composed_movable_names,
                "config_joint_order": config_joint_orders,
                "runtime_verification": (
                    "src/rocobrick/isaac_scene.py compares "
                    "articulation.dof_names with the configured order at "
                    "bootstrap; the USD file itself does not store DOF order."
                ),
            },
            "runtime_dof_order": {
                "stored_in_usd": False,
                "authoritative_source": "articulation.dof_names at bootstrap",
                "note": (
                    "The expected 8-DOF order is an expectation derived from "
                    "the evidence above, not a value read from the USD."
                ),
            },
        },
        "planning": {
            "srdf": _origin(srdf_path),
            "robot_name": srdf_root.attrib["name"],
            "groups": groups,
            "group_states": group_states,
            "isaac_bridge_topics": isaac_bridge_topics,
            "moveit_controllers": moveit_controllers,
            "ros2_controllers": ros2_controllers,
            "joint_limits_overrides": joint_limits,
            "kinematics_solver": kinematics,
            "initial_positions": initial_positions,
        },
        "tcp": {
            "status": "unresolved",
            "facts": [
                "The official URDF defines no tool0 or dedicated TCP frame; "
                "the deepest tool-flange link is link6.",
                "The fork uses EE_Frames=['link6'] in config/user_config.json, "
                "so IK targets land on link6, not on a documented TCP.",
                "The pinned submodule adds flange_link (identity from link6) "
                "and gripper_base (+0.0045 m from flange_link), but no TCP "
                "frame is declared in either variant.",
                "Finger frames are prismatic, so a finger-midpoint TCP would "
                "move with the gripper command and is not defined anywhere "
                "official.",
                "Manipulation offsets in config/user_config.json "
                "(Grasp_TCP_Top_Offset=-0.0053 m, Gripper_Open_Clearance, "
                "Pregrasp_Clearance) are locally tuned values, not documented "
                "official TCP parameters.",
                "Distance from link6 to the finger contact surfaces differs "
                "between the robot_assets snapshot (0.1358 m finger origin) "
                "and the pinned submodule (0.0045 + 0.138 m), so a "
                "link6-referenced grasp offset cannot be validated against a "
                "single official number yet.",
            ],
            "resolution_target": "Resolve before P1 grasp execution work.",
        },
        "license": {
            "caveat": (
                "Model data licensing is not fully confirmed. The pinned "
                "agx_arm_ros checkout and the agx_arm_urdf submodule carry an "
                "MIT LICENSE whose copyright holder is 'aalicecc' (2026), a "
                "name that does not match the agilexrobotics organization the "
                "repositories are fetched from. The agx_arm_sim checkout has "
                "no LICENSE file; its README claims 'MIT License' without a "
                "full license text, and third_party/manifest.lock.yaml still "
                "records assets_license_confirmed: false pending re-audit. "
                "URDFs are auto-generated by the SolidWorks to URDF exporter. "
                "Redistribution of the URDF/mesh/USD model data therefore "
                "remains blocked until upstream licensing is confirmed."
            ),
            "agx_arm_ros_license": {
                "id": "MIT",
                "source_path": "LICENSE",
                "sha256": repositories["agx_arm_ros"]["license"]["sha256"],
                "copyright_holder_observed": "aalicecc (2026)",
            },
            "agx_arm_urdf_license": {
                "id": "MIT",
                "sha256": file_sha256[_relative(AGX_ARM_URDF_DIR / "LICENSE")],
            },
            "agx_arm_sim_license": {
                "standalone_license_file": False,
                "license_claim_source": "README.md '## License' section",
                "sha256": repositories["agx_arm_sim"]["license"]["sha256"],
                "assets_license_confirmed": False,
            },
        },
    }
    return artifact


def main() -> int:
    """Regenerate the model contract artifact.

    Returns:
        Always 0; hard failures raise SystemExit with a reason instead.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/model-audit/piper_l_model_contract.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    artifact = build_artifact()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
