"""Sim-arbiter clearance proxy and gate contracts.

These tests pin three things that keep the gate honest:

* the embedded hull artifact still matches the pinned runtime meshes,
* the numpy GJK distance and the pure-numpy URDF FK agree with independent
  references (analytic box distances, a constrained QP, Pinocchio), and
* the offset12 support pre-phase -- the motion that jammed two wrist cameras
  together in simulation -- is rejected with the measured interpenetration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rocobrick.clearance import (
    UrdfChain,
    attachment_box,
    box_hull,
    gate_trajectory,
    load_hulls,
    region_clearance,
    robot_prefixes,
    sample_trajectory,
    trajectory_clearance,
)
from rocobrick.clearance.hulls import SimCollisionHulls, pair_distance

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "src/rocobrick/clearance/piper_l_sim_hulls.json"
TOOL_PATH = ROOT / "tools/sim_collision_hulls.py"
CONFIG = ROOT / "config/user_config_type1_scaled2_C.json"
RUNTIME_URDF = ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf"

# Recorded from task-c-top-offset12-camera2 (20261014T061028.688197Z): the
# staged end pose leaves piper_0's camera stand interpenetrating piper_1's
# finger-1 hull by 1.87 mm; the two trajectory controllers then fought and the
# piper_0 arm controller aborted on goal tolerance.
OFFSET12_ARM_JOINTS = (
    (
        -0.5080064535140991,
        1.2511368989944458,
        -0.6527838706970215,
        0.4604097008705139,
        0.6243300437927246,
        -0.8366326689720154,
        -3.905498840994781e-11,
        -4.967070088568448e-10,
        -3.9113981151195354e-13,
        -3.941559023612484e-10,
        -2.497049234051474e-09,
        9.47970976890311e-13,
    ),
    (
        0.5293111694252991,
        1.4605095694201062,
        -0.6803316934146542,
        0.7156202046644408,
        0.7404596251275292,
        -1.561829689993057,
        -0.7991335761802281,
        1.5923023198487563,
        -1.18300827163339,
        -0.01880879110100312,
        1.1604603846035608,
        -0.7948665900821488,
    ),
)
OFFSET12_GRIPPERS = {
    "piper_0_gripper_joint1": 0.01591438241302967,
    "piper_0_gripper_joint2": -0.015882106497883797,
    "piper_1_gripper_joint1": 0.0,
    "piper_1_gripper_joint2": 0.0,
}
OFFSET12_WORST_PAIR = ("piper_0_camera_stand_link", "piper_1_gripper_link1")
OFFSET12_WORST_DISTANCE_M = -0.001866
TRAJECTORY_JOINTS = tuple(
    [f"piper_0_joint{index}" for index in range(1, 7)]
    + [f"piper_1_joint{index}" for index in range(1, 7)]
)


def _load_module(name: str, path: Path):
    """Import a repository module by path for contract testing.

    Returns:
        The imported module.
    """
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dual_model_document() -> str:
    """Build the prefixed dual model the container generates for MoveIt.

    The gate is exercised against a model with the same kinematic content the
    container's generator produces: robot-prefixed links and joints, a
    ``dual_world`` root, and the base placements from the task config (the
    generator's pinned-origin validation targets the pinned vendor xacro, which
    is not expandable on a bare host).

    Returns:
        The URDF document text of the combined two-robot model.
    """
    import copy
    import xml.etree.ElementTree as ET

    from scipy.spatial.transform import Rotation

    config = json.loads(CONFIG.read_text())
    robots = config["Robot_Config"]["Robots"]
    source = ET.fromstring(RUNTIME_URDF.read_text())
    lines = [
        '<?xml version="1.0"?>',
        '<robot name="piper_l_dual">',
        '<link name="dual_world"/>',
    ]
    for robot in robots:
        name = robot["Name"]
        frame = robot["Robot_Base_Frame"]
        # The task config stores the base orientation as (w, x, y, z); scipy
        # wants (x, y, z, w).
        w, x, y, z = frame["Orientation"]
        rpy = Rotation.from_quat([x, y, z, w]).as_euler("xyz")
        xyz = " ".join(str(value) for value in frame["Position"])
        rotation = " ".join(str(value) for value in rpy)
        lines.append(
            f'<joint name="{name}_dual_world_joint" type="fixed">'
            '<parent link="dual_world"/>'
            f'<child link="{name}_base_link"/>'
            f'<origin xyz="{xyz}" rpy="{rotation}"/></joint>'
        )
        for element in source.findall("link"):
            if element.get("name") == "world":
                continue
            clone = copy.deepcopy(element)
            clone.set("name", f'{name}_{element.get("name")}')
            lines.append(ET.tostring(clone, encoding="unicode"))
        # The container's generated model adds the calibrated TCP link (the
        # runtime URDF does not carry it); the origin is the pinned TCP
        # calibration translation from link6.
        lines.append(f'<link name="{name}_rocobrick_tcp"/>')
        lines.append(
            f'<joint name="{name}_rocobrick_tcp_joint" type="fixed">'
            f'<parent link="{name}_link6"/><child link="{name}_rocobrick_tcp"/>'
            '<origin xyz="-1.887447531601083e-09 -1.3530843112619095e-16 '
            '0.1305895866844472" rpy="0 0 0"/></joint>'
        )
        for element in source.findall("joint"):
            if element.find("parent").get("link") == "world":
                continue
            clone = copy.deepcopy(element)
            clone.set("name", f'{name}_{element.get("name")}')
            clone.find("parent").set(
                "link", f'{name}_{clone.find("parent").get("link")}'
            )
            clone.find("child").set(
                "link", f'{name}_{clone.find("child").get("link")}'
            )
            mimic = clone.find("mimic")
            if mimic is not None:
                mimic.set("joint", f'{name}_{mimic.get("joint")}')
            lines.append(ET.tostring(clone, encoding="unicode"))
    lines.append("</robot>")
    return "\n".join(lines)


def test_proxy_artifact_tracks_the_pinned_meshes() -> None:
    """The embedded hulls still derive from the pinned runtime meshes."""
    tool = _load_module("sim_collision_hulls", TOOL_PATH)
    document = json.loads(ARTIFACT.read_text())
    assert document["schema"] == "rocobrick-sim-collision-hulls/v1"
    assert document["robot"]["approximation"] == "convexHull"
    assert (
        document["robot"]["runtime_urdf_sha256"] == tool.sha256_file(RUNTIME_URDF)
    )
    body = dict(document)
    recorded = body.pop("content_sha256")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == recorded
    import xml.etree.ElementTree as ET

    root = ET.fromstring(RUNTIME_URDF.read_text())
    bodies = {
        element.get("name"): ET.tostring(element, encoding="unicode")
        for element in root.findall("link")
    }
    for link, entry in document["links"].items():
        if entry["kind"] == "box":
            continue
        path, reference = tool.collision_mesh_for(link, bodies[link])
        assert entry["mesh"] == reference
        assert entry["mesh_sha256"] == tool.sha256_file(path)


def test_proxy_planes_hold_their_own_vertices() -> None:
    """Every stored plane uses the inside convention for its own quantized hull."""
    hulls = load_hulls()
    # The wrist-camera colliders are switched off by the scene bootstrap, so
    # the shipped proxy excludes them; the disabled hulls stay in the artifact
    # so the historical offset12 geometry remains reproducible.
    assert len(hulls.link_names) == 10
    assert hulls.disabled_names == ("camera_link", "camera_stand_link")
    assert len(SimCollisionHulls(include_disabled=True).link_names) == 12
    for name in hulls.link_names:
        hull = hulls.hull(name)
        inside = hull.vertices @ hull.planes[:, :3].T + hull.planes[:, 3]
        assert inside.min() >= -2e-6, name
        assert len(hull.vertices) >= 8


def test_gjk_distance_matches_analytic_box_cases() -> None:
    """Separated, touching and overlapping unit boxes measure exactly."""
    cube = box_hull("cube", np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
    identity = np.eye(4)
    separated = np.eye(4)
    separated[0, 3] = 1.05
    distance, intersecting = pair_distance(cube, identity, cube, separated)
    assert distance == pytest.approx(0.05, abs=1e-9)
    assert intersecting is False
    touching = np.eye(4)
    touching[0, 3] = 1.0
    _, intersecting = pair_distance(cube, identity, cube, touching)
    assert intersecting is True
    overlapping = np.eye(4)
    overlapping[0, 3] = 0.97
    _, intersecting = pair_distance(cube, identity, cube, overlapping)
    assert intersecting is True
    offset = np.eye(4)
    offset[0, 3] = 1.03
    offset[1, 3] = 0.04
    distance, intersecting = pair_distance(cube, identity, cube, offset)
    assert distance == pytest.approx(0.03, abs=1e-9)
    assert intersecting is False


def _world_planes(hull, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = hull.planes[:, :3] @ pose[:3, :3].T
    offsets = hull.planes[:, 3] - normals @ pose[:3, 3]
    return normals, offsets


def test_gjk_distance_matches_constrained_reference() -> None:
    """Independent QP reference: minimise |x - y| with both points inside."""
    from scipy.optimize import minimize

    hulls = SimCollisionHulls(include_disabled=True)
    rng = np.random.default_rng(11)
    for name_a, name_b in (
        ("gripper_base", "camera_stand_link"),
        ("camera_stand_link", "gripper_link1"),
        ("link5", "link6"),
    ):
        for _ in range(2):
            poses = []
            for _index in range(2):
                pose = np.eye(4)
                pose[:3, :3] = _random_rotation(rng)
                pose[:3, 3] = rng.uniform(-0.18, 0.18, 3)
                poses.append(pose)
            hull_a, hull_b = hulls.hull(name_a), hulls.hull(name_b)
            distance, intersecting = pair_distance(hull_a, poses[0], hull_b, poses[1])
            if intersecting:
                continue
            normals_a, offsets_a = _world_planes(hull_a, poses[0])
            normals_b, offsets_b = _world_planes(hull_b, poses[1])

            def objective(vector: np.ndarray) -> float:
                delta = vector[:3] - vector[3:]
                return float(delta @ delta)

            constraints = [
                {
                    "type": "ineq",
                    "fun": lambda vector, n=normal, o=offset: float(
                        n @ vector[:3] + o
                    ),
                }
                for normal, offset in zip(normals_a, offsets_a, strict=True)
            ] + [
                {
                    "type": "ineq",
                    "fun": lambda vector, n=normal, o=offset: float(
                        n @ vector[3:] + o
                    ),
                }
                for normal, offset in zip(normals_b, offsets_b, strict=True)
            ]
            start = np.concatenate(
                [
                    poses[0][:3, 3] + (poses[1][:3, 3] - poses[0][:3, 3]) * 0.5,
                    poses[1][:3, 3] + (poses[0][:3, 3] - poses[1][:3, 3]) * 0.5,
                ]
            )
            reference = minimize(
                objective, start, constraints=constraints, method="SLSQP",
                options={"maxiter": 400, "ftol": 1e-14},
            )
            assert reference.success, reference.message
            assert distance == pytest.approx(float(np.sqrt(reference.fun)), abs=5e-4)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.random(random_state=rng).as_matrix()


def test_urdf_chain_matches_pinocchio() -> None:
    """The gate's own FK equals Pinocchio on the pinned runtime model."""
    import pinocchio as pin

    chain = UrdfChain(RUNTIME_URDF.read_text())
    model = pin.buildModelFromUrdf(str(RUNTIME_URDF))
    data = model.createData()
    rng = np.random.default_rng(5)
    for _ in range(3):
        values = {
            f"joint{index}": float(rng.uniform(*limits_scale(index)))
            for index in range(1, 7)
        }
        values["gripper_joint1"] = 0.02
        values["gripper_joint2"] = -0.02
        configuration = pin.neutral(model)
        for name, value in values.items():
            configuration[model.idx_qs[model.getJointId(name)]] = value
        pin.framesForwardKinematics(model, data, configuration)
        poses = chain.forward(values)
        for link in (
        "link1",
        "link3",
        "link5",
        "link6",
        "gripper_base",
        "gripper_link1",
        "gripper_link2",
        "camera_stand_link",
        "camera_link",
    ):
            expected = np.array(data.oMf[model.getFrameId(link)].homogeneous)
            assert np.allclose(poses[link], expected, atol=1e-9), link


def limits_scale(index: int) -> tuple[float, float]:
    """Return a safe sampling range for one arm joint in tests."""
    return {
        1: (-2.6, 2.6),
        2: (0.0, 3.1),
        3: (-2.9, 0.0),
        4: (-1.7, 1.7),
        5: (-1.2, 1.2),
        6: (-2.0, 2.0),
    }[index]


def test_offset12_jam_is_reproduced_with_the_historical_geometry() -> None:
    """In the geometry of that run the gate rejects the motion with its pair.

    The wrist camera collided then; the scene bootstrap now disables it, so the
    historical verdict is pinned through ``include_disabled=True``.
    """
    document = _dual_model_document()
    report = gate_trajectory(
        document,
        list(TRAJECTORY_JOINTS),
        list(OFFSET12_ARM_JOINTS),
        static_joint_values=OFFSET12_GRIPPERS,
        hulls=SimCollisionHulls(include_disabled=True),
    )
    assert report.passed is False
    assert report.worst is not None
    assert report.worst.intersecting is True
    assert (report.worst.body_a, report.worst.body_b) == OFFSET12_WORST_PAIR
    assert report.worst.distance_m == pytest.approx(OFFSET12_WORST_DISTANCE_M, abs=5e-4)


def test_offset12_pre_phase_passes_with_the_camera_colliders_disabled() -> None:
    """The shipped geometry clears the same motion; the collars now bind.

    This pins the measured effect of disabling the vendor camera colliders:
    the staged pair moves from a 1.87 mm interpenetration of piper_0's stand
    with piper_1's finger to a 31.8 mm clearance between the two gripper
    collars.
    """
    document = _dual_model_document()
    report = gate_trajectory(
        document,
        list(TRAJECTORY_JOINTS),
        list(OFFSET12_ARM_JOINTS),
        static_joint_values=OFFSET12_GRIPPERS,
    )
    assert report.passed is True
    assert report.worst is not None
    assert report.worst.intersecting is False
    assert (report.worst.body_a, report.worst.body_b) == (
        "piper_0_gripper_base",
        "piper_1_gripper_base",
    )
    assert report.worst.distance_m == pytest.approx(0.0318, abs=1e-3)
    assert report.evidence["covered_bodies"] == {"piper_0": 10, "piper_1": 10}
    assert report.evidence["excluded_bodies"] == ["camera_link", "camera_stand_link"]


def test_offset12_staged_pose_is_the_worst_sample() -> None:
    """Clearance closes toward the staged pose: the endpoint is the binding one."""
    document = _dual_model_document()
    end = trajectory_clearance(
        document,
        [np.array(OFFSET12_ARM_JOINTS[-1])],
        list(TRAJECTORY_JOINTS),
        static_joint_values=OFFSET12_GRIPPERS,
        hulls=SimCollisionHulls(include_disabled=True),
    )
    assert end.passed is False
    assert end.worst is not None
    assert end.worst.intersecting is True
    assert end.worst.distance_m == pytest.approx(OFFSET12_WORST_DISTANCE_M, abs=5e-4)


def test_held_brick_is_measured_against_the_other_robot() -> None:
    """An attached box is a body like any other: it keeps the gate honest."""
    document = _dual_model_document()
    chain = UrdfChain(document)
    home = {
        f"{robot}_joint{index}": 0.0
        for robot in ("piper_0", "piper_1")
        for index in range(1, 7)
    }
    poses = {
        robot: {
            link: pose
            for link, pose in chain.forward(
                {
                    name: value
                    for name, value in home.items()
                    if name.startswith(f"{robot}_")
                }
            ).items()
            if link.startswith(f"{robot}_")
        }
        for robot in ("piper_0", "piper_1")
    }
    clear = region_clearance(poses, floor_m=0.015)
    # A foreign robot's link in one robot's map must never form a pair.
    polluted = {**poses, "piper_0": {**poses["piper_0"], **poses["piper_1"]}}
    assert region_clearance(polluted, floor_m=0.015).passed is True
    assert clear.passed is True
    tcp = poses["piper_0"]["piper_0_rocobrick_tcp"][:3, 3]
    reach = min(
        float(np.linalg.norm(tcp - pose[:3, 3]))
        for pose in poses["piper_1"].values()
    )
    half = reach + 0.25
    oversized = attachment_box(
        "piper_0_held_brick",
        "piper_0_rocobrick_tcp",
        minimum=(-half, -half, -half),
        maximum=(half, half, half),
    )
    blocked = region_clearance(
        poses, attached={"piper_0": [oversized]}, floor_m=0.015
    )
    assert blocked.passed is False
    assert blocked.worst is not None
    assert "piper_0_held_brick" in (blocked.worst.body_a, blocked.worst.body_b)


def test_robot_prefixes_and_sampling_budget() -> None:
    """Prefixes require a complete joint set; sampling respects its budget."""
    assert robot_prefixes(TRAJECTORY_JOINTS) == ("piper_0", "piper_1")
    assert robot_prefixes(("piper_0_joint1",)) == ()
    points = [np.zeros(12), np.full(12, 1.2)]
    samples = sample_trajectory(points, max_substep_rad=0.05, max_samples=600)
    assert len(samples) == 25
    assert np.allclose(samples[0], 0.0)
    assert np.allclose(samples[-1], 1.2)
    capped = sample_trajectory(points, max_substep_rad=0.05, max_samples=7)
    assert len(capped) == 7
    assert np.allclose(capped[-1], 1.2)
