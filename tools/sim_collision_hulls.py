"""Generate the sim-arbiter collision proxy for the Piper L wrist cluster.

The Isaac host bridge collides every robot-link mesh as a ``convexHull``
(``UsdPhysics.MeshCollisionAPI`` on the ``collisions`` Xforms of the pinned
vendor USD carries ``approximation=convexHull``, collision enabled) and keeps
all real contacts.  MoveIt, by contrast, checks the concave meshes, so both its
goal checking and its path validation are optimistic exactly where the hull
bridges a concavity — up to 54 mm on link2, 25 mm on the gripper base and the
wrist-camera stand, 12 mm on the fingers.  This tool materialises the hulls the
simulator actually collides so the runtime clearance gate can measure in the
arbiter geometry without needing the USD, Pinocchio, or any simulator.

The proxy is embedded in ``src/rocobrick/clearance/piper_l_sim_hulls.json``
because the ROS/MoveIt container mounts only ``src`` and ``ros_ws``.

Usage:
    python tools/sim_collision_hulls.py            # write the artifact
    python tools/sim_collision_hulls.py --check    # verify it is current
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO / "robot_assets/piper_l/piper_l_gripper_d435.urdf"
DEFAULT_STAND_MESH = (
    REPO / "ros_ws/src/rocobrick_piper_l_moveit_config/meshes/realsense_mid_stand.dae"
)
OUTPUT = REPO / "src/rocobrick/clearance/piper_l_sim_hulls.json"

SCHEMA = "rocobrick-sim-collision-hulls/v1"
# Bodies whose colliders the scene bootstrap disables; their data still ships
# so tests can rebuild the geometry that jammed the offset12 pre-phase.
DISABLED_BY_SCENE = ("camera_stand_link", "camera_link")
QUANT_M = 1e-6
PACKAGE_PREFIX = "package://agx_arm_description/"
PACKAGE_ROOT = REPO / "robot_assets/piper_l/agx_arm_description"

# Link names whose collision geometry the gate needs.  The chain links carry
# hulls; the d435 body is an exact authored box in both the runtime URDF and
# the USD ``camera_link/collisions/mesh_0/box`` Cube, so it stays a box.
COLLISION_LINKS = (
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "gripper_base",
    "gripper_link1",
    "gripper_link2",
    "camera_stand_link",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stl_vertices(path: Path) -> tuple[np.ndarray, int]:
    """Read a binary STL.

    Returns:
        The triangle vertices and the triangle count.
    """
    with path.open("rb") as handle:
        handle.read(80)
        count = struct.unpack("<I", handle.read(4))[0]
        raw = np.frombuffer(handle.read(count * 50), dtype=np.uint8).reshape(count, 50)
    return raw[:, 12:48].copy().view("<f4").reshape(-1, 3).astype(np.float64), count


def load_dae_vertices(path: Path) -> tuple[np.ndarray, int]:
    """Return (points, triangle_count) for the COLLADA mesh the stand uses."""
    namespace = "{http://www.collada.org/2005/11/COLLADASchema}"
    root = ET.parse(path).getroot()
    points = None
    for source in root.iter(namespace + "float_array"):
        if "positions" in source.get("id", ""):
            points = np.array(
                [float(value) for value in source.text.split()], dtype=np.float64
            ).reshape(-1, 3)
            break
    if points is None:
        raise SystemExit(f"{path}: no positions found")
    node = root.find(f".//{namespace}visual_scene/{namespace}node")
    matrix = np.eye(4)
    if node is not None and node.find(namespace + "matrix") is not None:
        matrix = np.array(
            [float(value) for value in node.find(namespace + "matrix").text.split()],
            dtype=np.float64,
        ).reshape(4, 4).T
    placed = (np.c_[points, np.ones(len(points))] @ matrix.T)[:, :3]
    triangles = root.find(f".//{namespace}triangles")
    count = int(triangles.get("count")) if triangles is not None else 0
    return placed, count


def collision_mesh_for(link: str, body: str) -> tuple[Path, str] | None:
    """Resolve the collision mesh file a URDF link body references.

    Returns:
        The resolved path and the reference as written, or None when the link
        declares no mesh collision.
    """
    """Resolve the collision mesh file the runtime URDF references."""
    match = re.search(r"<collision>(.*?)</collision>", body, re.S)
    if match is None:
        return None
    filename = re.search(r'filename="([^"]+)"', match.group(1))
    if filename is None:
        return None
    reference = filename.group(1)
    if reference.startswith(PACKAGE_PREFIX):
        path = PACKAGE_ROOT / reference[len(PACKAGE_PREFIX) :]
        if path.exists():
            return path, reference
    if reference.endswith("realsense_mid_stand.dae"):
        # The runtime URDF's own package path for the stand does not resolve on
        # disk; the USD carries the same mesh.  Our MoveIt copy was verified
        # against it (equal triangle count 4738, equal area 17987.9 mm^2 and
        # equal AABB in the USD link frame), and a convex hull only depends on
        # the surface, not on the triangulation.
        return DEFAULT_STAND_MESH, reference
    raise SystemExit(f"{link}: collision mesh {reference} does not resolve")


def hull_of(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (hull vertices, inside planes).

    The stored planes use the inside convention ``n.x + d >= 0``; qhull's own
    ``equations`` use ``n.x + d <= 0`` inside, so the sign is flipped here once
    and the runtime never has to care.

    Returns:
        The hull vertices and the inside-convention planes.
    """
    hull = ConvexHull(points, qhull_options="QJ" if len(points) < 12 else "")
    planes = -hull.equations.astype(np.float64)
    vertices = points[np.unique(hull.vertices)].astype(np.float64)
    inside = vertices @ planes[:, :3].T + planes[:, 3]
    if inside.min() < -1e-9:
        raise SystemExit(
            f"hull plane convention violated by {-inside.min() * 1e6:.3f} um"
        )
    return vertices, planes


def box_entry(minimum: np.ndarray, maximum: np.ndarray) -> dict[str, object]:
    """Return the exact box entry for the d435 body.

    Returns:
        The artifact entry with its vertices and inside planes.
    """
    corners = np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    planes = []
    for axis in range(3):
        above = np.zeros(3)
        above[axis] = 1.0
        planes.append([*above, -minimum[axis]])
        below = np.zeros(3)
        below[axis] = -1.0
        planes.append([*below, maximum[axis]])
    return {
        "kind": "box",
        "size_m": (maximum - minimum).tolist(),
        "center_m": ((minimum + maximum) / 2.0).tolist(),
        "vertices": flat_vertices(corners),
        "planes": [[round(float(value), 6) for value in plane] for plane in planes],
    }


def flat_vertices(points: np.ndarray) -> list[list[int]]:
    """Quantize points to whole micrometres for stable storage.

    Returns:
        Integer micrometre triples.
    """
    return [
        [int(round(float(value) / QUANT_M)) for value in point] for point in points
    ]


def build() -> dict[str, object]:
    """Build the complete proxy document.

    Returns:
        The artifact document including its content digest.
    """
    urdf_bytes = DEFAULT_URDF.read_bytes()
    root = ET.fromstring(urdf_bytes)
    links = {}
    for element in root.findall("link"):
        name = element.get("name")
        assert name is not None
        body = ET.tostring(element, encoding="unicode")
        if name not in COLLISION_LINKS:
            continue
        mesh = collision_mesh_for(name, body)
        if mesh is None:
            raise SystemExit(f"{name}: no collision mesh")
        path, reference = mesh
        if path.suffix == ".stl":
            points, triangles = load_stl_vertices(path)
        else:
            points, triangles = load_dae_vertices(path)
        vertices, planes = hull_of(points)
        links[name] = {
            "kind": "hull",
            "mesh": reference,
            "mesh_sha256": sha256_file(path),
            "mesh_triangles": triangles,
            "mesh_points": int(len(points)),
            "hull_vertices": int(len(vertices)),
            "vertices": flat_vertices(vertices),
            "planes": [
                [round(float(value), 6) for value in plane] for plane in planes
            ],
        }
    # The d435 body is an authored box, exactly as the MoveIt model has it.
    links["camera_link"] = box_entry(
        np.array([-0.012525, -0.0625, -0.0125]),
        np.array([0.012525, 0.0275, 0.0125]),
    )
    links["camera_link"]["mesh"] = "box 0.02505 x 0.09 x 0.025 at (0, -0.0175, 0)"
    links["camera_link"]["mesh_sha256"] = "authored-box"
    for disabled_link in DISABLED_BY_SCENE:
        links[disabled_link]["collision"] = "disabled-by-scene"
        links[disabled_link]["disabled_reason"] = (
            "rocobrick.isaac_scene._disable_camera_collisions switches this "
            "collider off when the scene is built; the vendor USD stays "
            "byte-identical.  The hull data is kept so the historical "
            "offset12 regression can still be reproduced with "
            "SimCollisionHulls(include_disabled=True)."
        )
    document = {
        "schema": SCHEMA,
        "robot": {
            "id": "piper_l_gripper_d435",
            "runtime_urdf": str(DEFAULT_URDF.relative_to(REPO)),
            "runtime_urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
            "vendor_usd": "robot_assets/piper_l/USD/piper_l_gripper_d435.usd",
            "vendor_usd_sha256": (
                "a781d52f9d55291fca9c8f2695abf9e311640d996314d272f47639d5323e2375"
            ),
            "approximation": "convexHull",
            "evidence": (
                "USD collision Xforms carry UsdPhysics.MeshCollisionAPI "
                "approximation=convexHull with collision enabled; the USD "
                "collision meshes are geometry-identical to the runtime URDF "
                "meshes (triangle count and total area matched per link); the "
                "d435 body collides as the authored box."
            ),
        },
        "units": "m",
        "quantization_m": QUANT_M,
        "plane_convention": "inside: n.x + d >= 0 (vertices quantized to 1 um)",
        "quantization_note": (
            "vertices are stored in whole micrometres, so a stored plane may "
            "sit up to 1 um inside its own quantized vertices"
        ),
        "links": links,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    document["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return document


def main() -> int:
    """Write the artifact, or verify it with ``--check``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify, do not write")
    args = parser.parse_args()
    document = build()
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.exists():
            print(f"missing artifact: {OUTPUT}", file=sys.stderr)
            return 1
        current = json.loads(OUTPUT.read_text())
        if current.get("content_sha256") != document["content_sha256"]:
            print(
                "artifact is stale: regenerate with tools/sim_collision_hulls.py",
                file=sys.stderr,
            )
            return 1
        print(f"artifact current: {OUTPUT}")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
