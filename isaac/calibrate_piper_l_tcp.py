#!/usr/bin/env python3
"""Bounded live project calibration of the Piper L TCP (Isaac Sim 5.1, headless).

Stage 1 (selector discovery, fail-closed): compose the CURRENT runtime USD at
``/World/Robot_0`` with no physics play/step, locate the exact finger collision
mesh prims (USD prim instances matching the historical runtime paths), read the
complete composed mesh topology, and map the reviewed STL candidate patch
(seven coplanar distal facets) into USD face indices by GEOMETRY ONLY — a
connected, coplanar patch whose area, centroid and polygon vertices match the
reviewed STL candidate within explicit tolerances, required to be unique. No
face-index assumption and no whole-link bounding box is used anywhere. The
reviewed local source JSON (``config/calibration/piper_l_tcp_source.json``) is
written from the selection; discovery is then rerun once against the finalized
source hash so the recorded evidence is self-consistent.

Stage 2 (15-sample calibration): the live articulation is initialized exactly
once (a single ``world.reset()`` — bounded initialization only) to read the
live DOF order and limits, then the timeline is stopped and never played or
stepped again. Each sample kinematically composes all 8 joint positions
(atomically, update-only) by authoring the composed forward-kinematics matrices
onto the in-memory session layer, and reads link6 and both finger meshes
through ONE ``UsdGeom.XformCache``. Patch triangle areas/centroids/normals are
computed in world, the TCP is the midpoint of the opposed patch centroids, and
``T_link6_tcp = inverse(T_world_link6) * T_world_tcp`` with the contract
identity orientation validated by axis measurement.

TCP definition (frozen): origin = midpoint of the area-weighted centroids of
the two opposed distal finger contact patches; parent ``link6``, child
``rocobrick_tcp``; orientation identity ``(0,0,0,1)``; +Y closing direction,
+X pad lateral, +Z same as link6. This is a PROJECT calibration scoped to the
exact runtime USD hash — simulation-only, never hardware validated, and never
an official/hardware TCP. No physics play/step/contact/grasp is involved and
no grasp is claimed.

Artifacts are written (and flushed) before Kit shutdown; ``app.close()`` may
terminate the process, so nothing is persisted after it.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_SELECTOR = "rocobrick-piper-l-tcp-selector-discovery/v1"
SCHEMA_CALIBRATION = "rocobrick-piper-l-isaac-tcp-calibration/v1"
SCHEMA_SOURCE = "rocobrick-piper-l-tcp-source/v1"

MODEL = "agilex_piper_l"
CALIBRATION_AUTHORITY = "project"
METHOD = "symmetric-opposed-contact-patch-centroids"
ALGORITHM_VERSION = 1
ISAAC_VERSION = "5.1"

USD_PATH = ROOT / "robot_assets" / "piper_l" / "USD" / "piper_l_gripper_d435.usd"
URDF_PATH = ROOT / "robot_assets" / "piper_l" / "piper_l_gripper_d435.urdf"
STL_DIR = (
    ROOT
    / "robot_assets"
    / "piper_l"
    / "agx_arm_description"
    / "agx_arm_urdf"
    / "piper_l"
    / "meshes"
)
STL_PATHS = {
    "gripper_link1": STL_DIR / "gripper_link1.stl",
    "gripper_link2": STL_DIR / "gripper_link2.stl",
}
SELECTOR_SOURCE_PATH = ROOT / "config" / "calibration" / "piper_l_tcp_source.json"
SELECTOR_ARTIFACT_PATH = (
    ROOT / "artifacts" / "model-audit" / "piper_l_tcp_selector_discovery.json"
)
CALIBRATION_ARTIFACT_PATH = (
    ROOT / "artifacts" / "model-audit" / "piper_l_tcp_calibration.json"
)

EXPECTED_USD_SHA256 = (
    "a781d52f9d55291fca9c8f2695abf9e311640d996314d272f47639d5323e2375"
)
EXPECTED_URDF_SHA256 = (
    "8a5e4a77c53947e234f7f8a15db14e30ca85eb2c9f7ae3dc1784f07057626b0d"
)

ROBOT_PRIM = "/World/Robot_0"
ARTICULATION_PRIM = ROBOT_PRIM + "/root_joint"
LINK6_PRIM = ROBOT_PRIM + "/link6"
FINGER_PRIMS = {
    "gripper_link1": ROBOT_PRIM + "/gripper_link1/collisions",
    "gripper_link2": ROBOT_PRIM + "/gripper_link2/collisions",
}
JOINT_SCOPE = ROBOT_PRIM + "/joints"
FK_BODIES = (
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "gripper_base",
    "gripper_link1",
    "gripper_link2",
)

# Reviewed STL candidate: 0-based facets 2..8 of both finger STLs, seven
# coplanar z=0.004 triangles. The signature is recomputed from the actual STL
# bytes at runtime and must match these frozen review values.
STL_FACET_RANGE = (2, 8)
FROZEN_PATCH_AREA_MM2 = 84.599979
FROZEN_PATCH_CENTROID_M = (0.0, -0.07279077972, 0.004)

TCP_DEFINITION = {
    "parent_frame": "link6",
    "child_frame": "rocobrick_tcp",
    "origin": (
        "midpoint of the area-weighted centroids of the two opposed distal "
        "finger contact patches"
    ),
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "axes": {
        "x": "pad lateral",
        "y": "closing direction",
        "z": "same as link6",
    },
    "expected_candidate_translation_m": [0.0, 0.0, 0.06300922028],
    "expected_patch_separation_m": "opening + 0.008",
    "authority": CALIBRATION_AUTHORITY,
    "official_or_hardware_tcp": False,
}

ARM_POSES = {
    "A": [0.0, 0.79, 0.0, -1.20, 0.0, 1.20],
    "B": [0.35, 0.95, -0.35, -0.90, 0.25, 0.85],
    "C": [-0.35, 0.65, -0.35, -1.40, -0.25, 1.45],
}
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGER_JOINTS = {
    "gripper_link1": "gripper_joint1",
    "gripper_link2": "gripper_joint2",
}
DOF_ORDER = ARM_JOINTS + ["gripper_joint1", "gripper_joint2"]
OPENINGS = [0.0, 0.025, 0.05, 0.075, 0.1]
GRIPPER_CENTER_OFFSET_M = 0.008
AXIS_VECTORS = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}

THRESHOLDS = {
    "measured_finger_vs_requested_m": 1e-7,
    "symmetry_m": 1e-7,
    "opening_error_m": 2e-7,
    "repeat_centroid_spread_m": 1e-8,
    "triangle_area_min_m2": 1e-10,
    "planarity_m": 5e-5,
    "opposed_normals_deg": 0.5,
    "separation_axis_vs_opening_axis_deg": 0.5,
    "per_pose_lateral_midpoint_spread_m": 5e-5,
    "per_pose_axial_midpoint_spread_m": 1e-4,
    "all15_translation_median_residual_m": 1e-4,
    "rotation_residual_deg": 0.05,
    "world_reconstruction_m": 1e-4,
    "separation_opening_r2_min": 0.999999,
    "separation_opening_slope_tolerance": 0.002,
    "separation_opening_max_residual_m": 5e-5,
    "expected_candidate_tolerance_m": 2e-4,
    "selector_vertex_m": 1e-9,
    "selector_area_m2": 1e-9,
    "selector_centroid_m": 1e-9,
    "selector_plane_m": 1e-9,
    "selector_normal_deg": 1e-3,
    "selector_required_unique_match": 1,
    "frozen_signature_area_mm2": 1e-6,
    "frozen_signature_centroid_m": 1e-9,
    "fk_urdf_zero_validation_m": 1e-6,
}


# ---------------------------------------------------------------------------
# Pure helpers (no simulator imports)
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of the file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_canonical_json(path: Path, data: dict) -> str:
    """Write canonical sorted-keys compact UTF-8 JSON with a trailing newline."""
    text = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    length = _norm(a)
    if length == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return (a[0] / length, a[1] / length, a[2] / length)


def _angle_deg(a, b):
    cosine = _dot(_unit(a), _unit(b))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def identity_matrix():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat_mul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def apply_matrix(m, point):
    return tuple(
        m[i][0] * point[0] + m[i][1] * point[1] + m[i][2] * point[2] + m[i][3]
        for i in range(3)
    )


def matrix_translation(m):
    return (m[0][3], m[1][3], m[2][3])


def matrix_rotation(m):
    return tuple(
        tuple(m[i][j] for j in range(3)) for i in range(3)
    )


def quat_to_matrix(quat_wxyz):
    w, x, y, z = quat_wxyz
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def frame_matrix(pos, quat_xyzw):
    rot = quat_to_matrix(
        (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
    )
    return [
        [rot[0][0], rot[0][1], rot[0][2], pos[0]],
        [rot[1][0], rot[1][1], rot[1][2], pos[1]],
        [rot[2][0], rot[2][1], rot[2][2], pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_part_matrix(rot3):
    return [
        [rot3[0][0], rot3[0][1], rot3[0][2], 0.0],
        [rot3[1][0], rot3[1][1], rot3[1][2], 0.0],
        [rot3[2][0], rot3[2][1], rot3[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def axis_rotation_matrix(axis_unit, angle_rad):
    x, y, z = axis_unit
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    one_c = 1.0 - c
    return [
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s, 0.0],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s, 0.0],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def translation_matrix(offset):
    return [
        [1.0, 0.0, 0.0, offset[0]],
        [0.0, 1.0, 0.0, offset[1]],
        [0.0, 0.0, 1.0, offset[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rigid_inverse(m):
    """Exact inverse of a rigid transform (rotation + translation)."""
    rot_t = [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
    t = (m[0][3], m[1][3], m[2][3])
    return [
        [rot_t[0][0], rot_t[0][1], rot_t[0][2],
         -_dot(rot_t[0], t)],
        [rot_t[1][0], rot_t[1][1], rot_t[1][2],
         -_dot(rot_t[1], t)],
        [rot_t[2][0], rot_t[2][1], rot_t[2][2],
         -_dot(rot_t[2], t)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def col_to_gf_values(m):
    """Column-convention matrix -> 16 row-major Gf.Matrix4d values.

    Gf uses the row-vector convention (translation in the LAST ROW, basis
    vectors as rows), so the rotation part is transposed and the translation
    occupies row 3: ``p_row' = p_row @ Gf`` equals ``p' = M @ p``.
    """
    return (
        m[0][0], m[1][0], m[2][0], 0.0,
        m[0][1], m[1][1], m[2][1], 0.0,
        m[0][2], m[1][2], m[2][2], 0.0,
        m[0][3], m[1][3], m[2][3], 1.0,
    )


def gf_to_col(m):
    """Gf.Matrix4d (USD row-vector convention) -> column-convention matrix."""
    rows = [list(m.GetRow(i)) for i in range(4)]
    return [
        [rows[0][0], rows[1][0], rows[2][0], rows[3][0]],
        [rows[0][1], rows[1][1], rows[2][1], rows[3][1]],
        [rows[0][2], rows[1][2], rows[2][2], rows[3][2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_to_list(m):
    return [value for row in m for value in row]


def median(values):
    ordered = sorted(values)
    size = len(ordered)
    if size % 2 == 1:
        return ordered[size // 2]
    return 0.5 * (ordered[size // 2 - 1] + ordered[size // 2])


def linear_fit(xs, ys):
    """Least-squares line fit with slope, intercept, R^2 and residuals."""
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2, residuals


def parse_binary_stl(path: Path):
    """Parse a binary STL into (facet count, per-facet 12-float tuples)."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path.name}: too small for a binary STL")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path.name}: not a binary STL of {count} facets")
    facets = [
        struct.unpack_from("<12f", data, 84 + 50 * index)
        for index in range(count)
    ]
    return count, facets


def triangle_metrics(v0, v1, v2):
    """Return (area, unit winding normal, vertex-mean centroid)."""
    cross = _cross(_sub(v1, v0), _sub(v2, v0))
    area = 0.5 * _norm(cross)
    normal = _unit(cross) if area > 0.0 else (0.0, 0.0, 0.0)
    centroid = (
        (v0[0] + v1[0] + v2[0]) / 3.0,
        (v0[1] + v1[1] + v2[1]) / 3.0,
        (v0[2] + v1[2] + v2[2]) / 3.0,
    )
    return area, normal, centroid


def stl_patch_signature(path: Path) -> dict:
    """Recompute the reviewed distal patch signature from actual STL bytes."""
    count, facets = parse_binary_stl(path)
    low, high = STL_FACET_RANGE
    if count <= high:
        raise ValueError(f"{path.name}: facet range {STL_FACET_RANGE} out of range")
    selected = facets[low : high + 1]
    areas, centroids, triangles = [], [], []
    normal_sum = (0.0, 0.0, 0.0)
    for facet in selected:
        v0, v1, v2 = facet[3:6], facet[6:9], facet[9:12]
        area, normal, centroid = triangle_metrics(v0, v1, v2)
        areas.append(area)
        centroids.append(centroid)
        triangles.append([list(v0), list(v1), list(v2)])
        normal_sum = tuple(normal_sum[k] + normal[k] for k in range(3))
    area_total = sum(areas)
    centroid = tuple(
        sum(a * c[k] for a, c in zip(areas, centroids)) / area_total
        for k in range(3)
    )
    unique = []
    seen = set()
    for triangle in triangles:
        for vertex in triangle:
            key = tuple(round(v, 9) for v in vertex)
            if key not in seen:
                seen.add(key)
                unique.append(vertex)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "facet_count": count,
        "reviewed_facets_0based": list(range(low, high + 1)),
        "triangle_count": len(selected),
        "area_m2": area_total,
        "area_mm2": area_total * 1e6,
        "centroid_m": list(centroid),
        "plane_normal": _unit(normal_sum),
        "unique_vertex_count": len(unique),
        "unique_vertices_m": unique,
        "triangles_m": triangles,
    }


# ---------------------------------------------------------------------------
# Selector discovery (geometry-only mapping into composed USD topology)
# ---------------------------------------------------------------------------


def _face_records(points, counts, indices):
    faces = []
    for face_index, count in enumerate(counts):
        if count != 3:
            continue
        ids = indices[3 * face_index : 3 * face_index + 3]
        verts = [tuple(points[i]) for i in ids]
        area, normal, centroid = triangle_metrics(*verts)
        faces.append({
            "face_index": face_index,
            "vertex_indices": ids,
            "vertices": verts,
            "area": area,
            "normal": normal,
            "centroid": centroid,
            "plane_offset": _dot(normal, verts[0]),
        })
    return faces


def coplanar_connected_components(faces):
    """Union-find faces into connected coplanar patches (geometry only)."""
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets = {}
    for index, face in enumerate(faces):
        buckets.setdefault(round(face["plane_offset"] / 1e-6), []).append(index)
    for key in sorted(buckets):
        nearby = buckets.get(key, []) + buckets.get(key + 1, [])
        for pos, i in enumerate(nearby):
            for j in nearby[pos + 1:]:
                if abs(faces[i]["plane_offset"] - faces[j]["plane_offset"]) > 1e-9:
                    continue
                if 1.0 - abs(_dot(faces[i]["normal"], faces[j]["normal"])) > 1e-9:
                    continue
                vertices_i = {
                    tuple(round(c, 9) for c in v) for v in faces[i]["vertices"]
                }
                vertices_j = {
                    tuple(round(c, 9) for c in v) for v in faces[j]["vertices"]
                }
                if vertices_i & vertices_j:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
    components = {}
    for index in range(len(faces)):
        components.setdefault(find(index), []).append(index)
    return [sorted(members) for members in components.values()]


def select_patch_geometry(topology, signature, thresholds):
    """Map the reviewed STL candidate into USD face indices by geometry only.

    Returns (selection dict or None, matching candidate count).
    """
    faces = _face_records(
        topology["points"],
        topology["face_vertex_counts"],
        topology["face_vertex_indices"],
    )
    signature_vertex_by_key = {}
    for triangle in signature["triangles_m"]:
        for vertex in triangle:
            signature_vertex_by_key.setdefault(
                tuple(round(v, 9) for v in vertex), vertex
            )
    signature_normal = tuple(signature["plane_normal"])
    matches = []
    for members in coplanar_connected_components(faces):
        if len(members) != signature["triangle_count"]:
            continue
        patch_faces = [faces[i] for i in members]
        area = sum(f["area"] for f in patch_faces)
        if area <= 0.0:
            continue
        centroid = tuple(
            sum(f["area"] * f["centroid"][k] for f in patch_faces) / area
            for k in range(3)
        )
        normal = _unit(tuple(
            sum(f["area"] * f["normal"][k] for f in patch_faces) / area
            for k in range(3)
        ))
        usd_vertex_by_key = {}
        for f in patch_faces:
            for vertex in f["vertices"]:
                usd_vertex_by_key.setdefault(
                    tuple(round(v, 9) for v in vertex), vertex
                )
        if len(usd_vertex_by_key) != len(signature_vertex_by_key):
            continue
        vertex_residual = 0.0
        for key, stl_vertex in signature_vertex_by_key.items():
            if key not in usd_vertex_by_key:
                vertex_residual = float("inf")
                break
            usd_vertex = usd_vertex_by_key[key]
            vertex_residual = max(
                vertex_residual,
                max(abs(usd_vertex[k] - stl_vertex[k]) for k in range(3)),
            )
        area_residual = abs(area - signature["area_m2"])
        centroid_residual = max(
            abs(centroid[k] - signature["centroid_m"][k]) for k in range(3)
        )
        normal_residual_deg = _angle_deg(normal, signature_normal)
        plane_residual = abs(
            _dot(normal, patch_faces[0]["vertices"][0])
            - _dot(normal, tuple(signature["triangles_m"][0][0]))
        )
        if (
            area_residual <= thresholds["selector_area_m2"]
            and centroid_residual <= thresholds["selector_centroid_m"]
            and vertex_residual <= thresholds["selector_vertex_m"]
            and normal_residual_deg <= thresholds["selector_normal_deg"]
            and plane_residual <= thresholds["selector_plane_m"]
        ):
            ordered = sorted(patch_faces, key=lambda f: f["face_index"])
            unique_vertices = sorted(
                usd_vertex_by_key.values(), key=lambda v: (v[0], v[1], v[2])
            )
            matches.append({
                "face_indices": [f["face_index"] for f in ordered],
                "ordered_vertex_indices": [f["vertex_indices"] for f in ordered],
                "ordered_vertex_coordinates_m": [
                    [list(v) for v in f["vertices"]] for f in ordered
                ],
                "unique_vertices_m": [list(v) for v in unique_vertices],
                "triangle_count": len(ordered),
                "area_m2": area,
                "area_mm2": area * 1e6,
                "centroid_m": list(centroid),
                "plane_normal": list(normal),
                "match_residuals": {
                    "area_m2": area_residual,
                    "centroid_m": centroid_residual,
                    "vertex_m": vertex_residual,
                    "normal_deg": normal_residual_deg,
                    "plane_m": plane_residual,
                },
            })
    if len(matches) != int(thresholds["selector_required_unique_match"]):
        return None, len(matches)
    return matches[0], len(matches)


def boot_simulation_app():
    """Start the headless Kit application."""
    from isaacsim import SimulationApp

    return SimulationApp({"headless": True})


def load_runtime_scene():
    """Compose the runtime USD at /World/Robot_0 and return (stage, meta)."""
    from isaacsim.core.utils.prims import create_prim
    import omni.usd
    from pxr import UsdGeom

    create_prim(prim_path=ROBOT_PRIM, usd_path=USD_PATH.as_posix())
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(ROBOT_PRIM).IsValid():
        raise RuntimeError(f"robot prim missing after load: {ROBOT_PRIM}")
    return stage, {
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
    }


def prim_record(prim) -> dict:
    """Record path/type/active/loaded/purpose/visibility/schemas/material."""
    from pxr import UsdGeom, UsdShade

    record = {
        "path": prim.GetPath().pathString,
        "type_name": str(prim.GetTypeName()),
        "active": bool(prim.IsActive()),
        "loaded": bool(prim.IsLoaded()),
        "instance": bool(prim.IsInstance()),
    }
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        record["purpose"] = str(imageable.ComputePurpose())
        record["visibility"] = str(imageable.ComputeVisibility())
    record["applied_schemas"] = sorted(str(a) for a in prim.GetAppliedSchemas())
    record["material"] = None
    try:
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material[0]:
            record["material"] = material[0].GetPath().pathString
    except Exception:
        record["material"] = None
    return record


def find_mesh_in_prototype(instance_prim):
    """Return (prototype prim, relative mesh path, mesh prim) for an instance."""
    from pxr import Usd

    proto_prim = instance_prim.GetPrototype()
    if proto_prim is None or not proto_prim.IsValid():
        raise RuntimeError("instance has no valid prototype")
    mesh_prim = None
    for candidate in Usd.PrimRange(proto_prim):
        if candidate.GetTypeName() == "Mesh":
            if mesh_prim is not None:
                raise RuntimeError("instance prototype has more than one Mesh")
            mesh_prim = candidate
    if mesh_prim is None:
        raise RuntimeError("instance prototype has no Mesh prim")
    relative = mesh_prim.GetPath().pathString.replace(
        proto_prim.GetPath().pathString, "", 1
    )
    return proto_prim, relative, mesh_prim


def _array_digest(values) -> str:
    items = []
    for item in values:
        try:
            items.append(tuple(float(v) for v in item))
        except TypeError:
            items.append(float(item))
    return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()


def read_mesh_topology(mesh_prim) -> dict:
    """Read complete Mesh points/faceVertexCounts/faceVertexIndices/meta."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    return {
        "prototype_path": mesh_prim.GetPath().pathString,
        "points_count": len(points),
        "face_vertex_counts_count": len(counts),
        "face_vertex_indices_count": len(indices),
        "points_sha256": _array_digest(points),
        "face_vertex_indices_sha256": _array_digest(indices),
        "orientation": str(mesh.GetOrientationAttr().Get()),
        "subdivision_scheme": str(mesh.GetSubdivisionSchemeAttr().Get()),
        "non_triangle_face_counts": sorted(
            {int(c) for c in counts if int(c) != 3}
        ),
        "points": [tuple(float(v) for v in p) for p in points],
        "face_vertex_counts": [int(c) for c in counts],
        "face_vertex_indices": [int(i) for i in indices],
    }


def physx_collision_record(mesh_prim, instance_prim) -> dict:
    """Record collision approximation / PhysX attributes and applied APIs."""
    from pxr import PhysxSchema, UsdPhysics

    collision_api = UsdPhysics.CollisionAPI(mesh_prim)
    record = {
        "mesh_applied_schemas": sorted(
            str(a) for a in mesh_prim.GetAppliedSchemas()
        ),
        "instance_applied_schemas": sorted(
            str(a) for a in instance_prim.GetAppliedSchemas()
        ),
        "usd_physics_collision_enabled": (
            collision_api.GetCollisionEnabledAttr().Get()
        ),
        "has_physx_collision_api": bool(
            PhysxSchema.PhysxCollisionAPI(mesh_prim)
        ),
    }
    attrs = {}
    for prim, label in ((mesh_prim, "mesh"), (instance_prim, "instance")):
        for attr_obj in prim.GetAttributes():
            name = attr_obj.GetName()
            value = attr_obj.Get()
            if name.startswith(("physics:", "physx:")) and value is not None:
                attrs[f"{label}:{name}"] = str(value)
    record["physics_physx_attributes"] = dict(sorted(attrs.items()))
    return record


def read_joint_specs(stage) -> dict:
    """Read the USD physics joint frames that drive the FK composition."""
    from pxr import UsdPhysics

    def xyzw(quat):
        return (quat.imaginary[0], quat.imaginary[1], quat.imaginary[2], quat.real)

    specs = {}
    for child in stage.GetPrimAtPath(JOINT_SCOPE).GetAllChildren():
        joint = UsdPhysics.Joint(child)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        pos0 = child.GetAttribute("physics:localPos0").Get()
        pos1 = child.GetAttribute("physics:localPos1").Get()
        rot0 = child.GetAttribute("physics:localRot0").Get()
        rot1 = child.GetAttribute("physics:localRot1").Get()
        axis_attr = child.GetAttribute("physics:axis").Get()
        lower = child.GetAttribute("physics:lowerLimit").Get()
        upper = child.GetAttribute("physics:upperLimit").Get()
        specs[child.GetName()] = {
            "prim_path": child.GetPath().pathString,
            "type": str(child.GetTypeName()),
            "body0": body0[0].name if body0 else None,
            "body1": body1[0].name if body1 else None,
            "local_pos0": tuple(float(v) for v in pos0),
            "local_rot0_xyzw": xyzw(rot0),
            "local_pos1": tuple(float(v) for v in pos1),
            "local_rot1_xyzw": xyzw(rot1),
            "axis": None if axis_attr is None else str(axis_attr),
            "lower_limit": None if lower is None else float(lower),
            "upper_limit": None if upper is None else float(upper),
        }
    return specs


def joint_displacement(spec, q):
    """Relative transform D(q) inserted between the two joint frames."""
    axis = AXIS_VECTORS[spec["axis"]] if spec["axis"] else (0.0, 0.0, 1.0)
    if spec["type"] == "PhysicsRevoluteJoint":
        return axis_rotation_matrix(axis, q)
    if spec["type"] == "PhysicsPrismaticJoint":
        return translation_matrix(tuple(axis[k] * q for k in range(3)))
    return identity_matrix()


def forward_kinematics(specs, root_matrix, joint_values):
    """Compose world transforms for every link body from the joint frames."""
    child_joints = {
        spec["body1"]: (name, spec)
        for name, spec in specs.items()
        if spec["body1"]
    }
    transforms = {"world": root_matrix}
    pending = list(child_joints)
    while pending:
        progressed = False
        for body in list(pending):
            name, spec = child_joints[body]
            if spec["body0"] not in transforms:
                continue
            l0 = frame_matrix(spec["local_pos0"], spec["local_rot0_xyzw"])
            l1_inv = rigid_inverse(
                frame_matrix(spec["local_pos1"], spec["local_rot1_xyzw"])
            )
            displacement = joint_displacement(
                spec, float(joint_values.get(name, 0.0))
            )
            transforms[body] = mat_mul(
                mat_mul(mat_mul(transforms[spec["body0"]], l0), displacement),
                l1_inv,
            )
            pending.remove(body)
            progressed = True
        if not progressed:
            raise RuntimeError(f"FK chain incomplete for: {sorted(pending)}")
    return transforms


def urdf_zero_fk(urdf_path: Path):
    """Independent URDF chain transforms at the zero configuration."""
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    joints = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        raw_xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        raw_rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        xyz = tuple(float(v) for v in raw_xyz.split())
        roll, pitch, yaw = (float(v) for v in raw_rpy.split())
        rot = mat_mul(
            mat_mul(
                axis_rotation_matrix((0.0, 0.0, 1.0), yaw),
                axis_rotation_matrix((0.0, 1.0, 0.0), pitch),
            ),
            axis_rotation_matrix((1.0, 0.0, 0.0), roll),
        )
        joints[joint.find("child").get("link")] = {
            "parent": joint.find("parent").get("link"),
            "matrix": [
                [rot[0][0], rot[0][1], rot[0][2], xyz[0]],
                [rot[1][0], rot[1][1], rot[1][2], xyz[1]],
                [rot[2][0], rot[2][1], rot[2][2], xyz[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    transforms = {"world": identity_matrix()}
    pending = list(joints)
    while pending:
        progressed = False
        for body in list(pending):
            joint = joints[body]
            if joint["parent"] not in transforms:
                continue
            transforms[body] = mat_mul(transforms[joint["parent"]], joint["matrix"])
            pending.remove(body)
            progressed = True
        if not progressed:
            raise RuntimeError(
                f"URDF chain incomplete for: {sorted(pending)}"
            )
    return transforms


def measured_patch_world(selected, matrix_world):
    """World-space per-triangle metrics for the selected USD patch."""
    triangles = []
    for vertex_indices, local_vertices in zip(
        selected["ordered_vertex_indices"],
        selected["ordered_vertex_coordinates_m"],
    ):
        world = [apply_matrix(matrix_world, v) for v in local_vertices]
        area, normal, centroid = triangle_metrics(*world)
        triangles.append({
            "vertex_indices": list(vertex_indices),
            "world_vertices_m": [list(v) for v in world],
            "area_m2": area,
            "normal": list(normal),
            "centroid_m": list(centroid),
        })
    area = sum(t["area_m2"] for t in triangles)
    centroid = tuple(
        sum(t["area_m2"] * t["centroid_m"][k] for t in triangles) / area
        for k in range(3)
    )
    normal = _unit(tuple(
        sum(t["area_m2"] * t["normal"][k] for t in triangles) / area
        for k in range(3)
    ))
    return {
        "triangles": triangles,
        "area_m2": area,
        "centroid_m": centroid,
        "normal": normal,
    }


def patch_lateral_axis(patch, closing_axis):
    """Measured pad-lateral direction from the patch boundary, orthogonalized.

    The contact pad is a rectangle-like patch whose lateral direction is
    carried by its long parallel boundary edges. The longest boundary edge
    (an edge used by exactly one patch triangle) defines the lateral
    direction; it is orthogonalized against the patch normal and left
    UNsigned here (evaluate_sample sign-fixes it against the contract +X).
    """
    edge_count = {}
    edge_endpoints = {}
    for triangle in patch["triangles"]:
        verts = triangle["world_vertices_m"]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ka = tuple(round(v, 9) for v in verts[a])
            kb = tuple(round(v, 9) for v in verts[b])
            key = (ka, kb) if ka <= kb else (kb, ka)
            edge_count[key] = edge_count.get(key, 0) + 1
            edge_endpoints[key] = (tuple(verts[a]), tuple(verts[b]))
    boundary = [key for key, count in edge_count.items() if count == 1]
    if not boundary:
        raise RuntimeError("patch has no boundary edges")
    longest = max(
        boundary,
        key=lambda key: _norm(_sub(edge_endpoints[key][0], edge_endpoints[key][1])),
    )
    direction = _unit(_sub(edge_endpoints[longest][0], edge_endpoints[longest][1]))
    # remove any residual normal component and normalize
    return _unit(_sub(
        direction,
        tuple(closing_axis[k] * _dot(direction, closing_axis) for k in range(3)),
    ))


def rotation_residual_deg(x_axis, y_axis):
    """Angle of the measured frame (x, y, x cross y) from identity."""
    x_u, y_u = _unit(x_axis), _unit(y_axis)
    z_u = _cross(x_u, y_u)
    trace = x_u[0] + y_u[1] + z_u[2]
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def author_link_transform(stage, prim_path, col_matrix):
    """Author one composed FK matrix on the in-memory session layer only.

    The session-layer ``xformOpOrder`` opinion replaces the captured
    ``[translate, orient, scale]`` order completely, so the composed local
    transform is exactly the authored FK matrix (``AddTransformOp`` alone
    would only APPEND the op and leave the captured pose applied first).
    """
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"missing prim for authoring: {prim_path}")
    attr = prim.GetAttribute("xformOp:transform")
    if not attr.IsValid():
        UsdGeom.Xform(prim).AddTransformOp(
            precision=UsdGeom.XformOp.PrecisionDouble
        )
        attr = prim.GetAttribute("xformOp:transform")
    order = prim.GetAttribute("xformOpOrder")
    order.Set(["xformOp:transform"])
    attr.Set(Gf.Matrix4d(*col_to_gf_values(col_matrix)))


# ---------------------------------------------------------------------------
# Main two-stage procedure
# ---------------------------------------------------------------------------


def run_selector_discovery(stage, signatures):
    """Two discovery passes; returns (initial, rerun) per-finger records."""
    from pxr import Usd, UsdGeom

    passes = {}
    for pass_name in ("initial", "rerun"):
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        link6_matrix = gf_to_col(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(LINK6_PRIM))
        )
        records = {}
        for name, prim_path in FINGER_PRIMS.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid() or not prim.IsInstance():
                raise RuntimeError(
                    f"{prim_path} missing or not a USD prim instance; the "
                    "historical collision prim topology changed"
                )
            _, relative, mesh_prim = find_mesh_in_prototype(prim)
            topology = read_mesh_topology(mesh_prim)
            selection, candidate_count = select_patch_geometry(
                topology, signatures[name], THRESHOLDS
            )
            instance_matrix = gf_to_col(cache.GetLocalToWorldTransform(prim))
            mesh_proto_matrix = gf_to_col(
                cache.GetLocalToWorldTransform(mesh_prim)
            )
            mesh_world = mat_mul(instance_matrix, mesh_proto_matrix)
            records[name] = {
                "instance": prim_record(prim),
                "instance_world_matrix": matrix_to_list(instance_matrix),
                "prototype_relative_mesh_path": relative,
                "prototype_mesh_prim_path": topology["prototype_path"],
                "in_prototype_mesh_transform_identity_residual_m": _norm(_sub(
                    apply_matrix(mesh_proto_matrix, (0.0, 0.0, 0.0)),
                    (0.0, 0.0, 0.0),
                )),
                "mesh_world_matrix": matrix_to_list(mesh_world),
                "topology": {
                    key: value
                    for key, value in topology.items()
                    if key
                    not in ("points", "face_vertex_counts", "face_vertex_indices")
                },
                "collision_record": physx_collision_record(mesh_prim, prim),
                "local_to_link6_matrix": matrix_to_list(
                    mat_mul(rigid_inverse(link6_matrix), mesh_world)
                ),
                "selection": selection,
                "usd_candidate_count": candidate_count,
                "passed": (
                    selection is not None
                    and candidate_count
                    == int(THRESHOLDS["selector_required_unique_match"])
                ),
            }
        passes[pass_name] = records
    return passes


def evaluate_sample(context) -> dict:
    """Compute every gated metric for one composed sample.

    The contract axes (+Y closing, +X pad lateral, +Z like link6) live in the
    link6 frame, so the measured directions (patch normals, separation axis,
    lateral axis) are rotated into link6 coordinates before comparison.
    """
    link6 = context["link6_matrix"]
    link6_inv = rigid_inverse(link6)
    rot6 = matrix_rotation(link6)
    rot6_t = [[rot6[j][i] for j in range(3)] for i in range(3)]

    def to_link6(vec):
        return tuple(sum(rot6_t[i][k] * vec[k] for k in range(3)) for i in range(3))

    patch1, patch2 = context["patch1"], context["patch2"]
    c1, c2 = patch1["centroid_m"], patch2["centroid_m"]
    n1, n2 = patch1["normal"], patch2["normal"]
    n1_link6 = _unit(to_link6(n1))
    separation_vector = _sub(c2, c1)
    separation = _norm(separation_vector)
    sep_axis_link6 = _unit(to_link6(separation_vector))
    midpoint = tuple((c1[k] + c2[k]) / 2.0 for k in range(3))
    translation_link6 = apply_matrix(link6_inv, midpoint)
    reconstruction = _norm(_sub(
        apply_matrix(link6, translation_link6), midpoint
    ))
    lateral = patch_lateral_axis(patch1, n1)
    lateral_link6 = _unit(to_link6(lateral))
    if _dot(lateral_link6, (1.0, 0.0, 0.0)) < 0.0:
        lateral_link6 = tuple(-v for v in lateral_link6)
    planarity = max(
        max(
            abs(_dot(_sub(tuple(v), patch["centroid_m"]), patch["normal"]))
            for triangle in patch["triangles"]
            for v in triangle["world_vertices_m"]
        )
        for patch in (patch1, patch2)
    )
    min_area = min(
        t["area_m2"] for patch in (patch1, patch2) for t in patch["triangles"]
    )
    translation = tuple(translation_link6)
    return {
        "translation_link6_m": list(translation),
        "midpoint_world_m": list(midpoint),
        "separation_m": separation,
        "opening_measured_m": separation - GRIPPER_CENTER_OFFSET_M,
        "opening_error_m": abs(
            separation - GRIPPER_CENTER_OFFSET_M - context["opening"]
        ),
        "finger_errors_m": {
            name: abs(
                context["measured_finger"][name]
                - context["requested_finger"][name]
            )
            for name in ("gripper_link1", "gripper_link2")
        },
        "symmetry_m": max(
            abs(
                context["measured_finger"]["gripper_link1"]
                + context["measured_finger"]["gripper_link2"]
            ),
            math.sqrt(translation[0] ** 2 + translation[1] ** 2),
        ),
        "planarity_m": planarity,
        "min_triangle_area_m2": min_area,
        "opposed_normals_error_deg": abs(180.0 - _angle_deg(n1, n2)),
        "closing_axis_error_deg": _angle_deg(n1_link6, (0.0, 1.0, 0.0)),
        "separation_axis_error_deg": _angle_deg(sep_axis_link6, (0.0, 1.0, 0.0)),
        "rotation_residual_deg": rotation_residual_deg(lateral_link6, n1_link6),
        "world_reconstruction_m": reconstruction,
        "repeat_centroid_spread_m": context["repeat_centroid_spread_m"],
        "measured_finger_m": dict(context["measured_finger"]),
    }


def measure_finger_displacements(joint_specs, gripper_base_matrix,
                                 instance_matrices):
    """Measured prismatic displacements from the read-back transforms.

    The displacement is the joint-axis projection of the read-back finger
    origin minus the gripper-base origin, expressed in the joint frame and
    referenced to the zero-pose projection constant, so the result is exactly
    the prismatic joint coordinate.
    """
    measured = {}
    base_origin = matrix_translation(gripper_base_matrix)
    base_rot = matrix_rotation(gripper_base_matrix)
    base_rot_t = [[base_rot[j][i] for j in range(3)] for i in range(3)]
    for name in FINGER_PRIMS:
        spec = joint_specs[FINGER_JOINTS[name]]
        axis = AXIS_VECTORS[spec["axis"]]
        l0_rot = quat_to_matrix(
            (spec["local_rot0_xyzw"][3], spec["local_rot0_xyzw"][0],
             spec["local_rot0_xyzw"][1], spec["local_rot0_xyzw"][2])
        )
        l0_rot_t = [[l0_rot[j][i] for j in range(3)] for i in range(3)]
        zero_constant = _dot(axis, tuple(
            sum(l0_rot_t[i][k] * spec["local_pos0"][k] for k in range(3))
            for i in range(3)
        ))
        delta = _sub(
            matrix_translation(instance_matrices[name]), base_origin
        )
        base_local = tuple(
            sum(base_rot_t[i][k] * delta[k] for k in range(3)) for i in range(3)
        )
        joint_frame = tuple(
            sum(l0_rot_t[i][k] * base_local[k] for k in range(3))
            for i in range(3)
        )
        measured[name] = _dot(axis, joint_frame) - zero_constant
    return measured


def aggregate_and_gate(samples, stage_meta, timeline_playing, sim_time_deltas,
                       dof_names, dof_limits):
    """Aggregate all 15 samples and evaluate every mandatory gate."""
    gate_results = {}
    gate_results["measured_finger_vs_requested"] = max(
        max(s["metrics"]["finger_errors_m"].values()) for s in samples
    ) <= THRESHOLDS["measured_finger_vs_requested_m"]
    gate_results["symmetry"] = max(
        s["metrics"]["symmetry_m"] for s in samples
    ) <= THRESHOLDS["symmetry_m"]
    gate_results["opening_error"] = max(
        s["metrics"]["opening_error_m"] for s in samples
    ) <= THRESHOLDS["opening_error_m"]
    gate_results["repeat_centroid_spread"] = max(
        s["metrics"]["repeat_centroid_spread_m"] for s in samples
    ) <= THRESHOLDS["repeat_centroid_spread_m"]
    gate_results["triangle_area_min"] = (
        min(s["metrics"]["min_triangle_area_m2"] for s in samples)
        > THRESHOLDS["triangle_area_min_m2"]
    )
    gate_results["planarity"] = max(
        s["metrics"]["planarity_m"] for s in samples
    ) <= THRESHOLDS["planarity_m"]
    gate_results["opposed_normals"] = max(
        s["metrics"]["opposed_normals_error_deg"] for s in samples
    ) <= THRESHOLDS["opposed_normals_deg"]
    gate_results["separation_axis_vs_opening_axis"] = max(
        s["metrics"]["separation_axis_error_deg"] for s in samples
    ) <= THRESHOLDS["separation_axis_vs_opening_axis_deg"]
    lateral_spreads, axial_spreads = {}, {}
    for pose_name in ARM_POSES:
        translations = [
            s["metrics"]["translation_link6_m"] for s in samples
            if s["pose"] == pose_name
        ]
        lateral = [(t[0], t[1]) for t in translations]
        axial = [t[2] for t in translations]
        lateral_spreads[pose_name] = max(
            math.dist(lateral[i], lateral[j])
            for i in range(len(lateral))
            for j in range(i + 1, len(lateral))
        )
        axial_spreads[pose_name] = max(axial) - min(axial)
    gate_results["per_pose_lateral_midpoint_spread"] = max(
        lateral_spreads.values()
    ) <= THRESHOLDS["per_pose_lateral_midpoint_spread_m"]
    gate_results["per_pose_axial_midpoint_spread"] = max(
        axial_spreads.values()
    ) <= THRESHOLDS["per_pose_axial_midpoint_spread_m"]
    translations = [s["metrics"]["translation_link6_m"] for s in samples]
    med = tuple(median([t[k] for t in translations]) for k in range(3))
    median_residual = max(math.dist(t, med) for t in translations)
    gate_results["all15_translation_median_residual"] = (
        median_residual <= THRESHOLDS["all15_translation_median_residual_m"]
    )
    expected = TCP_DEFINITION["expected_candidate_translation_m"]
    expected_deviation = max(abs(med[k] - expected[k]) for k in range(3))
    gate_results["expected_candidate_proximity"] = (
        expected_deviation <= THRESHOLDS["expected_candidate_tolerance_m"]
    )
    rotation_max = max(s["metrics"]["rotation_residual_deg"] for s in samples)
    gate_results["rotation_residual"] = (
        rotation_max <= THRESHOLDS["rotation_residual_deg"]
    )
    reconstruction_max = max(
        s["metrics"]["world_reconstruction_m"] for s in samples
    )
    gate_results["world_reconstruction"] = (
        reconstruction_max <= THRESHOLDS["world_reconstruction_m"]
    )
    slope, intercept, r2, residuals = linear_fit(
        [s["opening_m"] for s in samples],
        [s["metrics"]["separation_m"] for s in samples],
    )
    gate_results["separation_vs_opening"] = (
        r2 >= THRESHOLDS["separation_opening_r2_min"]
        and abs(slope - 1.0) <= THRESHOLDS["separation_opening_slope_tolerance"]
        and max(abs(r) for r in residuals)
        <= THRESHOLDS["separation_opening_max_residual_m"]
    )
    values = [v for t in translations for v in t]
    values += [slope, r2, intercept]
    for sample in samples:
        values += list(sample["metrics"]["translation_link6_m"])
        values += [
            sample["metrics"]["separation_m"],
            sample["metrics"]["opening_measured_m"],
            sample["metrics"]["symmetry_m"],
            sample["metrics"]["planarity_m"],
            sample["metrics"]["rotation_residual_deg"],
            sample["metrics"]["world_reconstruction_m"],
            sample["metrics"]["repeat_centroid_spread_m"],
            sample["metrics"]["min_triangle_area_m2"],
        ]
        values += [v for v in sample["measured_finger_displacement_m"].values()]
        values += [v for v in sample["metrics"]["finger_errors_m"].values()]
    gate_results["all_values_finite"] = all(math.isfinite(v) for v in values)
    gate_results["stage_units"] = (
        stage_meta["meters_per_unit"] == 1.0 and stage_meta["up_axis"] == "Z"
    )
    gate_results["exact_dof_order"] = dof_names == DOF_ORDER
    gate_results["no_foreign_contacts"] = (
        not timeline_playing and max(sim_time_deltas) == 0.0
    )
    # MANDATORY: every sampled state must lie inside the live articulation
    # limits; a single violated DOF on any sample fails the whole calibration.
    gate_results["joint_limit_violations_empty"] = not any(
        s["joint_limit_violations"] for s in samples
    )
    aggregate = {
        "translation_median_m": list(med),
        "translation_max_residual_from_median_m": median_residual,
        "expected_candidate_translation_m": expected,
        "expected_candidate_deviation_m": expected_deviation,
        "rotation_residual_max_deg": rotation_max,
        "world_reconstruction_max_m": reconstruction_max,
        "separation_vs_opening": {
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "max_abs_residual_m": max(abs(r) for r in residuals),
        },
        "per_pose_lateral_midpoint_spread_m": lateral_spreads,
        "per_pose_axial_midpoint_spread_m": axial_spreads,
        "sim_time_delta_max_s": max(sim_time_deltas),
        "timeline_playing_during_sampling": timeline_playing,
    }
    failures = [name for name, value in gate_results.items() if not value]
    return gate_results, aggregate, failures


def main(argv=None) -> int:
    """Run the bounded two-stage calibration and write both artifacts.

    Args:
        argv: Optional argument list; ``--discovery-only`` runs stage 1 only.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--discovery-only",
        action="store_true",
        help="run selector discovery only and write the selector artifacts",
    )
    args = parser.parse_args(argv)
    discovery_only = bool(args.discovery_only)
    started = time.time()
    app = None
    world = None
    cleanup_error = None
    try:
        usd_sha = sha256_file(USD_PATH)
        urdf_sha = sha256_file(URDF_PATH)
        if usd_sha != EXPECTED_USD_SHA256:
            raise RuntimeError(
                f"runtime USD hash {usd_sha} != expected {EXPECTED_USD_SHA256}"
            )
        if urdf_sha != EXPECTED_URDF_SHA256:
            raise RuntimeError(
                f"runtime URDF hash {urdf_sha} != expected {EXPECTED_URDF_SHA256}"
            )
        signatures = {
            name: stl_patch_signature(path) for name, path in STL_PATHS.items()
        }
        for name, signature in signatures.items():
            if abs(signature["area_mm2"] - FROZEN_PATCH_AREA_MM2) > 1e-6:
                raise RuntimeError(
                    f"{name}: reviewed area {signature['area_mm2']} mm^2 does "
                    f"not match the frozen candidate {FROZEN_PATCH_AREA_MM2}"
                )
            if max(
                abs(signature["centroid_m"][k] - FROZEN_PATCH_CENTROID_M[k])
                for k in range(3)
            ) > THRESHOLDS["frozen_signature_centroid_m"]:
                raise RuntimeError(
                    f"{name}: reviewed centroid does not match the frozen one"
                )

        app = boot_simulation_app()
        stage, stage_meta = load_runtime_scene()

        # -------- Stage 1: selector discovery (initial pass) -------------
        initial = run_selector_discovery(stage, signatures)["initial"]
        selector_passed = all(initial[name]["passed"] for name in FINGER_PRIMS)
        if not selector_passed:
            raise RuntimeError(
                "selector discovery failed: no unique geometric match for the "
                "reviewed STL patch in the composed USD topology"
            )
        source_document = {
            "schema": SCHEMA_SOURCE,
            "model": MODEL,
            "algorithm": {
                "name": METHOD,
                "version": ALGORITHM_VERSION,
                "tcp_definition": TCP_DEFINITION,
                "selector": (
                    "geometry-only connected coplanar patch match of the "
                    "reviewed STL candidate (area, centroid, polygon vertices) "
                    "inside the composed USD collision topology; no face-index "
                    "assumption and no whole-link bounding box is used"
                ),
            },
            "scope": {
                "runtime_usd_scoped": True,
                "runtime_usd_path": USD_PATH.relative_to(ROOT).as_posix(),
                "runtime_usd_sha256": usd_sha,
                "runtime_urdf_path": URDF_PATH.relative_to(ROOT).as_posix(),
                "runtime_urdf_sha256": urdf_sha,
                "isaac_sim_version": ISAAC_VERSION,
                "simulation_only": True,
                "hardware_validated": False,
            },
            "sources": {
                name: {
                    "path": signature["path"],
                    "sha256": signature["sha256"],
                    "facet_count": signature["facet_count"],
                    "reviewed_facets_0based": signature["reviewed_facets_0based"],
                    "patch_area_mm2": signature["area_mm2"],
                    "patch_centroid_m": signature["centroid_m"],
                    "unique_vertices_m": signature["unique_vertices_m"],
                    "triangles_m": signature["triangles_m"],
                }
                for name, signature in signatures.items()
            },
            "prims": {
                name: {
                    "collision_prim_path": FINGER_PRIMS[name],
                    "mesh_relative_prototype_path": initial[name][
                        "prototype_relative_mesh_path"
                    ],
                    "selected_usd_face_indices": initial[name]["selection"][
                        "face_indices"
                    ],
                    "selected_usd_face_vertex_indices": initial[name][
                        "selection"
                    ]["ordered_vertex_indices"],
                    "selected_usd_face_vertex_coordinates_m": initial[name][
                        "selection"
                    ]["ordered_vertex_coordinates_m"],
                    "selected_unique_vertices_m": initial[name]["selection"][
                        "unique_vertices_m"
                    ],
                    "selected_patch_area_mm2": initial[name]["selection"][
                        "area_mm2"
                    ],
                    "selected_patch_centroid_m": initial[name]["selection"][
                        "centroid_m"
                    ],
                    "match_residuals": initial[name]["selection"][
                        "match_residuals"
                    ],
                    "local_to_link6_matrix": initial[name][
                        "local_to_link6_matrix"
                    ],
                }
                for name in FINGER_PRIMS
            },
            "dof_order": DOF_ORDER,
            "arm_poses_rad": ARM_POSES,
            "openings_m": OPENINGS,
            "gripper_center_offset_m": GRIPPER_CENTER_OFFSET_M,
            "thresholds": THRESHOLDS,
        }
        source_sha = write_canonical_json(SELECTOR_SOURCE_PATH, source_document)

        # ---- rerun discovery once against the finalized source hash ------
        rerun = run_selector_discovery(stage, signatures)["rerun"]
        rerun_identical = all(
            rerun[name]["passed"]
            and rerun[name]["selection"] == initial[name]["selection"]
            for name in FINGER_PRIMS
        )
        selector_passed = selector_passed and rerun_identical
        selector_record = {
            "schema": SCHEMA_SELECTOR,
            "phase": "tcp-project-calibration/selector-discovery",
            "model": MODEL,
            "collected_date_utc": time.strftime("%Y-%m-%d", time.gmtime()),
            "headless": True,
            "physics_played": False,
            "stage": dict(
                stage_meta,
                meters_per_unit_expected=1.0,
                up_axis_expected="Z",
            ),
            "method": METHOD,
            "algorithm_version": ALGORITHM_VERSION,
            "inputs": {
                "runtime_usd": {
                    "path": USD_PATH.relative_to(ROOT).as_posix(),
                    "sha256": usd_sha,
                    "expected_sha256": EXPECTED_USD_SHA256,
                    "match": usd_sha == EXPECTED_USD_SHA256,
                },
                "runtime_urdf": {
                    "path": URDF_PATH.relative_to(ROOT).as_posix(),
                    "sha256": urdf_sha,
                    "expected_sha256": EXPECTED_URDF_SHA256,
                    "match": urdf_sha == EXPECTED_URDF_SHA256,
                },
                "stl_sources": {
                    name: {
                        "path": signature["path"],
                        "sha256": signature["sha256"],
                        "frozen_candidate_match": {
                            "area_mm2": {
                                "reviewed": signature["area_mm2"],
                                "frozen": FROZEN_PATCH_AREA_MM2,
                                "tolerance_mm2": THRESHOLDS[
                                    "frozen_signature_area_mm2"
                                ],
                            },
                            "centroid_m": {
                                "reviewed": signature["centroid_m"],
                                "frozen": list(FROZEN_PATCH_CENTROID_M),
                                "tolerance_m": THRESHOLDS[
                                    "frozen_signature_centroid_m"
                                ],
                            },
                        },
                    }
                    for name, signature in signatures.items()
                },
                "selector_source": {
                    "path": SELECTOR_SOURCE_PATH.relative_to(ROOT).as_posix(),
                    "sha256": source_sha,
                },
            },
            "prims": dict(initial),
            "rerun_verification": {
                name: {
                    "passed": rerun[name]["passed"],
                    "selection_identical": (
                        rerun[name]["selection"] == initial[name]["selection"]
                    ),
                }
                for name in FINGER_PRIMS
            },
            "failures": [] if selector_passed else [
                "rerun selection differs from the finalized source selection"
            ]
            if not rerun_identical
            else [],
            "passed": selector_passed,
        }
        selector_sha = write_canonical_json(SELECTOR_ARTIFACT_PATH, selector_record)
        if not selector_passed:
            print(json.dumps({
                "schema": SCHEMA_SELECTOR,
                "stage": "selector",
                "passed": False,
            }, sort_keys=True), flush=True)
            return 2
        print(json.dumps({
            "schema": SCHEMA_SELECTOR,
            "stage": "selector",
            "passed": True,
            "selector_artifact_sha256": selector_sha,
            "selector_source_sha256": source_sha,
            "selected_faces": {
                name: initial[name]["selection"]["face_indices"]
                for name in FINGER_PRIMS
            },
        }, sort_keys=True), flush=True)
        if discovery_only:
            return 0

        # ------------------------------------------------------------------
        # Stage 2 precondition: the finalized, PASSED discovery evidence on
        # disk (not in-memory state) authorizes the 15-sample calibration.
        # ------------------------------------------------------------------
        on_disk = json.loads(
            SELECTOR_ARTIFACT_PATH.read_text(encoding="utf-8")
        )
        source_on_disk = json.loads(
            SELECTOR_SOURCE_PATH.read_text(encoding="utf-8")
        )
        artifact_sha_on_disk = sha256_file(SELECTOR_ARTIFACT_PATH)
        source_sha_on_disk = sha256_file(SELECTOR_SOURCE_PATH)
        stage2_prerequisites = {
            "selector_artifact_sha256_matches_disk": (
                artifact_sha_on_disk == selector_sha
                and artifact_sha_on_disk == sha256_file(SELECTOR_ARTIFACT_PATH)
            ),
            "selector_source_sha256_matches_disk": (
                source_sha_on_disk == source_sha
                and source_sha_on_disk
                == sha256_file(SELECTOR_SOURCE_PATH)
            ),
            "artifact_records_passed": on_disk.get("passed") is True,
            "artifact_schema_ok": (
                on_disk.get("schema") == SCHEMA_SELECTOR
            ),
            "source_schema_ok": (
                source_on_disk.get("schema") == SCHEMA_SOURCE
            ),
            "artifact_source_hash_consistent": (
                on_disk.get("inputs", {}).get("selector_source", {}).get("sha256")
                == source_sha_on_disk
            ),
            "runtime_usd_hash_consistent": (
                on_disk.get("inputs", {}).get("runtime_usd", {}).get("sha256")
                == usd_sha
            ),
            "both_fingers_unique_reviewed_selection": all(
                on_disk.get("prims", {}).get(name, {}).get("passed") is True
                and on_disk.get("prims", {}).get(name, {}).get("usd_candidate_count")
                == THRESHOLDS["selector_required_unique_match"]
                and on_disk.get("prims", {}).get(name, {}).get("selection", {}).get(
                    "face_indices"
                )
                == list(range(STL_FACET_RANGE[0], STL_FACET_RANGE[1] + 1))
                for name in FINGER_PRIMS
            ),
        }
        if not all(stage2_prerequisites.values()):
            failed = [
                k for k, v in stage2_prerequisites.items() if not v
            ]
            raise RuntimeError(
                "stage 2 prerequisites on the finalized discovery evidence "
                f"failed: {failed}"
            )

        # -------- Stage 2: bounded 15-sample calibration ------------------
        from isaacsim.core.api import PhysicsContext, World
        from isaacsim.core.prims import SingleArticulation
        import omni.timeline
        from pxr import Usd, UsdGeom

        PhysicsContext()
        world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
        robot = world.scene.add(
            SingleArticulation(prim_path=ARTICULATION_PRIM, name="piper_l")
        )
        world.reset()
        dof_names = [str(n) for n in robot.dof_names]
        properties = robot.dof_properties
        dof_limits = {
            name: [float(properties["lower"][index]), float(properties["upper"][index])]
            for index, name in enumerate(dof_names)
        }
        live_joint_positions = {
            name: float(value)
            for name, value in zip(dof_names, robot.get_joint_positions())
        }

        # The physics reset zeroes the USD link transforms in this build (the
        # live sim poses live in the PhysX fabric, not the USD source layer),
        # so the FK convention is validated against the independent URDF
        # zero-pose chain and the live rest state is pinned by the recorded
        # live joint positions instead of USD link reads. The FK root is the
        # identity world frame; the calibrated quantity link6 -> tcp is
        # root-invariant because link6 and both finger meshes share it.
        base_matrix = identity_matrix()
        joint_specs = read_joint_specs(stage)
        urdf_zero = urdf_zero_fk(URDF_PATH)
        fk_zero = forward_kinematics(joint_specs, base_matrix, {})
        fk_vs_urdf = {
            body: _norm(_sub(
                apply_matrix(fk_zero[body], (0.0, 0.0, 0.0)),
                apply_matrix(urdf_zero[body], (0.0, 0.0, 0.0)),
            ))
            for body in FK_BODIES
        }
        if max(fk_vs_urdf.values()) > THRESHOLDS["fk_urdf_zero_validation_m"]:
            raise RuntimeError(
                "FK joint-frame convention disagrees with the independent "
                f"URDF zero-pose chain: {fk_vs_urdf}"
            )
        world.stop()
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        link6_prim = stage.GetPrimAtPath(LINK6_PRIM)
        instances = {
            name: stage.GetPrimAtPath(path)
            for name, path in FINGER_PRIMS.items()
        }
        proto_mesh_prims = {
            name: instances[name].GetPrototype().GetPath().AppendPath(
                initial[name]["prototype_relative_mesh_path"]
            )
            for name in FINGER_PRIMS
        }
        proto_mesh_prims = {
            name: stage.GetPrimAtPath(path)
            for name, path in proto_mesh_prims.items()
        }
        session = stage.GetSessionLayer()
        samples = []
        sim_time_deltas = []
        for pose_name, pose_values in ARM_POSES.items():
            for opening in OPENINGS:
                joint_values = dict(zip(ARM_JOINTS, pose_values))
                joint_values["gripper_joint1"] = opening / 2.0
                joint_values["gripper_joint2"] = -opening / 2.0
                requested_finger = {
                    "gripper_link1": opening / 2.0,
                    "gripper_link2": -opening / 2.0,
                }
                limit_violations = [
                    {
                        "dof": name,
                        "value": joint_values[name],
                        "live_limits": dof_limits[name],
                    }
                    for name in DOF_ORDER
                    if not (
                        dof_limits[name][0] <= joint_values[name]
                        <= dof_limits[name][1]
                    )
                ]
                fk = forward_kinematics(joint_specs, base_matrix, joint_values)
                sim_time_before = float(world.current_time)
                with Usd.EditContext(stage, session):
                    for body in FK_BODIES:
                        author_link_transform(
                            stage, ROBOT_PRIM + "/" + body, fk[body]
                        )
                cache.Clear()
                reads = []
                for _ in range(3):
                    link6_matrix = gf_to_col(
                        cache.GetLocalToWorldTransform(link6_prim)
                    )
                    instance_matrices = {
                        name: gf_to_col(
                            cache.GetLocalToWorldTransform(instances[name])
                        )
                        for name in FINGER_PRIMS
                    }
                    proto_matrices = {
                        name: gf_to_col(cache.GetLocalToWorldTransform(
                            proto_mesh_prims[name]
                        ))
                        for name in FINGER_PRIMS
                    }
                    reads.append((link6_matrix, instance_matrices, proto_matrices))
                sim_time_deltas.append(
                    float(world.current_time) - sim_time_before
                )
                link6_matrix = reads[0][0]
                patches = {}
                for name in FINGER_PRIMS:
                    mesh_world = mat_mul(reads[0][1][name], reads[0][2][name])
                    patches[name] = measured_patch_world(
                        initial[name]["selection"], mesh_world
                    )
                repeat_spread = 0.0
                for name in FINGER_PRIMS:
                    centroids = [
                        measured_patch_world(
                            initial[name]["selection"],
                            mat_mul(read[1][name], read[2][name]),
                        )["centroid_m"]
                        for read in reads
                    ]
                    for i in range(3):
                        for j in range(i + 1, 3):
                            repeat_spread = max(
                                repeat_spread,
                                _norm(_sub(centroids[i], centroids[j])),
                            )
                measured_finger = measure_finger_displacements(
                    joint_specs, fk["gripper_base"], reads[0][1]
                )
                metrics = evaluate_sample({
                    "opening": opening,
                    "link6_matrix": link6_matrix,
                    "patch1": patches["gripper_link1"],
                    "patch2": patches["gripper_link2"],
                    "measured_finger": measured_finger,
                    "requested_finger": requested_finger,
                    "repeat_centroid_spread_m": repeat_spread,
                })
                samples.append({
                    "pose": pose_name,
                    "opening_m": opening,
                    "requested_joints": dict(joint_values),
                    "requested_finger_displacement_m": requested_finger,
                    "measured_finger_displacement_m": measured_finger,
                    "within_live_limits": not limit_violations,
                    "joint_limit_violations": limit_violations,
                    "link6_world_matrix": matrix_to_list(link6_matrix),
                    "patches": {
                        name: {
                            "centroid_world_m": list(
                                patches[name]["centroid_m"]
                            ),
                            "normal_world": list(patches[name]["normal"]),
                            "area_m2": patches[name]["area_m2"],
                            "triangles": patches[name]["triangles"],
                        }
                        for name in FINGER_PRIMS
                    },
                    "metrics": metrics,
                })
        timeline_playing = bool(timeline.is_playing())
        gate_results, aggregate, failures = aggregate_and_gate(
            samples, stage_meta, timeline_playing, sim_time_deltas,
            dof_names, dof_limits,
        )
        calibration_passed = not failures
        calibration_record = {
            "schema": SCHEMA_CALIBRATION,
            "phase": "tcp-project-calibration/measurement",
            "model": MODEL,
            "collected_date_utc": time.strftime("%Y-%m-%d", time.gmtime()),
            "isaac": {
                "version": ISAAC_VERSION,
                "headless": True,
                "simulation_only": True,
                "hardware_validated": False,
            },
            "scope": {
                "kind": "project_calibration",
                "calibration_authority": CALIBRATION_AUTHORITY,
                "runtime_usd_scoped": True,
                "runtime_usd_sha256": usd_sha,
                "simulation_only": True,
                "hardware_validated": False,
            },
            "source": {
                "path": SELECTOR_SOURCE_PATH.relative_to(ROOT).as_posix(),
                "sha256": source_sha,
            },
            "selector_discovery": {
                "path": SELECTOR_ARTIFACT_PATH.relative_to(ROOT).as_posix(),
                "sha256": selector_sha,
                "passed": selector_passed,
            },
            "method": METHOD,
            "tcp_definition": TCP_DEFINITION,
            "articulation": {
                "dof_order_live": dof_names,
                "dof_order_expected": DOF_ORDER,
                "dof_limits_live_rad_or_m": dof_limits,
                "live_joint_positions_at_reset": live_joint_positions,
                "joint_limit_violations": [
                    {
                        "pose": s["pose"],
                        "opening_m": s["opening_m"],
                        "violations": s["joint_limit_violations"],
                    }
                    for s in samples
                    if s["joint_limit_violations"]
                ],
                "initialization": (
                    "exactly one world.reset() (bounded initialization, "
                    "required to create the live articulation physics view; "
                    "it internally plays and steps once); timeline stopped "
                    "immediately after and never played or stepped again. The "
                    "physics reset zeroes the USD link transforms in this "
                    "Isaac build (live poses live in the PhysX fabric), so "
                    "the FK convention is validated against the independent "
                    "URDF zero-pose chain, not against USD link reads"
                ),
                "script_driven_world_step_or_play_calls": 0,
                "fk_zero_vs_urdf_origin_m": fk_vs_urdf,
                "stage2_prerequisites_on_finalized_discovery": (
                    stage2_prerequisites
                ),
            },
            "samples": samples,
            "aggregate": aggregate,
            "thresholds": THRESHOLDS,
            "gates": {name: bool(v) for name, v in gate_results.items()},
            "failures": failures,
            "transform": (
                {
                    "parent_frame": "link6",
                    "child_frame": "rocobrick_tcp",
                    "translation_m": aggregate["translation_median_m"],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "measured_by": METHOD,
                    "authority": CALIBRATION_AUTHORITY,
                    "source_artifact": CALIBRATION_ARTIFACT_PATH.relative_to(
                        ROOT
                    ).as_posix(),
                }
                if calibration_passed
                else None
            ),
            "passed": calibration_passed,
        }
        write_canonical_json(CALIBRATION_ARTIFACT_PATH, calibration_record)
        print(json.dumps({
            "schema": SCHEMA_CALIBRATION,
            "stage": "calibration",
            "passed": calibration_passed,
            "selector_passed": selector_passed,
            "translation_m": (
                aggregate["translation_median_m"] if calibration_passed else None
            ),
            "failures": failures,
            "elapsed_s": round(time.time() - started, 3),
        }, sort_keys=True), flush=True)
        return 0 if calibration_passed else 3
    except Exception as error:
        import traceback

        traceback.print_exc()
        failures = [f"{type(error).__name__}: {error}"]
        try:
            if SELECTOR_ARTIFACT_PATH.exists():
                selector_record = json.loads(
                    SELECTOR_ARTIFACT_PATH.read_text(encoding="utf-8")
                )
                if selector_record.get("schema") == SCHEMA_SELECTOR:
                    selector_record["passed"] = False
                    selector_record["failures"] = failures
                    write_canonical_json(
                        SELECTOR_ARTIFACT_PATH, selector_record
                    )
            else:
                write_canonical_json(
                    SELECTOR_ARTIFACT_PATH,
                    {
                        "schema": SCHEMA_SELECTOR,
                        "model": MODEL,
                        "passed": False,
                        "failures": failures,
                    },
                )
        except Exception:
            pass
        print(json.dumps({
            "schema": SCHEMA_SELECTOR,
            "stage": "error",
            "error": failures[0],
        }, sort_keys=True), flush=True)
        return 1
    finally:
        cleanup = {}
        try:
            if world is not None:
                world.stop()
                cleanup["world_stopped"] = True
        except Exception as exc:
            cleanup["world_stopped"] = f"error: {exc}"
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
            cleanup["timeline_stopped"] = True
        except Exception as exc:
            cleanup["timeline_stopped"] = f"error: {exc}"
        if app is not None:
            cleanup["app_close_initiated"] = True
            try:
                app.close()
                cleanup["app_closed"] = True
            except Exception as exc:
                cleanup_error = exc
                cleanup["app_closed"] = f"error: {exc}"
        print(json.dumps({"cleanup": cleanup}, sort_keys=True), flush=True)
        if cleanup_error is not None:
            raise cleanup_error


if __name__ == "__main__":
    raise SystemExit(main())
