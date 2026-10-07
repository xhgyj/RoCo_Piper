"""Native CPU geometry tests; no ROS, controller, Isaac or motion is started."""

import hashlib
import itertools
import math
from dataclasses import replace

import pytest
from test_handover_joint_planning import planning_request

from rocobrick.adapters.moveit.handover_geometry import (
    HandoverModelFiles,
    IsolatedHandoverGeometry,
)
from rocobrick.skills.handover_contact import SimulationFingerContactPolicy
from rocobrick.skills.handover_simulation import SimulationHandoverAuthorization
from rocobrick.skills.observation import Pose


def native_scene(
    tmp_path, *, wrist_collision=False, cross_exemption=False, base_collision=False
):
    """Build a primitive two-arm model to exercise the actual native kernel.

    Returns:
        An isolated geometry mirror and its captured complete request.
    """
    pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    request = planning_request()
    request.scene_snapshot["robots"]["piper_0"]["tcp_pose"]["position_xyz"] = [
        0.35,
        0,
        0.35,
    ]
    request.scene_snapshot["environment_colliders"] = {
        "floor": {"aabb_min": [-2, -2, -0.05], "aabb_max": [2, 2, -0.03]},
    }
    start = dict(request.start_state)
    elements = ['<robot name="native_handover_test"><link name="world"/>']
    exclusions = []
    for robot, x, offset, rotation in (
        ("piper_0", 0.25, ".25 -.1 .35", f"0 0 {-math.pi / 2}"),
        ("piper_1", -0.25, ".25 -.26 .21", f"{math.pi} 0 {-math.pi / 2}"),
    ):
        base = robot + "_base_link"
        base_shape = (
            '<collision><origin xyz="0 0 -.04"/>'
            '<geometry><box size=".02 .02 .02"/></geometry></collision>'
            if base_collision
            else ""
        )
        elements.append(
            f'<link name="{base}">{base_shape}</link>'
            f'<joint name="{robot}_base" type="fixed">'
            f'<parent link="world"/><child link="{base}"/>'
            f'<origin xyz="{x} -.25 0" rpy="0 0 {math.pi / 2}"/></joint>'
        )
        links = [base]
        parent = base
        for index, (kind, axis) in enumerate(
            (
                ("prismatic", "1 0 0"),
                ("prismatic", "0 1 0"),
                ("prismatic", "0 0 1"),
                ("revolute", "1 0 0"),
                ("revolute", "0 1 0"),
                ("revolute", "0 0 1"),
            ),
            1,
        ):
            link = f"{robot}_link{index}"
            collision = (
                '<collision><geometry><sphere radius=".001"/></geometry>'
                f'<origin xyz="0 0 {index * 0.02}"/></collision>'
            )
            if robot == "piper_1" and index == 6 and wrist_collision:
                collision += (
                    '<collision><geometry><sphere radius=".015"/></geometry>'
                    f'<origin xyz="{offset}"/></collision>'
                )
            elements.append(
                f'<link name="{link}">{collision}</link>'
                f'<joint name="{robot}_joint{index}" type="{kind}">'
                f'<parent link="{parent}"/><child link="{link}"/>'
                f'<axis xyz="{axis}"/><limit lower="-1" upper="1" '
                'effort="100" velocity="1"/></joint>'
            )
            links.append(link)
            parent = link
        tcp = robot + "_rocobrick_tcp"
        elements.append(
            f'<link name="{tcp}"/><joint name="{robot}_tcp" type="fixed">'
            f'<parent link="{parent}"/><child link="{tcp}"/>'
            f'<origin xyz="{offset}" rpy="{rotation}"/></joint>'
        )
        links.append(tcp)
        for index in (1, 2):
            finger = f"{robot}_gripper_link{index}"
            joint = f"{robot}_gripper_joint{index}"
            start[joint] = 0.008 if index == 1 else -0.008
            elements.append(
                f'<link name="{finger}"><collision><geometry>'
                '<box size=".004 .001 .01"/></geometry></collision></link>'
                f'<joint name="{joint}" type="prismatic"><parent link="{tcp}"/>'
                f'<child link="{finger}"/><axis xyz="0 1 0"/>'
                '<limit lower="-.05" upper=".05" effort="10" velocity=".1"/></joint>'
            )
            links.append(finger)
        exclusions.extend(
            f'<disable_collisions link1="{a}" link2="{b}" reason="Adjacent"/>'
            for a, b in itertools.combinations(links, 2)
        )
    if cross_exemption:
        exclusions.append(
            '<disable_collisions link1="piper_0_link6" '
            'link2="piper_1_link6" reason="Never"/>'
        )
    elements.append("</robot>")
    urdf, srdf = tmp_path / "combined.urdf", tmp_path / "combined.srdf"
    urdf.write_text("".join(elements), encoding="utf-8")
    srdf.write_text(
        '<robot name="native_handover_test">' + "".join(exclusions) + "</robot>",
        encoding="utf-8",
    )
    request = replace(request, start_state=start)
    files = HandoverModelFiles(
        urdf,
        srdf,
        (),
        hashlib.sha256(urdf.read_bytes()).hexdigest(),
        hashlib.sha256(srdf.read_bytes()).hexdigest(),
        request.profile.runtime_model_sha256,
    )
    return IsolatedHandoverGeometry(files, request), request


def test_native_full_model_bounds_and_clearance(tmp_path):
    """Both complete arms and grippers are checked against native geometry."""
    scene, request = native_scene(tmp_path)
    verdict = scene.check(request.start_state, holder="piper_1")
    assert verdict.valid and verdict.clearance_m > 0


def finger_contact_scene(
    tmp_path, *, depth=0.0002, outside=False, wrist_collision=False
):
    """Deliberately place two primitive fixture fingers in a known contact.

    This is a synthetic native-kernel regression, not a Piper calibration edit.
    All other links, limits, world and object geometry remain in the fixture.

    Returns:
        The explicitly authorized fixture and its unchanged planning inputs.
    """
    base, request = native_scene(tmp_path, wrist_collision=wrist_collision)
    policy = SimulationFingerContactPolicy()
    token = SimulationHandoverAuthorization(
        request.scene_snapshot["session_id"],
        request.scene_snapshot["manifest_sha256"],
        request.profile.sha256,
        request.profile.runtime_model_sha256,
        policy.sha256,
    )
    scene = IsolatedHandoverGeometry(
        base.files,
        request,
        finger_contact_policy=policy,
        simulation_authorization=token,
    )
    scene.pin.framesForwardKinematics(
        scene.model, scene._data, scene.configuration(request.start_state)
    )
    center = scene.np.array(
        request.observation(request.donor.robot_id).object_pose.position_xyz
    )
    if outside:
        center[0] += 0.1
    for robot, offset in (
        (request.donor.robot_id, 0),
        (request.receiver.robot_id, 0.001 - depth),
    ):
        index = scene._link_names.index(robot + "_gripper_link1")
        shape = scene.geometry.geometryObjects[index]
        world = scene.pin.SE3(scene.np.eye(3), center + scene.np.array([0, offset, 0]))
        shape.placement = scene._data.oMi[shape.parentJoint].actInv(world)
    return scene, request


def test_native_contact_requires_exact_simulation_authorization(tmp_path):
    """Contact permission cannot bypass the exact trial authorization."""
    base, request = native_scene(tmp_path)
    with pytest.raises(ValueError, match="authorization"):
        IsolatedHandoverGeometry(
            base.files, request, finger_contact_policy=SimulationFingerContactPolicy()
        )
    with pytest.raises(ValueError, match="bound simulation"):
        base.check(
            request.start_state, holder=request.donor.robot_id, finger_contact=True
        )


def test_native_shallow_contact_is_not_a_global_collision_exemption(tmp_path):
    """A shallow fixture contact is allowed only with explicit contact mode."""
    scene, request = finger_contact_scene(tmp_path)
    kwargs = dict(holder=request.donor.robot_id, receiver_touch=True)
    assert not scene.check(request.start_state, **kwargs).valid
    assert scene.check(request.start_state, finger_contact=True, **kwargs).valid
    # Returning to ordinary checking must restore the strict inter-arm rule.
    assert not scene.check(request.start_state, **kwargs).valid
    assert scene.certify_segment(
        request.start_state,
        request.start_state,
        finger_contact=True,
        deadline_s=10,
        clock=lambda: 0,
        **kwargs,
    )


def test_native_guarded_segment_can_cross_zero_clearance(tmp_path):
    """A permitted contact path must not stall at infinitesimal positive gap."""
    scene, request = finger_contact_scene(tmp_path, depth=-0.0001)
    final = dict(request.start_state)
    final[request.receiver.robot_id + "_gripper_joint1"] -= 0.0002
    kwargs = dict(holder=request.donor.robot_id, receiver_touch=True)
    assert scene.check(request.start_state, **kwargs).valid
    assert not scene.check(final, **kwargs).valid
    assert scene.certify_segment(
        request.start_state,
        final,
        finger_contact=True,
        deadline_s=10,
        clock=lambda: 0,
        **kwargs,
    )


def test_native_contact_region_checks_whole_patch_not_single_witness(tmp_path):
    """A central contact point cannot exempt an oversized contact surface."""
    scene, request = finger_contact_scene(tmp_path)
    boxes = list(scene._local_collision_boxes)
    for robot in (request.donor.robot_id, request.receiver.robot_id):
        index = scene._link_names.index(robot + "_gripper_link1")
        scene.geometry.geometryObjects[index].geometry = scene.coal.Box(
            0.004, 0.001, 0.04
        )
        boxes[index] = (scene.np.zeros(3), scene.np.array([0.002, 0.0005, 0.02]))
    scene._local_collision_boxes = tuple(boxes)
    scene._geometry_data = scene.pin.GeometryData(scene.geometry)
    scene._relative_layout_key = None
    kwargs = dict(
        holder=request.donor.robot_id, receiver_touch=True, finger_contact=True
    )
    assert scene.check(request.start_state, **kwargs).valid
    assert not scene.certify_segment(
        request.start_state,
        request.start_state,
        deadline_s=10,
        max_checks=100,
        clock=lambda: 0,
        **kwargs,
    )
    assert scene.last_contact_region_failure is not None


@pytest.mark.parametrize(
    "values", [{"depth": 0.0008}, {"outside": True}, {"wrist_collision": True}]
)
def test_native_deep_outside_or_wrist_collision_remains_forbidden(tmp_path, values):
    """Finger policy cannot exempt deep contact, shafts or wrist collision."""
    scene, request = finger_contact_scene(tmp_path, **values)
    assert not scene.check(
        request.start_state,
        holder=request.donor.robot_id,
        receiver_touch=True,
        finger_contact=True,
    ).valid
    outside = {**request.start_state, "piper_0_joint1": 1.01}
    assert (
        scene.check(outside, holder="piper_1").reason == "MODEL_JOINT_LIMIT_VIOLATION"
    )
    roundoff = {**request.start_state, "piper_0_joint1": -1.0 - 5e-10}
    assert scene.configuration(roundoff)[scene._variables["piper_0_joint1"][0]] == -1
    actual_excursion = {**request.start_state, "piper_0_joint1": -1.0 - 1e-6}
    assert (
        scene.check(actual_excursion, holder="piper_1").reason
        == "MODEL_JOINT_LIMIT_VIOLATION"
    )


def test_attachment_parent_joint_does_not_exempt_wrist_collision(tmp_path):
    """The held object remains checked against geometry on its parent joint."""
    scene, request = native_scene(tmp_path, wrist_collision=True)
    result = scene.check(request.start_state, holder="piper_1")
    assert not result.valid
    assert any("held::brick-1" in pair for pair in result.collision_pairs)


def test_broadphase_never_overstates_native_clearance(tmp_path):
    """Mesh-derived broad-phase gaps remain conservative clearance bounds."""
    scene, request = native_scene(tmp_path)
    assert scene.check(request.start_state, holder="piper_1").valid
    for index in range(len(scene.geometry.collisionPairs)):
        if not scene._geometry_data.activeCollisionPairs[index]:
            continue
        assert not scene.pin.computeCollision(
            scene.geometry, scene._geometry_data, index
        )
        exact = scene.pin.computeDistance(
            scene.geometry, scene._geometry_data, index
        ).min_distance
        assert scene._pair_clearances[index] <= exact + 1e-8


def test_cross_robot_srdf_exemption_is_refused(tmp_path):
    """A combined SRDF cannot hide collision between the two arms."""
    with pytest.raises(ValueError, match="cross-robot"):
        native_scene(tmp_path, cross_exemption=True)


def test_clear_endpoints_cannot_admit_a_colliding_interior_segment(tmp_path):
    """Clearance certification examines the whole segment, not just endpoints."""
    scene, request = native_scene(tmp_path)
    # Rebuild from an explicitly changed request, rather than mutate the clone.
    request.scene_snapshot["environment_colliders"]["obstacle"] = {
        "aabb_min": [0.348, 0.028, 0.34],
        "aabb_max": [0.352, 0.052, 0.36],
    }
    scene = IsolatedHandoverGeometry(scene.files, request)
    final = {**request.start_state, "piper_0_joint1": 0.08}
    assert scene.check(request.start_state, holder="piper_1").valid
    assert scene.check(final, holder="piper_1").valid
    assert not scene.certify_segment(
        request.start_state, final, deadline_s=1.0, clock=lambda: 0.0
    )


def test_native_segment_and_ik_have_finite_budgets(tmp_path):
    """Real FK/IK and clearance checks do not relax bounds or wait forever."""
    scene, request = native_scene(tmp_path)
    final = {**request.start_state, "piper_0_joint1": 0.0001}
    assert scene.certify_segment(
        request.start_state, final, deadline_s=1.0, clock=lambda: 0.0
    )
    assert not scene.certify_segment(
        request.start_state, final, deadline_s=0.0, clock=lambda: 0.0
    )
    original = request.observation("piper_0").tcp_pose
    target = Pose(
        (original.position_xyz[0], 0.02, original.position_xyz[2]),
        original.orientation_xyzw,
    )
    solution = scene.solve_tcp_ik(
        "piper_0",
        target,
        request.start_state,
        deadline_s=1.0,
        clock=lambda: 0.0,
        attempts=2,
    )
    assert solution is not None
    assert solution["piper_0_joint1"] == pytest.approx(0.02, abs=0.001)
    assert (
        scene.solve_tcp_ik(
            "piper_0",
            target,
            request.start_state,
            deadline_s=0.0,
            clock=lambda: 0.0,
            attempts=2,
        )
        is None
    )


def test_fixed_mount_allowance_cannot_hide_moving_links(tmp_path):
    """Explicit installation contacts never exclude moving geometry."""
    scene, request = native_scene(tmp_path, base_collision=True)
    assert not scene.check(request.start_state, holder="piper_1").valid
    mounts = tuple(
        (r + "_base_link", "world::environment::floor") for r in ("piper_0", "piper_1")
    )
    mirror = IsolatedHandoverGeometry(scene.files, request, fixed_mount_contacts=mounts)
    assert mirror.check(request.start_state, holder="piper_1").valid
    with pytest.raises(ValueError, match="installation"):
        IsolatedHandoverGeometry(
            scene.files,
            request,
            fixed_mount_contacts=(("piper_0_link6", "world::environment::floor"),),
        )
    with pytest.raises(ValueError, match="world-fixed"):
        IsolatedHandoverGeometry(
            scene.files,
            request,
            fixed_mount_contacts=(
                ("piper_0_base_link", "world::environment::missing"),
            ),
        )


def test_static_tracking_uncertainty_cannot_be_ignored(tmp_path):
    """A nominally clear stationary path still checks its tracking envelope."""
    scene, request = native_scene(tmp_path)
    request.scene_snapshot["environment_colliders"]["nearby"] = {
        "aabb_min": [0.254, -0.251, 0.019],
        "aabb_max": [0.258, -0.249, 0.021],
    }
    nominal = IsolatedHandoverGeometry(scene.files, request)
    uncertain = IsolatedHandoverGeometry(scene.files, request, tracking_error_rad=0.005)
    args = dict(
        holder=request.donor.robot_id,
        deadline_s=1.0,
        clock=lambda: 0.0,
        max_checks=64,
    )
    assert nominal.certify_segment(request.start_state, request.start_state, **args)
    assert not uncertain.certify_segment(
        request.start_state, request.start_state, **args
    )


def test_aabb_lower_bound_is_refined_before_tracking_clearance_rejection(tmp_path):
    """A box near a sphere's AABB corner is farther from the actual sphere."""
    base, request = native_scene(tmp_path)
    request.scene_snapshot["environment_colliders"]["corner"] = {
        "aabb_min": [0.2511, -0.2491, 0.0199],
        "aabb_max": [0.2512, -0.2490, 0.0201],
    }
    scene = IsolatedHandoverGeometry(base.files, request, tracking_error_rad=0.00008)
    assert scene.check(request.start_state, holder=request.donor.robot_id).valid
    index = next(
        i
        for i, pair in enumerate(scene.geometry.collisionPairs)
        if {scene.geometry.geometryObjects[j].name for j in (pair.first, pair.second)}
        == {"piper_0_link1_0", "world::environment::corner"}
    )
    lower = scene._pair_clearances[index]
    assert 0 < lower < 2 * scene.tracking_error_rad
    assert index not in scene._exact_pair_distances
    assert scene._pairwise_clearance_certified(request.start_state, request.start_state)
    assert index in scene._exact_pair_distances
    assert scene._pair_clearances[index] > 2 * scene.tracking_error_rad
    assert scene.tracking_error_rad == 0.00008
    assert scene.certify_segment(
        request.start_state,
        request.start_state,
        holder=request.donor.robot_id,
        deadline_s=1.0,
        clock=lambda: 0.0,
    )


def test_captured_model_files_cannot_change_after_native_scene_load(tmp_path):
    """Runtime factory captures actual exports and rechecks immutable inputs."""
    scene, request = native_scene(tmp_path)
    files = HandoverModelFiles.capture(
        scene.files.urdf_path,
        scene.files.srdf_path,
        package_dirs=(),
        runtime_model_sha256=request.profile.runtime_model_sha256,
    )
    assert files == scene.files
    files.srdf_path.write_text('<robot name="changed"/>', encoding="utf-8")
    with pytest.raises(ValueError, match="changed on disk"):
        files.validate()


def test_collision_mesh_capture_requires_exact_asset_provenance(tmp_path):
    """Missing mesh files cannot become a collision-free primitive fallback."""
    scene, request = native_scene(tmp_path)
    scene.files.urdf_path.write_text(
        '<robot name="mesh"><link name="world"><collision><geometry>'
        '<mesh filename="package://missing/mesh.stl"/>'
        "</geometry></collision></link></robot>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing or ambiguous"):
        HandoverModelFiles.capture(
            scene.files.urdf_path,
            scene.files.srdf_path,
            package_dirs=(tmp_path,),
            runtime_model_sha256=request.profile.runtime_model_sha256,
        )
