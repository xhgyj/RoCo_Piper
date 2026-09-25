from rocobrick.robot_runtime_profile import RobotRuntimeProfile, assert_profiles_disjoint

def test_profiles_are_namespaced_and_disjoint():
    a = RobotRuntimeProfile.from_robot_id("piper_0")
    b = RobotRuntimeProfile.from_robot_id("piper_1")
    assert a.joint_names != b.joint_names
    assert a.arm_follow != b.arm_follow
    assert_profiles_disjoint((a, b))

def test_profile_rejects_invalid_namespace():
    try:
        RobotRuntimeProfile.from_robot_id("piper_0", "bad namespace")
    except ValueError:
        return
    raise AssertionError("invalid namespace was accepted")
