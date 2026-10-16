from pathlib import Path

import pytest

from tools.validate_piper_assets import validate


ROOT = Path(__file__).parents[1]
EXECUTION_URDF = (
    ROOT / "isaac/vendor/agx_arm_sim/agx_arm_description/urdf/piper_l_gripper_d435.urdf"
)


def test_runtime_urdf_matches_pinned_execution_asset():
    validate(ROOT / "robot_assets/piper_l/piper_l_gripper_d435.urdf", EXECUTION_URDF)


def test_widened_runtime_urdf_is_rejected(tmp_path):
    edited = tmp_path / "piper_l_gripper_d435.urdf"
    source = EXECUTION_URDF.read_text(encoding="utf-8")
    edited.write_text(
        source.replace('lower="-1.7453292"', 'lower="-2.2165681"'),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not byte-identical"):
        validate(edited, EXECUTION_URDF)
