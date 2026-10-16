"""Contract tests for the shared per-robot calibrated target context loader."""

import json
from pathlib import Path

import pytest

from rocobrick.skills.target_context import (
    GroundSpec,
    TargetContext,
    load_target_contexts,
)
from rocobrick.skills.targets import load_robot_base_frame, load_skill_params

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config/user_config.json"
SKILL_PARAMS_PATH = REPO_ROOT / "config/skill_params.json"
RUNTIME_MODEL_SHA256 = "a" * 64
SOURCE_SHA256 = "b" * 64
ROBOTS = ("piper_0", "piper_1")


def _config_with(tmp_path: Path, **manipulation) -> Path:
    """Return the real config with a derived Manipulation_Config override."""
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    document.setdefault("Manipulation_Config", {}).update(manipulation)
    path = tmp_path / "user_config.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _load(config_path=CONFIG_PATH, robot_ids=ROBOTS, **overrides):
    arguments = {
        "runtime_model_sha256": RUNTIME_MODEL_SHA256,
        "source_sha256": SOURCE_SHA256,
    }
    arguments.update(overrides)
    return load_target_contexts(
        config_path, SKILL_PARAMS_PATH, robot_ids, **arguments
    )


def test_loads_one_validated_context_per_robot():
    """Load exactly one validated context per requested robot."""
    contexts = _load()
    assert set(contexts) == set(ROBOTS)
    for robot_id, context in contexts.items():
        assert isinstance(context, TargetContext)
        assert context.base == load_robot_base_frame(CONFIG_PATH, robot_id)
        assert context.params == load_skill_params(SKILL_PARAMS_PATH)
        assert context.provenance.runtime_model_sha256 == RUNTIME_MODEL_SHA256
        assert context.provenance.source_sha256 == SOURCE_SHA256


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, None, 1])
def test_rejects_a_non_lowercase_sha256_runtime_model(digest):
    """Reject a runtime model digest that is not a lowercase SHA-256."""
    with pytest.raises(
        ValueError, match="runtime_model_sha256 must be a lowercase SHA-256 digest"
    ):
        _load(runtime_model_sha256=digest)


@pytest.mark.parametrize("digest", ["B" * 64, "b" * 65, "z" * 64, None, 1])
def test_rejects_a_non_lowercase_sha256_source(digest):
    """Reject a source digest that is not a lowercase SHA-256."""
    with pytest.raises(
        ValueError, match="source_sha256 must be a lowercase SHA-256 digest"
    ):
        _load(source_sha256=digest)


@pytest.mark.parametrize("robot_ids", [("piper_0", ""), ("piper_0", None), (1,)])
def test_rejects_unknown_robot_identifiers(robot_ids):
    """Reject robot identifiers that are not non-empty strings."""
    with pytest.raises(ValueError, match="robot_ids must contain non-empty strings"):
        _load(robot_ids=robot_ids)


def test_ground_spec_comes_from_the_finite_positive_geometry(tmp_path):
    """Read the ground spec from finite positive geometry."""
    config_path = _config_with(
        tmp_path,
        Ground_Size=[0.5, 0.4],
        Ground_Top_Z=0.02,
        Ground_Thickness=0.01,
    )
    ground = _load(config_path=config_path)["piper_0"].ground
    assert isinstance(ground, GroundSpec)
    assert len(ground.center_xy_m) == 2
    assert ground.size_xy_m == (0.5, 0.4)
    assert ground.top_z_m == 0.02
    assert ground.thickness_m == 0.01


@pytest.mark.parametrize(
    "manipulation",
    [
        {"Ground_Size": [0.5, 0.0], "Ground_Top_Z": 0.02, "Ground_Thickness": 0.01},
        {"Ground_Size": [0.5, 0.4], "Ground_Top_Z": 0.02, "Ground_Thickness": 0.0},
        {"Ground_Size": [0.5], "Ground_Top_Z": 0.02, "Ground_Thickness": 0.01},
        {"Ground_Size": [0.5, 0.4], "Ground_Top_Z": None, "Ground_Thickness": 0.01},
        {"Ground_Size": [0.5, 0.4], "Ground_Top_Z": "0.02", "Ground_Thickness": 0.01},
        {"Ground_Size": [0.5, 0.4], "Ground_Top_Z": True, "Ground_Thickness": 0.01},
    ],
)
def test_incomplete_or_nonpositive_geometry_degrades_to_no_ground(
    tmp_path, manipulation
):
    """Degrade to no ground on incomplete or non-positive geometry."""
    config_path = _config_with(tmp_path, **manipulation)
    assert _load(config_path=config_path)["piper_0"].ground is None


def test_a_missing_manipulation_section_degrades_to_no_ground(tmp_path):
    """Degrade to no ground when Manipulation_Config is missing."""
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    document.pop("Manipulation_Config", None)
    path = tmp_path / "user_config.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert _load(config_path=path)["piper_0"].ground is None
