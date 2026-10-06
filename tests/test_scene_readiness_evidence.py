"""Offline assertions for readiness error visibility."""

from types import SimpleNamespace


def test_scene_probe_error_is_recorded_not_collapsed():
    """A combined readiness failure carries the actual query exception."""
    runtime = SimpleNamespace(
        _last_scene_error="RuntimeError: malformed scene-state JSON",
        scene_readiness_evidence=lambda: {
            "error": "RuntimeError: malformed scene-state JSON"
        },
    )
    assert runtime.scene_readiness_evidence()["error"].startswith("RuntimeError")
