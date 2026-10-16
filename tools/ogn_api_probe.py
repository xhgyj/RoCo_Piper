#!/usr/bin/env python3
"""One-shot bounded probe: which Kit APIs actually reflect bridge OGN readiness."""
from __future__ import annotations

import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import json

import omni.graph.core as og
import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

result = {
    "schema": "rocobrick-ogn-api-probe/v1",
    "has_get_node_type_registry": hasattr(og, "get_node_type_registry"),
    "og_registry_names": [n for n in dir(og) if "regist" in n.lower() or "node_type" in n.lower()],
    "manager_names": [n for n in dir(manager) if "state" in n.lower() or "info" in n.lower()],
    "states": [],
}
for _ in range(120):
    app.update()
probe_types = [
    "isaacsim.ros2.bridge.ROS2Context",
    "isaacsim.ros2.bridge.ROS2PublishClock",
]
entry: dict = {"tick": 120}
for name in probe_types:
    try:
        entry[f"registry::{name}"] = repr(getattr(og, name, "absent"))
    except Exception as exc:  # noqa: BLE001
        entry[f"registry::{name}"] = f"error: {exc}"
try:
    stage = __import__("omni.usd").get_context().get_stage()
    entry["stage_available"] = stage is not None
except Exception as exc:  # noqa: BLE001
    entry["stage_available"] = f"error: {exc}"
result["states"].append(entry)
print(json.dumps(result, indent=2, sort_keys=True), flush=True)
app.close()
