#!/usr/bin/env python3
"""Publish a passed TCP calibration into the binding and MoveIt overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "artifacts/model-audit/piper_l_tcp_binding_contract.json"
CALIBRATION = ROOT / "artifacts/model-audit/piper_l_tcp_calibration.json"
SELECTOR = ROOT / "artifacts/model-audit/piper_l_tcp_selector_discovery.json"
SOURCE = ROOT / "config/calibration/piper_l_tcp_source.json"
OVERLAY = (ROOT / "ros_ws/src/rocobrick_piper_l_moveit_config/urdf"
           / "piper_l.tcp_calibration.xacro")


def digest(path: Path) -> str:
    """Return the SHA-256 of an input's original bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(template: Path) -> None:
    """Publish measured values only after checking calibration provenance."""
    calibration = json.loads(CALIBRATION.read_text())
    selector = json.loads(SELECTOR.read_text())
    source = json.loads(SOURCE.read_text())
    if calibration.get("passed") is not True or not all(calibration["gates"].values()):
        raise ValueError("calibration gates did not pass")
    if selector.get("passed") is not True:
        raise ValueError("selector did not pass")
    if selector["inputs"]["selector_source"]["sha256"] != digest(SOURCE):
        raise ValueError("selector source digest does not match")
    for kind in ("urdf", "usd"):
        scope = source["scope"]
        path = ROOT / scope[f"runtime_{kind}_path"]
        if digest(path) != scope[f"runtime_{kind}_sha256"]:
            raise ValueError(f"runtime {kind} changed since calibration")
    for entry in source["sources"].values():
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("finger mesh changed since calibration")
    runtime = ROOT / source["scope"]["runtime_urdf_path"]
    vendor = (ROOT / "isaac/vendor/agx_arm_sim/agx_arm_description/urdf"
              / "piper_l_gripper_d435.urdf")
    if runtime.read_bytes() != vendor.read_bytes():
        raise ValueError("runtime fingers are not the original vendor mounting")
    binding = json.loads(template.read_text())
    updated = {str(path.relative_to(ROOT)): digest(path)
               for path in (SOURCE, SELECTOR, CALIBRATION)}
    for entry in binding["inputs"]["pinned_files"]:
        if entry["path"] in updated:
            entry["sha256"] = updated[entry["path"]]
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"binding input drift: {entry['path']}")
    translation = calibration["aggregate"]["translation_median_m"]
    transform = binding["tcp_contract"]["transform"]
    transform["translation_m"] = translation
    transform["sha256"] = digest(CALIBRATION)
    evidence = next(item for item in binding["evidence"]
                    if item["id"] == "project-calibration-link6-rocobrick-tcp")
    evidence.update(source_sha256=digest(CALIBRATION), translation_m=translation,
                    selector_source_sha256=digest(SOURCE),
                    selector_artifact_sha256=digest(SELECTOR))
    binding["collected_date_utc"] = calibration["collected_date_utc"]
    binding["checks"]["runtime_vendor_urdf_byte_identical"] = True
    binding["checks"].pop("runtime_urdf_distal_pad_correction_applied", None)
    if not all(value is True for value in binding["checks"].values()):
        raise ValueError("binding template contains a failed check")
    BINDING.write_text(json.dumps(binding, indent=2) + "\n")
    text = OVERLAY.read_text()
    root = ET.fromstring(text)
    ns = {"xacro": "http://www.ros.org/wiki/xacro"}
    properties = {prop.attrib["name"]: prop.attrib["value"]
                  for prop in root.findall(".//xacro:property", ns)}
    replacements = {
        "tcp_binding_artifact_sha256": digest(BINDING),
        "tcp_calibration_source_sha256": digest(CALIBRATION),
        "tcp_translation_m": " ".join(repr(float(value)) for value in translation),
    }
    for name, value in replacements.items():
        text = text.replace(properties[name], value)
    OVERLAY.write_text(text)
    print(f"published TCP={translation}, calibration={digest(CALIBRATION)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    publish(parser.parse_args().template)
