"""Preflight entry point; no simulator is started before validation succeeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rocobrick.contracts import ContractError, load_and_validate_inputs
from rocobrick.evidence import EvidenceRecorder, FinalResult


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an assembly-plan/v2 and scene binding")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    recorder = EvidenceRecorder(args.evidence_root)
    try:
        recorder.capture_input("plan.json", args.plan)
        recorder.capture_input("scene.json", args.scene)
        plan, scene = load_and_validate_inputs(args.plan, args.scene)
        recorder.write_json("preflight.json", {"valid": True, "plan_id": plan.plan_id, "scene_id": scene.scene_id})
        recorder.finalize(FinalResult(True, False, False, True, False, True, "preflight only; no motion was requested"))
        print(f"valid plan={plan.plan_id} scene={scene.scene_id} evidence={recorder.directory}")
        return 0
    except (ContractError, OSError, ValueError) as error:
        recorder.write_json("preflight.json", {"valid": False, "reason": str(error)})
        recorder.finalize(FinalResult(False, False, False, True, False, True, str(error)))
        print(f"rejected: {error}; evidence={recorder.directory}", file=sys.stderr)
        return 2
