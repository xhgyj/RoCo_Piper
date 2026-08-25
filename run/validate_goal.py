#!/usr/bin/env python3
"""Validate one goal structure against RoCoBrick and BrickSim constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocobrick.task_config.symbolic_assembly import validate_goal_structure


def parse_args() -> argparse.Namespace:
    """Parse the goal path.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goal", type=Path, help="Path to structure_goal.json")
    return parser.parse_args()


def main() -> None:
    """Validate the requested goal and return a shell-friendly status."""
    args = parse_args()
    try:
        goal = json.loads(args.goal.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"INVALID {args.goal}: {exc}") from exc
    report = validate_goal_structure(goal)
    if not report.valid:
        print(f"INVALID {args.goal}")
        for error in report.errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(
        f"VALID {args.goal}: bricks={report.brick_count} "
        f"connections={report.connection_count}"
    )


if __name__ == "__main__":
    main()
