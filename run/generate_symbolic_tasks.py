#!/usr/bin/env python3
"""Generate validated one-step symbolic assembly tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from rocobrick.task_config.symbolic_assembly import (
    GenerationConfig,
    generate_symbolic_tasks,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tasks/type1"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count-per-family", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Generate a symbolic task corpus and print its coverage summary."""
    args = parse_args()
    report = generate_symbolic_tasks(
        GenerationConfig(
            output_dir=args.output,
            seed=args.seed,
            count_per_family=args.count_per_family,
        )
    )
    print(f"generated {report.total_tasks} tasks under {report.output_dir}")
    print(f"families: {report.family_counts}")


if __name__ == "__main__":
    main()
