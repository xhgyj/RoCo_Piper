#!/usr/bin/env python3
"""Validate the GT expert headlessly and write a live HTML report."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rocobrick.policy.expert_validation import (
    TaskValidationResult,
    discover_tasks,
    failure_from_log,
    progress_bar,
    write_report,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_SCRIPT = REPOSITORY_ROOT / "run/demo_symbolic_assembly.py"
KNOWN_FAMILIES = ("basic", "adjacent", "multilevel", "dense", "bridge")


def parse_args() -> argparse.Namespace:
    """Parse batch-validation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=Path("tasks/type1"))
    parser.add_argument(
        "--family",
        action="append",
        choices=KNOWN_FAMILIES,
        default=[],
        help="validate only this family; may be repeated",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip tasks previously passed with a retained log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report directory (default: validation_reports/<timestamp>)",
    )
    return parser.parse_args()


def main() -> int:
    """Run each task in an isolated headless Isaac process."""
    args = parse_args()
    tasks_root = args.tasks_root.resolve()
    if not tasks_root.is_dir():
        raise FileNotFoundError(tasks_root)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    tasks = discover_tasks(tasks_root, tuple(args.family))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise RuntimeError("no task directories found")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else REPOSITORY_ROOT / "validation_reports" / timestamp
    )
    logs_dir = output_dir / "task_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results = [
        TaskValidationResult(
            task=str(path.relative_to(tasks_root)),
            family=path.parent.name,
            log_name=f"task_logs/{path.parent.name}_{path.name}.log",
        )
        for path in tasks
    ]
    if args.resume:
        _restore_passed_results(output_dir, results)
    report = write_report(output_dir, tasks_root, results, running=True)
    print(f"HTML report: {report}", flush=True)

    bricksim = shutil.which("bricksim")
    if bricksim is None:
        raise RuntimeError("bricksim executable not found; run with uv run python")

    for index, (task_dir, result) in enumerate(zip(tasks, results), start=1):
        if result.status == "passed":
            completed = sum(item.status == "passed" for item in results[:index])
            _show_progress(completed, len(tasks), result.task, result.duration_seconds)
            continue
        result.status = "running"
        write_report(output_dir, tasks_root, results, running=True)
        started = time.monotonic()
        _show_progress(index - 1, len(tasks), result.task, 0.0)
        command = [
            bricksim,
            "--/app/window/enabled=false",
            "--/app/livestream/enabled=false",
            str(DEMO_SCRIPT),
            "--task-dir",
            str(task_dir),
            "--inspect-seconds",
            "0",
            "--final-hold-seconds",
            "0",
        ]
        log_path = output_dir / result.log_name
        temporary_log_path = output_dir / (
            f".{task_dir.parent.name}_{task_dir.name}.running.log"
        )
        timed_out = False
        with temporary_log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    _show_progress(index - 1, len(tasks), result.task, elapsed)
                    if elapsed >= args.timeout_seconds:
                        timed_out = True
                        _stop_process(process)
                        break
                    time.sleep(1.0)
            except BaseException:
                _stop_process(process)
                logs_dir.mkdir(parents=True, exist_ok=True)
                temporary_log_path.replace(log_path)
                result.status = "failed"
                result.failure = "validation interrupted"
                result.duration_seconds = time.monotonic() - started
                write_report(output_dir, tasks_root, results, running=False)
                raise

        logs_dir.mkdir(parents=True, exist_ok=True)
        temporary_log_path.replace(log_path)
        result.duration_seconds = time.monotonic() - started
        if timed_out:
            result.status = "timeout"
            result.failure = f"exceeded {args.timeout_seconds:.0f} s"
        elif process.returncode == 0:
            result.status = "passed"
        else:
            result.status = "failed"
            result.failure = failure_from_log(
                log_path.read_text(encoding="utf-8", errors="replace"),
                int(process.returncode or 1),
            )
        write_report(output_dir, tasks_root, results, running=True)
        _show_progress(index, len(tasks), result.task, result.duration_seconds)

    report = write_report(output_dir, tasks_root, results, running=False)
    passed = sum(item.status == "passed" for item in results)
    failed = len(results) - passed
    print()
    print(f"Completed: passed={passed} failed={failed}")
    print(f"HTML report: {report}")
    return 0 if failed == 0 else 1


def _show_progress(completed: int, total: int, task: str, elapsed: float) -> None:
    text = f"\r{progress_bar(completed, total)} {task} {elapsed:6.1f}s"
    print(text.ljust(90), end="", flush=True)


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate one exact simulator process, escalating after a grace period."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _restore_passed_results(
    output_dir: Path, results: list[TaskValidationResult]
) -> None:
    """Restore only completed passes whose detailed log is still available."""
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    previous = {
        item["task"]: item
        for item in payload.get("results", [])
        if item.get("status") == "passed"
    }
    for result in results:
        saved = previous.get(result.task)
        if saved is None:
            continue
        saved_log = output_dir / saved.get("log_name", "")
        if not saved_log.is_file():
            continue
        result.status = "passed"
        result.duration_seconds = float(saved.get("duration_seconds", 0.0))
        result.log_name = str(saved_log.relative_to(output_dir))


if __name__ == "__main__":
    sys.exit(main())
