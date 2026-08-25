"""Persistent reporting utilities for batch GT-expert validation."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Literal

ValidationStatus = Literal["pending", "running", "passed", "failed", "timeout"]


@dataclass
class TaskValidationResult:
    """Current validation state for one saved symbolic task."""

    task: str
    family: str
    status: ValidationStatus = "pending"
    duration_seconds: float = 0.0
    failure: str = ""
    log_name: str = ""
    initial_yaw_degrees: float | None = None


def discover_tasks(
    tasks_root: Path, families: tuple[str, ...] = ()
) -> list[Path]:
    """Return naturally sorted task directories containing both structures.

    Returns:
        Saved Type-1 task directories.
    """
    selected = set(families)
    tasks = []
    for family_dir in tasks_root.iterdir():
        if not family_dir.is_dir() or (selected and family_dir.name not in selected):
            continue
        for task_dir in family_dir.iterdir():
            if not task_dir.is_dir():
                continue
            required = ("structure_start.json", "structure_goal.json")
            if all((task_dir / name).is_file() for name in required):
                tasks.append(task_dir)
    return sorted(
        tasks,
        key=lambda path: (
            _natural_key(path.parent.name),
            _natural_key(path.name),
        ),
    )


def failure_from_log(log_text: str, return_code: int) -> str:
    """Extract a concise failure reason from one complete simulator log.

    Returns:
        Human-readable failure summary.
    """
    expert = re.findall(r"failure_reason=(?:'([^']*)'|\"([^\"]*)\")", log_text)
    if expert:
        return next((item for item in expert[-1] if item), "expert failed")
    safe_start = re.findall(r"SafeStartError:\s*([^\r\n]+)", log_text)
    if safe_start:
        return safe_start[-1].strip()
    runtime = re.findall(r"RuntimeError:\s*([^\r\n]+)", log_text)
    if runtime:
        return runtime[-1].strip()
    return f"process exited with code {return_code}"


def deterministic_validation_yaws(
    task: str, count: int, seed: int
) -> tuple[float, ...]:
    """Return task-stable continuous yaw samples in ``[-180, 180)``.

    Returns:
        Reproducible yaw sequence independent of task iteration order.
    """
    if count <= 0:
        raise ValueError("yaw sample count must be positive")
    digest = hashlib.sha256(f"{seed}:{task}".encode()).digest()
    task_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = Random(task_seed)
    return tuple(rng.uniform(-180.0, 180.0) for _ in range(count))


def write_report(
    output_dir: Path,
    tasks_root: Path,
    results: list[TaskValidationResult],
    running: bool,
) -> Path:
    """Atomically write HTML and JSON validation reports.

    Returns:
        Absolute HTML report path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = sum(item.status in {"passed", "failed", "timeout"} for item in results)
    passed = sum(item.status == "passed" for item in results)
    failed = sum(item.status in {"failed", "timeout"} for item in results)
    total = len(results)
    percent = 100.0 if total == 0 else completed * 100.0 / total
    rows = "\n".join(_result_row(item) for item in results)
    refresh = '<meta http-equiv="refresh" content="3">' if running else ""
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
{refresh}
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GT 专家策略验证报告</title>
<style>
body {{
  font-family: system-ui, sans-serif; margin: 2rem;
  color: #172033; background: #f5f7fb;
}}
.panel {{
  max-width: 1100px; margin: auto; background: white; padding: 1.5rem;
  border-radius: 12px; box-shadow: 0 4px 20px #16213e18;
}}
.bar {{ height: 20px; background: #e5e9f2; border-radius: 10px; overflow: hidden; }}
.fill {{
  height: 100%; width: {percent:.2f}%;
  background: #3478f6; transition: width .3s;
}}
.summary {{ display: flex; gap: 1.5rem; margin: 1rem 0; flex-wrap: wrap; }}
.passed {{ color: #147a3d; font-weight: 700; }}
.failed, .timeout {{ color: #c23434; font-weight: 700; }}
.running {{ color: #8a5a00; font-weight: 700; }} .pending {{ color: #697386; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 1.25rem; }}
th, td {{
  padding: .7rem; text-align: left;
  border-bottom: 1px solid #e6e9ef; vertical-align: top;
}}
th {{ background: #f1f4f9; }} code {{ white-space: pre-wrap; }}
</style>
</head>
<body><main class="panel">
<h1>GT 专家策略验证报告</h1>
<p>任务目录：<code>{html.escape(str(tasks_root.resolve()))}</code></p>
<div class="bar"><div class="fill"></div></div>
<div class="summary">
<span>进度 {completed}/{total}（{percent:.1f}%）</span>
<span class="passed">通过 {passed}</span>
<span class="failed">失败 {failed}</span>
</div>
<table><thead><tr>
<th>任务</th><th>类别</th><th>初始 yaw</th><th>状态</th>
<th>耗时</th><th>结果/错误</th><th>日志</th>
</tr></thead>
<tbody>{rows}</tbody></table>
</main></body></html>
"""
    report = output_dir / "report.html"
    temporary = output_dir / ".report.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(report)
    payload = {
        "tasks_root": str(tasks_root.resolve()),
        "running": running,
        "completed": completed,
        "total": total,
        "passed": passed,
        "failed": failed,
        "results": [asdict(item) for item in results],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report.resolve()


def progress_bar(completed: int, total: int, width: int = 28) -> str:
    """Return a fixed-width textual progress bar.

    Returns:
        Bar and numeric completion fraction.
    """
    ratio = 1.0 if total == 0 else min(1.0, max(0.0, completed / total))
    filled = round(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total}"


def _result_row(result: TaskValidationResult) -> str:
    status_text = {
        "pending": "等待",
        "running": "运行中",
        "passed": "PASS",
        "failed": "FAIL",
        "timeout": "TIMEOUT",
    }[result.status]
    log = (
        f'<a href="{html.escape(result.log_name)}">查看</a>'
        if result.log_name
        else "—"
    )
    yaw = (
        f"{result.initial_yaw_degrees:.3f}°"
        if result.initial_yaw_degrees is not None
        else "arranger 默认"
    )
    return (
        "<tr>"
        f"<td><code>{html.escape(result.task)}</code></td>"
        f"<td>{html.escape(result.family)}</td>"
        f"<td>{yaw}</td>"
        f'<td class="{result.status}">{status_text}</td>'
        f"<td>{result.duration_seconds:.1f} s</td>"
        f"<td>{html.escape(result.failure) if result.failure else '—'}</td>"
        f"<td>{log}</td>"
        "</tr>"
    )


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", value)
    )
