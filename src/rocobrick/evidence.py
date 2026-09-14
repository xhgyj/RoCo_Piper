"""Append-only per-run evidence, including unsuccessful starts and shutdowns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping
from uuid import uuid4


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


@dataclass(frozen=True)
class FinalResult:
    input_valid: bool
    tasks_complete: bool
    structure_valid: bool
    execution_safe: bool
    retreat_clear: bool
    evidence_complete: bool
    reason: str | None = None

    @property
    def passed(self) -> bool:
        return all((
            self.input_valid,
            self.tasks_complete,
            self.structure_valid,
            self.execution_safe,
            self.retreat_clear,
            self.evidence_complete,
        ))


class EvidenceRecorder:
    def __init__(self, root: Path, run_id: str | None = None) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_id = run_id or str(uuid4())
        self.directory = root / f"{timestamp}_{self.run_id}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._events_path = self.directory / "events.jsonl"

    def capture_input(self, name: str, source: Path) -> None:
        data = source.read_bytes()
        (self.directory / name).write_bytes(data)
        self.write_json(f"{name}.metadata.json", {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})

    def capture_provenance(self, repository: Path, extra: Mapping[str, object] | None = None) -> None:
        def git(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=repository, check=False, text=True, capture_output=True).stdout.strip()
        payload: dict[str, object] = {
            "commit": git("rev-parse", "HEAD"),
            "status": git("status", "--short", "--branch"),
        }
        payload.update(extra or {})
        self.write_json("provenance.json", payload)

    def append_event(self, event: object) -> None:
        record = {"recorded_at": datetime.now(UTC).isoformat(), "run_id": self.run_id, "event": event}
        with self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")
            stream.flush()

    def write_json(self, name: str, value: object) -> None:
        destination = self.directory / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def finalize(self, result: FinalResult) -> None:
        self.write_json("result.json", {**asdict(result), "passed": result.passed})
