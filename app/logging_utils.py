"""Pipeline trace logging.

Every query run through the pipeline produces a structured trace record
(input -> retrieved chunks -> LLM response -> translated text -> audio path)
so the whole pipeline can be inspected/debugged after the fact, per the
assignment's "log the full pipeline trace" requirement. Traces are appended
as JSON Lines to logs/pipeline_trace.jsonl and also written individually
under logs/traces/<trace_id>.json for easy single-run inspection.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

# On Windows, sys.stdout/stderr often default to the legacy console codepage
# (cp1252), which cannot encode a lot of what this app prints: Amharic (and
# other non-Latin) script from translation, or even stray Unicode punctuation
# (e.g. U+2011 non-breaking hyphen) an LLM may produce in English. Force
# UTF-8 with a safe fallback so logging/printing never crashes the pipeline
# over an encoding issue. Not all streams support reconfigure (e.g. some
# test/capture harnesses), so this is best-effort.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

logger = logging.getLogger("captain_assistant")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _json_default(obj: Any):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class PipelineTrace:
    """Accumulates the stages of a single pipeline run and persists them."""

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or new_trace_id()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.stages: list[dict] = []

    def record(self, stage: str, **data: Any) -> None:
        entry = {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        self.stages.append(entry)
        logger.info("[%s] %s: %s", self.trace_id, stage, _summarize(data))

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "stages": self.stages,
        }

    def persist(self) -> Path:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = settings.logs_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        record = self.to_dict()

        # Individual file, easy to open for one run.
        single_path = traces_dir / f"{self.trace_id}.json"
        single_path.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")

        # Append-only JSONL log of every run, for tailing/aggregating.
        jsonl_path = settings.logs_dir / "pipeline_trace.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

        return single_path


def _summarize(data: dict, max_len: int = 160) -> str:
    parts = []
    for k, v in data.items():
        s = str(v)
        if len(s) > max_len:
            s = s[:max_len] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts)
