# src/graphs/store.py
"""Shared handles for both graphs: the SQLite repository and the section-summary cache."""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

from src.db.repository import Repository

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = os.getenv("STUDYMATE_DB", str(PROJECT_ROOT / "Data" / "Database" / "app.db"))
SUMMARY_DIR = PROJECT_ROOT / "Data" / "Cache" / "summaries"


@lru_cache(maxsize=1)
def repo() -> Repository:
    return Repository(DB_PATH)


def _summary_path(doc_id: str) -> Path:
    return SUMMARY_DIR / f"{doc_id}.json"


def has_doc_summary(doc_id: str) -> bool:
    return _summary_path(doc_id).exists()


def load_doc_summary(doc_id: str) -> Optional[dict]:
    path = _summary_path(doc_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_doc_summary(doc_id: str, data: dict) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = _summary_path(doc_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)          # atomic, so a reader never sees a half-written file
    return path


# A marker file while the ingest graph is summarizing, so the chat can say "still building"
# instead of starting a second, duplicate summary run. It holds the progress as JSON
# ({"stage": "summarizing", "done": 12, "total": 40}) for the web UI. The running job touches the
# marker every minute; one that has not been touched for 5 minutes was left behind by a crash.
HEARTBEAT_SECONDS = 60
STALE_SECONDS = 5 * 60


def _marker(doc_id: str) -> Path:
    return SUMMARY_DIR / f"{doc_id}.building"


@contextmanager
def building(doc_id: str) -> Iterator[None]:
    """Keep the marker fresh while the block runs (a big book can take well over 30 minutes)."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    marker, stop = _marker(doc_id), threading.Event()
    marker.write_text(json.dumps({"stage": "planning"}), encoding="utf-8")

    def heartbeat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            os.utime(marker)       # fresh mtime, progress left as it is

    beat = threading.Thread(target=heartbeat, daemon=True)
    beat.start()
    try:
        yield
    finally:
        stop.set()
        beat.join()
        marker.unlink(missing_ok=True)


def report_progress(doc_id: str, stage: str, done: int = 0, total: int = 0) -> None:
    """Called by the ingest graph as it goes; read back by building_progress()."""
    marker = _marker(doc_id)
    if marker.exists():
        marker.write_text(json.dumps({"stage": stage, "done": done, "total": total}), encoding="utf-8")


def is_building(doc_id: str) -> bool:
    m = _marker(doc_id)
    return m.exists() and time.time() - m.stat().st_mtime < STALE_SECONDS


def building_progress(doc_id: str) -> Optional[dict]:
    """The progress of a running summary build, {} if it has not reported yet, None if none runs."""
    if not is_building(doc_id):
        return None
    try:
        progress = json.loads(_marker(doc_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):            # gone or mid-write
        return {}
    return progress if isinstance(progress, dict) else {}
