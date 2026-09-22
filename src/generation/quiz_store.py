# src/generation/quiz_store.py
from __future__ import annotations

"""Quiz persistence (by quiz_id).

Requirements:
- Every generated quiz must have a stable quiz_id.
- The grader must be able to load a quiz by quiz_id.

Implementation notes:
- One JSON file per quiz_id.
- Default storage is inside the repository at: Data/Cache/quizzes
- Environment override supported via QUIZ_CACHE_DIR.
  * If QUIZ_CACHE_DIR is relative, it is resolved relative to the repository root.

This module is deliberately dependency-free and deterministic.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    # quiz_store.py -> generation -> src -> <repo root>
    return Path(__file__).resolve().parents[2]


def get_quiz_cache_dir() -> str:
    """Return quiz cache dir.

    If QUIZ_CACHE_DIR is unset, uses <repo_root>/Data/Cache/quizzes.
    """
    raw = os.getenv("QUIZ_CACHE_DIR")
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (_repo_root() / p).resolve()
        return str(p)

    return str((_repo_root() / "Data/Cache/quizzes").resolve())


def quiz_path(quiz_id: str, quiz_cache_dir: Optional[str] = None) -> Path:
    quiz_cache_dir = quiz_cache_dir or get_quiz_cache_dir()
    return Path(quiz_cache_dir) / f"{quiz_id}.json"


def save_quiz(quiz_payload: Dict[str, Any], quiz_cache_dir: Optional[str] = None) -> str:
    """Persist a quiz payload by quiz_id.

    Returns the written file path.
    """
    quiz_id = str((quiz_payload or {}).get("quiz_id") or "").strip()
    if not quiz_id:
        raise ValueError("quiz_payload must contain a non-empty 'quiz_id'")

    fp = quiz_path(quiz_id, quiz_cache_dir)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(quiz_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(fp)


def load_quiz(quiz_id: str, quiz_cache_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load quiz JSON by quiz_id. Returns None if not found."""
    quiz_id = str(quiz_id or "").strip()
    if not quiz_id:
        return None
    fp = quiz_path(quiz_id, quiz_cache_dir)
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))
