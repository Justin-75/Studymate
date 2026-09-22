# src/generation/quiz_grader.py
from __future__ import annotations

"""Deterministic quiz grading.

Contract:
- Quizzes carry a stable `quiz_id`.
- Each question carries a stable `question_id` (also duplicated as `id` for backward compatibility).
- Grader accepts:
    * `quiz_id` + `answers`
    * OR full `quiz` payload + `answers`

Answer formats accepted:
- MCQ: "A"/"B"/"C"/"D" (case-insensitive), "1".."4", or 0..3.
- Short answers: free text.

Because this is deterministic and offline, short-answer grading is approximate:
we check whether the user's answer contains a sufficient fraction of the
expected answer's *content words*.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.generation.quiz_store import load_quiz


# ----------------------------
# Normalization helpers
# ----------------------------


def _normalize_mcq_answer(ans: Any) -> Optional[str]:
    """Normalize user answer into "A"/"B"/"C"/"D"."""
    if ans is None:
        return None

    if isinstance(ans, int):
        if 0 <= ans <= 3:
            return ["A", "B", "C", "D"][ans]
        return None

    s = str(ans).strip()
    if not s:
        return None

    first = s[0].upper()
    if first in {"A", "B", "C", "D"}:
        return first

    # allow "A. ..." or "B) ..."
    m = re.match(r"^\s*([ABCD])\s*[\.)\-:]?\s*", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    if s.isdigit():
        v = int(s)
        if 1 <= v <= 4:
            return ["A", "B", "C", "D"][v - 1]
        if 0 <= v <= 3:
            return ["A", "B", "C", "D"][v]

    return None


def _norm_text(s: Any) -> str:
    t = str(s or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    # keep hyphens as spaces to reduce mismatch ("divide-and-conquer" vs "divide and conquer")
    t = t.replace("-", " ")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Lightweight stopword list for short-answer scoring
_STOP = {
    "a","an","the","and","or","to","of","in","on","for","with","by","as","at","from",
    "is","are","was","were","be","been","being","this","that","these","those","it","its",
    "we","you","they","their","our","your","i","he","she","them","us",
}


def _content_words(s: Any) -> List[str]:
    t = _norm_text(s)
    if not t:
        return []
    toks = re.findall(r"[a-z0-9]{3,}", t)
    return [w for w in toks if w not in _STOP]


def _short_answer_match(expected: str, user: str) -> Tuple[bool, Dict[str, Any]]:
    """Return (correct, debug)."""
    exp_words = set(_content_words(expected))
    usr_words = set(_content_words(user))

    if not exp_words:
        # fall back to substring
        exp = _norm_text(expected)
        usr = _norm_text(user)
        ok = bool(exp) and (exp in usr or usr in exp)
        return ok, {"method": "substring"}

    overlap = len(exp_words & usr_words)
    needed = max(2, int(0.45 * len(exp_words)))
    needed = min(needed, 6)  # avoid over-demanding on long expected sentences

    ok = overlap >= needed
    return ok, {"method": "content_overlap", "overlap": overlap, "needed": needed, "expected_terms": sorted(list(exp_words))[:12]}


def _grade_short_answer(q: Dict[str, Any], user_ans: Any) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Grade short answers.

    Returns (correct|None, debug).
    None means "cannot grade automatically".
    """
    ua = _norm_text(user_ans)
    if not ua:
        return False, {"reason": "empty"}

    acc = q.get("acceptable_answers")
    if isinstance(acc, list) and acc:
        # Accept if any acceptable answer is matched (content-overlap)
        for a in acc:
            ok, dbg = _short_answer_match(str(a or ""), ua)
            if ok:
                return True, {"matched": "acceptable_answers", **dbg}
        return False, {"matched": "acceptable_answers", "reason": "no_match"}

    at = q.get("answer_text")
    if at:
        ok, dbg = _short_answer_match(str(at or ""), ua)
        return ok, {"matched": "answer_text", **dbg}

    return None, {"reason": "no_expected_answer"}


# ----------------------------
# Main grading API
# ----------------------------


def grade_quiz(quiz: Dict[str, Any], answers: Dict[str, Any]) -> Dict[str, Any]:
    """Grade a quiz payload using the provided answers.

    Answers mapping may key by:
      - question_id
      - id (backward compatible)
      - or "1", "2", ... (question position)

    Returns:
      {
        "quiz_id": ...,
        "score": int,
        "total": int,
        "details": [...],
        "feedback": "..."
      }
    """
    qlist = (quiz or {}).get("quiz", [])
    if not isinstance(qlist, list):
        qlist = []

    score = 0
    total_gradable = 0
    details: List[Dict[str, Any]] = []

    for idx, q in enumerate(qlist, start=1):
        qid = str(q.get("question_id") or q.get("id") or "").strip()
        legacy_id = str(q.get("id") or "").strip()
        qtype = str(q.get("type", "mcq") or "mcq").lower()
        page_no = q.get("page_no")

        # Find user answer by preferred keys
        user_raw = None
        if qid and qid in (answers or {}):
            user_raw = (answers or {}).get(qid)
        elif legacy_id and legacy_id in (answers or {}):
            user_raw = (answers or {}).get(legacy_id)
        elif str(idx) in (answers or {}):
            user_raw = (answers or {}).get(str(idx))

        if qtype in ("short", "fill_blank"):
            correct, dbg = _grade_short_answer(q, user_raw)
            graded = correct is not None
            if graded:
                total_gradable += 1
                if bool(correct):
                    score += 1

            details.append(
                {
                    "question_id": qid,
                    "question": q.get("question", ""),
                    "type": qtype,
                    "difficulty": q.get("difficulty", ""),
                    "graded": graded,
                    "correct": correct,
                    "expected": q.get("answer_text") or q.get("answer", ""),
                    "user_answer": user_raw,
                    "page_no": page_no,
                    "debug": dbg,
                }
            )
            continue

        # Default: MCQ
        correct_ans = _normalize_mcq_answer(q.get("answer"))
        user_ans = _normalize_mcq_answer(user_raw)

        graded = correct_ans is not None
        is_correct = (user_ans is not None) and (correct_ans is not None) and (user_ans == correct_ans)

        if graded:
            total_gradable += 1
            if is_correct:
                score += 1

        details.append(
            {
                "question_id": qid,
                "question": q.get("question", ""),
                "options": q.get("options", []),
                "type": "mcq",
                "difficulty": q.get("difficulty", ""),
                "graded": graded,
                "correct": bool(is_correct) if graded else None,
                "correct_answer": correct_ans,
                "user_answer": user_ans,
                "page_no": page_no,
            }
        )

    if total_gradable == 0:
        feedback = "No automatically gradable questions."
    else:
        pct = score / total_gradable
        if pct >= 0.85:
            feedback = "Strong performance. Keep going."
        elif pct >= 0.6:
            feedback = "Decent understanding. Review the incorrect items and retry."
        else:
            feedback = "Needs review. Re-read the cited pages and try again."

    return {
        "quiz_id": (quiz or {}).get("quiz_id"),
        "score": score,
        "total": total_gradable,
        "details": details,
        "feedback": feedback,
    }


def grade_quiz_flexible(
    *,
    answers: Dict[str, Any],
    quiz: Optional[Dict[str, Any]] = None,
    quiz_id: Optional[str] = None,
    quiz_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Grade by quiz payload or quiz_id.

    Preferred usage (tool-level):
      - grade_quiz_flexible(quiz_id="...", answers={...})

    Backward compatible:
      - grade_quiz_flexible(quiz=<full quiz dict>, answers={...})
    """
    if quiz is None:
        if quiz_id:
            quiz = load_quiz(quiz_id, quiz_cache_dir=quiz_cache_dir)
        if quiz is None:
            return {
                "quiz_id": quiz_id,
                "score": 0,
                "total": 0,
                "details": [],
                "feedback": "Quiz not found. Generate the quiz first to create a cached quiz_id.",
            }

    return grade_quiz(quiz, answers)
