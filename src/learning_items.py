# src/learning_items.py
"""High-precision learning item generation (flashcards + quizzes).

This module is designed to prevent **topic drift**.

Non-negotiable contract enforced here:
  - Every downstream artifact must be anchored to the user's query.
  - If the query concept appears in the summary/keypoints, it MUST appear in:
      * important_terms
      * flashcards
      * quizzes
      * knowledge map (handled in src/knowledge_graph.py)

We intentionally generate **fewer** items but of **higher quality**.
No external LLM is required; everything is deterministic.

Inputs expected:
  - summary_bundle from `src.semantic_outline.build_summary`
    * main_idea: str
    * why_it_matters: str
    * key_points: list[str]
    * key_points_cited: list[{text,page_no}]
    * concepts: list[{term,definition,evidence,page_no,...}]
  - page_contexts from `src.semantic_outline.build_page_contexts_from_hits`

Outputs:
  - important_terms: strict, query-anchored concept set
  - flashcards: Type A (definition) and Type B (direct recall)
  - quiz: mixed MCQ + short-answer

"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Stopwords list for term hygiene (sklearn is already in your project via TF-IDF retrieval)
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _EN_STOP  # type: ignore
except Exception:  # pragma: no cover
    _EN_STOP = frozenset()

from src.query_anchor import (
    QueryAnchor,
    best_supporting_span,
    build_query_anchor,
    is_strongly_relevant,
    relevance_score,
    stable_quiz_id,
)


# ----------------------------
# Term hygiene
# ----------------------------

# Words that are "valid English" but useless as study concepts in technical PDFs.
# Expanded to cover the exact drift you observed: "solve", "running", "method" etc.
_GENERIC_BAD_TERMS: Set[str] = {
    # filler / glue
    "way", "ways", "use", "uses", "used", "using", "key", "keys", "term", "terms",
    "simple", "simpler", "simply", "need", "needs", "needed", "want", "wants",
    "make", "makes", "made", "example", "examples", "note", "notes",
    "value", "values", "thing", "things", "case", "cases",
    "consider", "let", "lets", "suppose", "assume",
    "there", "here", "thus", "therefore", "however", "moreover",
    "first", "second", "third", "finally", "also", "often", "sometimes", "usually",
    "we", "our", "you", "your", "they", "their", "it", "its",
    "this", "that", "these", "those",
    "one", "two", "three", "four", "five",

    # generic verbs/nouns that caused drift in your outputs
    "solve", "solves", "solving",
    "run", "runs", "running",
    "method", "methods",
    "algorithm", "algorithms",
    "procedure", "procedures",
    "approach", "approaches",

    # document scaffolding
    "figure", "fig", "table", "section", "chapter", "page",
}

# Allow a few short technical tokens if they are explicitly in the query.
_SHORT_TECH_WHITELIST: Set[str] = {"os", "db", "io", "ai", "cpu", "ram", "nlp", "ml"}

# Variable-like tokens (common in algorithm pseudocode) that should NOT be flashcards by default.
# Examples seen in your output: a0p, l0i, r0i, a1, tj
_VARLIKE_RE = re.compile(r"^([a-zA-Z]\d+|[a-zA-Z]\d+[a-zA-Z]|\w\d+\w|\w\w?\d+)$")

# Bracketed / slice-ish tokens like A[0:p], A[i], A[0..n]
_BRACKET_TOKEN_RE = re.compile(r"[\[\]\(\)\{\}:]")

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _query_tokens(query: str) -> Set[str]:
    """A lightweight tokenization used by other modules (knowledge_graph).

    NOTE: this intentionally keeps behavior stable across the repo.
    """
    q = (query or "").lower()
    toks = set(_WORD_RE.findall(q))
    for w in _SHORT_TECH_WHITELIST:
        if re.search(rf"\b{re.escape(w)}\b", q):
            toks.add(w)
    return toks


def _looks_like_variable(term: str) -> bool:
    t = term.strip()
    if not t:
        return False
    if _BRACKET_TOKEN_RE.search(t):
        return True
    if _VARLIKE_RE.match(t):
        return True
    if len(t) == 1 and t.isalpha():
        return True
    return False


def _is_bad_term(term: str, query: str = "") -> bool:
    """Return True if term should NOT be used as a study concept."""
    t = _norm_space(term)
    if not t:
        return True

    tl = t.lower()

    # If query explicitly contains the term, allow it (user intent override)
    q = (query or "").lower()
    if tl and tl in q:
        return False

    # Stopwords / generic fillers
    if tl in _EN_STOP or tl in _GENERIC_BAD_TERMS:
        return True
    if tl in {"a", "an", "the", "of", "to", "in", "on", "for", "as", "at", "by", "and", "or"}:
        return True

    # Very short is almost always junk
    if len(tl) < 3 and tl not in _SHORT_TECH_WHITELIST:
        return True

    # Variable-like tokens should not become concepts by default
    if _looks_like_variable(t) and tl not in _query_tokens(query):
        return True

    # Punctuation-only
    if re.fullmatch(r"[^A-Za-z0-9]+", t):
        return True

    return False


def _term_in_text(term: str, text: str) -> bool:
    """Boundary-safe English mention check."""
    term = _norm_space(term)
    if not term or not text:
        return False

    words = re.findall(r"[A-Za-z0-9]+", term)
    if not words:
        return False
    if len(words) == 1:
        w = re.escape(words[0])
        return re.search(rf"\b{w}\b", text, flags=re.IGNORECASE) is not None

    sep = r"[\s\-]+"
    pat = r"\b" + sep.join(re.escape(w) for w in words) + r"\b"
    return re.search(pat, text, flags=re.IGNORECASE) is not None


def _extract_clause_after(text: str, marker: str) -> Optional[str]:
    if not text:
        return None
    idx = text.lower().find(marker.lower())
    if idx < 0:
        return None
    out = text[idx + len(marker) :].strip()
    out = re.split(r"[.;!?]\s", out, maxsplit=1)[0].strip()
    return out if len(out) >= 8 else None


def _truncate(s: str, max_chars: int = 240) -> str:
    s = _norm_space(s)
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars].rsplit(" ", 1)[0]
    return cut.strip() + "…"


def _stable_seed(*parts: str) -> int:
    raw = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


# ----------------------------
# Data models
# ----------------------------


@dataclass(frozen=True)
class Concept:
    term: str
    definition: str
    page_no: Optional[int] = None
    evidence: Optional[str] = None
    score: float = 0.0
    covered_keypoints: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        slug = re.sub(r"[^a-z0-9]+", "-", self.term.lower()).strip("-")
        return {
            "id": slug or self.term.lower(),
            "term": self.term,
            "definition": self.definition,
            "page_no": self.page_no,
            "evidence": self.evidence or self.definition,
            "score": float(self.score),
            "covered_keypoints": list(self.covered_keypoints),
        }


@dataclass(frozen=True)
class Flashcard:
    id: str
    type: str  # "A" (definition) or "B" (direct recall)
    front: str
    back: str
    page_no: Optional[int] = None
    concept_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "id": self.id,
            "type": self.type,
            "front": self.front,
            "back": self.back,
        }
        if self.page_no is not None:
            out["page_no"] = int(self.page_no)
        if self.concept_id:
            out["concept_id"] = self.concept_id
        return out


@dataclass(frozen=True)
class QuizQuestion:
    # Stable identifiers
    question_id: str
    id: str

    type: str  # "mcq" | "short"
    question: str
    page_no: Optional[int] = None

    # MCQ
    choices: Optional[List[str]] = None
    answer: Optional[str] = None  # "A"/"B"/"C"/"D"

    # Short answer
    answer_text: Optional[str] = None
    acceptable_answers: Optional[List[str]] = None

    concept_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "question_id": self.question_id,
            "id": self.id,  # backward compatible
            "type": self.type,
            "question": self.question,
        }
        if self.page_no is not None:
            out["page_no"] = int(self.page_no)
        if self.choices is not None:
            out["choices"] = self.choices
        if self.answer is not None:
            out["answer"] = self.answer
        if self.answer_text is not None:
            out["answer_text"] = self.answer_text
        if self.acceptable_answers is not None:
            out["acceptable_answers"] = self.acceptable_answers
        if self.concept_id:
            out["concept_id"] = self.concept_id
        return out


# ----------------------------
# Query-anchored keypoint pool
# ----------------------------


def _summary_text_pool(summary_bundle: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ["main_idea", "why_it_matters"]:
        v = summary_bundle.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    for kp in (summary_bundle.get("key_points", []) or []):
        if isinstance(kp, str) and kp.strip():
            parts.append(kp.strip())
    return "\n".join(parts)


def _keypoints_cited(summary_bundle: Dict[str, Any]) -> List[Tuple[str, Optional[int]]]:
    out: List[Tuple[str, Optional[int]]] = []
    kpc = summary_bundle.get("key_points_cited", []) or []
    if isinstance(kpc, list) and kpc:
        for item in kpc:
            if not isinstance(item, dict):
                continue
            txt = str(item.get("text", "") or "").strip()
            if not txt:
                continue
            p = item.get("page_no", None)
            try:
                page_no = int(p) if p is not None else None
            except Exception:
                page_no = None
            out.append((txt, page_no))
        return out

    # fallback: key_points without page numbers
    for kp in (summary_bundle.get("key_points", []) or []):
        if isinstance(kp, str) and kp.strip():
            out.append((kp.strip(), None))
    return out


def _adaptive_limits(relevant_kp_count: int) -> Tuple[int, int, int]:
    """Return (max_concepts, max_cards, max_questions) based on density."""
    if relevant_kp_count <= 2:
        return 3, 4, 4
    if relevant_kp_count <= 4:
        return 4, 6, 5
    return 6, 8, 6


# ----------------------------
# Key concept selection (query-anchored, keypoint-only)
# ----------------------------


def _mk_anchor_concept(summary_bundle: Dict[str, Any], anchor: QueryAnchor) -> Concept:
    """Create a Concept for the anchor phrase from summary spans (keypoint-only)."""
    spans: List[Tuple[str, Optional[int]]] = []
    mi = str(summary_bundle.get("main_idea", "") or "").strip()
    if mi:
        spans.append((mi, None))
    why = str(summary_bundle.get("why_it_matters", "") or "").strip()
    if why:
        spans.append((why, None))
    spans.extend(_keypoints_cited(summary_bundle))

    best_text, best_page = best_supporting_span(anchor, spans)
    if not best_text:
        best_text = mi or (spans[0][0] if spans else anchor.anchor_phrase)

    # Make it teacher-friendly: if the span is long, truncate.
    definition = _truncate(best_text, 260)

    return Concept(
        term=_norm_space(anchor.anchor_phrase or anchor.raw_query),
        definition=definition,
        page_no=best_page,
        evidence=definition,
        score=100.0,
        covered_keypoints=(),
    )


def _is_definitional(s: str) -> bool:
    sl = (s or "").lower()
    return any(p in sl for p in [" is ", " means ", " refers to ", " defined as ", " we define ", " we call "])


def select_important_concepts(
    summary_bundle: Dict[str, Any],
    query: str,
    max_concepts: int = 6,
    min_concepts: int = 3,
) -> List[Concept]:
    """Select a SMALL set of query-anchored concepts.

    Only concepts supported by:
      - main_idea
      - key_points (and key_points_cited)
      - (optionally) concept definitions that are strongly query-relevant

    Drift prevention:
      - The query anchor concept is always injected and always selected.
      - Generic filler terms are banned.
    """
    query = query or ""
    text_pool = _summary_text_pool(summary_bundle)
    anchor = build_query_anchor(query, text_pool=text_pool)

    # Determine which keypoints are strongly relevant to the query.
    kp_spans = _keypoints_cited(summary_bundle)
    relevant_kps: List[Tuple[str, Optional[int], int]] = []  # (text,page,orig_idx)
    for i, (txt, pno) in enumerate(kp_spans):
        if is_strongly_relevant(txt, anchor):
            relevant_kps.append((txt, pno, i))

    # If none are marked relevant, fall back to all keypoints (still anchor-injected later).
    kp_used = relevant_kps if relevant_kps else [(t, p, i) for i, (t, p) in enumerate(kp_spans)]
    kp_texts = [t for (t, _, _) in kp_used]

    # Always include the anchor as a concept.
    anchor_concept = _mk_anchor_concept(summary_bundle, anchor)

    # Prevent fragment drift: if the anchor is multi-word, do not treat its individual words
    # as separate study concepts (e.g., avoid 'divide'/'conquer' when anchor is 'divide and conquer').
    anchor_words: Set[str] = set(re.findall(r"[A-Za-z0-9]+", (anchor_concept.term or "").lower()))

    # Build candidate concepts.
    raw_concepts = summary_bundle.get("concepts", []) or []
    candidates: List[Concept] = []

    def concept_page(c: Dict[str, Any]) -> Optional[int]:
        p = c.get("page_no", None)
        try:
            return int(p) if p is not None else None
        except Exception:
            return None

    # Helper: pick the best keypoint sentence supporting a term.
    def best_kp_support(term: str) -> Tuple[str, Optional[int], Tuple[int, ...]]:
        matches: List[Tuple[str, Optional[int], int]] = []
        for idx, (kp_txt, kp_pno, orig_idx) in enumerate(kp_used):
            if _term_in_text(term, kp_txt):
                matches.append((kp_txt, kp_pno, idx))

        if not matches:
            return "", None, ()

        # Prefer definitional, then higher query relevance, then shorter.
        best = None
        best_key = (-1_000, -1_000, 1_000)
        for kp_txt, kp_pno, local_i in matches:
            rel = relevance_score(kp_txt, anchor)
            key = (1 if _is_definitional(kp_txt) else 0, rel, -len(kp_txt))
            if key > best_key:
                best_key = key
                best = (kp_txt, kp_pno, local_i)

        assert best is not None
        # covered local indices in kp_used list
        covered_local = tuple(sorted({local_i for _, _, local_i in matches}))
        return best[0], best[1], covered_local

    for c in raw_concepts:
        if not isinstance(c, dict):
            continue
        term = _norm_space(str(c.get("term", "") or c.get("id", "") or ""))
        if not term:
            continue

        # Forcefully skip garbage terms
        # Prevent fragment drift: drop single-word tokens that are just pieces of the anchor phrase
        if len(anchor_words) >= 2 and len(term.split()) == 1 and term.lower() in anchor_words:
            continue

        if _is_bad_term(term, query=query):
            continue

        # Do not allow the anchor to be duplicated via a slightly different concept object.
        if term.lower() == anchor_concept.term.lower():
            continue

        # Prefer a keypoint-supported definition (keypoint-only contract)
        kp_def, kp_page, covered = best_kp_support(term)

        # If no keypoint supports this term, only keep it if its definition is strongly query relevant.
        raw_def = _norm_space(str(c.get("definition", "") or c.get("evidence", "") or ""))
        if kp_def:
            definition = _truncate(kp_def, 260)
            page_no = kp_page
        else:
            if not raw_def:
                continue
            if not is_strongly_relevant(raw_def, anchor):
                continue
            definition = _truncate(raw_def, 260)
            page_no = concept_page(c)

        # Score
        score = 0.0
        score += 2.0 * len(covered)
        score += float(relevance_score(definition, anchor))
        if _is_definitional(definition):
            score += 2.0
        if len(term.split()) >= 2:
            score += 1.5

        # Penalize too-generic singletons
        tl = term.lower()
        if tl in {"algorithm", "method", "procedure", "approach"} and tl not in (anchor.anchor_tokens | anchor.query_tokens):
            score -= 2.0

        candidates.append(
            Concept(
                term=term,
                definition=definition,
                page_no=page_no,
                evidence=_truncate(str(c.get("evidence", "") or definition), 260),
                score=score,
                covered_keypoints=covered,
            )
        )

    # If no candidates, return just the anchor.
    if not candidates:
        return [anchor_concept]

    # Greedy set cover over keypoints (local indices).
    all_kp = set(range(len(kp_texts)))
    uncovered = set(all_kp)

    # Start with anchor concept (always selected)
    selected: List[Concept] = [anchor_concept]

    # Mark keypoints covered by anchor if the anchor phrase appears.
    anchor_covered: Set[int] = set()
    for i, kp in enumerate(kp_texts):
        if anchor.anchor_phrase and _term_in_text(anchor.anchor_phrase, kp):
            anchor_covered.add(i)
        elif is_strongly_relevant(kp, anchor):
            # even if not exact phrase, treat as relevant
            anchor_covered.add(i)
    uncovered -= anchor_covered

    # Deterministic ordering
    cand_sorted = sorted(candidates, key=lambda x: (-x.score, x.term.lower()))

    used_terms: Set[str] = {anchor_concept.term.lower()}

    # Ensure we don't exceed max_concepts
    while len(selected) < max_concepts and cand_sorted:
        best = None
        best_gain = -1
        best_value = -1.0

        for c in cand_sorted:
            if c.term.lower() in used_terms:
                continue
            gain = len(set(c.covered_keypoints) & uncovered)
            value = gain * 10.0 + c.score
            if gain > best_gain or (gain == best_gain and value > best_value):
                best = c
                best_gain = gain
                best_value = value

        if best is None:
            break

        # Stop if we already have minimum concepts and new concept adds no coverage.
        if best_gain <= 0 and len(selected) >= min_concepts:
            break

        selected.append(best)
        used_terms.add(best.term.lower())
        uncovered -= set(best.covered_keypoints)
        cand_sorted = [c for c in cand_sorted if c.term.lower() != best.term.lower()]

        # Early stop if coverage is sufficient
        if all_kp and (len(uncovered) / max(1, len(all_kp))) <= 0.2 and len(selected) >= min_concepts:
            break

    # Always keep anchor first.
    return selected[:max_concepts]


# ----------------------------
# Flashcards (strict A/B types + quality gate)
# ----------------------------


def _mk_card_id(doc_id: str, kind: str, term: str, idx: int) -> str:
    raw = f"{doc_id}|{kind}|{term}|{idx}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def _good_question_stem(q: str) -> bool:
    if not q or len(q) < 10:
        return False
    ql = q.lower()
    # ban vague stems
    banned = [
        "what is way",
        "what is use",
        "what is key",
        "what is terms",
        "what is simple",
        "what is method",
        "what is solve",
        "what is running",
    ]
    if any(b in ql for b in banned):
        return False
    if not re.match(r"^(what|why|how|when|which)\b", ql):
        return False
    # must mention a concrete concept name (at least one long word)
    if len(re.findall(r"[A-Za-z]{4,}", q)) == 0:
        return False
    return True


def _good_answer(a: str) -> bool:
    if not a:
        return False
    a = _norm_space(a)
    if len(a) < 16:
        return False
    # avoid answers that are just one word
    if len(a.split()) <= 1 and len(a) < 24:
        return False
    return True


def _pick_best_evidence_sentence(term: str, page_text: str, anchor: QueryAnchor) -> Optional[str]:
    """Choose a query-relevant evidence sentence for a term."""
    if not page_text:
        return None

    # Rough sentence split
    sents = re.split(r"(?<=[\.\?\!])\s+|\n+", page_text)
    sents = [_norm_space(s) for s in sents if _norm_space(s)]
    if not sents:
        return None

    best = None
    best_score = -1.0

    for s in sents:
        if len(s) < 30:
            continue
        if not _term_in_text(term, s):
            continue
        # Strict query anchoring for evidence
        if not is_strongly_relevant(s, anchor):
            continue

        sl = s.lower()
        score = 0.0
        if any(p in sl for p in [" is ", " means ", " refers to ", " defined as ", " we define ", " we call "]):
            score += 3.0
        if any(p in sl for p in [" used to ", " used for ", " helps ", " allows ", " enables ", " so that", " because"]):
            score += 2.0
        if any(p in sl for p in ["must", "always", "remains true", "invariant", "correct"]):
            score += 2.0
        if len(s) > 260:
            score -= 1.0

        # Add anchor relevance bonus
        score += float(relevance_score(s, anchor)) * 0.2

        if score > best_score:
            best_score = score
            best = s

    return _truncate(best, 240) if best else None


def _make_type_b_recall(term: str, page_text: str, anchor: QueryAnchor) -> Optional[Tuple[str, str]]:
    """Build a Type-B recall Q/A from query-relevant evidence patterns."""
    s = _pick_best_evidence_sentence(term, page_text, anchor) or ""
    if not s:
        return None

    sl = s.lower()

    if any(k in sl for k in ["used to", "helps", "so that", "because", "allows", "enables", "important"]):
        q = f"Why is {term} useful?"
        a = s
        if _good_question_stem(q) and _good_answer(a):
            return q, a

    if any(k in sl for k in ["must", "always", "invariant", "remains true"]):
        q = f"What property must always hold for {term}?"
        a = s
        if _good_question_stem(q) and _good_answer(a):
            return q, a

    clause = _extract_clause_after(s, "is that")
    if clause:
        q = f"What condition is stated for {term}?"
        a = clause
        if _good_question_stem(q) and _good_answer(a):
            return q, a

    return None


def generate_flashcards_high_precision(
    doc_id: str,
    summary_bundle: Dict[str, Any],
    page_contexts: Dict[int, Dict[str, Any]],
    query: str,
    cjk: bool = False,
    max_cards: int = 10,
) -> Dict[str, Any]:
    """Generate strictly query-anchored flashcards."""

    text_pool = _summary_text_pool(summary_bundle)
    anchor = build_query_anchor(query, text_pool=text_pool)

    # Determine density for adaptive output.
    kp_spans = _keypoints_cited(summary_bundle)
    relevant_kp_count = sum(1 for (t, _) in kp_spans if is_strongly_relevant(t, anchor))
    max_concepts, max_cards_adapt, _ = _adaptive_limits(relevant_kp_count)
    max_cards_final = min(max_cards, max_cards_adapt)

    concepts = select_important_concepts(summary_bundle, query=query, max_concepts=max_concepts, min_concepts=min(3, max_concepts))
    important_terms = [c.to_dict() for c in concepts]

    cards: List[Flashcard] = []
    rejected = 0

    # Type A: definition cards (anchor must exist)
    for i, c in enumerate(concepts):
        term = c.term
        # Never drop anchor even if it looks generic
        if i != 0 and _is_bad_term(term, query=query):
            continue
        front = f"什么是{term}？" if cjk else f"What is {term}?"
        back = c.definition or c.evidence or ""
        if not _good_question_stem(front) or not _good_answer(back):
            rejected += 1
            continue
        cards.append(
            Flashcard(
                id=_mk_card_id(doc_id, "A", term, i),
                type="A",
                front=front,
                back=_truncate(back, 220),
                page_no=c.page_no,
                concept_id=c.to_dict().get("id"),
            )
        )

    # Type B: direct recall (query-relevant evidence only)
    # Attempt recall for anchor first, then others.
    for j, c in enumerate(concepts):
        if len(cards) >= max_cards_final:
            break

        page_text = ""
        if c.page_no is not None and c.page_no in page_contexts:
            page_text = str(page_contexts[c.page_no].get("text", "") or "")
        if not page_text:
            # fallback: any page containing term
            for _, v in sorted(page_contexts.items(), key=lambda kv: kv[0]):
                txt = str(v.get("text", "") or "")
                if _term_in_text(c.term, txt):
                    page_text = txt
                    break

        if not page_text:
            rejected += 1
            continue

        qa = _make_type_b_recall(c.term, page_text, anchor)
        if not qa:
            rejected += 1
            continue

        q, a = qa
        if not _good_question_stem(q) or not _good_answer(a):
            rejected += 1
            continue

        cards.append(
            Flashcard(
                id=_mk_card_id(doc_id, "B", c.term, j),
                type="B",
                front=q,
                back=_truncate(a, 240),
                page_no=c.page_no,
                concept_id=c.to_dict().get("id"),
            )
        )

        # keep recall cards low-volume
        if len([x for x in cards if x.type == "B"]) >= 3:
            break

    # Stable ordering: anchor first, then Type A, then Type B
    anchor_cid = concepts[0].to_dict().get("id") if concepts else None

    def _card_key(c: Flashcard):
        is_anchor = 1 if (anchor_cid and c.concept_id == anchor_cid) else 0
        return (-is_anchor, 0 if c.type == "A" else 1, c.front.lower())

    cards_sorted = sorted(cards, key=_card_key)[:max_cards_final]

    # Deduplicate by front
    dedup: List[Flashcard] = []
    seen_front: Set[str] = set()
    for c in cards_sorted:
        f = c.front.strip().lower()
        if f in seen_front:
            continue
        seen_front.add(f)
        dedup.append(c)

    # Contract enforcement: ensure anchor phrase appears in at least one card.
    if anchor.anchor_phrase:
        has_anchor = any(anchor.anchor_phrase.lower() in (cd.front + " " + cd.back).lower() for cd in dedup)
        if not has_anchor and concepts:
            # Force insert anchor definition card at top.
            ac = concepts[0]
            front = f"什么是{ac.term}？" if cjk else f"What is {ac.term}?"
            back = ac.definition or ac.evidence or ac.term
            forced = Flashcard(
                id=_mk_card_id(doc_id, "A", ac.term, 999),
                type="A",
                front=front,
                back=_truncate(back, 220),
                page_no=ac.page_no,
                concept_id=ac.to_dict().get("id"),
            )
            dedup = [forced] + [c for c in dedup if c.front.strip().lower() != forced.front.strip().lower()]
            dedup = dedup[:max_cards_final]

    return {
        "cards": [c.to_dict() for c in dedup],
        "important_terms": important_terms,
        "rejected_count": int(rejected),
        "anchor": {
            "phrase": anchor.anchor_phrase,
            "query_tokens": sorted(anchor.query_tokens),
        },
    }


# ----------------------------
# Quiz (MCQ + Short answer + strict anchoring)
# ----------------------------


def _mk_question_id(quiz_id: str, idx: int, qtype: str, concept_id: str, question: str) -> str:
    raw = f"{quiz_id}|{idx}|{qtype}|{concept_id}|{question}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _make_mcq_from_definition(
    quiz_id: str,
    idx: int,
    term: str,
    correct_def: str,
    distractor_defs: List[str],
    page_no: Optional[int],
    concept_id: Optional[str],
) -> Optional[QuizQuestion]:
    """MCQ template: definition discrimination."""
    correct_def = _truncate(correct_def, 210)
    dist = [_truncate(d, 210) for d in distractor_defs if d and d.strip()]
    dist = [d for d in dist if not _term_in_text(term, d)]
    if len(dist) < 3:
        return None

    dist = dist[:3]
    choices_raw = [correct_def] + dist

    rng = random.Random(_stable_seed(quiz_id, term, str(idx)))
    rng.shuffle(choices_raw)

    labels = ["A", "B", "C", "D"]
    choices = [f"{labels[i]}. {choices_raw[i]}" for i in range(4)]
    correct_idx = choices_raw.index(correct_def)
    answer = labels[correct_idx]

    question = f"Which statement best describes {term}?"
    if len(question) < 12 or not _good_answer(correct_def):
        return None

    qid = _mk_question_id(quiz_id, idx, "mcq", concept_id or "", question)
    return QuizQuestion(
        question_id=qid,
        id=qid,
        type="mcq",
        question=question,
        choices=choices,
        answer=answer,
        page_no=page_no,
        concept_id=concept_id,
    )


def _make_short_answer_definition(
    quiz_id: str,
    idx: int,
    term: str,
    definition: str,
    page_no: Optional[int],
    concept_id: Optional[str],
) -> Optional[QuizQuestion]:
    question = f"In 1–2 sentences, define {term}."
    if not _good_question_stem("What " + term + "?"):
        # reuse gate: ensure term is concrete
        if len(re.findall(r"[A-Za-z]{4,}", term)) == 0:
            return None
    if not _good_answer(definition):
        return None

    acc = [definition]
    clause = _extract_clause_after(definition, "means")
    if clause:
        acc.append(clause)

    qid = _mk_question_id(quiz_id, idx, "short", concept_id or "", question)
    return QuizQuestion(
        question_id=qid,
        id=qid,
        type="short",
        question=question,
        answer_text=_truncate(definition, 260),
        acceptable_answers=[_truncate(a, 260) for a in acc if a],
        page_no=page_no,
        concept_id=concept_id,
    )


def _make_short_answer_why(
    quiz_id: str,
    idx: int,
    term: str,
    page_text: str,
    anchor: QueryAnchor,
    page_no: Optional[int],
    concept_id: Optional[str],
) -> Optional[QuizQuestion]:
    sent = _pick_best_evidence_sentence(term, page_text, anchor) or ""
    if not sent:
        return None

    if not any(k in sent.lower() for k in ["because", "so that", "used to", "helps", "allows", "enables", "advantage", "benefit", "important"]):
        return None

    question = f"According to the text, why is {term} useful or important?"
    if not _good_question_stem(question):
        return None
    if not _good_answer(sent):
        return None

    acc = [sent]
    clause = _extract_clause_after(sent, "because")
    if clause:
        acc.append(clause)

    qid = _mk_question_id(quiz_id, idx, "short", concept_id or "", question)
    return QuizQuestion(
        question_id=qid,
        id=qid,
        type="short",
        question=question,
        answer_text=_truncate(sent, 260),
        acceptable_answers=[_truncate(a, 260) for a in acc if a],
        page_no=page_no,
        concept_id=concept_id,
    )


def generate_quiz_high_precision(
    doc_id: str,
    summary_bundle: Dict[str, Any],
    page_contexts: Dict[int, Dict[str, Any]],
    query: str,
    cjk: bool = False,
    max_questions: int = 6,
) -> Dict[str, Any]:
    """Generate a teacher-like quiz anchored to the query."""

    text_pool = _summary_text_pool(summary_bundle)
    anchor = build_query_anchor(query, text_pool=text_pool)

    # Density => adaptive volume
    kp_spans = _keypoints_cited(summary_bundle)
    relevant_kp_count = sum(1 for (t, _) in kp_spans if is_strongly_relevant(t, anchor))
    max_concepts, _, max_q_adapt = _adaptive_limits(relevant_kp_count)
    max_q_final = min(max_questions, max_q_adapt)

    # Concepts (anchor included as first)
    concepts = select_important_concepts(summary_bundle, query=query, max_concepts=max_concepts, min_concepts=min(3, max_concepts))
    important_terms = [c.to_dict() for c in concepts]

    # Build quiz_id (stable)
    quiz_id = stable_quiz_id(doc_id, query, version="v1")

    # Build distractor definitions from other concepts (still query-related due to selection)
    defs: List[Tuple[str, str, Optional[int], str]] = []
    for c in concepts:
        cid = c.to_dict().get("id")
        defs.append((c.term, c.definition or c.evidence or "", c.page_no, cid))

    quiz: List[QuizQuestion] = []
    rejected = 0
    idx = 1

    # 1) MCQ on anchor definition (mandatory)
    if defs:
        term, dfn, page_no, cid = defs[0]
        distractors = [d for (t, d, _, _) in defs[1:] if d]
        # If we don't have enough distractors, borrow from summary_bundle concepts but require anchoring.
        if len(distractors) < 3:
            for c in (summary_bundle.get("concepts", []) or []):
                if not isinstance(c, dict):
                    continue
                dd = _norm_space(str(c.get("definition", "") or c.get("evidence", "") or ""))
                tt = _norm_space(str(c.get("term", "") or ""))
                if not dd or not tt:
                    continue
                if _is_bad_term(tt, query=query):
                    continue
                if tt.lower() == term.lower():
                    continue
                if not is_strongly_relevant(dd, anchor):
                    continue
                distractors.append(dd)
                if len(distractors) >= 3:
                    break

        q = _make_mcq_from_definition(quiz_id, idx, term, dfn, distractors, page_no, cid)
        idx += 1
        if q:
            quiz.append(q)
        else:
            rejected += 1

    # 2) MCQ on a second concept (if available)
    if len(defs) >= 2 and len(quiz) < max_q_final:
        term, dfn, page_no, cid = defs[1]
        distractors = [d for (t, d, _, _) in defs if t != term and d]
        q = _make_mcq_from_definition(quiz_id, idx, term, dfn, distractors, page_no, cid)
        idx += 1
        if q:
            quiz.append(q)
        else:
            rejected += 1

    # 3) Short-answer definition of anchor (mandatory if we still have budget)
    if defs and len(quiz) < max_q_final:
        term, dfn, page_no, cid = defs[0]
        q = _make_short_answer_definition(quiz_id, idx, term, dfn, page_no, cid)
        idx += 1
        if q:
            quiz.append(q)
        else:
            rejected += 1

    # 4) Short-answer "why/useful" (from query-relevant evidence)
    if len(quiz) < max_q_final:
        # Prefer anchor, else other concepts.
        for (term, _, page_no, cid) in defs[:3]:
            if page_no is None or page_no not in page_contexts:
                continue
            page_text = str(page_contexts[page_no].get("text", "") or "")
            if not page_text:
                continue
            q = _make_short_answer_why(quiz_id, idx, term, page_text, anchor, page_no, cid)
            idx += 1
            if q:
                quiz.append(q)
                break
            rejected += 1

    # Final: ensure at least one question mentions the anchor phrase explicitly.
    if anchor.anchor_phrase:
        if not any(anchor.anchor_phrase.lower() in (qq.question or "").lower() for qq in quiz):
            # Force a simple short-answer definition question on the anchor.
            if defs:
                term, dfn, page_no, cid = defs[0]
                q = _make_short_answer_definition(quiz_id, idx, term, dfn, page_no, cid)
                if q:
                    quiz.insert(0, q)
                    quiz = quiz[:max_q_final]

    # Stable ordering: MCQ first then short
    quiz_sorted = sorted(quiz, key=lambda x: (0 if x.type == "mcq" else 1, x.question_id))
    quiz_sorted = quiz_sorted[:max_q_final]

    return {
        "quiz_id": quiz_id,
        "doc_id": doc_id,
        "query": query,
        "quiz": [q.to_dict() for q in quiz_sorted],
        "important_terms": important_terms,
        "rejected_count": int(rejected),
        "anchor": {
            "phrase": anchor.anchor_phrase,
            "query_tokens": sorted(anchor.query_tokens),
        },
    }
