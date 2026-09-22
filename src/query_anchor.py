# src/query_anchor.py
"""Query anchoring utilities (deterministic).

The repository generates multiple artifacts from PDFs (summary, flashcards, quizzes, knowledge maps).
A common failure mode is **topic drift**: downstream artifacts pick generic surface tokens instead of
staying anchored to the user's query.

This module implements a single, deterministic query-anchoring contract:
- Choose a stable *anchor phrase* that represents the user's intent.
- Provide robust phrase matching (space/hyphen tolerant).
- Provide relevance scoring used to filter evidence and prevent drift.

Important design note:
- We **preserve internal stopwords** inside the anchor phrase (e.g., "divide and conquer"),
  because removing them breaks phrase matching.
- We still compute content-token sets for scoring (stopwords removed).

No external NLP models are required.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple


# Conservative token regex: words with optional internal hyphens/apostrophes
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


# Stopwords: small built-in list + optional sklearn stopwords if present.
_BASE_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else",
    "to", "of", "in", "on", "for", "with", "by", "as", "at", "from", "into", "over", "under",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "they", "their", "we", "our", "you", "your",
}

_QUERY_VERBS: Set[str] = {
    "explain", "describe", "define", "give", "tell", "show", "summarize", "outline",
    "compare", "contrast", "discuss", "analyze", "derive", "prove", "compute", "calculate",
}

_QUERY_WH_WORDS: Set[str] = {"what", "why", "how", "when", "where", "who", "which"}

# Words that often appear in queries but are rarely the target concept by themselves.
_GENERIC_QUERY_FILLERS: Set[str] = {
    "concept", "idea", "topic", "approach", "algorithm", "method", "procedure", "technique",
    "overview", "introduction", "basics", "definition",
}

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SK_STOP  # type: ignore

    _BASE_STOPWORDS |= set(_SK_STOP)
except Exception:  # pragma: no cover
    pass


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_for_match(s: str) -> str:
    """Lowercase, remove most punctuation (keeping hyphens as spaces), collapse whitespace."""
    s = (s or "").strip().lower()
    # convert hyphens/underscores to spaces for matching
    s = re.sub(r"[-_]+", " ", s)
    # remove other punctuation
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return _norm_space(s)


def tokenize(query_or_phrase: str) -> List[str]:
    """Tokenize into lowercase tokens (keeps alphanumerics and internal hyphens)."""
    q = (query_or_phrase or "").strip().lower()
    return [t.lower() for t in _WORD_RE.findall(q)]


def content_tokens(query: str, *, keep_short_whitelist: Optional[Set[str]] = None) -> List[str]:
    """Return content tokens for scoring (removes stopwords + wh/query verbs)."""
    keep_short_whitelist = keep_short_whitelist or {"ai", "ml", "nlp", "io", "os", "db", "cpu", "ram"}
    toks = tokenize(query)
    out: List[str] = []
    for t in toks:
        if t in keep_short_whitelist:
            out.append(t)
            continue
        if t in _BASE_STOPWORDS or t in _QUERY_WH_WORDS or t in _QUERY_VERBS:
            continue
        if len(t) < 3:
            continue
        out.append(t)
    return out


def _phrase_regex(phrase: str) -> re.Pattern:
    """Build a regex that matches phrase with flexible spaces/hyphens."""
    phrase = _norm_space(phrase)
    words = [w for w in _WORD_RE.findall(phrase) if w]
    if not words:
        return re.compile(r"a^")
    if len(words) == 1:
        return re.compile(rf"\b{re.escape(words[0])}\b", flags=re.IGNORECASE)

    sep = r"[\s\-]+"
    pat = r"\b" + sep.join(re.escape(w) for w in words) + r"\b"
    return re.compile(pat, flags=re.IGNORECASE)


def phrase_in_text(phrase: str, text: str) -> bool:
    if not phrase or not text:
        return False
    return _phrase_regex(phrase).search(text) is not None


def _strip_query_scaffolding(query: str) -> str:
    """Remove leading question scaffolding ("what is", "explain", etc.) but keep internal stopwords."""
    q = _norm_space(query)
    if not q:
        return ""

    # strip trailing punctuation
    q = q.strip().strip("?!.:")

    ql = q.lower()

    # common multi-word prefixes (ordered longest-first)
    prefixes = [
        "give an overview of ",
        "give overview of ",
        "what is the ",
        "what are the ",
        "what is ",
        "what are ",
        "explain the ",
        "explain ",
        "describe the ",
        "describe ",
        "define the ",
        "define ",
        "summarize the ",
        "summarize ",
        "outline the ",
        "outline ",
        "compare ",
        "contrast ",
    ]

    for p in prefixes:
        if ql.startswith(p):
            q = q[len(p):].strip()
            ql = q.lower()
            break

    # remove leading stopwords repeatedly
    toks = q.split()
    while toks and toks[0].lower() in _BASE_STOPWORDS:
        toks = toks[1:]
    q = " ".join(toks).strip()

    return q


def choose_anchor_phrase(query: str, *, text_pool: Optional[str] = None, max_ngram: int = 6) -> str:
    """Choose an anchor phrase for the query.

    Preference order:
      1) Longest n-gram (>=2) from the *stripped query phrase* that appears in text_pool (summary/keypoints).
      2) If none appear, use the stripped query phrase.
      3) If query becomes empty, fall back to normalized query.

    This preserves phrases like "divide and conquer" (the "and" is important).
    """
    q = _strip_query_scaffolding(query or "")
    if not q:
        return normalize_for_match(query or "")

    pool = (text_pool or "")
    toks = tokenize(q)

    # If we have a summary/keypoint pool, pick the best matching n-gram.
    if pool and toks:
        n_max = min(max_ngram, len(toks))
        for n in range(n_max, 1, -1):
            for i in range(0, len(toks) - n + 1):
                cand_toks = toks[i : i + n]
                if not cand_toks:
                    continue
                # avoid candidates that start/end with stopwords (usually fragments)
                if cand_toks[0] in _BASE_STOPWORDS or cand_toks[-1] in _BASE_STOPWORDS:
                    continue
                # must contain at least one non-stopword content token
                if not any(t not in _BASE_STOPWORDS for t in cand_toks):
                    continue
                cand = " ".join(cand_toks)
                if phrase_in_text(cand, pool):
                    return cand

    # fallback: stripped phrase (limit length)
    if toks:
        return " ".join(toks[:12])

    return normalize_for_match(query or "")


def stable_quiz_id(doc_id: str, query: str, *, version: str = "v1") -> str:
    """Stable quiz_id based on (doc_id, normalized query, version)."""
    raw = f"{doc_id}|{normalize_for_match(query)}|{version}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


@dataclass(frozen=True)
class QueryAnchor:
    """Compact representation of user intent for strict anchoring."""

    raw_query: str
    anchor_phrase: str
    query_tokens: Set[str]
    anchor_tokens: Set[str]

    def matches(self, text: str) -> bool:
        return phrase_in_text(self.anchor_phrase, text)


def build_query_anchor(query: str, *, text_pool: Optional[str] = None) -> QueryAnchor:
    """Build QueryAnchor with robust anchor phrase and token sets."""
    anchor = choose_anchor_phrase(query or "", text_pool=text_pool or "")
    qt = set(content_tokens(query or ""))
    # anchor_tokens derived from anchor phrase but stopwords removed
    at = set(content_tokens(anchor or "")) if anchor else set()
    return QueryAnchor(raw_query=query or "", anchor_phrase=anchor or (query or ""), query_tokens=qt, anchor_tokens=at)


def relevance_score(text: str, anchor: QueryAnchor) -> int:
    """A small integer score: higher means more query-relevant."""
    if not text:
        return 0
    score = 0
    if anchor.anchor_phrase and phrase_in_text(anchor.anchor_phrase, text):
        score += 6
    tl = (text or "").lower()
    for tok in anchor.query_tokens:
        if tok and tok in tl:
            score += 1
    return score


def is_strongly_relevant(text: str, anchor: QueryAnchor, *, min_token_hits: int = 2) -> bool:
    """True if text is strongly relevant to query.

    Strong relevance is:
      - anchor phrase appears, OR
      - at least `min_token_hits` query content tokens appear.
    """
    if not text:
        return False
    if anchor.anchor_phrase and phrase_in_text(anchor.anchor_phrase, text):
        return True
    tl = (text or "").lower()
    hits = sum(1 for tok in anchor.query_tokens if tok and tok in tl)
    return hits >= min_token_hits


def best_supporting_span(anchor: QueryAnchor, spans: Sequence[Tuple[str, Optional[int]]]) -> Tuple[str, Optional[int]]:
    """Pick the best supporting span (text, page_no) for the anchor.

    Preference:
      1) Spans that contain the anchor phrase
      2) Higher relevance_score
      3) Shorter span (less rambling)
    """
    best: Tuple[str, Optional[int]] = ("", None)
    best_key = (-10_000, -10_000, 10_000)

    for text, page_no in spans:
        t = _norm_space(text)
        if not t:
            continue
        rel = relevance_score(t, anchor)
        has_phrase = 1 if (anchor.anchor_phrase and phrase_in_text(anchor.anchor_phrase, t)) else 0
        key = (has_phrase, rel, -len(t))
        if key > best_key:
            best_key = key
            best = (t, page_no)

    return best
