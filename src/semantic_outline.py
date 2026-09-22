
# src/semantic_outline.py
"""
Deterministic, explainable document-understanding utilities for:
  - extractive summarization that preserves "main idea"
  - concept extraction (definitions + keyphrases)
  - flashcards + quizzes derived from concepts
  - knowledge map (concept graph) built from true concepts, not glue words

Design goals:
  - No LLM required (works offline / before Ollama rewriting)
  - Deterministic (seeded where randomness is needed)
  - Debuggable (intermediate artifacts are easy to inspect)
  - Language-aware (English / Chinese / mixed)

This module is intentionally dependency-light:
  - numpy + scikit-learn are used if available
  - If scikit-learn is missing, code falls back to frequency-based heuristics
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ------------------------------------------------------------
# Language + token hygiene
# ------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _EN_STOP  # type: ignore
except Exception:  # pragma: no cover
    _EN_STOP = frozenset(
        {
            "the","a","an","and","or","to","of","in","on","for","with","by","as","at","from",
            "this","that","these","those","it","its","is","are","was","were","be","been",
            "we","you","they","he","she","i","our","your","their",
        }
    )

_EXTRA_BAD_TERMS = {
    # Terms that are usually structural or low-value as study concepts
    # (keep this conservative; concept filtering also uses sentence hit-count).

    "etc","e.g","i.e","eg","ie","et",
    "figure","fig","table","section","chapter","pages","page",
    "example","examples","exercise","exercises","problem","problems",
}


_WEAK_PHRASE_TAILS = {
    # Multi-word phrases ending with these are often not concepts, but fragments.
    "representing",
    "followed",
    "including",
    "using",
    "called",
    "based",
}

# Remove common PDF artifacts
_SOFT_HYPHEN = "\u00ad"

def has_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))

def cjk_ratio(texts: Sequence[str], sample_chars: int = 20000) -> float:
    s = "".join(t for t in texts if t)[:sample_chars]
    if not s:
        return 0.0
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(1, len(s))

def is_cjk_heavy(texts: Sequence[str], threshold: float = 0.15) -> bool:
    return cjk_ratio(texts) >= threshold

def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    t = (text or "").replace("\x00", " ").replace(_SOFT_HYPHEN, "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces but keep paragraph breaks
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def repair_hyphenation(text: str) -> str:
    """
    Fix common PDF line-wrap hyphenation artifacts:
      - "in- dex" -> "index"
      - "rep- resenting" -> "representing"
    We only remove hyphen when it is followed by whitespace and a letter.
    """
    if not text:
        return ""
    t = text
    # hyphen + whitespace + letter => join
    t = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", t)
    return t

def normalize_for_sentence_split(text: str) -> str:
    """
    Prepare text for sentence splitting while preserving paragraph breaks.
    Handles:
      - single newlines (line-wrap) -> spaces
      - paragraph newlines -> kept as \n\n
    """
    if not text:
        return ""
    t = normalize_whitespace(text)
    t = repair_hyphenation(t)

    # Mark paragraph breaks
    t = re.sub(r"\n{2,}", " <PARA> ", t)
    # Convert remaining single newlines to spaces
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    # Restore paragraphs
    t = t.replace("<PARA>", "\n\n")
    return t.strip()

_SENT_SPLIT_RE = re.compile(
    r"(?<=[\.\!\?])\s+|(?<=[。！？])\s+|(?<=[。！？])(?=\S)|\n{2,}"
)

def split_sentences(text: str) -> List[str]:
    """
    Robust(ish) EN/ZH sentence splitter, tolerant to PDF newlines.
    """
    t = normalize_for_sentence_split(text)
    if not t:
        return []
    parts = _SENT_SPLIT_RE.split(t)
    out: List[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        # Filter short Latin fragments (keep short CJK)
        if len(s) < 18 and not has_cjk(s):
            continue
        out.append(s)
    return out

def _is_bad_english_token(tok: str) -> bool:
    t = (tok or "").strip().lower()
    if not t:
        return True
    if len(t) < 3:
        return True
    if t.isdigit():
        return True
    if re.fullmatch(r"[^a-z0-9]+", t):
        return True
    if t in _EXTRA_BAD_TERMS:
        return True
    if t in _EN_STOP:
        return True
    return False

def _is_bad_term(term: str) -> bool:
    term = (term or "").strip()
    if not term:
        return True
    if term.isdigit():
        return True
    if has_cjk(term):
        # For CJK, allow short but not single character.
        return len(term) < 2
    # English / Latin
    t = term.lower().strip()
    parts = [p for p in re.split(r"\s+", t) if p]
    if not parts:
        return True
    if any(_is_bad_english_token(p) for p in parts):
        return True
    # Kill phrases that are just glue
    if len(parts) == 1 and _is_bad_english_token(parts[0]):
        return True
    return False

def normalize_concept_id(term: str) -> str:
    term = (term or "").strip()
    if not term:
        return ""
    if has_cjk(term):
        return term
    t = term.lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_]", "", t)
    return t

# ------------------------------------------------------------
# Chunk stitching (fixes boundary truncation when chunking is char-based)
# ------------------------------------------------------------

def merge_with_overlap(a: str, b: str, max_overlap: int = 220, min_overlap: int = 40) -> str:
    """
    Join two strings, removing a repeated overlap if detected.

    This is used to stitch overlapping chunks on the same page.
    Works best when chunks are produced with fixed overlap (e.g., 120 chars).
    """
    a = a or ""
    b = b or ""
    if not a:
        return b
    if not b:
        return a

    a_tail = a[-max_overlap:]
    b_head = b[:max_overlap]

    # Exact overlap search from large -> small
    max_L = min(len(a_tail), len(b_head), max_overlap)
    for L in range(max_L, min_overlap - 1, -1):
        if a_tail[-L:] == b[:L]:
            return a + b[L:]

    # Softer check: if b_head is contained in a_tail, drop duplicate prefix
    # (helps when overlap isn't perfectly aligned)
    idx = a_tail.find(b_head[: min(80, len(b_head))])
    if idx != -1 and idx > 0:
        # If the prefix fragment appears near the end, assume overlap-ish
        # and avoid duplicating the first line of b.
        # Conservative: only drop if match is late.
        if idx >= max(0, len(a_tail) - 120):
            return a + "\n" + b

    return a + "\n" + b

def build_page_contexts_from_hits(
    hits: Sequence[Any],
    *,
    max_chars_per_page: int = 9000,
) -> Dict[int, Dict[str, Any]]:
    """
    Group SearchHit-like objects by page_no, sort by chunk_index, stitch into a page context.

    Returns:
      { page_no: {"page_no": int, "text": str, "chunk_ids": [..], "chunk_indexes":[..]} }
    """
    by_page: Dict[int, List[Any]] = {}
    for h in hits or []:
        try:
            p = int(getattr(h, "page_no", 0) or 0)
        except Exception:
            continue
        if p <= 0:
            continue
        by_page.setdefault(p, []).append(h)

    out: Dict[int, Dict[str, Any]] = {}
    for p, items in by_page.items():
        items_sorted = sorted(
            items,
            key=lambda x: (int(getattr(x, "chunk_index", 0) or 0), -float(getattr(x, "score", 0.0) or 0.0)),
        )
        text = ""
        chunk_ids: List[str] = []
        chunk_indexes: List[int] = []
        for h in items_sorted:
            t = getattr(h, "text", "") or ""
            if not t.strip():
                continue
            text = merge_with_overlap(text, t)
            cid = str(getattr(h, "chunk_id", "") or "")
            if cid:
                chunk_ids.append(cid)
            chunk_indexes.append(int(getattr(h, "chunk_index", 0) or 0))
            if max_chars_per_page > 0 and len(text) >= max_chars_per_page:
                break

        text = normalize_whitespace(text)
        text = repair_hyphenation(text)

        out[p] = {
            "page_no": p,
            "text": text,
            "chunk_ids": list(dict.fromkeys(chunk_ids)),
            "chunk_indexes": chunk_indexes,
        }

    return dict(sorted(out.items(), key=lambda kv: kv[0]))

# ------------------------------------------------------------
# Sentence objects + rhetorical roles
# ------------------------------------------------------------

@dataclass(frozen=True)
class Sentence:
    text: str
    page_no: int
    sent_index: int  # within page context

@dataclass(frozen=True)
class SentenceFeatures:
    is_heading: bool
    is_definition: bool
    is_purpose: bool
    is_example: bool
    is_warning: bool
    starts_fragment: bool

_DEF_CUES_EN = re.compile(r"\b(is|are|means|refers\s+to|denotes|represents|defined\s+as)\b", re.IGNORECASE)
_PURPOSE_CUES_EN = re.compile(r"\b(goal|purpose|we\s+will|this\s+(section|chapter|paper)\s+)\b", re.IGNORECASE)
_EXAMPLE_CUES_EN = re.compile(r"\b(for\s+example|e\.g\.|such\s+as)\b", re.IGNORECASE)
_WARNING_CUES_EN = re.compile(r"\b(note\s+that|warning|be\s+careful|caution)\b", re.IGNORECASE)

_DEF_CUES_ZH = re.compile(r"(定义|指|称为|是指|表示|意味着)")
_PURPOSE_CUES_ZH = re.compile(r"(目的|目标|本章|本节|我们将)")
_EXAMPLE_CUES_ZH = re.compile(r"(例如|比如)")
_WARNING_CUES_ZH = re.compile(r"(注意|警告|小心)")

def detect_sentence_features(s: str, *, cjk: bool) -> SentenceFeatures:
    t = (s or "").strip()
    if not t:
        return SentenceFeatures(False, False, False, False, False, False)

    # Heading heuristic: short-ish, no sentence-ending punctuation, or ends with ':'
    is_heading = False
    if len(t) <= 90:
        if t.endswith(":"):
            is_heading = True
        elif not re.search(r"[\.!\?。！？]$", t) and sum(ch.isalpha() for ch in t) >= 10:
            # title-ish line
            caps = sum(1 for ch in t if ch.isupper())
            letters = sum(1 for ch in t if ch.isalpha())
            if letters > 0 and (caps / letters) >= 0.6:
                is_heading = True

    if cjk:
        is_def = bool(_DEF_CUES_ZH.search(t))
        is_purpose = bool(_PURPOSE_CUES_ZH.search(t))
        is_example = bool(_EXAMPLE_CUES_ZH.search(t))
        is_warning = bool(_WARNING_CUES_ZH.search(t))
        starts_fragment = False  # harder to detect reliably for CJK
    else:
        is_def = bool(_DEF_CUES_EN.search(t))
        is_purpose = bool(_PURPOSE_CUES_EN.search(t))
        is_example = bool(_EXAMPLE_CUES_EN.search(t))
        is_warning = bool(_WARNING_CUES_EN.search(t))
        # Fragment heuristic: starts with lowercase and not a proper sentence start
        starts_fragment = bool(re.match(r"^[a-z]", t)) and not t.lower().startswith(("i ", "i'", "we ", "in ", "on ", "for ", "to "))
    return SentenceFeatures(is_heading, is_def, is_purpose, is_example, is_warning, starts_fragment)

# ------------------------------------------------------------
# Vectorization + LexRank centrality
# ------------------------------------------------------------

def _safe_import_sklearn():
    try:
        import numpy as np  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore
        from sklearn.preprocessing import normalize  # type: ignore
        return np, TfidfVectorizer, KMeans, TruncatedSVD, normalize
    except Exception:
        return None

def _lexrank(sim: "Any", *, damping: float = 0.85, max_iter: int = 200, tol: float = 1e-7) -> "Any":
    """
    sim: dense numpy array (n,n) non-negative; diagonal should be zero.
    Returns: centrality vector shape (n,)
    """
    import numpy as np  # type: ignore

    n = int(sim.shape[0])
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)

    W = np.maximum(sim, 0.0)
    # Row-normalize
    row_sums = W.sum(axis=1)
    P = np.zeros_like(W, dtype=float)
    for i in range(n):
        if row_sums[i] > 0:
            P[i, :] = W[i, :] / row_sums[i]
        else:
            P[i, :] = 1.0 / n

    v = np.ones(n, dtype=float) / n
    teleport = (1.0 - damping) / n

    for _ in range(max_iter):
        v_new = teleport + damping * (P.T @ v)
        if float(np.linalg.norm(v_new - v, 1)) < tol:
            v = v_new
            break
        v = v_new
    # Normalize
    s = float(v.sum())
    if s > 0:
        v = v / s
    return v

def sentence_embeddings(
    sentences: Sequence[str],
    *,
    cjk: bool,
) -> Tuple["Any", "Any", List[str]]:
    """
    Build TF-IDF embeddings for sentences.

    Returns:
      - vectorizer (or None)
      - X: sentence matrix (scipy sparse) or None
      - cleaned_sentences (aligned with X)
    """
    sk = _safe_import_sklearn()
    if sk is None:
        return None, None, [s for s in sentences]

    np, TfidfVectorizer, _, _, _ = sk

    sents = [normalize_for_sentence_split(s) for s in sentences]
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return None, None, []

    if cjk:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False, norm="l2")
    else:
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=(1.0 if len(sents) < 3 else 0.95),
            norm="l2",
        )

    X = vec.fit_transform(sents)
    return vec, X, sents

def compute_centrality(vec: "Any", X: "Any") -> List[float]:
    """
    Centrality via LexRank/PageRank on cosine similarity graph.
    """
    sk = _safe_import_sklearn()
    if sk is None or X is None:
        return []

    np, _, _, _, _ = sk
    n = int(X.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # Cosine similarity (X is L2 normalized)
    # Use dense if small; otherwise approximate by taking top neighbors
    if n <= 500:
        S = (X @ X.T).toarray()
        np.fill_diagonal(S, 0.0)
        # prune tiny weights to reduce noise
        S[S < 0.08] = 0.0
        c = _lexrank(S)
        return [float(x) for x in c.tolist()]

    # Large: approximate by keeping strongest similarities per row
    S_sparse = (X @ X.T)
    S_sparse.setdiag(0)
    S_sparse.eliminate_zeros()
    # Convert to dense is too big; fallback: uniform
    return [1.0 / n for _ in range(n)]

# ------------------------------------------------------------
# Topic clustering (optional but useful for "main point" detection)
# ------------------------------------------------------------

def cluster_topics(X: "Any", *, seed: int, max_topics: int = 6) -> Optional[List[int]]:
    """
    Cluster sentence vectors into topics (KMeans on reduced SVD space).
    Returns list of labels per sentence or None if unavailable.
    """
    sk = _safe_import_sklearn()
    if sk is None or X is None:
        return None

    np, _, KMeans, TruncatedSVD, normalize = sk
    n = int(X.shape[0])
    if n < 8:
        return [0 for _ in range(n)]

    # Choose k heuristically
    k = int(round(math.sqrt(n) / 2.0))
    k = max(2, min(max_topics, k))
    # Reduce dimension
    dim = min(64, max(8, k * 8), int(X.shape[1]) - 1) if int(X.shape[1]) > 1 else 1
    if dim <= 1:
        return [0 for _ in range(n)]

    svd = TruncatedSVD(n_components=dim, random_state=seed)
    Y = svd.fit_transform(X)
    Y = normalize(Y)

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(Y)
    return [int(x) for x in labels.tolist()]

def topic_labels_from_clusters(vec: "Any", X: "Any", labels: Sequence[int], *, top_terms: int = 4) -> Dict[int, str]:
    """
    For each cluster label, derive a short label from top TF-IDF features.
    """
    if vec is None or X is None or not labels:
        return {}
    try:
        import numpy as np  # type: ignore
    except Exception:
        return {}

    labels = list(labels)
    n = int(X.shape[0])
    if n == 0:
        return {}

    feat_names = getattr(vec, "get_feature_names_out", None)
    if feat_names is None:
        return {}
    names = vec.get_feature_names_out()

    out: Dict[int, str] = {}
    for lab in sorted(set(labels)):
        idx = [i for i, L in enumerate(labels) if L == lab]
        if not idx:
            continue
        # mean vector for cluster
        v = X[idx].mean(axis=0)
        # v may be matrix; convert
        arr = getattr(v, "A1", None)
        if arr is None:
            try:
                arr = np.asarray(v).ravel()
            except Exception:
                continue
        # pick top terms
        order = arr.argsort()[::-1]
        terms: List[str] = []
        for j in order[: 80]:
            term = str(names[int(j)]).strip()
            if _is_bad_term(term):
                continue
            terms.append(term)
            if len(terms) >= top_terms:
                break
        out[lab] = ", ".join(terms) if terms else f"Topic {lab+1}"
    return out

# ------------------------------------------------------------
# Scoring + selection
# ------------------------------------------------------------

def _query_vector(vec: "Any", query: str) -> Optional["Any"]:
    if vec is None:
        return None
    q = (query or "").strip()
    if not q:
        return None
    try:
        return vec.transform([q])
    except Exception:
        return None

def _cosine_scores(X: "Any", qv: "Any") -> List[float]:
    if X is None or qv is None:
        return []
    try:
        scores = (X @ qv.T).toarray().ravel()
        return [float(x) for x in scores.tolist()]
    except Exception:
        return []

def score_sentences(
    sentences: Sequence[Sentence],
    sentence_texts: Sequence[str],
    *,
    query: str,
    cjk: bool,
    seed: int,
) -> Dict[str, Any]:
    """
    Return:
      {
        "vectorizer": vec,
        "X": X,
        "centrality": [...],
        "query_relevance": [...],
        "features": [SentenceFeatures...],
        "scores": [...],
        "topic_labels": [... or None],
        "topic_names": {lab: name}
      }
    """
    vec, X, sents_clean = sentence_embeddings(sentence_texts, cjk=cjk)
    if X is None:
        # fallback: naive scores based on position + cue words
        feats = [detect_sentence_features(s.text, cjk=cjk) for s in sentences]
        scores: List[float] = []
        for s, f in zip(sentences, feats):
            sc = 0.0
            sc += max(0.0, 0.15 - 0.03 * float(s.sent_index))
            if f.is_definition:
                sc += 0.25
            if f.is_purpose:
                sc += 0.25
            if f.is_heading:
                sc += 0.12
            if f.is_warning:
                sc += 0.10
            if f.starts_fragment:
                sc -= 0.20
            scores.append(sc)
        return {
            "vectorizer": None,
            "X": None,
            "centrality": [0.0 for _ in scores],
            "query_relevance": [0.0 for _ in scores],
            "features": feats,
            "scores": scores,
            "topic_labels": None,
            "topic_names": {},
        }

    centrality = compute_centrality(vec, X)
    if not centrality:
        centrality = [0.0 for _ in range(int(X.shape[0]))]

    qv = _query_vector(vec, query)
    qrel = _cosine_scores(X, qv)
    if not qrel:
        qrel = [0.0 for _ in range(int(X.shape[0]))]

    feats = [detect_sentence_features(s.text, cjk=cjk) for s in sentences]

    # Topic clustering
    labels = cluster_topics(X, seed=seed)
    topic_names = topic_labels_from_clusters(vec, X, labels or [], top_terms=4) if labels else {}

    scores: List[float] = []
    for i, (s, f) in enumerate(zip(sentences, feats)):
        # Base weights
        sc = 0.0
        sc += 0.38 * float(centrality[i])
        sc += 0.38 * float(qrel[i])

        # Position (topic sentences often appear early)
        sc += max(0.0, 0.12 - 0.03 * float(s.sent_index))

        # Rhetorical role bonuses
        if f.is_purpose:
            sc += 0.14
        if f.is_definition:
            sc += 0.12
        if f.is_heading:
            sc += 0.10
        if f.is_warning:
            sc += 0.06
        if f.is_example:
            sc += 0.03

        # Penalize fragments
        if f.starts_fragment:
            sc -= 0.14

        # Penalize extreme symbol noise
        t = s.text
        sym = sum(1 for ch in t if not (ch.isalnum() or ch.isspace() or has_cjk(ch)))
        sym_ratio = sym / max(1, len(t))
        if sym_ratio > (0.55 if not cjk else 0.45):
            sc -= 0.18

        scores.append(float(sc))

    return {
        "vectorizer": vec,
        "X": X,
        "centrality": centrality,
        "query_relevance": qrel,
        "features": feats,
        "scores": scores,
        "topic_labels": labels,
        "topic_names": topic_names,
    }

def mmr_select(
    X: "Any",
    candidates: List[int],
    cand_scores: List[float],
    *,
    top_n: int,
    lambda_mult: float = 0.75,
) -> List[int]:
    """
    Maximal Marginal Relevance selection among candidates.
    Uses cosine similarity between candidate vectors (X is L2 normalized).
    """
    if top_n <= 0 or not candidates:
        return []
    if X is None:
        # Fallback: just top by score
        ranked = sorted(zip(candidates, cand_scores), key=lambda x: x[1], reverse=True)
        return [i for i, _ in ranked[:top_n]]

    try:
        import numpy as np  # type: ignore
    except Exception:
        ranked = sorted(zip(candidates, cand_scores), key=lambda x: x[1], reverse=True)
        return [i for i, _ in ranked[:top_n]]

    top_n = min(top_n, len(candidates))
    # Start from best score
    order = sorted(range(len(candidates)), key=lambda k: cand_scores[k], reverse=True)
    selected: List[int] = [candidates[order[0]]]
    selected_set = {selected[0]}

    while len(selected) < top_n:
        best_i = None
        best_val = -1e18
        for idx in order:
            s_idx = candidates[idx]
            if s_idx in selected_set:
                continue
            rel = float(cand_scores[idx])
            # diversity penalty: max similarity to any selected
            vi = X[s_idx]
            max_sim = 0.0
            for j in selected:
                sim = float((vi @ X[j].T).toarray()[0][0])
                if sim > max_sim:
                    max_sim = sim
            mmr_val = lambda_mult * rel - (1.0 - lambda_mult) * max_sim
            if mmr_val > best_val:
                best_val = mmr_val
                best_i = s_idx
        if best_i is None:
            break
        selected.append(best_i)
        selected_set.add(best_i)

    return selected

# ------------------------------------------------------------
# Concept extraction
# ------------------------------------------------------------

def _token_spans_en(s: str) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_\-]*", s):
        out.append((m.group(0), m.start(), m.end()))
    return out

def _extract_lhs_term_before_def_cue_en(sentence: str) -> Optional[str]:
    """
    Attempt to extract a term being defined in a definitional sentence.
    Example: "A pointer is a variable ..." -> "pointer"
    """
    s = (sentence or "").strip()
    if not s:
        return None

    m = re.search(r"\b(is|are|means|refers\s+to|denotes|represents|defined\s+as)\b", s, flags=re.IGNORECASE)
    if not m:
        return None
    cue_start = m.start()

    toks = _token_spans_en(s)
    if not toks:
        return None

    # tokens that end before cue
    left = [t for t in toks if t[2] <= cue_start]
    if not left:
        return None

    # walk backwards, collect up to 4 content tokens
    collected: List[str] = []
    for tok, _, _ in reversed(left):
        tl = tok.lower()
        if tl in {"a","an","the","this","that","these","those","it","we","you","they","he","she","i","our","your","their"}:
            if collected:
                break
            continue
        if _is_bad_english_token(tl):
            if collected:
                break
            continue
        collected.append(tok)

        if len(collected) >= 4:
            break

    if not collected:
        return None
    term = " ".join(reversed(collected)).strip()
    # Normalize common patterns (plural -> singular naive)
    term = re.sub(r"\s+", " ", term)
    if _is_bad_term(term):
        return None
    return term

def _extract_lhs_term_before_def_cue_zh(sentence: str) -> Optional[str]:
    s = (sentence or "").strip()
    if not s:
        return None
    # Very simple: take up to 10 chars before the first "是/指/称为/定义"
    m = _DEF_CUES_ZH.search(s)
    if not m:
        return None
    left = s[: m.start()].strip()
    if not left:
        return None
    # take last "word-like" chunk (CJK or alnum)
    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_\-]{3,}", left)
    if not parts:
        return None
    term = parts[-1]
    if _is_bad_term(term):
        return None
    return term

def extract_concepts(
    page_texts: Sequence[str],
    sentences: Sequence[Sentence],
    *,
    query: str,
    cjk: bool,
    max_concepts: int = 18,
) -> List[Dict[str, Any]]:
    """
    Extract a ranked list of study-worthy concepts with definitions + evidence.

    Strategy:
      1) High-precision candidates from definitional sentences (pattern-based)
      2) High-recall candidates from TF-IDF keyphrases on page_texts
      3) For each candidate, attach best definition sentence + page citation
    """
    all_text = "\n\n".join(t for t in page_texts if t).strip()
    if not all_text:
        return []

    # 1) definitional candidates
    defs: List[str] = []
    for s in sentences:
        f = detect_sentence_features(s.text, cjk=cjk)
        if not f.is_definition:
            continue
        if cjk:
            term = _extract_lhs_term_before_def_cue_zh(s.text)
        else:
            term = _extract_lhs_term_before_def_cue_en(s.text)
        if term and term not in defs:
            defs.append(term)

    # 2) TF-IDF keyphrase candidates
    tfidf_terms: List[str] = []
    sk = _safe_import_sklearn()
    if sk is not None:
        np, TfidfVectorizer, _, _, _ = sk
        try:
            if cjk:
                vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False, norm="l2")
            else:
                vec = TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=(1.0 if len([t for t in page_texts if t.strip()]) < 3 else 0.90),
                    norm="l2",
                )
            X = vec.fit_transform([t for t in page_texts if t.strip()])
            names = vec.get_feature_names_out()
            scores = X.mean(axis=0).A1
            idx = scores.argsort()[::-1]
            for i in idx[:400]:
                term = str(names[int(i)]).strip()
                if _is_bad_term(term):
                    continue
                tfidf_terms.append(term)
                if len(tfidf_terms) >= max(2 * max_concepts, 30):
                    break
        except Exception:
            tfidf_terms = []
    else:
        # fallback: frequency
        blob = all_text.lower()
        toks = re.findall(r"[a-zA-Z]{4,}", blob)
        freq: Dict[str, int] = {}
        for t in toks:
            if _is_bad_english_token(t):
                continue
            freq[t] = freq.get(t, 0) + 1
        tfidf_terms = [t for t, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:40]]

    # Merge candidates: definitional terms first, then tfidf terms
    candidates = []
    for t in defs + tfidf_terms:
        if t not in candidates:
            candidates.append(t)

    # If query is provided, lightly promote candidates that overlap the query
    q_low = (query or "").lower()
    def cand_key(t: str) -> Tuple[int, int]:
        tl = t.lower()
        # Higher rank if overlaps query
        overlap = 1 if (q_low and tl in q_low) or (q_low and any(w in tl for w in q_low.split())) else 0
        # prefer shorter terms (more likely atomic concept)
        return (-overlap, len(tl))

    candidates = sorted(candidates, key=cand_key)

    # Filter weak single-word candidates.
    # Rationale: TF-IDF on short context can surface generic words ("write", "followed", ...)
    # We keep:
    #   - definitional candidates
    #   - terms overlapping the query
    #   - multi-word phrases
    #   - single-word terms that appear in >=2 sentences (or are ALLCAPS like NIL)
    sent_texts = [s.text for s in sentences]
    q_terms = set((query or "").lower().split())

    filtered: List[str] = []
    for term in candidates[:400]:
        if _is_bad_term(term):
            continue
        tl = term.lower().strip()
        parts = tl.split() if not cjk else [tl]
        is_multi = (len(parts) >= 2) if not cjk else False
        # Drop obvious phrase fragments like 'variable representing'
        if (not cjk) and len(parts) >= 2:
            tail = parts[-1]
            if (tail in _WEAK_PHRASE_TAILS) and (not overlaps_query) and (not in_defs):
                continue
        in_defs = term in defs
        overlaps_query = bool(q_terms) and (tl in q_terms or any(q in tl for q in q_terms))

        # Sentence hit-count
        hit_count = 0
        if cjk or has_cjk(term):
            for st in sent_texts:
                if term in st:
                    hit_count += 1
        else:
            phrase = re.sub(r"\s+", " ", tl).strip()
            body = r"\s+".join(re.escape(p) for p in phrase.split(" ") if p)
            patt = re.compile(rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])", flags=re.IGNORECASE)
            for st in sent_texts:
                if patt.search(" " + st.lower() + " "):
                    hit_count += 1

        if (not is_multi) and (hit_count < 2) and (not in_defs) and (not overlaps_query) and (not term.isupper()):
            continue

        filtered.append(term)
        if len(filtered) >= max(3 * max_concepts, 60):
            break

    candidates = filtered

    # Attach best definition/evidence for each candidate
    concepts: List[Dict[str, Any]] = []
    for term in candidates:
        if len(concepts) >= max_concepts:
            break
        if _is_bad_term(term):
            continue

        definition, page_no, evidence = find_best_definition(sentences, term, cjk=cjk)
        if not definition:
            # If we can't define it, it's a weak concept for learning outputs
            continue

        concepts.append(
            {
                "id": normalize_concept_id(term),
                "term": term,
                "definition": definition,
                "page_no": page_no,
                "evidence": evidence,
            }
        )

    return concepts

def find_best_definition(
    sentences: Sequence[Sentence],
    term: str,
    *,
    cjk: bool,
) -> Tuple[str, int, str]:
    """
    Pick the most 'definition-like' sentence containing the term.
    Returns (definition_sentence, page_no, evidence_sentence).
    """
    term = (term or "").strip()
    if not term:
        return "", 0, ""

    term_l = term.lower()

    best = ("", 0, "", -1e18)  # def, page, evidence, score

    for s in sentences:
        txt = s.text.strip()
        if not txt:
            continue

        if cjk or has_cjk(term):
            if term not in txt:
                continue
        else:
            # boundary-safe match
            patt = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term_l)}(?![A-Za-z0-9_])", flags=re.IGNORECASE)
            if not patt.search(" " + txt.lower() + " "):
                continue

        f = detect_sentence_features(txt, cjk=cjk)
        sc = 1.0
        if f.is_definition:
            sc += 2.2
        if f.is_purpose:
            sc += 0.6
        if f.is_heading:
            sc += 0.3
        if f.is_example:
            sc -= 0.2
        if f.starts_fragment:
            sc -= 0.4

        # Prefer shorter, cleaner definitions
        if len(txt) > 260 and not cjk:
            sc -= (len(txt) - 260) / 260
        if len(txt) < 35 and not cjk:
            sc -= 0.3

        # Prefer "term is ..." direct pattern
        if not cjk:
            direct = re.search(rf"(?<![A-Za-z0-9_]){re.escape(term_l)}(?![A-Za-z0-9_])\s+(is|are|means|refers\s+to)\b", txt.lower())
            if direct:
                sc += 1.6

        if sc > best[3]:
            best = (txt, s.page_no, txt, sc)

    if not best[0]:
        return "", 0, ""

    # Clip
    defn = best[0].strip()
    if len(defn) > 320:
        defn = defn[:320].rstrip() + "..."
    return defn, int(best[1] or 0), best[2]

# ------------------------------------------------------------
# Summary building (main idea + why it matters + key points)
# ------------------------------------------------------------

def build_summary(
    page_contexts: Dict[int, Dict[str, Any]],
    *,
    query: str,
    seed: int,
    max_key_points: int = 7,
) -> Dict[str, Any]:
    """
    Build a structured summary from stitched page contexts.
    """
    page_nos = sorted(page_contexts.keys())
    page_texts = [page_contexts[p]["text"] for p in page_nos if page_contexts[p].get("text")]

    cjk = is_cjk_heavy(page_texts)

    # Build sentence list with metadata
    sentences: List[Sentence] = []
    for p in page_nos:
        text = page_contexts[p].get("text") or ""
        sents = split_sentences(text)
        for i, s in enumerate(sents):
            sentences.append(Sentence(text=s, page_no=p, sent_index=i))

    if not sentences:
        return {
            "language": "cjk" if cjk else "en",
            "pages": page_nos,
            "main_idea": "",
            "why_it_matters": "",
            "key_points": [],
            "key_points_cited": [],
            "concepts": [],
            "topics": [],
        }

    # Score sentences
    scored = score_sentences(sentences, [s.text for s in sentences], query=query, cjk=cjk, seed=seed)
    scores: List[float] = list(scored["scores"])
    X = scored.get("X")
    labels = scored.get("topic_labels")
    topic_names = scored.get("topic_names") or {}

    # Candidate indices sorted by score desc
    order = sorted(range(len(sentences)), key=lambda i: (scores[i], -sentences[i].page_no, -sentences[i].sent_index), reverse=True)

    # Main idea selection:
    # prefer purpose/heading/definition, and early on page
    best_main = None
    best_val = -1e18
    for i in order[: min(50, len(order))]:
        s = sentences[i]
        f = detect_sentence_features(s.text, cjk=cjk)
        val = scores[i]
        if s.sent_index <= 2 and (f.is_purpose or f.is_heading or f.is_definition):
            val += 0.25
        if val > best_val:
            best_val = val
            best_main = i
    if best_main is None:
        best_main = order[0]

    main_idea = sentences[best_main].text.strip()

    # Why it matters: choose a sentence with purpose/warning cue, different from main
    why = ""
    for i in order:
        if i == best_main:
            continue
        s = sentences[i]
        f = detect_sentence_features(s.text, cjk=cjk)
        if f.is_purpose or f.is_warning:
            why = s.text.strip()
            break

    # Select key points via MMR (diverse but relevant)
    cand = order[: min(120, len(order))]
    cand_scores = [scores[i] for i in cand]
    chosen = mmr_select(X, cand, cand_scores, top_n=max_key_points, lambda_mult=0.78)
    # ensure main idea is included at top
    if best_main not in chosen:
        chosen = [best_main] + chosen
        chosen = chosen[:max_key_points]
    else:
        # move to front
        chosen = [best_main] + [i for i in chosen if i != best_main]
        chosen = chosen[:max_key_points]

    # Order chosen by document flow
    chosen_sorted = sorted(chosen, key=lambda i: (sentences[i].page_no, sentences[i].sent_index))

    key_points: List[str] = []
    key_points_cited: List[Dict[str, Any]] = []
    for i in chosen_sorted:
        txt = sentences[i].text.strip()
        if len(txt) > (220 if not cjk else 160):
            txt = txt[: (220 if not cjk else 160)].rstrip() + "..."
        key_points.append(txt)
        key_points_cited.append({"text": txt, "page_no": sentences[i].page_no})

    # Topic outline (optional): pick one representative per cluster
    topics: List[Dict[str, Any]] = []
    if labels:
        by_lab: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            by_lab.setdefault(int(lab), []).append(i)

        # rank clusters by total score mass
        cluster_rank = sorted(
            by_lab.keys(),
            key=lambda lab: sum(scores[i] for i in by_lab[lab]),
            reverse=True,
        )
        for lab in cluster_rank[:6]:
            idxs = by_lab[lab]
            # best sentence in cluster
            best_i = max(idxs, key=lambda i: scores[i])
            name = topic_names.get(int(lab)) or f"Topic {int(lab)+1}"
            topics.append(
                {
                    "topic_id": int(lab),
                    "label": name,
                    "representative": sentences[best_i].text.strip(),
                    "pages": sorted({sentences[i].page_no for i in idxs}),
                }
            )

    # Concepts (shared foundation for flashcards/quiz/knowledge map)
    concepts = extract_concepts(page_texts, sentences, query=query, cjk=cjk, max_concepts=18)

    # Final human-readable summary string
    if cjk:
        summary_text = f"核心内容：{main_idea}"
        if why:
            summary_text += f"\n\n为什么重要：{why}"
        if key_points:
            summary_text += "\n\n要点：\n" + "\n".join(f"- {kp}" for kp in key_points[:max_key_points])
    else:
        summary_text = f"Main idea: {main_idea}"
        if why:
            summary_text += f"\n\nWhy it matters: {why}"
        if key_points:
            summary_text += "\n\nKey points:\n" + "\n".join(f"- {kp}" for kp in key_points[:max_key_points])

    return {
        "language": "cjk" if cjk else "en",
        "pages": page_nos,
        "main_idea": main_idea,
        "why_it_matters": why,
        "topics": topics,
        "summary": summary_text.strip(),
        "key_points": key_points,
        "key_points_cited": key_points_cited,
        "concepts": concepts,
        # Attach small debug slice (safe)
        "debug": {
            "num_pages": len(page_nos),
            "num_sentences": len(sentences),
        },
    }

# ------------------------------------------------------------
# Flashcards + quiz generation from concepts
# ------------------------------------------------------------

def _seed_from(doc_id: str, mode: str, query: str) -> int:
    raw = f"{doc_id}|{mode}|{query}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)

def generate_flashcards_from_concepts(
    concepts: Sequence[Dict[str, Any]],
    *,
    cjk: bool,
    max_cards: int = 10,
) -> Dict[str, Any]:
    cards: List[Dict[str, Any]] = []
    for i, c in enumerate(concepts[:max_cards], start=1):
        term = str(c.get("term", "") or "").strip()
        definition = str(c.get("definition", "") or "").strip()
        page_no = int(c.get("page_no", 0) or 0)
        if not term or not definition:
            continue
        front = f"什么是「{term}」？" if cjk else f"What is {term}?"
        cards.append({"id": str(i), "front": front, "back": definition, "page_no": page_no})
    return {"cards": cards}

def generate_quiz_from_concepts(
    concepts: Sequence[Dict[str, Any]],
    *,
    doc_id: str,
    query: str,
    cjk: bool,
    max_questions: int = 5,
) -> Dict[str, Any]:
    concepts = [c for c in concepts if (c.get("term") and c.get("definition"))]
    if len(concepts) < 4:
        return {"quiz": []}

    rng = random.Random(_seed_from(doc_id, "quiz", query))

    # Choose question targets: prefer earlier concepts (higher salience)
    targets = list(concepts[: min(8, len(concepts))])
    rng.shuffle(targets)
    targets = targets[: min(max_questions, len(targets))]

    pool_terms = [str(c.get("term")) for c in concepts]
    quiz: List[Dict[str, Any]] = []

    labels = ["A", "B", "C", "D"]

    for qi, target in enumerate(targets, start=1):
        term = str(target.get("term", "")).strip()
        definition = str(target.get("definition", "")).strip()
        page_no = int(target.get("page_no", 0) or 0)
        if not term or not definition:
            continue

        # Cloze if possible
        cloze = definition
        if cjk:
            if term in cloze:
                cloze = cloze.replace(term, "____")
        else:
            patt = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term.lower())}(?![A-Za-z0-9_])", flags=re.IGNORECASE)
            cloze2, n = patt.subn("____", cloze, count=1)
            if n > 0:
                cloze = cloze2

        if "____" in cloze:
            question = f"填空题：{cloze}" if cjk else f"Fill in the blank: {cloze}"
        else:
            question = (
                f"以下哪一个术语最符合描述：{definition}"
                if cjk
                else f"Which term best matches this description: {definition}"
            )

        distractors = [t for t in pool_terms if t and t != term]
        # simple de-dup & avoid substring overlaps (English)
        if not cjk:
            tl = term.lower()
            distractors = [t for t in distractors if t.lower() not in tl and tl not in t.lower() and not _is_bad_term(t)]
        rng.shuffle(distractors)

        options = [term] + distractors[:3]
        if len(options) < 4:
            # pad deterministically
            for t in pool_terms:
                if t != term and t not in options:
                    options.append(t)
                if len(options) >= 4:
                    break

        options = options[:4]
        rng.shuffle(options)

        choices = [f"{labels[i]}. {options[i]}" for i in range(4)]
        answer = labels[options.index(term)]

        quiz.append(
            {
                "id": f"q{qi}",
                "question": question,
                "choices": choices,
                "answer": answer,
                "page_no": page_no,
            }
        )

    return {"quiz": quiz}

# ------------------------------------------------------------
# Knowledge map from chunks (doc-level)
# ------------------------------------------------------------

def build_concept_graph_from_chunks(
    doc_id: str,
    chunks: Sequence[Any],
    *,
    max_nodes: int = 60,
    max_edges: int = 200,
) -> Dict[str, Any]:
    """
    Build concept graph from all chunks in a doc.

    Key idea:
      - nodes are *concepts* extracted with term hygiene
      - edges come from co-occurrence of concepts in the same chunk

    Output:
      { doc_id, nodes:[{id,label,pages,chunk_ids?}], edges:[{source,target,type,weight,pages}] }
    """
    chunks = list(chunks or [])
    texts = [str(getattr(c, "text", "") or "") for c in chunks if str(getattr(c, "text", "") or "").strip()]
    if not texts:
        return {"doc_id": doc_id, "nodes": [], "edges": []}

    # Build stitched pseudo-pages from chunks (for more stable concept extraction)
    # Group by page_no
    page_map: Dict[int, List[Any]] = {}
    for c in chunks:
        p = int(getattr(c, "page_no", 0) or 0)
        if p <= 0:
            continue
        page_map.setdefault(p, []).append(c)

    page_texts: List[str] = []
    sentence_objs: List[Sentence] = []
    cjk = is_cjk_heavy(texts)

    for p in sorted(page_map.keys()):
        items = sorted(page_map[p], key=lambda x: int(getattr(x, "chunk_index", 0) or 0))
        t = ""
        for it in items:
            t = merge_with_overlap(t, str(getattr(it, "text", "") or ""))
        t = normalize_whitespace(t)
        t = repair_hyphenation(t)
        if not t:
            continue
        page_texts.append(t)
        sents = split_sentences(t)
        for i, s in enumerate(sents):
            sentence_objs.append(Sentence(text=s, page_no=p, sent_index=i))

    # Extract concepts from the entire doc (no query)
    concepts = extract_concepts(page_texts, sentence_objs, query="", cjk=cjk, max_concepts=max(18, min(2 * max_nodes, 120)))

    # Keep top nodes
    concepts = concepts[:max_nodes]

    # Node presence (pages, chunk_ids)
    node_pages: Dict[str, set] = {}
    node_chunks: Dict[str, set] = {}
    id_to_label: Dict[str, str] = {}

    # Precompile matchers
    matchers: List[Tuple[str, str, Any]] = []
    for c in concepts:
        term = str(c.get("term", "") or "").strip()
        cid = str(c.get("id", "") or "").strip()
        if not term or not cid:
            continue
        if _is_bad_term(term):
            continue
        id_to_label[cid] = term
        node_pages.setdefault(cid, set())
        node_chunks.setdefault(cid, set())
        if cjk or has_cjk(term):
            matchers.append((term, cid, None))
        else:
            phrase = re.sub(r"\s+", " ", term.lower()).strip()
            parts = [p for p in phrase.split(" ") if p]
            body = r"\s+".join(re.escape(p) for p in parts)
            patt = re.compile(rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])", flags=re.IGNORECASE)
            matchers.append((term, cid, patt))

    # Edge weights
    from collections import Counter, defaultdict
    edge_weight = Counter()
    edge_pages = defaultdict(set)

    # Cap concepts per chunk to avoid dense cliques
    max_terms_per_chunk = 10

    for c in chunks:
        text = str(getattr(c, "text", "") or "")
        if not text.strip():
            continue
        p = int(getattr(c, "page_no", 0) or 0)
        cid_chunk = str(getattr(c, "chunk_id", "") or "")

        present: List[str] = []
        if cjk:
            for term, node_id, _ in matchers:
                if term and term in text:
                    present.append(node_id)
                    node_pages[node_id].add(p)
                    if cid_chunk:
                        node_chunks[node_id].add(cid_chunk)
                    if len(present) >= max_terms_per_chunk:
                        break
        else:
            text_norm = " " + re.sub(r"\s+", " ", text.lower()).strip() + " "
            for _, node_id, patt in matchers:
                if patt and patt.search(text_norm):
                    present.append(node_id)
                    node_pages[node_id].add(p)
                    if cid_chunk:
                        node_chunks[node_id].add(cid_chunk)
                    if len(present) >= max_terms_per_chunk:
                        break

        # De-dupe present nodes
        present_u = []
        seen = set()
        for nid in present:
            if nid in seen:
                continue
            seen.add(nid)
            present_u.append(nid)

        # Co-occurrence edges
        for i in range(len(present_u)):
            for j in range(i + 1, len(present_u)):
                a, b = present_u[i], present_u[j]
                key = (a, b) if a < b else (b, a)
                edge_weight[key] += 1
                edge_pages[key].add(p)

    # Build nodes
    nodes: List[Dict[str, Any]] = []
    for c in concepts:
        nid = str(c.get("id", "") or "")
        if not nid:
            continue
        pages = sorted(node_pages.get(nid, set()))
        if not pages:
            continue
        nodes.append(
            {
                "id": nid,
                "label": id_to_label.get(nid, nid),
                "pages": pages,
                "chunk_ids": sorted(node_chunks.get(nid, set())),
            }
        )

    # Build edges: filter weak edges if possible
    edges: List[Dict[str, Any]] = []
    ranked = edge_weight.most_common()
    strong = [(pair, w) for pair, w in ranked if w >= 2]
    use = strong if strong else ranked

    for (a, b), w in use[: max_edges]:
        edges.append(
            {
                "source": a,
                "target": b,
                "type": "related",
                "weight": float(w),
                "pages": sorted(edge_pages[(a, b)]),
            }
        )

    return {"doc_id": doc_id, "nodes": nodes, "edges": edges}
