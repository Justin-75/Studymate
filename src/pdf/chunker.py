
# src/pdf/chunker.py
from __future__ import annotations

import hashlib
import re
from typing import List, Tuple

from src.db.models import Chunk


# ------------------------------------------------------------
# Sentence helpers (moved here from the old semantic_outline.py, unchanged,
# so chunk boundaries and chunk_ids stay exactly the same as before)
# ------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SOFT_HYPHEN = "\u00ad"
_SENT_SPLIT_RE = re.compile(
    r"(?<=[\.\!\?])\s+|(?<=[。！？])\s+|(?<=[。！？])(?=\S)|\n{2,}"
)


def has_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))


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
    """Fix PDF line-wrap hyphenation: "in- dex" -> "index"."""
    if not text:
        return ""
    return re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)


def normalize_for_sentence_split(text: str) -> str:
    """Single newlines (line-wrap) -> spaces; paragraph breaks kept as blank lines."""
    if not text:
        return ""
    t = normalize_whitespace(text)
    t = repair_hyphenation(t)
    t = re.sub(r"\n{2,}", " <PARA> ", t)
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("<PARA>", "\n\n")
    return t.strip()


def split_sentences(text: str) -> List[str]:
    """Robust(ish) EN/ZH sentence splitter, tolerant to PDF newlines."""
    t = normalize_for_sentence_split(text)
    if not t:
        return []
    out: List[str] = []
    for p in _SENT_SPLIT_RE.split(t):
        s = p.strip()
        if not s:
            continue
        # Filter short Latin fragments (keep short CJK)
        if len(s) < 18 and not has_cjk(s):
            continue
        out.append(s)
    return out


def make_chunk_id(doc_id: str, page_no: int, chunk_index: int) -> str:
    """
    Create a stable chunk_id from (doc_id, page_no, chunk_index).
    We hash it to keep IDs short and uniform.

    Returns:
        chunk_id as a hex string (16 chars).
    """
    raw = f"{doc_id}:{page_no}:{chunk_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _char_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Legacy char-based chunking (kept as fallback).

    NOTE:
      - This can cut words/sentences. Use sentence-aware chunking by default.
    """
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end == n:
            break
        start += step
    return chunks


def _sentence_aware_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Sentence-aware chunking:
      - Split into sentences (robust EN/ZH)
      - Pack sentences into ~chunk_size character chunks
      - Apply overlap in characters by carrying over trailing sentences

    This prevents:
      - "f elements..." (missing first letter from chunk boundary)
      - fragmented sentences used by summarization/flashcards/quiz
    """
    text = normalize_whitespace(text or "")
    text = repair_hyphenation(text)
    if not text:
        return []

    sents = split_sentences(text)
    if len(sents) < 2:
        return _char_chunks(text, chunk_size=chunk_size, overlap=overlap)

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        chunk = " ".join(s.strip() for s in cur if s.strip()).strip()
        if chunk:
            chunks.append(chunk)
        cur = []
        cur_len = 0

    for s in sents:
        s = s.strip()
        if not s:
            continue
        # If a single sentence is huge, hard-split it (rare but possible in PDFs)
        if len(s) > chunk_size * 1.6:
            flush()
            chunks.extend(_char_chunks(s, chunk_size=chunk_size, overlap=max(0, min(overlap, chunk_size - 1))))
            continue

        if cur and (cur_len + 1 + len(s) > chunk_size):
            # finalize current chunk
            flush()
            # overlap: carry over trailing sentences from previous chunk
            if overlap > 0 and chunks:
                prev = chunks[-1]
                # Take last sentences from prev until overlap char budget
                prev_sents = split_sentences(prev)
                carry: List[str] = []
                budget = 0
                for ps in reversed(prev_sents):
                    if budget + len(ps) > overlap and carry:
                        break
                    carry.append(ps)
                    budget += len(ps)
                carry = list(reversed(carry))
                cur = carry[:]
                cur_len = sum(len(x) for x in cur) + max(0, len(cur) - 1)

        cur.append(s)
        cur_len += len(s) + (1 if cur_len > 0 else 0)

    flush()

    # De-dup accidental identical chunks
    dedup: List[str] = []
    seen = set()
    for c in chunks:
        if c in seen:
            continue
        seen.add(c)
        dedup.append(c)

    return dedup


def chunk_pages(
    doc_id: str,
    pages: List[Tuple[int, str]],
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[Chunk]:
    """
    Convert cleaned page texts into Chunk objects.

    Args:
        doc_id: document id
        pages: [(page_no, clean_text)] where page_no is 1-based
        chunk_size: target max characters per chunk (sentence-aware)
        overlap: overlapping characters between consecutive chunks (sentence-aware)

    Returns:
        List[Chunk] in stable order: by page_no asc, then chunk_index asc.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    out: List[Chunk] = []

    pages_sorted = sorted(pages, key=lambda x: x[0])

    for page_no, text in pages_sorted:
        parts = _sentence_aware_chunks(text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk_text in enumerate(parts):
            chunk_id = make_chunk_id(doc_id, page_no, chunk_index)
            out.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_no=page_no,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

    return out
