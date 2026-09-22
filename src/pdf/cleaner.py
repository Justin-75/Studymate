# src/pdf/cleaner.py
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List, Tuple

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

# Common bullet / list markers at line start
# Note: we will apply this conservatively in code (only for short lines)
_BULLET_PREFIX_RE = re.compile(
    r"^\s*([•·●▪◦‣]|(\(\d+\))|(\d+[\.\)]))\s+",
    re.UNICODE,
)

# Join words split by hyphen at line breaks: "exam-\nple" -> "example"
_HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z])-\n\s*([A-Za-z])")

# Lines that are usually just page numbers / separators (to help header/footer removal)
_PAGE_NUMBER_ONLY_RE = re.compile(r"^\s*(page\s*)?\d+\s*$", re.IGNORECASE)
_SEPARATOR_ONLY_RE = re.compile(r"^\s*[-—_=]{3,}\s*$")


def _strip_control_chars(text: str) -> str:
    """Remove most control characters while keeping newlines and tabs."""
    out = []
    for ch in text:
        if ch in ("\n", "\t"):
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        # Categories starting with 'C' are control/surrogate/unassigned
        if cat.startswith("C"):
            continue
        out.append(ch)
    return "".join(out)


def clean_page_text(text: str) -> str:
    """
    Clean a single page of extracted text.

    Safe-by-default cleaning:
    - Unicode normalize (NFKC)
    - Remove URLs
    - Remove control characters
    - Remove soft hyphen (\u00ad)
    - Fix hyphenation across line breaks (exam-\\nple -> example)
    - Conservative bullet/list prefix removal (ONLY on short lines)
    - Normalize whitespace (trim line spaces, collapse excessive blank lines, collapse multi-spaces)
    """
    if not text:
        return ""

    # Normalize line endings + Unicode
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs that often come from hyperlink text
    text = _URL_RE.sub("", text)

    # Remove soft hyphen
    text = text.replace("\u00ad", "")

    # Remove control characters (keep \n and \t)
    text = _strip_control_chars(text)

    # Fix hyphenation across line breaks (English)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    cleaned_lines: List[str] = []
    for line in text.split("\n"):
        # Normalize common whitespace characters
        line = line.replace("\u00a0", " ")  # non-breaking space -> normal space
        line = line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        # Conservative bullet/list removal:
        # - only attempt on relatively short lines (typical bullet points)
        # - avoids damaging long lines that legitimately start with punctuation
        if len(line) <= 220:
            line = _BULLET_PREFIX_RE.sub("", line)

        # Collapse repeated spaces/tabs inside line (helps TF-IDF)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # OCR often inserts whitespace between consecutive Chinese characters
    # (e.g., "微 积 分"), which breaks retrieval by exact matching and by
    # char n-gram TF-IDF. Join CJK characters back together.
    #
    # Only removes whitespace between CJK chars (keeps spaces between
    # English words and around numbers/symbols).
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

    # Collapse 3+ blank lines to max 2 (keeps readability)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_repeated_headers_footers(
    pages: List[Tuple[int, str]],
    top_n: int = 2,
    bottom_n: int = 2,
    min_ratio: float = 0.6,
    min_len: int = 3,
) -> List[Tuple[int, str]]:
    """
    Remove repeated header/footer lines across many pages.

    Args:
        pages: [(page_no, page_text)] usually AFTER basic cleaning.
        top_n: number of top lines per page to consider as header candidates
        bottom_n: number of bottom lines per page to consider as footer candidates
        min_ratio: line must appear in at least this fraction of pages to be removed
        min_len: ignore very short lines

    Returns:
        Same structure [(page_no, cleaned_text)] with repeated header/footer removed.

    Notes:
        - Conservative heuristic.
        - Removes exact repeated lines (case/whitespace normalized).
        - Also treats pure page-number lines / separator lines as removable candidates when repeated.
    """
    if not pages:
        return []

    def norm_line(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    total_pages = len(pages)
    freq = Counter()

    # Collect candidate header/footer lines
    for _, text in pages:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue

        header = lines[:top_n]
        footer = lines[-bottom_n:] if bottom_n > 0 else []

        for ln in header + footer:
            n = norm_line(ln)
            if len(n) < min_len:
                continue

            # Boost candidates that are just page numbers/separators (common noise)
            if _PAGE_NUMBER_ONLY_RE.match(n) or _SEPARATOR_ONLY_RE.match(n):
                freq[n] += 2
            else:
                freq[n] += 1

    # Decide which lines to remove
    threshold = max(1, int(total_pages * min_ratio))
    to_remove = {ln for ln, c in freq.items() if c >= threshold}

    # Remove them
    out: List[Tuple[int, str]] = []
    for page_no, text in pages:
        lines = text.split("\n")
        kept: List[str] = []
        for ln in lines:
            n = norm_line(ln)
            if n and n in to_remove:
                continue
            kept.append(ln)
        out.append((page_no, "\n".join(kept).strip()))

    return out
