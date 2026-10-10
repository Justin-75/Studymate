# src/db/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple

ISO8601 = str

# ----------------------------
# Core storage models
# ----------------------------

@dataclass(frozen=True)
class Document:
    doc_id: str
    filename: str
    file_hash: str
    created_at: ISO8601


@dataclass(frozen=True)
class Page:
    doc_id: str
    page_no: int  # 1-based
    text_raw: str
    text_clean: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_no: int      # 1-based
    chunk_index: int  # 0-based within page
    text: str


@dataclass(frozen=True)
class ChildChunk:
    """A piece of one chunk (the parent): searched in parent-child retrieval, the parent goes to the LLM."""
    child_id: str
    parent_id: str    # chunks.chunk_id
    doc_id: str
    page_no: int      # same page as the parent
    child_index: int  # 0-based within the parent
    text: str
    child_words: int  # the size this child set was cut to (several sizes can sit side by side)
    child_overlap: int = 0   # words each child shares with the one before it (0 = no overlap)


def child_unit(words: int, overlap: int = 0):
    """Name of a child set: 128 (128 words, no overlap) or "150o40" (150 words, 40 shared with the previous child)."""
    return int(words) if not overlap else f"{int(words)}o{int(overlap)}"


def parse_child_unit(unit) -> Tuple[int, int]:
    """128, "128" -> (128, 0); "150o40" -> (150, 40). The inverse of child_unit()."""
    words, _, overlap = str(unit).partition("o")
    if not words.isdigit() or (overlap and not overlap.isdigit()):
        raise ValueError(f"not a child set: {unit!r} (expected e.g. 128 or 150o40)")
    return int(words), int(overlap or 0)


@dataclass(frozen=True)
class ChunkCard:
    chunk_id: str
    doc_id: str
    topic: str        # what the chunk is about, a few words
    description: str  # 1-2 sentences that represent the chunk

    @property
    def text(self) -> str:
        """What the card indexes search: the topic, then the description."""
        return f"{self.topic}\n{self.description}"


# ----------------------------
# Ingestion output model
# ----------------------------

@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    used_ocr: bool
    ocr_required: bool = False  # True if scanned/empty but OCR not run
    language: Optional[str] = None  # "zh" | "en", from the chunker's detect_lang
    error: Optional[str] = None     # why nothing was stored, e.g. an English scan (OCR only reads Chinese)


# ----------------------------
# Retrieval models
# ----------------------------

@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    page_no: int
    chunk_index: int
    score: float
    text: str


# ----------------------------
# Generation models
# ----------------------------

