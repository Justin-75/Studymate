# src/db/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

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

