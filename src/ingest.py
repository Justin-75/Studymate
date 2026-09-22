# src/ingest.py
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, List, Tuple

from src.db.schema import init_db
from src.db.repository import Repository
from src.db.models import Page, IngestResult
from src.pdf.extractor import extract_pages_text_pymupdf
from src.pdf.cleaner import clean_page_text, remove_repeated_headers_footers
from src.pdf.chunker import chunk_pages
from src.pdf.ocr_fallback import needs_ocr, extract_pages_text_ocr, normalize_ocr_lang


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../src -> project root


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def sha256_file(file_path: str | Path, buf_size: int = 1024 * 1024) -> str:
    fp = Path(file_path)
    h = hashlib.sha256()
    with fp.open("rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ingest_pdf(
    file_path: str,
    db_path: str | Path = "Data/Database/app.db",
    *,
    ocr_enabled: bool = True,
    # Avoid mixed-language OCR by default; chi_sim produces better Chinese OCR.
    ocr_lang: str = "chi_sim",
    ocr_dpi: int = 300,
    ocr_psm: int = 3,
    ocr_preprocess: bool = False,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    max_pages: Optional[int] = None,
) -> IngestResult:
    """
    End-to-end ingestion (crash-safe):
      - init DB
      - upsert document by file hash
      - extract pages (PyMuPDF)
      - OCR fallback if needed (optional)
      - clean per page
      - remove repeated headers/footers across pages
      - chunk per page
      - ATOMICALLY replace pages + chunks in SQLite
    """
    pdf_path = _resolve_path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    db_file = _resolve_path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # 1) init db + repo
    init_db(str(db_file))
    repo = Repository(str(db_file))

    # 2) doc dedup (hash)
    file_hash = sha256_file(pdf_path)
    doc = repo.upsert_document(filename=pdf_path.name, file_hash=file_hash)

    # 3) extract
    pages_raw: List[Tuple[int, str]] = extract_pages_text_pymupdf(str(pdf_path))
    if max_pages is not None:
        pages_raw = pages_raw[: max_pages]

    # 4) OCR decision + fallback
    ocr_required = needs_ocr(pages_raw)
    used_ocr = False

    # Normalize to a single language code (prevents accidental "eng+chi_sim").
    ocr_lang = normalize_ocr_lang(ocr_lang)

    if ocr_required and ocr_enabled:
        # Avoid mixed-language OCR ("eng+chi_sim" etc.)
        ocr_lang = normalize_ocr_lang(ocr_lang)
        pages_raw = extract_pages_text_ocr(
            str(pdf_path),
            dpi=ocr_dpi,
            lang=ocr_lang,
            psm=ocr_psm,
            use_preprocess=ocr_preprocess,
        )
        if max_pages is not None:
            pages_raw = pages_raw[: max_pages]
        used_ocr = True

    # 5) clean per page
    pages_clean: List[Tuple[int, str]] = [(pno, clean_page_text(txt)) for pno, txt in pages_raw]

    # 5.5) remove repeated headers/footers (major retrieval quality improvement)
    pages_clean = remove_repeated_headers_footers(
        pages_clean,
        top_n=2,
        bottom_n=2,
        min_ratio=0.6,
        min_len=3,
    )

    # 6) build Page rows
    page_rows: List[Page] = [
        Page(
            doc_id=doc.doc_id,
            page_no=pno,
            text_raw=(pages_raw[i][1] or ""),
            text_clean=pages_clean[i][1] or "",
        )
        for i, (pno, _) in enumerate(pages_clean)
    ]

    # 7) chunk + build Chunk rows
    chunks = chunk_pages(
        doc_id=doc.doc_id,
        pages=pages_clean,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    # 8) ATOMIC replace (fixes corrupted partial-ingest states)
    repo.replace_doc_content_atomic(doc.doc_id, page_rows, chunks)

    return IngestResult(
        doc_id=doc.doc_id,
        filename=pdf_path.name,
        num_pages=len(pages_raw),
        num_chunks=len(chunks),
        used_ocr=used_ocr,
        ocr_required=ocr_required,
    )
