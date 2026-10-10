# src/ingest.py
r"""
PDF ingest as a LangGraph: PDF -> pages -> language -> (OCR) -> clean -> chunks -> SQLite.
Runs at upload, before the ingest graph (src/graphs/ingest_graph.py) builds the section summaries.

START -> extract -> detect_lang --(text layer)--------------------> clean -> chunk --(text)-----> save -> END
                               \--(scanned, zh)--> ocr -----------/               \--(no text)--> END
                               \--(scanned, en)--> ocr_failed -> END

1. extract      text layer per page; OCR needed or not
2. detect_lang  the chunker's detect_lang: on the text layer, or for a scan on chi_sim OCR of sample pages
3. ocr_failed   a scan that is English: error, OCR only reads Chinese (chi_sim)
4. clean        whitespace and noise (cleaner.py), then chunk: whole sentences up to 512 words,
                Moses for English, jieba for Chinese (chunker.py). Each chunk is also split into
                children of up to CHILD_WORDS words (optionally overlapping by CHILD_OVERLAP words):
                retrieval searches the children (src/retrieval/hybrid.py).

Run it:
    python -m src.ingest path/to/file.pdf
    python -m src.ingest --children <doc_id | all> [size] [overlap]   # add a child set to stored documents
                                                                       # (default CHILD_WORDS, CHILD_OVERLAP)
    python -m src.ingest --drop-children <size> [overlap]             # remove one child set (e.g. tried in the benchmark)
"""
from __future__ import annotations

import hashlib
import random
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from src.db.schema import init_db
from src.db.repository import Repository
from src.db.models import ChildChunk, Chunk, Document, Page, IngestResult
from src.pdf.extractor import extract_pages_text_pymupdf
from src.pdf.cleaner import clean_page_text, remove_repeated_headers_footers
from src.pdf.chunker import (
    CHILD_OVERLAP, CHILD_WORDS, MAX_WORDS, MIN_PAGE_CHARS, OVERLAP, SAMPLE_PAGES, Chunker, NoTextLayerError,
    detect_lang,
)
from src.pdf.ocr_fallback import needs_ocr, extract_pages_text_ocr


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../src -> project root
OCR_LANG = {"zh": "chi_sim", "en": "eng"}           # chunker language -> Tesseract language


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


class PdfState(TypedDict, total=False):
    pdf_path: str
    db_path: str
    ocr_enabled: bool
    ocr_dpi: int
    ocr_psm: int
    ocr_preprocess: bool
    max_pages: Optional[int]
    max_words: int
    overlap_sentences: int
    child_words: int
    child_overlap: int
    pages_raw: List[Tuple[int, str]]     # [(page_no, text)], 1-based: text layer, or OCR after the ocr node
    ocr_required: bool
    ocr_pages: Dict[int, str]            # page_no -> OCR text of the sample pages detect_lang already read
    used_ocr: bool
    pages_clean: List[Tuple[int, str]]
    language: Optional[str]              # "zh" | "en" | None (no text found)
    doc: Document
    chunks: List[Chunk]
    children: List[ChildChunk]            # the chunks split again for parent-child retrieval
    result: IngestResult


def _ocr(state: PdfState, lang: str, pages: List[int]) -> Dict[int, str]:
    return dict(extract_pages_text_ocr(
        state["pdf_path"], dpi=state["ocr_dpi"], lang=lang, psm=state["ocr_psm"],
        use_preprocess=state["ocr_preprocess"], pages=pages,
    ))


def _result(state: PdfState, doc_id: str = "", num_chunks: int = 0, **extra) -> IngestResult:
    return IngestResult(
        doc_id=doc_id,
        filename=Path(state["pdf_path"]).name,
        num_pages=len(state["pages_raw"]),
        num_chunks=num_chunks,
        used_ocr=state.get("used_ocr", False),
        ocr_required=state["ocr_required"],
        language=state.get("language"),
        **extra,
    )


# ---------------------------------------------------------------------------
# 1. Extract the text layer; OCR needed or not
# ---------------------------------------------------------------------------
def extract(state: PdfState) -> dict:
    pages_raw = extract_pages_text_pymupdf(state["pdf_path"])
    if state.get("max_pages") is not None:
        pages_raw = pages_raw[: state["max_pages"]]
    return {"pages_raw": pages_raw, "ocr_required": needs_ocr(pages_raw), "used_ocr": False}


# ---------------------------------------------------------------------------
# 2. Language from the chunker's detect_lang; 3. an English scan is an error
# ---------------------------------------------------------------------------
def _scan_ocr_needed(state: PdfState) -> bool:
    return state["ocr_required"] and state["ocr_enabled"]


def detect_language(state: PdfState) -> dict:
    """
    Text layer: detect_lang on its pages. Scan: OCR pages with chi_sim, 5 at a time in a fixed random
    order until 5 have text (blank pages don't vote), and detect_lang on those.
    """
    done: Dict[int, str] = {}
    if _scan_ocr_needed(state):
        order = random.Random(0).sample([p for p, _ in state["pages_raw"]], len(state["pages_raw"]))
        while order and sum(len(clean_page_text(t)) >= MIN_PAGE_CHARS for t in done.values()) < SAMPLE_PAGES:
            batch, order = sorted(order[:SAMPLE_PAGES]), order[SAMPLE_PAGES:]
            done.update(_ocr(state, OCR_LANG["zh"], batch))
        texts = [clean_page_text(t) for t in done.values()]     # chi_sim OCR spaces out 汉字; cleaning rejoins them
    else:
        texts = [t for _, t in state["pages_raw"]]
    try:
        language = detect_lang(texts)
    except NoTextLayerError:
        language = None
    print(f"[ingest] language: {language or 'no text'}" + (f" (OCR of {len(done)} sample pages)" if done else ""))
    return {"language": language, "ocr_pages": done}


def route_after_detect(state: PdfState) -> str:
    if not _scan_ocr_needed(state):
        return "clean"
    return "ocr" if state["language"] == "zh" else "ocr_failed"


def ocr_failed(state: PdfState) -> dict:
    if state["language"] == "en":
        error = ("OCR failed: this is a scanned English document. OCR only reads Chinese (chi_sim), "
                 "so OCR the English document first and upload the searchable PDF.")
    else:
        error = "OCR failed: no text found in this scanned PDF."
    print(f"[ingest] {error}")
    return {"result": _result(state, error=error)}


def ocr(state: PdfState) -> dict:
    done = state["ocr_pages"]
    todo = [p for p, _ in state["pages_raw"] if p not in done]          # the sample pages are read already
    pages = {**done, **_ocr(state, OCR_LANG[state["language"]], todo)}
    return {"pages_raw": sorted(pages.items()), "used_ocr": True}


# ---------------------------------------------------------------------------
# 4. Clean, chunk, save
# ---------------------------------------------------------------------------
def clean(state: PdfState) -> dict:
    pages_clean = [(pno, clean_page_text(txt)) for pno, txt in state["pages_raw"]]
    # remove repeated headers/footers (major retrieval quality improvement)
    pages_clean = remove_repeated_headers_footers(pages_clean, top_n=2, bottom_n=2, min_ratio=0.6, min_len=3)
    return {"pages_clean": pages_clean}


def chunk(state: PdfState) -> dict:
    language = state["language"]
    if language is None:                    # detect_lang found no text on any page
        print("[ingest] no text in this PDF")
        return {"result": _result(state)}
    repo = Repository(state["db_path"])
    doc = repo.create_document(filename=Path(state["pdf_path"]).name, file_hash=sha256_file(state["pdf_path"]))
    with Chunker(state["max_words"], state["overlap_sentences"]) as chunker:
        chunks = chunker.chunk_pages(doc.doc_id, state["pages_clean"], language)
        children = chunker.children(chunks, language, state["child_words"], state["child_overlap"])
    print(f"[ingest] {len(chunks)} chunks, {len(children)} children ({language})")
    return {"doc": doc, "chunks": chunks, "children": children}


def route_after_chunk(state: PdfState) -> str:
    return END if state.get("result") else "save"


def save(state: PdfState) -> dict:
    doc, raw = state["doc"], dict(state["pages_raw"])
    page_rows = [
        Page(doc_id=doc.doc_id, page_no=pno, text_raw=raw.get(pno) or "", text_clean=txt or "")
        for pno, txt in state["pages_clean"]
    ]
    # inserting page + chunk + children with rollback system
    Repository(state["db_path"]).replace_doc_content_atomic(doc.doc_id, page_rows, state["chunks"], state["children"])
    return {"result": _result(state, doc.doc_id, len(state["chunks"]))}


def build_pdf_graph():
    g = StateGraph(PdfState)
    g.add_node("extract", extract)
    g.add_node("detect_lang", detect_language)
    g.add_node("ocr_failed", ocr_failed)
    g.add_node("ocr", ocr)
    g.add_node("clean", clean)
    g.add_node("chunk", chunk)
    g.add_node("save", save)
    g.add_edge(START, "extract")
    g.add_edge("extract", "detect_lang")
    g.add_conditional_edges("detect_lang", route_after_detect, ["clean", "ocr", "ocr_failed"])
    g.add_edge("ocr_failed", END)
    g.add_edge("ocr", "clean")
    g.add_edge("clean", "chunk")
    g.add_conditional_edges("chunk", route_after_chunk, ["save", END])
    g.add_edge("save", END)
    return g.compile()


def ingest_pdf(
    file_path: str,
    db_path: str | Path = "Data/Database/app.db",
    *,
    ocr_enabled: bool = True,
    ocr_dpi: int = 300,
    ocr_psm: int = 3,
    ocr_preprocess: bool = False,
    max_words: int = MAX_WORDS,
    overlap_sentences: int = OVERLAP,
    child_words: int = CHILD_WORDS,
    child_overlap: int = CHILD_OVERLAP,
    max_pages: Optional[int] = None,
) -> IngestResult:
    """
    Run the PDF graph on one file. Crash-safe: pages + chunks + children replace the old ones in one
    transaction. The document row is created only once there is text to chunk; check result.error and
    result.num_chunks (0 = nothing stored).
    """
    pdf_path = _resolve_path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    db_file = _resolve_path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    init_db(str(db_file))

    return build_pdf_graph().invoke({
        "pdf_path": str(pdf_path),
        "db_path": str(db_file),
        "ocr_enabled": ocr_enabled,
        "ocr_dpi": ocr_dpi,
        "ocr_psm": ocr_psm,
        "ocr_preprocess": ocr_preprocess,
        "max_pages": max_pages,
        "max_words": max_words,
        "overlap_sentences": overlap_sentences,
        "child_words": child_words,
        "child_overlap": child_overlap,
    })["result"]


def build_children(doc_id: str, db_path: str | Path = "Data/Database/app.db", child_words: int = CHILD_WORDS,
                   child_overlap: int = CHILD_OVERLAP) -> int:
    """
    Split a stored document's chunks into children, for documents ingested before parent-child existed
    (or to try another child set). Uses the chunks already in the database: no PDF, no LLM.
    """
    db_file = _resolve_path(db_path)
    init_db(str(db_file))
    repo = Repository(str(db_file))
    chunks = repo.get_chunks(doc_id)
    if not chunks:
        raise ValueError(f"No chunks for doc_id={doc_id}. Ingest the PDF first.")
    language = detect_lang([p.text_clean or "" for p in repo.get_pages(doc_id)])
    with Chunker() as chunker:
        children = chunker.children(chunks, language, child_words, child_overlap)
    repo.replace_children(doc_id, children, child_words, child_overlap)
    print(f"[ingest] {doc_id}: {len(chunks)} chunks -> {len(children)} children of <= {child_words} words, "
          f"{child_overlap} overlap ({language})")
    return len(children)


if __name__ == "__main__":
    db = _resolve_path("Data/Database/app.db")
    if len(sys.argv) in (3, 4, 5) and sys.argv[1] == "--children":
        size = int(sys.argv[3]) if len(sys.argv) >= 4 else CHILD_WORDS
        overlap = int(sys.argv[4]) if len(sys.argv) == 5 else CHILD_OVERLAP
        ids = ([d for (d,) in sqlite3.connect(str(db)).execute("SELECT doc_id FROM documents")]
               if sys.argv[2] == "all" else [sys.argv[2]])
        for d in ids:
            build_children(d, db, size, overlap)
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "--drop-children":
        init_db(str(db))
        overlap = int(sys.argv[3]) if len(sys.argv) == 4 else 0
        removed = Repository(str(db)).delete_children(int(sys.argv[2]), overlap)
        print(f"removed {removed} children of size {sys.argv[2]}, overlap {overlap}")
    elif len(sys.argv) == 2:
        print(ingest_pdf(sys.argv[1]))
    else:
        print("usage: python -m src.ingest path/to/file.pdf\n"
              "       python -m src.ingest --children <doc_id | all> [size] [overlap]\n"
              "       python -m src.ingest --drop-children <size> [overlap]")
