from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.YA_Common.utils.logger import get_logger
from tools import YA_MCPServer_Tool  # template decorator

# Make sure we can import your backend package `src/`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- Backend imports ----
from src.db.schema import init_db
from src.db.repository import Repository
from src.ingest import ingest_pdf as ingest_pdf_backend
from src.retrieval.tfidf_index import TfidfIndex
from src.retrieval.search import build_index_for_doc, search_chunks
from src.db.models import GenerateRequest, SearchHit
from src.generation.generator import generate_material
from src.generation.quiz_grader import grade_quiz_flexible as grade_quiz_core
from src.pdf.ocr_fallback import normalize_ocr_lang
from src.knowledge_map import (
    build_knowledge_map,
    save_knowledge_map,
    load_knowledge_map,
)


KM_CACHE_DIR = os.getenv("KM_CACHE_DIR", "Data/Cache/knowledge_maps")
logger = get_logger("StudyPartnerTools")


def _resolve_repo_path(env_key: str, default_rel: str) -> str:
    """Resolve paths relative to template root unless absolute."""
    raw = os.getenv(env_key)
    if raw:
        p = Path(raw)
        return str(p) if p.is_absolute() else str((PROJECT_ROOT / p).resolve())
    return str((PROJECT_ROOT / default_rel).resolve())


def _trim_hits_by_chars(hits: List[SearchHit], max_total_chars: int) -> List[SearchHit]:
    """Keep hits in order until total text length reaches max_total_chars."""
    if max_total_chars <= 0:
        return hits
    kept: List[SearchHit] = []
    total = 0
    for h in hits:
        t = (h.text or "").strip()
        if not t:
            continue
        if total + len(t) > max_total_chars and kept:
            break
        kept.append(h)
        total += len(t)
    return kept


def _hits_debug_view(hits: List[SearchHit], preview_chars: int = 200) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits:
        t = (h.text or "").replace("\n", " ").strip()
        if len(t) > preview_chars:
            t = t[:preview_chars] + "..."
        out.append(
            {
                "page_no": int(getattr(h, "page_no", 0) or 0),
                "score": float(getattr(h, "score", 0.0) or 0.0),
                "preview": t,
            }
        )
    return out


# Defaults (can be set via env.yaml -> environment variables)
DB_PATH = _resolve_repo_path("DB_PATH", "Data/Database/app.db")
TFIDF_CACHE_ROOT = _resolve_repo_path("TFIDF_CACHE_ROOT", "Data/Cache/tfidf")

# Lower default improves output quality for extractive generation
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
# Default to Chinese OCR for this project; avoid mixed-language OCR like "eng+chi_sim".
OCR_LANG = normalize_ocr_lang(os.getenv("OCR_LANG", "chi_sim"))
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_PSM = int(os.getenv("OCR_PSM", "3"))
OCR_PREPROCESS = os.getenv("OCR_PREPROCESS", "false").lower() == "true"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Create reusable objects (repo opens connections per call)
init_db(DB_PATH)
_repo = Repository(DB_PATH)
_index = TfidfIndex(index_root=TFIDF_CACHE_ROOT)


@YA_MCPServer_Tool(
    name="ingest_pdf",
    title="Ingest PDF",
    description="Parse a PDF (with OCR fallback if needed), clean, chunk, store into SQLite, and build TF-IDF index.",
)
async def ingest_pdf(file_path: str) -> Dict[str, Any]:
    """
    Tool: ingest_pdf(file_path)

    Args:
        file_path: Absolute path or path relative to template root.
                  Recommended: "Data/Raw PDF/xxx.pdf"

    Returns:
        {doc_id, filename, num_pages, num_chunks, used_ocr, ocr_required}
    """
    res = ingest_pdf_backend(
        file_path=file_path,
        db_path=DB_PATH,
        ocr_enabled=OCR_ENABLED,
        ocr_lang=OCR_LANG,
        ocr_dpi=OCR_DPI,
        ocr_psm=OCR_PSM,
        ocr_preprocess=OCR_PREPROCESS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        max_pages=None,
    )

    # Build TF-IDF now (so later generate() is fast)
    try:
        build_index_for_doc(_repo, _index, res.doc_id)
    except Exception as e:
        logger.warning(f"TF-IDF build failed for doc_id={res.doc_id}: {e}")

    return asdict(res)


@YA_MCPServer_Tool(
    name="generate",
    title="Generate Study Material",
    description='Generate learning materials using retrieval. mode: "summary" | "flashcards" | "quiz".',
)
async def generate(
    doc_id: str,
    mode: str,
    query: str,
    top_k: Optional[int] = None,
    debug: bool = False,
    max_context_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tool: generate(doc_id, mode, query, top_k=None, debug=False, max_context_chars=None)

    Quality-first behavior:
    - Retrieval is limited (top_k + max_context_chars) to prevent "dump" outputs.
    - User-facing response is CLEAN:
        * content: summary/flashcards/quiz
        * citations: pages only
    - Debug metadata is only included when debug=True.
    """
    mode = (mode or "").strip().lower()
    if mode not in {"summary", "flashcards", "quiz"}:
        raise ValueError('mode must be one of: "summary", "flashcards", "quiz"')

    k = int(top_k) if top_k is not None else TOP_K_DEFAULT
    max_chars = int(max_context_chars) if max_context_chars is not None else MAX_CONTEXT_CHARS

    # Retrieve candidates
    hits = search_chunks(_repo, _index, doc_id, query, top_k=k, auto_build=True) or []
    # Cap total context size (prevents garbage / repetition)
    hits = _trim_hits_by_chars(hits, max_total_chars=max_chars)

    req = GenerateRequest(doc_id=doc_id, mode=mode, query=query, top_k=k)
    result = generate_material(req, hits)
    payload = asdict(result)

    # Enforce clean payload (preference A)
    clean = {
        "doc_id": payload.get("doc_id"),
        "mode": payload.get("mode"),
        "query": payload.get("query"),
        "content": payload.get("content") or {},
        "citations": payload.get("citations") or [],
    }

    if debug:
        clean["debug"] = {
            "requested_top_k": k,
            "applied_max_context_chars": max_chars,
            "num_hits_after_trim": len(hits),
            "hits": _hits_debug_view(hits),
        }

    return clean


@YA_MCPServer_Tool(
    name="grade_quiz",
    title="Grade Quiz",
    description="Grade quiz answers using either quiz_id (recommended) or an inline quiz payload.",
)
async def grade_quiz(
    quiz: Optional[Dict[str, Any]] = None,
    answers: Optional[Dict[str, Any]] = None,
    quiz_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Tool: grade_quiz(quiz_id=..., answers=...)

    Preferred usage:
      - grade_quiz(quiz_id="...", answers={question_id: "A" ...})

    Backward compatible:
      - grade_quiz(quiz=<quiz payload>, answers={...})
    """
    return grade_quiz_core(answers=answers or {}, quiz=quiz, quiz_id=quiz_id)

@YA_MCPServer_Tool(
    name="knowledge_map",
    title="Knowledge Map",
    description="Build (or load) a lightweight concept graph (nodes/edges) from a document's chunks.",
)
async def knowledge_map(
    doc_id: str,
    max_nodes: int = 60,
    max_edges: int = 200,
    query: Optional[str] = None,
    focus_depth: int = 2,
    persist: bool = True,
    force_rebuild: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Tool: knowledge_map(doc_id, max_nodes=60, max_edges=200, persist=True, force_rebuild=False, debug=False)

    Behavior:
      - Loads cached JSON map if persist=True and force_rebuild=False and file exists.
      - Otherwise builds from DB chunks (deterministic).
      - Returns CLEAN output by default (no chunk_ids).
      - If debug=True, includes chunk_ids for nodes.
    """
    if max_nodes <= 0 or max_nodes > 200:
        raise ValueError("max_nodes must be between 1 and 200")
    if max_edges < 0 or max_edges > 1000:
        raise ValueError("max_edges must be between 0 and 1000")

    # 1) Load from cache (optional)
    if (not (query and str(query).strip())) and persist and (not force_rebuild):
        cached = load_knowledge_map(doc_id, KM_CACHE_DIR)
        if cached is not None:
            return cached if debug else _strip_km_debug_fields(cached)

    # 2) Build from DB chunks
    # Requires your global repository instance: _repo = Repository(DB_PATH)
    chunks = _repo.get_chunks(doc_id)
    if not chunks:
        return {"doc_id": doc_id, "nodes": [], "edges": [], "message": "No chunks found for this doc_id."}

    km = build_knowledge_map(doc_id, chunks, max_nodes=max_nodes, max_edges=max_edges, query=(query or None), focus_depth=int(focus_depth))

    # 3) Save (optional)
    if (not (query and str(query).strip())) and persist:
        try:
            save_knowledge_map(doc_id, km, KM_CACHE_DIR)
        except Exception:
            # Don't fail the tool if saving fails; still return the computed map.
            pass

    return km if debug else _strip_km_debug_fields(km)


def _strip_km_debug_fields(km: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove noisy internal fields from the knowledge map for user-facing output.

    We keep the parts a frontend needs:
      - doc_id
      - query / focus / paths (if present)
      - nodes(id,label,pages,evidence)
      - edges(source,target,type,weight,pages)
    """
    nodes_out = []
    for n in km.get("nodes", []) or []:
        nodes_out.append(
            {
                "id": n.get("id"),
                "label": n.get("label"),
                "pages": n.get("pages", []),
                "evidence": n.get("evidence", None),
            }
        )

    edges_out = []
    for e in km.get("edges", []) or []:
        edges_out.append(
            {
                "source": e.get("source"),
                "target": e.get("target"),
                "type": e.get("type", "related"),
                "weight": e.get("weight", 0.0),
                "pages": e.get("pages", []),
            }
        )

    out = {
        "doc_id": km.get("doc_id"),
        "nodes": nodes_out,
        "edges": edges_out,
    }
    # Optional fields from the new knowledge graph builder
    if "query" in km:
        out["query"] = km.get("query")
    if "focus" in km:
        out["focus"] = km.get("focus")
    if "paths" in km:
        out["paths"] = km.get("paths")

    return out
