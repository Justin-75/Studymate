# src/retrieval/search.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional

import numpy as np

from src.db.repository import Repository
from src.db.models import SearchHit
from src.retrieval.tfidf_index import TfidfIndex


def build_index_for_doc(repo: Repository, index: TfidfIndex, doc_id: str) -> None:
    """
    Build TF-IDF index from chunks stored in SQLite for this doc_id.
    """
    chunks = repo.get_chunks(doc_id)
    if not chunks:
        raise ValueError(f"No chunks found in DB for doc_id={doc_id}. Ingest first.")

    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    index.build_for_doc(doc_id, chunk_ids, texts)


def _mmr_rerank(
    cand_ids: List[str],
    cand_scores: np.ndarray,
    cand_matrix,
    *,
    top_k: int,
    lambda_mult: float = 0.75,
) -> List[str]:
    """
    Maximal Marginal Relevance selection.

    Assumptions:
      - cand_scores are cosine similarities query<->chunk (higher better)
      - cand_matrix rows are L2-normalized TF-IDF vectors (same as index build)
      - similarity between chunks = dot product

    Returns: selected chunk_ids in selection order
    """
    if top_k <= 0 or not cand_ids:
        return []

    n = len(cand_ids)
    top_k = min(top_k, n)

    rel = np.asarray(cand_scores, dtype=float)
    # All candidate<->candidate similarities in one sparse product (n is small, ~50),
    # instead of one sparse dot product per (candidate, selected) pair.
    sims = (cand_matrix @ cand_matrix.T).toarray()

    # Start from best query score
    first = int(np.argmax(rel))
    selected: List[int] = [first]
    is_selected = np.zeros(n, dtype=bool)
    is_selected[first] = True
    # diversity penalty: max similarity of each candidate to any selected one
    max_sim = np.maximum(sims[:, first], 0.0)

    # Greedy MMR
    while len(selected) < top_k:
        mmr_val = lambda_mult * rel - (1.0 - lambda_mult) * max_sim
        mmr_val[is_selected] = -np.inf
        best_idx = int(np.argmax(mmr_val))  # first max wins ties, same as before
        if is_selected[best_idx]:
            break

        selected.append(best_idx)
        is_selected[best_idx] = True
        max_sim = np.maximum(max_sim, sims[:, best_idx])

    return [cand_ids[i] for i in selected]


def _score_threshold_for_mode(mode: str) -> float:
    """
    TF-IDF cosine scores vary by vectorizer mode.
    These defaults are conservative (quality-first).
    """
    # char n-gram scores are usually smaller
    if mode == "char":
        return 0.03
    if mode == "word":
        return 0.08
    return 0.05


def search_chunks(
    repo: Repository,
    index: TfidfIndex,
    doc_id: str,
    query: str,
    top_k: int = 8,
    *,
    auto_build: bool = True,
    fetch_k_multiplier: int = 5,
    mmr_lambda: float = 0.75,
    max_per_page: int = 2,
    score_threshold: Optional[float] = None,
) -> List[SearchHit]:
    """
    High-quality retrieval for a doc_id:
      1) ensure index exists
      2) fetch_k = top_k * fetch_k_multiplier candidates from TF-IDF
      3) apply score threshold
      4) MMR rerank for diversity
      5) page balancing (max_per_page)
      6) return SearchHit list sorted by final selection order

    Notes:
      - No neighbor expansion here (keeps it safe / no DB API changes).
      - Works with your current schema + repository.
    """
    query = (query or "").strip()
    if not query or top_k <= 0:
        return []

    if auto_build and not index.index_exists(doc_id):
        build_index_for_doc(repo, index, doc_id)

    meta = index.get_meta(doc_id)
    mode = (meta.get("mode") or "auto").strip().lower()
    thr = _score_threshold_for_mode(mode) if score_threshold is None else float(score_threshold)

    # --- Step 1: score all chunks and get top fetch_k indices ---
    scores, chunk_ids_all = index.score_query(doc_id, query)
    if scores.size == 0 or not chunk_ids_all:
        return []

    fetch_k = min(len(chunk_ids_all), max(top_k * fetch_k_multiplier, top_k))

    # take top fetch_k by score
    if fetch_k == scores.size:
        cand_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(-scores, fetch_k - 1)[:fetch_k]
        cand_idx = top_idx[np.argsort(-scores[top_idx])]

    # filter by threshold early
    cand_idx = [int(i) for i in cand_idx if float(scores[int(i)]) >= thr]
    if not cand_idx:
        return []

    cand_ids = [chunk_ids_all[i] for i in cand_idx]
    cand_scores = scores[cand_idx]

    # --- Step 2: load artifacts and slice candidate vectors ---
    vectorizer, matrix, chunk_ids_loaded = index.load_artifacts(doc_id)

    # Safety check: chunk_ids_loaded should match chunk_ids_all order
    # If not, build index->row map
    if chunk_ids_loaded != chunk_ids_all:
        row_map: Dict[str, int] = {cid: i for i, cid in enumerate(chunk_ids_loaded)}
        cand_rows = [row_map[cid] for cid in cand_ids if cid in row_map]
        cand_ids = [cid for cid in cand_ids if cid in row_map]
        cand_scores = np.array([float(scores[row_map[cid]]) for cid in cand_ids], dtype=float)
    else:
        cand_rows = cand_idx

    cand_matrix = matrix[cand_rows]  # sparse rows

    # --- Step 3: MMR selection (diversity) ---
    mmr_order_ids = _mmr_rerank(
        cand_ids,
        cand_scores,
        cand_matrix,
        top_k=min(top_k * 3, len(cand_ids)),  # pick a bit more; page balancing will filter
        lambda_mult=mmr_lambda,
    )
    if not mmr_order_ids:
        return []

    # --- Step 4: fetch chunk metadata from DB ---
    chunks = repo.get_chunks_by_ids(mmr_order_ids)
    chunk_map = {c.chunk_id: c for c in chunks}

    # --- Step 5: page balancing + final selection ---
    selected_ids: List[str] = []
    page_counts: Dict[int, int] = {}

    for cid in mmr_order_ids:
        c = chunk_map.get(cid)
        if c is None:
            continue
        p = int(getattr(c, "page_no", 0) or 0)
        if p <= 0:
            continue
        if max_per_page > 0 and page_counts.get(p, 0) >= max_per_page:
            continue
        selected_ids.append(cid)
        page_counts[p] = page_counts.get(p, 0) + 1
        if len(selected_ids) >= top_k:
            break

    if not selected_ids:
        return []

    # --- Step 6: build SearchHit output in selected order ---
    # Keep score_map from original query scores (stable)
    score_map = {}
    if chunk_ids_loaded == chunk_ids_all:
        # score array aligns with chunk_ids_all
        for cid in selected_ids:
            # find row index
            i = chunk_ids_all.index(cid)
            score_map[cid] = float(scores[i])
    else:
        # use row_map
        row_map = {cid: i for i, cid in enumerate(chunk_ids_loaded)}
        for cid in selected_ids:
            ri = row_map.get(cid)
            if ri is not None:
                score_map[cid] = float(scores[ri])

    out: List[SearchHit] = []
    for cid in selected_ids:
        c = chunk_map.get(cid)
        if c is None:
            continue
        out.append(
            SearchHit(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                page_no=c.page_no,
                chunk_index=c.chunk_index,
                score=float(score_map.get(c.chunk_id, 0.0)),
                text=c.text,
            )
        )

    # already in final order
    return out
