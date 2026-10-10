# src/retrieval/hybrid.py
"""
Hybrid retrieval: BM25 + BGE-M3 dense, merged with Reciprocal Rank Fusion,
then reordered by a cross-encoder reranker.

This is the ONE retrieval pipeline. The app (server/http_server.py) and the
benchmark (scripts/eval_retrieval.py) both import it from here, so the numbers
in the results table describe exactly what users get. It searches the chunk texts;
the chunk cards (topic + description) are only extra reference for the LLM afterwards.

Parent-child retrieval (STUDYMATE_RETRIEVAL_UNIT=children, the default):
    1. the 512-word chunks are the parents; each is cut into children of up to CHILD_WORDS words that share
       ~CHILD_OVERLAP words with their neighbour (the app: 150 words, no overlap; chunker.py)
    2. only the children are indexed and embedded (BM25, BGE-M3 dense, RRF)
    3. a query retrieves and reranks the children, then swaps each child for its parent, drops
       duplicate parents (each keeps its best child's rank) and returns the top k parents
STUDYMATE_RETRIEVAL_UNIT=chunks searches the 512-word chunks directly (the old way). A document
without children falls back to chunks (python -m src.ingest --children all).

`unit` in the functions below: "chunks" (the 512-word chunks) or a child set: a size in words, e.g. 128,
or size + overlap, e.g. "150o40" (the benchmark compares several sets, the app uses CHILD_WORDS + CHILD_OVERLAP).

    hybrid_search(repo, doc_id, query, top_k)    -> List[SearchHit]   (used by the app)
    hybrid_rrf_rerank(repo, doc_id, query, k)    -> List[Hit]         (used by the eval)
    parent_child_rerank(repo, doc_id, query, k)  -> List[Hit]         (parents, ranked by their children)
    warm_up(repo, doc_id)                        build the indexes right after ingest
    forget(doc_id)                               drop loaded indexes after a re-ingest
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

from src.db.models import SearchHit, child_unit, parse_child_unit
from src.pdf.chunker import CHILD_OVERLAP, CHILD_WORDS
from src.retrieval import bm25_index, dense_index
from src.retrieval.reranker import rerank

Hit = Tuple[str, int, float, str]   # (chunk_id, page_no, score, text)

RRF_K = 60                      # the constant from the paper; bigger k = ranks matter less, agreement matters more
CANDIDATES_PER_RETRIEVER = 50   # how many chunks BM25 and dense each contribute before fusion
RERANK_POOL = 30                # how many fused chunks (or children) the reranker re-scores
RETRIEVAL_UNIT = os.getenv("STUDYMATE_RETRIEVAL_UNIT", "children")   # "chunks" = search the 512-word chunks


def rrf_fuse(ranked_lists: List[List[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
    """
    ranked_lists: one list of chunk_ids per retriever, each best first
                  e.g. [["A", "B", "C"], ["C", "A", "D"]]
    returns:      [(chunk_id, rrf_score), ...] sorted, best first
    """
    scores: Dict[str, float] = {}
    for one_list in ranked_lists:
        for rank, cid in enumerate (one_list, 1 ):
            scores[cid] = scores.get(cid,0) + 1/(k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _to_hits(repo, scored: List[Tuple[str, float]], unit="chunks") -> List[Hit]:
    """(id, score) pairs -> Hit tuples, looking up page and text in the DB, order kept."""
    ids = [cid for cid, _ in scored]
    if unit != "chunks":
        items = {c.child_id: c for c in repo.get_children_by_ids(ids)}
    else:
        items = {c.chunk_id: c for c in repo.get_chunks_by_ids(ids)}
    return [(cid, items[cid].page_no, s, items[cid].text) for cid, s in scored if cid in items]


def bm25_hits(repo, doc_id: str, q: str, k: int, unit="chunks") -> List[Hit]:
    # BM25 (jieba tokens) scores only: no threshold, no MMR, no page cap.
    return _to_hits(repo, bm25_index.get_index(repo, doc_id, unit).search(q, top_k=k), unit)


def dense_hits(repo, doc_id: str, q: str, k: int, unit="chunks") -> List[Hit]:
    # BGE-M3 dense vectors + FAISS exact search. Each document is encoded once,
    # saved in Data/Cache/bge_m3/<doc_id>/ (children: bge_m3_children/w<size>/) and reused.
    return _to_hits(repo, dense_index.search(repo, doc_id, q, top_k=k, unit=unit), unit)


def hybrid_rrf(repo, doc_id: str, q: str, k: int, n: int = CANDIDATES_PER_RETRIEVER,
               unit="chunks") -> List[Hit]:
    # BM25 + BGE-M3 dense, merged by Reciprocal Rank Fusion (ranks only, no scores)
    bm25_hit = bm25_hits(repo, doc_id, q, n, unit)    # already sorted best first, no need to re-sort
    dense_hit = dense_hits(repo, doc_id, q, n, unit)
    fuse = rrf_fuse([[h[0] for h in bm25_hit], [h[0] for h in dense_hit]])
    lookup = {h[0]: h for h in bm25_hit + dense_hit}   # id -> (id, page, score, text)
    return [(cid, lookup[cid][1], score, lookup[cid][3]) for cid, score in fuse[:k]]


def hybrid_rrf_rerank(repo, doc_id: str, q: str, k: int, pool: int = RERANK_POOL,
                      unit="chunks") -> List[Hit]:
    candidates = hybrid_rrf(repo, doc_id, q, pool, unit=unit)   # stage 1: candidate pool
    return rerank(q, candidates, k)                            # stage 2: reorder with the cross-encoder


def parent_child_rerank(repo, doc_id: str, q: str, k: int, pool: int = RERANK_POOL,
                        unit=child_unit(CHILD_WORDS, CHILD_OVERLAP)) -> List[Hit]:
    """
    Parent-child: rank the children (hybrid RRF pool -> reranker), then return their parent chunks,
    each once, in the order of its best child (with that child's score). Several children of one
    parent in the pool count once, so k parents may need the whole pool.
    """
    children = hybrid_rrf_rerank(repo, doc_id, q, pool, pool=pool, unit=unit)   # the whole pool, reranked
    parent_of = {c.child_id: c.parent_id for c in repo.get_children_by_ids([h[0] for h in children])}
    order: List[str] = []
    best: Dict[str, Tuple[int, float]] = {}
    for cid, page, score, _ in children:
        pid = parent_of.get(cid)
        if pid and pid not in best:
            best[pid] = (page, score)
            order.append(pid)
    order = order[:k]
    parents = {c.chunk_id: c for c in repo.get_chunks_by_ids(order)}
    return [(pid, best[pid][0], best[pid][1], parents[pid].text) for pid in order if pid in parents]


def search_unit(repo, doc_id: str):
    """What the app searches: the CHILD_WORDS/CHILD_OVERLAP child set, or "chunks" (STUDYMATE_RETRIEVAL_UNIT=chunks,
    or the document has no such children)."""
    unit = child_unit(CHILD_WORDS, CHILD_OVERLAP)
    if RETRIEVAL_UNIT == "children" and repo.has_children(doc_id, *parse_child_unit(unit)):
        return unit
    return "chunks"


def hybrid_search(repo, doc_id: str, query: str, top_k: int = 8) -> List[SearchHit]:
    """The app's retrieval: hybrid RRF + reranker (chunks, or children -> parents), as SearchHit objects."""
    query = (query or "").strip()
    if not query or top_k <= 0:
        return []
    unit = search_unit(repo, doc_id)
    if unit != "chunks":
        hits = parent_child_rerank(repo, doc_id, query, top_k, unit=unit)
    else:
        hits = hybrid_rrf_rerank(repo, doc_id, query, top_k)
    chunk_map = {c.chunk_id: c for c in repo.get_chunks_by_ids([h[0] for h in hits])}
    return [
        SearchHit(
            chunk_id=cid,
            doc_id=doc_id,
            page_no=page,
            chunk_index=chunk_map[cid].chunk_index,
            score=float(score),
            text=text,
        )
        for cid, page, score, text in hits
        if cid in chunk_map
    ]


def warm_up(repo, doc_id: str) -> None:
    """Build (or load) the BM25 and dense indexes of what the app searches (the children), so the first question isn't slow."""
    unit = search_unit(repo, doc_id)
    bm25_index.get_index(repo, doc_id, unit)
    dense_index.get_index(repo, doc_id, unit)


def forget(doc_id: str) -> None:
    """Drop the loaded indexes of a document whose chunks changed (re-ingested)."""
    bm25_index.forget(doc_id)
    dense_index.forget(doc_id)


if __name__ == "__main__":
    # Toy test from the hand calculation. Expected order: A, C, B, D
    for cid, s in rrf_fuse([["A", "B", "C"], ["C", "A", "D"]]):
        print(cid, round(s, 5))
