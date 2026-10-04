# src/retrieval/hybrid.py
"""
Hybrid retrieval: BM25 + BGE-M3 dense, merged with Reciprocal Rank Fusion,
then reordered by a cross-encoder reranker.

This is the ONE retrieval pipeline. The app (server/http_server.py) and the
benchmark (scripts/eval_retrieval.py) both import it from here, so the numbers
in the results table describe exactly what users get.

    hybrid_search(repo, doc_id, query, top_k)   -> List[SearchHit]   (used by the app)
    hybrid_rrf_rerank(repo, doc_id, query, k)   -> List[Hit]         (used by the eval)
    warm_up(repo, doc_id)                       build both indexes right after ingest
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.db.models import SearchHit
from src.retrieval import bm25_index, dense_index
from src.retrieval.reranker import rerank

Hit = Tuple[str, int, float, str]   # (chunk_id, page_no, score, text)

RRF_K = 60                      # the constant from the paper; bigger k = ranks matter less, agreement matters more
CANDIDATES_PER_RETRIEVER = 50   # how many chunks BM25 and dense each contribute before fusion
RERANK_POOL = 30                # how many fused chunks the reranker re-scores


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


def _to_hits(repo, scored: List[Tuple[str, float]]) -> List[Hit]:
    """(chunk_id, score) pairs -> Hit tuples, looking up page and text in the DB, order kept."""
    chunk_map = {c.chunk_id: c for c in repo.get_chunks_by_ids([cid for cid, _ in scored])}
    return [(cid, chunk_map[cid].page_no, s, chunk_map[cid].text) for cid, s in scored if cid in chunk_map]


def bm25_hits(repo, doc_id: str, q: str, k: int) -> List[Hit]:
    # BM25 (jieba tokens) scores only: no threshold, no MMR, no page cap.
    return _to_hits(repo, bm25_index.get_index(repo, doc_id).search(q, top_k=k))


def dense_hits(repo, doc_id: str, q: str, k: int) -> List[Hit]:
    # BGE-M3 dense vectors + FAISS exact search. Each document is encoded once,
    # saved in Data/Cache/bge_m3/<doc_id>/ and reused.
    return _to_hits(repo, dense_index.search(repo, doc_id, q, top_k=k))


def hybrid_rrf(repo, doc_id: str, q: str, k: int, n: int = CANDIDATES_PER_RETRIEVER) -> List[Hit]:
    # BM25 + BGE-M3 dense, merged by Reciprocal Rank Fusion (ranks only, no scores)
    bm25_hit = bm25_hits(repo, doc_id, q, n)    # already sorted best first, no need to re-sort
    dense_hit = dense_hits(repo, doc_id, q, n)
    fuse = rrf_fuse([[h[0] for h in bm25_hit], [h[0] for h in dense_hit]])
    lookup = {h[0]: h for h in bm25_hit + dense_hit}   # chunk_id -> (cid, page, score, text)
    return [(cid, lookup[cid][1], score, lookup[cid][3]) for cid, score in fuse[:k]]


def hybrid_rrf_rerank(repo, doc_id: str, q: str, k: int, pool: int = RERANK_POOL) -> List[Hit]:
    candidates = hybrid_rrf(repo, doc_id, q, pool)   # stage 1: candidate pool
    return rerank(q, candidates, k)                 # stage 2: reorder with the cross-encoder


def hybrid_search(repo, doc_id: str, query: str, top_k: int = 8) -> List[SearchHit]:
    """The app's retrieval: hybrid RRF + reranker, returned as SearchHit objects."""
    query = (query or "").strip()
    if not query or top_k <= 0:
        return []
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
    """Build (or load) the BM25 and dense indexes now, so the first question isn't slow."""
    bm25_index.get_index(repo, doc_id)
    dense_index.get_index(repo, doc_id)


if __name__ == "__main__":
    # Toy test from the hand calculation. Expected order: A, C, B, D
    for cid, s in rrf_fuse([["A", "B", "C"], ["C", "A", "D"]]):
        print(cid, round(s, 5))
