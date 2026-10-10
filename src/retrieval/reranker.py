# src/retrieval/reranker.py
"""Stage 2: re-score a candidate pool with a cross-encoder (bge-reranker-v2-m3)."""
from typing import List, Tuple

import numpy as np
from FlagEmbedding import FlagReranker

Hit = Tuple[str, int, float, str]   # (chunk_id, page_no, score, text)
MAX_LENGTH = 8192    # tokens per (question, chunk) pair = the model's own limit, so no chunk is cut
BATCH_SIZE = 16      # pairs per step; FlagEmbedding shrinks it by itself if long pairs run out of GPU memory
_reranker = None


def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True,   # fp16 OK on your RTX 5070
                                 max_length=MAX_LENGTH, batch_size=BATCH_SIZE)
    return _reranker


def rerank(q: str, hits: List[Hit], k: int) -> List[Hit]:
    if not hits:
        return []

    pairs = [[q, h[3]] for h in hits]
    scores = get_reranker().compute_score(pairs, normalize=True)
    scores = np.atleast_1d(scores).tolist()   # 1 pair -> float, many -> list; this makes it always a list
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return [(h[0], h[1], float(s), h[3]) for h, s in ranked[:k]]

