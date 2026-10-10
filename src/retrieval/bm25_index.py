# src/retrieval/bm25_index.py
"""
BM25 retriever (Okapi BM25, Stanford IR book §11.4.3).

The same algorithm as src/bm25.py, packaged so other code (eval_retrieval.py,
later the app) can call it:

    PASS 1  build_for_doc(): once per document, collect N, df, tf, L_d, L_ave
    PASS 2  search():        per question, score every chunk and return the top k

Tokenizer: jieba + lowercase + drop pure punctuation, used for BOTH chunks and questions.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple

import jieba

from src.db.models import parse_child_unit

jieba.setLogLevel(60)   # hide jieba's "Building prefix dict..." messages

K1 = 1.5    # how fast repeated words stop adding score 
B = 0.75    # how strongly long chunks are pulled down 
#According to Okapi BM25 Books

def tokenize(text: str) -> List[str]:
    tokens = jieba.lcut(text or "")                    # 1. split into words
    tokens = [t.strip().lower() for t in tokens]       # 2. trim spaces, lowercase English (ReLU == relu)
    return [t for t in tokens                          # 3. keep a token only if it is not empty and
            if t and any(ch.isalnum() for ch in t)]    #    has at least one letter/digit/汉字


class BM25Index:
    """BM25 statistics for ONE document's chunks, kept in memory."""

    def __init__(self, chunk_ids: List[str], texts: List[str], k1: float = K1, b: float = B):
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts length mismatch")
        self.k1 = k1
        self.b = b
        self.chunk_ids = chunk_ids

        # PASS 1: statistics that do not depend on the question
        self.chunk_length: List[int] = []   # number of tokens in each chunk              (L_d)
        self.chunk_tf: List[Counter] = []   # per chunk: times each word appears           (tf_td)
        self.df: Counter = Counter()        # per word: chunks containing it at least once (df_t)
        for text in texts:
            tokens = tokenize(text)
            self.df.update(set(tokens))     # set(): each word counts once per chunk
            self.chunk_tf.append(Counter(tokens))
            self.chunk_length.append(len(tokens))

        self.N = len(chunk_ids)                                             # number of chunks (N)
        self.avg_length = sum(self.chunk_length) / max(1, self.N)           # L_ave

    def score_query(self, query: str) -> List[float]:
        """PASS 2: BM25 score of every chunk for this question (same order as chunk_ids)."""
        q_set = set(tokenize(query))
        scores = []
        for i in range(self.N):                       # every chunk
            score = 0.0
            for t in q_set:                           # every question word
                if self.df[t] == 0:
                    continue                          # word in no chunk: skip (log(N/0) would crash)
                idf = math.log(self.N / self.df[t])
                tf_td = self.chunk_tf[i][t]
                l_d = self.chunk_length[i]
                tf_part = ((self.k1 + 1) * tf_td) / (
                    self.k1 * ((1 - self.b) + (self.b * l_d / self.avg_length)) + tf_td #BM25 Formula according to Okapi books 
                )
                score += idf * tf_part
            scores.append(score)
        return scores

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Top-k (chunk_id, score), highest first. Chunks scoring 0 (no shared word) are left out."""
        scores = self.score_query(query)
        order = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunk_ids[i], scores[i]) for i in order if scores[i] > 0]


# One index per doc_id per process: tokenizing a whole book with jieba takes a few
# seconds, so it is built on first use and reused for every later question.
_CACHE: Dict[Tuple[object, str], BM25Index] = {}


def get_index(repo, doc_id: str, unit="chunks") -> BM25Index:
    """unit="chunks": the 512-word chunks; a child set (128, "150o40"): their child chunks (parent-child retrieval)."""
    key = (unit if unit == "chunks" else parse_child_unit(unit), doc_id)
    if key not in _CACHE:
        if unit != "chunks":
            items = [(c.child_id, c.text) for c in repo.get_children(doc_id, *parse_child_unit(unit))]
        else:
            items = [(c.chunk_id, c.text) for c in repo.get_chunks(doc_id)]
        if not items:
            raise ValueError(f"No {unit} found in DB for doc_id={doc_id}. Ingest first.")
        _CACHE[key] = BM25Index([i for i, _ in items], [t for _, t in items])
    return _CACHE[key]


def forget(doc_id: str) -> None:
    """Drop the in-memory indexes (chunks and every child size) of a document whose chunks changed."""
    for key in [k for k in _CACHE if k[1] == doc_id]:
        del _CACHE[key]
