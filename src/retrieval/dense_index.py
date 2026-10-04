# src/retrieval/dense_index.py
"""
Dense retriever: BGE-M3 vectors searched with FAISS.

Two classes, split by what is heavy and shared vs what is per document:

    Encoder     loads BGE-M3 ONCE for the whole program (2+ GB on the GPU)
                encode_chunks(texts)    -> (N, 1024) float32, each row length 1
                encode_questions(texts) -> (Q, 1024) float32, each row length 1

    DenseIndex  one per document (doc_id); a FAISS index lives inside it
                build(encoder, chunk_ids, texts)   PASS 1: encode every chunk once, save to disk
                load()                             read the saved index back (no model needed)
                search(q_vec, k)                   PASS 2: top-k (chunk_id, score)

Saved per document in Data/Cache/bge_m3/<doc_id>/:
    index.faiss      the chunk vectors (FAISS IndexFlatIP = exact dot-product search)
    chunk_ids.json   chunk ids in the SAME order as the vectors (row i <-> chunk_ids[i])
    meta.json        model, number of chunks, dimension, build time

Model location: the folder in the BGE_M3_PATH environment variable, else E:\\models\\bge-m3,
else the Hugging Face name "BAAI/bge-m3" (download).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "Data" / "Cache" / "bge_m3"

DEFAULT_LOCAL_MODEL = r"E:\models\bge-m3"
MAX_LENGTH = 512      # tokens; chunks are ~800 characters, so 512 is plenty
BATCH_SIZE = 32       # chunks encoded per step; lower it if the GPU runs out of memory


def _model_path() -> str:
    p = os.environ.get("BGE_M3_PATH") or DEFAULT_LOCAL_MODEL
    return p if Path(p).exists() else "BAAI/bge-m3"


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """float32, every row length 1 -> dot product == cosine similarity."""
    vecs = np.asarray(vecs, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)     # (n, 1) so each row divides by its own length
    return vecs / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Encoder: the model, loaded once
# ---------------------------------------------------------------------------
class Encoder:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        import torch
        from FlagEmbedding import BGEM3FlagModel

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_path = model_path or _model_path()

        t0 = time.time()
        self.model = BGEM3FlagModel(
            self.model_path,
            use_fp16=device.startswith("cuda"),   # fp16 only helps (and only works well) on a GPU
            devices=[device],
        )
        print(f"  [bge-m3] loaded from {self.model_path} on {device} in {time.time() - t0:.1f}s")

    def _encode(self, fn, texts: List[str], batch_size: int) -> np.ndarray:
        out = fn(
            texts,
            batch_size=batch_size,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = np.asarray(out["dense_vecs"])
        if vecs.ndim == 1:                   # a single text comes back as (1024,) -> make it (1, 1024)
            vecs = vecs[None, :]
        return _normalize(vecs)

    def encode_chunks(self, texts: List[str]) -> np.ndarray:
        return self._encode(self.model.encode_corpus, list(texts), BATCH_SIZE)       # (N, 1024)

    def encode_questions(self, texts: List[str]) -> np.ndarray:
        return self._encode(self.model.encode_queries, list(texts), BATCH_SIZE)      # (Q, 1024)


_ENCODER: Optional[Encoder] = None


def get_encoder() -> Encoder:
    """The one shared Encoder; created on first use."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = Encoder()
    return _ENCODER


# ---------------------------------------------------------------------------
# DenseIndex: one document's vectors inside FAISS
# ---------------------------------------------------------------------------
class DenseIndex:
    def __init__(self, doc_id: str, cache_root: Path = CACHE_ROOT):
        self.doc_id = doc_id
        self.dir = Path(cache_root) / doc_id
        self.index = None                   # faiss.IndexFlatIP after build() / load()
        self.chunk_ids: List[str] = []

    @property
    def _index_file(self) -> Path:
        return self.dir / "index.faiss"

    @property
    def _ids_file(self) -> Path:
        return self.dir / "chunk_ids.json"

    def exists(self) -> bool:
        return self._index_file.exists() and self._ids_file.exists()

    def build(self, encoder: Encoder, chunk_ids: List[str], texts: List[str]) -> None:
        """PASS 1: encode every chunk once, put the vectors in FAISS, save everything."""
        import faiss

        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts length mismatch")
        t0 = time.time()
        vecs = encoder.encode_chunks(texts)                  # (N, 1024), the slow part
        self.index = faiss.IndexFlatIP(vecs.shape[1])        # Flat = exact, IP = dot product
        self.index.add(vecs)
        self.chunk_ids = list(chunk_ids)

        self.dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_file))
        self._ids_file.write_text(json.dumps(self.chunk_ids), encoding="utf-8")
        meta = {
            "doc_id": self.doc_id, "model": encoder.model_path, "num_chunks": len(self.chunk_ids),
            "dim": int(vecs.shape[1]), "max_length": MAX_LENGTH,
            "build_seconds": round(time.time() - t0, 1),
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  [bge-m3] encoded {len(texts)} chunks of {self.doc_id[:8]} in {meta['build_seconds']}s")

    def load(self) -> None:
        import faiss

        self.index = faiss.read_index(str(self._index_file))
        self.chunk_ids = json.loads(self._ids_file.read_text(encoding="utf-8"))

    def search(self, q_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """PASS 2: top-k (chunk_id, cosine score), best first. q_vec: one normalized (1024,) vector."""
        q = np.asarray(q_vec, dtype="float32").reshape(1, -1)      # FAISS wants (num_questions, dim)
        k = min(top_k, self.index.ntotal)
        scores, positions = self.index.search(q, k)                # both (1, k): scores FIRST, then positions
        return [(self.chunk_ids[p], float(s)) for s, p in zip(scores[0], positions[0]) if p != -1]


# One loaded DenseIndex per doc_id per process.
_INDEXES: Dict[str, DenseIndex] = {}


def get_index(repo, doc_id: str) -> DenseIndex:
    """Load the saved index if it matches the chunks in the DB; otherwise (re)build it."""
    if doc_id in _INDEXES:
        return _INDEXES[doc_id]

    chunks = repo.get_chunks(doc_id)
    if not chunks:
        raise ValueError(f"No chunks found in DB for doc_id={doc_id}. Ingest first.")
    chunk_ids = [c.chunk_id for c in chunks]

    idx = DenseIndex(doc_id)
    if idx.exists():
        idx.load()
    if idx.chunk_ids != chunk_ids:          # nothing saved yet, or the PDF was re-ingested -> rebuild
        idx.build(get_encoder(), chunk_ids, [c.text for c in chunks])
    _INDEXES[doc_id] = idx
    return idx


def search(repo, doc_id: str, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Convenience: encode the question and search one document."""
    q_vec = get_encoder().encode_questions([query])[0]
    return get_index(repo, doc_id).search(q_vec, top_k)
