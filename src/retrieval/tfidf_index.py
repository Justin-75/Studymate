# src/retrieval/tfidf_index.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional, Literal, Dict, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../src/retrieval -> project root
TfidfMode = Literal["auto", "word", "char"]

# Loaded (vectorizer, matrix, chunk_ids) per doc directory, shared by every TfidfIndex
# instance in the process. Unpickling a large vectorizer takes ~1-2 s, so reloading it on
# every query dominated search time. Entries are keyed by file mtimes, so a rebuild
# (in this process or another) is picked up automatically.
_ARTIFACT_CACHE: Dict[Path, Tuple[Tuple[int, ...], Any]] = {}
_CACHE_MAX_DOCS = 8


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _cjk_ratio(texts: List[str], sample_chars: int = 20000) -> float:
    s = "".join(texts)[:sample_chars]
    if not s:
        return 0.0
    cjk = sum(1 for ch in s if _is_cjk(ch))
    return cjk / max(1, len(s))


class TfidfIndex:
    """
    Per-document TF-IDF index stored on disk.

    Files per doc_id:
      - vectorizer.joblib
      - matrix.npz
      - chunk_ids.json
      - meta.json (chosen mode + basic stats)

    Notes:
      - TF-IDF vectors are L2-normalized => cosine similarity == dot product.
      - For Chinese-heavy text, word tokenization is weak; use char n-grams.
    """

    def __init__(self, index_root: str | Path = "Data/Cache/tfidf"):
        self.index_root = _resolve_path(index_root)
        self.index_root.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Paths
    # ----------------------------

    def _doc_dir(self, doc_id: str) -> Path:
        d = self.index_root / doc_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _vectorizer_path(self, doc_id: str) -> Path:
        return self._doc_dir(doc_id) / "vectorizer.joblib"

    def _matrix_path(self, doc_id: str) -> Path:
        return self._doc_dir(doc_id) / "matrix.npz"

    def _chunk_ids_path(self, doc_id: str) -> Path:
        return self._doc_dir(doc_id) / "chunk_ids.json"

    def _meta_path(self, doc_id: str) -> Path:
        return self._doc_dir(doc_id) / "meta.json"

    def index_exists(self, doc_id: str) -> bool:
        return (
            self._vectorizer_path(doc_id).exists()
            and self._matrix_path(doc_id).exists()
            and self._chunk_ids_path(doc_id).exists()
        )

    # ----------------------------
    # Build / Load
    # ----------------------------

    def _make_vectorizer(self, texts: List[str], mode: TfidfMode = "auto"):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for TF-IDF. Install: pip install scikit-learn"
            ) from e

        chosen: TfidfMode = mode
        if mode == "auto":
            ratio = _cjk_ratio(texts)
            chosen = "char" if ratio >= 0.15 else "word"

        if chosen == "char":
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                lowercase=False,
                norm="l2",
            )
        else:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                norm="l2",
                ngram_range=(1, 2),
            )

        return vectorizer, chosen

    def build_for_doc(
        self,
        doc_id: str,
        chunk_ids: List[str],
        texts: List[str],
        *,
        mode: TfidfMode = "auto",
    ) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts length mismatch")
        if not texts:
            raise ValueError("No texts provided to build TF-IDF index")

        try:
            import joblib
        except ImportError as e:
            raise ImportError("joblib is required. Install: pip install joblib") from e

        try:
            from scipy import sparse
        except ImportError as e:
            raise ImportError("scipy is required. Install: pip install scipy") from e

        vectorizer, chosen = self._make_vectorizer(texts, mode=mode)
        matrix = vectorizer.fit_transform(texts)  # (n_chunks, n_terms) sparse

        # Persist
        joblib.dump(vectorizer, self._vectorizer_path(doc_id))
        sparse.save_npz(self._matrix_path(doc_id), matrix)

        with self._chunk_ids_path(doc_id).open("w", encoding="utf-8") as f:
            json.dump(chunk_ids, f, ensure_ascii=False)

        meta = {
            "doc_id": doc_id,
            "mode": chosen,
            "num_chunks": int(matrix.shape[0]),
            "num_features": int(matrix.shape[1]),
        }
        with self._meta_path(doc_id).open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        _ARTIFACT_CACHE.pop(self._doc_dir(doc_id), None)

    def _load(self, doc_id: str):
        """
        Load (vectorizer, matrix, chunk_ids), from the in-process cache when the files
        on disk haven't changed since the last load.
        """
        if not self.index_exists(doc_id):
            raise FileNotFoundError(
                f"TF-IDF index not found for doc_id={doc_id}. Build it first."
            )

        files = (self._vectorizer_path(doc_id), self._matrix_path(doc_id), self._chunk_ids_path(doc_id))
        stamp = tuple(p.stat().st_mtime_ns for p in files)
        key = self._doc_dir(doc_id)
        cached = _ARTIFACT_CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]

        try:
            import joblib
        except ImportError as e:
            raise ImportError("joblib is required. Install: pip install joblib") from e

        try:
            from scipy import sparse
        except ImportError as e:
            raise ImportError("scipy is required. Install: pip install scipy") from e

        vectorizer = joblib.load(self._vectorizer_path(doc_id))
        matrix = sparse.load_npz(self._matrix_path(doc_id))

        with self._chunk_ids_path(doc_id).open("r", encoding="utf-8") as f:
            chunk_ids = json.load(f)

        _ARTIFACT_CACHE.pop(key, None)
        _ARTIFACT_CACHE[key] = (stamp, (vectorizer, matrix, chunk_ids))
        while len(_ARTIFACT_CACHE) > _CACHE_MAX_DOCS:
            _ARTIFACT_CACHE.pop(next(iter(_ARTIFACT_CACHE)))  # drop the oldest entry
        return vectorizer, matrix, chunk_ids

    # ----------------------------
    # NEW: Public helpers (safe additions)
    # ----------------------------

    def get_meta(self, doc_id: str) -> Dict[str, Any]:
        """
        Return stored meta.json if present; else minimal fallback.
        """
        fp = self._meta_path(doc_id)
        if fp.exists():
            try:
                return json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"doc_id": doc_id, "mode": "auto"}

    def load_artifacts(self, doc_id: str):
        """
        Public wrapper for loading artifacts. (vectorizer, matrix, chunk_ids)
        """
        return self._load(doc_id)

    def score_query(
        self, doc_id: str, query: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute cosine scores for all chunks for a query.
        Returns: (scores ndarray shape (n_chunks,), chunk_ids list)
        """
        query = (query or "").strip()
        if not query:
            return np.array([], dtype=float), []

        vectorizer, matrix, chunk_ids = self._load(doc_id)
        qv = vectorizer.transform([query])  # (1, n_terms)
        scores = (matrix @ qv.T).toarray().ravel()
        return scores, chunk_ids

    # ----------------------------
    # Search (existing)
    # ----------------------------

    def search(self, doc_id: str, query: str, top_k: int = 8) -> List[Tuple[str, float]]:
        query = (query or "").strip()
        if not query:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        scores, chunk_ids = self.score_query(doc_id, query)
        if scores.size == 0:
            return []

        k = min(top_k, scores.size)
        if k == scores.size:
            idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, k - 1)[:k]
            idx = top_idx[np.argsort(-scores[top_idx])]

        results: List[Tuple[str, float]] = []
        for i in idx:
            s = float(scores[i])
            if s <= 0.0:
                continue
            results.append((chunk_ids[i], s))
        return results
