from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
import joblib


def inspect(doc_dir: str, top_terms: int = 8) -> None:
    doc_path = Path(doc_dir)
    vec_path = doc_path / "vectorizer.joblib"
    mat_path = doc_path / "matrix.npz"
    ids_path = doc_path / "chunk_ids.json"

    vectorizer = joblib.load(vec_path)
    matrix = sparse.load_npz(mat_path)
    chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))

    feature_names = vectorizer.get_feature_names_out()

    print("Index dir:", doc_path)
    print("Chunks:", matrix.shape[0])
    print("Features (vocab size):", matrix.shape[1])

    # Show top tf-idf features per chunk
    for i in range(min(3, matrix.shape[0])):  # show first 3 chunks
        row = matrix.getrow(i)
        if row.nnz == 0:
            print(f"\nChunk {i} ({chunk_ids[i]}): [NO FEATURES]")
            continue
        # get indices of highest weights
        idx = row.indices
        data = row.data
        top_idx = idx[np.argsort(-data)[:top_terms]]
        top_words = [feature_names[j] for j in top_idx]
        print(f"\nChunk {i} ({chunk_ids[i]}): top terms -> {top_words}")


if __name__ == "__main__":
    # Example:
    # python -m src.retrieval.inspect_tfidf "Data/Cache/_tfidf_smoke/<doc_id>"
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.retrieval.inspect_tfidf <path_to_doc_index_dir>")
    inspect(sys.argv[1])
