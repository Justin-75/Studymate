# src/retrieval/smoke_test.py
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.schema import init_db
from src.db.repository import Repository
from src.db.models import Chunk
from src.retrieval.tfidf_index import TfidfIndex
from src.retrieval.search import build_index_for_doc, search_chunks


def main() -> None:
    db_path = PROJECT_ROOT / "Data" / "Database" / "_retrieval_smoke_test.db"
    cache_root = PROJECT_ROOT / "Data" / "Cache" / "_tfidf_smoke"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Recreate DB
    if db_path.exists():
        db_path.unlink()

    print("[1] init_db...")
    init_db(str(db_path))
    repo = Repository(str(db_path))

    print("[2] insert dummy document + chunks...")
    doc = repo.upsert_document(filename="dummy.pdf", file_hash="dummy-hash")

    chunks = [
        Chunk(chunk_id="c1", doc_id=doc.doc_id, page_no=1, chunk_index=0,
              text="Binary search tree supports ordered insert and search."),
        Chunk(chunk_id="c2", doc_id=doc.doc_id, page_no=1, chunk_index=1,
              text="Hash table uses hashing and handles collisions."),
        Chunk(chunk_id="c3", doc_id=doc.doc_id, page_no=2, chunk_index=0,
              text="Time complexity of binary search is O(log n)."),
    ]
    repo.insert_chunks(chunks)

    print("[3] build TF-IDF index...")
    index = TfidfIndex(index_root=str(cache_root))
    build_index_for_doc(repo, index, doc.doc_id)
    assert index.index_exists(doc.doc_id), "Index files were not created."

    print("[4] query retrieval...")
    hits = search_chunks(repo, index, doc.doc_id, query="hash collisions", top_k=3)
    for h in hits:
        print(f"- score={h.score:.4f} page={h.page_no} chunk={h.chunk_id}: {h.text}")

    assert hits, "No hits returned."
    assert hits[0].chunk_id == "c2", "Expected chunk about hash table to rank highest."

    print("\n✅ Retrieval smoke test PASSED")


if __name__ == "__main__":
    main()
