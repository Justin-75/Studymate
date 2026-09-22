from __future__ import annotations

from pathlib import Path
import sqlite3

from src.db.schema import init_db
from src.db.repository import Repository
from src.db.models import Page, Chunk


# ✅ Put the DB file INSIDE your Database folder
DB_PATH = "Data/Database/smoke_test.db"


def main() -> None:
    db_file = Path(DB_PATH)

    # Ensure folder exists
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Clean previous test DB
    if db_file.exists():
        db_file.unlink()

    print("[0] DB file will be created at:", db_file)

    print("[1] init_db...")
    init_db(DB_PATH)
    assert db_file.exists(), "DB file was not created. Check folder permissions/path."

    repo = Repository(DB_PATH)

    print("[2] upsert_document + get_document + dedup...")
    doc1 = repo.upsert_document(filename="demo.pdf", file_hash="hash-abc")
    assert doc1.doc_id, "doc_id missing"
    assert doc1.file_hash == "hash-abc"

    fetched = repo.get_document(doc1.doc_id)
    assert fetched is not None, "get_document returned None"
    assert fetched.doc_id == doc1.doc_id

    # Dedup check
    doc2 = repo.upsert_document(filename="demo_renamed.pdf", file_hash="hash-abc")
    assert doc2.doc_id == doc1.doc_id, "dedup failed: doc_id changed for same file_hash"

    print("    OK: doc_id =", doc1.doc_id)

    print("[3] insert_pages + get_pages ordering...")
    pages = [
        Page(doc_id=doc1.doc_id, page_no=2, text_raw="RAW P2", text_clean="CLEAN P2"),
        Page(doc_id=doc1.doc_id, page_no=1, text_raw="RAW P1", text_clean="CLEAN P1"),
        Page(doc_id=doc1.doc_id, page_no=3, text_raw="RAW P3", text_clean="CLEAN P3"),
    ]
    repo.insert_pages(pages)

    got_pages = repo.get_pages(doc1.doc_id)
    assert [p.page_no for p in got_pages] == [1, 2, 3], "pages not ordered by page_no"
    assert got_pages[0].text_clean == "CLEAN P1"
    print("    OK: pages stored/read")

    print("[4] insert_chunks + get_chunks ordering...")
    chunks = [
        Chunk(chunk_id="c1", doc_id=doc1.doc_id, page_no=1, chunk_index=0, text="chunk p1-0"),
        Chunk(chunk_id="c2", doc_id=doc1.doc_id, page_no=1, chunk_index=1, text="chunk p1-1"),
        Chunk(chunk_id="c3", doc_id=doc1.doc_id, page_no=2, chunk_index=0, text="chunk p2-0"),
    ]
    repo.insert_chunks(chunks)

    got_chunks = repo.get_chunks(doc1.doc_id)
    assert [c.chunk_id for c in got_chunks] == ["c1", "c2", "c3"], "chunks not ordered correctly"
    print("    OK: chunks stored/read")

    print("[5] get_chunks_by_ids preserves input order...")
    ranked_ids = ["c3", "c1", "c2"]
    ranked_chunks = repo.get_chunks_by_ids(ranked_ids)
    assert [c.chunk_id for c in ranked_chunks] == ranked_ids, "get_chunks_by_ids order not preserved"
    print("    OK: ranking order preserved")

    print("[6] Foreign key cascade delete test...")
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc1.doc_id,))
        conn.commit()
    finally:
        conn.close()

    assert repo.get_document(doc1.doc_id) is None, "document not deleted"
    assert repo.get_pages(doc1.doc_id) == [], "pages not cascaded deleted"
    assert repo.get_chunks(doc1.doc_id) == [], "chunks not cascaded deleted"
    print("    OK: cascade delete works")

    print("\n✅ DB smoke test PASSED")
    print("DB created at:", db_file)


if __name__ == "__main__":
    main()
