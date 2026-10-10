# src/db/schema.py
from __future__ import annotations

import sqlite3
from pathlib import Path


def init_db(db_path: str) -> None:
    """
    Initialize SQLite schema for:
      - documents
      - pages
      - chunks

    Design notes:
      - documents.file_hash is UNIQUE to deduplicate identical PDFs.
      - pages uses (doc_id, page_no) composite PK to preserve page reference.
      - chunks uses chunk_id PK + UNIQUE(doc_id,page_no,chunk_index) for stability.
      - Foreign keys ON + ON DELETE CASCADE so deleting a document clears children.
      - WAL mode improves read/write behavior for local apps.
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id     TEXT PRIMARY KEY,
                filename   TEXT NOT NULL,
                file_hash  TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                doc_id     TEXT NOT NULL,
                page_no    INTEGER NOT NULL,
                text_raw   TEXT,
                text_clean TEXT,
                PRIMARY KEY (doc_id, page_no),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                page_no     INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL,
                UNIQUE (doc_id, page_no, chunk_index),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );
            """
        )

        # Parent-child retrieval: each chunk (parent, up to 512 words) split into children (up to CHILD_WORDS,
        # optionally overlapping by CHILD_OVERLAP words). Children are searched; their parent is what the LLM
        # reads. Several child sets can sit side by side (child_words, child_overlap) so the benchmark can
        # compare them; the app uses one.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(child_chunks)")]
        if cols and "child_words" not in cols:     # children from before sizes existed: derived data, rebuilt by
            conn.execute("DROP TABLE child_chunks")  # python -m src.ingest --children all
            print("[db] child_chunks rebuilt with a child_words column; run: python -m src.ingest --children all")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS child_chunks (
                child_id    TEXT PRIMARY KEY,
                parent_id   TEXT NOT NULL,
                doc_id      TEXT NOT NULL,
                page_no     INTEGER NOT NULL,
                child_index INTEGER NOT NULL,
                text        TEXT NOT NULL,
                child_words INTEGER NOT NULL,
                child_overlap INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (parent_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
            );
            """
        )
        # A child set is (child_words, child_overlap). Sets from before overlap existed have none, so 0 is right.
        if "child_overlap" not in [r[1] for r in conn.execute("PRAGMA table_info(child_chunks)")]:
            conn.execute("ALTER TABLE child_chunks ADD COLUMN child_overlap INTEGER NOT NULL DEFAULT 0")
        conn.execute("DROP INDEX IF EXISTS idx_children_doc")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_children_set ON child_chunks(doc_id, child_words, child_overlap);")

        # One card per chunk, written by the summary job: what the chunk is about (topic) and a short
        # description. Part of the document: deleting or re-ingesting the chunks deletes their cards.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_cards (
                chunk_id    TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                topic       TEXT NOT NULL,
                description TEXT NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
            );
            """
        )

        # Indexes (practical for retrieval + citations)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cards_doc ON chunk_cards(doc_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_doc ON pages(doc_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks(doc_id, page_no);")

        # NEW: helps neighbor expansion + ordered per-page chunk fetch
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_page_index ON chunks(doc_id, page_no, chunk_index);")

        conn.commit()
    finally:
        conn.close()
