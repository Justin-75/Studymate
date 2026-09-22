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

        # Indexes (practical for retrieval + citations)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_doc ON pages(doc_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks(doc_id, page_no);")

        # NEW: helps neighbor expansion + ordered per-page chunk fetch
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_page_index ON chunks(doc_id, page_no, chunk_index);")

        conn.commit()
    finally:
        conn.close()
