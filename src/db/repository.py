# src/db/repository.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from src.db.models import Document, Page, Chunk


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Repository:
    """
    Thin SQLite repository layer.

    Key guarantees:
      - Deterministic ordering for get_pages/get_chunks
      - Dedup on file_hash via upsert_document
      - get_chunks_by_ids preserves input ranking order
      - Atomic replace_doc_content_atomic for crash-safe ingestion
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    # ----------------------------
    # Documents
    # ----------------------------

    def upsert_document(self, filename: str, file_hash: str) -> Document:
        """
        If file_hash exists, returns the existing document (dedup).
        Otherwise inserts a new document and returns it.

        NOTE:
          - We keep filename from the first time by design (stable doc identity).
          - If you want filename updates on re-ingest, add UPDATE.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT doc_id, filename, file_hash, created_at FROM documents WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()

            if row:
                return Document(
                    doc_id=row["doc_id"],
                    filename=row["filename"],
                    file_hash=row["file_hash"],
                    created_at=row["created_at"],
                )

            doc_id = str(uuid4())
            created_at = _utc_now_iso()

            conn.execute(
                "INSERT INTO documents(doc_id, filename, file_hash, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, filename, file_hash, created_at),
            )
            conn.commit()

            return Document(
                doc_id=doc_id,
                filename=filename,
                file_hash=file_hash,
                created_at=created_at,
            )
        finally:
            conn.close()

    def get_document(self, doc_id: str) -> Optional[Document]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT doc_id, filename, file_hash, created_at FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()

            if not row:
                return None

            return Document(
                doc_id=row["doc_id"],
                filename=row["filename"],
                file_hash=row["file_hash"],
                created_at=row["created_at"],
            )
        finally:
            conn.close()

    # ----------------------------
    # Pages
    # ----------------------------

    def insert_pages(self, pages: List[Page]) -> None:
        if not pages:
            return

        conn = self._connect()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO pages(doc_id, page_no, text_raw, text_clean)
                VALUES (?, ?, ?, ?)
                """,
                [(p.doc_id, p.page_no, p.text_raw, p.text_clean) for p in pages],
            )
            conn.commit()
        finally:
            conn.close()

    def get_pages(self, doc_id: str) -> List[Page]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT doc_id, page_no, text_raw, text_clean
                FROM pages
                WHERE doc_id = ?
                ORDER BY page_no ASC
                """,
                (doc_id,),
            ).fetchall()

            return [
                Page(
                    doc_id=r["doc_id"],
                    page_no=int(r["page_no"]),
                    text_raw=r["text_raw"] or "",
                    text_clean=r["text_clean"] or "",
                )
                for r in rows
            ]
        finally:
            conn.close()

    # ----------------------------
    # Chunks
    # ----------------------------

    def insert_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        conn = self._connect()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks(chunk_id, doc_id, page_no, chunk_index, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(c.chunk_id, c.doc_id, c.page_no, c.chunk_index, c.text) for c in chunks],
            )
            conn.commit()
        finally:
            conn.close()

    def get_chunks(self, doc_id: str) -> List[Chunk]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, page_no, chunk_index, text
                FROM chunks
                WHERE doc_id = ?
                ORDER BY page_no ASC, chunk_index ASC
                """,
                (doc_id,),
            ).fetchall()

            return [
                Chunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    page_no=int(r["page_no"]),
                    chunk_index=int(r["chunk_index"]),
                    text=r["text"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Fetch chunks by IDs and preserve the input order (important for ranked retrieval).

        If an ID is missing, it is skipped.
        Duplicate IDs will appear duplicated in output.
        """
        if not chunk_ids:
            return []

        placeholders = ",".join(["?"] * len(chunk_ids))

        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT chunk_id, doc_id, page_no, chunk_index, text
                FROM chunks
                WHERE chunk_id IN ({placeholders})
                """,
                tuple(chunk_ids),
            ).fetchall()

            got = {
                r["chunk_id"]: Chunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    page_no=int(r["page_no"]),
                    chunk_index=int(r["chunk_index"]),
                    text=r["text"],
                )
                for r in rows
            }

            return [got[cid] for cid in chunk_ids if cid in got]
        finally:
            conn.close()

    # ----------------------------
    # NEW: Crash-safe atomic replace (fixes your issue)
    # ----------------------------

    def replace_doc_content_atomic(
        self,
        doc_id: str,
        pages: List[Page],
        chunks: List[Chunk],
    ) -> None:
        """
        Atomically replace ALL pages + chunks for a doc_id.

        If anything fails (e.g. OCR crash, insert error), DB is rolled back so you never
        end up with a document that has missing/half content.

        This should be used by ingest_pdf() instead of:
          - clear_doc_content
          - insert_pages
          - insert_chunks
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN;")

            # Clear old rows
            conn.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

            # Insert pages
            if pages:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO pages(doc_id, page_no, text_raw, text_clean)
                    VALUES (?, ?, ?, ?)
                    """,
                    [(p.doc_id, p.page_no, p.text_raw, p.text_clean) for p in pages],
                )

            # Insert chunks
            if chunks:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO chunks(chunk_id, doc_id, page_no, chunk_index, text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [(c.chunk_id, c.doc_id, c.page_no, c.chunk_index, c.text) for c in chunks],
                )

            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()
