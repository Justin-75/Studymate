# src/db/repository.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from src.db.models import Document, Page, Chunk, ChildChunk, ChunkCard


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Repository:

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection: #Create SQL3 DB 
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    # ----------------------------
    # Documents
    # ----------------------------

    def create_document(self, filename: str, file_hash: str) -> Document:
        """
        Create a new document via SQL3, and check the document whether it's created via hash, if yes then return the doc and the 
        corresponding data (doc_id, filename, file_hash, created_at) directly
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT doc_id, filename, file_hash, created_at FROM documents WHERE file_hash = ?",
                (file_hash,),
            ).fetchone() #Check if the document is inside 

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
                (doc_id, filename, file_hash, created_at), #Insert Document
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
            ).fetchone() #Get data from the document

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
        Fetch chunks by IDs and preserve the input order (for RRF)
        Edge case:
        1. If an ID is missing = it is skipped.
        2. Duplicate IDs will appear duplicated in output.
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
    # Chunk cards (topic + description per chunk, written by the summary job)
    # ----------------------------

    def upsert_cards(self, cards: List[ChunkCard]) -> None:
        if not cards:
            return
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO chunk_cards(chunk_id, doc_id, topic, description) VALUES (?, ?, ?, ?)",
                [(c.chunk_id, c.doc_id, c.topic, c.description) for c in cards],
            )
            conn.commit()
        finally:
            conn.close()

    def get_cards(self, doc_id: str) -> List[ChunkCard]:
        """Cards in reading order (page, then chunk index)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT k.chunk_id, k.doc_id, k.topic, k.description
                FROM chunk_cards k JOIN chunks c ON c.chunk_id = k.chunk_id
                WHERE k.doc_id = ?
                ORDER BY c.page_no ASC, c.chunk_index ASC
                """,
                (doc_id,),
            ).fetchall()
            return [ChunkCard(r["chunk_id"], r["doc_id"], r["topic"], r["description"]) for r in rows]
        finally:
            conn.close()

    def get_cards_by_ids(self, chunk_ids: List[str]) -> Dict[str, ChunkCard]:
        """chunk_id -> its card; chunks without a card are left out."""
        if not chunk_ids:
            return {}
        placeholders = ",".join(["?"] * len(chunk_ids))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT chunk_id, doc_id, topic, description FROM chunk_cards WHERE chunk_id IN ({placeholders})",
                tuple(chunk_ids),
            ).fetchall()
            return {r["chunk_id"]: ChunkCard(r["chunk_id"], r["doc_id"], r["topic"], r["description"]) for r in rows}
        finally:
            conn.close()

    def has_all_cards(self, doc_id: str) -> bool:
        """True when every chunk of the document has a card (the summary job finished its card pass)."""
        conn = self._connect()
        try:
            n_chunks, n_cards = conn.execute(
                "SELECT (SELECT COUNT(*) FROM chunks WHERE doc_id = ?), (SELECT COUNT(*) FROM chunk_cards WHERE doc_id = ?)",
                (doc_id, doc_id),
            ).fetchone()
            return n_chunks > 0 and n_cards >= n_chunks
        finally:
            conn.close()

    # ----------------------------
    # Child chunks (parent-child retrieval: children are searched, the parent chunk goes to the LLM)
    # ----------------------------

    # A child set is (child_words, child_overlap); every method below takes child_overlap=0 = no overlap.
    _CHILD_COLS = "child_id, parent_id, doc_id, page_no, child_index, text, child_words, child_overlap"
    _CHILD_VALUES = ", ".join(["?"] * 8)

    @staticmethod
    def _child(r) -> ChildChunk:
        return ChildChunk(r["child_id"], r["parent_id"], r["doc_id"], int(r["page_no"]), int(r["child_index"]),
                          r["text"], int(r["child_words"]), int(r["child_overlap"]))

    @staticmethod
    def _child_row(c: ChildChunk) -> tuple:
        return (c.child_id, c.parent_id, c.doc_id, c.page_no, c.child_index, c.text, c.child_words, c.child_overlap)

    def replace_children(self, doc_id: str, children: List[ChildChunk], child_words: int,
                         child_overlap: int = 0) -> None:
        """(Re)build one child set of a document; the other sets stay."""
        conn = self._connect()
        try:
            conn.execute("BEGIN;")
            conn.execute("DELETE FROM child_chunks WHERE doc_id = ? AND child_words = ? AND child_overlap = ?",
                         (doc_id, child_words, child_overlap))
            conn.executemany(f"INSERT INTO child_chunks({self._CHILD_COLS}) VALUES ({self._CHILD_VALUES})",
                             [self._child_row(c) for c in children])
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def delete_children(self, child_words: int, child_overlap: int = 0) -> int:
        """Remove one child set from every document (e.g. a size tried in the benchmark). Returns rows removed."""
        conn = self._connect()
        try:
            n = conn.execute("DELETE FROM child_chunks WHERE child_words = ? AND child_overlap = ?",
                             (child_words, child_overlap)).rowcount
            conn.commit()
            return n
        finally:
            conn.close()

    def get_children(self, doc_id: str, child_words: int, child_overlap: int = 0) -> List[ChildChunk]:
        """One child set of a document, in reading order (page, parent, child)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT k.child_id, k.parent_id, k.doc_id, k.page_no, k.child_index, k.text, k.child_words,
                       k.child_overlap
                FROM child_chunks k JOIN chunks c ON c.chunk_id = k.parent_id
                WHERE k.doc_id = ? AND k.child_words = ? AND k.child_overlap = ?
                ORDER BY c.page_no ASC, c.chunk_index ASC, k.child_index ASC
                """,
                (doc_id, child_words, child_overlap),
            ).fetchall()
            return [self._child(r) for r in rows]
        finally:
            conn.close()

    def get_children_by_ids(self, child_ids: List[str]) -> List[ChildChunk]:
        """Children by id, in the input order (missing ids skipped). Ids hold the size, so they are unique."""
        if not child_ids:
            return []
        placeholders = ",".join(["?"] * len(child_ids))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {self._CHILD_COLS} FROM child_chunks WHERE child_id IN ({placeholders})", tuple(child_ids)
            ).fetchall()
            got = {r["child_id"]: self._child(r) for r in rows}
            return [got[cid] for cid in child_ids if cid in got]
        finally:
            conn.close()

    def has_children(self, doc_id: str, child_words: int, child_overlap: int = 0) -> bool:
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT 1 FROM child_chunks WHERE doc_id = ? AND child_words = ? AND child_overlap = ? LIMIT 1",
                (doc_id, child_words, child_overlap)).fetchone() is not None
        finally:
            conn.close()


    def replace_doc_content_atomic(
        self,
        doc_id: str,
        pages: List[Page],
        chunks: List[Chunk],
        children: Optional[List[ChildChunk]] = None,
    ) -> None:
        """
        Using the properties ACID on transaction in DB, Making sure that each
        Doc has a proper content of chunks and page by deleting old page and chunk
        and inserting new page and chunk with rollback system if there's error in the progress
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN;")

            # A chunk card is LLM work: keep the card of every chunk whose id and text come back unchanged
            kept_cards = conn.execute(
                "SELECT k.chunk_id, k.doc_id, k.topic, k.description, c.text FROM chunk_cards k "
                "JOIN chunks c ON c.chunk_id = k.chunk_id WHERE k.doc_id = ?",
                (doc_id,),
            ).fetchall()
            new_text = {c.chunk_id: c.text for c in chunks}
            kept_cards = [r for r in kept_cards if new_text.get(r["chunk_id"]) == r["text"]]

            # Clear old rows (cards of changed chunks and the children of every size describe the old text)
            conn.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunk_cards WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM child_chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

            # Insert new content pages
            if pages:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO pages(doc_id, page_no, text_raw, text_clean)
                    VALUES (?, ?, ?, ?)
                    """,
                    [(p.doc_id, p.page_no, p.text_raw, p.text_clean) for p in pages],
                )

            # Insert new content chunks
            if chunks:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO chunks(chunk_id, doc_id, page_no, chunk_index, text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [(c.chunk_id, c.doc_id, c.page_no, c.chunk_index, c.text) for c in chunks],
                )

            # Insert the children of those chunks (parent-child retrieval)
            if children:
                conn.executemany(
                    f"INSERT INTO child_chunks({self._CHILD_COLS}) VALUES ({self._CHILD_VALUES})",
                    [self._child_row(c) for c in children],
                )

            # Put back the cards of unchanged chunks (the summary job only labels the changed ones)
            if kept_cards:
                conn.executemany(
                    "INSERT INTO chunk_cards(chunk_id, doc_id, topic, description) VALUES (?, ?, ?, ?)",
                    [(r["chunk_id"], r["doc_id"], r["topic"], r["description"]) for r in kept_cards],
                )

            conn.commit()
        except Exception:
            try:
                conn.rollback() #Rollback to the older version 
            except Exception:
                pass
            raise
        finally:
            conn.close()
