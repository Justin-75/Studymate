# scripts/clean_noise.py
"""
Remove known extraction noise from documents that are ALREADY in the database, so you
don't have to re-ingest the PDFs:

    MXNet log lines   [07:00:31] ../src/storage/storage.cc:196: Using Pooled (Naive) StorageManager for CPU
    page-break marks  (continued from previous page) · (continues on next page)
    code wrap marks   ,→  (d2l-zh)   ↩→  (LLMBook)

The same rules run on every new ingest (src/pdf/cleaner.py: strip_known_noise).

    python scripts/clean_noise.py           # dry run: show what would change, write nothing
    python scripts/clean_noise.py --apply   # back up the DB, clean chunks + pages, clear caches

`python scripts/eval_retrieval.py ablation` calls this for you between its runs.

After --apply, the TF-IDF and BGE-M3 caches of every changed document are deleted, so
the next eval rebuilds them (BGE-M3 re-encodes those books once on the GPU, about a
minute per book). Chunk ids and page numbers do not change, so gold pages in
eval/questions.jsonl stay valid.
"""
from __future__ import annotations

import re
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pdf.cleaner import strip_known_noise  # noqa: E402

DB_PATH = ROOT / "Data/Database/app.db"
CACHE_DIRS = [ROOT / "Data/Cache/tfidf", ROOT / "Data/Cache/bge_m3"]
MIN_CHUNK_CHARS = 3   # a chunk that is (almost) only noise is deleted


def tidy(text: str) -> str:
    """Clean, then fix the double spaces / empty lines left where noise was removed.
    Text without noise is returned unchanged (byte for byte)."""
    cleaned = strip_known_noise(text)
    if cleaned == text:
        return text
    text = re.sub(r"[ \t]{2,}", " ", cleaned)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_changes() -> Dict:
    """Scan the DB; nothing is written."""
    con = sqlite3.connect(str(DB_PATH))
    try:
        chunk_updates, chunk_deletes, page_updates = [], [], []
        per_doc: Counter = Counter()
        for cid, doc_id, text in con.execute("SELECT chunk_id, doc_id, text FROM chunks"):
            new = tidy(text)
            if new != text:
                per_doc[doc_id] += 1
                if len(new) < MIN_CHUNK_CHARS:
                    chunk_deletes.append((cid,))
                else:
                    chunk_updates.append((new, cid))
        for doc_id, page_no, text in con.execute("SELECT doc_id, page_no, text_clean FROM pages"):
            if text and tidy(text) != text:
                page_updates.append((tidy(text), doc_id, page_no))
        names = dict(con.execute("SELECT doc_id, filename FROM documents"))
    finally:
        con.close()
    docs = set(per_doc) | {d for _, d, _ in page_updates}
    return {"chunk_updates": chunk_updates, "chunk_deletes": chunk_deletes,
            "page_updates": page_updates, "per_doc": per_doc, "docs": docs,
            "n_docs": len(docs), "names": names}


def clean_database(apply: bool = False) -> Dict:
    ch = find_changes()
    print(f"Chunks to clean: {len(ch['chunk_updates'])}   chunks that were only noise (delete): "
          f"{len(ch['chunk_deletes'])}   pages to clean: {len(ch['page_updates'])}")
    for doc_id, n in ch["per_doc"].most_common():
        print(f"  {ch['names'].get(doc_id, doc_id)[:40]:<40} {doc_id[:8]}  {n} chunks")

    if not apply:
        print("\nDry run only. Run again with --apply to write the changes.")
        return ch
    if not ch["n_docs"]:
        print("Nothing to do: the database is already clean.")
        return ch

    con = sqlite3.connect(str(DB_PATH))
    try:
        backup = DB_PATH.with_name(f"app.db.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        dst = sqlite3.connect(str(backup))
        con.backup(dst)                       # safe copy even with the WAL file present
        dst.close()
        print(f"Backup written: {backup}")
        with con:
            con.executemany("UPDATE chunks SET text=? WHERE chunk_id=?", ch["chunk_updates"])
            con.executemany("DELETE FROM chunks WHERE chunk_id=?", ch["chunk_deletes"])
            con.executemany("UPDATE pages SET text_clean=? WHERE doc_id=? AND page_no=?", ch["page_updates"])
    finally:
        con.close()
    print("Database updated.")

    for doc_id in ch["docs"]:
        for root in CACHE_DIRS:
            d = root / doc_id
            if d.exists():
                shutil.rmtree(d)
                print(f"  cleared cache {d.relative_to(ROOT)}")
    print("The next eval run rebuilds the cleared indexes automatically.")
    return ch


if __name__ == "__main__":
    try:  # Windows consoles (GBK) cannot print some markers like ↩; never crash on printing
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    clean_database(apply="--apply" in sys.argv[1:])
