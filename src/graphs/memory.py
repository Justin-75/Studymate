# src/graphs/memory.py
"""
Long-term memory per document, shared by every chat on that PDF. It lives in the LangGraph store
(Data/Database/memory.db), which the graph hands to nodes as runtime.store:

    ("studymate", doc_id, "topics")    key = the topic     {"topic", "takeaway", "pages", "intent", "count"}
    ("studymate", doc_id, "mastery")   key = the concept   {"concept", "score"}

Short-term memory (this chat) stays in the thread's state: the messages, the last turn's chunks and
a running conversation summary (nodes.remember).

Readers and writers:
    router    reads recent topics + missed concepts, so "continue where we stopped" or "quiz me on
              what I got wrong" works in a new chat too
    remember  writes the topic of each answered turn and its one-line takeaway (no extra LLM call)
    grade     writes the quiz score of each concept
"""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, Iterator, List, Optional

from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

MEMORY_DB = Path(__file__).resolve().parents[2] / "Data" / "Database" / "memory.db"
RECENT_TOPICS = 8      # topics the router sees
SCAN_LIMIT = 500       # items read per namespace


def _ns(doc_id: str, kind: str) -> tuple:
    return ("studymate", doc_id, kind)


def recent_topics(store: Optional[BaseStore], doc_id: str, n: int = RECENT_TOPICS) -> List[dict]:
    """The n most recently studied topics of this document, newest first."""
    if store is None:
        return []
    items = store.search(_ns(doc_id, "topics"), limit=SCAN_LIMIT)
    items.sort(key=lambda it: it.updated_at, reverse=True)
    return [it.value for it in items[:n]]


def save_topic(store: Optional[BaseStore], doc_id: str, topic: str, takeaway: str,
               pages: List[int], intent: str) -> None:
    if store is None or not topic.strip():
        return
    ns, key = _ns(doc_id, "topics"), topic.strip().lower()[:200]
    old = store.get(ns, key)
    count = (old.value.get("count", 0) if old else 0) + 1
    store.put(ns, key, {"topic": topic.strip(), "takeaway": takeaway.strip(), "pages": sorted(set(pages)),
                        "intent": intent, "count": count})


def mastery(store: Optional[BaseStore], doc_id: str) -> Dict[str, float]:
    if store is None:
        return {}
    return {it.value["concept"]: float(it.value["score"])
            for it in store.search(_ns(doc_id, "mastery"), limit=SCAN_LIMIT)}


def save_mastery(store: Optional[BaseStore], doc_id: str, scores: Dict[str, float]) -> None:
    if store is None:
        return
    for concept, score in scores.items():
        store.put(_ns(doc_id, "mastery"), concept.strip().lower()[:200], {"concept": concept, "score": score})


def weak_concepts(store: Optional[BaseStore], doc_id: str, threshold: float) -> List[str]:
    return [c for c, s in mastery(store, doc_id).items() if s < threshold]


# ---------------------------------------------------------------------------
# Opening the store
# ---------------------------------------------------------------------------
@contextmanager
def open_store(path: Path = MEMORY_DB) -> Iterator[SqliteStore]:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(sqlite3.connect(str(path), check_same_thread=False, isolation_level=None))
    store.setup()
    try:
        yield store
    finally:
        store.conn.close()


@asynccontextmanager
async def open_store_async(path: Path = MEMORY_DB) -> AsyncIterator[AsyncSqliteStore]:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteStore.from_conn_string(str(path)) as store:
        await store.setup()
        yield store
