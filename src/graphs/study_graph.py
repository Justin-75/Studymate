# src/graphs/study_graph.py
r"""
The study graph: wiring only. All logic lives in nodes.py.

START -> router --(whole-doc summary)--> summary_doc -> END
                \-> retrieve --> answer | summarize_topic | make_flashcards -> END
                             \-> make_quiz -> collect_answers (pause) -> grade
                                    ^    \-(no usable questions)-> END        |
                                    +------------- review <--- weak concepts left?
                                                                          \-> END

Try it in the terminal:
    python -m src.graphs.study_graph <doc_id>
"""
from __future__ import annotations

import sqlite3
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.graphs import nodes as N
from src.graphs.state import StudyState

CHECKPOINT_DB = Path(__file__).resolve().parents[2] / "Data" / "Database" / "checkpoints.db"


def build_study_graph(checkpointer=None):
    g = StateGraph(StudyState)

    g.add_node("router", N.router)
    g.add_node("summary_doc", N.summary_doc)
    g.add_node("retrieve", N.retrieve)
    g.add_node("no_context", N.no_context)
    g.add_node("answer", N.answer)
    g.add_node("summarize_topic", N.summarize_topic)
    g.add_node("make_flashcards", N.make_flashcards)
    g.add_node("make_quiz", N.make_quiz)
    g.add_node("collect_answers", N.collect_answers)
    g.add_node("grade", N.grade)
    g.add_node("review", N.review)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", N.route_after_router, ["summary_doc", "retrieve"])
    g.add_conditional_edges(
        "retrieve", N.route_after_retrieve,
        ["answer", "summarize_topic", "make_flashcards", "make_quiz", "no_context"],
    )
    for leaf in ("summary_doc", "no_context", "answer", "summarize_topic", "make_flashcards"):
        g.add_edge(leaf, END)

    g.add_conditional_edges("make_quiz", N.route_after_make_quiz, ["collect_answers", END])
    g.add_edge("collect_answers", "grade")
    g.add_conditional_edges("grade", N.route_after_grade, ["review", END])
    g.add_edge("review", "make_quiz")

    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: Path = CHECKPOINT_DB) -> SqliteSaver:
    """Saves every thread's state to a file, so a quiz survives a server restart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def get_study_graph():
    return build_study_graph(sqlite_checkpointer())


@asynccontextmanager
async def open_study_graph_async(path: Path = CHECKPOINT_DB) -> AsyncIterator:
    """
    The same graph for async callers (the web server): use ainvoke / astream / aget_state.
    Same checkpoint file as the terminal demo, so chats can be continued from either.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield build_study_graph(saver)


# ---------------------------------------------------------------------------
# Terminal demo
# ---------------------------------------------------------------------------
def _print_new_messages(result: dict, seen: int) -> int:
    msgs = result.get("messages", [])
    for m in msgs[seen:]:
        if m.type == "ai":
            print(f"\nStudyMate: {m.content}\n")
    return len(msgs)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m src.graphs.study_graph <doc_id> [thread_id]")
        return
    doc_id = sys.argv[1]
    thread_id = sys.argv[2] if len(sys.argv) > 2 else f"{doc_id}-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_study_graph()
    print(f"thread_id = {thread_id}   (pass it again to continue this session)")

    seen = len(graph.get_state(config).values.get("messages", []))
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("exit", "quit", "q"):
            break
        if not text:
            continue

        result = graph.invoke({"messages": [HumanMessage(text)], "doc_id": doc_id}, config)
        seen = _print_new_messages(result, seen)

        while "__interrupt__" in result:                  # the quiz is waiting for answers
            quiz = result["__interrupt__"][0].value["quiz"]
            answers = {}
            for q in quiz["questions"]:
                answers[q["id"]] = input(f"  {q['id']} answer: ").strip()
            result = graph.invoke(Command(resume={"answers": answers}), config)
            seen = _print_new_messages(result, seen)


if __name__ == "__main__":
    main()
