# src/graphs/study_graph.py
r"""
The study graph: wiring only. All logic lives in nodes.py.

START -> router --(whole-doc summary)--> summary_doc -> END
                \-> retrieve --(nothing found)--> no_context -> END
                      ^   \
                      |    v
                      +-- summarize_topic        agent loop: notes not covered -> search again (<= MAX_HOPS)
                              |--> topic_summary ---+
                              |--> answer ----------+--> remember -> END
                              |--> make_flashcards -+                ^
                              \--> make_quiz -> collect_answers (pause) -> grade --(no weak concepts)--+
                                      ^    \-(no usable questions)-> END        |
                                      +------------- review <------ weak concepts left?

summarize_topic always runs first, so answers, flashcards and quizzes are written from its notes
plus the chunks. remember writes the document's long-term memory (LangGraph store, memory.db) and
folds older messages into the chat's running summary.

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
from src.graphs.memory import open_store, open_store_async
from src.graphs.state import StudyState

CHECKPOINT_DB = Path(__file__).resolve().parents[2] / "Data" / "Database" / "checkpoints.db"


ANSWER_ROUTES = ["topic_summary", "answer", "make_flashcards", "make_quiz"]


def _add_topic_loop(g: StateGraph, routes: list) -> None:
    """retrieve <-> summarize_topic (the agent loop), shared by the study graph and the answer graph."""
    g.add_node("retrieve", N.retrieve)
    g.add_node("no_context", N.no_context)
    g.add_node("summarize_topic", N.summarize_topic)
    g.add_conditional_edges("retrieve", N.route_after_retrieve, ["summarize_topic", "no_context"])
    g.add_conditional_edges("summarize_topic", N.route_after_notes, ["retrieve", *routes])
    g.add_edge("no_context", END)


def build_study_graph(checkpointer=None, store=None):
    g = StateGraph(StudyState)

    g.add_node("router", N.router)
    g.add_node("summary_doc", N.summary_doc)
    _add_topic_loop(g, ANSWER_ROUTES)
    g.add_node("topic_summary", N.topic_summary)
    g.add_node("answer", N.answer)
    g.add_node("make_flashcards", N.make_flashcards)
    g.add_node("make_quiz", N.make_quiz)
    g.add_node("collect_answers", N.collect_answers)
    g.add_node("grade", N.grade)
    g.add_node("review", N.review)
    g.add_node("remember", N.remember)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", N.route_after_router, ["summary_doc", "retrieve"])
    g.add_edge("summary_doc", END)
    for leaf in ("topic_summary", "answer", "make_flashcards"):
        g.add_edge(leaf, "remember")
    g.add_edge("remember", END)

    g.add_conditional_edges("make_quiz", N.route_after_make_quiz, ["collect_answers", END])
    g.add_edge("collect_answers", "grade")
    g.add_conditional_edges("grade", N.route_after_grade, ["review", "remember"])
    g.add_edge("review", "make_quiz")

    return g.compile(checkpointer=checkpointer, store=store)


def build_answer_graph():
    """
    The study graph's question path without the router or memory: retrieve <-> summarize_topic -> answer.
    The answer benchmark (scripts/eval_generation.py) runs it, so it scores exactly what the app does.
    Input: {"doc_id", "topic": question, "intent": "ask", "scope": "topic", "messages": []}
    """
    g = StateGraph(StudyState)
    _add_topic_loop(g, ["answer"])
    g.add_node("answer", N.answer)
    g.add_edge(START, "retrieve")
    g.add_edge("answer", END)
    return g.compile()


def sqlite_checkpointer(path: Path = CHECKPOINT_DB) -> SqliteSaver:
    """Saves every thread's state to a file, so a quiz survives a server restart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


@asynccontextmanager
async def open_study_graph_async(path: Path = CHECKPOINT_DB) -> AsyncIterator:
    """
    The same graph for async callers (the web server): use ainvoke / astream / aget_state.
    Same checkpoint file and memory store as the terminal demo, so chats can be continued from either.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver, open_store_async() as store:
        yield build_study_graph(saver, store)


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
    with open_store() as store:
        graph = build_study_graph(sqlite_checkpointer(), store)
        print(f"thread_id = {thread_id}   (pass it again to continue this session)")
        _chat(graph, doc_id, config)


def _chat(graph, doc_id: str, config: dict) -> None:
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
