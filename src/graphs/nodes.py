# src/graphs/nodes.py
"""
Study-graph nodes. Each node: read state -> (retrieve / call the LLM) -> return the keys it changes.

Code decides what the LLM sees and checks what comes back; the LLM only writes language.
Grading of mcq, mastery scores and the "understanding not enough" threshold are plain code.

Every reply is an AIMessage with:
    content            plain-text version (logs, terminal demo)
    additional_kwargs  {"kind": ..., "data": {...}}  structured version the web UI renders
kinds: answer | doc_summary | topic_summary | flashcards | quiz | grade | review | info
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import interrupt

from src.graphs import prompts as P
from src.graphs.schemas import (
    Answer, FlashcardSet, Intent, Quiz, ShortAnswerGrade, TopicSummary,
)
from src.graphs.state import StudyState
from src.graphs.store import is_building, load_doc_summary, repo
from src.llm_client import structured_llm
from src.retrieval.hybrid import hybrid_search

TOP_K = 8                 # chunks retrieved per request
DOC_KEY_POINTS = 30       # summary key points used for a whole-document quiz / flashcard set
QUIZ_SIZE = 5
MASTERY_THRESHOLD = 0.6   # a concept scoring below this goes to review
MAX_ROUNDS = 3            # quiz -> review -> quiz ... at most this many quizzes per loop


def reply(kind: str, text: str, data: dict | None = None) -> AIMessage:
    return AIMessage(content=text, additional_kwargs={"kind": kind, "data": data or {}})


def _last_user_message(state: StudyState) -> Optional[HumanMessage]:
    for m in reversed(state["messages"]):
        if m.type == "human":
            return m
    return None


def _search(doc_id: str, query: str, k: int = TOP_K) -> List[dict]:
    return [asdict(h) for h in hybrid_search(repo(), doc_id, query, top_k=k)]


def _spread(items: List[dict], k: int) -> List[dict]:
    """k items at evenly spaced positions, so the beginning of the document doesn't crowd out the rest."""
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _document_context(doc_id: str) -> List[dict]:
    """
    Material from across the whole document, for "quiz me on this PDF" without a topic:
    the key points of the cached section summaries, else chunks spread over the document.
    """
    summary = load_doc_summary(doc_id)
    if summary:
        points = [{"page_no": kp["page"], "text": f"[{s['title']}] {kp['text']}"}
                  for s in summary["sections"] for kp in s["key_points"]]
        return _spread(points, DOC_KEY_POINTS)
    chunks = [{"page_no": c.page_no, "text": c.text} for c in repo().get_chunks(doc_id)]
    return _spread(chunks, TOP_K)


def _pages(items) -> str:
    pages = sorted({int(p) for p in items if p})
    return ", ".join(f"p.{p}" for p in pages)


# ---------------------------------------------------------------------------
# 1. Router: what does the user want, and what should we search for?
# ---------------------------------------------------------------------------
def router(state: StudyState) -> dict:
    last = _last_user_message(state)
    user_msg = str(last.content) if last else ""
    # set when the user pressed the Quiz / Flashcards / Summary button instead of only typing
    chosen = last.additional_kwargs.get("intent") if last else None
    cached = load_doc_summary(state["doc_id"]) or {}
    history = P.format_history(state["messages"][:-1])

    decision: Intent = structured_llm(Intent, 0.0).invoke(
        P.router_prompt(history, user_msg, cached.get("subject", ""), chosen)
    )
    intent = chosen or decision.intent
    topic = decision.topic.strip()
    if decision.scope == "document" and intent in ("quiz", "flashcard"):
        topic = cached.get("subject") or "the whole document"   # what the quiz / cards are about
    topic = topic or user_msg                    # whole-document requests still need a search query
    print(f"[router] intent={intent}{' (button)' if chosen else ''} scope={decision.scope} topic={topic!r}")
    return {
        "intent": intent,
        "scope": decision.scope,
        "topic": topic,
        "round": 0,                 # a new request starts a new learning loop
        "weak_concepts": [],
    }


def route_after_router(state: StudyState) -> str:
    if state["intent"] == "summary" and state["scope"] == "document":
        return "summary_doc"
    return "retrieve"


# ---------------------------------------------------------------------------
# 2. Retrieve: one shared node for ask / topic summary / flashcards / quiz
# ---------------------------------------------------------------------------
def retrieve(state: StudyState) -> dict:
    if state["scope"] == "document" and state["intent"] in ("quiz", "flashcard"):
        hits = _document_context(state["doc_id"])          # no topic: cover the whole document
    else:
        hits = _search(state["doc_id"], state["topic"])
    print(f"[retrieve] {len(hits)} passages from pages {_pages(h['page_no'] for h in hits)}")
    return {"context": hits}


def route_after_retrieve(state: StudyState) -> str:
    if not state["context"]:
        return "no_context"
    return {
        "ask": "answer",
        "summary": "summarize_topic",
        "flashcard": "make_flashcards",
        "quiz": "make_quiz",
    }[state["intent"]]


def no_context(state: StudyState) -> dict:
    msg = "I couldn't find anything about that in this document. Try other keywords?"
    return {"messages": [reply("info", msg, {"code": "no_context"})]}


# ---------------------------------------------------------------------------
# 3. Answer a question
# ---------------------------------------------------------------------------
def answer(state: StudyState) -> dict:
    result: Answer = structured_llm(Answer, 0.2).invoke(
        P.answer_prompt(P.format_context(state["context"]), state["topic"])
    )
    text = result.answer
    if result.found and result.pages:
        text += f"\n\n(Source: {_pages(result.pages)})"
    return {"messages": [reply("answer", text, result.model_dump())]}


# ---------------------------------------------------------------------------
# 4. Summaries
# ---------------------------------------------------------------------------
def summary_doc(state: StudyState) -> dict:
    """
    Whole-PDF summary: read the cached section summaries made by the ingest graph.
    Never builds them here: that takes minutes to an hour and would hold up the whole chat.
    """
    doc_id = state["doc_id"]
    doc = load_doc_summary(doc_id)
    if doc is None and is_building(doc_id):
        msg = "The document summary is still being built. Ask questions in the meantime, then try again in a minute."
        return {"messages": [reply("info", msg, {"code": "summary_building"})]}
    if doc is None:
        msg = ("This document has no summary yet. Build it first (the Build summary button, or "
               f"`python -m src.graphs.ingest_graph {doc_id} <path to the PDF>`), then ask again.")
        return {"messages": [reply("info", msg, {"code": "summary_missing"})]}

    lines = [doc["subject"], "", doc["overview"], ""]
    lines += [f"- {s['title']} (p.{s['start']}-{s['end']}): {s['main_idea']}" for s in doc["sections"]]
    data = {k: doc[k] for k in ("subject", "overview", "sections")}
    return {"messages": [reply("doc_summary", "\n".join(lines), data)]}


def summarize_topic(state: StudyState) -> dict:
    result: TopicSummary = structured_llm(TopicSummary, 0.3).invoke(
        P.topic_summary_prompt(P.format_context(state["context"]), state["topic"])
    )
    lines = [result.main_idea, ""] + [f"- {kp.text} (p.{kp.page})" for kp in result.key_points]
    data = {"topic": state["topic"], **result.model_dump()}
    return {"messages": [reply("topic_summary", "\n".join(lines), data)]}


# ---------------------------------------------------------------------------
# 5. Flashcards
# ---------------------------------------------------------------------------
def make_flashcards(state: StudyState) -> dict:
    result: FlashcardSet = structured_llm(FlashcardSet, 0.5).invoke(
        P.flashcard_prompt(P.format_context(state["context"]), state["topic"])
    )
    pages = {c["page_no"] for c in state["context"]}
    cards = [c.model_dump() for c in result.cards if c.page in pages]   # drop cards citing pages we never showed it
    text = "\n".join(f"{i}. {c['front']} — {c['back']} (p.{c['page']})" for i, c in enumerate(cards, 1))
    if not cards:
        return {"flashcards": [], "messages": [reply("info", "No flashcards could be made from these pages.",
                                                     {"code": "no_flashcards"})]}
    return {"flashcards": cards,
            "messages": [reply("flashcards", text, {"topic": state["topic"], "cards": cards})]}


# ---------------------------------------------------------------------------
# 6. Quiz loop: make_quiz -> collect_answers (pause) -> grade -> review -> make_quiz ...
# ---------------------------------------------------------------------------
def _valid_question(q: dict) -> bool:
    if q["type"] == "mcq":
        return bool(q.get("options")) and len(q["options"]) == 4 and q["answer"].strip().upper()[:1] in "ABCD"
    return bool(q["answer"].strip())


def make_quiz(state: StudyState) -> dict:
    focus = state.get("weak_concepts") or None
    result: Quiz = structured_llm(Quiz, 0.4).invoke(
        P.quiz_prompt(P.format_context(state["context"]), state["topic"], QUIZ_SIZE, focus)
    )
    questions = [q.model_dump() for q in result.questions]
    questions = [q for q in questions if _valid_question(q)]
    for n, q in enumerate(questions, 1):
        q["id"] = f"q{n}"          # the model's own ids can repeat, and answers are keyed by id
    if not questions:
        return {"quiz": {"questions": []}, "messages": [reply(
            "info", "I couldn't write a usable quiz from these pages. Try again or name a narrower topic.",
            {"code": "no_quiz"})]}
    rnd = state.get("round", 0) + 1

    lines = [f"Quiz round {rnd}:"]
    for q in questions:
        lines.append(f"\n{q['id']}. {q['question']}")
        lines += [f"   {o}" for o in q.get("options") or []]
    data = {"round": rnd, "topic": state["topic"], "focus": focus or [], "questions": questions}
    # NOTE: data includes the answers; the server strips them before sending a quiz to the browser
    return {"quiz": {"questions": questions}, "messages": [reply("quiz", "\n".join(lines), data)]}


def route_after_make_quiz(state: StudyState) -> str:
    return "collect_answers" if state["quiz"]["questions"] else END


def collect_answers(state: StudyState) -> dict:
    """
    Pause the graph until the user answers.
    Resume with: Command(resume={"answers": {"q1": "A", "q2": "...", ...}})
    (wrapped in "answers" because LangGraph reads a bare dict of ids as interrupt-id -> value)
    """
    payload = interrupt({"type": "quiz", "quiz": state["quiz"]})
    if isinstance(payload, str):
        payload = json.loads(payload)
    answers = payload.get("answers", payload) if isinstance(payload, dict) else {}
    return {"answers": {str(k): str(v) for k, v in answers.items()}}


def _norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", (s or "").lower())


def _mcq_label(q: dict, letter: str) -> str:
    """'B' -> 'B. the option text' (or just the letter if not found)."""
    letter = (letter or "").strip().upper()[:1]
    for o in q.get("options") or []:
        if o.strip().upper().startswith(letter):
            return o
    return letter


def grade(state: StudyState) -> dict:
    answers = state.get("answers") or {}
    questions = state["quiz"]["questions"]
    given_by_id = {q["id"]: answers.get(q["id"], "").strip() for q in questions}

    # short answers and inexact fill-blanks go to the LLM judge, all in one parallel batch
    def needs_judge(q: dict) -> bool:
        given = given_by_id[q["id"]]
        if q["type"] == "mcq" or not given:
            return False
        return not (q["type"] == "fill_blank" and _norm(given) == _norm(q["answer"]))

    to_judge = [q for q in questions if needs_judge(q)]
    verdicts: List[ShortAnswerGrade] = structured_llm(ShortAnswerGrade, 0.0).batch(
        [P.grade_short_prompt(q["question"], q["answer"], given_by_id[q["id"]]) for q in to_judge]
    ) if to_judge else []
    judged = {q["id"]: v for q, v in zip(to_judge, verdicts)}

    graded = []
    for q in questions:
        given = given_by_id[q["id"]]
        if q["type"] == "mcq":                                   # plain code
            correct = bool(given) and given.upper()[:1] == q["answer"].strip().upper()[:1]
            feedback = ""
            shown_given, shown_answer = _mcq_label(q, given) if given else "", _mcq_label(q, q["answer"])
        else:
            shown_given, shown_answer = given, q["answer"]
            if q["id"] in judged:
                correct, feedback = judged[q["id"]].correct, judged[q["id"]].feedback
            else:                                                 # empty = wrong, exact fill-blank = right
                correct, feedback = bool(given), ""
        graded.append({"id": q["id"], "question": q["question"], "type": q["type"], "concept": q["concept"],
                       "correct": correct, "your_answer": shown_given, "correct_answer": shown_answer,
                       "feedback": feedback, "page": q["page"]})

    # per-concept score for this quiz, then update mastery
    per_concept: Dict[str, List[bool]] = {}
    for g in graded:
        per_concept.setdefault(g["concept"], []).append(g["correct"])
    scores = {c: sum(v) / len(v) for c, v in per_concept.items()}
    mastery = {**state.get("mastery", {}), **scores}
    weak = [c for c, s in scores.items() if s < MASTERY_THRESHOLD]
    rnd = state.get("round", 0) + 1
    will_review = bool(weak) and rnd < MAX_ROUNDS

    n_right = sum(g["correct"] for g in graded)
    lines = [f"Score: {n_right}/{len(graded)}"]
    for g in graded:
        mark = "✓" if g["correct"] else "✗"
        extra = "" if g["correct"] else f": correct answer {g['correct_answer']} (p.{g['page']})"
        lines.append(f"{mark} {g['id']} ({g['concept']}){extra}")
    if will_review:
        lines.append(f"\nLet's review: {', '.join(weak)}")

    data = {"round": rnd, "score": n_right, "total": len(graded), "results": graded,
            "weak": weak, "will_review": will_review, "max_rounds": MAX_ROUNDS}
    return {
        "graded": graded,
        "mastery": mastery,
        "weak_concepts": weak,
        "round": rnd,
        "messages": [reply("grade", "\n".join(lines), data)],
    }


def route_after_grade(state: StudyState) -> str:
    if state["weak_concepts"] and state["round"] < MAX_ROUNDS:
        return "review"
    return END


def review(state: StudyState) -> dict:
    """Short targeted summary of only the missed concepts, then back to make_quiz."""
    weak = state["weak_concepts"]
    hits = _search(state["doc_id"], " ".join(weak))
    result: TopicSummary = structured_llm(TopicSummary, 0.3).invoke(
        P.review_prompt(P.format_context(hits), weak)
    )
    lines = ["Review:", result.main_idea, ""] + [f"- {kp.text} (p.{kp.page})" for kp in result.key_points]
    data = {"concepts": weak, **result.model_dump()}
    return {"context": hits or state["context"], "messages": [reply("review", "\n".join(lines), data)]}
