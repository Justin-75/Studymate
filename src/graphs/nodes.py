# src/graphs/nodes.py
"""
Study-graph nodes. Each node: read state -> (retrieve / call the LLM) -> return the keys it changes.

Code decides what the LLM sees and checks what comes back; the LLM only writes language.
Grading of mcq, mastery scores and the "understanding not enough" threshold are plain code.

Every turn goes topic first: retrieve -> summarize_topic writes notes on the chunks -> the answer,
flashcards or quiz are written from those notes AND the chunks (the chunks keep exact facts and pages).
Retrieval searches the chunk texts; once the document is summarized, each chunk also comes with its
card (topic + description) as extra reference for the LLM.

Agent loop (retrieve <-> summarize_topic): the notes say whether the chunks cover the task. If not,
summarize_topic writes the next search query and the graph searches again, adding to the chunks,
at most MAX_HOPS rounds per turn. A follow-up ("and A2?" after A) starts from the previous turn's
chunks, so it searches only when they don't already hold the answer.

Memory: this chat = messages + a running conversation summary (remember); this document, all chats =
the LangGraph store (memory.py), read by the router and written by remember and grade.

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
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.graphs import memory as M
from src.graphs import prompts as P
from src.graphs.schemas import (
    Answer, ConversationSummary, FlashcardSet, Intent, Quiz, ShortAnswerGrade, TopicNotes, TopicSummary,
)
from src.graphs.state import StudyState
from src.graphs.store import doc_language, is_building, load_doc_summary, repo
from src.llm_client import structured_llm
from src.retrieval.hybrid import hybrid_search

TOP_K = 5                 # parent chunks (512 words) the first search of a turn gives the LLM
MORE_K = 2                # parents added by each extra search of the agent loop
MAX_CONTEXT = 5           # parents the LLM sees at most: extra searches replace the lowest-ranked ones
MAX_HOPS = 3              # retrieval rounds per turn: the first search + up to 2 more
DOC_KEY_POINTS = 30       # summary key points used for a whole-document quiz / flashcard set
QUIZ_SIZE = 5
MASTERY_THRESHOLD = 0.6   # a concept scoring below this goes to review
MAX_ROUNDS = 3            # quiz -> review -> quiz ... at most this many quizzes per loop
KEEP_RECENT = 4           # messages always given word for word; older ones go into the running summary
SUMMARIZE_EVERY = 8       # fold messages into the summary once this many older ones are not in it yet


def reply(kind: str, text: str, data: dict | None = None) -> AIMessage:
    return AIMessage(content=text, additional_kwargs={"kind": kind, "data": data or {}})


def _last_user_message(state: StudyState) -> Optional[HumanMessage]:
    for m in reversed(state["messages"]):
        if m.type == "human":
            return m
    return None


def _search(doc_id: str, query: str, k: Optional[int] = None) -> List[dict]:
    """
    The app's retrieval over the chunk texts (hybrid.py). Each hit also carries its chunk card
    (card_topic, card_description) when the document was summarized: extra reference for the LLM.
    """
    hits = [asdict(h) for h in hybrid_search(repo(), doc_id, query, top_k=k or TOP_K)]
    cards = repo().get_cards_by_ids([h["chunk_id"] for h in hits])
    for h in hits:
        if h["chunk_id"] in cards:
            h["card_topic"], h["card_description"] = cards[h["chunk_id"]].topic, cards[h["chunk_id"]].description
    return hits


def _merge(old: List[dict], new: List[dict]) -> List[dict]:
    """Add newly found chunks to the turn's chunks; when over MAX_CONTEXT, the oldest extras make room."""
    seen = {c["chunk_id"] for c in old if c.get("chunk_id")}
    fresh = [c for c in new if c.get("chunk_id") not in seen]
    return old[: max(0, MAX_CONTEXT - len(fresh))] + fresh


def _spread(items: List[dict], k: int) -> List[dict]:
    """k items at evenly spaced positions, so the beginning of the document doesn't crowd out the rest."""
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _document_context(doc_id: str) -> List[dict]:
    """
    Material from across the whole document, for "quiz me on this PDF" without a topic:
    the key points of the cached section summaries, else the chunk cards, else chunks spread over it.
    """
    summary = load_doc_summary(doc_id)
    if summary:
        points = [{"page_no": kp["page"], "text": f"[{s['title']}] {kp['text']}"}
                  for s in summary["sections"] for kp in s["key_points"]]
        return _spread(points, DOC_KEY_POINTS)
    r = repo()
    if r.has_all_cards(doc_id):
        page_of = {c.chunk_id: c.page_no for c in r.get_chunks(doc_id)}
        cards = [{"chunk_id": k.chunk_id, "page_no": page_of[k.chunk_id], "text": f"[{k.topic}] {k.description}"}
                 for k in r.get_cards(doc_id)]
        return _spread(cards, DOC_KEY_POINTS)
    chunks = [{"chunk_id": c.chunk_id, "page_no": c.page_no, "text": c.text} for c in r.get_chunks(doc_id)]
    return _spread(chunks, TOP_K)


def _pages(items) -> str:
    pages = sorted({int(p) for p in items if p})
    return ", ".join(f"p.{p}" for p in pages)


def _conversation(state: StudyState) -> str:
    """This chat for a prompt: the running summary + the recent messages, without the message being answered."""
    msgs = state.get("messages", [])
    recent = msgs[state.get("summarized_upto", 0):]
    if recent and recent[-1].type == "human":
        recent = recent[:-1]
    return P.format_conversation(state.get("conversation_summary", ""), recent, KEEP_RECENT)


# ---------------------------------------------------------------------------
# 1. Router: what does the user want, is it a follow-up, and what should we search for?
# ---------------------------------------------------------------------------
def router(state: StudyState, runtime: Runtime) -> dict:
    last = _last_user_message(state)
    user_msg = str(last.content) if last else ""
    # set when the user pressed the Quiz / Flashcards / Summary button instead of only typing
    chosen = last.additional_kwargs.get("intent") if last else None
    doc_id = state["doc_id"]
    cached = load_doc_summary(doc_id) or {}
    memory = P.format_memory(M.recent_topics(runtime.store, doc_id),
                             M.weak_concepts(runtime.store, doc_id, MASTERY_THRESHOLD))

    decision: Intent = structured_llm(Intent, 0.0).invoke(
        P.router_prompt(_conversation(state), memory, user_msg, cached.get("subject", ""), chosen)
    )
    intent = chosen or decision.intent
    topic = decision.topic.strip()
    if decision.scope == "document" and intent in ("quiz", "flashcard"):
        topic = cached.get("subject") or "the whole document"   # what the quiz / cards are about
    topic = topic or user_msg                    # whole-document requests still need a search query
    follow_up = decision.follow_up and decision.scope == "topic" and bool(state.get("context"))
    print(f"[router] intent={intent}{' (button)' if chosen else ''} scope={decision.scope} "
          f"follow_up={follow_up} topic={topic!r}")
    return {
        "intent": intent,
        "scope": decision.scope,
        "topic": topic,
        "follow_up": follow_up,
        "notes": None,
        "hops": 0,
        "queries": [],
        "round": 0,                 # a new request starts a new learning loop
        "weak_concepts": [],
    }


def route_after_router(state: StudyState) -> str:
    if state["intent"] == "summary" and state["scope"] == "document":
        return "summary_doc"
    return "retrieve"


# ---------------------------------------------------------------------------
# 2. Retrieve (agent loop step 1): first search, the previous turn's chunks, or one more search
# ---------------------------------------------------------------------------
def retrieve(state: StudyState) -> dict:
    doc_id, hops = state["doc_id"], state.get("hops", 0)
    query = None
    if state["scope"] == "document" and state["intent"] in ("quiz", "flashcard"):
        hits = _document_context(doc_id)                     # no topic: cover the whole document
    elif hops == 0 and state.get("follow_up"):
        hits = state["context"]                              # continue from the last turn's chunks
    elif hops == 0:
        query = state["topic"]
        hits = _search(doc_id, query)
    else:                                                    # the notes asked for more
        query = state["notes"]["next_query"].strip()
        hits = _merge(state["context"], _search(doc_id, query, MORE_K))
    how = f"search {query!r}" if query else ("previous turn's chunks" if state.get("follow_up") else "whole document")
    print(f"[retrieve] round {hops + 1}, {how}: {len(hits)} passages from pages {_pages(h['page_no'] for h in hits)}")
    return {"context": hits, "hops": hops + 1, "queries": state.get("queries", []) + ([query] if query else [])}


def route_after_retrieve(state: StudyState) -> str:
    return "summarize_topic" if state["context"] else "no_context"


def no_context(state: StudyState) -> dict:
    msg = "I couldn't find anything about that in this document. Try other keywords?"
    return {"messages": [reply("info", msg, {"code": "no_context"})]}


# ---------------------------------------------------------------------------
# 3. Summarize the topic (agent loop step 2): notes on the chunks + "covered, or search for X"
# ---------------------------------------------------------------------------
def summarize_topic(state: StudyState) -> dict:
    notes: TopicNotes = structured_llm(TopicNotes, 0.2).invoke(
        P.topic_notes_prompt(P.format_context(state["context"]), state["topic"], state["intent"],
                             _conversation(state), doc_language(state["doc_id"]))
    )
    # an extra search that gave the notes no new page to cite won't be helped by a third one
    cited = sorted({kp.page for kp in notes.key_points})
    stalled = state["hops"] > 1 and set(cited) <= set((state.get("notes") or {}).get("cited", []))
    print(f"[summarize_topic] round {state['hops']}: covered={notes.covered}"
          + ("" if notes.covered else f", next query {notes.next_query!r}") + (", no new pages: stop" if stalled else ""))
    return {"notes": {**notes.model_dump(), "cited": cited, "stalled": stalled}}


def route_after_notes(state: StudyState) -> str:
    notes = state["notes"]
    next_query = (notes.get("next_query") or "").strip()
    if (not notes["covered"] and not notes.get("stalled") and next_query
            and next_query not in state.get("queries", [])
            and state["hops"] < MAX_HOPS and state["scope"] != "document"):
        return "retrieve"
    return {
        "ask": "answer",
        "summary": "topic_summary",
        "flashcard": "make_flashcards",
        "quiz": "make_quiz",
    }[state["intent"]]


def topic_summary(state: StudyState) -> dict:
    """The summary intent's reply is the notes themselves."""
    notes = state["notes"]
    lines = [notes["main_idea"], ""] + [f"- {kp['text']} (p.{kp['page']})" for kp in notes["key_points"]]
    data = {"topic": state["topic"], "main_idea": notes["main_idea"], "key_points": notes["key_points"]}
    return {"messages": [reply("topic_summary", "\n".join(lines), data)]}


# ---------------------------------------------------------------------------
# 4. Answer a question: from the notes AND the chunks, continuing the conversation
# ---------------------------------------------------------------------------
def answer(state: StudyState) -> dict:
    result: Answer = structured_llm(Answer, 0.2).invoke(
        P.answer_prompt(P.format_context(state["context"]), state["topic"],
                        P.format_notes(state.get("notes")), _conversation(state), doc_language(state["doc_id"]))
    )
    text = result.answer
    if result.found and result.pages:
        text += f"\n\n(Source: {_pages(result.pages)})"
    return {"messages": [reply("answer", text, result.model_dump())]}


# ---------------------------------------------------------------------------
# 5. Whole-document summary (cached by the ingest graph)
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


# ---------------------------------------------------------------------------
# 6. Flashcards
# ---------------------------------------------------------------------------
def make_flashcards(state: StudyState) -> dict:
    result: FlashcardSet = structured_llm(FlashcardSet, 0.5).invoke(
        P.flashcard_prompt(P.format_context(state["context"]), state["topic"], P.format_notes(state.get("notes")),
                           lang=doc_language(state["doc_id"]))
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
# 7. Quiz loop: make_quiz -> collect_answers (pause) -> grade -> review -> make_quiz ...
# ---------------------------------------------------------------------------
def _valid_question(q: dict) -> bool:
    if q["type"] == "mcq":
        return bool(q.get("options")) and len(q["options"]) == 4 and q["answer"].strip().upper()[:1] in "ABCD"
    return bool(q["answer"].strip())


def make_quiz(state: StudyState) -> dict:
    focus = state.get("weak_concepts") or None
    result: Quiz = structured_llm(Quiz, 0.4).invoke(
        P.quiz_prompt(P.format_context(state["context"]), state["topic"], QUIZ_SIZE, focus,
                      P.format_notes(state.get("notes")), doc_language(state["doc_id"]))
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


def grade(state: StudyState, runtime: Runtime) -> dict:
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

    # per-concept score for this quiz, then update mastery: this chat's state + the document's memory
    per_concept: Dict[str, List[bool]] = {}
    for g in graded:
        per_concept.setdefault(g["concept"], []).append(g["correct"])
    scores = {c: sum(v) / len(v) for c, v in per_concept.items()}
    M.save_mastery(runtime.store, state["doc_id"], scores)
    mastery = {**M.mastery(runtime.store, state["doc_id"]), **state.get("mastery", {}), **scores}
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
    return "remember"


def review(state: StudyState) -> dict:
    """Short targeted summary of only the missed concepts, then back to make_quiz."""
    weak = state["weak_concepts"]
    hits = _search(state["doc_id"], " ".join(weak))
    result: TopicSummary = structured_llm(TopicSummary, 0.3).invoke(
        P.review_prompt(P.format_context(hits), weak, doc_language(state["doc_id"]))
    )
    lines = ["Review:", result.main_idea, ""] + [f"- {kp.text} (p.{kp.page})" for kp in result.key_points]
    data = {"concepts": weak, **result.model_dump()}
    return {"context": hits or state["context"], "notes": result.model_dump(),
            "messages": [reply("review", "\n".join(lines), data)]}


# ---------------------------------------------------------------------------
# 8. Remember: the document's long-term memory + this chat's running summary
# ---------------------------------------------------------------------------
def remember(state: StudyState, runtime: Runtime) -> dict:
    """End of a turn. Never fails the turn: the reply is already written, memory is a bonus."""
    update: dict = {}
    try:
        notes = state.get("notes") or {}
        if state.get("scope") == "topic" and notes.get("main_idea"):
            pages = [int(kp["page"]) for kp in notes.get("key_points", []) if kp.get("page")]
            M.save_topic(runtime.store, state["doc_id"], state["topic"], notes["main_idea"], pages, state["intent"])

        msgs, upto = state["messages"], state.get("summarized_upto", 0)
        if len(msgs) - KEEP_RECENT - upto >= SUMMARIZE_EVERY:
            new_upto = len(msgs) - KEEP_RECENT
            older = P.format_history(msgs[upto:new_upto], last_n=new_upto - upto)
            result: ConversationSummary = structured_llm(ConversationSummary, 0.0).invoke(
                P.conversation_summary_prompt(state.get("conversation_summary", ""), older)
            )
            update = {"conversation_summary": result.summary, "summarized_upto": new_upto}
            print(f"[remember] conversation summary now covers {new_upto} messages")
    except Exception as e:
        print(f"[remember] skipped: {type(e).__name__}: {e}")
    return update
