# src/graphs/prompts.py
"""
Every prompt in one place. Each function returns a plain string; the node pairs it with
llm.with_structured_output(<Schema>), so the prompts describe the TASK, not the JSON format.

Documents are Chinese or English, so every prompt says: write in the document's language.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from langchain_core.messages import AnyMessage

LANG_RULE = "Write in the same language as the document text (Chinese document -> Chinese, English -> English)."
GROUNDING_RULE = "Use ONLY the provided pages. Do not add facts from outside them."


def format_context(chunks: Sequence[dict], max_chars: int = 12000) -> str:
    """[p.12] text ... — the page label lets the model cite pages."""
    parts, used = [], 0
    for c in chunks:
        block = f"[p.{c['page_no']}] {c['text']}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def format_history(messages: Sequence[AnyMessage], last_n: int = 6) -> str:
    lines = []
    for m in list(messages)[-last_n:]:
        who = "User" if m.type == "human" else "Assistant"
        lines.append(f"{who}: {str(m.content)[:500]}")
    return "\n".join(lines) or "(no earlier messages)"


# ---------------------------------------------------------------------------
# Study graph
# ---------------------------------------------------------------------------
def router_prompt(history: str, user_msg: str, subject: str, chosen: str | None = None) -> str:
    if chosen:      # the user pressed a button, so only scope and topic are left to decide
        task = f"""The user pressed the "{chosen}" button, so the intent is {chosen}.
Decide whether it is about the whole document or one topic: the topic is the one named in the
message; if the message names none, it is the topic of the recent conversation; if there is no
earlier conversation either, the scope is the whole document."""
    else:
        task = """Decide what the user wants (ask / summary / flashcard / quiz) and whether it is about the whole
document or one topic."""
    return f"""You route requests in a study app. The user has uploaded a document.
Document subject: {subject or "unknown"}

Recent conversation:
{history}

New user message: {user_msg}

{task}
Then rewrite the request as a standalone search query: replace words like
"that", "it", "这个", "那个" with what they refer to in the conversation, and keep technical terms."""


def answer_prompt(context: str, question: str) -> str:
    return f"""Answer the student's question using the pages below.
{GROUNDING_RULE} If the pages do not contain the answer, set found=false and say so briefly.
{LANG_RULE}

Pages:
{context}

Question: {question}"""


def topic_summary_prompt(context: str, topic: str) -> str:
    return f"""Summarize what the pages below say about: {topic}
{GROUNDING_RULE} Give each key point the page it comes from.
{LANG_RULE}

Pages:
{context}"""


def flashcard_prompt(context: str, topic: str, n: int = 8) -> str:
    return f"""Make up to {n} study flashcards about "{topic}" from the pages below.
Front = a term or a short question; back = a precise definition or answer; add the source page.
Prefer definitions, key properties and comparisons. {GROUNDING_RULE}
{LANG_RULE}

Pages:
{context}"""


def quiz_prompt(context: str, topic: str, n: int = 5, focus: List[str] | None = None) -> str:
    focus_line = (
        f"Focus on these concepts the student got wrong last time: {', '.join(focus)}. "
        "Ask about them in a different way than before.\n" if focus else ""
    )
    return f"""Write a {n}-question quiz about "{topic}" from the pages below.
{focus_line}Mix types: mostly mcq, plus fill_blank and short. For each question give the concept it tests,
the correct answer and the source page. mcq answers must be one letter A-D, with exactly 4 options.
Wrong options must be plausible. {GROUNDING_RULE}
{LANG_RULE}

Pages:
{context}"""


def grade_short_prompt(question: str, reference: str, student: str) -> str:
    return f"""Grade a student's answer.
Question: {question}
Reference answer: {reference}
Student answer: {student}

Mark correct if it has the same meaning as the reference, even with different wording or language.
Give one sentence of feedback in the language of the question."""


def review_prompt(context: str, concepts: List[str]) -> str:
    return f"""The student missed quiz questions on: {', '.join(concepts)}.
Write a short review of exactly these concepts from the pages below, so they can try again.
{GROUNDING_RULE} Give each key point its source page.
{LANG_RULE}

Pages:
{context}"""


# ---------------------------------------------------------------------------
# Ingest graph (per-section summaries)
# ---------------------------------------------------------------------------
def section_summary_prompt(title: str, start: int, end: int, text: str, issues: List[str] | None = None) -> str:
    fix = ""
    if issues:
        fix = "A reviewer found problems in your previous summary. Fix them:\n- " + "\n- ".join(issues) + "\n\n"
    return f"""{fix}Summarize this section of a study document.
Section: {title or "(untitled)"}, pages {start}-{end}
Cover every major topic in the section. Each key point must cite a page between {start} and {end}.
{GROUNDING_RULE}
{LANG_RULE}

Text:
{text}"""


def faithfulness_prompt(summary: Dict, text: str) -> str:
    points = "\n".join(f"- (p.{kp['page']}) {kp['text']}" for kp in summary.get("key_points", []))
    return f"""Check a summary against its source text.
List every key point that the source text does NOT support (wrong facts, invented details).
Return an empty list if all points are supported.

Key points:
{points}

Source text:
{text}"""


def overview_prompt(section_summaries: List[Dict]) -> str:
    lines = "\n".join(
        f"- {s['title']} (p.{s['start']}-{s['end']}): {s['main_idea']}" for s in section_summaries
    )
    return f"""Here are summaries of every section of one document:
{lines}

Name the subject of the document and write a short overview of what it covers.
{LANG_RULE}"""
