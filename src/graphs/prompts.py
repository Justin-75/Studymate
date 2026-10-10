# src/graphs/prompts.py
"""
Every prompt in one place. Each function returns a plain string; the node pairs it with
llm.with_structured_output(<Schema>), so the prompts describe the TASK, not the JSON format.

Every prompt has the same four parts, so the model reads what to do instead of guessing it:
    ROLE    who the model is in StudyMate (the persona) and its one job
    INPUT   the labelled blocks that follow
    STEPS   numbered actions in order, each with a concrete output
    RULES   grounding and language

Documents are Chinese or English. The code detects the document's language (store.doc_language) and every
prompt that writes for the student names it (lang_rule), so the reply follows the PDF, not the question.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from langchain_core.messages import AnyMessage

LANGS = {"en": "English", "zh": "Chinese (简体中文)"}


def lang_rule(lang: str = "") -> str:
    """The language to write in: the document's, which the code detected (lang = "en" / "zh")."""
    if lang not in LANGS:
        return "- Language: write in the language of the document text (Chinese document -> Chinese, English -> English)."
    return (f"- Language: write in {LANGS[lang]}, the language of the document, even when the question or the"
            f" conversation is in another language. Keep names, formulas and technical terms as the pages write them.")
GROUNDING_RULE = "- Use ONLY the given pages. Every fact you write must appear on a page you cite."
CARD_RULE = ("- A page block may start with a card line (topic + description of that chunk, written from the page"
             " text). Use the card to see what the block is about; take facts and page numbers from the page text.")
PAGES_LABEL = "Pages (each block: [p.N], the chunk's card line if it has one, then the page text):"
TASKS = {
    "ask": "answer the question",
    "summary": "summarize the topic",
    "flashcard": "make flashcards on the topic",
    "quiz": "write a quiz on the topic",
}


def format_block(c: dict) -> str:
    """One chunk for a prompt: the page label (lets the model cite pages), its card if any, the text."""
    card = f"Card: {c['card_topic']} - {c['card_description']}\n" if c.get("card_topic") else ""
    return f"[p.{c['page_no']}] {card}{c['text']}"


def format_context(chunks: Sequence[dict], max_chars: int = 40000) -> str:
    """The page blocks of a prompt. 40000 chars fits 8 chunks of 512 words plus their cards."""
    parts, used = [], 0
    for c in chunks:
        block = format_block(c)
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


def format_conversation(summary: str, messages: Sequence[AnyMessage], last_n: int = 4) -> str:
    """The running summary of older turns, then the last few messages word for word."""
    head = f"Summary of earlier turns:\n{summary}\n\n" if summary else ""
    return f"{head}Recent messages:\n{format_history(messages, last_n)}"


def format_notes(notes: Optional[Dict]) -> str:
    if not notes:
        return "(no notes)"
    points = "\n".join(f"- {kp['text']} (p.{kp['page']})" for kp in notes.get("key_points", []))
    return f"{notes.get('main_idea', '')}\n{points}"


def format_memory(topics: List[Dict], weak: List[str]) -> str:
    """Long-term memory of this document (all chats): topics studied and concepts missed in quizzes."""
    lines = [f"- {t['topic']}: {t['takeaway']}" for t in topics]
    if weak:
        lines.append(f"- Concepts missed in quizzes: {', '.join(weak)}")
    return "\n".join(lines) or "(nothing yet)"


# ---------------------------------------------------------------------------
# Study graph
# ---------------------------------------------------------------------------
def router_prompt(conversation: str, memory: str, user_msg: str, subject: str, chosen: str | None = None) -> str:
    if chosen:      # the user pressed a button, so the intent is fixed
        step1 = f'1. Set intent = "{chosen}" (the student pressed the {chosen} button).'
    else:
        step1 = """1. Set intent from the new message:
   ask = a question or a request for an explanation; summary = summarize / outline / overview;
   flashcard = flashcards; quiz = test me / quiz / exercises."""
    return f"""ROLE
You are StudyMate's request router. You turn one chat message from a student into a routing
decision for a study app about one uploaded document.

INPUT
Document subject: {subject or "unknown"}
Earlier study of this document (all chats):
{memory}
Conversation in this chat:
{conversation}
New message: {user_msg}

STEPS
{step1}
2. Set scope = "document" if the message is about the whole document ("this PDF", "the whole book",
   "all chapters"), or names no topic while the conversation and the earlier study are empty.
   Otherwise set scope = "topic".
3. Set follow_up = true if the new message continues the topic of the last assistant reply: it asks
   for more detail, the next step, an example, or uses "it", "that", "这个", "那个" for that topic.
   Set follow_up = false if it starts another topic or there is no assistant reply yet.
4. Write topic = the request as one standalone search query. Replace "it", "that", "这个", "那个"
   with the thing they point to in the conversation or the earlier study. Keep technical terms
   exactly as written. Write topic in the language of the new message. If scope = "document",
   write an empty string."""


def topic_notes_prompt(context: str, topic: str, intent: str, conversation: str, lang: str = "") -> str:
    return f"""ROLE
You are StudyMate's research note-taker. You read the retrieved pages of a document, write notes on
one topic, and report what the pages are still missing for the task.

INPUT
Topic: {topic}
Task the notes are for: {TASKS.get(intent, TASKS["ask"])}
Conversation in this chat:
{conversation}
{PAGES_LABEL}
{context}

STEPS
1. Read each block's card line (if it has one), then mark the page blocks that talk about the topic.
2. Write main_idea: one sentence that states what the marked pages say about the topic. If no page
   block is marked, write main_idea = "The pages do not cover this topic.", key_points = [],
   covered = false, and go to step 5.
3. Write 3-7 key_points. Each key point is one fact from one marked page, with that page number.
   Copy numbers, names, formulas and definitions exactly as the page writes them.
4. Set covered = true if the key points hold everything the task needs: for a question, the facts
   that answer it; for a summary, flashcards or a quiz, at least 3 facts about the topic.
   Otherwise set covered = false.
5. If covered = false: write missing = the facts the pages lack, in one sentence, and next_query = a
   search query for exactly those facts, in the language of the document. If covered = true: write
   missing = "" and next_query = "".

RULES
{GROUNDING_RULE}
{CARD_RULE}
{lang_rule(lang)}"""


def answer_prompt(context: str, question: str, notes: str = "", conversation: str = "", lang: str = "") -> str:
    return f"""ROLE
You are StudyMate's tutor. You answer a student's question about their document, using only the
pages shown to you, and you cite the pages.

INPUT
Conversation in this chat:
{conversation or "(no earlier messages)"}
Topic notes (written from the same pages):
{notes or "(no notes)"}
{PAGES_LABEL}
{context}
Question: {question}

STEPS
1. Find the sentences in the pages that answer the question. Use the topic notes to locate them.
2. If no page answers the question: set found = false, pages = [], and write one sentence saying
   the pages shown do not answer it. Stop here.
3. Write the answer: first the direct answer in 1-2 sentences, then the supporting details from the
   pages. If the question follows up on the previous reply, start from what that reply said and add
   only the new information.
4. Set pages = the page numbers of the sentences you used, and found = true.

RULES
{GROUNDING_RULE}
{CARD_RULE}
{lang_rule(lang)}"""


def flashcard_prompt(context: str, topic: str, notes: str = "", n: int = 8, lang: str = "") -> str:
    return f"""ROLE
You are StudyMate's flashcard writer. You turn the pages of a document into study cards.

INPUT
Topic: {topic}
Topic notes (written from the same pages):
{notes or "(no notes)"}
{PAGES_LABEL}
{context}

STEPS
1. Pick up to {n} facts about the topic from the notes and the pages: definitions first, then key
   properties, then comparisons.
2. For each fact write one card: front = the term or a short question (max 15 words); back = the
   definition or answer from the page (max 40 words); page = the page the fact is on.
3. Delete any card that repeats the fact of another card.

RULES
{GROUNDING_RULE}
{CARD_RULE}
{lang_rule(lang)}"""


def quiz_prompt(context: str, topic: str, n: int = 5, focus: List[str] | None = None, notes: str = "",
                lang: str = "") -> str:
    focus_step = (f"\n   Give each of these concepts the student missed last time at least one question, with a"
                  f" different fact or wording than a plain definition: {', '.join(focus)}." if focus else "")
    n_mcq = max(1, n - 2)
    return f"""ROLE
You are StudyMate's quiz writer. You write quiz questions that test a student on a document.

INPUT
Topic: {topic}
Topic notes (written from the same pages):
{notes or "(no notes)"}
{PAGES_LABEL}
{context}

STEPS
1. Pick {n} different facts about the topic from the notes and the pages.{focus_step}
2. Write {n_mcq} mcq questions: exactly 4 options written "A. ...", "B. ...", "C. ...", "D. ...";
   exactly one option is correct; the 3 wrong options use terms from the same pages that are false
   for this question; answer = the letter of the correct option.
3. Write 1 fill_blank question: a sentence from a page with one key term replaced by "____";
   answer = the removed term.
4. Write 1 short question that takes 1-2 sentences to answer; answer = a 1-2 sentence reference
   answer from the page.
5. For every question set concept = the concept it tests (1-4 words) and page = the page of its fact.

RULES
{GROUNDING_RULE}
{CARD_RULE}
{lang_rule(lang)}"""


def grade_short_prompt(question: str, reference: str, student: str) -> str:
    return f"""ROLE
You are StudyMate's grader. You grade one written quiz answer against a reference answer.

INPUT
Question: {question}
Reference answer: {reference}
Student answer: {student}

STEPS
1. List the key facts in the reference answer.
2. Look for each key fact in the student answer. Different wording or another language still counts.
3. Set correct = true if the student answer states every key fact and nothing that contradicts the
   reference. Otherwise set correct = false.
4. Write feedback: one sentence. If correct, name what was right; if not, name the missing or
   wrong fact.

RULES
- Language: write the feedback in the language of the question."""


def review_prompt(context: str, concepts: List[str], lang: str = "") -> str:
    return f"""ROLE
You are StudyMate's review tutor. A student missed quiz questions; you re-teach exactly those
concepts from the document before the next quiz.

INPUT
Missed concepts: {', '.join(concepts)}
{PAGES_LABEL}
{context}

STEPS
1. For each missed concept, find the page sentences that explain it.
2. Write main_idea: one sentence the student must remember about these concepts.
3. Write key_points: 1-2 per missed concept, each one fact with its page number.

RULES
{GROUNDING_RULE}
{CARD_RULE}
{lang_rule(lang)}"""


def conversation_summary_prompt(previous: str, new_messages: str) -> str:
    return f"""ROLE
You are StudyMate's memory keeper. You keep a short running summary of one study chat, so later
replies can build on it without rereading every message.

INPUT
Summary so far:
{previous or "(empty)"}
New messages:
{new_messages}

STEPS
1. Start from the summary so far.
2. For each new student request, add one bullet: the topic asked and the main fact of the reply,
   with its pages.
3. For each quiz result in the new messages, add one bullet: concepts right and concepts wrong.
4. Delete bullets that a newer message replaced.
5. Keep the summary under 150 words.

RULES
- Language: write in the language the student uses."""


# ---------------------------------------------------------------------------
# Ingest graph (chunk cards, then per-section summaries)
# ---------------------------------------------------------------------------
def chunk_cards_prompt(chunks: str) -> str:
    return f"""ROLE
You are StudyMate's indexer. You label each chunk of a document with a card, so the chunk can be
found later by its topic.

INPUT
Chunks, each starting with its id and page, like "[c1] (p.12)":
{chunks}

STEPS
For every chunk, in the given order:
1. id = the chunk's id exactly as given (c1, c2, ...).
2. topic = the chunk's main subject in 2-8 words, a noun phrase with the document's own terms
   (e.g. "Red-black tree insertion"). For a chunk that is only references, an index, a table of
   contents or code, name what it is (e.g. "Bibliography", "Code: training loop").
3. description = 1-2 sentences (max 40 words) saying what the chunk explains, defines, proves or
   lists, with its key terms.

RULES
- Write exactly one card per chunk.
- Use only the chunk's own text.
- Language: write in the language of the chunk."""


def section_summary_prompt(title: str, start: int, end: int, cards: str, issues: List[str] | None = None,
                           lang: str = "") -> str:
    fix = ""
    if issues:
        fix = ("\n5. A reviewer found these problems in your previous summary. Fix each one:\n   - "
               + "\n   - ".join(issues))
    return f"""ROLE
You are StudyMate's section summarizer. You summarize one section of a document from the cards of
its chunks (each card = the topic and description of one chunk, with its page).

INPUT
Section: {title or "(untitled)"}, pages {start}-{end}
Cards:
{cards}

STEPS
1. Group the cards by topic.
2. Write title = the section title as given (or a 2-8 word title if none is given).
3. Write main_idea = one sentence on what the whole section covers.
4. Write 3-7 key_points, one per major topic group, each citing a page between {start} and {end}
   taken from that group's cards. Then write key_terms = the important terms the cards name.{fix}

RULES
- Use only the cards. Every key point must cite a page from a card.
{lang_rule(lang)}"""


def faithfulness_prompt(summary: Dict, text: str) -> str:
    points = "\n".join(f"- (p.{kp['page']}) {kp['text']}" for kp in summary.get("key_points", []))
    return f"""ROLE
You are StudyMate's fact checker. You check a section summary against the section's source text.

INPUT
Key points:
{points}
Source text:
{text}

STEPS
1. For each key point, find the sentence(s) of the source text that it restates.
2. List every key point that has no such sentence, or that changes a number, name or relation.
3. Return an empty list if every key point has a source sentence.

RULES
- Copy each listed key point word for word."""


def overview_prompt(section_summaries: List[Dict], lang: str = "") -> str:
    lines = "\n".join(
        f"- {s['title']} (p.{s['start']}-{s['end']}): {s['main_idea']}" for s in section_summaries
    )
    return f"""ROLE
You are StudyMate's overview writer. You describe a whole document from its section summaries.

INPUT
Section summaries, in document order:
{lines}

STEPS
1. Write subject = the document type and subject in under 12 words (e.g. "Algorithms textbook, chapters 1-35").
2. Write overview = 3-5 sentences on what the document covers, in section order.

RULES
- Use only the section summaries.
{lang_rule(lang)}"""
