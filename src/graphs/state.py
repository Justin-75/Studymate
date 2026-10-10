# src/graphs/state.py
"""
What flows between the study-graph nodes, and what the checkpointer saves per thread_id.

Big data (all chunks, indexes, section summaries) stays in the database / cache and is
looked up by doc_id. State only holds the conversation and this turn's working data.
Long-term memory per document (topics studied, mastery, across all chats) lives in the
LangGraph store, not here: see memory.py.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from langgraph.graph import MessagesState


class StudyState(MessagesState):          # inherits: messages + add_messages reducer
    doc_id: str
    intent: Optional[Literal["ask", "summary", "flashcard", "quiz"]]
    scope: Optional[Literal["document", "topic"]]
    topic: Optional[str]                  # standalone search query written by the router
    follow_up: bool                       # the message continues the last reply's topic
    context: List[dict]                   # THIS turn's chunks; kept after the turn, so a follow-up starts from them
    notes: Optional[dict]                 # summarize_topic's TopicNotes for this turn
    hops: int                             # retrieval rounds this turn (the agent loop stops at MAX_HOPS)
    queries: List[str]                    # search queries used this turn
    conversation_summary: str             # running summary of the turns before the last few messages
    summarized_upto: int                  # messages[:summarized_upto] are inside conversation_summary
    flashcards: Optional[List[dict]]
    quiz: Optional[dict]                  # Quiz.model_dump()
    answers: Optional[Dict[str, str]]     # question id -> the student's answer
    graded: Optional[List[dict]]          # per-question results of the last quiz
    mastery: Dict[str, float]             # concept -> score of its latest quiz (0..1), all chats of this document
    weak_concepts: List[str]              # concepts below the threshold in the last quiz
    round: int                            # quiz rounds in the current learning loop
