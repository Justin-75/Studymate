# src/graphs/state.py
"""
What flows between the study-graph nodes, and what the checkpointer saves per thread_id.

Big data (all chunks, indexes, section summaries) stays in the database / cache and is
looked up by doc_id. State only holds the conversation and this turn's working data.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from langgraph.graph import MessagesState


class StudyState(MessagesState):          # inherits: messages + add_messages reducer
    doc_id: str
    intent: Optional[Literal["ask", "summary", "flashcard", "quiz"]]
    scope: Optional[Literal["document", "topic"]]
    topic: Optional[str]                  # standalone search query written by the router
    context: List[dict]                   # THIS turn's retrieved chunks; no reducer, so replaced each turn
    flashcards: Optional[List[dict]]
    quiz: Optional[dict]                  # Quiz.model_dump()
    answers: Optional[Dict[str, str]]     # question id -> the student's answer
    graded: Optional[List[dict]]          # per-question results of the last quiz
    mastery: Dict[str, float]             # concept -> score of its latest quiz (0..1)
    weak_concepts: List[str]              # concepts below the threshold in the last quiz
    round: int                            # quiz rounds in the current learning loop
