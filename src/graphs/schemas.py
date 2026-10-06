# src/graphs/schemas.py
"""
What the LLM must return. Each class is passed to llm.with_structured_output(...),
so the reply comes back as a validated Python object instead of a string to parse.

Field descriptions are sent to the model as instructions, so keep them precise.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class Intent(BaseModel):
    intent: Literal["ask", "summary", "flashcard", "quiz"] = Field(
        description="ask = a question about the document; summary = summarize; "
                    "flashcard = make flashcards; quiz = test me / make a quiz"
    )
    scope: Literal["document", "topic"] = Field(
        description="document = the whole PDF (e.g. 'summarize this PDF'); topic = one specific topic"
    )
    topic: str = Field(
        description="The user's request rewritten as a standalone search query, using the chat history "
                    "to resolve words like 'that' or '这个'. Same language as the user. "
                    "Empty string if scope is document."
    )


# ---------------------------------------------------------------------------
# Answers and summaries
# ---------------------------------------------------------------------------
class Answer(BaseModel):
    answer: str = Field(description="The answer, based only on the provided pages")
    pages: List[int] = Field(description="Page numbers the answer is taken from")
    found: bool = Field(description="False if the provided pages do not contain the answer")


class KeyPoint(BaseModel):
    text: str
    page: int = Field(description="Page number this point comes from")


class TopicSummary(BaseModel):
    main_idea: str = Field(description="One-sentence summary")
    key_points: List[KeyPoint] = Field(description="3-7 key points, each with its source page")


class SectionSummary(BaseModel):
    title: str = Field(description="Section title (keep the original title if one is given)")
    main_idea: str = Field(description="One-sentence summary of the section")
    key_points: List[KeyPoint] = Field(description="3-7 key points, each with its source page")
    key_terms: List[str] = Field(description="Important terms or concepts introduced in this section")


class DocOverview(BaseModel):
    subject: str = Field(description="Subject of the document, e.g. 'Data Structures textbook, Ch.1-5'")
    overview: str = Field(description="A short paragraph describing what the whole document covers")


class FaithfulnessCheck(BaseModel):
    unsupported: List[str] = Field(
        description="Key points from the summary that the source pages do NOT support. Empty list if all are supported."
    )


# ---------------------------------------------------------------------------
# Flashcards and quizzes
# ---------------------------------------------------------------------------
class Flashcard(BaseModel):
    front: str = Field(description="Term or question")
    back: str = Field(description="Definition or answer")
    page: int = Field(description="Source page")


class FlashcardSet(BaseModel):
    cards: List[Flashcard]


class Question(BaseModel):
    id: str = Field(description="q1, q2, ...")
    type: Literal["mcq", "fill_blank", "short"]
    concept: str = Field(description="The concept this question tests, 1-4 words, used to track mastery")
    question: str
    options: Optional[List[str]] = Field(
        None, description="mcq only: exactly 4 options written as 'A. ...', 'B. ...', 'C. ...', 'D. ...'"
    )
    answer: str = Field(description="mcq: a single letter A-D; fill_blank: the missing word(s); short: a 1-2 sentence reference answer")
    page: int = Field(description="Source page")


class Quiz(BaseModel):
    questions: List[Question]


class ShortAnswerGrade(BaseModel):
    correct: bool = Field(description="True if the student's answer has the same meaning as the reference answer")
    feedback: str = Field(description="One sentence: what was right or missing")
