# src/generation/smoke_test.py
from __future__ import annotations

from src.db.models import GenerateRequest, SearchHit
from src.generation.generator import generate_material
from src.generation.quiz_grader import grade_quiz


def main() -> None:
    hits = [
        SearchHit(chunk_id="c1", doc_id="d1", page_no=1, chunk_index=0, score=0.9,
                  text="A hash table uses a hash function to map keys to buckets. Collisions occur when two keys map to the same bucket."),
        SearchHit(chunk_id="c2", doc_id="d1", page_no=2, chunk_index=0, score=0.6,
                  text="Binary search runs in O(log n) time on a sorted array. It repeatedly halves the search range."),
    ]

    print("=== SUMMARY ===")
    req = GenerateRequest(doc_id="d1", mode="summary", query="hash collisions", top_k=8)
    out = generate_material(req, hits)
    print(out.content)

    print("\n=== FLASHCARDS ===")
    req = GenerateRequest(doc_id="d1", mode="flashcards", query="hash collisions", top_k=8)
    out = generate_material(req, hits)
    print(out.content)

    print("\n=== QUIZ ===")
    req = GenerateRequest(doc_id="d1", mode="quiz", query="hash collisions", top_k=8)
    out = generate_material(req, hits)
    print(out.content)

    quiz = out.content
    answers = {"q1": "A", "q2": "B"}  # example
    graded = grade_quiz(quiz, answers)
    print("\n=== GRADED ===")
    print(graded)

    print("\n✅ Generation smoke test ran")


if __name__ == "__main__":
    main()
