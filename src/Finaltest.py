import json
from pathlib import Path

from src.ingest import ingest_pdf
from src.db.repository import Repository
from src.retrieval.tfidf_index import TfidfIndex
from src.retrieval.search import build_index_for_doc, search_chunks
from src.db.models import GenerateRequest
from src.generation.generator import generate_material
from src.generation.quiz_grader import grade_quiz
from src.knowledge_map import build_knowledge_map

DB_PATH = "Data/Database/app.db"
TFIDF_CACHE_ROOT = "Data/Cache/tfidf"

# ✅ Change this to your textbook PDF path (absolute is safest)
#PDF_PATH = r"E:\Kuliah\Project AI\MCP Template\Data\Raw PDF\Algolab.pdf"
PDF_PATH = "Data/Raw PDF/chp15.pdf"


def main():
    pdf = Path(PDF_PATH)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    print("\n=== 1) INGEST ===")
    res = ingest_pdf(
        file_path=str(pdf),
        db_path=DB_PATH,
        ocr_enabled=True,
        ocr_lang="eng+chi_sim",
        ocr_dpi=300,
        ocr_psm=3,
        ocr_preprocess=False,
        chunk_size=800,
        chunk_overlap=120,
        max_pages=None,  # set to 5 for quick tests
    )
    print("IngestResult:", res)

    repo = Repository(DB_PATH)
    index = TfidfIndex(TFIDF_CACHE_ROOT)

    print("\n=== 2) DB SANITY ===")
    pages = repo.get_pages(res.doc_id)
    chunks = repo.get_chunks(res.doc_id)
    print(f"Pages in DB: {len(pages)}")
    print(f"Chunks in DB: {len(chunks)}")

    # Quick look at cleaned text from a few pages
    for pno in [1, min(2, len(pages)), len(pages)]:
        if pno <= 0 or pno > len(pages):
            continue
        pg = pages[pno - 1]
        preview = (pg.text_clean or "").replace("\n", " ")
        print(f"Page {pg.page_no} clean preview:", preview[:250], "..." if len(preview) > 250 else "")

    print("\n=== 3) BUILD TF-IDF INDEX (if missing) ===")
    if not index.index_exists(res.doc_id):
        build_index_for_doc(repo, index, res.doc_id)
        print("Index built.")
    else:
        print("Index already exists.")

    print("\n=== 4) RETRIEVAL TEST ===")
    query = "introduction"  # change to something from your textbook topic
    hits = search_chunks(repo, index, res.doc_id, query, top_k=5, auto_build=True)
    print(f"Query: {query}")
    print(f"Hits: {len(hits)}")
    for h in hits:
        prev = (h.text or "").replace("\n", " ")
        print(f"- page={h.page_no} score={h.score:.4f} text={prev[:140]}{'...' if len(prev)>140 else ''}")

    print("\n=== 5) GENERATE: SUMMARY ===")
    req = GenerateRequest(doc_id=res.doc_id, mode="summary", query=query, top_k=5)
    out = generate_material(req, hits)
    print(json.dumps(out.content, ensure_ascii=False, indent=2))
    print("Citations:", out.citations)

    print("\n=== 6) GENERATE: FLASHCARDS ===")
    req = GenerateRequest(doc_id=res.doc_id, mode="flashcards", query=query, top_k=5)
    out_fc = generate_material(req, hits)
    print(json.dumps(out_fc.content, ensure_ascii=False, indent=2))

    print("\n=== 7) GENERATE: QUIZ + GRADE ===")
    req = GenerateRequest(doc_id=res.doc_id, mode="quiz", query=query, top_k=5)
    out_q = generate_material(req, hits)
    quiz = out_q.content
    print(json.dumps(quiz, ensure_ascii=False, indent=2))

    # Simulate answers: choose "A" for all questions
    answers = {}
    for q in quiz.get("quiz", []):
        answers[q["id"]] = "A"

    grade = grade_quiz(quiz, answers)
    print("Grade:", json.dumps(grade, ensure_ascii=False, indent=2))

    print("\n=== 8) KNOWLEDGE MAP ===")
    km = build_knowledge_map(res.doc_id, chunks, max_nodes=40, max_edges=120)
    # Print just counts + sample
    print(f"KM nodes={len(km.get('nodes', []))}, edges={len(km.get('edges', []))}")
    if km.get("nodes"):
        print("Sample node:", {k: km["nodes"][0].get(k) for k in ["id", "label", "pages"]})
    if km.get("edges"):
        print("Sample edge:", km["edges"][0])

    print("\n✅ SMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()
