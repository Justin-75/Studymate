import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import ingest_pdf
from src.db.repository import Repository
from src.retrieval.tfidf_index import TfidfIndex
from src.retrieval.search import search_chunks

PDF = sys.argv[1]
QUERY = sys.argv[2]
DB = "Data/Database/app.db"

res = ingest_pdf(PDF, DB)
print(f"doc_id={res.doc_id}  pages={res.num_pages}  chunks={res.num_chunks}  ocr={res.used_ocr}")

hits = search_chunks(Repository(DB), TfidfIndex("Data/Cache/tfidf"), res.doc_id, QUERY, top_k=5)
for rank, h in enumerate(hits, 1):
    print(f"#{rank}  page {h.page_no:>4}  score {h.score:.3f}  {h.text[:80]!r}")