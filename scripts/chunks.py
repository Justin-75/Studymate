"""Sample chunks spread across a document, to be turned into benchmark questions."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db.repository import Repository

DOC_ID   = sys.argv[1]
OUT      = sys.argv[2]                    # e.g. Data/samples/algo.jsonl
N        = int(sys.argv[3]) if len(sys.argv) > 3 else 60
PAGE_MIN = int(sys.argv[4]) if len(sys.argv) > 4 else 1      # skip front matter
PAGE_MAX = int(sys.argv[5]) if len(sys.argv) > 5 else 10**6  # skip index
MIN_CHARS = 300

random.seed(42)

chunks = [c for c in Repository("Data/Database/app.db").get_chunks(DOC_ID)
          if PAGE_MIN <= c.page_no <= PAGE_MAX and len(c.text) >= MIN_CHARS]
print(f"{len(chunks)} usable chunks between pages {PAGE_MIN}-{PAGE_MAX}")

# one chunk per bin, so the sample is spread across the whole document
bins, size = [], max(1, len(chunks) // N)
for i in range(0, len(chunks), size):
    group = chunks[i:i + size]
    if group:
        bins.append(random.choice(group))
sample = bins[:N]

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for c in sample:
        f.write(json.dumps({"chunk_id": c.chunk_id, "doc": DOC_ID,
                            "page_no": c.page_no, "text": c.text},
                           ensure_ascii=False) + "\n")
print(f"wrote {len(sample)} chunks to {OUT}, pages {sample[0].page_no}–{sample[-1].page_no}")