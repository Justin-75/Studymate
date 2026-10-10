# scripts/eval_retrieval.py
"""
Interactive retrieval benchmark for StudyMate. No arguments needed:

    python scripts/eval_retrieval.py

1. Pick a document from the numbered list (or paste a PDF path to add a new one).
2. Type a question -> you see the top 5 pages the current retriever returns.
3. Type the gold page(s) that really answer it -> the question is saved.
4. Type `run` any time to score every saved question (Recall@k, MRR) and
   write CSVs to eval_out/<timestamp>/ for Excel.

To only score (no menu):  python scripts/eval_retrieval.py run
Only some retrievers:     python scripts/eval_retrieval.py run hybrid_rrf+reranker hybrid_rrf+reranker@128
(names as in RETRIEVERS below)

Child-size sweep (parent-child retrieval: children are searched, the LLM reads their 512-word parent):
    python scripts/eval_retrieval.py sweep                          hybrid_rrf+reranker, sizes 64 80 90 100 110 128 140
    python scripts/eval_retrieval.py sweep hybrid_rrf 64 100 128    another retriever / other sizes
    python scripts/eval_retrieval.py sweep hybrid_rrf 150 150o40    "150o40" = 150 words, 40 overlapping
Every "<name>@<size>" row is the same retriever run over child chunks of <size> words (cut from each
512-word chunk) instead of the 512-word chunks; "<name>@<size>o<overlap>" uses children that also share
~<overlap> words with their neighbour. bm25_raw, bge_m3_dense, hybrid_rrf and hybrid_rrf+reranker can take
a size, and `run` accepts such names too (hybrid_rrf+reranker@100, hybrid_rrf@150o40).
A missing child set is cut from the stored chunks on first use (no PDF, no LLM; stays in app.db:
python -m src.ingest --drop-children <size> [overlap] removes it). Before a row is timed, its indexes and models
are loaded, so ms/query is query time only; embed s is the one-time BGE-M3 cost of that size.
The per-size table (quality next to cost) is printed and saved as size_compare.csv.
hybrid_rrf+reranker@150 is what the app runs (CHILD_WORDS + CHILD_OVERLAP) before swapping in the parents; parents
share their children's pages, so the page-level scores are the app's.

Gold pages come in groups ("gold_groups"). Each group is one part of the answer;
any page inside a group satisfies that part. Two scores per cut-off k:
    Recall@k  (Hit)  at least one gold page is in the top k
    Full@k           every group has a page in the top k (all parts of the answer found)
For a one-group question the two are the same. Older rows that only have
"gold_pages" are read as one group. Unanswerable (L5) questions have no gold pages
and are scored by NoAnswer: the share where the retriever returned nothing above
its threshold. A row with "excluded": true is skipped by `run`.

Ranking is page-level: each retriever returns chunks, and a page is kept once, at
the position of its best (max-score) chunk. Set DEDUPE_PAGES = False to score raw
chunk lists the old way.

Saved data:
    eval/questions.jsonl   every question you added (one JSON per line)
    eval/docs.json         the short names you gave your documents
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys  
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db.repository import Repository  # noqa: E402
from src.retrieval.search import build_index_for_doc, search_chunks  # noqa: E402
from src.retrieval.tfidf_index import TfidfIndex  # noqa: E402
from src.retrieval import hybrid  # noqa: E402  (the same pipeline the app uses)
from src.pdf.chunker import CHILD_OVERLAP, CHILD_WORDS  # noqa: E402
from src.db.models import child_unit, parse_child_unit  # noqa: E402
from src.ingest import build_children  # noqa: E402
from src.retrieval import bm25_index, dense_index  # noqa: E402
from src.retrieval.reranker import get_reranker  # noqa: E402


DB_PATH = ROOT / "Data/Database/app.db"
TFIDF_ROOT = ROOT / "Data/Cache/tfidf"
EVAL_DIR = ROOT / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
NAMES_FILE = EVAL_DIR / "docs.json"
OUT_ROOT = ROOT / "eval_out"
KS = (1, 3, 5, 10, 20, 50)  # Recall@k cut-offs; MRR is computed at max(KS)
DEDUPE_PAGES = True         # one page appears once in a ranking, scored by its best chunk
CHUNK_DEPTH = 3             # fetch k * CHUNK_DEPTH chunks so k distinct pages remain after dedupe

LEVELS = {
    "1": ("L1", "same words as the page"),
    "2": ("L2", "paraphrased, different words"),
    "3": ("L3", "asked in the other language"),
    "4": ("L4", "needs 2+ pages"),
}

repo = Repository(str(DB_PATH))
index = TfidfIndex(str(TFIDF_ROOT))


# ---------------------------------------------------------------------------
# Retrievers. Each takes (doc_id, query, k) and returns a ranked list of
# (chunk_id, page_no, score, text). Add other retrievers here later to compare.
# ---------------------------------------------------------------------------
Hit = Tuple[str, int, float, str]


def tfidf_pipeline(doc_id: str, q: str, k: int) -> List[Hit]:
    # The app's OLD retrieval (before Oct 2026): TF-IDF + threshold + MMR diversity + max 2 chunks/page
    hits = search_chunks(repo, index, doc_id, q, top_k=k, auto_build=True)
    return [(h.chunk_id, h.page_no, h.score, h.text) for h in hits]


def _to_hits(scored: List[Tuple[str, float]]) -> List[Hit]:
    """(chunk_id, score) pairs -> Hit tuples, looking up page and text in the DB, order kept."""
    chunk_map = {c.chunk_id: c for c in repo.get_chunks_by_ids([cid for cid, _ in scored])}
    return [(cid, chunk_map[cid].page_no, s, chunk_map[cid].text) for cid, s in scored if cid in chunk_map]


def tfidf_raw(doc_id: str, q: str, k: int) -> List[Hit]:
    # TF-IDF scores only: no threshold, no MMR, no page cap. The fair baseline for bm25_raw.
    if not index.index_exists(doc_id):
        build_index_for_doc(repo, index, doc_id)
    return _to_hits(index.search(doc_id, q, top_k=k))


# The four retrievers below live in src/retrieval/hybrid.py and are shared with the app.
# These are thin wrappers that only plug in this script's repo.

def bm25_raw(doc_id: str, q: str, k: int) -> List[Hit]:
    # BM25 (jieba tokens) scores only: no threshold, no MMR, no page cap.
    return hybrid.bm25_hits(repo, doc_id, q, k)


def bge_m3_dense(doc_id: str, q: str, k: int) -> List[Hit]:
    # BGE-M3 dense vectors + FAISS exact search.
    return hybrid.dense_hits(repo, doc_id, q, k)


def hybrid_rrf(doc_id: str, q: str, k: int) -> List[Hit]:
    # BM25 + BGE-M3 dense, merged by Reciprocal Rank Fusion (50 candidates per retriever)
    return hybrid.hybrid_rrf(repo, doc_id, q, k)


def hybrid_rrf_rerank(doc_id: str, q: str, k: int) -> List[Hit]:
    # hybrid_rrf pool of 30 -> bge-reranker-v2-m3. This is what the app runs (STUDYMATE_RETRIEVAL_UNIT=chunks).
    return hybrid.hybrid_rrf_rerank(repo, doc_id, q, k)


# The same four retrievers over child chunks of any size, scored on the children themselves.
# A child's page is its parent's page, so the page-level scores compare directly with the 512-word rows.
# Row name "<retriever>@<size>", e.g. "hybrid_rrf+reranker@100"; the app uses CHILD_WORDS.
SIZED: Dict[str, Callable] = {        # retrievers that take unit= "chunks" or a child size
    "bm25_raw": hybrid.bm25_hits,
    "bge_m3_dense": hybrid.dense_hits,
    "hybrid_rrf": hybrid.hybrid_rrf,
    "hybrid_rrf+reranker": hybrid.hybrid_rrf_rerank,
}
SWEEP_SIZES = (64, 80, 90, 100, 110, 128, 140)   # child sizes `sweep` compares by default


CHILD_SET = re.compile(r"[1-9]\d*(o\d+)?")   # a child set: "100" (100 words) or "150o40" (150 words, 40 overlap)


def parse_row(name: str):
    """'hybrid_rrf+reranker@100' -> ('hybrid_rrf+reranker', 100); '...@150o40' -> (..., '150o40');
    any other name -> (name, None)."""
    base, _, size = name.partition("@")
    if CHILD_SET.fullmatch(size) and base in SIZED:
        return base, child_unit(*parse_child_unit(size))
    return name, None


def child_retriever(base: str, size: int) -> Callable[[str, str, int], List[Hit]]:
    fn = SIZED[base]

    def run(doc_id: str, q: str, k: int) -> List[Hit]:
        return fn(repo, doc_id, q, k, unit=size)

    run.__name__ = f"{base}@{size}"
    return run


def dedupe_pages(hits: List[Hit]) -> List[Hit]:
    """Keep each page once, at the position of its first (best-ranked) chunk, with the
    page's max chunk score. Lists sorted by score are unchanged apart from the removed
    duplicates; for MMR-ordered lists (tfidf_pipeline) the first-seen order is kept."""
    best: Dict[int, float] = {}
    for _, page, score, _ in hits:
        best[page] = max(score, best.get(page, score))
    seen, out = set(), []
    for cid, page, _, text in hits:
        if page in seen:
            continue
        seen.add(page)
        out.append((cid, page, best[page], text))
    return out


def page_level(fn: Callable[[str, str, int], List[Hit]]) -> Callable[[str, str, int], List[Hit]]:
    """Wrap a chunk retriever so it returns k distinct pages."""
    if not DEDUPE_PAGES:
        return fn

    def run(doc_id: str, q: str, k: int) -> List[Hit]:
        return dedupe_pages(fn(doc_id, q, k * CHUNK_DEPTH))[:k]

    run.__name__ = fn.__name__
    return run


RETRIEVERS: Dict[str, Callable[[str, str, int], List[Hit]]] = {
    name: page_level(fn) for name, fn in {
        "hybrid_rrf+reranker": hybrid_rrf_rerank,   # the 512-word chunks (STUDYMATE_RETRIEVAL_UNIT=chunks)
        "tfidf_pipeline": tfidf_pipeline,   # the app's old retrieval (kept as a baseline)
        "tfidf_raw": tfidf_raw,             # compare these two:
        "bm25_raw": bm25_raw,               # same conditions, only the scoring differs
        "bge_m3_dense": bge_m3_dense,       # meaning-based: BGE-M3 vectors + FAISS
        "hybrid_rrf": hybrid_rrf,           # BM25 + Dense + RRF
    }.items()
}
# the same four over the children the app uses (the rows above search the 512-word chunks)
APP_CHILDREN = child_unit(CHILD_WORDS, CHILD_OVERLAP)
RETRIEVERS.update({f"{b}@{APP_CHILDREN}": page_level(child_retriever(b, APP_CHILDREN)) for b in SIZED})
ALL_RETRIEVERS = dict(RETRIEVERS)


def retriever_for(name: str) -> Optional[Callable[[str, str, int], List[Hit]]]:
    """A listed retriever, or "<sized retriever>@<any size>"; None if the name is unknown."""
    if name in ALL_RETRIEVERS:
        return ALL_RETRIEVERS[name]
    base, size = parse_row(name)
    return page_level(child_retriever(base, size)) if size else None


def prepare(name: str, doc_ids: List[str]) -> Dict:
    """
    Before a row is timed: cut its child size if missing, load (or build) its BM25 and dense indexes and
    the models, so ms/query measures queries only. Returns the index size and the one-time embed cost.
    """
    base, size = parse_row(name)
    if base not in SIZED:
        return {}
    unit = size or "chunks"
    items = embed_s = 0.0
    for d in doc_ids:
        if size and not repo.has_children(d, *parse_child_unit(size)):
            build_children(d, str(DB_PATH), *parse_child_unit(size))
        items += bm25_index.get_index(repo, d, unit).N
        meta = dense_index.get_index(repo, d, unit).dir / "meta.json"
        embed_s += json.loads(meta.read_text(encoding="utf-8")).get("build_seconds", 0) if meta.exists() else 0
    if "reranker" in base:
        rerank_warm = get_reranker()
        rerank_warm.compute_score([["warm up", "warm up"]])
    return {"unit": size or 512, "items": int(items), "embed_s": round(embed_s, 1)}

LIVE = "tfidf_pipeline"  # the one shown while you type questions


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_questions() -> List[Dict]:
    if not QUESTIONS_FILE.exists():
        return []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_questions(questions: List[Dict]) -> None:
    EVAL_DIR.mkdir(exist_ok=True)
    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def save_names(names: Dict[str, str]) -> None:
    EVAL_DIR.mkdir(exist_ok=True)
    NAMES_FILE.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")


def list_documents() -> List[Dict]:
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute(
        "SELECT d.doc_id, d.filename, MAX(c.page_no), COUNT(c.chunk_id) FROM documents d "
        "JOIN chunks c ON c.doc_id = d.doc_id GROUP BY d.doc_id ORDER BY d.created_at"
    ).fetchall()
    docs = []
    for doc_id, fname, pages, chunks in rows:
        first = con.execute(
            "SELECT text FROM chunks WHERE doc_id=? ORDER BY page_no, chunk_index LIMIT 1", (doc_id,)
        ).fetchone()
        preview = re.sub(r"\s+", " ", first[0] if first else "")[:60]
        docs.append({"doc_id": doc_id, "filename": fname, "pages": pages, "chunks": chunks, "preview": preview})
    return docs


def page_text(doc_id: str, page: int) -> str:
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute(
        "SELECT text FROM chunks WHERE doc_id=? AND page_no=? ORDER BY chunk_index", (doc_id, page)
    ).fetchall()
    return "\n".join(r[0] for r in rows)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "quit"


def is_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def snippet(text: str, n: int = 110) -> str:
    return re.sub(r"\s+", " ", text)[:n]


def gold_groups(q: Dict) -> List[set]:
    """The question's answer parts. Old rows (gold_pages only) are one group."""
    if q.get("gold_groups"):
        return [set(g) for g in q["gold_groups"] if g]
    return [set(q["gold_pages"])] if q.get("gold_pages") else []


def first_hit_rank(hits: List[Hit], gold: set) -> int:
    for rank, (_, page, _, _) in enumerate(hits, start=1):
        if page in gold:
            return rank
    return 0  # 0 = not found (written as blank in the CSV)


def full_hit_rank(hits: List[Hit], groups: List[set]) -> int:
    """Rank at which EVERY group has a page in the list (0 = some group never found)."""
    ranks = [first_hit_rank(hits, g) for g in groups]
    return 0 if not ranks or 0 in ranks else max(ranks)


def parse_pages(text: str) -> Optional[List[int]]:
    nums = re.findall(r"\d+", text)
    return sorted({int(n) for n in nums}) if nums else None


def parse_groups(text: str) -> Optional[List[List[int]]]:
    """'508 | 509 | 516 518' -> [[508], [509], [516, 518]]. Without '|' it is one group."""
    parts = [parse_pages(p) for p in text.split("|")]
    if not parts or any(p is None for p in parts):
        return None
    return parts


def show_page(doc_id: str, page: int) -> None:
    t = page_text(doc_id, page)
    print(f"\n----- page {page} -----\n{t}\n----- end of page {page} -----\n" if t else "  (no text on that page)")


# ---------------------------------------------------------------------------
# Choosing / adding a document
# ---------------------------------------------------------------------------
def add_pdf(path_text: str) -> Optional[str]:
    path = Path(path_text.strip().strip('"').strip("'"))
    if not path.is_file() or path.suffix.lower() != ".pdf":
        print(f"  Not a PDF file: {path}")
        return None
    from src.ingest import ingest_pdf

    print(f"  Reading {path.name} ... (large books take a minute)")
    res = ingest_pdf(str(path), str(DB_PATH))
    build_index_for_doc(repo, index, res.doc_id)
    print(f"  Added: {res.num_pages} pages, {res.num_chunks} chunks, OCR used: {res.used_ocr}")
    if res.used_ocr:
        print("  Warning: this PDF had no text layer, so OCR was used. Results may be noisy.")
    return res.doc_id


def choose_document(names: Dict[str, str]) -> Optional[Dict]:
    while True:
        docs = list_documents()
        counts: Dict[str, int] = defaultdict(int)
        for q in load_questions():
            counts[q["doc_id"]] += 1

        print("\nDocuments")
        for i, d in enumerate(docs, start=1):
            label = names.get(d["doc_id"]) or d["filename"]
            print(f"  [{i}] {label:<28} {d['pages']:>5} pages  {counts[d['doc_id']]:>3} questions   \"{d['preview']}\"")
        print("\nType a number, paste a PDF path to add a new document, `run` to score everything, or `quit`.")
        choice = ask("> ")

        if choice.lower() in ("quit", "q", "exit"):
            return None
        if choice.lower() == "run":
            run_benchmark(names)
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(docs):
            doc = docs[int(choice) - 1]
        elif choice.lower().strip('"\'').endswith(".pdf"):
            doc_id = add_pdf(choice)
            if not doc_id:
                continue
            doc = next(d for d in list_documents() if d["doc_id"] == doc_id)
            if doc_id not in names:
                names[doc_id] = Path(choice.strip('"\'')).stem
        else:
            print("  Didn't understand that.")
            continue

        if doc["doc_id"] not in names:
            print(f"\n  This document has no name yet. It starts with: \"{doc['preview']}\"")
            name = ask("  Give it a short name (e.g. clrs, d2l-zh): ") or doc["filename"]
            names[doc["doc_id"]] = re.sub(r"\s+", "-", name)
        save_names(names)
        doc["name"] = names[doc["doc_id"]]
        return doc


# ---------------------------------------------------------------------------
# Adding questions for one document
# ---------------------------------------------------------------------------
HELP = """
  Type a question to search it. Other commands:
    p 400     show the text of page 400 (to check a gold page)
    list      show the questions saved for this document
    undo      delete the last question you saved
    run       score all saved questions and write the Excel files
    back      choose another document
    quit      exit
"""


def question_loop(doc: Dict, names: Dict[str, str]) -> bool:
    """Returns False when the user wants to quit the program."""
    print(f"\n=== {doc['name']}  ({doc['pages']} pages) ===")
    print(HELP)
    while True:
        text = ask(f"[{doc['name']}] question> ")
        low = text.lower()
        if not text:
            continue
        if low in ("quit", "exit", "q"):
            return False
        if low == "back":
            return True
        if low in ("help", "?"):
            print(HELP)
            continue
        if low == "run":
            run_benchmark(names)
            continue
        if low == "list":
            mine = [q for q in load_questions() if q["doc_id"] == doc["doc_id"]]
            for q in mine:
                gold = " | ".join(" ".join(map(str, sorted(g))) for g in gold_groups(q)) or "-"
                print(f"  {q['id']:<14} {q['level']}  gold {gold:<20} {q['q']}")
            print(f"  ({len(mine)} questions)")
            continue
        if low == "undo":
            qs = load_questions()
            if qs:
                removed = qs.pop()
                save_questions(qs)
                print(f"  Deleted {removed['id']}: {removed['q']}")
            continue
        m = re.fullmatch(r"p\s*(\d+)", low)
        if m:
            show_page(doc["doc_id"], int(m.group(1)))
            continue

        add_question(doc, text)


def add_question(doc: Dict, q: str) -> None:
    hits = RETRIEVERS[LIVE](doc["doc_id"], q, 5)
    print("\n  Top 5 from the current retriever:")
    if not hits:
        print("    (nothing above the score threshold)")
    for rank, (_, page, score, text) in enumerate(hits, start=1):
        print(f"    #{rank}  page {page:>5}  score {score:.3f}  {snippet(text)}")

    print("\n  Which page(s) really answer it? e.g. `400` or `400 405` (either page is enough)")
    print("  Answer needs several parts? Separate the parts with |, e.g. `400 | 512 513`")
    print("  `none` = the answer is not in this document · `p 400` = read a page first · Enter = don't save")
    while True:
        ans = ask("  gold pages> ")
        m = re.fullmatch(r"p\s*(\d+)", ans.lower())
        if not m:
            break
        show_page(doc["doc_id"], int(m.group(1)))
    if not ans:
        print("  Not saved.\n")
        return

    if ans.lower() == "none":
        gold, groups, level = [], [], "L5"
    else:
        groups = parse_groups(ans)
        if not groups:
            print("  No page numbers found. Not saved.\n")
            return
        gold = sorted({p for g in groups for p in g})
        bad = [p for p in gold if p < 1 or p > doc["pages"]]
        if bad:
            print(f"  Page(s) {bad} don't exist (this document has {doc['pages']} pages). Not saved.\n")
            return
        print("  Level: " + " · ".join(f"{k}={v[0]} {v[1]}" for k, v in LEVELS.items()))
        print("  L4 only if NO single page answers it (check for a page that already compares them).")
        default = "4" if len(groups) > 1 else "1"
        pick = ask(f"  level [{default}]> ") or default
        level = LEVELS.get(pick, LEVELS[default])[0]

    questions = load_questions()
    used = {x["id"] for x in questions}
    n = 1
    while f"{doc['name']}-{n:03d}" in used:
        n += 1
    row = {
        "id": f"{doc['name']}-{n:03d}",
        "doc_id": doc["doc_id"],
        "doc": doc["name"],
        "lang": "zh" if is_chinese(q) else "en",
        "level": level,
        "q": q,
        "gold_pages": gold,
    }
    if gold:
        row["gold_groups"] = groups
    questions.append(row)
    save_questions(questions)

    if gold:
        rank = first_hit_rank(hits, set(gold))
        result = f"found at rank {rank}" if rank else "NOT in the top 5"
    else:
        result = "unanswerable (L5), scored by whether the retriever returns nothing"
    print(f"  Saved {row['id']} ({level}, {row['lang']}), {result}. Total questions: {len(questions)}\n")


# ---------------------------------------------------------------------------
# Scoring everything
# ---------------------------------------------------------------------------
def run_benchmark(names: Dict[str, str]) -> None:
    questions = [q for q in load_questions() if not q.get("excluded")]
    if not questions:
        print("  No questions saved yet.")
        return
    kmax = max(KS)
    out_dir = OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_q_rows, text_rows = [], []
    # (hit_rank, full_rank, returned_nothing) per question; ranks are None for L5
    buckets: Dict[Tuple[str, str], List[Tuple[Optional[int], Optional[int], bool]]] = defaultdict(list)

    doc_ids = sorted({q["doc_id"] for q in questions})
    costs: Dict[str, Dict] = {}
    for name, fn in RETRIEVERS.items():
        costs[name] = prepare(name, doc_ids)
        t0 = time.time()
        for q in questions:
            groups = gold_groups(q)
            gold = set().union(*groups) if groups else set()
            hits = fn(q["doc_id"], q["q"], kmax)
            rank = first_hit_rank(hits, gold) if gold else 0
            full = full_hit_rank(hits, groups) if gold else 0
            pages = [h[1] for h in hits[:5]] + [""] * (5 - min(5, len(hits)))
            doc = names.get(q["doc_id"], q.get("doc", q["doc_id"][:8]))

            row = {
                "retriever": name, "id": q["id"], "doc": doc, "lang": q.get("lang", ""),
                "level": q.get("level", ""), "question": q["q"],
                "gold_groups": " | ".join(" ".join(map(str, sorted(g))) for g in groups) or "(none)",
                "n_groups": len(groups),
                **{f"top{i + 1}": pages[i] for i in range(5)},
                # blank = gold page not in the top kmax (never 0, so sorting by rank works)
                "first_hit_rank": (rank or "") if gold else "n/a",
                "full_rank": (full or "") if gold else "n/a",
            }
            for k in KS:
                row[f"hit@{k}"] = int(0 < rank <= k) if gold else "n/a"
            for k in KS:
                row[f"full@{k}"] = int(0 < full <= k) if gold else "n/a"
            row["RR"] = round(1 / rank, 4) if gold and rank else (0 if gold else "n/a")
            row["returned_nothing"] = int(not hits)
            per_q_rows.append(row)

            for i, (_, page, score, text) in enumerate(hits[:5], start=1):
                text_rows.append({
                    "retriever": name, "id": q["id"], "question": q["q"], "rank": i,
                    "page": page, "is_gold": int(page in gold), "score": round(score, 4),
                    "snippet": snippet(text, 300),
                })

            if gold:
                slices = ["ALL", f"doc={doc}", f"lang={q.get('lang', '?')}", f"level={q.get('level', '?')}"]
                if len(groups) > 1:
                    slices.append("multi-part")
                for sl in slices:
                    buckets[(name, sl)].append((rank, full, not hits))
            else:  # unanswerable: no Recall/MRR, only whether the retriever correctly returned nothing
                buckets[(name, f"level={q.get('level') or 'L5'}")].append((None, None, not hits))
        if costs[name]:
            costs[name]["ms_per_q"] = round((time.time() - t0) / len(questions) * 1000, 1)
        print(f"  {name} scored {len(questions)} questions in {time.time() - t0:.1f}s")

    summary_rows = []
    for (name, sl), items in sorted(buckets.items(), key=lambda kv: (kv[0][1] != "ALL", kv[0][1], kv[0][0])):
        ranks = [r for r, _, _ in items if r is not None]
        fulls = [f for _, f, _ in items if f is not None]
        row = {"retriever": name, "slice": sl, "n": len(items)}
        for k in KS:
            row[f"Recall@{k}"] = round(sum(0 < r <= k for r in ranks) / len(ranks), 4) if ranks else ""
        row[f"MRR@{kmax}"] = round(sum(1 / r for r in ranks if r) / len(ranks), 4) if ranks else ""
        for k in KS:
            row[f"Full@{k}"] = round(sum(0 < f <= k for f in fulls) / len(fulls), 4) if fulls else ""
        row["NoAnswer"] = round(sum(nothing for _, _, nothing in items) / len(items), 4)
        summary_rows.append(row)

    def write(path: Path, rows: List[Dict]) -> None:
        if rows:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    write(out_dir / "per_question.csv", per_q_rows)
    write(out_dir / "summary.csv", summary_rows)
    write(out_dir / "top5_text.csv", text_rows)

    def cell(v) -> str:
        return f"{v:>8.3f}" if isinstance(v, float) else f"{'-':>8}"

    print(f"\n  {'retriever':<25}{'slice':<20}{'n':>4}" + "".join(f"{'R@' + str(k):>8}" for k in KS)
          + f"{'MRR':>8}{'Full@5':>8}{'Full@10':>8}{'NoAns':>8}")
    for r in summary_rows:
        print(f"  {r['retriever']:<25}{r['slice'][:19]:<20}{r['n']:>4}"
              + "".join(cell(r[f'Recall@{k}']) for k in KS) + cell(r[f'MRR@{kmax}'])
              + cell(r["Full@5"]) + cell(r["Full@10"]) + cell(r["NoAnswer"]))
    print("\n  R@k = at least one gold page in the top k. Full@k = every part (gold group) found in the"
          "\n  top k; it differs from R@k only on multi-part questions (see the multi-part slice)."
          "\n  Rankings are page-level (each page once, best chunk score)." if DEDUPE_PAGES else "")
    print("\n  NoAns = share of questions where the retriever returned nothing (nothing above its score"
          "\n  threshold). On L5 (unanswerable) that is the correct answer, so higher is better; on every"
          "\n  other row it is a missed answer, so lower is better.")
    compare_sizes(summary_rows, costs, kmax, out_dir / "size_compare.csv", write)
    print(f"\n  Excel files written to {out_dir}\n")


def compare_sizes(summary_rows: List[Dict], costs: Dict[str, Dict], kmax: int, path: Path, write) -> None:
    """
    Per retriever: the 512-word chunks and every child size that was run, quality next to cost
    (index items, one-time embed seconds, ms per query), to find the efficient child size.
    """
    by = {(r["retriever"], r["slice"]): r for r in summary_rows}
    names = list(dict.fromkeys(r["retriever"] for r in summary_rows))
    mrr, rows = f"MRR@{kmax}", []
    for base in SIZED:
        sized = sorted(((parse_row(n)[1], n) for n in names if parse_row(n)[0] == base and parse_row(n)[1]),
                       key=lambda sn: parse_child_unit(sn[0]))       # by size, then overlap
        if not sized:
            continue
        group = ([(512, base)] if base in names else []) + sized
        for size, name in group:
            a, c = by.get((name, "ALL")), costs.get(name, {})
            if not a or not isinstance(a[mrr], float):
                continue
            words, overlap = parse_child_unit(size)
            label = f"{words}-word children" + (f" +{overlap}" if overlap else "")
            row = {"retriever": base, "unit": "512 (chunks)" if name == base else label,
                   "items": c.get("items", ""), "embed_s": c.get("embed_s", ""), "ms_per_q": c.get("ms_per_q", ""),
                   "R@1": a["Recall@1"], "R@3": a["Recall@3"], "R@5": a["Recall@5"], "R@10": a["Recall@10"], "MRR": a[mrr]}
            for sl in sorted(sl for (n, sl) in by if n == name and (sl.startswith("level=") or sl.startswith("lang="))):
                if isinstance(by[(name, sl)][mrr], float):
                    row[f"MRR {sl.split('=')[1]}"] = by[(name, sl)][mrr]
            rows.append(row)
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    write(path, [{k: r.get(k, "") for k in keys} for r in rows])
    levels = [k for k in keys if k.startswith("MRR L")]
    print("\n  Child size: quality next to cost (items = indexed units, embed s = one-time BGE-M3 time,"
          "\n  ms/q = query time with everything loaded). * = best MRR of the retriever.")
    print(f"  {'retriever':<20}{'unit':<20}{'items':>7}{'embed s':>9}{'ms/q':>8}{'R@1':>7}{'R@3':>7}{'R@5':>7}"
          f"{'R@10':>7}{'MRR':>8}" + "".join(f"{k[4:]:>7}" for k in levels))
    for base in dict.fromkeys(r["retriever"] for r in rows):
        group = [r for r in rows if r["retriever"] == base]
        best = max(r["MRR"] for r in group)
        for r in group:
            num = lambda v, w, f: f"{v:>{w}{f}}" if isinstance(v, (int, float)) else f"{'-':>{w}}"   # noqa: E731
            print(f"  {base:<20}{r['unit']:<20}{num(r['items'], 7, 'd')}{num(r['embed_s'], 9, '.1f')}"
                  f"{num(r['ms_per_q'], 8, '.0f')}" + "".join(num(r[k], 7, '.3f') for k in ("R@1", "R@3", "R@5", "R@10"))
                  + f"{r['MRR']:>7.3f}{'*' if r['MRR'] == best else ' '}"
                  + "".join(num(r.get(k, ""), 7, '.3f') for k in levels))


def main() -> None:
    print("StudyMate retrieval benchmark")
    names = load_json(NAMES_FILE, {})
    if sys.argv[1:2] in (["run"], ["sweep"]):
        if sys.argv[1] == "sweep":            # one retriever: the 512-word chunks + each child size
            args = sys.argv[2:]
            base = args[0] if args and not CHILD_SET.fullmatch(args[0]) else "hybrid_rrf+reranker"
            sizes = [a for a in args if CHILD_SET.fullmatch(a)] or list(SWEEP_SIZES)
            if base not in SIZED:
                raise SystemExit(f"sweep needs one of: {', '.join(SIZED)}")
            pick = [base] + [f"{base}@{n}" for n in sizes]
        else:
            pick = sys.argv[2:]               # optional: score only these retrievers
        unknown = [n for n in pick if retriever_for(n) is None]
        if unknown:
            raise SystemExit(f"Unknown retriever(s) {unknown}. Choose from: {', '.join(ALL_RETRIEVERS)}"
                             f", or <{' | '.join(SIZED)}>@<size> or @<size>o<overlap>")
        if pick:
            RETRIEVERS.clear()
            RETRIEVERS.update({n: retriever_for(n) for n in pick})
        run_benchmark(names)
        return
    while True:
        doc = choose_document(names)
        if doc is None or not question_loop(doc, names):
            print("Bye. Your questions are saved in", QUESTIONS_FILE)
            return


if __name__ == "__main__":
    main()
