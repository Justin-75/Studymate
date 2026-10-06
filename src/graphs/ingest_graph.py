# src/graphs/ingest_graph.py
r"""
Ingest graph: run ONCE per PDF, after ingest.py has stored its pages.
Builds one summary per section, checks them, and saves the result to
Data/Cache/summaries/<doc_id>.json for the study graph to reuse.

START -> plan_sections -> summarize -> check --(issues left, <= 2 revisions)--> summarize
                                           \--(clean, or out of revisions)----> finalize -> END

check = coverage rules in plain code + an LLM faithfulness check (are the key points on the cited pages?).

Run it:
    python -m src.graphs.ingest_graph <doc_id> [path/to/file.pdf]
(the PDF path is optional: with it, sections follow the PDF's table of contents)
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.graphs import prompts as P
from src.graphs.schemas import DocOverview, FaithfulnessCheck, SectionSummary
from src.graphs.store import building, repo, report_progress, save_doc_summary
from src.llm_client import structured_llm

MAX_SECTION_CHARS = 12000   # text per summary call; bigger chapters are split into parts
MAX_REVISIONS = 2
MAX_CONCURRENCY = 4         # parallel LLM calls in .batch()
CHECK_FAITHFULNESS = os.getenv("CHECK_FAITHFULNESS", "0") in ("1", "true", "yes", "on")   # off by default; set CHECK_FAITHFULNESS=1 in .env


class IngestState(TypedDict, total=False):
    doc_id: str
    pdf_path: Optional[str]
    sections: List[dict]                 # {title, start, end, text}
    summaries: List[Optional[dict]]      # aligned with sections; None = not written yet
    issues: Dict[int, List[str]]         # section index -> problems found by check
    checks: int                          # how many times check has run
    last_error: str                      # last LLM failure, reported if no section could be summarized
    result: dict


# ---------------------------------------------------------------------------
# 1. Plan sections: from the PDF's table of contents, else by page windows
# ---------------------------------------------------------------------------
def _toc_ranges(pdf_path: str, num_pages: int) -> List[dict]:
    try:
        try:
            import pymupdf                        # PyMuPDF, already used by ingest.py
        except ImportError:
            import fitz as pymupdf                # older PyMuPDF versions
        toc = pymupdf.open(pdf_path).get_toc()    # [[level, title, page], ...]
    except Exception:
        return []
    level = 1 if sum(1 for lv, _, _ in toc if lv == 1) >= 3 else 2
    starts = [(title.strip(), page) for lv, title, page in toc if lv <= level and page >= 1]
    starts = sorted({p: t for t, p in starts}.items())          # one entry per start page
    ranges = []
    for i, (page, title) in enumerate(starts):
        end = (starts[i + 1][0] - 1) if i + 1 < len(starts) else num_pages
        if end >= page:
            ranges.append({"title": title, "start": page, "end": end})
    return ranges


def _windows(pages: List[tuple], title: str = "") -> List[dict]:
    """Group consecutive pages into chunks of at most MAX_SECTION_CHARS characters."""
    out, cur = [], []
    for page_no, text in pages:
        if cur and sum(len(t) for _, t in cur) + len(text) > MAX_SECTION_CHARS:
            out.append(cur)
            cur = []
        cur.append((page_no, text))
    if cur:
        out.append(cur)
    multi = len(out) > 1
    return [
        {
            "title": f"{title} (part {i})" if (title and multi) else title,
            "start": grp[0][0],
            "end": grp[-1][0],
            "text": "\n\n".join(f"[p.{n}] {t}" for n, t in grp),
        }
        for i, grp in enumerate(out, 1)
    ]


def plan_sections(state: IngestState) -> dict:
    pages = [(p.page_no, (p.text_clean or "").strip()) for p in repo().get_pages(state["doc_id"])]
    pages = [(n, t) for n, t in pages if t]
    if not pages:
        raise ValueError(f"No pages for doc_id={state['doc_id']}. Run ingest first.")

    ranges = _toc_ranges(state["pdf_path"], pages[-1][0]) if state.get("pdf_path") else []
    sections: List[dict] = []
    if ranges:
        for r in ranges:
            in_range = [(n, t) for n, t in pages if r["start"] <= n <= r["end"]]
            sections += _windows(in_range, r["title"])
    else:
        sections = _windows(pages)

    print(f"[ingest] {len(sections)} sections ({'from TOC' if ranges else 'by page windows'})")
    return {"sections": sections, "summaries": [None] * len(sections), "issues": {}, "checks": 0}


# ---------------------------------------------------------------------------
# 2. Summarize every section that has no summary yet, or that check flagged
# ---------------------------------------------------------------------------
def summarize(state: IngestState) -> dict:
    sections, summaries, issues = state["sections"], list(state["summaries"]), state["issues"]
    todo = [i for i, s in enumerate(summaries) if s is None or i in issues]
    prompts = [
        P.section_summary_prompt(sections[i]["title"], sections[i]["start"], sections[i]["end"],
                                 sections[i]["text"], issues.get(i))
        for i in todo
    ]
    llm = structured_llm(SectionSummary, 0.3)
    stage = "summarizing" if state["checks"] == 0 else "revising"
    report_progress(state["doc_id"], stage, 0, len(todo))

    update: dict = {}
    batch = llm.batch_as_completed(prompts, config={"max_concurrency": MAX_CONCURRENCY}, return_exceptions=True)
    for done, (j, r) in enumerate(batch, 1):
        i = todo[j]
        if isinstance(r, Exception):
            print(f"[ingest] section {i} failed: {r}")      # a revision that fails keeps the earlier summary
            update["last_error"] = str(r)
        else:
            summaries[i] = r.model_dump()
        report_progress(state["doc_id"], stage, done, len(todo))
    print(f"[ingest] summarized {len(todo)} sections")
    return {"summaries": summaries, **update}


# ---------------------------------------------------------------------------
# 3. Check: rules in code first, then the LLM faithfulness check
# ---------------------------------------------------------------------------
def _rule_issues(summary: Optional[dict], sec: dict) -> List[str]:
    if summary is None:
        return ["The summary could not be generated; write it again."]
    problems = []
    if len(summary["key_points"]) < 3:
        problems.append("Give at least 3 key points.")
    bad = sorted({kp["page"] for kp in summary["key_points"] if not sec["start"] <= kp["page"] <= sec["end"]})
    if bad:
        problems.append(f"Pages {bad} are outside this section; cite pages {sec['start']}-{sec['end']} only.")
    if not summary["key_terms"]:
        problems.append("List the key terms of the section.")
    return problems


def check(state: IngestState) -> dict:
    sections, summaries = state["sections"], state["summaries"]
    to_check = range(len(sections)) if state["checks"] == 0 else list(state["issues"])  # re-check only revised ones
    issues: Dict[int, List[str]] = {i: p for i in to_check if (p := _rule_issues(summaries[i], sections[i]))}

    if CHECK_FAITHFULNESS:
        rule_clean = [i for i in to_check if i not in issues]
        report_progress(state["doc_id"], "checking", 0, len(rule_clean))
        llm = structured_llm(FaithfulnessCheck, 0.0)
        results = llm.batch(
            [P.faithfulness_prompt(summaries[i], sections[i]["text"]) for i in rule_clean],
            config={"max_concurrency": MAX_CONCURRENCY}, return_exceptions=True,
        )
        for i, r in zip(rule_clean, results):
            if not isinstance(r, Exception) and r.unsupported:
                issues[i] = [f"Not supported by the text, remove or correct: {u}" for u in r.unsupported]

    checks = state["checks"] + 1
    print(f"[ingest] check {checks}: {len(issues)} sections with issues")
    return {"issues": issues, "checks": checks}


def route_after_check(state: IngestState) -> str:
    if state["issues"] and state["checks"] <= MAX_REVISIONS:
        return "summarize"
    return "finalize"


# ---------------------------------------------------------------------------
# 4. Finalize: whole-document overview + save
# ---------------------------------------------------------------------------
def finalize(state: IngestState) -> dict:
    sections = []
    for i, (sec, summ) in enumerate(zip(state["sections"], state["summaries"])):
        if summ is None:
            continue
        sections.append({
            "title": sec["title"] or summ["title"] or f"Pages {sec['start']}-{sec['end']}",   # TOC title wins
            "start": sec["start"],
            "end": sec["end"],
            "main_idea": summ["main_idea"],
            "key_points": summ["key_points"],
            "key_terms": summ["key_terms"],
            "unresolved_issues": state["issues"].get(i, []),
        })
    if not sections:       # don't save an empty summary that would then count as "ready"
        raise RuntimeError(f"No section could be summarized. Last error: {state.get('last_error', 'unknown')}")
    report_progress(state["doc_id"], "overview")
    try:
        ov: DocOverview = structured_llm(DocOverview, 0.3).invoke(P.overview_prompt(sections))
        subject, overview = ov.subject, ov.overview
    except Exception as e:     # keep the section summaries of a long run even if this last call fails
        print(f"[ingest] overview failed, saving the sections without it: {e}")
        doc = repo().get_document(state["doc_id"])
        subject, overview = (doc.filename if doc else ""), ""
    result = {"doc_id": state["doc_id"], "subject": subject, "overview": overview, "sections": sections}
    path = save_doc_summary(state["doc_id"], result)
    print(f"[ingest] saved {len(sections)} section summaries to {path}")
    return {"result": result}


def build_ingest_graph():
    g = StateGraph(IngestState)
    g.add_node("plan_sections", plan_sections)
    g.add_node("summarize", summarize)
    g.add_node("check", check)
    g.add_node("finalize", finalize)
    g.add_edge(START, "plan_sections")
    g.add_edge("plan_sections", "summarize")
    g.add_edge("summarize", "check")
    g.add_conditional_edges("check", route_after_check, ["summarize", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


def summarize_document(doc_id: str, pdf_path: Optional[str] = None) -> dict:
    """Build and cache the section summaries for one document. Returns the saved summary."""
    with building(doc_id):
        return build_ingest_graph().invoke({"doc_id": doc_id, "pdf_path": pdf_path})["result"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.graphs.ingest_graph <doc_id> [path/to/file.pdf]")
    else:
        out = summarize_document(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        print(f"\n{out['subject']}\n{out['overview']}")
