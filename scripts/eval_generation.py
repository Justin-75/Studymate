# scripts/eval_generation.py
"""
Answer-quality benchmark for StudyMate: does the answer stick to the pages it was given?

    python scripts/eval_generation.py run                  # answer + judge every question (k = 8)
    python scripts/eval_generation.py run --k 5            # same with the top 5 chunks
    python scripts/eval_generation.py run --limit 10       # quick smoke test on 10 questions
    python scripts/eval_generation.py labels               # 50 answers for you to label by hand
    python scripts/eval_generation.py agree                # how often the judge agrees with you

For every question in eval/questions.jsonl:
  1. retrieve   the app's own question path, build_answer_graph() in src/graphs/study_graph.py:
                hybrid retrieval (top k chunks, each with its card if the document has cards), then
                the agent loop: summarize_topic writes notes and searches again while they don't cover
                the question (at most MAX_HOPS rounds)
  2. answer     the app's own `answer` node: notes + chunks, same prompt, same temperature
  3. judge      DeepEval FaithfulnessMetric with the app's LLM at temperature 0: split the answer
                into claims, mark each claim yes / borderline / no against the chunks the model saw.
                Faithfulness = yes claims / all claims.

Judge settings, and why:
  penalize_ambiguous_claims=True  DeepEval's default only fails a claim the pages CONTRADICT; a claim
                                  the pages never mention ("borderline") still passes. The model has
                                  read CLRS and d2l, so answering from memory is exactly the failure
                                  this benchmark has to catch.
  --truths raw (default)          Claims are checked against the chunks verbatim. Stock DeepEval
                                  (--truths extracted) first has the LLM rewrite the chunks into a list
                                  of "truths" and checks against those, so a detail the rewrite drops
                                  turns a supported claim into "borderline". Judge the same answers
                                  both ways and let `agree` show which matches your labels better.
  No router                       The benchmark question goes straight to retrieval, so a misrouted
                                  intent or a rewritten query doesn't leak into an answer-quality number.

Answers with found=false (the model said the pages don't answer it) are not judged: a refusal has no
claims, so DeepEval would score it 1.0. They are counted instead. On L5 (unanswerable) a refusal is
right (Abstain); on L1-L4 it is a missed answer (Answered).

Answers and verdicts are cached in eval_out/generation_k<k>/, so an interrupted run continues where it
stopped, and `--truths extracted` re-judges without re-answering. --fresh throws the cache away (do
that after changing the answer prompt, the retriever or the model).

Hand labels (label_sheet.csv, written by `labels`): human_faithful = 1 if every statement in the answer
is supported by the context shown, 0 if any statement is not (wrong, or true but not in the pages).
The sheet is half judge-unfaithful, half judge-faithful answers, so the disagreements show up in 50 rows;
the agreement numbers describe the judge, not the share of faithful answers.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Chinese text in a Windows console

from deepeval.metrics import FaithfulnessMetric  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from src.graphs import nodes as N  # noqa: E402  (the app's own retrieval + answer node)
from src.graphs import prompts as P  # noqa: E402
from src.graphs.study_graph import build_answer_graph  # noqa: E402
from src.llm_client import get_llm, llm_info  # noqa: E402
from src.retrieval import hybrid  # noqa: E402

EVAL_DIR = ROOT / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
NAMES_FILE = EVAL_DIR / "docs.json"
OUT_ROOT = ROOT / "eval_out"
SEED = 0


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
class AppJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by the app's own chat model (llm_client.get_llm) at temperature 0."""

    def __init__(self):
        self.llm = get_llm(0.0)
        super().__init__(llm_info()["model"])

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            try:
                out = self.llm.with_structured_output(schema).invoke(prompt)
                if out is not None:
                    return out
            except Exception:
                pass          # fall back to plain text; DeepEval parses the JSON out of it
        return self.llm.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            try:
                out = await self.llm.with_structured_output(schema).ainvoke(prompt)
                if out is not None:
                    return out
            except Exception:
                pass
        return (await self.llm.ainvoke(prompt)).content

    def get_model_name(self) -> str:
        return self.name


class ChunkFaithfulness(FaithfulnessMetric):
    """FaithfulnessMetric that checks claims against the retrieved chunks verbatim, not LLM-extracted truths."""

    def _generate_truths(self, retrieval_context, multimodal):
        return list(retrieval_context)

    async def _a_generate_truths(self, retrieval_context, multimodal):
        return list(retrieval_context)

    def _get_prompt(self, method, *, template_class=None, **kwargs):
        # DeepEval finds prompts by class name; borrow FaithfulnessMetric's (same text as the stock metric)
        return super()._get_prompt(method, template_class=template_class or "FaithfulnessMetric", **kwargs)


_judge: Optional[AppJudge] = None


def make_metric(truths: str) -> FaithfulnessMetric:
    global _judge
    _judge = _judge or AppJudge()
    cls = ChunkFaithfulness if truths == "raw" else FaithfulnessMetric
    return cls(model=_judge, penalize_ambiguous_claims=True, include_reason=False)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_questions() -> List[Dict]:
    names = json.loads(NAMES_FILE.read_text(encoding="utf-8")) if NAMES_FILE.exists() else {}
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if not r.get("excluded")]
    for r in rows:
        r["doc"] = names.get(r["doc_id"], r.get("doc", r["doc_id"][:8]))
    return rows


def gold_groups(q: Dict) -> List[set]:
    """The question's answer parts (same rule as eval_retrieval.py). Empty for L5."""
    if q.get("gold_groups"):
        return [set(g) for g in q["gold_groups"] if g]
    return [set(q["gold_pages"])] if q.get("gold_pages") else []


def load_jsonl(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: r for r in rows}


def write_csv(path: Path, rows: List[Dict]) -> None:
    if rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def read_csv(path: Path) -> List[Dict]:
    """utf-8 first; Excel on a Chinese Windows saves plain "CSV" as GBK."""
    for enc in ("utf-8-sig", "gbk"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"Can't read {path}: save it as 'CSV UTF-8' in Excel.")


def out_dir(k: int, unit: str = "children") -> Path:
    d = OUT_ROOT / f"generation_k{k}_{unit}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1 + 2. Retrieve and answer, exactly as the app does
# ---------------------------------------------------------------------------
def block(hit: Dict) -> str:
    return P.format_block(hit)       # exactly the block the model saw (page label, card line, text)


_GPU = threading.Lock()          # BGE-M3 and the reranker: one search at a time across the worker threads
_app_search = N._search


def _locked_search(*args, **kwargs):
    with _GPU:
        return _app_search(*args, **kwargs)


N._search = _locked_search      # the graph's retrieve node looks N._search up at call time
_answer_graph = build_answer_graph()


def answer_one(q: Dict, k: int) -> Dict:
    out = _answer_graph.invoke({"doc_id": q["doc_id"], "topic": q["q"], "intent": "ask", "scope": "topic",
                                "follow_up": False, "messages": [], "hops": 0, "queries": []})
    hits = out.get("context") or []
    prompt_context = P.format_context(hits)
    shown = [h for h in hits if block(h) in prompt_context]     # format_context stops at max_chars
    notes = out.get("notes") or {}
    rec = {"id": q["id"], "k": k, "model": llm_info()["model"],
           "hops": out.get("hops", 0), "queries": out.get("queries", []), "covered": notes.get("covered"),
           "shown_pages": [h["page_no"] for h in shown], "context": [block(h) for h in shown]}
    data = out["messages"][-1].additional_kwargs.get("data", {}) if out.get("messages") else {}
    if not hits or "answer" not in data:                         # the app replied no_context
        return {**rec, "answer": "", "found": False, "cited_pages": []}
    return {**rec, "answer": data["answer"], "found": bool(data["found"]),
            "cited_pages": [int(p) for p in data["pages"]]}


def generate_answers(questions: List[Dict], k: int, path: Path, workers: int) -> Dict[str, Dict]:
    done = load_jsonl(path)
    todo = [q for q in questions if q["id"] not in done]
    if not todo:
        return done
    t0 = time.time()
    print(f"  answering {len(todo)} questions with {llm_info()['model']} ({workers} at a time; "
          f"retrieval, notes loop and answer as in the app) ...")
    with ThreadPoolExecutor(workers) as pool, open(path, "a", encoding="utf-8") as f:
        futures = {pool.submit(answer_one, q, k): q for q in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:
                print(f"    ! {futures[fut]['id']}: {type(e).__name__}: {e}")
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[rec["id"]] = rec
            if i % 20 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)} answered")
    print(f"  answering took {time.time() - t0:.0f}s")
    return done


# ---------------------------------------------------------------------------
# 3. Judge
# ---------------------------------------------------------------------------
async def judge_answers(questions: List[Dict], answers: Dict[str, Dict], truths: str,
                        path: Path, workers: int) -> Dict[str, Dict]:
    done = load_jsonl(path)
    todo = [(q, answers[q["id"]]) for q in questions
            if q["id"] in answers and q["id"] not in done
            and answers[q["id"]]["found"] and answers[q["id"]]["answer"].strip()]
    if not todo:
        return done
    t0, finished = time.time(), 0
    print(f"  judging {len(todo)} answers (truths={truths}, {workers} at a time) ...")
    sem = asyncio.Semaphore(workers)

    async def one(q: Dict, a: Dict) -> Optional[Dict]:
        async with sem:
            metric = make_metric(truths)
            case = LLMTestCase(input=q["q"], actual_output=a["answer"], retrieval_context=a["context"])
            try:
                await metric.a_measure(case, _show_indicator=False)
            except Exception as e:
                print(f"    ! {q['id']}: {type(e).__name__}: {e}")
                return None
            verdicts = [{"verdict": str(v.verdict), "reason": v.reason or ""} for v in metric.verdicts]
            return {"id": q["id"], "truths": truths, "judge": metric.evaluation_model,
                    "score": float(metric.score), "claims": metric.claims, "verdicts": verdicts}

    with open(path, "a", encoding="utf-8") as f:
        for coro in asyncio.as_completed([one(q, a) for q, a in todo]):
            rec = await coro
            finished += 1
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[rec["id"]] = rec
            if finished % 20 == 0 or finished == len(todo):
                print(f"    {finished}/{len(todo)} judged")
    print(f"  judging took {time.time() - t0:.0f}s")
    return done


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def unsupported(j: Dict) -> str:
    """'[no] claim - reason' for every claim that didn't get a yes."""
    lines = []
    for claim, v in zip(j["claims"], j["verdicts"]):
        if v["verdict"] != "yes":
            lines.append(f"[{v['verdict']}] {claim}" + (f" - {v['reason']}" if v["reason"] else ""))
    return "\n".join(lines)


def per_question_rows(questions: List[Dict], answers: Dict[str, Dict], judged: Dict[str, Dict]) -> List[Dict]:
    rows = []
    for q in questions:
        a = answers.get(q["id"])
        if a is None:
            continue
        j = judged.get(q["id"])
        groups = gold_groups(q)
        gold = set().union(*groups) if groups else set()
        shown, cited = set(a["shown_pages"]), set(a["cited_pages"])
        counts = defaultdict(int)
        for v in (j["verdicts"] if j else []):
            counts[v["verdict"]] += 1
        rows.append({
            "id": q["id"], "doc": q["doc"], "lang": q.get("lang", ""), "level": q.get("level", ""),
            "question": q["q"],
            "gold_groups": " | ".join(" ".join(map(str, sorted(g))) for g in groups) or "(none)",
            "shown_pages": " ".join(map(str, a["shown_pages"])),
            # any gold page among the chunks the model saw / every answer part among them
            "retrieval_hit": int(bool(gold & shown)) if gold else "n/a",
            "retrieval_full": int(all(g & shown for g in groups)) if gold else "n/a",
            "found": int(a["found"]),
            "hops": a.get("hops", ""),                       # retrieval rounds of the agent loop
            "queries": " | ".join(a.get("queries", [])),
            "cited_pages": " ".join(map(str, sorted(cited))),
            # cited pages were all shown to the model / at least one cited page is a gold page
            "cite_valid": int(bool(cited) and cited <= shown) if a["found"] else "n/a",
            "cite_gold": int(bool(cited & gold)) if a["found"] and gold else "n/a",
            "faithfulness": round(j["score"], 4) if j else "",
            "claims": len(j["claims"]) if j else "",
            "yes": counts["yes"] if j else "", "borderline": counts["borderline"] if j else "",
            "no": counts["no"] if j else "",
            "unsupported": unsupported(j) if j else "",
            "answer": a["answer"],
        })
    return rows


def mean(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def summary_rows(rows: List[Dict]) -> List[Dict]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        slices = ["ALL", f"doc={r['doc']}", f"lang={r['lang']}", f"level={r['level']}"]
        if r["retrieval_hit"] != "n/a":
            slices.append("retrieval=hit" if r["retrieval_hit"] else "retrieval=miss")
        for sl in slices:
            buckets[sl].append(r)

    order = lambda sl: (sl != "ALL", sl.split("=")[0] == "retrieval", sl)   # noqa: E731
    out = []
    for sl in sorted(buckets, key=order):
        items = buckets[sl]
        gold_rows = [r for r in items if r["retrieval_hit"] != "n/a"]
        l5_rows = [r for r in items if r["retrieval_hit"] == "n/a"]
        answered = [r for r in items if r["found"]]
        judged = [r for r in items if r["faithfulness"] != ""]
        out.append({
            "slice": sl, "n": len(items),
            "Hops": mean([r["hops"] for r in items if r["hops"] != ""]),
            "RetrievalHit": mean([r["retrieval_hit"] for r in gold_rows]),
            "Answered": mean([r["found"] for r in gold_rows]),
            "Abstain": mean([1 - r["found"] for r in l5_rows]),
            "Judged": len(judged),
            "Faithfulness": mean([r["faithfulness"] for r in judged]),
            "FullyFaithful": mean([float(r["faithfulness"] == 1.0) for r in judged]),
            "CiteValid": mean([r["cite_valid"] for r in answered]),
            "CiteGold": mean([r["cite_gold"] for r in answered if r["cite_gold"] != "n/a"]),
        })
    return out


def print_summary(summary: List[Dict]) -> None:
    cols = ["Hops", "RetrievalHit", "Answered", "Abstain", "Judged", "Faithfulness", "FullyFaithful", "CiteValid", "CiteGold"]
    heads = ["Hops", "RetHit", "Answered", "Abstain", "Judged", "Faithful", "Fully", "CiteOK", "CiteGold"]

    def cell(v) -> str:
        if v is None:
            return f"{'-':>9}"
        return f"{v:>9}" if isinstance(v, int) else f"{v:>9.3f}"

    print(f"\n  {'slice':<18}{'n':>4}" + "".join(f"{h:>9}" for h in heads))
    for r in summary:
        print(f"  {r['slice'][:17]:<18}{r['n']:>4}" + "".join(cell(r[c]) for c in cols))
    print("""
  Hops      mean retrieval rounds per question (1 = the first search was enough)
  RetHit    a gold page is among the chunks the model saw (L1-L4)
  Answered  the model gave an answer (found=true) on L1-L4; a refusal there is a missed answer
  Abstain   the model refused (found=false) on L5, where the document has no answer: higher is better
  Faithful  mean DeepEval faithfulness of the answered rows (yes claims / all claims, borderline = fail)
  Fully     share of answered rows where every claim is supported
  CiteOK    every cited page was among the chunks shown;  CiteGold  a cited page is a gold page
  retrieval=hit / miss splits L1-L4 by RetHit: unfaithful on a hit = generation problem;
  answered on a miss = the model filled in from memory (or found the answer on another page).""")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_run(args) -> None:
    questions = load_questions()
    if args.limit:
        picked = {q["id"] for q in random.Random(SEED).sample(questions, min(args.limit, len(questions)))}
        questions = [q for q in questions if q["id"] in picked]
    N.TOP_K = args.k                # chunks of the first search; the agent loop may add MORE_K per round
    hybrid.RETRIEVAL_UNIT = args.unit   # children = parent-child retrieval (the LLM still gets the parent chunks)
    d = out_dir(args.k, args.unit)
    answers_path, judged_path = d / "answers.jsonl", d / f"judged_{args.truths}.jsonl"
    if args.fresh:
        for p in d.glob("*.jsonl"):
            p.unlink()
    info = llm_info()
    print(f"StudyMate answer benchmark: {len(questions)} questions, top {args.k} chunks, "
          f"{info['provider']} {info['model']} answers and judges")

    answers = generate_answers(questions, args.k, answers_path, args.workers)
    judged = asyncio.run(judge_answers(questions, answers, args.truths, judged_path, args.workers))

    rows = per_question_rows(questions, answers, judged)
    summary = summary_rows(rows)
    write_csv(d / f"per_question_{args.truths}.csv", rows)
    write_csv(d / f"summary_{args.truths}.csv", summary)
    print_summary(summary)
    missing = len(questions) - len(rows) + sum(
        1 for r in rows if r["found"] and r["answer"].strip() and r["faithfulness"] == "")
    if missing:
        print(f"\n  {missing} questions failed (see the ! lines); run the same command again to retry them.")
    print(f"\n  Excel files written to {d}\n")


def cmd_labels(args) -> None:
    d = out_dir(args.k, args.unit)
    sheet = d / "label_sheet.csv"
    if sheet.exists() and not args.force:
        raise SystemExit(f"{sheet} already exists (it may hold your labels). Use --force to replace it.")
    questions = {q["id"]: q for q in load_questions()}
    answers = load_jsonl(d / "answers.jsonl")
    judged = load_jsonl(d / f"judged_{args.truths}.jsonl")
    if not judged:
        raise SystemExit(f"No judged answers in {d}. Run: python scripts/eval_generation.py run --k {args.k}")

    rng = random.Random(SEED)
    ids = sorted(judged)
    bad = [i for i in ids if judged[i]["score"] < 1.0]
    good = [i for i in ids if judged[i]["score"] == 1.0]
    picked = rng.sample(bad, min(len(bad), args.n // 2))
    picked += rng.sample(good, min(len(good), args.n - len(picked)))
    rng.shuffle(picked)

    rows = [{"id": i, "doc": questions[i]["doc"], "level": questions[i].get("level", ""),
             "question": questions[i]["q"], "context": "\n\n".join(answers[i]["context"]),
             "answer": answers[i]["answer"], "human_faithful": "", "human_note": ""} for i in picked]
    write_csv(sheet, rows)
    print(f"Wrote {len(rows)} answers to {sheet}")
    print("Fill human_faithful: 1 = every statement in the answer is supported by the context,"
          "\n0 = at least one is not (wrong, or true but not in these pages). Leave blank to skip."
          "\nDon't look at the judge's columns in per_question_*.csv first. Then run `agree`.")


def cmd_agree(args) -> None:
    d = out_dir(args.k, args.unit)
    sheet = d / "label_sheet.csv"
    if not sheet.exists():
        raise SystemExit(f"No {sheet}. Run `labels` first.")
    human = {r["id"]: int(r["human_faithful"].strip()) for r in read_csv(sheet)
             if r.get("human_faithful", "").strip() in ("0", "1")}
    if not human:
        raise SystemExit("No labels yet: fill the human_faithful column with 1 or 0.")

    compare = {r["id"]: {"id": r["id"], "human": human[r["id"]]} for r in read_csv(sheet) if r["id"] in human}
    for path in sorted(d.glob("judged_*.jsonl")):
        mode = path.stem.removeprefix("judged_")
        judged = load_jsonl(path)
        pairs = [(h, int(judged[i]["score"] == 1.0)) for i, h in human.items() if i in judged]
        if not pairs:
            continue
        n = len(pairs)
        po = sum(h == j for h, j in pairs) / n
        ph, pj = sum(h for h, _ in pairs) / n, sum(j for _, j in pairs) / n
        pe = ph * pj + (1 - ph) * (1 - pj)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        cm = {(h, j): sum(1 for p in pairs if p == (h, j)) for h in (1, 0) for j in (1, 0)}
        print(f"\n  truths={mode}: {n} labelled answers, agreement {po:.1%}, Cohen's kappa {kappa:.2f}")
        print(f"    you faithful,   judge faithful   {cm[(1, 1)]:>3}    judge unfaithful {cm[(1, 0)]:>3}"
              f"   <- judge too strict")
        print(f"    you unfaithful, judge faithful   {cm[(0, 1)]:>3}    judge unfaithful {cm[(0, 0)]:>3}"
              f"   (judge faithful here = judge too lenient)")
        for i in human:
            if i in judged:
                compare[i][f"judge_{mode}"] = round(judged[i]["score"], 4)
                compare[i][f"unsupported_{mode}"] = unsupported(judged[i])
    rows = list(compare.values())
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "id", k != "human", k))
    write_csv(d / "agreement.csv", [{k: r.get(k, "") for k in keys} for r in rows])
    print(f"\n  Row-by-row comparison written to {d / 'agreement.csv'}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="StudyMate answer-faithfulness benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "labels", "agree"):
        p = sub.add_parser(name)
        p.add_argument("--k", type=int, default=N.TOP_K, help=f"chunks given to the model (app: {N.TOP_K})")
        p.add_argument("--unit", choices=["children", "chunks"], default="children",
                       help="what retrieval searches: children (the app: children -> their parent chunks) "
                            "or the 512-word chunks. Results in generation_k<k>_<unit>/")
        if name != "agree":
            p.add_argument("--truths", choices=["raw", "extracted"], default="raw",
                           help="judge against the chunks verbatim (raw) or DeepEval's extracted truths")
    run = sub.choices["run"]
    run.add_argument("--limit", type=int, default=0, help="only this many questions (random, fixed seed)")
    run.add_argument("--workers", type=int, default=8, help="LLM calls in flight at once")
    run.add_argument("--fresh", action="store_true", help="discard cached answers and verdicts for this k")
    labels = sub.choices["labels"]
    labels.add_argument("--n", type=int, default=50)
    labels.add_argument("--force", action="store_true", help="overwrite an existing label_sheet.csv")
    args = ap.parse_args()
    {"run": cmd_run, "labels": cmd_labels, "agree": cmd_agree}[args.cmd](args)


if __name__ == "__main__":
    main()
