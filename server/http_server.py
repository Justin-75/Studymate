# server/http_server.py
"""
StudyMate web server: the LangGraph study graph behind an async FastAPI app, plus the web UI.

    python server/http_server.py        then open http://localhost:5000

API
    GET  /api/health                         server + which LLM is active
    GET  /api/documents                      uploaded PDFs with their status
    POST /api/upload                         PDF (multipart "file") -> ingest; indexing + summaries run in the background
    GET  /api/documents/{doc_id}/summary     cached section summaries (or their status)
    POST /api/documents/{doc_id}/summarize   start building the summaries for an older document
    GET  /api/documents/{doc_id}/pdf         the PDF itself (citations open it at #page=N)
    POST /api/chat                           {thread_id, doc_id, message, intent?} -> server-sent events
    POST /api/quiz/answer                    {thread_id, answers: {q1: "A"}}  -> server-sent events
    GET  /api/threads/{thread_id}            full chat history of a thread (to restore after reload)

Chat and quiz answers stream "step" events (one per finished graph node) and end with a
"done" event holding the new messages, or an "error" event. Errors elsewhere are {"error": "..."}.

Nothing slow runs on the event loop: the graph's (sync) nodes run in LangGraph's thread pool,
and PDF ingest, index building and summaries run in asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import BinaryIO, Coroutine, Literal, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from src.db.schema import init_db  # noqa: E402
from src.graphs.ingest_graph import summarize_document  # noqa: E402
from src.graphs.store import (  # noqa: E402
    DB_PATH, building_progress, has_doc_summary, is_building, load_doc_summary, repo,
)
from src.graphs.study_graph import open_study_graph_async  # noqa: E402
from src.ingest import ingest_pdf  # noqa: E402
from src.llm_client import llm_info  # noqa: E402
from src.retrieval.hybrid import warm_up  # noqa: E402

WEB_DIR = ROOT / "web"                      # the chat UI: web/index.html
UPLOAD_DIR = ROOT / "uploads"               # kept PDFs: uploads/<doc_id>.pdf
OLD_PDF_DIR = ROOT / "Data" / "pdfs"        # PDFs ingested before this server existed
HOST = os.getenv("STUDYMATE_HOST", "127.0.0.1")   # set to 0.0.0.0 to open it to other devices
PORT = int(os.getenv("STUDYMATE_PORT", "5000"))

# One graph run or index build at a time: both use the GPU models, and this is a single-user local app.
GPU_LOCK = asyncio.Lock()
# Background work per document: {"index": "building" | "ready" | "error", "error": str,
#                                "summarizing": bool, "summary_error": str}
# Only touched from the event loop, so no lock is needed.
JOBS: dict[str, dict] = {}
# asyncio keeps only weak references to tasks; this keeps background jobs alive until they finish.
TASKS: set[asyncio.Task] = set()

NO_TEXT = "No text could be read from this PDF (if it is scanned, OCR found nothing either)."
SUMMARY_BUSY = "The document summary is being built. The chat unlocks when it is done."


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(exist_ok=True)
    init_db(DB_PATH)
    async with open_study_graph_async() as graph:
        app.state.graph = graph
        yield


app = FastAPI(title="StudyMate", lifespan=lifespan)
# The UI is normally served by this app (same origin). This only lets a page on another local
# port (e.g. an editor's live preview) call the API; other websites still cannot read it.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_error(_request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def invalid_request(_request, exc: RequestValidationError) -> JSONResponse:
    fields = ", ".join(".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors())
    return JSONResponse({"error": f"Missing or invalid: {fields}"}, status_code=400)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def pdf_path_for(doc_id: str, filename: Optional[str] = None) -> Optional[Path]:
    p = UPLOAD_DIR / f"{doc_id}.pdf"
    if p.exists():
        return p
    if filename and (OLD_PDF_DIR / filename).exists():
        return OLD_PDF_DIR / filename
    return None


def summary_status(doc_id: str) -> str:
    job = JOBS.get(doc_id, {})
    if has_doc_summary(doc_id):
        return "ready"
    if job.get("summarizing") or is_building(doc_id):
        return "building"
    if job.get("summary_error"):
        return "error"
    return "none"


def doc_info(doc_id: str, filename: str, created_at: str, pages: int, chunks: int) -> dict:
    job = JOBS.get(doc_id, {})
    if job.get("index"):
        index, error = job["index"], job.get("error")      # uploaded during this run
    elif chunks == 0:
        index, error = "error", NO_TEXT                     # nothing to search in
    else:
        index, error = "ready", None                        # indexed by an earlier run
    summary = summary_status(doc_id)
    return {
        "doc_id": doc_id,
        "filename": filename,
        "created_at": created_at,
        "pages": pages,
        "index": index,
        "index_error": error,
        "summary": summary,
        # {"stage", "done", "total"} while building ({} until the job reports); the chat is locked meanwhile
        "summary_progress": (building_progress(doc_id) or {}) if summary == "building" else None,
        "summary_error": job.get("summary_error"),
        "has_pdf": pdf_path_for(doc_id, filename) is not None,
    }


def _query_documents(doc_id: Optional[str] = None) -> list[tuple]:
    where, args = ("WHERE d.doc_id = ?", (doc_id,)) if doc_id else ("", ())
    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute(
            "SELECT d.doc_id, d.filename, d.created_at, "
            "       (SELECT COUNT(*) FROM pages p WHERE p.doc_id = d.doc_id), "
            "       (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.doc_id) "
            f"FROM documents d {where} ORDER BY d.created_at DESC",
            args,
        ).fetchall()
    finally:
        conn.close()


async def list_documents() -> list[dict]:
    return [doc_info(*r) for r in await asyncio.to_thread(_query_documents)]


async def find_document(doc_id: str) -> Optional[dict]:
    rows = await asyncio.to_thread(_query_documents, doc_id)
    return doc_info(*rows[0]) if rows else None


# ---------------------------------------------------------------------------
# Background jobs: indexing after upload, then section summaries
# ---------------------------------------------------------------------------
def spawn(coro: Coroutine) -> None:
    task = asyncio.create_task(coro)
    TASKS.add(task)
    task.add_done_callback(TASKS.discard)


async def prepare_document(doc_id: str, pdf_path: str) -> None:
    """After upload: build the BM25 + BGE-M3 indexes (chat waits for this), then the summaries."""
    job = JOBS.setdefault(doc_id, {})
    job.update(index="building", error=None)
    try:
        async with GPU_LOCK:
            await asyncio.to_thread(warm_up, repo(), doc_id)
    except Exception as e:
        traceback.print_exc()
        job.update(index="error", error=str(e))
        return
    job["index"] = "ready"
    start_summary(doc_id, pdf_path)


def start_summary(doc_id: str, pdf_path: Optional[str]) -> str:
    """Start building the section summaries unless they exist or are being built. Returns the status."""
    status = summary_status(doc_id)
    if status in ("ready", "building"):
        return status
    JOBS.setdefault(doc_id, {}).update(summarizing=True, summary_error=None)
    spawn(_summarize(doc_id, pdf_path))
    return "building"


async def _summarize(doc_id: str, pdf_path: Optional[str]) -> None:
    job = JOBS[doc_id]
    try:
        await asyncio.to_thread(summarize_document, doc_id, pdf_path)
    except Exception as e:
        traceback.print_exc()
        job["summary_error"] = str(e)
    finally:
        job["summarizing"] = False


# ---------------------------------------------------------------------------
# Graph runs, streamed as server-sent events
# ---------------------------------------------------------------------------
def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def awaiting_answers(snapshot) -> bool:
    return any(t.interrupts for t in snapshot.tasks)


def serialize(msg) -> dict:
    if msg.type == "human":
        return {"role": "user", "text": str(msg.content), "intent": msg.additional_kwargs.get("intent")}
    kind = msg.additional_kwargs.get("kind", "text")
    data = json.loads(json.dumps(msg.additional_kwargs.get("data", {}), ensure_ascii=False))
    if kind == "quiz":                                   # never send the answers to the browser
        for q in data.get("questions", []):
            q.pop("answer", None)
    return {"role": "assistant", "kind": kind, "text": str(msg.content), "data": data}


async def thread_payload(thread_id: str, start: int = 0) -> dict:
    snap = await app.state.graph.aget_state(thread_config(thread_id))
    values = snap.values or {}
    return {
        "messages": [serialize(m) for m in values.get("messages", [])[start:]],
        "awaiting_answers": awaiting_answers(snap),
        "mastery": values.get("mastery", {}),
        "round": values.get("round", 0),
        "doc_id": values.get("doc_id"),
    }


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_graph(thread_id: str, graph_input, events: asyncio.Queue) -> None:
    graph, config = app.state.graph, thread_config(thread_id)
    try:
        async with GPU_LOCK:
            snap = await graph.aget_state(config)
            if isinstance(graph_input, Command) and not awaiting_answers(snap):   # answered twice
                await events.put(sse("error", {"message": "No quiz is waiting for answers in this chat"}))
                return
            before = len((snap.values or {}).get("messages", []))
            await events.put(sse("start", {"thread_id": thread_id}))
            async for chunk in graph.astream(graph_input, config, stream_mode="updates"):
                for node in chunk:
                    if node != "__interrupt__":
                        await events.put(sse("step", {"node": node}))
            await events.put(sse("done", await thread_payload(thread_id, start=before)))
    except Exception as e:
        traceback.print_exc()
        await events.put(sse("error", {"message": str(e)}))
    finally:
        await events.put(None)


def stream_graph(thread_id: str, graph_input) -> StreamingResponse:
    """
    The run is its own task and the response only relays its events, so closing the tab does
    not cancel a run halfway: it finishes, is checkpointed, and shows up after a reload.
    """
    events: asyncio.Queue = asyncio.Queue()
    spawn(_run_graph(thread_id, graph_input, events))

    async def relay():
        while (item := await events.get()) is not None:
            yield item

    return StreamingResponse(relay(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    # set by the Quiz / Flashcards / Summary buttons; without it the router's LLM decides
    intent: Optional[Literal["ask", "summary", "flashcard", "quiz"]] = None


class QuizAnswerRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    answers: dict[str, str]


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "llm": llm_info()}


@app.get("/api/documents")
async def documents():
    return {"documents": await list_documents()}


def _save_upload(src: BinaryIO, dest: Path) -> None:
    with dest.open("wb") as out:
        shutil.copyfileobj(src, out)


@app.post("/api/upload")
async def upload(file: Optional[UploadFile] = File(None)):
    if not file or not file.filename:
        raise HTTPException(400, "No file uploaded")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # keep the original file name (ingest stores it as the document name); a unique folder avoids clashes
    tmp_dir = Path(tempfile.mkdtemp(prefix="studymate-"))
    tmp = tmp_dir / Path(file.filename.replace("\\", "/")).name
    try:
        await asyncio.to_thread(_save_upload, file.file, tmp)
        result = await asyncio.to_thread(
            ingest_pdf, file_path=str(tmp), db_path=DB_PATH,
            ocr_enabled=True, ocr_lang="eng+chi_sim", ocr_dpi=300, ocr_psm=3, ocr_preprocess=False,
            chunk_size=800, chunk_overlap=120, max_pages=None,
        )
        if result.num_chunks == 0:
            raise HTTPException(422, NO_TEXT)
        kept = UPLOAD_DIR / f"{result.doc_id}.pdf"
        await asyncio.to_thread(shutil.move, str(tmp), kept)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Could not process the PDF: {e}") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    JOBS[result.doc_id] = {"index": "building"}
    spawn(prepare_document(result.doc_id, str(kept)))
    return {"document": doc_info(result.doc_id, result.filename, "", result.num_pages, result.num_chunks)}


@app.get("/api/documents/{doc_id}/summary")
async def document_summary(doc_id: str):
    summary = await asyncio.to_thread(load_doc_summary, doc_id)
    if summary:
        return {"status": "ready", "summary": summary}
    return {"status": summary_status(doc_id), "error": JOBS.get(doc_id, {}).get("summary_error")}


@app.post("/api/documents/{doc_id}/summarize")
async def document_summarize(doc_id: str):
    doc = await find_document(doc_id)
    if not doc:
        raise HTTPException(404, "Unknown document")
    pdf = pdf_path_for(doc_id, doc["filename"])
    return {"status": start_summary(doc_id, str(pdf) if pdf else None)}


@app.get("/api/documents/{doc_id}/pdf")
async def document_pdf(doc_id: str):
    doc = await find_document(doc_id)
    pdf = pdf_path_for(doc_id, doc["filename"] if doc else None)
    if not pdf:
        raise HTTPException(404, "PDF file not available")
    return FileResponse(pdf, media_type="application/pdf")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "message is empty")
    doc = await find_document(req.doc_id)
    if not doc:
        raise HTTPException(404, "Unknown document")
    if doc["index"] == "building":
        raise HTTPException(409, "This document is still being indexed. Please wait a moment.")
    if doc["index"] == "error":
        raise HTTPException(409, f"This document could not be indexed: {doc['index_error']}")
    if doc["summary"] == "building":
        raise HTTPException(409, SUMMARY_BUSY)
    human = HumanMessage(message, additional_kwargs={"intent": req.intent} if req.intent else {})
    return stream_graph(req.thread_id, {"messages": [human], "doc_id": req.doc_id})


@app.post("/api/quiz/answer")
async def quiz_answer(req: QuizAnswerRequest):
    snap = await app.state.graph.aget_state(thread_config(req.thread_id))
    if not awaiting_answers(snap):
        raise HTTPException(409, "No quiz is waiting for answers in this chat")
    doc_id = (snap.values or {}).get("doc_id")
    if doc_id and summary_status(doc_id) == "building":
        raise HTTPException(409, SUMMARY_BUSY)
    return stream_graph(req.thread_id, Command(resume={"answers": req.answers}))


@app.get("/api/threads/{thread_id}")
async def thread_history(thread_id: str):
    return await thread_payload(thread_id)


if __name__ == "__main__":
    info = llm_info()
    print("=" * 60)
    print(f"StudyMate   http://localhost:{PORT}      LLM: {info['provider']} ({info['model']})")
    print("=" * 60)
    uvicorn.run(app, host=HOST, port=PORT)
