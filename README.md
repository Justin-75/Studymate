## StudyMate — PDF study assistant (RAG)

A chat-style study assistant for PDFs: ask questions, read chapter summaries, flip flashcards and take
quizzes that are graded and followed by a review of what you missed. Chinese and English documents.

**Retrieval:** BM25 (jieba) + BGE-M3 dense (FAISS) → Reciprocal Rank Fusion → bge-reranker-v2-m3.
On the 181-question bilingual benchmark (`eval/questions.jsonl`) this reaches Recall@1 0.873 and
MRR 0.923, against 0.591 / 0.691 for the earlier TF-IDF pipeline. See `eval_out/`.

**Generation:** DeepSeek through LangChain (`ChatDeepSeek`), or local Ollama when no key is set, orchestrated with LangGraph.

### Requirements

- Windows 10/11 with either [Miniconda / Anaconda](https://www.anaconda.com/download) (option A) or
  [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the WSL 2 backend (option B)
- An NVIDIA GPU is recommended for BGE-M3 and the reranker; without one they run on the CPU, slower
- A [DeepSeek](https://platform.deepseek.com/) API key, or [Ollama](https://ollama.com/) running locally
- Disk: about 7 GB for the two models, plus about 6 GB for the Docker image (option B only)

### Installation

**1. Get the code**

```powershell
git clone https://github.com/Justin-75/studymate.git
cd studymate
```

**2. Configure the LLM.** Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_key_here      # exactly this name; leave out to use local Ollama instead
OLLAMA_MODEL=mistral-nemo:12b       # used only when there is no DeepSeek key
CHECK_FAITHFULNESS=0                # 1 = LLM-check every section summary (about 2x the cost)
```

**3. Download BGE-M3** (dense retrieval, about 4.3 GB) into `E:\models\bge-m3`. Both CLIs come with the conda
environment from step 4A; for option B, `pip install huggingface_hub` is enough.

```powershell
hf download BAAI/bge-m3 --local-dir E:\models\bge-m3
# or, from mainland China:
modelscope download --model BAAI/bge-m3 --local-dir E:\models\bge-m3
```

Another folder works too: set `BGE_M3_PATH` in `.env` (and the `E:/models/bge-m3` volume in `compose.yaml` for
Docker). Without the folder the app downloads `BAAI/bge-m3` from Hugging Face on first use. The reranker
(`BAAI/bge-reranker-v2-m3`, about 2.3 GB) always downloads itself into the Hugging Face cache on first use.

**4A. Option A: conda**

```powershell
conda env create -f environment.yml
conda activate studymate
python server/http_server.py
```

`environment.yml` holds everything: Python 3.11, PyTorch with CUDA 12.8 (for RTX 50-series and older NVIDIA
cards), the Tesseract OCR engine with Chinese and English data (for scanned PDFs), and the Python packages.
To bring an existing environment up to date after pulling: `conda env update -f environment.yml`.

**4B. Option B: Docker**

With Docker Desktop running (Settings → Resources → Advanced → *Disk image location* can put its data on a
drive other than C:):

```powershell
docker compose up --build
```

The first build takes about 15 minutes and produces a 6 GB image; later starts take seconds. The image installs
the same `environment.yml` on Linux; `compose.yaml` passes in the GPU, the `.env` file, BGE-M3 from
`E:\models\bge-m3`, the Hugging Face cache (for the reranker) and the project's `Data/` and `uploads/`, so it
sees the same documents as a local run. Run option A or B, not both at once: they share the database and the
port. `docker compose down` stops it. Without a DeepSeek key the container uses Ollama on Windows through
`host.docker.internal`.

### Run

Open `http://localhost:5000` (the server serves the UI; opening `web/index.html` as a file just redirects
there). Upload a PDF; it is indexed and its chapter summaries are built in the background. The chat stays
locked until both are done, with a progress card ("Summarizing section 14 of 40") in the chat; the same
happens when you press **Build summary** for an older document. Opening a document shows a short guide to
what you can do with it.

Then chat in your own words: the router's LLM decides whether you want an answer, a summary, flashcards or a
quiz. Or press **Summary / Flashcards / Quiz** under the input box to get that directly: with a topic typed in
the box it focuses on that topic; with the box empty it continues the current conversation, or covers the whole
document if there is none. Quizzes are graded, weak concepts get a short review and a new quiz (up to 3 rounds),
and the sidebar tracks mastery per concept.

The local server listens on `127.0.0.1` only; set `STUDYMATE_HOST=0.0.0.0` to open it to other devices on your
network.

Terminal versions of the same graphs:

```powershell
python -m src.graphs.study_graph <doc_id>                      # chat in the terminal
python -m src.graphs.ingest_graph <doc_id> path\to\book.pdf    # build chapter summaries
```

### How it works

- **Ingest graph** (`src/graphs/ingest_graph.py`): sections from the PDF's table of contents (or page
  windows) -> summary per section -> rule check (+ optional LLM faithfulness check) -> revise, at most twice -> cached.
- **Study graph** (`src/graphs/study_graph.py`): router (intent + standalone query) -> hybrid retrieval ->
  answer / topic summary / flashcards / quiz. Quiz -> pause for answers (`interrupt`) -> grade -> review
  weak concepts -> new quiz. State is checkpointed in SQLite per chat thread, so chats survive restarts.
  A whole-document quiz or flashcard set draws on key points spread over the chapter summaries.
- **Server** (`server/http_server.py`): async FastAPI. Graph runs stream over server-sent events and keep going
  (and get checkpointed) if the browser disconnects; PDF ingest, indexing and summaries run off the event loop.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/documents` | Uploaded PDFs with index / summary status |
| `POST /api/upload` | Upload a PDF (multipart `file`) |
| `POST /api/chat` | `{thread_id, doc_id, message, intent?}`, streams progress as server-sent events. `intent` (`ask` / `summary` / `flashcard` / `quiz`) skips the router's intent guess |
| `POST /api/quiz/answer` | `{thread_id, answers: {q1: "A", ...}}`, streams the grading |
| `GET /api/threads/<thread_id>` | Full chat history of a thread |
| `GET /api/documents/<doc_id>/summary` | Cached chapter summaries |
| `POST /api/documents/<doc_id>/summarize` | Start building the chapter summaries |
| `GET /api/documents/<doc_id>/pdf` | The PDF (page citations open it at `#page=N`) |

Interactive API docs: `http://localhost:5000/docs`.

### Evaluate

```powershell
python scripts/eval_retrieval.py                   # retrieval: Recall@k and MRR on eval/questions.jsonl
python scripts/eval_generation.py run              # answers: DeepEval faithfulness to the retrieved pages (k = 8)
python scripts/eval_generation.py run --k 5        # same with the top 5 chunks
```

Both import the same code the server uses (`src/retrieval/hybrid.py`, the study graph's `answer` node), so their
numbers describe the live app. Results land in `eval_out/`.

### Project structure

```
studymate/
├── environment.yml            # conda env: Python 3.11, CUDA PyTorch, Tesseract OCR, all packages
├── Dockerfile                 # Linux image built from environment.yml
├── compose.yaml               # runs the image with the GPU, .env, models and data folders
├── .env                       # your API key (create it; not in git)
├── server/
│   └── http_server.py         # async FastAPI app around the study graph; also serves the UI
├── web/
│   └── index.html             # the chat UI
├── src/
│   ├── ingest.py              # PDF -> pages -> clean -> chunks -> SQLite
│   ├── llm_client.py          # DeepSeek, or Ollama as fallback (LangChain chat models)
│   ├── pdf/                   # extractor (PyMuPDF), OCR fallback (Tesseract), cleaner, chunker
│   ├── retrieval/
│   │   ├── hybrid.py          # THE retrieval pipeline (app + eval)
│   │   ├── bm25_index.py  dense_index.py  reranker.py
│   │   └── tfidf_index.py  search.py   # old TF-IDF baseline, kept for the benchmark
│   ├── graphs/                # LangGraph
│   │   ├── study_graph.py     # router -> retrieval -> answer / summary / flashcards / quiz
│   │   ├── ingest_graph.py    # chapter summaries: write -> check -> revise
│   │   ├── nodes.py  prompts.py  schemas.py  state.py
│   │   └── store.py           # shared repository + summary cache
│   └── db/                    # SQLite schema, models, repository
├── scripts/
│   ├── eval_retrieval.py      # retrieval benchmark
│   ├── eval_generation.py     # answer-faithfulness benchmark (DeepEval)
│   ├── chunks.py              # sample chunks to write benchmark questions from
│   └── clean_noise.py         # strip extraction noise from documents already in the database
├── eval/                      # the benchmark: questions.jsonl, docs.json, CHANGELOG.md
├── eval_out/                  # benchmark results
├── sample/                    # sampled chunks and draft questions per book
├── Data/                      # created at runtime, not in git
│   ├── Database/              # app.db (documents, pages, chunks), checkpoints.db (chat threads, quiz mastery)
│   ├── Cache/                 # BM25 and BGE-M3 indexes, chapter summaries
│   └── pdfs/                  # PDFs ingested before the server existed
└── uploads/                   # PDFs uploaded through the UI, not in git
```
