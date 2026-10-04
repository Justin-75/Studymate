## StudyMate — PDF study assistant (RAG)

Upload a PDF, then generate summaries, flashcards and quizzes from it, with automatic grading.
Works on Chinese and English documents.

**Retrieval:** BM25 (jieba) + BGE-M3 dense (FAISS) → Reciprocal Rank Fusion → bge-reranker-v2-m3.
On the 181-question bilingual benchmark (`eval/questions.jsonl`) this reaches Recall@1 0.873 and
MRR 0.923, against 0.591 / 0.691 for the earlier TF-IDF pipeline. See `eval_out/`.

**Generation:** any OpenAI-compatible LLM — DeepSeek (default), local Ollama, or Groq.

### Install

```powershell
conda env create -f environment.yml
conda activate studymate
```

### Configure the LLM

Create a `.env` file in the project root:

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key_here
# LLM_MODEL=deepseek-flash      (optional override)
```

For a local model instead: `LLM_PROVIDER=ollama` and `LLM_MODEL=<model you pulled>`, with `ollama serve` running.
BGE-M3 is read from `BGE_M3_PATH` (default `E:\models\bge-m3`), otherwise downloaded from Hugging Face.

### Run

```powershell
conda activate studymate
python server/http_server.py
```

The server starts at `http://localhost:5000`. Open `test_frontend.html` in your browser.

### API

| Endpoint | Input | Output |
|---|---|---|
| `POST /api/upload` | PDF file (multipart/form-data) | `{doc_id}` — also builds the BM25 + dense indexes |
| `POST /api/generate` | `{doc_id, mode, query, top_k}`, mode = summary / flashcards / quiz | `{data: {...}}` |
| `POST /api/grade` | `{quiz, answers}` | `{score, total, details}` |

### Evaluate retrieval

```powershell
python scripts/eval_retrieval.py
```

The eval imports the same `src/retrieval/hybrid.py` the server uses, so its numbers describe the live app.

### Project structure

```
studymate/
├── server/http_server.py      # Flask API
├── src/
│   ├── ingest.py              # PDF -> pages -> clean -> chunks -> SQLite
│   ├── pdf/                   # extractor, OCR fallback, cleaner, chunker
│   ├── retrieval/
│   │   ├── hybrid.py          # THE retrieval pipeline (app + eval)
│   │   ├── bm25_index.py  dense_index.py  reranker.py
│   │   └── tfidf_index.py  search.py   # old TF-IDF baseline, kept for the benchmark
│   ├── generation/            # LLM prompts, quiz grading
│   ├── llm_client.py          # the one place that calls the LLM
│   └── db/                    # SQLite schema + repository
├── eval/                      # benchmark questions + changelog
├── scripts/eval_retrieval.py  # benchmark runner
└── test_frontend.html
```
