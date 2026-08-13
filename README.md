# Self-Correcting RAG Pipeline — Exam & Interview Prep

A modular Retrieval-Augmented Generation (RAG) pipeline with hybrid retrieval, self-correction agents, cross-encoder reranking, and grounded answer verification — extended with multi-format ingestion (PDF/DOCX/OCR'd screenshots), topic-scoped retrieval, a FastAPI backend, a Next.js frontend, and a faithfulness-based evaluation harness.

This project answers questions using information from your own notes, cheatsheets, and saved posts (e.g. Instagram/LinkedIn screenshots) instead of relying purely on an LLM's general knowledge. It retrieves relevant chunks from your material, generates an answer grounded in that material, and verifies the answer before returning it — so you get answers that match *your* course, *your* cheat sheet phrasing, and *your* saved references, with sources attached.

---
   title: exam-rag-api
   sdk: docker
   app_port: 7860
   ---
   
**New in this fork** (on top of the original self-correcting pipeline):
- Multi-format ingestion: `.txt`, `.md`, `.pdf`, `.docx`, and OCR'd images (`.png`/`.jpg`) — see `indexes/ingestion.py`
- Topic-scoped retrieval: chunks are tagged by subject folder (`data/raw/<topic>/...`) so an OS question doesn't pull DBMS notes
- Doc-type-aware chunking: cheat sheets and OCR'd social posts get smaller, denser chunks than full notes (`config.yaml → chunking.overrides`)
- Faithfulness scoring in `evaluate.py`: an LLM-judge groundedness score (0.0–1.0) alongside the original keyword recall
- `api/` — a FastAPI wrapper exposing `/ask`, `/ingest`, `/topics`, `/eval` for the frontend (or any client)
- `frontend/` — a minimal Next.js + Tailwind UI: pick a topic, drop in files, ask questions
- `Dockerfile` + `render.yaml` for deployment (see [Deployment](#deployment))

---

## What This Project Does

1. Searches documents using **vector similarity (FAISS)** and **keyword search (BM25)**, merged via **Reciprocal Rank Fusion (RRF)**
2. Rewrites weak queries automatically using a query rewrite agent
3. Reranks retrieved chunks using a **cross-encoder model** (falls back to LLM reranker)
4. Runs a **guardrail agent** (deterministic token overlap + LLM check) to ensure relevance
5. Generates answers using an LLM, strictly grounded in retrieved context
6. Verifies the answer using an **evaluator agent** with one automatic retry
7. Returns the answer with **source attribution**

---

## Key Features

- Hybrid retrieval — **FAISS + BM25 + RRF fusion**
- Query rewrite agent for weak queries
- **Cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) with LLM fallback
- Guardrail agent — deterministic fast-fail + LLM relevance check
- Evaluator agent with automatic retry on failure
- **Config-driven** — all parameters in `config.yaml`, no hardcoded values
- **Prompt templates** as separate files in `prompts/`
- **Streamlit web UI** (`app.py`)
- **Benchmarking script** (`evaluate.py`) with keyword recall metrics
- **63 unit + integration tests** across retrieval, agents, and pipeline
- Debug mode to inspect every pipeline step
- Source attribution on every answer

---

## High Level Flow

```
User Question
      ↓
Hybrid Retrieval (FAISS + BM25 + RRF)
      ↓
Query Rewrite Agent  ← only if retrieval is weak
      ↓
Cross-Encoder Reranker
      ↓
Guardrail Agent  ← blocks irrelevant context
      ↓
Answer Generation
      ↓
Evaluator Agent  ← retries once on failure
      ↓
Final Answer + Sources
```

---

## Project Structure

```
self-correcting-rag/
│
├── data/
│   ├── raw/                        # Input documents (.txt, .md)
│   └── processed/                  # Generated indexes (gitignored)
│
├── indexes/
│   ├── ingestion.py                # Document loading and chunking
│   ├── embed.py                    # Embedding generation
│   ├── index.py                    # FAISS HNSW index builder
│   └── bm25_index.py               # BM25 index builder
│
├── retrieval/
│   ├── retrieve.py                 # FAISS search
│   ├── bm25_retrieve.py            # BM25 search
│   └── cross_encoder_reranker.py   # Cross-encoder reranker (NEW)
│
├── rag/
│   └── rag.py                      # Full pipeline orchestration
│
├── llm/
│   └── llm.py                      # Groq LLM wrapper
│
├── prompts/                        # LLM prompt templates (NEW)
│   ├── query_rewrite.txt
│   ├── reranker.txt
│   ├── guardrail.txt
│   ├── answer.txt
│   └── evaluator.txt
│
├── tests/                          # Test suite (NEW)
│   ├── test_retrieval.py           # BM25, FAISS, RRF, cross-encoder tests
│   ├── test_agents.py              # Guardrail and prompt template tests
│   └── test_pipeline.py            # End-to-end integration tests
│
├── config.yaml                     # Central configuration (NEW)
├── evaluate.py                     # Benchmarking script (NEW)
├── app.py                          # Streamlit web UI (NEW)
├── main.py                         # CLI entry point
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Python 3.9 or higher is recommended.

```bash
git clone https://github.com/praharshithavishnubhatla/self-correcting-rag.git
cd self-correcting-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

---

## Configuration

All tunable parameters are in `config.yaml` — no hardcoded values anywhere in the codebase:

```yaml
llm:
  model: llama-3.3-70b-versatile
  temperature: 0.2
  max_tokens: 1024

embedding:
  index_model: BAAI/bge-base-en-v1.5
  query_model: BAAI/bge-base-en-v1.5   # must match index_model

retrieval:
  top_k: 8
  relevance_threshold: 1.3

reranker:
  use_cross_encoder: true
  cross_encoder_model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Running the Project

**CLI (interactive):**
```bash
python main.py
```

**CLI (single query):**
```bash
python main.py --query "explain scalability"
python main.py --query "how does caching work?" --debug
```

**Web UI:**
```bash
streamlit run app.py
```

**Benchmark:**
```bash
python evaluate.py
```

The pipeline auto-builds all indexes on first run.

---

## Debug Mode

```bash
python main.py --debug
```

Example output:
```
[DEBUG] Best FAISS distance: 0.4907
[DEBUG] Reranked sources:
  - design.txt
  - system.txt
  - system.md
[DEBUG] Guardrail: YES
[DEBUG] Evaluator: PASS
```

---

## Running Tests

```bash
# Unit tests — no API key needed
pytest tests/test_retrieval.py tests/test_agents.py -v

# Full pipeline tests — requires GROQ_API_KEY
pytest tests/test_pipeline.py -v -s

# All tests
pytest tests/ -v
```

**Test results: 63/63 passing**

---

## Benchmark Results

Run against 10 questions from the knowledge base:

| Metric | Score |
|---|---|
| Avg keyword recall | 0.89 |
| Guardrail PASS rate | 100% |
| Evaluator PASS rate | 100% |
| Avg latency | ~3–6s |

---

## Example Query

```
Ask: explain scalability
```

Output:
```
Scalability refers to a system's ability to handle growing workloads
without performance degradation.

Sources:
  - system.md
  - system.txt
```

---

## Technologies Used

| Tool | Purpose |
|---|---|
| FAISS | Vector similarity search (HNSW index) |
| BM25 | Keyword search |
| Sentence Transformers | Document + query embeddings (`BAAI/bge-base-en-v1.5`) |
| Cross-Encoder | Reranking (`ms-marco-MiniLM-L-6-v2`) |
| Groq API | LLM inference (`llama-3.3-70b-versatile`) |
| Streamlit | Web UI |
| pytest | Test suite |
| PyYAML | Config management |

---

## Running the API + Frontend

**Backend (FastAPI):**

```
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Endpoints: `GET /health`, `GET /topics`, `POST /ask`, `POST /ingest` (multipart upload), `POST /eval`, `GET /eval/latest`.

**Frontend (Next.js):**

```
cd frontend
npm install
cp .env.example .env.local     # point NEXT_PUBLIC_API_URL at your backend
npm run dev
```

Open `http://localhost:3000` — pick a topic, drop in a note/cheatsheet/screenshot, and ask a question.

---

## Deployment

**Backend + Postgres → Render**, using the included `render.yaml` blueprint:

```
# from the Render dashboard: New → Blueprint → point at this repo
```

This provisions a Docker web service (built from `Dockerfile`, which installs `tesseract-ocr` and `poppler-utils` for OCR), a persistent disk mounted at `/app/data` (so uploaded files and FAISS/BM25 indexes survive redeploys), and a managed Postgres instance. Set `GROQ_API_KEY` in the Render dashboard after the first deploy.

Avoid Render's free tier for a link you're putting on a resume — services spin down after 15 minutes idle and cold-start in ~30-60s. The `starter` plan in `render.yaml` avoids this.

**Frontend → Vercel** (recommended over Render for this piece — Vercel is purpose-built for Next.js):

```
cd frontend
vercel
```

Set `NEXT_PUBLIC_API_URL` to your deployed Render backend URL in the Vercel project's environment variables.

---

## License

This project is licensed under the MIT License.

## Author

Vishnubhatla Praharshitha