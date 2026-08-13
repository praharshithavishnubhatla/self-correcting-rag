"""
api/main.py
-----------
Thin FastAPI wrapper around the existing self-correcting RAG pipeline
(rag/rag.py) and ingestion pipeline (indexes/*). No pipeline logic lives
here — this module only handles HTTP, file uploads, and (re)index
triggering.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health
    GET  /topics
    POST /ask
    POST /ask/stream     (Server-Sent Events — live pipeline stage progress)
    POST /ingest         (multipart file upload)
    POST /ingest/batch   (multipart, multiple files)
    POST /eval
    GET  /eval/latest
"""

import json
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from threading import Lock

import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from api.schemas import (
    AskRequest,
    AskResponse,
    BatchIngestResponse,
    BatchIngestResult,
    EvalRunRequest,
    EvalSummary,
    IngestResponse,
    TopicsResponse,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────
_cfg_path = Path(__file__).parent.parent / "config.yaml"
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)

_paths = _cfg["paths"]
_api_cfg = _cfg.get("api", {})
RAW_DATA_DIR = Path(_paths["raw_data"])
MAX_UPLOAD_BYTES = _api_cfg.get("max_upload_mb", 20) * 1024 * 1024

app = FastAPI(title="Exam Prep RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_api_cfg.get("cors_origins", ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth ──────────────────────────────────────────────────
# Protects the state-changing / expensive endpoints (uploads that write to
# disk and trigger a full reindex, and eval runs that burn LLM calls).
# /health, /topics, /ask, /ask/stream, and /eval/latest stay open so the
# demo link is still usable read-only without a key.
#
# Set API_KEY in the environment (Render dashboard / .env) to turn this on.
# If it's unset, auth is a no-op — convenient for local dev, but you should
# always set it before sharing a public demo URL.
API_KEY = os.getenv("API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

if not API_KEY:
    print(
        "[api.main] WARNING: API_KEY is not set — /ingest, /ingest/batch, and "
        "/eval are UNAUTHENTICATED. Set API_KEY before deploying publicly."
    )


def require_api_key(provided: str | None = Depends(_api_key_header)):
    if not API_KEY:
        return  # auth disabled (no key configured) — local/dev mode
    if not provided or not secrets.compare_digest(provided, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


# ── Path safety ───────────────────────────────────────────
# `topic`, `doc_type`, and the uploaded filename are all attacker-controlled
# strings that used to be joined directly onto RAW_DATA_DIR. A filename like
# "../../../etc/cron.d/evil" or a topic of "../../app" would previously let
# a caller write files outside data/raw entirely. Both helpers below collapse
# their input to a single safe path component before it ever touches disk.

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]")


def _sanitize_component(value: str, field_name: str) -> str:
    """Reduce a user-supplied topic/doc_type to one safe directory name."""
    candidate = Path((value or "").strip()).name  # drops any '/', '\', or '..' segments
    candidate = _UNSAFE_CHARS.sub("_", candidate).strip(". ")
    if not candidate:
        raise HTTPException(status_code=400, detail=f"Invalid `{field_name}`.")
    return candidate


def _safe_upload_path(target_dir: Path, filename: str) -> Path:
    """Resolve an uploaded filename to a path guaranteed to stay inside target_dir."""
    name = Path((filename or "").strip()).name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    target_dir_resolved = target_dir.resolve()
    target_path = (target_dir_resolved / name).resolve()
    if target_path.parent != target_dir_resolved:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return target_path

# Reindexing touches shared pickle/FAISS files on disk — serialize it.
_reindex_lock = Lock()


def _rebuild_indexes():
    """
    Full pipeline rebuild: ingest -> chunk -> embed -> FAISS -> BM25.
    Synchronous and blocking — fine at portfolio/demo scale. At real
    scale this would be an async job (Celery/RQ or a Render background
    worker) so uploads don't block the request thread.
    """
    from indexes.ingestion import load_docs, chunk_docs
    from indexes.embed import main as embed_main
    from indexes.index import main as index_main
    from indexes.bm25_index import main as bm25_main

    docs = load_docs()
    chunk_docs(docs)
    embed_main()
    index_main()
    bm25_main(force=True)

    # The retrieval modules cache the FAISS index/chunks/BM25 index in memory
    # for the life of the process. Without this, a running server would keep
    # serving the pre-upload index until restarted.
    from retrieval.retrieve import invalidate as invalidate_faiss_cache
    from retrieval.bm25_retrieve import invalidate as invalidate_bm25_cache
    invalidate_faiss_cache()
    invalidate_bm25_cache()


# ── Health ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Topics ────────────────────────────────────────────────

@app.get("/topics", response_model=TopicsResponse)
def list_topics():
    if not RAW_DATA_DIR.exists():
        return TopicsResponse(topics=[])
    topics = sorted(
        p.name for p in RAW_DATA_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    return TopicsResponse(topics=topics)


# ── Ask ───────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    from rag.rag import run_pipeline, build_revision_notes, build_practice_questions

    if req.mode == "explain":
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="`question` is required for mode='explain'.")
        history = [{"role": t.role, "content": t.content} for t in req.history] if req.history else None
        try:
            result = run_pipeline(req.question, debug=req.debug, topic=req.topic, history=history)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Indexes not built yet. Upload documents via /ingest first.",
            )
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            topic=req.topic,
            mode=req.mode,
            guardrail_passed=result["guardrail_passed"],
            evaluator_verdict=result["evaluator_verdict"],
            debug_info=result.get("debug_info") if req.debug else None,
        )

    # revise / practice: broad-coverage, whole-topic modes — no single query
    if not req.topic:
        raise HTTPException(status_code=400, detail=f"`topic` is required for mode='{req.mode}'.")

    try:
        if req.mode == "revise":
            result = build_revision_notes(req.topic, debug=req.debug)
        else:
            result = build_practice_questions(req.topic, debug=req.debug)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Indexes not built yet. Upload documents via /ingest first.",
        )

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        topic=req.topic,
        mode=req.mode,
        guardrail_passed=None,
        evaluator_verdict=None,
        debug_info=result.get("debug_info"),
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """
    Server-Sent Events version of /ask. Streams one event per pipeline
    stage ({"stage": "retrieve", "message": "..."}, etc.) so the UI can show
    live progress through the ~5-6 sequential LLM calls a single question
    makes, instead of a single blank spinner. The final event has
    stage="done" and carries the same fields as AskResponse; on failure the
    final event has stage="error".

    Note: uses a plain POST + fetch/ReadableStream on the client rather than
    the browser's native EventSource, since EventSource only supports GET
    and can't send a JSON body (needed for question/topic/history/mode).
    """
    from rag.rag import (
        build_practice_questions_stream,
        build_revision_notes_stream,
        run_pipeline_stream,
    )

    if req.mode == "explain" and not req.question.strip():
        raise HTTPException(status_code=400, detail="`question` is required for mode='explain'.")
    if req.mode != "explain" and not req.topic:
        raise HTTPException(status_code=400, detail=f"`topic` is required for mode='{req.mode}'.")

    def event_generator():
        try:
            if req.mode == "explain":
                history = (
                    [{"role": t.role, "content": t.content} for t in req.history]
                    if req.history else None
                )
                gen = run_pipeline_stream(req.question, debug=req.debug, topic=req.topic, history=history)
            elif req.mode == "revise":
                gen = build_revision_notes_stream(req.topic, debug=req.debug)
            else:
                gen = build_practice_questions_stream(req.topic, debug=req.debug)

            for event in gen:
                if event["stage"] == "done":
                    result = event["result"]
                    yield _sse({
                        "stage": "done",
                        "message": event.get("message", ""),
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "topic": req.topic,
                        "mode": req.mode,
                        "guardrail_passed": result.get("guardrail_passed"),
                        "evaluator_verdict": result.get("evaluator_verdict"),
                        "debug_info": result.get("debug_info") if req.debug else None,
                    })
                else:
                    yield _sse({"stage": event["stage"], "message": event.get("message", "")})
        except FileNotFoundError:
            yield _sse({
                "stage": "error",
                "message": "Indexes not built yet. Upload documents via /ingest first.",
            })
        except Exception as e:
            yield _sse({"stage": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering so events flush immediately
        },
    )


# ── Ingest ────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(
    file: UploadFile = File(...),
    topic: str = Form(default="general"),
    doc_type: str = Form(default=None),
):
    """
    Upload a note, cheatsheet, or screenshot (Instagram/LinkedIn export).
    Saves under data/raw/<topic>/[<doc_type>/]<filename>, then triggers
    a full reindex so the file is immediately queryable.
    """
    topic = _sanitize_component(topic, "topic")
    if doc_type:
        doc_type = _sanitize_component(doc_type, "doc_type")

    ext = Path(file.filename).suffix.lower()
    from indexes.ingestion import SUPPORTED_EXTENSIONS
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    target_dir = RAW_DATA_DIR / topic
    if doc_type:
        target_dir = target_dir / doc_type
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = _safe_upload_path(target_dir, file.filename)
    with open(target_path, "wb") as f:
        f.write(contents)

    with _reindex_lock:
        # Clear cached chunks/embeddings so the rebuild picks up the new file.
        for key in ("chunks", "chunks_with_embeddings", "faiss_index", "bm25_index"):
            p = Path(_paths[key])
            if p.exists():
                p.unlink()
        _rebuild_indexes()

    from indexes.ingestion import _infer_doc_type
    rel_path = target_path.relative_to(RAW_DATA_DIR.resolve())
    inferred_type = doc_type or _infer_doc_type(rel_path, ext)

    with open(_paths["chunks"], "rb") as f:
        import pickle
        chunks = pickle.load(f)
    chunks_added = sum(1 for c in chunks if c["source"] == str(rel_path))

    return IngestResponse(
        source=str(rel_path),
        topic=topic,
        doc_type=inferred_type,
        chunks_added=chunks_added,
        reindexed=True,
    )


@app.post("/ingest/batch", response_model=BatchIngestResponse, dependencies=[Depends(require_api_key)])
async def ingest_batch(
    files: list[UploadFile] = File(...),
    topic: str = Form(default="general"),
    doc_type: str = Form(default=None),
):
    """
    Upload several files at once (e.g. a dump of notes + cheatsheets +
    screenshots for one subject). Saves every file to disk first, then
    triggers a single reindex at the end — avoids rebuilding FAISS/BM25
    once per file, which is what the plain /ingest endpoint does when
    called in a loop.
    """
    from indexes.ingestion import SUPPORTED_EXTENSIONS, _infer_doc_type

    topic = _sanitize_component(topic, "topic")
    if doc_type:
        doc_type = _sanitize_component(doc_type, "doc_type")

    saved = []  # (target_path, rel_path)
    results: list[BatchIngestResult] = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append(BatchIngestResult(
                filename=file.filename, ok=False,
                error=f"Unsupported file type '{ext}'",
            ))
            continue

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            results.append(BatchIngestResult(
                filename=file.filename, ok=False, error="File too large",
            ))
            continue

        target_dir = RAW_DATA_DIR / topic
        if doc_type:
            target_dir = target_dir / doc_type
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            target_path = _safe_upload_path(target_dir, file.filename)
        except HTTPException as e:
            results.append(BatchIngestResult(
                filename=file.filename, ok=False, error=e.detail,
            ))
            continue
        with open(target_path, "wb") as f:
            f.write(contents)

        rel_path = target_path.relative_to(RAW_DATA_DIR.resolve())
        saved.append((target_path, rel_path, file.filename, ext))

    if not saved:
        return BatchIngestResponse(results=results, reindexed=False)

    with _reindex_lock:
        for key in ("chunks", "chunks_with_embeddings", "faiss_index", "bm25_index"):
            p = Path(_paths[key])
            if p.exists():
                p.unlink()
        _rebuild_indexes()

    import pickle
    with open(_paths["chunks"], "rb") as f:
        chunks = pickle.load(f)

    for target_path, rel_path, filename, ext in saved:
        inferred_type = doc_type or _infer_doc_type(rel_path, ext)
        chunks_added = sum(1 for c in chunks if c["source"] == str(rel_path))
        results.append(BatchIngestResult(
            filename=filename,
            ok=True,
            source=str(rel_path),
            doc_type=inferred_type,
            chunks_added=chunks_added,
        ))

    return BatchIngestResponse(results=results, reindexed=True)


# ── Evaluation ────────────────────────────────────────────

@app.post("/eval", response_model=EvalSummary, dependencies=[Depends(require_api_key)])
def run_eval(req: EvalRunRequest):
    from evaluate import DEFAULT_QA_PAIRS, run_evaluation

    qa_pairs = DEFAULT_QA_PAIRS
    results = run_evaluation(
        qa_pairs, debug=False, topic=req.topic,
        include_faithfulness=req.include_faithfulness,
    )

    n = len(results) or 1
    summary = EvalSummary(
        questions_evaluated=len(results),
        avg_keyword_recall=round(sum(r["keyword_recall"] for r in results) / n, 3),
        avg_faithfulness=(
            round(sum(r.get("faithfulness", 0) for r in results) / n, 3)
            if req.include_faithfulness else None
        ),
        guardrail_pass_rate=round(sum(r["guardrail_passed"] for r in results) / n, 3),
        evaluator_pass_rate=round(
            sum(r["evaluator_verdict"] == "PASS" for r in results) / n, 3
        ),
        avg_latency_s=round(sum(r["latency_s"] for r in results) / n, 2),
    )

    out = Path(_paths["eval_output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    return summary


@app.get("/eval/latest")
def latest_eval():
    out = Path(_paths["eval_output"])
    if not out.exists():
        raise HTTPException(status_code=404, detail="No evaluation has been run yet.")
    with open(out) as f:
        return json.load(f)