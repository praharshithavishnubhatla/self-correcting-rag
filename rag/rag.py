import os
import re
import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import yaml

from retrieval.retrieve import faiss_search, get_topic_chunks
from retrieval.bm25_retrieve import bm25_search
from llm.llm import generate_answer, DEFAULT_SYSTEM_MESSAGE, FAST_MODEL

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── Load config ───────────────────────────────────────────
def _load_config():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_cfg = _load_config()
_retrieval = _cfg["retrieval"]
_reranker_cfg = _cfg["reranker"]

RELEVANCE_THRESHOLD = _retrieval["relevance_threshold"]
TOP_K = _retrieval["top_k"]
RRF_K = _retrieval["rrf_k"]
RERANK_TOP_N = _reranker_cfg["top_n"]
GUARDRAIL_AUTO_PASS_DISTANCE = _cfg.get("guardrail", {}).get("auto_pass_distance")


# ── Load prompt templates ─────────────────────────────────
def _load_prompt(name: str) -> str:
    path = Path(__file__).parent.parent / "prompts" / f"{name}.txt"
    with open(path) as f:
        return f.read()

_PROMPT_QUERY_REWRITE = _load_prompt("query_rewrite")
_PROMPT_RERANKER      = _load_prompt("reranker")
_PROMPT_GUARDRAIL     = _load_prompt("guardrail")
_PROMPT_ANSWER        = _load_prompt("answer")
_PROMPT_EVALUATOR     = _load_prompt("evaluator")
_PROMPT_REVISE        = _load_prompt("revise")
_PROMPT_REVISE_MAP    = _load_prompt("revise_map")
_PROMPT_PRACTICE      = _load_prompt("practice")

_MODES_CFG = _cfg.get("modes", {})
_REVISE_CFG = _MODES_CFG.get("revise", {})
_PRACTICE_CFG = _MODES_CFG.get("practice", {})


# ── Cross-encoder (lazy loaded) ───────────────────────────
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None and _reranker_cfg.get("use_cross_encoder", False):
        from retrieval.cross_encoder_reranker import get_reranker
        _cross_encoder = get_reranker()
    return _cross_encoder


# ── Hybrid Retrieval ──────────────────────────────────────

def hybrid_retrieve(query: str, k: int = None, topic: str = None):
    """Merge FAISS dense + BM25 sparse results via Reciprocal Rank Fusion.

    If topic is given, results are scoped to that topic/subject so retrieval
    doesn't cross-contaminate unrelated exam material (e.g. an OS question
    pulling DBMS notes).
    """
    k = k or TOP_K

    faiss_results, distances = faiss_search(query, k=k, topic=topic)
    bm25_results = bm25_search(query, k=k, topic=topic)

    rrf_scores = {}
    doc_map = {}

    for rank, result in enumerate(faiss_results):
        key = (result["source"], result["text"][:80])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        doc_map[key] = result

    for rank, result in enumerate(bm25_results):
        key = (result["source"], result["text"][:80])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        doc_map[key] = result

    merged = sorted(
        doc_map.values(),
        key=lambda r: rrf_scores[(r["source"], r["text"][:80])],
        reverse=True,
    )

    return merged[:k], distances


# ── Query Rewrite ─────────────────────────────────────────

def rewrite_query(question: str) -> str:
    prompt = _PROMPT_QUERY_REWRITE.format(question=question)
    return generate_answer(prompt, model=FAST_MODEL, max_tokens=64).strip()


# ── Reranker ──────────────────────────────────────────────

def rerank(question: str, contexts: list, top_n: int = None) -> list:
    """
    Rerank using cross-encoder if enabled, otherwise fall back to LLM reranker.
    """
    top_n = top_n or RERANK_TOP_N

    ce = _get_cross_encoder()
    if ce and ce.available:
        return ce.rerank(question, contexts, top_n=top_n)

    # ── LLM fallback reranker ─────────────────────────────
    context_text = "\n\n".join(
        f"ID:{i}\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    prompt = _PROMPT_RERANKER.format(
        top_n=top_n,
        question=question,
        chunks=context_text,
    )
    response = generate_answer(prompt, model=FAST_MODEL, max_tokens=128)
    ids = list(dict.fromkeys(
        int(x) for x in re.findall(r"\d+", response)
        if int(x) < len(contexts)
    ))

    reranked = [contexts[i] for i in ids]
    return reranked if reranked else contexts[:top_n]


# ── Guardrail ─────────────────────────────────────────────

def guardrail_check(question: str, contexts: list, best_distance: float = None) -> str:
    # Skip the LLM call entirely when retrieval was already clearly
    # confident — saves a full Groq round-trip on the common case.
    if (
        GUARDRAIL_AUTO_PASS_DISTANCE is not None
        and best_distance is not None
        and best_distance <= GUARDRAIL_AUTO_PASS_DISTANCE
    ):
        return "YES"

    context_text = "\n\n".join(c["text"] for c in contexts)

    # Deterministic fallback: token overlap check
    q_tokens = set(question.lower().split())
    c_tokens = set(context_text.lower().split())
    if len(q_tokens & c_tokens) >= 2:
        pass
    else:
        return "NO"  # Fast-fail: no lexical overlap at all

    prompt = _PROMPT_GUARDRAIL.format(
        question=question,
        context=context_text,
    )
    decision = generate_answer(prompt, max_tokens=10).strip().upper()
    return "YES" if "YES" in decision else "NO"


# ── Chat history ───────────────────────────────────────────

def _format_history(history: list, max_turns: int = 6) -> str:
    """
    Render the last few turns as plain text for the answer prompt. Kept
    short (default last 6 turns = 3 exchanges) to avoid ballooning the
    prompt — this is for conversational continuity, not full transcript
    recall.
    """
    if not history:
        return "(no prior conversation)"
    turns = history[-max_turns:]
    lines = []
    for turn in turns:
        role = "Student" if turn.get("role") == "user" else "Tutor"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no prior conversation)"


# ── Answer generation ─────────────────────────────────────

def build_answer(question: str, contexts: list, history: list = None) -> str:
    context_block = "\n\n".join(
        f"SOURCE: {c['source']}\n{c['text']}"
        for c in contexts
    )
    prompt = _PROMPT_ANSWER.format(
        context=context_block,
        question=question,
        history=_format_history(history),
    )
    return generate_answer(prompt)


# ── Evaluator ─────────────────────────────────────────────

def evaluator_check(question: str, answer: str, contexts: list) -> str:
    context_text = "\n\n".join(c["text"] for c in contexts)
    prompt = _PROMPT_EVALUATOR.format(
        question=question,
        context=context_text,
        answer=answer,
    )
    verdict = generate_answer(prompt, max_tokens=10).strip().upper()
    return "PASS" if "PASS" in verdict else "FAIL"


# ── Broad-coverage modes (revise / practice) ───────────────
#
# These don't use hybrid_retrieve/rerank/guardrail at all — those are built
# for "find the chunks most relevant to this specific question." Revise and
# practice want the opposite: full coverage of a topic. So they pull every
# chunk tagged with the topic via get_topic_chunks() instead.

def _render_chunks(chunks: list) -> str:
    return "\n\n".join(f"SOURCE: {c['source']}\n{c['text']}" for c in chunks)


def _batch_chunks(chunks: list, char_budget: int) -> list:
    """Group chunks into batches that each stay under char_budget characters."""
    batches, current, current_len = [], [], 0
    for c in chunks:
        text_len = len(c["text"])
        if current and current_len + text_len > char_budget:
            batches.append(current)
            current, current_len = [], 0
        current.append(c)
        current_len += text_len
    if current:
        batches.append(current)
    return batches


def _compress_chunks_stream(chunks: list, topic: str, batch_budget: int, label: str = "mode"):
    """
    Map step, streaming version: yields a progress event per batch as it's
    compressed, then a final event carrying the merged text + batch count.
    """
    batches = _batch_chunks(chunks, batch_budget)
    compressed = []
    for i, batch in enumerate(batches):
        yield {
            "stage": "compress",
            "message": f"{label}: compressing batch {i + 1}/{len(batches)} ({len(batch)} chunks)…",
        }
        prompt = _PROMPT_REVISE_MAP.format(topic=topic or "general", context=_render_chunks(batch))
        compressed.append(generate_answer(prompt, temperature=0.2, max_tokens=1024))
    yield {"stage": "compress_done", "merged_text": "\n\n".join(compressed), "batch_count": len(batches)}


def _compress_chunks(chunks: list, topic: str, batch_budget: int, debug: bool = False, label: str = "mode"):
    """Non-streaming wrapper around _compress_chunks_stream."""
    merged_text, batch_count = "", 0
    for event in _compress_chunks_stream(chunks, topic, batch_budget, label=label):
        if debug and event["stage"] == "compress":
            print(f"[DEBUG] {event['message']}")
        if event["stage"] == "compress_done":
            merged_text, batch_count = event["merged_text"], event["batch_count"]
    return merged_text, batch_count


def build_revision_notes_stream(topic: str, debug: bool = False):
    """
    Streaming version of build_revision_notes — yields progress events while
    gathering/compressing material, then a final "done" event with the same
    result dict shape build_revision_notes() returns.
    """
    yield {"stage": "gather", "message": f"Gathering material for '{topic}'…"}
    chunks = get_topic_chunks(topic)

    if not chunks:
        result = {
            "answer": f"I don't have any material for '{topic}' yet — upload some notes, "
                      f"a cheatsheet, or saved screenshots for this topic first.",
            "sources": [],
            "debug_info": {"chunk_count": 0} if debug else None,
        }
        yield {"stage": "done", "message": "No material found.", "result": result}
        return

    sources = sorted({c["source"] for c in chunks})
    total_chars = sum(len(c["text"]) for c in chunks)
    single_pass_budget = _REVISE_CFG.get("single_pass_char_budget", 12000)
    batch_budget = _REVISE_CFG.get("batch_char_budget", 6000)
    debug_info = {"chunk_count": len(chunks), "total_chars": total_chars, "sources": sources}

    if total_chars <= single_pass_budget:
        context = _render_chunks(chunks)
        debug_info["mode"] = "single_pass"
    else:
        merged_text, batch_count = "", 0
        for event in _compress_chunks_stream(chunks, topic, batch_budget, label="Revise"):
            if event["stage"] == "compress_done":
                merged_text, batch_count = event["merged_text"], event["batch_count"]
            else:
                yield event
        context = merged_text
        debug_info["mode"] = "map_reduce"
        debug_info["batch_count"] = batch_count

    yield {"stage": "generate", "message": "Writing revision notes…"}
    prompt = _PROMPT_REVISE.format(topic=topic or "general", context=context)
    answer = generate_answer(
        prompt,
        temperature=_REVISE_CFG.get("temperature"),
        max_tokens=_REVISE_CFG.get("max_tokens"),
    )

    result = {"answer": answer, "sources": sources, "debug_info": debug_info if debug else None}
    yield {"stage": "done", "message": "Done.", "result": result}


def build_revision_notes(topic: str, debug: bool = False) -> dict:
    """
    Coach-style revision notes covering an entire topic. Single LLM pass if
    the topic's material fits in one context window, otherwise map-reduce:
    compress in batches, then run the final coach pass over the compressed
    material. Non-streaming wrapper around build_revision_notes_stream.
    """
    result = None
    for event in build_revision_notes_stream(topic, debug=debug):
        if debug and event["stage"] != "done":
            print(f"[DEBUG] {event['message']}")
        if event["stage"] == "done":
            result = event["result"]
    return result


def build_practice_questions_stream(topic: str, debug: bool = False):
    """Streaming version of build_practice_questions — same shape as
    build_revision_notes_stream."""
    yield {"stage": "gather", "message": f"Gathering material for '{topic}'…"}
    chunks = get_topic_chunks(topic)

    if not chunks:
        result = {
            "answer": f"I don't have any material for '{topic}' yet — upload some notes, "
                      f"a cheatsheet, or saved screenshots for this topic first.",
            "sources": [],
            "debug_info": {"chunk_count": 0} if debug else None,
        }
        yield {"stage": "done", "message": "No material found.", "result": result}
        return

    sources = sorted({c["source"] for c in chunks})
    total_chars = sum(len(c["text"]) for c in chunks)
    single_pass_budget = _PRACTICE_CFG.get("single_pass_char_budget", 12000)
    batch_budget = _PRACTICE_CFG.get("batch_char_budget", 6000)
    num_questions = _PRACTICE_CFG.get("num_questions", 8)
    debug_info = {"chunk_count": len(chunks), "total_chars": total_chars, "sources": sources}

    if total_chars <= single_pass_budget:
        context = _render_chunks(chunks)
        debug_info["mode"] = "single_pass"
    else:
        merged_text, batch_count = "", 0
        for event in _compress_chunks_stream(chunks, topic, batch_budget, label="Practice"):
            if event["stage"] == "compress_done":
                merged_text, batch_count = event["merged_text"], event["batch_count"]
            else:
                yield event
        context = merged_text
        debug_info["mode"] = "map_reduce"
        debug_info["batch_count"] = batch_count

    yield {"stage": "generate", "message": "Writing practice questions…"}
    prompt = _PROMPT_PRACTICE.format(topic=topic or "general", context=context, num_questions=num_questions)
    answer = generate_answer(
        prompt,
        temperature=_PRACTICE_CFG.get("temperature"),
        max_tokens=_PRACTICE_CFG.get("max_tokens"),
    )

    result = {"answer": answer, "sources": sources, "debug_info": debug_info if debug else None}
    yield {"stage": "done", "message": "Done.", "result": result}


def build_practice_questions(topic: str, debug: bool = False) -> dict:
    """
    Coach-generated practice Q&A covering an entire topic, same broad-coverage
    retrieval and map-reduce strategy as build_revision_notes. Non-streaming
    wrapper around build_practice_questions_stream.
    """
    result = None
    for event in build_practice_questions_stream(topic, debug=debug):
        if debug and event["stage"] != "done":
            print(f"[DEBUG] {event['message']}")
        if event["stage"] == "done":
            result = event["result"]
    return result


# ── Core pipeline ─────────────────────────────────────────

def ask(question: str, debug: bool = False, topic: str = None) -> str:
    """
    Run the full self-correcting RAG pipeline.
    Returns a formatted string with the answer and sources.
    """
    result = run_pipeline(question, debug=debug, topic=topic)
    sources_block = "\n".join(f"  - {s}" for s in result["sources"])
    return f"{result['answer']}\n\nSources:\n{sources_block}"


def run_pipeline_stream(question: str, debug: bool = False, topic: str = None, history: list = None):
    """
    Streaming version of run_pipeline — yields a progress event before each
    of the pipeline's sequential LLM/retrieval calls, then a final "done"
    event carrying the same result dict run_pipeline() returns. This is what
    powers the /ask/stream SSE endpoint, so the UI can show live stage
    progress ("Retrieving…", "Reranking…", "Generating answer…") instead of
    a blank spinner for the ~5-6 sequential calls a single question makes.

    Yields
    ------
    dict with keys:
        stage  : str   "retrieve" | "rewrite" | "rerank" | "guardrail"
                        | "generate" | "verify" | "done"
        message: str   human-readable status line
        result : dict  only present on the final "done" event
    """
    debug_info = {}
    final_query = question

    # Fold the previous user turn into the retrieval query so short
    # follow-ups still retrieve the right chunks. Cheap string concat, no
    # extra LLM call — deliberately not a full conversational-rewrite step
    # since that would add another sequential Groq call per question.
    retrieval_query = question
    if history:
        prev_user_turns = [t.get("content", "") for t in history if t.get("role") == "user" and t.get("content")]
        if prev_user_turns:
            retrieval_query = f"{prev_user_turns[-1]} {question}"

    # STEP 1 — Hybrid Retrieve
    yield {"stage": "retrieve", "message": "Retrieving relevant material…"}
    contexts, distances = hybrid_retrieve(retrieval_query, topic=topic)
    best_distance = float(min(distances))
    debug_info["best_faiss_distance"] = best_distance

    # STEP 2 — Query Rewrite if weak retrieval
    if best_distance > RELEVANCE_THRESHOLD:
        yield {"stage": "rewrite", "message": "Weak match — rewriting your question…"}
        better_query = rewrite_query(question)
        debug_info["rewritten_query"] = better_query

        yield {"stage": "retrieve", "message": "Retrieving again with the rewritten query…"}
        new_contexts, new_distances = hybrid_retrieve(better_query, topic=topic)
        new_best = float(min(new_distances))

        if new_best < best_distance:
            final_query = better_query
            contexts = new_contexts
            best_distance = new_best

        if best_distance > RELEVANCE_THRESHOLD:
            result = {
                "answer": "I don't have information about this in my knowledge base.",
                "sources": [],
                "guardrail_passed": False,
                "evaluator_verdict": "SKIP",
                "debug_info": debug_info,
            }
            yield {"stage": "done", "message": "No relevant material found.", "result": result}
            return

    # STEP 3 — Rerank
    yield {"stage": "rerank", "message": "Reranking retrieved chunks…"}
    contexts = rerank(final_query, contexts)

    # Deduplicate by source
    seen = set()
    unique_contexts = []
    for c in contexts:
        if c["source"] not in seen:
            unique_contexts.append(c)
            seen.add(c["source"])
    contexts = unique_contexts
    debug_info["reranked_sources"] = [c["source"] for c in contexts]

    # STEP 4 — Guardrail
    yield {"stage": "guardrail", "message": "Checking relevance…"}
    decision = guardrail_check(final_query, contexts, best_distance=best_distance)
    debug_info["guardrail"] = decision

    if decision != "YES":
        result = {
            "answer": "I don't have enough relevant information to answer that.",
            "sources": [],
            "guardrail_passed": False,
            "evaluator_verdict": "SKIP",
            "debug_info": debug_info,
        }
        yield {"stage": "done", "message": "Not enough relevant information.", "result": result}
        return

    # STEP 5 — Generate Answer
    yield {"stage": "generate", "message": "Generating answer…"}
    answer = build_answer(final_query, contexts, history=history)

    # STEP 6 — Evaluator (with one retry)
    yield {"stage": "verify", "message": "Verifying the answer is grounded…"}
    verdict = evaluator_check(final_query, answer, contexts)
    debug_info["evaluator"] = verdict

    if verdict != "PASS":
        yield {"stage": "generate", "message": "Answer wasn't well-grounded — regenerating…"}
        answer = build_answer(final_query, contexts, history=history)

        yield {"stage": "verify", "message": "Re-verifying…"}
        verdict = evaluator_check(final_query, answer, contexts)
        debug_info["evaluator_retry"] = verdict

        if verdict != "PASS":
            result = {
                "answer": "I'm not confident enough in my answer. Please rephrase your question.",
                "sources": [],
                "guardrail_passed": True,
                "evaluator_verdict": "FAIL",
                "debug_info": debug_info,
            }
            yield {"stage": "done", "message": "Low confidence answer.", "result": result}
            return

    sources = sorted({c["source"] for c in contexts})
    result = {
        "answer": answer,
        "sources": sources,
        "contexts": contexts,  # retrieved chunks used to ground the answer
        "guardrail_passed": True,
        "evaluator_verdict": verdict,
        "debug_info": debug_info,
    }
    yield {"stage": "done", "message": "Done.", "result": result}


def run_pipeline(question: str, debug: bool = False, topic: str = None, history: list = None) -> dict:
    """
    Run the full self-correcting RAG pipeline. Non-streaming wrapper around
    run_pipeline_stream — runs the generator to completion and returns the
    final result dict. Used by callers that don't need stage progress
    (evaluate.py, tests, app.py, and the non-streaming /ask endpoint).

    history, if given, is a list of {"role": "user"|"assistant", "content": str}
    in chronological order. It's used two ways: (1) folded into the retrieval
    query so short follow-ups like "what about the second one" can still find
    the right chunks, without an extra LLM call, and (2) passed to the answer
    prompt so the tutor stays consistent with what's already been said.

    Returns
    -------
    dict with keys:
        answer          : str
        sources         : list[str]
        guardrail_passed: bool
        evaluator_verdict: str   (PASS / FAIL / SKIP)
        debug_info      : dict   (only populated when debug=True)
    """
    result = None
    for event in run_pipeline_stream(question, debug=debug, topic=topic, history=history):
        if debug and event["stage"] != "done":
            print(f"[DEBUG] stage={event['stage']}: {event['message']}")
        if event["stage"] == "done":
            result = event["result"]
    return result