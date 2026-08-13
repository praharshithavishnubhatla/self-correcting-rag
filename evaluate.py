"""
evaluate.py
-----------
Benchmarks the Self-Correcting RAG pipeline against your knowledge base.

Usage:
    python evaluate.py                       # default questions from your docs
    python evaluate.py --qa path/to/qa.json  # custom QA pairs
    python evaluate.py --debug               # show pipeline debug output
    python evaluate.py --output results.json # custom output path

Output: summary table printed + full results saved to data/eval_results.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── QA pairs grounded in your actual documents ────────────
# design.md (URL shortener), design.txt (scaling/caching/CDN),
# system.md (scalable design), system.txt (system design basics)

DEFAULT_QA_PAIRS = [
    {
        "id": "q1",
        "question": "what is scalability?",
        "expected_keywords": ["scalability", "workload", "performance", "handle", "growing"],
    },
    {
        "id": "q2",
        "question": "explain horizontal and vertical scaling",
        "expected_keywords": ["horizontal", "vertical", "server", "scale", "cpu"],
    },
    {
        "id": "q3",
        "question": "how does a load balancer work?",
        "expected_keywords": ["load balancer", "traffic", "distribut", "server", "request"],
    },
    {
        "id": "q4",
        "question": "what is caching and when should I use it?",
        "expected_keywords": ["cache", "memory", "database", "performance", "expir"],
    },
    {
        "id": "q5",
        "question": "what is a CDN and how does it work?",
        "expected_keywords": ["cdn", "static", "content", "server", "deliver"],
    },
    {
        "id": "q6",
        "question": "what are the differences between SQL and NoSQL databases?",
        "expected_keywords": ["sql", "nosql", "relational", "schema", "database"],
    },
    {
        "id": "q7",
        "question": "explain the CAP theorem",
        "expected_keywords": ["cap", "consistency", "availability", "partition"],
    },
    {
        "id": "q8",
        "question": "how would you design a URL shortener?",
        "expected_keywords": ["short", "url", "redirect", "id", "hash"],
    },
    {
        "id": "q9",
        "question": "what is database sharding?",
        "expected_keywords": ["shard", "database", "split", "partition"],
    },
    {
        "id": "q10",
        "question": "what causes single points of failure and how do you avoid them?",
        "expected_keywords": ["single point", "failure", "redundan", "availab", "failover"],
    },
]


# ── Metrics ───────────────────────────────────────────────

def keyword_recall(answer: str, keywords: list) -> float:
    """Fraction of expected keywords found (substring match, case-insensitive)."""
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 3)


_FAITHFULNESS_PROMPT = """You are grading whether an answer is faithful to (fully
supported by) the provided context. Faithful means every factual claim in the
answer can be traced back to the context — no invented facts, no unsupported
specifics.

Context:
{context}

Answer:
{answer}

Score faithfulness from 0.0 to 1.0, where 1.0 = fully grounded in the context
and 0.0 = mostly unsupported/hallucinated. Reply with ONLY the number."""


def faithfulness_score(answer: str, contexts: list) -> float:
    """
    LLM-as-judge groundedness score. Reuses the same LLM wrapper as the
    pipeline's evaluator agent, but scores 0.0-1.0 instead of PASS/FAIL —
    gives a continuous signal for tracking retrieval/prompt quality over time,
    on top of the pipeline's own binary evaluator_verdict.
    """
    if not answer or not contexts:
        return 0.0

    from llm.llm import generate_answer

    context_text = "\n\n".join(c["text"] for c in contexts)
    prompt = _FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)

    try:
        response = generate_answer(prompt, temperature=0.0).strip()
        score = float("".join(ch for ch in response if ch.isdigit() or ch == "."))
        return max(0.0, min(1.0, score))
    except (ValueError, Exception):
        return 0.0


# ── Runner ────────────────────────────────────────────────

def run_evaluation(
    qa_pairs: list,
    debug: bool = False,
    topic: str = None,
    include_faithfulness: bool = True,
) -> list:
    try:
        from rag.rag import run_pipeline
    except ImportError as e:
        raise ImportError(
            f"Could not import rag.rag.run_pipeline: {e}\n"
            "Make sure indexes are built by running: python main.py"
        )

    results = []
    print(f"\n{'─'*65}")
    print(f"  Self-Correcting RAG — Evaluation  ({len(qa_pairs)} questions)")
    if topic:
        print(f"  Scoped to topic: {topic}")
    print(f"{'─'*65}\n")

    for i, qa in enumerate(qa_pairs, 1):
        q = qa["question"]
        expected = qa.get("expected_keywords", [])
        print(f"[{i:02d}/{len(qa_pairs)}] {q}")

        start = time.time()
        try:
            result = run_pipeline(q, debug=debug, topic=topic)
            latency = round(time.time() - start, 2)

            recall = keyword_recall(result.get("answer", ""), expected)
            guard = result.get("guardrail_passed", False)
            verdict = result.get("evaluator_verdict", "—")
            sources = result.get("sources", [])

            faithfulness = None
            if include_faithfulness and result.get("answer") and guard:
                retrieved_contexts = result.get("contexts", [])
                if retrieved_contexts:
                    faithfulness = faithfulness_score(result["answer"], retrieved_contexts)

            status = "✓" if recall >= 0.4 else "✗"
            faith_str = f"  faith={faithfulness}" if faithfulness is not None else ""
            print(
                f"       {status}  recall={recall}  "
                f"guard={'PASS' if guard else 'FAIL'}  "
                f"eval={verdict}{faith_str}  "
                f"sources={len(sources)}  "
                f"{latency}s\n"
            )

            rows = {
                "id": qa.get("id", str(i)),
                "question": q,
                "answer": result.get("answer", ""),
                "sources": sources,
                "keyword_recall": recall,
                "faithfulness": faithfulness,
                "guardrail_passed": guard,
                "evaluator_verdict": verdict,
                "latency_s": latency,
                "error": None,
            }

        except Exception as e:
            latency = round(time.time() - start, 2)
            print(f"       ✗  ERROR: {e}\n")
            rows = {
                "id": qa.get("id", str(i)),
                "question": q,
                "answer": "",
                "sources": [],
                "keyword_recall": 0.0,
                "faithfulness": None,
                "guardrail_passed": False,
                "evaluator_verdict": "ERROR",
                "latency_s": latency,
                "error": str(e),
            }

        results.append(rows)

    return results


# ── Summary ───────────────────────────────────────────────

def print_summary(results: list):
    n = len(results)
    avg_recall  = round(sum(r["keyword_recall"] for r in results) / n, 3)
    avg_latency = round(sum(r["latency_s"] for r in results) / n, 2)
    guard_pct   = round(sum(r["guardrail_passed"] for r in results) / n * 100)
    pass_pct    = round(sum(r["evaluator_verdict"] == "PASS" for r in results) / n * 100)
    errors      = sum(1 for r in results if r["error"])
    high_recall = sum(1 for r in results if r["keyword_recall"] >= 0.4)

    print(f"\n{'═'*65}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'═'*65}")
    print(f"  Questions evaluated  : {n}")
    print(f"  Avg keyword recall   : {avg_recall}  ({high_recall}/{n} scored ≥ 0.4)")
    print(f"  Avg latency          : {avg_latency}s")
    print(f"  Guardrail PASS rate  : {guard_pct}%")
    print(f"  Evaluator PASS rate  : {pass_pct}%")
    print(f"  Errors               : {errors}")
    print(f"{'═'*65}\n")


# ── CLI ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark the RAG pipeline")
    parser.add_argument("--qa", type=str, default=None,
                        help="Path to JSON file: [{question, expected_keywords}]")
    parser.add_argument("--debug", action="store_true",
                        help="Show pipeline debug output per question")
    parser.add_argument("--output", type=str, default="data/eval_results.json",
                        help="Where to save results JSON")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Add it to your .env file.")
        return

    if args.qa:
        qa_path = Path(args.qa)
        if not qa_path.exists():
            print(f"ERROR: QA file not found: {qa_path}")
            return
        with open(qa_path) as f:
            qa_pairs = json.load(f)
    else:
        qa_pairs = DEFAULT_QA_PAIRS

    results = run_evaluation(qa_pairs, debug=args.debug)
    print_summary(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {out}\n")


if __name__ == "__main__":
    main()