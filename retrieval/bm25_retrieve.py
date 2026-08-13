import pickle
from pathlib import Path
import yaml


# ── Load config ───────────────────────────────────────────
def _load_config():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_cfg = _load_config()
_paths = _cfg["paths"]

_chunks = None
_bm25 = None


def load():
    global _chunks, _bm25

    if _chunks is None:
        with open(_paths["chunks_with_embeddings"], "rb") as f:
            _chunks = pickle.load(f)

    if _bm25 is None:
        with open(_paths["bm25_index"], "rb") as f:
            _bm25 = pickle.load(f)


def invalidate():
    """Force the next load() to re-read chunks/bm25 index from disk. See
    retrieval/retrieve.py::invalidate() for why this matters."""
    global _chunks, _bm25
    _chunks = None
    _bm25 = None


def bm25_search(query: str, k: int = 5, topic: str = None):
    load()

    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    fetch_k = k
    overfetch = _cfg.get("topics", {}).get("filter_overfetch_multiplier", 4)
    if topic:
        fetch_k = min(k * overfetch, len(_chunks))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked[:fetch_k]:
        chunk = _chunks[i]
        if topic and chunk.get("topic") != topic:
            continue
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "topic": chunk.get("topic"),
            "score": float(scores[i]),
        })
        if len(results) >= k:
            break

    return results