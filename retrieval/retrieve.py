import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import yaml


# ── Load config ───────────────────────────────────────────
def _load_config():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_cfg = _load_config()
_paths = _cfg["paths"]
_query_model_name = _cfg["embedding"]["query_model"]

_index = None
_chunks = None
_model = None


def load():
    global _index, _chunks, _model

    if _index is None:
        _index = faiss.read_index(_paths["faiss_index"])

    if _chunks is None:
        with open(_paths["chunks_with_embeddings"], "rb") as f:
            _chunks = pickle.load(f)

    if _model is None:
        _model = SentenceTransformer(_query_model_name)


def invalidate():
    """
    Force the next load() to re-read faiss.index/chunks from disk. Call this
    after a reindex — without it, a long-running server process keeps
    serving the FAISS index/chunks it had in memory at first request, so
    newly-ingested documents silently wouldn't show up in search results
    until the process restarted. The embedding model itself doesn't need
    reloading, only the data.
    """
    global _index, _chunks
    _index = None
    _chunks = None


def faiss_search(query: str, k: int = 8, topic: str = None):
    load()

    fetch_k = k
    overfetch = _cfg.get("topics", {}).get("filter_overfetch_multiplier", 4)
    if topic:
        fetch_k = min(k * overfetch, len(_chunks))

    q = _model.encode([query]).astype("float32")
    distances, indices = _index.search(q, fetch_k)

    results = []
    result_distances = []
    for j, i in enumerate(indices[0]):
        chunk = _chunks[i]
        if topic and chunk.get("topic") != topic:
            continue
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "topic": chunk.get("topic"),
            "distance": float(distances[0][j]),
        })
        result_distances.append(distances[0][j])
        if len(results) >= k:
            break

    return results, (result_distances or distances[0])


def get_topic_chunks(topic: str = None) -> list:
    """
    Fetch ALL chunks for a topic (or every chunk if topic is None), in
    original ingestion order (grouped by source file, in the order they
    were written).

    Used by broad-coverage modes (revision notes, practice questions) where
    the goal is "cover everything important" rather than "find the chunks
    most similar to this question" — similarity search against a vague
    prompt like "give me quick notes" doesn't target any single point in
    embedding space, so top-k retrieval would return an arbitrary slice
    instead of full topic coverage.
    """
    load()

    chunks = _chunks
    if topic:
        chunks = [c for c in chunks if c.get("topic") == topic]

    return sorted(chunks, key=lambda c: (c["source"], c.get("chunk_id", 0)))