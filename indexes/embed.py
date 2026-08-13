import pickle
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
_embed_model_name = _cfg["embedding"]["index_model"]

# Cached across calls within the same process. Loading SentenceTransformer
# weights from disk takes several seconds — reloading it from scratch on
# every single /ingest request (as this used to do) was the single biggest
# hidden cost in ingestion latency, especially since it happened silently
# with no progress output between the "Embedding N chunks..." print and the
# tqdm bar actually starting.
_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {_embed_model_name}")
        _model = SentenceTransformer(_embed_model_name)
    return _model


def _load_embedding_cache() -> dict:
    """
    (source, text) -> embedding, built from whatever was embedded last run.
    Lets re-ingestion reuse embeddings for chunks that haven't changed
    instead of re-encoding the entire corpus every time a single new file
    is uploaded — the full-reindex-per-upload behavior is otherwise the
    dominant ingestion cost once a topic has more than a handful of files.
    """
    path = Path(_paths["chunks_with_embeddings"])
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            prev_chunks = pickle.load(f)
        return {(c["source"], c["text"]): c["embedding"] for c in prev_chunks}
    except Exception:
        # Cache file missing/corrupt — fall back to embedding everything.
        return {}


def main():
    with open(_paths["chunks"], "rb") as f:
        chunks = pickle.load(f)

    cache = _load_embedding_cache()
    to_embed = [c for c in chunks if (c["source"], c["text"]) not in cache]

    print(
        f"{len(chunks)} total chunks — {len(chunks) - len(to_embed)} unchanged "
        f"(reused cached embeddings), {len(to_embed)} new/changed to embed "
        f"with model: {_embed_model_name}"
    )

    if to_embed:
        model = _get_model()
        texts = [c["text"] for c in to_embed]
        embeddings = model.encode(texts, show_progress_bar=True)
        for c, emb in zip(to_embed, embeddings):
            cache[(c["source"], c["text"])] = emb

    for chunk in chunks:
        chunk["embedding"] = cache[(chunk["source"], chunk["text"])]

    with open(_paths["chunks_with_embeddings"], "wb") as f:
        pickle.dump(chunks, f)

    print("Embeddings created")


if __name__ == "__main__":
    main()