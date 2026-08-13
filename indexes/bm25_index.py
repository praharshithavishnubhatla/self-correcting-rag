import os
import pickle
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi  # type: ignore
import yaml


# ── Load config ───────────────────────────────────────────
def _load_config():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_cfg = _load_config()
_paths = _cfg["paths"]


def main(force: bool = False):
    if os.path.exists(_paths["bm25_index"]) and not force:
        print("BM25 index already exists. Skipping rebuild.")
        print("Run with --force to rebuild.")
        return

    if not os.path.exists(_paths["chunks_with_embeddings"]):
        print("Embeddings not found. Run embed.py first.")
        return

    with open(_paths["chunks_with_embeddings"], "rb") as f:
        chunks = pickle.load(f)

    print(f"Building BM25 index over {len(chunks)} chunks...")

    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(_paths["bm25_index"], "wb") as f:
        pickle.dump(bm25, f)

    print(f"BM25 index saved to {_paths['bm25_index']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 keyword index.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists.")
    args = parser.parse_args()
    main(force=args.force)