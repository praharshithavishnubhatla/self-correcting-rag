import pickle
import numpy as np
import faiss
from pathlib import Path
import yaml


# ── Load config ───────────────────────────────────────────
def _load_config():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_cfg = _load_config()
_paths = _cfg["paths"]
_faiss_cfg = _cfg["faiss"]


def main():
    with open(_paths["chunks_with_embeddings"], "rb") as f:
        chunks = pickle.load(f)

    embeddings = np.array(
        [c["embedding"] for c in chunks],
        dtype="float32"
    )

    dimension = embeddings.shape[1]
    print(f"Building FAISS HNSW index (dim={dimension}, M={_faiss_cfg['hnsw_m']})")

    index = faiss.IndexHNSWFlat(dimension, _faiss_cfg["hnsw_m"])
    index.hnsw.efConstruction = _faiss_cfg["ef_construction"]
    index.hnsw.efSearch = _faiss_cfg["ef_search"]
    index.add(embeddings)

    faiss.write_index(index, _paths["faiss_index"])
    print(f"FAISS index saved to {_paths['faiss_index']}")


if __name__ == "__main__":
    main()