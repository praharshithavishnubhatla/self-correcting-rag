"""
retrieval/cross_encoder_reranker.py
------------------------------------
Drop-in cross-encoder reranker.
Used by rag/rag.py when config reranker.use_cross_encoder = true.
Falls back gracefully if the model can't load.
"""

import logging
from typing import List, Dict
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def _load_config():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class CrossEncoderReranker:
    def __init__(self):
        cfg = _load_config()
        model_name = cfg["reranker"]["cross_encoder_model"]

        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self._available = True
            logger.info(f"CrossEncoder loaded: {model_name}")
        except Exception as e:
            self._available = False
            logger.warning(
                f"CrossEncoder could not be loaded ({e}). "
                "Falling back to LLM reranker."
            )

    @property
    def available(self) -> bool:
        return self._available

    def rerank(self, query: str, chunks: List[Dict], top_n: int = 3) -> List[Dict]:
        """
        Score each (query, chunk) pair and return top_n by score.
        Each chunk gets a 'rerank_score' key added.
        """
        if not self._available or not chunks:
            return chunks[:top_n]

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        scored = [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in zip(chunks, scores)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.debug(
            "Cross-encoder top scores: "
            + str([round(c["rerank_score"], 3) for c in scored[:top_n]])
        )
        return scored[:top_n]


# Singleton — loaded once per process
_reranker_instance = None

def get_reranker() -> CrossEncoderReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance