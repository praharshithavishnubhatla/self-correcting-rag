"""
tests/test_retrieval.py
-----------------------
Unit tests for FAISS + BM25 retrieval and the cross-encoder reranker.

Run with:  pytest tests/test_retrieval.py -v
No API key required.
"""

import pytest
import numpy as np

# ── Sample chunks matching your actual schema ─────────────
SAMPLE_CHUNKS = [
    {"chunk_id": 0, "text": "Scalability refers to a system's ability to handle growing workloads without performance degradation.", "source": "system.md"},
    {"chunk_id": 1, "text": "Horizontal scaling means adding more servers into your pool of resources.", "source": "system.txt"},
    {"chunk_id": 2, "text": "A load balancer evenly distributes incoming traffic among web servers.", "source": "design.txt"},
    {"chunk_id": 3, "text": "Non-relational databases are useful for flexible schemas and massive scale. Popular ones include DynamoDB and Cassandra.", "source": "design.txt"},
    {"chunk_id": 4, "text": "Caching stores frequently accessed data in memory to reduce database calls. Tools: Redis, Memcached.", "source": "system.txt"},
    {"chunk_id": 5, "text": "A CDN is a network of geographically dispersed servers used to deliver static content.", "source": "design.txt"},
    {"chunk_id": 6, "text": "CAP Theorem: a distributed system can only guarantee two of Consistency, Availability, Partition Tolerance.", "source": "system.txt"},
]


# ── BM25 ──────────────────────────────────────────────────

class TestBM25:
    def setup_method(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank_bm25 not installed")

        corpus = [c["text"].lower().split() for c in SAMPLE_CHUNKS]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(corpus)

    def test_returns_score_per_chunk(self):
        scores = self.bm25.get_scores("scalability".split())
        assert len(scores) == len(SAMPLE_CHUNKS)

    def test_scalability_query_top_result(self):
        scores = self.bm25.get_scores("scalability workloads".split())
        top_idx = int(np.argmax(scores))
        assert "scalability" in SAMPLE_CHUNKS[top_idx]["text"].lower()

    def test_caching_query_top_result(self):
        scores = self.bm25.get_scores("caching redis memory".split())
        top_idx = int(np.argmax(scores))
        assert "cach" in SAMPLE_CHUNKS[top_idx]["text"].lower()

    def test_unrelated_query_low_score(self):
        scores = self.bm25.get_scores("quantum photon laser".split())
        assert max(scores) < 2.0

    def test_all_zeros_for_empty_query(self):
        scores = self.bm25.get_scores([])
        assert all(s == 0 for s in scores)


# ── FAISS ─────────────────────────────────────────────────

class TestFAISS:
    def setup_method(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("faiss-cpu or sentence-transformers not installed")

        import faiss
        from sentence_transformers import SentenceTransformer

        # Use your actual query model from config
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c["text"] for c in SAMPLE_CHUNKS]
        embeddings = self.model.encode(texts, convert_to_numpy=True).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def test_index_contains_all_chunks(self):
        assert self.index.ntotal == len(SAMPLE_CHUNKS)

    def test_search_returns_k_results(self):
        q = self.model.encode(["load balancing"]).astype("float32")
        distances, indices = self.index.search(q, k=3)
        assert len(indices[0]) == 3

    def test_load_balancer_query(self):
        q = self.model.encode(["how does load balancing work?"]).astype("float32")
        _, indices = self.index.search(q, k=1)
        result = SAMPLE_CHUNKS[indices[0][0]]
        assert "load balancer" in result["text"].lower()

    def test_distances_non_negative(self):
        q = self.model.encode(["caching strategy"]).astype("float32")
        distances, _ = self.index.search(q, k=4)
        assert all(d >= 0 for d in distances[0])

    def test_top1_distance_less_than_top4(self):
        q = self.model.encode(["database NoSQL"]).astype("float32")
        distances, _ = self.index.search(q, k=4)
        assert distances[0][0] <= distances[0][-1]


# ── RRF Fusion ────────────────────────────────────────────

class TestRRF:
    """Test the Reciprocal Rank Fusion logic used in hybrid_retrieve()."""

    def _rrf_merge(self, faiss_results, bm25_results, k=60):
        rrf_scores = {}
        doc_map = {}
        for rank, r in enumerate(faiss_results):
            key = (r["source"], r["text"][:80])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
            doc_map[key] = r
        for rank, r in enumerate(bm25_results):
            key = (r["source"], r["text"][:80])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
            doc_map[key] = r
        return sorted(doc_map.values(), key=lambda r: rrf_scores[(r["source"], r["text"][:80])], reverse=True)

    def test_both_lists_merged(self):
        faiss_res = SAMPLE_CHUNKS[:3]
        bm25_res = SAMPLE_CHUNKS[3:6]
        merged = self._rrf_merge(faiss_res, bm25_res)
        assert len(merged) == 6

    def test_shared_doc_ranked_higher(self):
        """A chunk appearing in both FAISS and BM25 should score higher via RRF."""
        shared = SAMPLE_CHUNKS[0]
        faiss_res = [shared, SAMPLE_CHUNKS[1]]
        bm25_res = [shared, SAMPLE_CHUNKS[2]]
        merged = self._rrf_merge(faiss_res, bm25_res)
        assert merged[0]["text"] == shared["text"]

    def test_empty_lists_returns_empty(self):
        merged = self._rrf_merge([], [])
        assert merged == []


# ── Cross-Encoder Reranker ────────────────────────────────

class TestCrossEncoderReranker:
    def setup_method(self):
        try:
            from retrieval.cross_encoder_reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker()
        except Exception:
            pytest.skip("CrossEncoderReranker could not be initialised")

    def test_returns_top_n(self):
        result = self.reranker.rerank("scalability", SAMPLE_CHUNKS, top_n=3)
        assert len(result) == 3

    def test_rerank_score_present(self):
        result = self.reranker.rerank("caching redis", SAMPLE_CHUNKS, top_n=2)
        for chunk in result:
            assert "rerank_score" in chunk

    def test_scores_are_sorted_descending(self):
        result = self.reranker.rerank("load balancer traffic", SAMPLE_CHUNKS, top_n=5)
        scores = [r["rerank_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_handles_empty_chunks(self):
        result = self.reranker.rerank("anything", [], top_n=5)
        assert result == []

    def test_relevant_chunk_ranked_first(self):
        result = self.reranker.rerank("CDN static content delivery", SAMPLE_CHUNKS, top_n=7)
        assert "cdn" in result[0]["text"].lower() or "static" in result[0]["text"].lower()