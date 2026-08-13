"""
tests/test_pipeline.py
----------------------
End-to-end smoke tests for the full RAG pipeline.

Requires:
  - Built indexes (run `python main.py` once first)
  - GROQ_API_KEY in .env

Skips gracefully if indexes or API key are missing.
Run with:  pytest tests/test_pipeline.py -v -s
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def indexes_ready():
    required = [
        "data/processed/faiss.index",
        "data/processed/bm25.pkl",
        "data/processed/chunks_with_embeddings.pkl",
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        pytest.skip(f"Indexes not built yet. Missing: {missing}. Run `python main.py` first.")
    return True


@pytest.fixture(scope="module")
def api_key_set():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set in .env")
    return True


@pytest.fixture(scope="module")
def pipeline(indexes_ready, api_key_set):
    from rag.rag import run_pipeline
    return run_pipeline


# ── Ingestion tests (no API key needed) ───────────────────

class TestIngestion:
    def test_load_docs_finds_your_files(self):
        from indexes.ingestion import load_docs
        docs = load_docs("data/raw")
        assert len(docs) >= 4
        sources = [d["source"] for d in docs]
        assert "design.md" in sources
        assert "system.md" in sources

    def test_chunk_text_produces_chunks(self):
        from indexes.ingestion import chunk_text
        long_text = "word " * 400
        chunks = chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunk_overlap(self):
        from indexes.ingestion import chunk_text
        text = "word " * 400
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # Each chunk should be ~100 words
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_min_chunk_words_filter(self):
        from indexes.ingestion import chunk_text
        short_text = "only ten words in this short text here now"
        chunks = chunk_text(short_text, chunk_size=300, overlap=60, min_words=50)
        assert chunks == []  # too short, filtered out

    def test_clean_text_removes_filename_header(self):
        from indexes.ingestion import clean_text
        raw = "File Name: design_url_shortener.md\nActual content here."
        cleaned = clean_text(raw)
        assert "File Name:" not in cleaned
        assert "Actual content here" in cleaned


# ── Config loading ────────────────────────────────────────

class TestConfig:
    def test_config_loads(self):
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "llm" in cfg
        assert "retrieval" in cfg
        assert "paths" in cfg
        assert "chunking" in cfg

    def test_config_model_name_set(self):
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["llm"]["model"] == "llama-3.3-70b-versatile"

    def test_config_paths_match_actual_files(self):
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        # raw data should always exist
        assert Path(cfg["paths"]["raw_data"]).exists()


# ── Pipeline integration tests ────────────────────────────

class TestPipelineIntegration:
    """
    Queries that match your actual data/raw documents:
    design.md / design.txt / system.md / system.txt
    """

    def test_scalability_query(self, pipeline):
        result = pipeline("what is scalability?")
        assert "answer" in result
        assert len(result["answer"]) > 20
        assert result["guardrail_passed"] is True
        assert result["evaluator_verdict"] == "PASS"

    def test_load_balancer_query(self, pipeline):
        result = pipeline("explain load balancing")
        assert "answer" in result
        assert result["guardrail_passed"] is True

    def test_database_query(self, pipeline):
        result = pipeline("what is the difference between SQL and NoSQL?")
        assert "answer" in result
        assert len(result["sources"]) > 0

    def test_caching_query(self, pipeline):
        result = pipeline("how does caching improve performance?")
        assert result["guardrail_passed"] is True
        assert "sources" in result

    def test_sources_are_your_files(self, pipeline):
        result = pipeline("explain horizontal scaling")
        known_sources = {"design.md", "design.txt", "system.md", "system.txt"}
        for s in result["sources"]:
            assert s in known_sources, f"Unexpected source: {s}"

    def test_irrelevant_query_handled(self, pipeline):
        result = pipeline("what is the boiling point of tungsten on Jupiter?")
        # Should not crash; either guardrail blocks it or evaluator fails
        assert "answer" in result
        assert isinstance(result["sources"], list)

    def test_result_has_all_keys(self, pipeline):
        result = pipeline("explain CDN")
        expected_keys = {"answer", "sources", "guardrail_passed", "evaluator_verdict", "debug_info"}
        assert expected_keys.issubset(result.keys())

    def test_debug_mode_populates_debug_info(self, pipeline):
        result = pipeline("what is vertical scaling?", debug=True)
        assert isinstance(result["debug_info"], dict)
        assert "best_faiss_distance" in result["debug_info"]
        assert "guardrail" in result["debug_info"]