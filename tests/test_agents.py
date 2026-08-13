"""
tests/test_agents.py
--------------------
Tests for guardrail logic and prompt templates.
No API key or LLM calls needed.

Run with:  pytest tests/test_agents.py -v
"""

import pytest
from pathlib import Path


# ── Deterministic guardrail (token overlap) ───────────────
# This mirrors the fast-fail logic added to guardrail_check() in rag/rag.py

def token_overlap_passes(question: str, context: str, min_overlap: int = 3) -> bool:
    q_tokens = set(question.lower().split())
    c_tokens = set(context.lower().split())
    return len(q_tokens & c_tokens) >= min_overlap


class TestDeterministicGuardrail:
    def test_relevant_context_passes(self):
        q = "How does horizontal scaling work?"
        ctx = "Horizontal scaling means adding more servers into your pool of resources."
        assert token_overlap_passes(q, ctx, min_overlap=2) is True

    def test_completely_unrelated_fails(self):
        q = "How does authentication work?"
        ctx = "The weather in Paris is sunny and warm in summer."
        assert token_overlap_passes(q, ctx) is False

    def test_overlap_exactly_at_threshold(self):
        q = "database caching load strategy"
        ctx = "database caching improves system load handling"
        assert token_overlap_passes(q, ctx, min_overlap=3) is True

    def test_overlap_below_threshold_fails(self):
        q = "database connection pooling explained"
        ctx = "the system uses a cloud provider"
        assert token_overlap_passes(q, ctx, min_overlap=3) is False

    def test_scalability_query_against_your_data(self):
        q = "explain scalability"
        ctx = "Scalability refers to a system ability to handle growing workloads without performance degradation"
        assert token_overlap_passes(q, ctx, min_overlap=1) is True

    def test_load_balancer_query(self):
        q = "how does a load balancer distribute traffic"
        ctx = "A load balancer evenly distributes incoming traffic among web servers that are defined in a load balanced set"
        assert token_overlap_passes(q, ctx) is True


# ── Prompt template tests ─────────────────────────────────
# Validates all 5 prompt files exist, have correct placeholders, and render.

PROMPTS_DIR = Path("prompts")

EXPECTED_PROMPTS = {
    "query_rewrite": ["{question}"],
    "reranker":      ["{question}", "{chunks}", "{top_n}"],
    "guardrail":     ["{question}", "{context}"],
    "answer":        ["{question}", "{context}"],
    "evaluator":     ["{question}", "{context}", "{answer}"],
}


@pytest.mark.parametrize("prompt_name,placeholders", EXPECTED_PROMPTS.items())
class TestPromptTemplates:
    def test_file_exists(self, prompt_name, placeholders):
        path = PROMPTS_DIR / f"{prompt_name}.txt"
        assert path.exists(), f"Missing prompt file: {path}"

    def test_placeholders_present(self, prompt_name, placeholders):
        path = PROMPTS_DIR / f"{prompt_name}.txt"
        if not path.exists():
            pytest.skip(f"{path} not found")
        content = path.read_text()
        for ph in placeholders:
            assert ph in content, f"Placeholder '{ph}' missing from {prompt_name}.txt"

    def test_renders_without_error(self, prompt_name, placeholders):
        path = PROMPTS_DIR / f"{prompt_name}.txt"
        if not path.exists():
            pytest.skip(f"{path} not found")
        template = path.read_text()
        dummy = {ph.strip("{}"): f"TEST_{ph.strip('{}').upper()}" for ph in placeholders}
        rendered = template.format(**dummy)
        for val in dummy.values():
            assert val in rendered

    def test_not_empty(self, prompt_name, placeholders):
        path = PROMPTS_DIR / f"{prompt_name}.txt"
        if not path.exists():
            pytest.skip(f"{path} not found")
        assert len(path.read_text().strip()) > 20


# ── Evaluator-specific: PASS/FAIL keywords ────────────────

class TestEvaluatorPrompt:
    def setup_method(self):
        path = PROMPTS_DIR / "evaluator.txt"
        if not path.exists():
            pytest.skip("prompts/evaluator.txt not found")
        self.template = path.read_text()

    def test_instructs_pass_or_fail(self):
        assert "PASS" in self.template and "FAIL" in self.template

    def test_does_not_ask_for_explanation(self):
        # The evaluator should return a single word, not an essay
        assert "explain" not in self.template.lower() or "do not explain" in self.template.lower()


# ── Guardrail-specific: YES/NO keywords ───────────────────

class TestGuardrailPrompt:
    def setup_method(self):
        path = PROMPTS_DIR / "guardrail.txt"
        if not path.exists():
            pytest.skip("prompts/guardrail.txt not found")
        self.template = path.read_text()

    def test_instructs_yes_or_no(self):
        assert "YES" in self.template and "NO" in self.template