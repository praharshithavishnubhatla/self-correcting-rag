"""
tests/test_api_security.py
---------------------------
Regression tests for two things fixed in api/main.py:

1. Path-traversal: `topic`, `doc_type`, and the uploaded filename are all
   attacker-controlled strings that get joined onto RAW_DATA_DIR. These
   tests assert that traversal attempts ('../', absolute paths, bare '..')
   can never resolve to a path outside data/raw.
2. API-key auth: /ingest, /ingest/batch, and /eval must reject requests
   without a valid X-API-Key when API_KEY is configured, and must allow
   through when it isn't (local/dev mode).

Does not require built indexes or a GROQ_API_KEY — these test the request
layer, not the RAG pipeline itself.
"""

import importlib

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.fixture()
def main_module(monkeypatch):
    """Reload api.main fresh so API_KEY changes in each test take effect."""
    import api.main as main
    importlib.reload(main)
    return main


# ── Path traversal: _sanitize_component ─────────────────────

@pytest.mark.parametrize("malicious", ["..", "."])
def test_sanitize_component_rejects_bare_dotdot(main_module, malicious):
    with pytest.raises(HTTPException) as exc:
        main_module._sanitize_component(malicious, "topic")
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "malicious,expected_basename",
    [("../../etc", "etc"), ("../secret", "secret"), ("....//etc", "etc")],
)
def test_sanitize_component_collapses_traversal_to_safe_basename(main_module, malicious, expected_basename):
    # These don't raise — they collapse to a single safe path component with
    # no separators or ".." left in it, so joining onto RAW_DATA_DIR can
    # never escape it.
    result = main_module._sanitize_component(malicious, "topic")
    assert result == expected_basename
    assert "/" not in result and ".." not in result


def test_sanitize_component_allows_normal_values(main_module):
    assert main_module._sanitize_component("system design", "topic") == "system design"
    assert main_module._sanitize_component("cheatsheet", "doc_type") == "cheatsheet"


def test_sanitize_component_strips_embedded_traversal(main_module):
    # "../../etc/passwd" as a topic should collapse to just the basename,
    # never let a caller climb out of RAW_DATA_DIR.
    assert main_module._sanitize_component("../../etc/passwd", "topic") == "passwd"


# ── Path traversal: _safe_upload_path ────────────────────────

def test_safe_upload_path_rejects_dotdot_filename(main_module, tmp_path):
    with pytest.raises(HTTPException):
        main_module._safe_upload_path(tmp_path, "..")


def test_safe_upload_path_strips_directory_components(main_module, tmp_path):
    result = main_module._safe_upload_path(tmp_path, "../../../etc/cron.d/evil")
    assert result.parent == tmp_path.resolve()
    assert result.name == "evil"


def test_safe_upload_path_stays_inside_target_dir(main_module, tmp_path):
    result = main_module._safe_upload_path(tmp_path, "notes.txt")
    assert tmp_path.resolve() in result.parents


# ── API-key auth ──────────────────────────────────────────────

def test_ingest_rejected_without_key_when_configured(main_module, monkeypatch):
    monkeypatch.setattr(main_module, "API_KEY", "test-secret-key")
    client = TestClient(main_module.app)
    resp = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
        data={"topic": "general"},
    )
    assert resp.status_code == 401


def test_ingest_rejected_with_wrong_key(main_module, monkeypatch):
    monkeypatch.setattr(main_module, "API_KEY", "test-secret-key")
    client = TestClient(main_module.app)
    resp = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
        data={"topic": "general"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_health_and_topics_do_not_require_key(main_module, monkeypatch):
    monkeypatch.setattr(main_module, "API_KEY", "test-secret-key")
    client = TestClient(main_module.app)
    assert client.get("/health").status_code == 200
    assert client.get("/topics").status_code == 200


def test_auth_is_noop_when_api_key_unset(main_module, monkeypatch):
    # No API_KEY configured -> require_api_key should be a pass-through.
    monkeypatch.setattr(main_module, "API_KEY", None)
    # Should not raise even with no header supplied.
    main_module.require_api_key(provided=None)
