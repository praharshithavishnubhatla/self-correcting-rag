import os
import pickle
import re
from pathlib import Path
import yaml


# ── Load config ───────────────────────────────────────────
def _load_config():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_cfg = _load_config()
_chunking = _cfg["chunking"]
_paths = _cfg["paths"]
_ingestion_cfg = _cfg.get("ingestion", {})
_topics_cfg = _cfg.get("topics", {"enabled": False})

SUPPORTED_EXTENSIONS = tuple(
    _ingestion_cfg.get("supported_extensions", [".txt", ".md"])
)
DEFAULT_TOPIC = _ingestion_cfg.get("default_topic", "general")
OCR_ENGINE = _ingestion_cfg.get("ocr", {}).get("engine", "none")
OCR_LANG = _ingestion_cfg.get("ocr", {}).get("lang", "eng")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def clean_text(text: str) -> str:
    text = re.sub(r'File Name:.*?\n', '', text)
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


# ── Per-format loaders ────────────────────────────────────
# Every loader returns raw extracted text (uncleaned). clean_text()
# is applied uniformly afterwards so formatting quirks don't leak
# into individual loaders.

def _load_txt_or_md(full_path: str) -> str:
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_pdf(full_path: str) -> str:
    """Extract text per page; falls back to OCR for image-only pages."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required for PDF ingestion: pip install pypdf")

    reader = PdfReader(full_path)
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)
        elif OCR_ENGINE != "none":
            # Scanned/image-only page — try OCR via pdf2image + pytesseract.
            # Requires poppler to be installed on the host; skipped silently
            # (with a warning) if unavailable rather than crashing ingestion.
            try:
                from pdf2image import convert_from_path
                page_num = reader.pages.index(page)
                images = convert_from_path(
                    full_path, first_page=page_num + 1, last_page=page_num + 1
                )
                for img in images:
                    text_parts.append(_ocr_image(img))
            except Exception as e:
                print(f"  [WARN] OCR fallback failed for a page in {full_path}: {e}")

    return "\n".join(text_parts)


def _load_docx(full_path: str) -> str:
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx is required for DOCX ingestion: pip install python-docx")

    document = docx.Document(full_path)
    return "\n".join(p.text for p in document.paragraphs)


def _ocr_image(img) -> str:
    """Run OCR on a PIL Image object."""
    if OCR_ENGINE != "tesseract":
        return ""
    import pytesseract
    return pytesseract.image_to_string(img, lang=OCR_LANG)


def _load_image(full_path: str) -> str:
    """Screenshots of notes / Instagram / LinkedIn carousels."""
    if OCR_ENGINE == "none":
        print(f"  [WARN] OCR disabled — skipping image file {full_path}")
        return ""
    from PIL import Image
    with Image.open(full_path) as img:
        return _ocr_image(img)


_LOADERS = {
    ".txt": _load_txt_or_md,
    ".md": _load_txt_or_md,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".png": _load_image,
    ".jpg": _load_image,
    ".jpeg": _load_image,
}

# Heuristic doc_type inference — drives per-type chunking (see config.yaml).
# Override by placing files under data/raw/<topic>/cheatsheets/... etc.
_DOC_TYPE_DIR_HINTS = {
    "cheatsheet": "cheatsheet",
    "cheatsheets": "cheatsheet",
    "social": "social_post",
    "social_post": "social_post",
    "social_posts": "social_post",
}


def _infer_doc_type(rel_path: Path, ext: str) -> str:
    for part in rel_path.parts[:-1]:  # any directory segment in the path
        hint = _DOC_TYPE_DIR_HINTS.get(part.lower())
        if hint:
            return hint
    if ext in IMAGE_EXTENSIONS:
        return "social_post"
    return "notes"


def _infer_topic(rel_path: Path) -> str:
    """First-level subfolder under raw_data is treated as the topic/subject."""
    if not _topics_cfg.get("enabled", False):
        return DEFAULT_TOPIC
    parts = rel_path.parts
    if len(parts) > 1:
        return parts[0]
    return DEFAULT_TOPIC


def load_docs(path: str = None) -> list:
    """
    Walk the raw data directory recursively, loading every supported file.
    Supports .txt, .md, .pdf, .docx, and OCR'd images (.png/.jpg/.jpeg).

    Directory layout convention (topics.enabled=true):
        data/raw/<topic>/<file>
        data/raw/<topic>/cheatsheets/<file>
        data/raw/<topic>/social_posts/<file>
    """
    raw_path = Path(path or _paths["raw_data"])
    docs = []

    if not raw_path.exists():
        return docs

    for full_path in sorted(raw_path.rglob("*")):
        if not full_path.is_file():
            continue

        ext = full_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        loader = _LOADERS.get(ext)
        if loader is None:
            continue

        rel_path = full_path.relative_to(raw_path)

        try:
            raw_text = loader(str(full_path))
        except Exception as e:
            print(f"  [WARN] Failed to load {full_path}: {e}")
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        docs.append({
            "source": str(rel_path),
            "text": cleaned,
            "topic": _infer_topic(rel_path),
            "doc_type": _infer_doc_type(rel_path, ext),
        })

    return docs


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
    min_words: int = None,
) -> list:
    """Split a text string into overlapping word-level chunks."""
    chunk_size = chunk_size or _chunking["chunk_size"]
    overlap = overlap or _chunking["chunk_overlap"]
    min_words = min_words or _chunking["min_chunk_words"]

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        if len(chunk_words) < min_words:
            break

        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap

    return chunks


def _chunk_params_for(doc_type: str) -> dict:
    overrides = _chunking.get("overrides", {})
    return overrides.get(doc_type, {})


def chunk_docs(docs: list) -> list:
    """Chunk all documents (doc-type-aware) and save to disk."""
    all_chunks = []
    chunk_id = 0

    for doc in docs:
        params = _chunk_params_for(doc.get("doc_type", "notes"))
        for chunk in chunk_text(
            doc["text"],
            chunk_size=params.get("chunk_size"),
            overlap=params.get("chunk_overlap"),
            min_words=params.get("min_chunk_words"),
        ):
            all_chunks.append({
                "chunk_id": chunk_id,
                "source": doc["source"],
                "text": chunk,
                "topic": doc.get("topic", DEFAULT_TOPIC),
                "doc_type": doc.get("doc_type", "notes"),
            })
            chunk_id += 1

    os.makedirs("data/processed", exist_ok=True)

    with open(_paths["chunks"], "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    docs = load_docs()
    chunk_docs(docs)
