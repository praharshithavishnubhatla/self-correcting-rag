import os
import argparse

from indexes.ingestion import load_docs, chunk_docs
from indexes.embed import main as embed_main
from indexes.index import main as index_main
from indexes.bm25_index import main as bm25_main

from rag.rag import ask


CHUNKS_PATH = "data/processed/chunks.pkl"
EMBED_PATH = "data/processed/chunks_with_embeddings.pkl"
FAISS_PATH = "data/processed/faiss.index"
BM25_PATH = "data/processed/bm25.pkl"


def ensure_ingestion(debug=False):
    if os.path.exists(CHUNKS_PATH):
        if debug:
            print("[BOOTSTRAP] Chunks already exist ✓")
        return

    print("[BOOTSTRAP] Running ingestion...")

    docs = load_docs()

    if not docs:
        print("No documents found in data/raw/")
        exit()

    chunk_docs(docs)

    if debug:
        print("[BOOTSTRAP] Document ingestion completed.")


def ensure_embeddings(debug=False):
    if os.path.exists(EMBED_PATH):
        if debug:
            print("[BOOTSTRAP] Embeddings already exist ✓")
        return

    print("[BOOTSTRAP] Generating embeddings...")
    embed_main()

    if debug:
        print("[BOOTSTRAP] Embeddings created.")


def ensure_faiss(debug=False):
    if os.path.exists(FAISS_PATH):
        if debug:
            print("[BOOTSTRAP] FAISS index already exists ✓")
        return

    print("[BOOTSTRAP] Building FAISS index...")
    index_main()

    if debug:
        print("[BOOTSTRAP] FAISS index built.")


def ensure_bm25(debug=False):
    if os.path.exists(BM25_PATH):
        if debug:
            print("[BOOTSTRAP] BM25 index already exists ✓")
        return

    print("[BOOTSTRAP] Building BM25 index...")
    bm25_main()

    if debug:
        print("[BOOTSTRAP] BM25 index built.")


def bootstrap_pipeline(debug=False):

    print("\n=== Bootstrapping RAG pipeline ===\n")

    ensure_ingestion(debug)
    ensure_embeddings(debug)
    ensure_faiss(debug)
    ensure_bm25(debug)

    print("\n=== Pipeline ready ===\n")


def interactive_chat(debug=False):

    print("Ask questions. Press Ctrl+C to exit.\n")

    while True:
        try:
            q = input("Ask: ").strip()

            if not q:
                continue

            print()

            answer = ask(q, debug=debug)

            print("\n=== FINAL ANSWER ===\n")
            print(answer)
            print("\n====================\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        help="Ask a single question and exit"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debugging logs"
    )

    args = parser.parse_args()

    bootstrap_pipeline(debug=args.debug)

    if args.query:

        print("\n=== QUERY MODE ===\n")

        answer = ask(args.query, debug=args.debug)

        print(answer)

    else:

        interactive_chat(debug=args.debug)


if __name__ == "__main__":
    main()