"""Build (or rebuild) the FAISS index from data/knowledge_base/.

Run this whenever the knowledge base documents change:
    python scripts/build_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.rag.knowledge_base import load_and_chunk
from app.rag.retriever import VectorStore


def main() -> int:
    chunks = load_and_chunk()
    if not chunks:
        print(f"No documents found in {settings.knowledge_base_dir}", file=sys.stderr)
        return 1

    print(f"Loaded {len(chunks)} chunks from {settings.knowledge_base_dir}")
    print(f"Embedding with model: {settings.embedding_model} (first run downloads the model, ~90MB)")

    store = VectorStore()
    store.build(chunks)
    store.save()

    print(f"Index built and saved to {settings.index_dir} ({len(store)} vectors).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
