"""Embedding + FAISS vector store for the RAG pipeline.

Design:
- `Embedder` is a tiny interface so the embedding backend is swappable
  (default: sentence-transformers, running locally, no per-call API cost
  or extra API key -- see README "Tools/APIs used" for the justification).
- `VectorStore` wraps a FAISS `IndexFlatIP` over L2-normalized vectors,
  which is equivalent to cosine similarity search. At 10-50 documents a
  brute-force index is more than fast enough; FAISS is used anyway because
  it is the standard, swap-in-place path to scale to a much larger corpus
  without changing any calling code.
- The index (vectors + chunk metadata) is persisted to disk under
  data/index/ so the app doesn't re-embed the knowledge base on every
  process start.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.config import settings
from app.rag.knowledge_base import Chunk, load_and_chunk


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dim(self) -> int: ...


class SentenceTransformerEmbedder:
    """Local embedding model via sentence-transformers.

    Loaded lazily so importing this module never triggers a model
    download; only actually building/searching the index does.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")

    @property
    def dim(self) -> int:
        model = self._load()
        return int(model.get_sentence_embedding_dimension())


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore:
    """FAISS-backed vector index over knowledge-base chunks."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self._index = None
        self._chunks: list[Chunk] = []

    # -- building -----------------------------------------------------
    def build(self, chunks: list[Chunk] | None = None) -> "VectorStore":
        import faiss

        chunks = chunks if chunks is not None else load_and_chunk()
        if not chunks:
            raise ValueError("No knowledge-base documents found to index.")

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._index = index
        self._chunks = chunks
        return self

    # -- persistence ----------------------------------------------------
    def save(self, directory: Path | None = None) -> None:
        import faiss

        directory = directory or settings.index_dir
        directory.mkdir(parents=True, exist_ok=True)
        if self._index is None:
            raise RuntimeError("Index has not been built yet.")

        faiss.write_index(self._index, str(directory / "index.faiss"))
        meta = {
            "embedding_model": self.embedder.model_name if hasattr(self.embedder, "model_name") else None,
            "chunks": [c.__dict__ for c in self._chunks],
        }
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, directory: Path | None = None) -> "VectorStore":
        import faiss

        directory = directory or settings.index_dir
        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.json"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No index found at {directory}. Run `python scripts/build_index.py` first."
            )

        self._index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                title=c["title"],
                tags=tuple(c["tags"]),
                text=c["text"],
                source_path=c["source_path"],
            )
            for c in meta["chunks"]
        ]
        return self

    def load_or_build(self, directory: Path | None = None) -> "VectorStore":
        try:
            return self.load(directory)
        except FileNotFoundError:
            self.build()
            self.save(directory)
            return self

    # -- querying -------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        if self._index is None:
            raise RuntimeError("Index has not been built/loaded yet.")
        top_k = top_k or settings.retrieval_top_k
        top_k = min(top_k, len(self._chunks))

        query_vec = self.embedder.embed([query])
        scores, indices = self._index.search(query_vec, top_k)

        results: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(ScoredChunk(chunk=self._chunks[int(idx)], score=float(score)))
        return results

    def __len__(self) -> int:
        return len(self._chunks)


class Retriever:
    """High-level RAG retrieval API used by the pipeline."""

    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore().load_or_build()

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        return self.store.search(query, top_k=top_k)

    def retrieve_relevant(
        self, query: str, top_k: int | None = None, threshold: float | None = None
    ) -> list[ScoredChunk]:
        """Retrieve chunks and drop anything below the relevance threshold,
        so the generator only ever sees context worth grounding on."""
        threshold = settings.relevance_threshold if threshold is None else threshold
        results = self.retrieve(query, top_k=top_k)
        return [r for r in results if r.score >= threshold]
