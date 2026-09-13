"""Retriever tests use a small, hand-crafted fake embedder instead of the
real sentence-transformers model, so they run fast/offline and the expected
ranking is fully deterministic and easy to reason about.
"""
import numpy as np

from app.rag.knowledge_base import Chunk
from app.rag.retriever import VectorStore

# A tiny "vocabulary -> direction" embedding: each text is embedded as the
# (normalized) sum of one-hot vectors for the keywords it contains, so
# cosine similarity behaves like a controllable keyword-overlap score.
VOCAB = ["fire", "anchor", "overboard", "pilot", "cargo"]


class FakeEmbedder:
    model_name = "fake-keyword-embedder"

    @property
    def dim(self) -> int:
        return len(VOCAB)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), len(VOCAB)), dtype="float32")
        for i, text in enumerate(texts):
            lower = text.lower()
            for j, word in enumerate(VOCAB):
                if word in lower:
                    vectors[i, j] = 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


def make_chunks():
    return [
        Chunk("fire#0", "fire", "Fire Emergency Response", (), "Fire fire fire alarm procedure.", "fire.md"),
        Chunk("anchor#0", "anchor", "Anchoring Procedures", (), "Anchor chain scope depth.", "anchor.md"),
        Chunk("mob#0", "mob", "Man Overboard Procedure", (), "overboard overboard lifebuoy rescue.", "mob.md"),
        Chunk("pilot#0", "pilot", "Pilot Boarding", (), "pilot ladder boarding arrangement.", "pilot.md"),
    ]


def build_store():
    store = VectorStore(embedder=FakeEmbedder())
    store.build(make_chunks())
    return store


def test_search_returns_most_relevant_chunk_first():
    store = build_store()
    results = store.search("What do we do if someone goes overboard?", top_k=4)
    assert results[0].chunk.doc_id == "mob"
    assert results[0].score > results[1].score


def test_search_respects_top_k():
    store = build_store()
    results = store.search("fire", top_k=2)
    assert len(results) == 2


def test_search_is_empty_for_unrelated_query_with_no_keyword_overlap():
    store = build_store()
    results = store.search("cargo stowage plan", top_k=4)
    # No chunk mentions "cargo" (not part of these 4 docs), so every score is 0.
    assert all(r.score == 0.0 for r in results)


def test_retriever_relevance_threshold_filters_low_scores(monkeypatch):
    from app.rag.retriever import Retriever

    retriever = Retriever(store=build_store())
    relevant = retriever.retrieve_relevant("overboard emergency", threshold=0.5)
    assert all(sc.score >= 0.5 for sc in relevant)
    assert any(sc.chunk.doc_id == "mob" for sc in relevant)

    none_relevant = retriever.retrieve_relevant("completely unrelated topic", threshold=0.5)
    assert none_relevant == []
