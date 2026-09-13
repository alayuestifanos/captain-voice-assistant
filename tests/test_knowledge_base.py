from app.config import settings
from app.rag.knowledge_base import Document, chunk_document, load_documents


def test_load_documents_finds_the_shipped_knowledge_base():
    docs = load_documents()
    # Assignment requires 10-50 documents.
    assert 10 <= len(docs) <= 50
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids)), "document ids must be unique"
    for d in docs:
        assert d.title
        assert d.body


def test_frontmatter_parsing_extracts_tags():
    docs = load_documents()
    man_overboard = next(d for d in docs if d.doc_id == "man-overboard")
    assert man_overboard.title == "Man Overboard Procedure"
    assert "emergency" in man_overboard.tags


def test_chunk_document_splits_long_bodies():
    long_body = "\n\n".join([f"Paragraph {i} " + ("x" * 300) for i in range(5)])
    doc = Document(doc_id="d1", title="Doc 1", tags=(), body=long_body, source_path="d1.md")
    chunks = chunk_document(doc, max_chars=400)
    assert len(chunks) > 1
    assert all(c.doc_id == "d1" for c in chunks)
    # chunk ids must be unique and ordered
    assert [c.chunk_id for c in chunks] == [f"d1#{i}" for i in range(len(chunks))]


def test_chunk_document_keeps_short_bodies_as_one_chunk():
    doc = Document(doc_id="d2", title="Doc 2", tags=(), body="Just one short paragraph.", source_path="d2.md")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one short paragraph."
