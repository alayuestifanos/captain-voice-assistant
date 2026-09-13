"""Loading and chunking of the knowledge base documents.

Documents live as plain Markdown files under data/knowledge_base/, each
with a small YAML-ish frontmatter block (id, title, tags). We parse the
frontmatter by hand with no external YAML dependency required, since the
format is fixed and simple -- one less thing that can break the setup.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import settings

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    tags: tuple[str, ...]
    body: str
    source_path: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    tags: tuple[str, ...]
    text: str
    source_path: str


def _parse_frontmatter(raw: str, fallback_id: str) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {"id": fallback_id, "title": fallback_id, "tags": []}, raw.strip()

    header, body = match.group(1), match.group(2)
    meta: dict = {"id": fallback_id, "title": fallback_id, "tags": []}
    for line in header.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "tags":
            # tags: [a, b, c]
            value = value.strip("[]")
            meta["tags"] = [t.strip() for t in value.split(",") if t.strip()]
        else:
            meta[key] = value
    return meta, body.strip()


def load_documents(directory: Path | None = None) -> list[Document]:
    directory = directory or settings.knowledge_base_dir
    docs: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw, fallback_id=path.stem)
        try:
            source_path = str(path.relative_to(settings.base_dir))
        except ValueError:
            source_path = str(path)
        docs.append(
            Document(
                doc_id=meta.get("id", path.stem),
                title=meta.get("title", path.stem),
                tags=tuple(meta.get("tags", [])),
                body=body,
                source_path=source_path,
            )
        )
    return docs


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_document(doc: Document, max_chars: int = 700) -> list[Chunk]:
    """Split a document into chunks of roughly `max_chars` characters,
    on paragraph boundaries where possible. Most of our knowledge-base
    documents are short enough to stay a single chunk, which keeps
    retrieval simple and each chunk self-contained/citable.
    """
    paragraphs = _split_paragraphs(doc.body)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    def flush():
        nonlocal idx, current, current_len
        if not current:
            return
        text = "\n\n".join(current)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{idx}",
                doc_id=doc.doc_id,
                title=doc.title,
                tags=doc.tags,
                text=text,
                source_path=doc.source_path,
            )
        )
        idx += 1
        current = []
        current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            flush()
        current.append(para)
        current_len += len(para)
    flush()

    if not chunks:
        # empty doc guard
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#0",
                doc_id=doc.doc_id,
                title=doc.title,
                tags=doc.tags,
                text=doc.body,
                source_path=doc.source_path,
            )
        )
    return chunks


def load_and_chunk(directory: Path | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents(directory):
        chunks.extend(chunk_document(doc))
    return chunks
