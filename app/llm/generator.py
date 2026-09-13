"""Grounded response generation.

The LLM is only ever shown the chunks the retriever judged relevant; it is
instructed to answer strictly from that context and to cite the source
document titles it drew on. If retrieval found nothing relevant enough
(see `Retriever.retrieve_relevant`), we short-circuit before ever calling
the LLM and return a fixed "not covered by the manual" answer -- this is
a stronger anti-hallucination guarantee than prompting alone, since an
empty-context question never reaches generation at all.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.llm.providers import LLMProvider, get_llm_provider
from app.rag.retriever import ScoredChunk

NOT_COVERED_MESSAGE = (
    "I don't have information about that in the ship's operations manual. "
    "Please rephrase, or consult the relevant department directly."
)

SYSTEM_PROMPT = """You are the Captain's onboard assistant for a merchant vessel.
You answer questions strictly using the CONTEXT excerpts provided below, which
are drawn from the ship's operations manual.

Rules:
- Only use facts present in the CONTEXT. Do not use outside knowledge and do
  not speculate or invent procedures, numbers, or regulations.
- If the CONTEXT does not contain enough information to answer, say so plainly
  instead of guessing.
- Keep the answer concise and operational -- the kind of direct, factual
  answer a Captain needs, in 2-5 sentences unless the question needs a list.
- After the answer, on a new line, cite the source titles you actually used,
  formatted exactly as: Sources: <title 1>; <title 2>
- Do not mention "context" or "excerpts" in the answer itself; just answer
  as if you know the ship's procedures."""


@dataclass
class GenerationResult:
    answer: str
    citations: list[str]
    grounded: bool
    raw_context_used: list[str]


def _format_context(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for i, sc in enumerate(chunks, start=1):
        blocks.append(f"[{i}] {sc.chunk.title}\n{sc.chunk.text}")
    return "\n\n".join(blocks)


class ResponseGenerator:
    def __init__(self, provider: LLMProvider | None = None):
        # Resolved lazily (see _resolve_provider) so constructing a
        # ResponseGenerator never requires an API key -- only actually
        # calling the LLM does, and that never happens when there are no
        # relevant chunks (see the short-circuit below).
        self._provider = provider

    def _resolve_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    def generate(self, query: str, chunks: list[ScoredChunk]) -> GenerationResult:
        if not chunks:
            return GenerationResult(
                answer=NOT_COVERED_MESSAGE, citations=[], grounded=False, raw_context_used=[]
            )

        context = _format_context(chunks)
        user_prompt = f"CONTEXT:\n{context}\n\nCAPTAIN'S QUESTION:\n{query}"

        provider = self._resolve_provider()
        try:
            text = provider.complete(
                system=SYSTEM_PROMPT, user=user_prompt, max_tokens=settings.llm_max_tokens
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Response generation failed: {exc}") from exc

        answer, citations = _split_citations(text)
        return GenerationResult(
            answer=answer,
            citations=citations,
            grounded=True,
            raw_context_used=[sc.chunk.title for sc in chunks],
        )


def _split_citations(text: str) -> tuple[str, list[str]]:
    marker = "Sources:"
    if marker in text:
        answer, _, tail = text.rpartition(marker)
        citations = [c.strip() for c in tail.strip().split(";") if c.strip()]
        return answer.strip(), citations
    return text.strip(), []
