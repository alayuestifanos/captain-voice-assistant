from unittest.mock import MagicMock

import pytest

from app.llm.generator import NOT_COVERED_MESSAGE, ResponseGenerator
from app.rag.knowledge_base import Chunk
from app.rag.retriever import ScoredChunk


def test_generate_short_circuits_when_no_chunks_retrieved():
    """No relevant context -> fixed answer, the LLM provider is never
    resolved or called (no hallucination risk, and no API key needed)."""
    generator = ResponseGenerator()
    result = generator.generate("What's the capital of France?", chunks=[])
    assert result.grounded is False
    assert result.answer == NOT_COVERED_MESSAGE
    assert result.citations == []
    assert generator._provider is None  # never resolved


def test_generate_grounds_answer_and_parses_citations():
    chunk = Chunk("mob#0", "mob", "Man Overboard Procedure", (), "Throw a lifebuoy immediately.", "mob.md")
    fake_provider = MagicMock()
    fake_provider.complete.return_value = (
        "Throw the lifebuoy and sound the alarm immediately.\nSources: Man Overboard Procedure"
    )
    generator = ResponseGenerator(provider=fake_provider)

    result = generator.generate("What do I do for man overboard?", chunks=[ScoredChunk(chunk, 0.9)])

    assert result.grounded is True
    assert "lifebuoy" in result.answer.lower()
    assert result.citations == ["Man Overboard Procedure"]

    # The retrieved chunk text must have been placed in the prompt sent to the provider.
    _, kwargs = fake_provider.complete.call_args
    assert "Throw a lifebuoy immediately." in kwargs["user"]
    assert kwargs["system"]  # the grounding system prompt was passed through


def test_generate_reports_a_clean_error_from_the_provider():
    """A provider failure (bad key, rate limit, etc.) must surface as a
    RuntimeError with the provider's message, not an unhandled exception."""
    chunk = Chunk("mob#0", "mob", "Man Overboard Procedure", (), "Throw a lifebuoy immediately.", "mob.md")
    fake_provider = MagicMock()
    fake_provider.complete.side_effect = RuntimeError("the Groq API key was rejected (401)")
    generator = ResponseGenerator(provider=fake_provider)

    with pytest.raises(RuntimeError, match="API key was rejected"):
        generator.generate("What do I do for man overboard?", chunks=[ScoredChunk(chunk, 0.9)])
