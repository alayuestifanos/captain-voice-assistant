import json
from unittest.mock import MagicMock

import pytest

from app.config import settings
from app.llm.generator import GenerationResult
from app.pipeline import CaptainAssistantPipeline
from app.rag.knowledge_base import Chunk
from app.rag.retriever import ScoredChunk
from app.translation.translator import TranslationResult
from app.tts.base import SynthesisResult


@pytest.fixture()
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "logs_dir", tmp_path / "logs")
    monkeypatch.setattr(settings, "audio_output_dir", tmp_path / "audio")
    settings.ensure_dirs()
    return tmp_path


def _pipeline_with_mocks(retrieved, grounded=True, tts_warning=None):
    retriever = MagicMock()
    retriever.retrieve_relevant.return_value = retrieved

    generator = MagicMock()
    generator.generate.return_value = GenerationResult(
        answer="Throw the lifebuoy immediately.",
        citations=["Man Overboard Procedure"],
        grounded=grounded,
        raw_context_used=[sc.chunk.title for sc in retrieved],
    )

    translator = MagicMock()
    translator.translate.return_value = TranslationResult(
        text="ወዲያውኑ የህይወት ማዳኛውን ይወርውሩ።", target_lang="am", skipped=False
    )

    tts = MagicMock()

    def fake_synthesize(text, language, output_path):
        output_path.write_bytes(b"fake-mp3-bytes")
        return SynthesisResult(
            audio_path=output_path, engine="local", voice_label="test-voice", language=language, warning=tts_warning
        )

    tts.synthesize.side_effect = fake_synthesize

    pipeline = CaptainAssistantPipeline(
        retriever=retriever, generator=generator, translator=translator, tts_engine=tts
    )
    return pipeline, retriever, generator, translator, tts


def test_pipeline_runs_all_stages_in_order(isolated_dirs):
    chunk = Chunk("mob#0", "mob", "Man Overboard Procedure", (), "Throw a lifebuoy immediately.", "mob.md")
    retrieved = [ScoredChunk(chunk, 0.87)]
    pipeline, retriever, generator, translator, tts = _pipeline_with_mocks(retrieved)

    result = pipeline.run("Man overboard, what do I do?", target_lang="am")

    retriever.retrieve_relevant.assert_called_once()
    generator.generate.assert_called_once_with("Man overboard, what do I do?", retrieved)
    translator.translate.assert_called_once_with(
        "Throw the lifebuoy immediately.", target_lang="am", source_lang="en"
    )
    tts.synthesize.assert_called_once()

    assert result.synthesis is not None
    assert result.synthesis.audio_path.exists()
    assert result.warnings == []

    api_dict = result.to_api_dict()
    assert api_dict["grounded"] is True
    assert api_dict["citations"] == ["Man Overboard Procedure"]
    assert api_dict["audio_url"] == f"/api/audio/{result.trace_id}.mp3"


def test_pipeline_persists_trace_with_all_stages(isolated_dirs):
    chunk = Chunk("mob#0", "mob", "Man Overboard Procedure", (), "Throw a lifebuoy immediately.", "mob.md")
    pipeline, *_ = _pipeline_with_mocks([ScoredChunk(chunk, 0.87)])

    result = pipeline.run("Man overboard, what do I do?", target_lang="am")

    assert result.trace_path.exists()
    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["input", "retrieval", "generation", "translation", "synthesis"]


def test_pipeline_warns_when_nothing_relevant_retrieved(isolated_dirs):
    pipeline, *_ = _pipeline_with_mocks([], grounded=False)
    result = pipeline.run("Unrelated question", target_lang="am")
    assert any("relevant" in w.lower() for w in result.warnings)


def test_pipeline_can_skip_audio_synthesis(isolated_dirs):
    chunk = Chunk("mob#0", "mob", "Man Overboard Procedure", (), "Throw a lifebuoy immediately.", "mob.md")
    pipeline, retriever, generator, translator, tts = _pipeline_with_mocks([ScoredChunk(chunk, 0.87)])

    result = pipeline.run("Man overboard, what do I do?", target_lang="am", synthesize_audio=False)

    tts.synthesize.assert_not_called()
    assert result.synthesis is None
    assert result.to_api_dict()["audio_url"] is None
