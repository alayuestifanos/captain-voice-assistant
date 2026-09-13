"""End-to-end orchestration: retrieve -> generate -> translate -> synthesize.

This is the one place that wires the four stages together and writes the
pipeline trace, per the assignment's "clean separation of concerns"
criterion -- each stage module (rag/, llm/, translation/, tts/) knows
nothing about the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.llm.generator import GenerationResult, ResponseGenerator
from app.logging_utils import PipelineTrace
from app.rag.retriever import Retriever, ScoredChunk
from app.translation.translator import LLMTranslator, TranslationResult, language_name
from app.tts.base import SynthesisResult, TTSEngine
from app.tts.factory import get_tts_engine


@dataclass
class PipelineResult:
    trace_id: str
    query: str
    target_lang: str
    retrieved: list[ScoredChunk]
    generation: GenerationResult
    translation: TranslationResult
    synthesis: SynthesisResult | None
    trace_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def to_api_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "target_lang": self.target_lang,
            "target_lang_name": language_name(self.target_lang),
            "retrieved_chunks": [
                {
                    "title": sc.chunk.title,
                    "source": sc.chunk.source_path,
                    "score": round(sc.score, 4),
                    "text": sc.chunk.text,
                }
                for sc in self.retrieved
            ],
            "grounded": self.generation.grounded,
            "answer_en": self.generation.answer,
            "citations": self.generation.citations,
            "translated_text": self.translation.text,
            "translation_skipped": self.translation.skipped,
            "audio_url": f"/api/audio/{self.synthesis.audio_path.name}" if self.synthesis else None,
            "tts_engine": self.synthesis.engine if self.synthesis else None,
            "voice_label": self.synthesis.voice_label if self.synthesis else None,
            "warnings": self.warnings,
        }


class CaptainAssistantPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: ResponseGenerator | None = None,
        translator: LLMTranslator | None = None,
        tts_engine: TTSEngine | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.generator = generator or ResponseGenerator()
        self.translator = translator or LLMTranslator()
        self._tts_engine = tts_engine  # lazily resolved via factory if None

    def _resolve_tts(self) -> TTSEngine:
        if self._tts_engine is None:
            self._tts_engine = get_tts_engine()
        return self._tts_engine

    def run(
        self,
        query: str,
        target_lang: str | None = None,
        synthesize_audio: bool = True,
    ) -> PipelineResult:
        target_lang = target_lang or settings.default_target_lang
        trace = PipelineTrace()
        warnings: list[str] = []

        trace.record("input", query=query, target_lang=target_lang)

        # 1. Retrieval
        retrieved = self.retriever.retrieve_relevant(query)
        trace.record(
            "retrieval",
            top_k=settings.retrieval_top_k,
            threshold=settings.relevance_threshold,
            results=[
                {"title": sc.chunk.title, "score": round(sc.score, 4), "chunk_id": sc.chunk.chunk_id}
                for sc in retrieved
            ],
        )

        # 2. Grounded generation
        generation = self.generator.generate(query, retrieved)
        trace.record(
            "generation",
            grounded=generation.grounded,
            answer=generation.answer,
            citations=generation.citations,
        )
        if not generation.grounded:
            warnings.append("No sufficiently relevant knowledge-base content was found for this query.")

        # 3. Translation
        translation = self.translator.translate(generation.answer, target_lang=target_lang, source_lang="en")
        trace.record(
            "translation",
            target_lang=target_lang,
            skipped=translation.skipped,
            text=translation.text,
        )

        # 4. Speech synthesis
        synthesis: SynthesisResult | None = None
        if synthesize_audio:
            tts = self._resolve_tts()
            output_path = settings.audio_output_dir / f"{trace.trace_id}.mp3"
            try:
                synthesis = tts.synthesize(translation.text, target_lang, output_path)
                if synthesis.warning:
                    warnings.append(synthesis.warning)
                trace.record(
                    "synthesis",
                    engine=synthesis.engine,
                    voice=synthesis.voice_label,
                    audio_path=str(synthesis.audio_path),
                    warning=synthesis.warning,
                )
            except Exception as exc:  # noqa: BLE001 - surface as trace + warning, don't crash the API
                warnings.append(f"Speech synthesis failed: {exc}")
                trace.record("synthesis_error", error=str(exc))

        trace_path = trace.persist()

        return PipelineResult(
            trace_id=trace.trace_id,
            query=query,
            target_lang=target_lang,
            retrieved=retrieved,
            generation=generation,
            translation=translation,
            synthesis=synthesis,
            trace_path=trace_path,
            warnings=warnings,
        )
