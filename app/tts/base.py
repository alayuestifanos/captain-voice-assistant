"""Common TTS engine interface.

Every backend (local/offline, ElevenLabs, Azure) implements `synthesize`
and reports whether it can plausibly speak a given language, so the
factory/pipeline can pick a working engine and the UI can warn the user
about language coverage instead of silently producing bad audio.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class SynthesisResult:
    audio_path: Path
    engine: str
    voice_label: str
    language: str
    warning: str | None = None


class TTSEngine(Protocol):
    name: str

    def synthesize(self, text: str, language: str, output_path: Path) -> SynthesisResult: ...

    def supports_language(self, language: str) -> bool: ...
