"""Local, offline TTS fallback using pyttsx3 (wraps the OS's SAPI5/NSSpeech
voices). Always available, no API key required -- this is the guaranteed
"consistent voice profile" fallback described in the assignment when true
voice cloning isn't set up (see README "Voice Synthesis" tradeoffs).

Known limitation: the voices bundled with a typical Windows install only
cover a handful of languages (usually just English unless extra language
packs are installed), so this engine is a good demonstration voice for
English but is NOT expected to pronounce Amharic correctly. It is kept as
the always-on fallback so the app never fails to produce *some* audio;
ElevenLabs/Azure (see elevenlabs_tts.py / azure_tts.py) are the paths to
real multilingual, cloned-voice output.
"""
from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.tts.base import SynthesisResult


class LocalTTSEngine:
    name = "local"

    def __init__(self, voice_name: str | None = None, rate: int | None = None):
        self.voice_name = voice_name if voice_name is not None else settings.local_voice_name
        self.rate = rate or settings.local_voice_rate
        self._chosen_voice_id: str | None = None
        self._chosen_voice_label = "default"

    def _pick_voice(self, engine) -> None:
        voices = engine.getProperty("voices")
        if not voices:
            return

        if self.voice_name:
            for v in voices:
                if self.voice_name.lower() in (v.name or "").lower():
                    self._chosen_voice_id = v.id
                    self._chosen_voice_label = v.name
                    return

        # Deterministic default: always the same voice (sorted by id) so
        # the "Captain" persona sounds consistent across runs/machines.
        chosen = sorted(voices, key=lambda v: v.id)[0]
        self._chosen_voice_id = chosen.id
        self._chosen_voice_label = chosen.name

    def supports_language(self, language: str) -> bool:
        # pyttsx3/SAPI5 on a stock Windows install reliably covers English
        # only. Treat everything else as "unsupported" so callers can warn.
        return language == "en"

    def synthesize(self, text: str, language: str, output_path: Path) -> SynthesisResult:
        import pyttsx3

        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine = pyttsx3.init()
        self._pick_voice(engine)
        if self._chosen_voice_id:
            engine.setProperty("voice", self._chosen_voice_id)
        engine.setProperty("rate", self.rate)

        engine.save_to_file(text, str(output_path))
        engine.runAndWait()

        warning = None
        if not self.supports_language(language):
            warning = (
                f"Local TTS voice '{self._chosen_voice_label}' is not verified to support "
                f"'{language}'; audio may mispronounce the translated text. Configure "
                f"ELEVENLABS_API_KEY or AZURE_SPEECH_KEY for proper {language} speech."
            )

        return SynthesisResult(
            audio_path=output_path,
            engine=self.name,
            voice_label=self._chosen_voice_label,
            language=language,
            warning=warning,
        )
