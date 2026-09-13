"""ElevenLabs TTS backend.

Two ways to get a voice_id, with a real plan restriction discovered
empirically (not just from the docs) while building this:

  1. True voice cloning (Instant Voice Cloning): record a sample and run
     `python scripts/clone_voice.py "Captain Voice" <sample.wav>`, which
     calls the ElevenLabs Voice Add API and prints a voice_id.
     *Requires a paid ElevenLabs plan* -- a free-tier key gets a 400
     "can_not_use_instant_voice_cloning" error from that endpoint.
  2. A stock "premade" voice (bundled with every account, free or paid):
     `GET /v1/voices` and pick one with `"category": "premade"`. This
     *does* work on the free tier via the API. Voices from the Voice
     *Library* (`"category": "professional"` or similar) do NOT work on
     the free tier via the API even if added to "My Voices" -- ElevenLabs
     returns a 402 "paid_plan_required" for those specifically.

So on a free ElevenLabs account, this backend gives the assignment's
explicitly-accepted fallback -- "a well-configured single consistent
voice profile" -- rather than true cloning, same role Azure's fixed
per-language voice plays. `voice_label` reflects this honestly (see
below) instead of always claiming "cloned".

Language coverage: eleven_multilingual_v2 (the default model here) covers
~29 languages. As of this writing Amharic is not one of them, so we still
attempt synthesis (the model will do its best phonetic approximation) but
flag a warning -- see `_MODEL_SUPPORTED_LANGS` below. Azure Speech
(azure_tts.py) is the recommended backend specifically for Amharic output,
since it ships dedicated am-ET neural voices.
"""
from __future__ import annotations

from pathlib import Path

import requests

from app.config import settings
from app.tts.base import SynthesisResult

API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# Languages eleven_multilingual_v2 is documented to support well.
_MODEL_SUPPORTED_LANGS = {
    "en", "ja", "zh", "de", "hi", "fr", "ko", "pt", "it", "es", "id", "nl",
    "tr", "fil", "pl", "sv", "bg", "ro", "ar", "cs", "el", "fi", "hr", "ms",
    "sk", "da", "ta", "uk", "ru",
}


class ElevenLabsTTSEngine:
    name = "elevenlabs"

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
    ):
        self.api_key = api_key or settings.elevenlabs_api_key
        self.voice_id = voice_id or settings.elevenlabs_voice_id
        self.model_id = model_id or settings.elevenlabs_model_id
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set.")
        if not self.voice_id:
            raise RuntimeError(
                "ELEVENLABS_VOICE_ID is not set. Run scripts/clone_voice.py to create a "
                "cloned Captain voice, or set it to a stock ElevenLabs voice id."
            )

    def supports_language(self, language: str) -> bool:
        return language in _MODEL_SUPPORTED_LANGS

    def synthesize(self, text: str, language: str, output_path: Path) -> SynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.post(
            API_URL.format(voice_id=self.voice_id),
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"ElevenLabs TTS request failed ({response.status_code}): {response.text[:300]}"
            )

        output_path.write_bytes(response.content)

        warning = None
        if not self.supports_language(language):
            warning = (
                f"'{language}' is not an officially supported language for {self.model_id}; "
                "pronunciation quality is not guaranteed."
            )

        return SynthesisResult(
            audio_path=output_path,
            engine=self.name,
            # Not necessarily a true clone (see module docstring) -- this
            # backend is used the same way whether voice_id points at an
            # Instant Voice Clone (paid plans) or a stock premade voice
            # (free tier), so the label doesn't assert which one it is.
            voice_label=f"elevenlabs:{self.voice_id}",
            language=language,
            warning=warning,
        )
