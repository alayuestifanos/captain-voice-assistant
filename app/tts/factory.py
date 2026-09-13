"""Picks a TTS backend based on configuration/available credentials.

Priority when TTS_ENGINE=auto (the default):
  1. ElevenLabs   -- if ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID are set
                      (true voice cloning; best quality for covered languages)
  2. Azure Speech -- if AZURE_SPEECH_KEY + AZURE_SPEECH_REGION are set
                      (best coverage for languages like Amharic)
  3. Local pyttsx3 -- always available, offline, no credentials needed

Set TTS_ENGINE explicitly (elevenlabs|azure|local) to force one backend.
This lets the whole pipeline run end-to-end with zero paid services
configured (just ANTHROPIC_API_KEY), while upgrading transparently once
real credentials are added -- no code changes required.
"""
from __future__ import annotations

from app.config import settings
from app.tts.base import TTSEngine
from app.tts.local_tts import LocalTTSEngine


def get_tts_engine(preferred: str | None = None) -> TTSEngine:
    choice = (preferred or settings.tts_engine or "auto").lower()

    if choice == "local":
        return LocalTTSEngine()

    if choice == "elevenlabs":
        from app.tts.elevenlabs_tts import ElevenLabsTTSEngine

        return ElevenLabsTTSEngine()

    if choice == "azure":
        from app.tts.azure_tts import AzureTTSEngine

        return AzureTTSEngine()

    if choice != "auto":
        raise ValueError(f"Unknown TTS_ENGINE '{choice}'. Use auto, elevenlabs, azure, or local.")

    # auto: try best-quality backends first, fall back to local.
    if settings.elevenlabs_api_key and settings.elevenlabs_voice_id:
        from app.tts.elevenlabs_tts import ElevenLabsTTSEngine

        try:
            return ElevenLabsTTSEngine()
        except RuntimeError:
            pass

    if settings.azure_speech_key and settings.azure_speech_region:
        from app.tts.azure_tts import AzureTTSEngine

        try:
            return AzureTTSEngine()
        except RuntimeError:
            pass

    return LocalTTSEngine()
