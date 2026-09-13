"""Azure Cognitive Services Speech backend.

This is the recommended engine for languages ElevenLabs' multilingual
model doesn't cover well -- notably Amharic, which Azure ships dedicated
neural voices for (am-ET-AmehaNeural / am-ET-MekdesNeural). True voice
*cloning* on Azure requires the separate Custom Neural Voice program
(an approval + training process out of scope for this prototype); instead
we pick one fixed neural voice per language so the "Captain" has a single,
consistent, well-configured voice profile per language, per the
assignment's explicitly-accepted fallback for when full cloning isn't
feasible in the time given.

Implemented as plain REST calls (token endpoint + SSML synthesis) rather
than the azure-cognitiveservices-speech SDK, to avoid pulling in its large
native binary dependency for a prototype.
"""
from __future__ import annotations

from pathlib import Path

import requests

from app.config import settings
from app.tts.base import SynthesisResult

TOKEN_URL = "https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
SYNTH_URL = "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

# One consistent "Captain" voice per language. Override by editing this
# map if your Azure resource/region offers a preferred alternative voice.
DEFAULT_VOICE_MAP: dict[str, str] = {
    "en": "en-US-GuyNeural",
    "am": "am-ET-AmehaNeural",
    "ar": "ar-SA-HamedNeural",
    "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural",
    "sw": "sw-KE-RafikiNeural",
    "de": "de-DE-ConradNeural",
    "pt": "pt-BR-AntonioNeural",
    "hi": "hi-IN-MadhurNeural",
    "zh": "zh-CN-YunxiNeural",
}


class AzureTTSEngine:
    name = "azure"

    def __init__(
        self,
        api_key: str | None = None,
        region: str | None = None,
        voice_map: dict[str, str] | None = None,
    ):
        self.api_key = api_key or settings.azure_speech_key
        self.region = region or settings.azure_speech_region
        self.voice_map = voice_map or DEFAULT_VOICE_MAP
        if not self.api_key or not self.region:
            raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not set.")

    def supports_language(self, language: str) -> bool:
        return language in self.voice_map

    def _get_token(self) -> str:
        resp = requests.post(
            TOKEN_URL.format(region=self.region),
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text

    def synthesize(self, text: str, language: str, output_path: Path) -> SynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        voice = self.voice_map.get(language)
        warning = None
        if not voice:
            voice = self.voice_map["en"]
            warning = f"No Azure voice configured for '{language}'; falling back to {voice}."

        locale = "-".join(voice.split("-")[:2])
        ssml = (
            f'<speak version="1.0" xml:lang="{locale}">'
            f'<voice name="{voice}">{_xml_escape(text)}</voice></speak>'
        )

        token = self._get_token()
        resp = requests.post(
            SYNTH_URL.format(region=self.region),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
                "User-Agent": "captain-voice-assistant",
            },
            data=ssml.encode("utf-8"),
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Azure TTS request failed ({resp.status_code}): {resp.text[:300]}")

        output_path.write_bytes(resp.content)

        return SynthesisResult(
            audio_path=output_path,
            engine=self.name,
            voice_label=voice,
            language=language,
            warning=warning,
        )


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
