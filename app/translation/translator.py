"""Translation stage.

Choice: LLM-based translation (via the same Anthropic model used for
generation) rather than a dedicated translation API such as Google
Translate or DeepL. Reasoning, per the assignment's "justify the choice":

  1. One fewer external API/credential to provision for the demo.
  2. DeepL does not support Amharic at all, and general MT engines are
     historically weak on it; large instruction-tuned LLMs handle
     low-resource languages like Amharic noticeably better in practice
     because they translate meaning/register rather than word-by-word.
  3. It lets us pass the *source context* (the grounded answer, not just
     an isolated sentence) plus a short glossary of maritime terms, which
     keeps domain terminology (e.g. "Man Overboard", "MAYDAY", "COLREGs")
     consistent instead of being awkwardly transliterated.

The trade-off (documented in the README) is turnaround latency/cost vs a
dedicated translation API, and no formal BLEU-style quality guarantee --
acceptable for a prototype at this scale.

`Translator` is a Protocol so a Google/DeepL backend can be dropped in
later without touching the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.config import settings
from app.llm.providers import LLMProvider, get_llm_provider

# ISO 639-1 (or common short code) -> human-readable name, used both to
# render a friendly label in the UI and inside the translation prompt.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "am": "Amharic",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "sw": "Swahili",
    "de": "German",
    "pt": "Portuguese",
    "hi": "Hindi",
    "zh": "Chinese (Simplified)",
}

MARITIME_GLOSSARY = (
    "Captain, Officer of the Watch (OOW), Man Overboard, MAYDAY, PAN PAN, "
    "SECURITE, COLREGs, SOLAS, MARPOL, GMDSS, bridge, muster station, "
    "lifeboat, liferaft, pilot ladder, bunkering, EPIRB, VHF Channel 16"
)


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


class Translator(Protocol):
    def translate(self, text: str, target_lang: str, source_lang: str = "en") -> str: ...


@dataclass
class TranslationResult:
    text: str
    target_lang: str
    skipped: bool  # True when source == target and translation was a no-op


class LLMTranslator:
    def __init__(self, provider: LLMProvider | None = None):
        # Resolved lazily -- translating en->en is a no-op below and never
        # needs a provider/API key at all.
        self._provider = provider

    def _resolve_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    def translate(self, text: str, target_lang: str, source_lang: str = "en") -> TranslationResult:
        if target_lang == source_lang:
            return TranslationResult(text=text, target_lang=target_lang, skipped=True)

        target_name = language_name(target_lang)
        source_name = language_name(source_lang)

        system = (
            f"You are a professional maritime translator. Translate the user's text from "
            f"{source_name} to {target_name}. Preserve the meaning naturally (do not translate "
            f"word-for-word); keep it faithful and operational in tone, suitable to be read aloud. "
            f"Keep these maritime terms recognizable, transliterating rather than inventing new "
            f"terms for them if {target_name} commonly borrows them: {MARITIME_GLOSSARY}. "
            f"Return ONLY the translated text, with no preamble, quotes, or explanation."
        )

        provider = self._resolve_provider()
        try:
            translated = provider.complete(
                system=system, user=text, max_tokens=settings.llm_max_tokens, temperature=0
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Translation failed: {exc}") from exc

        return TranslationResult(text=translated, target_lang=target_lang, skipped=False)
