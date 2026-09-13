from unittest.mock import MagicMock

import pytest

from app.translation.translator import LLMTranslator


def test_translate_skips_when_target_equals_source():
    translator = LLMTranslator()
    result = translator.translate("Proceed to the muster station.", target_lang="en", source_lang="en")
    assert result.skipped is True
    assert result.text == "Proceed to the muster station."
    assert translator._provider is None  # never resolved -- no key needed for a no-op


def test_translate_calls_provider_and_returns_translated_text():
    fake_provider = MagicMock()
    fake_provider.complete.return_value = "ወደ መሰብሰቢያ ጣቢያ ይሂዱ።"
    translator = LLMTranslator(provider=fake_provider)

    result = translator.translate("Proceed to the muster station.", target_lang="am", source_lang="en")

    assert result.skipped is False
    assert result.target_lang == "am"
    assert result.text == "ወደ መሰብሰቢያ ጣቢያ ይሂዱ።"

    # Verify the prompt told the provider the correct source/target languages.
    _, kwargs = fake_provider.complete.call_args
    assert "Amharic" in kwargs["system"]
    assert "English" in kwargs["system"]
    assert kwargs["user"] == "Proceed to the muster station."
    assert kwargs["temperature"] == 0


def test_translate_reports_a_clean_error_from_the_provider():
    """A provider failure (bad key, rate limit, etc.) must surface as a
    RuntimeError with the provider's message, not an unhandled exception."""
    fake_provider = MagicMock()
    fake_provider.complete.side_effect = RuntimeError("the Groq API key was rejected (401)")
    translator = LLMTranslator(provider=fake_provider)

    with pytest.raises(RuntimeError, match="API key was rejected"):
        translator.translate("Proceed to the muster station.", target_lang="am", source_lang="en")
