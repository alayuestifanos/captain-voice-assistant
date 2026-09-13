"""LLM chat-completion providers, used by both grounded generation
(app/llm/generator.py) and translation (app/translation/translator.py).

Three backends, behind one small `LLMProvider` interface -- mirroring the
existing cloud/cloud/local fallback design in app/tts/factory.py:

  - GroqProvider     -- free tier, open-source models (Llama 3.3 70B by
                        default) via Groq's OpenAI-compatible REST API.
                        No credit card required to get a key. Default
                        cloud option so the pipeline can run at zero cost.
  - AnthropicProvider -- Claude, via the official `anthropic` SDK. A
                        pluggable alternative (the assignment's suggested
                        stack lists Claude explicitly) for anyone who
                        already has an Anthropic key or wants Claude's
                        quality instead of a free open-source model.
  - OllamaProvider    -- fully local, via a locally-running Ollama server
                        (https://ollama.com). No account, no API key, no
                        web signup/CAPTCHA of any kind -- just install
                        Ollama and `ollama pull` a model once. The
                        guaranteed no-credentials fallback, same role
                        pyttsx3 plays for TTS.

No provider's constructor requires a key/server to be present -- the
"not available" check happens lazily inside `complete()`, so building a
ResponseGenerator/LLMTranslator never fails just because nothing is
configured yet (matching the existing short-circuit behaviour: a query
that never needs the LLM, e.g. no relevant chunks retrieved, must never
require a key at all).
"""
from __future__ import annotations

from typing import Protocol

from app.config import settings


class LLMProvider(Protocol):
    name: str

    def complete(
        self, system: str, user: str, max_tokens: int, temperature: float | None = None
    ) -> str: ...


class GroqProvider:
    """Free tier via Groq's OpenAI-compatible chat completions API.

    Get a free key (no credit card) at https://console.groq.com/keys.
    """

    name = "groq"
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None, reasoning_effort: str | None = None):
        # `is None` (not falsy-`or`) so an explicitly-passed "" means "no
        # key" rather than silently falling back to the ambient .env value.
        self.api_key = settings.groq_api_key if api_key is None else api_key
        self.model = model or settings.groq_model
        self.reasoning_effort = settings.groq_reasoning_effort if reasoning_effort is None else reasoning_effort

    def complete(
        self, system: str, user: str, max_tokens: int, temperature: float | None = None
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys (no credit card required) and set it "
                "in .env -- or set ANTHROPIC_API_KEY / LLM_PROVIDER=anthropic to use "
                "Claude instead."
            )

        import requests

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if self.reasoning_effort:
            # Several current Groq models (e.g. openai/gpt-oss-*) are
            # "reasoning" models that spend tokens on a hidden chain-of-
            # thought before writing the final answer -- without capping
            # that effort, they can burn the whole max_tokens budget on
            # reasoning and return an EMPTY `content` (finish_reason
            # "length"). Capping effort low leaves room for the actual
            # answer. Non-reasoning models on Groq ignore this field.
            payload["reasoning_effort"] = self.reasoning_effort

        try:
            response = requests.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach the Groq API: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(_describe_groq_error(response))

        data = response.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Groq API response shape: {data}") from exc

        if not content:
            # Most commonly: a reasoning model spent its whole token
            # budget on hidden reasoning and never wrote a final answer.
            if choice.get("finish_reason") == "length":
                raise RuntimeError(
                    f"Groq model '{self.model}' used its entire token budget on internal "
                    "reasoning and returned no answer (finish_reason=length). Raise "
                    "LLM_MAX_TOKENS in .env, or lower GROQ_REASONING_EFFORT (low/minimal)."
                )
            raise RuntimeError(f"Groq model '{self.model}' returned an empty response.")

        return content


def _describe_groq_error(response) -> str:
    status = response.status_code
    try:
        detail = response.json().get("error", {}).get("message", response.text[:300])
    except ValueError:
        detail = response.text[:300]

    if status == 401:
        return (
            "the Groq API key was rejected (401). Double-check GROQ_API_KEY in your "
            ".env is a full, unmodified key copied from https://console.groq.com/keys."
        )
    if status == 429:
        return "rate limited by the Groq free tier (429). Wait a moment and try again."
    if status == 400 and "decommissioned" in detail.lower():
        return f"Groq model error (400): {detail}. Set GROQ_MODEL to a currently supported model."
    return f"Groq API returned {status}: {detail}"


class AnthropicProvider:
    """Claude via the official `anthropic` SDK -- a pluggable, non-default
    alternative to the free Groq provider (see module docstring)."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        # `is None` (not falsy-`or`) so an explicitly-passed "" means "no
        # key" rather than silently falling back to the ambient .env value.
        self.api_key = settings.anthropic_api_key if api_key is None else api_key
        self.model = model or settings.anthropic_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(
        self, system: str, user: str, max_tokens: int, temperature: float | None = None
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in your .env to use Claude -- or "
                "leave it unset and set GROQ_API_KEY instead for the free default provider."
            )

        from app.llm.errors import describe_anthropic_error

        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if temperature is not None:
            kwargs["temperature"] = temperature

        client = self._get_client()
        try:
            message = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalized below
            raise RuntimeError(describe_anthropic_error(exc)) from exc

        return "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()


class OllamaProvider:
    """Fully local via a locally-running Ollama server -- no account, no
    API key, no web signup or CAPTCHA of any kind.

    Setup (one-time):
      1. Install from https://ollama.com/download
      2. `ollama pull llama3.1` (or another model; ~4-5GB, one-time download)
      3. Ollama runs its own local server automatically -- nothing else to do.
    """

    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    def complete(
        self, system: str, user: str, max_tokens: int, temperature: float | None = None
    ) -> str:
        import requests

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if temperature is not None:
            payload["options"]["temperature"] = temperature

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url} ({exc}). Install it from "
                "https://ollama.com/download, make sure it's running, and run "
                f"`ollama pull {self.model}` -- or set GROQ_API_KEY/ANTHROPIC_API_KEY "
                "instead to use a cloud provider."
            ) from exc

        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{self.model}' is not pulled yet. Run `ollama pull {self.model}` "
                "and try again."
            )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned {response.status_code}: {response.text[:300]}")

        data = response.json()
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response shape: {data}") from exc


def get_llm_provider(preferred: str | None = None) -> LLMProvider:
    """Pick a provider based on config/available credentials.

    Priority when LLM_PROVIDER=auto (the default):
      1. Groq      -- if GROQ_API_KEY is set (free tier, open-source model)
      2. Anthropic -- if ANTHROPIC_API_KEY is set (Claude)
      3. Ollama     -- otherwise: the guaranteed no-credentials local
         fallback. If Ollama isn't installed/running, its `complete()`
         call raises a clear setup error rather than silently doing
         nothing.

    Set LLM_PROVIDER explicitly (groq|anthropic|ollama) to force one backend.
    """
    choice = (preferred or settings.llm_provider or "auto").lower()

    if choice == "groq":
        return GroqProvider()
    if choice == "anthropic":
        return AnthropicProvider()
    if choice == "ollama":
        return OllamaProvider()
    if choice != "auto":
        raise ValueError(f"Unknown LLM_PROVIDER '{choice}'. Use auto, groq, anthropic, or ollama.")

    if settings.groq_api_key:
        return GroqProvider()
    if settings.anthropic_api_key:
        return AnthropicProvider()
    return OllamaProvider()
