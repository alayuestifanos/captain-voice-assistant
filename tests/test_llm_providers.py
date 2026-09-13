from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest

from app.config import settings
from app.llm.providers import AnthropicProvider, GroqProvider, OllamaProvider, get_llm_provider


# --- GroqProvider (default, free tier) --------------------------------------


def _fake_groq_response(status_code=200, content="translated text", finish_reason="stop", json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    }
    resp.text = str(json_body or content)
    return resp


def test_groq_provider_requires_api_key():
    provider = GroqProvider(api_key="")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        provider.complete(system="sys", user="hi", max_tokens=100)


def test_groq_provider_returns_completion_text():
    provider = GroqProvider(api_key="fake-key", model="llama-3.3-70b-versatile", reasoning_effort="")
    with patch("requests.post", return_value=_fake_groq_response(content="hello there")) as post:
        result = provider.complete(system="You are helpful.", user="hi", max_tokens=100)

    assert result == "hello there"
    _, kwargs = post.call_args
    assert kwargs["json"]["model"] == "llama-3.3-70b-versatile"
    assert kwargs["json"]["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert kwargs["json"]["messages"][1] == {"role": "user", "content": "hi"}
    assert kwargs["headers"]["Authorization"] == "Bearer fake-key"
    assert "reasoning_effort" not in kwargs["json"]  # explicitly disabled above


def test_groq_provider_includes_reasoning_effort_by_default():
    """gpt-oss and similar reasoning models need this capped, or they can
    spend the whole token budget on hidden reasoning (see the empty-content
    tests below) -- 'low' is the default."""
    provider = GroqProvider(api_key="fake-key")
    with patch("requests.post", return_value=_fake_groq_response(content="hello there")) as post:
        provider.complete(system="sys", user="hi", max_tokens=100)

    assert post.call_args.kwargs["json"]["reasoning_effort"] == "low"


def test_groq_provider_raises_clear_error_when_reasoning_budget_exhausted():
    """A reasoning model (e.g. gpt-oss) can burn the whole max_tokens
    budget on hidden chain-of-thought and return empty `content` with
    finish_reason=length -- must surface as an actionable error, not a
    silently empty translation/answer."""
    provider = GroqProvider(api_key="fake-key")
    with patch("requests.post", return_value=_fake_groq_response(content="", finish_reason="length")):
        with pytest.raises(RuntimeError, match="entire token budget on internal reasoning"):
            provider.complete(system="sys", user="hi", max_tokens=100)


def test_groq_provider_raises_error_on_empty_content():
    provider = GroqProvider(api_key="fake-key")
    with patch("requests.post", return_value=_fake_groq_response(content="", finish_reason="stop")):
        with pytest.raises(RuntimeError, match="returned an empty response"):
            provider.complete(system="sys", user="hi", max_tokens=100)


def test_groq_provider_reports_clean_error_on_bad_key():
    provider = GroqProvider(api_key="bad-key")
    error_body = {"error": {"message": "Invalid API Key"}}
    with patch("requests.post", return_value=_fake_groq_response(status_code=401, json_body=error_body)):
        with pytest.raises(RuntimeError, match="Groq API key was rejected"):
            provider.complete(system="sys", user="hi", max_tokens=100)


def test_groq_provider_reports_clean_error_on_rate_limit():
    provider = GroqProvider(api_key="fake-key")
    with patch("requests.post", return_value=_fake_groq_response(status_code=429, json_body={"error": {}})):
        with pytest.raises(RuntimeError, match="rate limited"):
            provider.complete(system="sys", user="hi", max_tokens=100)


# --- AnthropicProvider (optional, pluggable alternative) ---------------------


def _fake_anthropic_message(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    message = MagicMock()
    message.content = [block]
    return message


def test_anthropic_provider_requires_api_key():
    provider = AnthropicProvider(api_key="")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        provider.complete(system="sys", user="hi", max_tokens=100)


def test_anthropic_provider_returns_completion_text():
    provider = AnthropicProvider(api_key="fake-key")
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_anthropic_message("hello there")
    provider._client = fake_client

    result = provider.complete(system="You are helpful.", user="hi", max_tokens=100)

    assert result == "hello there"
    _, kwargs = fake_client.messages.create.call_args
    assert kwargs["system"] == "You are helpful."
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_provider_reports_clean_error_on_invalid_key():
    provider = AnthropicProvider(api_key="bad-key")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request, json={"error": {"message": "invalid x-api-key"}})
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = anthropic.AuthenticationError(
        "invalid x-api-key", response=response, body=None
    )
    provider._client = fake_client

    with pytest.raises(RuntimeError, match="API key was rejected"):
        provider.complete(system="sys", user="hi", max_tokens=100)


# --- get_llm_provider() factory selection ------------------------------------


def test_factory_picks_groq_when_only_groq_key_set(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "groq_api_key", "groq-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    assert isinstance(get_llm_provider(), GroqProvider)


def test_factory_picks_anthropic_when_only_anthropic_key_set(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "anthropic-key")
    assert isinstance(get_llm_provider(), AnthropicProvider)


def test_factory_prefers_groq_when_both_keys_set(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "groq_api_key", "groq-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "anthropic-key")
    assert isinstance(get_llm_provider(), GroqProvider)


def test_factory_defaults_to_ollama_when_no_keys_set(monkeypatch):
    """Even with nothing configured, fall back to the local, no-signup
    Ollama provider -- mirrors the TTS factory's local-fallback design."""
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    assert isinstance(get_llm_provider(), OllamaProvider)


def test_factory_honors_explicit_provider_override(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "groq-key")
    assert isinstance(get_llm_provider(preferred="anthropic"), AnthropicProvider)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm_provider(preferred="not-a-real-provider")


# --- OllamaProvider (fully local, no signup/key/CAPTCHA) ---------------------


def _fake_ollama_response(status_code=200, content="hello there", json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {"message": {"role": "assistant", "content": content}}
    resp.text = str(json_body or content)
    return resp


def test_ollama_provider_returns_completion_text():
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.1")
    with patch("requests.post", return_value=_fake_ollama_response(content="hello there")) as post:
        result = provider.complete(system="You are helpful.", user="hi", max_tokens=100)

    assert result == "hello there"
    _, kwargs = post.call_args
    assert post.call_args[0][0] == "http://localhost:11434/api/chat"
    assert kwargs["json"]["model"] == "llama3.1"
    assert kwargs["json"]["stream"] is False
    assert kwargs["json"]["messages"][0] == {"role": "system", "content": "You are helpful."}


def test_ollama_provider_reports_clean_error_when_server_unreachable():
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.1")
    import requests

    with patch("requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(RuntimeError, match="Could not reach Ollama"):
            provider.complete(system="sys", user="hi", max_tokens=100)


def test_ollama_provider_reports_clean_error_when_model_not_pulled():
    provider = OllamaProvider(base_url="http://localhost:11434", model="mystery-model")
    with patch("requests.post", return_value=_fake_ollama_response(status_code=404)):
        with pytest.raises(RuntimeError, match="not pulled yet"):
            provider.complete(system="sys", user="hi", max_tokens=100)
