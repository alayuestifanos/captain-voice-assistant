"""Shared helper to turn Anthropic SDK exceptions into clean, user-facing
messages instead of a raw traceback bubbling up to the CLI/API caller.
Used by app/llm/providers.py:AnthropicProvider. Groq's REST errors are
normalized in app/llm/providers.py directly (plain HTTP status codes, no
SDK exception types involved).
"""
from __future__ import annotations


def describe_anthropic_error(exc: Exception) -> str:
    try:
        import anthropic
    except ImportError:  # pragma: no cover - anthropic is optional now that Groq is the default
        return str(exc)

    if isinstance(exc, anthropic.AuthenticationError):
        return (
            "the Anthropic API key was rejected (401). Double-check ANTHROPIC_API_KEY in your "
            ".env is a full, unmodified key copied from https://console.anthropic.com/settings/keys."
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "the Anthropic API key does not have permission for this request (403)."
    if isinstance(exc, anthropic.RateLimitError):
        return "rate limited by the Anthropic API (429). Wait a moment and try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "could not reach the Anthropic API. Check your network connection."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API returned {exc.status_code}: {exc.message}"
    return str(exc)


# Backwards-compatible alias (previous name, before Groq was added as the default provider).
describe_api_error = describe_anthropic_error
