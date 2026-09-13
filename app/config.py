"""Central configuration for the Captain Voice Assistant.

All settings are read from environment variables (loaded from a local
.env file if present) so the same code runs unmodified across dev,
CI, and demo environments. Nothing here requires a specific provider —
every external integration (LLM, translation, TTS) is optional/pluggable
and the app degrades gracefully when a key is missing (see app/tts/factory.py
and app/llm/generator.py for the fallback behaviour).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a thin convenience only
    pass

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- Paths -----------------------------------------------------------
    base_dir: Path = BASE_DIR
    knowledge_base_dir: Path = BASE_DIR / "data" / "knowledge_base"
    index_dir: Path = BASE_DIR / "data" / "index"
    audio_output_dir: Path = BASE_DIR / "audio_output"
    logs_dir: Path = BASE_DIR / "logs"

    # --- LLM (generation + LLM-based translation) -------------------------
    # LLM_PROVIDER=auto (default) picks Groq if GROQ_API_KEY is set, else
    # Anthropic if ANTHROPIC_API_KEY is set, else falls back to a local
    # Ollama server (no key/signup needed at all). Set explicitly to force
    # one backend: groq | anthropic | ollama. See app/llm/providers.py.
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "600")))

    # Groq (free tier -- https://console.groq.com/keys, no card required)
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    # Caps hidden chain-of-thought token spend on reasoning models (e.g.
    # openai/gpt-oss-*) so they don't burn the whole max_tokens budget on
    # reasoning and return an empty answer. low|medium|high, or "" to omit
    # the field entirely (for a Groq model that doesn't support it).
    groq_reasoning_effort: str = field(default_factory=lambda: os.getenv("GROQ_REASONING_EFFORT", "low"))

    # Anthropic / Claude (optional, pluggable alternative)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"))

    # Ollama (fully local, no key/signup -- https://ollama.com; the guaranteed fallback)
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1"))

    # --- Retrieval ---------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "4")))
    # Cosine-similarity floor (index uses normalized vectors + inner product).
    # Below this we treat the query as "not covered by the knowledge base"
    # rather than let the LLM improvise, to avoid ungrounded answers.
    relevance_threshold: float = field(
        default_factory=lambda: float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
    )

    # --- Translation ---------------------------------------------------------
    default_target_lang: str = field(default_factory=lambda: os.getenv("DEFAULT_TARGET_LANG", "am"))

    # --- TTS ("Captain" voice) ---------------------------------------------
    tts_engine: str = field(default_factory=lambda: os.getenv("TTS_ENGINE", "auto"))
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", ""))
    elevenlabs_model_id: str = field(
        default_factory=lambda: os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    )
    azure_speech_key: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_KEY", ""))
    azure_speech_region: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_REGION", ""))
    local_voice_name: str = field(default_factory=lambda: os.getenv("LOCAL_VOICE_NAME", ""))
    local_voice_rate: int = field(default_factory=lambda: int(os.getenv("LOCAL_VOICE_RATE", "170")))

    # --- Server --------------------------------------------------------------
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    debug: bool = field(default_factory=lambda: _bool("DEBUG", False))

    def ensure_dirs(self) -> None:
        for d in (self.index_dir, self.audio_output_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
