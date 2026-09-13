# Captain Voice Assistant — RAG-Based Translation

A prototype voice assistant for a ship's Captain: type a question in
English, get an answer **grounded in a ship operations manual** via RAG,
**translated** into a target language (default: Amharic), and **spoken
back** in a consistent "Captain" voice. Runs entirely on **free
services** (a local Ollama model, or Groq's free API tier + local
embeddings + offline TTS) -- Claude/ElevenLabs/Azure are supported as
pluggable, higher-quality upgrades, not requirements.

## Requirements checklist

Mapping the assignment's core requirements and deliverables to what's in
this repo, for quick grading/verification.

| Requirement | Status | Where |
|---|---|---|
| Text input (CLI / web UI / API) | ✅ | [app/cli.py](app/cli.py), [static/](static/), [app/main.py](app/main.py) |
| ≥1 non-English language pair | ✅ | English → Amharic default; also fr/es/ar/sw/de/pt/hi/zh — [app/translation/translator.py](app/translation/translator.py) |
| Knowledge base, 10–50 docs | ✅ | 20 docs — [data/knowledge_base/](data/knowledge_base/) |
| Vector store + embedding model | ✅ | FAISS + sentence-transformers — [app/rag/retriever.py](app/rag/retriever.py) |
| Grounded, non-hallucinated responses | ✅ | Relevance threshold gate + citation-checked prompt — [Anti-hallucination design](#anti-hallucination-design) |
| Translation before speech, choice justified | ✅ | [Tools/APIs used, and why](#toolsapis-used-and-why) |
| TTS / consistent Captain voice, tradeoff explained if not true cloning | ✅ | [Voice synthesis](#voice-synthesis-the-captains-voice) |
| Audio returned/played | ✅ | Web UI player, CLI saves file, `GET /api/audio/{file}` |
| Full pipeline trace logged | ✅ | [Pipeline trace / logging](#pipeline-trace--logging) |
| Source code in a repo | ✅ | this repo (see [Repository](#repository) below) |
| README: architecture, setup, tools+why, limitations | ✅ | this file |
| Sample knowledge base included | ✅ | [data/knowledge_base/](data/knowledge_base/) |
| Demo video (2-5 min) | ⚠️ **action needed** | script ready at [demo/DEMO_SCRIPT.md](demo/DEMO_SCRIPT.md); recording requires a live screen/audio capture, which has to be done by a human |
| Automated tests | ✅ (bonus, not required) | 34 tests, `pytest` — see [Tests](#tests) |

### Repository

This project is a local git repository (one commit at the time of
writing). To submit it as required ("GitHub repo link"), push it to
GitHub:
```bash
gh repo create captain-voice-assistant --private --source=. --push
# or, without the GitHub CLI:
#   create an empty repo on github.com, then:
git remote add origin https://github.com/<you>/captain-voice-assistant.git
git branch -M main
git push -u origin main
```

## Table of contents

- [Requirements checklist](#requirements-checklist)
- [Architecture](#architecture)
- [Setup & run](#setup--run)
- [Tools/APIs used, and why](#toolsapis-used-and-why)
- [Anti-hallucination design](#anti-hallucination-design)
- [Knowledge base](#knowledge-base)
- [Pipeline trace / logging](#pipeline-trace--logging)
- [Tests](#tests)
- [Known limitations & what I'd improve](#known-limitations--what-id-improve-with-more-time)
- [Demo](#demo)

## Architecture

```
┌────────────┐     text query      ┌───────────────────┐
│   Captain   │ ──────────────────▶│   Input Layer      │
│ (CLI / Web) │                    │  CLI · FastAPI ·UI │
└────────────┘                    └─────────┬──────────┘
                                             │
                                             ▼
                                   ┌───────────────────────┐
                                   │   1. RAG Retrieval      │
                                   │  app/rag/               │
                                   │  • chunk KB docs         │
                                   │  • embed (sentence-      │
                                   │    transformers)         │
                                   │  • FAISS cosine search    │
                                   │  • relevance threshold    │
                                   └─────────┬───────────────┘
                                             │ top-k chunks
                                             ▼
                                   ┌───────────────────────┐
                                   │  2. Grounded Generation │
                                   │  app/llm/generator.py    │
                                   │  Groq > Anthropic >       │
                                   │  Ollama (auto-selected),  │
                                   │  system prompt: "answer   │
                                   │  only from CONTEXT, cite  │
                                   │  sources" (no chunks ->   │
                                   │  canned "not covered"     │
                                   │  answer, LLM never called)│
                                   └─────────┬───────────────┘
                                             │ English answer
                                             ▼
                                   ┌───────────────────────┐
                                   │  3. Translation          │
                                   │  app/translation/        │
                                   │  same LLM provider,       │
                                   │  dedicated translation    │
                                   │  prompt + maritime        │
                                   │  glossary                 │
                                   └─────────┬───────────────┘
                                             │ translated text
                                             ▼
                                   ┌───────────────────────┐
                                   │  4. Speech Synthesis     │
                                   │  app/tts/                │
                                   │  ElevenLabs (clone) >     │
                                   │  Azure Speech >           │
                                   │  pyttsx3 (local, offline) │
                                   └─────────┬───────────────┘
                                             │ audio file
                                             ▼
                                   ┌───────────────────────┐
                                   │  Output: audio + trace   │
                                   │  logs/traces/<id>.json    │
                                   │  logs/pipeline_trace.jsonl│
                                   └───────────────────────┘
```

`app/pipeline.py` is the only module that knows about all four stages;
each stage module (`rag/`, `llm/`, `translation/`, `tts/`) is independent,
swappable, and unit-tested in isolation with fakes/mocks (see
[Tests](#tests)) — this is the "clean separation of concerns" the
assignment asks for. The LLM stage itself is pluggable one level deeper:
`app/llm/providers.py` defines a small `LLMProvider` interface with three
implementations — `GroqProvider` (free cloud API), `OllamaProvider`
(fully local, no signup), and `AnthropicProvider` (optional, paid) —
selected by `get_llm_provider()`, mirroring the same cloud→cloud→local
fallback pattern already used for TTS (`app/tts/factory.py`). Both the
generator and translator use whichever provider resolves, so switching
providers is a one-line `.env` change, not a code change.

## Setup & run

Requires Python 3.10+ (developed/tested on 3.14).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure an LLM backend (pick ONE -- all free or effectively free):
cp .env.example .env
#   a) Groq (cloud, free, no card):    set GROQ_API_KEY=gsk_...
#   b) Ollama (fully local, no signup): install https://ollama.com/download,
#      run `ollama pull llama3.1`, then set LLM_PROVIDER=ollama in .env
#   c) Anthropic/Claude (paid, optional): set ANTHROPIC_API_KEY=sk-ant-...

# 3. Build the vector index from the knowledge base
python scripts/build_index.py

# 4. Run
python -m app.cli ask "What do I do if a crew member falls overboard?" --lang am
#   ...or start the web UI / API:
uvicorn app.main:app --reload
# then open http://localhost:8000
```

Only one LLM backend needs to be configured to run the whole pipeline
end-to-end — generation, translation, and (via the local pyttsx3
fallback) speech synthesis all work with just that. `LLM_PROVIDER=auto`
(the default) picks `GROQ_API_KEY` if set, else `ANTHROPIC_API_KEY` if
set, else falls back to a locally-running **Ollama** server — no key, no
account, no CAPTCHA of any kind, just `ollama pull llama3.1` once (see
https://ollama.com/download). See `app/llm/providers.py:get_llm_provider()`
for the selection logic. Adding `ELEVENLABS_API_KEY`/`ELEVENLABS_VOICE_ID` or
`AZURE_SPEECH_KEY`/`AZURE_SPEECH_REGION` upgrades the voice quality and
language coverage — see [Voice synthesis](#voice-synthesis-the-captains-voice)
below — with no code changes; `app/tts/factory.py` picks the best
available backend automatically.

### CLI

```bash
python -m app.cli ask "How much chain scope do I use when anchoring?" --lang en
python -m app.cli ask "What's the MAYDAY call format?" --lang am --json
python -m app.cli ask "Quick check, no audio" --lang fr --no-audio
```

### Web UI / API

```bash
uvicorn app.main:app --reload
```
- `http://localhost:8000/` — minimal web UI (textarea, language picker,
  answer, translated text, audio player, retrieved-chunks panel, trace id)
- `POST /api/query {"text": "...", "target_lang": "am"}` — the same
  pipeline as an API endpoint
- `GET /api/audio/{filename}` — serves generated audio

### Tests

```bash
pytest
```
34 tests covering chunking, retrieval/ranking (deterministic fake
embedder), the Groq/Anthropic LLM providers and provider-selection
factory, grounded generation (incl. the no-hallucination short-circuit
and clean error handling on API failures), translation, and full pipeline
orchestration/trace logging — all mocked at the network boundary, so they
run in a few seconds with no API keys or model downloads required.

## Tools/APIs used, and why

| Stage | Choice | Why |
|---|---|---|
| LLM | **Groq (Llama 3.3 70B)**, **Ollama (fully local Llama 3.1)**, or **Claude** — three interchangeable providers (`app/llm/providers.py`) | Groq's free tier (no credit card) and a local Ollama model both make the pipeline runnable at zero cost; Ollama additionally needs no web account/key/CAPTCHA at all, so it's the guaranteed fallback when a cloud signup isn't an option. `LLM_PROVIDER=anthropic` switches to Claude with no code change, for anyone who wants its somewhat stronger multilingual/low-resource-language output instead. Both stages (generation + translation) share whichever provider is active — one config either way. |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Local, free, no extra API key, fast (~90MB, CPU-friendly). Deliberately small for a 10-50 document knowledge base; swap `EMBEDDING_MODEL` for a larger model if the corpus grows. |
| Vector store | **FAISS** (`IndexFlatIP` over normalized vectors = cosine similarity) | The suggested, standard choice. Brute-force is overkill-proof at this corpus size and is a drop-in path to scale later without touching calling code. Index is persisted to `data/index/` so it's not rebuilt on every boot. |
| Translation | **LLM-based (same provider as generation)**, not Google Translate/DeepL | (1) One fewer API key to provision — reuses the generation provider. (2) **DeepL does not support Amharic at all**, and general MT engines are historically weak on it; large instruction-tuned LLMs handle low-resource languages meaningfully better because they translate meaning/register rather than word-by-word. (3) Lets us pass a maritime glossary in the prompt so domain terms (MAYDAY, COLREGs, muster station...) stay consistent instead of being awkwardly transliterated. Trade-off: higher per-call latency/cost than a dedicated MT API, and no formal BLEU-style quality guarantee — acceptable for a prototype at this scale; see [limitations](#known-limitations--what-id-improve-with-more-time) for the Groq-vs-Claude Amharic quality trade-off specifically. |
| TTS / "Captain" voice | **ElevenLabs → Azure Speech → pyttsx3**, auto-selected | See below. |

### Voice synthesis (the Captain's voice)

Three backends behind one interface (`app/tts/base.py`), auto-selected by
`app/tts/factory.py` based on which credentials are present:

1. **ElevenLabs** — true voice cloning *on a paid plan*: `scripts/clone_voice.py`
   uploads a few audio samples of the Captain and returns a `voice_id`;
   every response after that is spoken in that cloned voice. **Found
   empirically while building this** (not documented clearly by
   ElevenLabs): a **free-tier** key cannot use Instant Voice Cloning via
   the API at all (`400 can_not_use_instant_voice_cloning`), and also
   cannot use Voice *Library* voices via the API (`402 paid_plan_required`)
   even after adding one to "My Voices". What a free key *can* use is one
   of the ~20 stock **"premade"** voices bundled with every account
   (`GET /v1/voices`, filter `"category": "premade"`) — that's the
   assignment's "well-configured single consistent voice profile"
   fallback, not true cloning, despite going through the same backend
   code. Either way, its multilingual model does not officially cover
   Amharic.
2. **Azure Speech** — not cloning, but a fixed, well-configured neural
   voice per language (`app/tts/azure_tts.py`), including dedicated
   **am-ET (Amharic) voices** that ElevenLabs lacks. This is the
   assignment's explicitly-accepted fallback: *"a well-configured single
   consistent voice profile is acceptable"* — recommended specifically
   for Amharic output.
3. **pyttsx3 (local/offline)** — the guaranteed fallback so the pipeline
   never fails to produce audio even with zero paid services configured.
   Uses one deterministically-chosen installed voice for a consistent
   persona. **Limitation:** stock Windows SAPI voices only reliably cover
   English — non-English text will be mispronounced. The engine reports
   this via a `warning` on the result, surfaced in the API/UI, rather
   than silently producing bad audio.

This three-tier design means the assignment's core trade-off (cloning vs.
a consistent branded voice) is not an either/or choice made once — it's
resolved per-deployment based on which credentials are available, and the
system is honest in its output about the gap when it falls back.

## Anti-hallucination design

RAG "grounding" is enforced at two layers, not just prompted for:

1. **Retrieval gate**: `Retriever.retrieve_relevant()` drops any chunk
   below `RELEVANCE_THRESHOLD` (cosine similarity, default 0.30). If
   *nothing* clears the bar, the generator is never even called —
   `ResponseGenerator.generate()` short-circuits to a fixed "not covered
   by the manual" message. An empty-context question literally cannot
   reach the LLM, which is a stronger guarantee than prompting alone.
2. **Prompt constraint**: when there *is* context, the system prompt
   (`app/llm/generator.py:SYSTEM_PROMPT`) instructs the model to answer
   only from the provided excerpts, say so if the context is
   insufficient, and cite the source document titles it used — citations
   are parsed out and returned/logged separately (`result.citations`),
   so grounding is auditable per response, not just claimed.

## Knowledge base

`data/knowledge_base/` — 20 short Markdown documents (one topic each)
from a **merchant ship operations manual**: bridge watchkeeping, COLREGs,
man overboard, fire/abandon-ship emergencies, anchoring, mooring, engine
room watch, ballast management, cargo loading, heavy weather, piracy/
security (ISPS), radio/distress protocols (GMDSS), life-saving equipment,
MARPOL pollution prevention, pilot boarding, crew rest hours, medical
emergencies, port entry, drills, and fuel bunkering. Each file has a
small frontmatter block (`id`, `title`, `tags`) parsed by
`app/rag/knowledge_base.py`, then chunked on paragraph boundaries
(`chunk_document`, default max 700 chars — most documents here stay a
single, fully self-contained/citable chunk).

## Pipeline trace / logging

Every run writes a structured trace covering all five stages
(input → retrieval → generation → translation → synthesis):
- `logs/traces/<trace_id>.json` — one file per run, easy to inspect
- `logs/pipeline_trace.jsonl` — append-only log of every run
- the same `trace_id` is returned in the API/CLI response and shown in
  the web UI, so a specific answer can always be traced back to exactly
  which chunks were retrieved and what was generated/translated.

## Known limitations & what I'd improve with more time

- **Llama-based translation quality (Groq or Ollama) is a step below
  Claude's for a low-resource language like Amharic.** Both are the
  default/fallback specifically because they're free — Groq needs no
  card, Ollama needs no signup at all — which matters more for a
  prototype/take-home than squeezing out the last bit of fluency, but
  Llama 3.x's Amharic output is less polished than Claude's. Setting
  `ANTHROPIC_API_KEY` (`LLM_PROVIDER=anthropic`, or leave `auto`) switches
  to Claude with no code change if translation fidelity matters more than
  cost for a given deployment. Ollama's *local* model size/quality also
  depends entirely on what you `ollama pull` — a small quantized model
  will ground and translate noticeably worse than Groq's hosted 70B.
- **Amharic TTS quality depends on the backend.** The always-available
  local fallback (pyttsx3/SAPI) cannot pronounce Amharic correctly; Azure
  Speech's `am-ET` neural voices are the accurate option, and neither
  ElevenLabs cloning nor Azure's fixed voice is a like-for-like
  replacement for a cloned voice actually speaking Amharic (Azure's
  Custom Neural Voice program, which would enable that, requires a
  longer approval/training process out of scope here).
- **Chunking is paragraph-based, not semantic.** Fine for this
  20-document manual where most docs are already one coherent chunk; a
  larger/more heterogeneous corpus would benefit from semantic chunking
  and possibly a re-ranking step after FAISS retrieval.
- **No conversation memory.** Each query is independent; a natural next
  step is a short rolling context so the Captain can ask follow-ups
  ("and how long does that take?") without repeating context.
- **Translation isn't benchmarked.** LLM-based translation was chosen and
  justified above, but I did not have a native Amharic speaker available
  to score fluency/accuracy against a reference — the README claim is a
  design justification, not a measured quality result.
- **Retrieval threshold is a single global constant.** `0.30` was tuned
  by eyeballing a handful of queries (see the smoke test below); a larger
  knowledge base would want per-domain calibration or a learned
  threshold instead of one fixed cosine cutoff.
- **Single-turn, single-user.** No auth, no persistence beyond the
  filesystem logs/index — appropriate for a prototype, not for a
  multi-Captain fleet deployment.
- **Retrieval smoke test** (real embeddings, run via
  `python -c "from app.rag.retriever import Retriever; ..."`, see repo
  history) correctly ranked "Man Overboard Procedure" top for *"What do I
  do if a crew member falls overboard?"* (score 0.54) and "Anchoring
  Procedures" top for *"How much chain scope should I use when
  anchoring?"* (score 0.67) — included here since it's not captured by
  the mocked unit tests.

## Demo

See [`demo/DEMO_SCRIPT.md`](demo/DEMO_SCRIPT.md) for a suggested 2-5
minute walkthrough script (text query in → retrieved chunks → grounded
English answer → Amharic translation → spoken audio out), covering both
the CLI and the web UI.
