# Demo walkthrough script (2-5 min)

Goal: show a text command going in and voice output coming out, with the
RAG grounding and translation visible in between.

## Setup (before recording)

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in GROQ_API_KEY at minimum (free -- console.groq.com/keys)
python scripts/build_index.py
uvicorn app.main:app --reload
```
Open `http://localhost:8000`.

## Suggested flow

1. **Intro (15s)** — "This is a Captain voice assistant. It answers from
   a ship operations manual using RAG, translates the answer, and speaks
   it back in a consistent Captain voice. It runs on free-tier services
   by default: Groq for generation/translation, ElevenLabs' free tier
   for speech -- Claude and Azure are supported as pluggable upgrades."

2. **Show the knowledge base (15s)** — briefly show `data/knowledge_base/`
   in the editor: 20 short docs, one topic each (man overboard, fire,
   anchoring, COLREGs, etc.).

3. **Ask a grounded question (60-90s)** — in the web UI, type:
   > "What do I do if a crew member falls overboard?"

   Select target language **Amharic**, click **Ask**. Narrate as it
   returns:
   - the retrieved chunks panel (open the "Retrieved context" details) —
     point out "Man Overboard Procedure" was retrieved with the highest
     score
   - the English answer, and the `Sources:` citation line
   - the Amharic translation text
   - press play on the audio player — the spoken Amharic response

4. **Show the anti-hallucination guardrail (30-45s)** — ask something the
   manual doesn't cover, e.g.:
   > "What's the weather going to be like next week?"

   Show that the answer says it isn't covered by the manual rather than
   making something up, and that `grounded: false` / the warning banner
   appears — this demonstrates the retrieval-threshold gate, not just a
   prompted instruction.

5. **Show the pipeline trace (20s)** — open `logs/traces/<trace_id>.json`
   (the trace id shown at the bottom of the result) to show the full
   input → retrieval → generation → translation → synthesis log.

6. **CLI alternative (optional, 15s)** — show the same thing from the
   terminal:
   ```bash
   python -m app.cli ask "How much chain scope do I use when anchoring?" --lang am
   ```

7. **Wrap-up (15s)** — mention the LLM fallback chain (Groq free tier →
   Anthropic Claude → local Ollama) and the TTS fallback chain (ElevenLabs
   → Azure Amharic neural voice → local pyttsx3) -- both configurable with
   no code changes -- and point to the README limitations section.

## Recording tips

- Keep the terminal/browser font large enough to read on video.
- If the audio warning banner appears ("'am' is not an officially
  supported language for eleven_multilingual_v2" or the local-pyttsx3
  equivalent), mention on camera that this is a documented, known
  tradeoff -- Azure's dedicated am-ET neural voices are the path to
  guaranteed-correct Amharic pronunciation (see README), and the
  assignment explicitly accepts a well-configured consistent voice
  profile as a fallback when that isn't set up.
