"""Create a cloned "Captain" voice on ElevenLabs from local audio samples.

Usage:
    python scripts/clone_voice.py "Captain Voice" sample1.wav sample2.wav

Requires ELEVENLABS_API_KEY in the environment/.env. On success, prints the
new voice_id -- put it in .env as ELEVENLABS_VOICE_ID.

Provide 1-25 clean audio samples (a few minutes total is usually enough
for a recognizable clone; ElevenLabs recommends WAV/MP3, minimal
background noise). The recording's language doesn't need to match the
language you'll later synthesize -- cloning captures voice characteristics
(tone/timbre), not words; ElevenLabs' multilingual model handles the
language switch at synthesis time (see app/tts/elevenlabs_tts.py for that
model's language coverage/limitations).

IMPORTANT (found empirically, not just from the docs): this Voice Add
endpoint requires a **paid** ElevenLabs plan. A free-tier key gets a 400
"can_not_use_instant_voice_cloning" error. If you're on the free tier,
skip this script and instead set ELEVENLABS_VOICE_ID to one of your
account's bundled "premade" voices -- list them with:
    GET https://api.elevenlabs.io/v1/voices  (header: xi-api-key)
and pick one with "category": "premade" (voices from the Voice Library,
category "professional"/etc., also require a paid plan to use via the
API even once added to "My Voices" -- only "premade" works free). That
gives the assignment's explicitly-accepted fallback -- a well-configured
consistent voice profile -- instead of true cloning.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from app.config import settings

ADD_VOICE_URL = "https://api.elevenlabs.io/v1/voices/add"


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python scripts/clone_voice.py <voice_name> <sample1.wav> [sample2.wav ...]", file=sys.stderr)
        return 1

    if not settings.elevenlabs_api_key:
        print("ELEVENLABS_API_KEY is not set.", file=sys.stderr)
        return 1

    voice_name = sys.argv[1]
    sample_paths = [Path(p) for p in sys.argv[2:]]
    for p in sample_paths:
        if not p.exists():
            print(f"Sample file not found: {p}", file=sys.stderr)
            return 1

    files = [("files", (p.name, p.open("rb"), "audio/mpeg")) for p in sample_paths]
    data = {"name": voice_name, "description": "Cloned Captain voice for the voice assistant demo"}

    resp = requests.post(
        ADD_VOICE_URL,
        headers={"xi-api-key": settings.elevenlabs_api_key},
        data=data,
        files=files,
        timeout=120,
    )
    for _, (_, fh, _) in files:
        fh.close()

    if resp.status_code != 200:
        print(f"Voice clone request failed ({resp.status_code}): {resp.text}", file=sys.stderr)
        return 1

    voice_id = resp.json().get("voice_id")
    print(f"Created cloned voice '{voice_name}' -> voice_id={voice_id}")
    print("Add this to your .env file:")
    print(f"ELEVENLABS_VOICE_ID={voice_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
