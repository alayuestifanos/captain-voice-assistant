"""Command-line entry point for the Captain Voice Assistant.

Usage:
  python -m app.cli ask "What do I do if a man goes overboard?" --lang am
  python -m app.cli ask "Give me the anchoring depth ratio" --lang en --no-audio
"""
from __future__ import annotations

import argparse
import json
import sys

from app.pipeline import CaptainAssistantPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="captain-assistant", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask the Captain assistant a question")
    ask.add_argument("text", help="The Captain's text command/question")
    ask.add_argument("--lang", "--target-lang", dest="lang", default=None, help="Target language code (default: am)")
    ask.add_argument("--no-audio", action="store_true", help="Skip speech synthesis (text-only output)")
    ask.add_argument("--json", action="store_true", help="Print the full pipeline result as JSON")

    args = parser.parse_args(argv)

    if args.command == "ask":
        pipeline = CaptainAssistantPipeline()
        try:
            result = pipeline.run(args.text, target_lang=args.lang, synthesize_audio=not args.no_audio)
        except RuntimeError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(result.to_api_dict(), indent=2, ensure_ascii=False))
            return 0

        print(f"\nTrace ID: {result.trace_id}")
        print(f"\nQuery: {result.query}")
        print("\nRetrieved sources:")
        for sc in result.retrieved:
            print(f"  - [{sc.score:.3f}] {sc.chunk.title}")
        print(f"\nAnswer (EN): {result.generation.answer}")
        if result.generation.citations:
            print(f"Citations: {', '.join(result.generation.citations)}")
        print(f"\nTranslated ({result.target_lang}): {result.translation.text}")
        if result.synthesis:
            print(f"\nAudio saved to: {result.synthesis.audio_path}")
            print(f"TTS engine: {result.synthesis.engine} (voice: {result.synthesis.voice_label})")
        for w in result.warnings:
            print(f"\n[warning] {w}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
