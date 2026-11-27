"""
Build the Listening recordings once, with Gemini text-to-speech, and store them in the repository.

    GEMINI_API_KEY=... uv run python -m scripts.build_listening_audio           # rebuild stale
    GEMINI_API_KEY=... uv run python -m scripts.build_listening_audio --force   # rebuild all
    uv run python -m scripts.build_listening_audio --check                      # no network

This is the only place a model speaks. A learner's attempt never calls it: the gateway serves the
mp3 files written here (`ielts/data/listening_audio/`) and `ielts/data/listening_audio.json` says
which script each file was made from, so only sections whose script changed are rebuilt. Needs
`ffmpeg` on the path to turn the model's raw PCM into a small mono mp3.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from typing import Any

import httpx

from services.gateway.src.gateway.ielts.listening import (
    AUDIO_DIR,
    MANIFEST_PATH,
    ListeningSet,
    Section,
    fingerprint,
    get_set,
    tts_text,
)

MODEL = "gemini-2.5-flash-preview-tts"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
PCM_RATE = 24000  # the model returns 16-bit mono PCM at 24 kHz
MP3_KBPS = 32
ATTEMPTS = 4


def _voice(name: str) -> dict[str, Any]:
    return {"prebuiltVoiceConfig": {"voiceName": name}}


def _speech_config(section: Section) -> dict[str, Any]:
    if len(section.speakers) == 1:
        return {"voiceConfig": _voice(section.speakers[0].voice)}
    return {"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": [
        {"speaker": s.name, "voiceConfig": _voice(s.voice)} for s in section.speakers
    ]}}


def synthesise(paper: ListeningSet, section: Section, key: str) -> bytes:
    """Raw 16-bit PCM for one section."""
    body = {
        "contents": [{"parts": [{"text": tts_text(paper.style, section)}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"], "speechConfig": _speech_config(section),
        },
    }
    last = ""
    for attempt in range(ATTEMPTS):
        resp = httpx.post(URL, json=body, headers={"x-goog-api-key": key}, timeout=300)
        if resp.status_code == 200:
            try:
                parts = resp.json()["candidates"][0]["content"]["parts"]
                return base64.b64decode(parts[0]["inlineData"]["data"])
            except (KeyError, IndexError, ValueError):
                last = "the reply had no audio"
        else:
            last = f"HTTP {resp.status_code}"
        print(f"  {section.id}: {last}; retrying", file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f"{section.id}: text-to-speech failed ({last})")


def to_mp3(pcm: bytes, out_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(PCM_RATE), "-ac", "1",
         "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", f"{MP3_KBPS}k", out_path],
        input=pcm, check=True,
    )


def load_manifest() -> dict[str, Any]:
    try:
        manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"model": MODEL, "sections": {}}
    return manifest


def stale(paper: ListeningSet, manifest: dict[str, Any]) -> list[Section]:
    out = []
    for s in paper.sections:
        entry = manifest.get("sections", {}).get(s.id)
        if not entry or entry.get("sha256") != fingerprint(paper.style, s) or not (
            AUDIO_DIR / entry["file"]
        ).is_file():
            out.append(s)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="rebuild every section")
    parser.add_argument("--check", action="store_true", help="only report stale sections")
    args = parser.parse_args()

    paper = get_set()
    manifest = load_manifest()
    todo = list(paper.sections) if args.force else stale(paper, manifest)
    if args.check:
        print("stale: " + (", ".join(s.id for s in todo) or "none"))
        return 1 if todo else 0
    if not todo:
        print("every recording is up to date")
        return 0
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        return 2

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    manifest["model"] = MODEL
    entries = manifest.setdefault("sections", {})
    for s in todo:
        print(f"building {s.id}: {s.title}")
        pcm = synthesise(paper, s, key)
        name = f"{s.id}.mp3"
        to_mp3(pcm, str(AUDIO_DIR / name))
        entries[s.id] = {
            "file": name, "sha256": fingerprint(paper.style, s),
            "duration_s": round(len(pcm) / (2 * PCM_RATE), 1),
            "voices": {sp.name: sp.voice for sp in s.speakers},
        }
        # Written after every section, so a failure later keeps the sections already built.
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
