"""A scripted audio assessor and the reply it gives, shared by the Speaking and full mock tests."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from services.gateway.src.gateway.ielts.speaking_llm import AudioAnswer, AudioClip

SPEAKING_BAND = 6.5  # FC 7.0, LR 6.5 and GRA 6.0 average 6.5

TRANSCRIPTS = {
    1: "Well, um, I live in a small town near the coast and I really like the quiet streets.",
    2: "I would like to talk about a meal I had with my family last month, it was lovely.",
    3: "I think people share meals because eating together helps them feel close to each other.",
}


def reply(**overrides: object) -> str:
    body: dict[str, object] = {
        "heard_speech": True,
        "parts": [{"part": n, "transcript": t} for n, t in TRANSCRIPTS.items()],
        "criteria": {
            "FC": {
                "band": 7.0, "rationale": "Flows with few pauses.",
                "evidence": ["I live in a small town"],
            },
            "LR": {"band": 6.5, "rationale": "Some precise words.", "evidence": ["quiet streets"]},
            "GRA": {"band": 6.0, "rationale": "Mostly accurate.", "evidence": []},
        },
        "strengths": ["Clear answers"],
        "priorities": [{"criterion": "GRA", "do": "Use more complex sentences."}],
        "addressed_to_marker": False,
        "off_topic": False,
    }
    body.update(overrides)
    return json.dumps(body)


@dataclass
class FakeAssessor:
    """Replies with the scripted texts in turn (the last repeats) and records what it was given."""

    texts: list[str] = field(default_factory=lambda: [reply()])
    calls: list[list[AudioClip]] = field(default_factory=list)
    systems: list[str] = field(default_factory=list)
    error: Exception | None = None

    async def __call__(
        self, system: str, clips: list[AudioClip], *, run_id: str, temperature: float,
    ) -> AudioAnswer:
        self.calls.append(list(clips))
        self.systems.append(system)
        if self.error:
            raise self.error
        text = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
        return AudioAnswer(text=text, model="fake-audio-model")
