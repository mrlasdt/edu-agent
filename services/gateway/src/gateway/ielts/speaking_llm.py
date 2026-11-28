"""
Audio in, text out: the one call the Speaking mock makes to a model.

The other IELTS roles go through the LiteLLM Router, which cannot carry audio to Gemma or to the
OpenAI chat models. This client posts the recordings straight to the Gemini REST API
(`models/<model>:generateContent`, the recording as `inlineData`, the same call
`scripts/build_listening_audio.py` makes for speech), using the `ielts_speaking` role of the model
config for the model and its fallbacks. Gemini only, as decided for Speaking: if the key or the
role is missing, the call fails with `ModelUnavailableError` (a retryable failure the learner can
re-run) and no other provider is tried.

A busy model (HTTP 429, 500 or 503, which Google returns often for audio) is retried with a short
backoff, then the next model in the chain is tried. Spend goes through the same ledger as the
other IELTS calls: the worst case is checked before the call (`BudgetExceededError`, not
wrapped) and the reported usage is recorded after it.
"""
from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from services.gateway.src.gateway.guarded_completion import failure_reason
from services.gateway.src.gateway.ielts.llm import ModelUnavailableError
from shared.src.shared.budget import Ledger, estimate_cost, spend_ledger
from shared.src.shared.config import get_llm_settings, load_model_config

SPEAKING_ROLE = "ielts_speaking"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
# Gemini reads audio at about 32 tokens a second, and 1 second of audio is what the ledger
# is charged for, so a long clip cannot be under-priced.
AUDIO_TOKENS_PER_S = 32
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
ATTEMPTS_PER_MODEL = 3
BACKOFF_S = (1.0, 3.0)
TIMEOUT_S = 120.0
MAX_OUTPUT_TOKENS = 6000


@dataclass(frozen=True)
class AudioClip:
    label: str  # e.g. "Part 1" with the questions: sent as text just before the audio
    data: bytes
    mime: str
    duration_s: float


@dataclass(frozen=True)
class AudioAnswer:
    text: str
    model: str  # the model that answered


class AudioAssessor(Protocol):
    async def __call__(
        self, system: str, clips: Sequence[AudioClip], *, run_id: str, temperature: float,
    ) -> AudioAnswer: ...


def _bare(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("gemini/") else model


def _chain() -> list[str]:
    config = load_model_config(get_llm_settings().model_config_path)
    role = config.role(SPEAKING_ROLE)
    if role.provider != "gemini":
        raise ModelUnavailableError(
            f"Speaking needs a Gemini model that hears audio, but {SPEAKING_ROLE} is "
            f"{role.provider}/{role.model}"
        )
    models = [role.model, *[m for m in role.fallbacks if m.startswith("gemini/")]]
    return [_bare(m) for m in models]


def _text_of(body: dict[str, Any]) -> str:
    candidates = body.get("candidates") or []
    parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
    # parts can include a thought signature as well as text: join only the text ones
    return "".join(
        p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)
    )


class GeminiAudioAssessor:
    def __init__(
        self, ledger: Ledger | None = None, transport: httpx.AsyncBaseTransport | None = None,
        api_key: str | None = None, sleep: Any = asyncio.sleep,
    ) -> None:
        self._ledger = ledger
        self._transport = transport
        self._api_key = api_key
        self._sleep = sleep

    @property
    def ledger(self) -> Ledger:
        return self._ledger or spend_ledger()

    def _key(self) -> str:
        key = self._api_key if self._api_key is not None else get_llm_settings().gemini_api_key
        if not key:
            raise ModelUnavailableError("GEMINI_API_KEY is not set, so audio cannot be marked")
        return key

    def _projected(self, model: str, system: str, clips: Sequence[AudioClip]) -> float:
        audio = sum(int(c.duration_s) + 1 for c in clips) * AUDIO_TOKENS_PER_S
        prompt = audio + (len(system) + sum(len(c.label) for c in clips)) // 4
        return estimate_cost(model, prompt, MAX_OUTPUT_TOKENS)

    async def __call__(
        self, system: str, clips: Sequence[AudioClip], *, run_id: str, temperature: float,
    ) -> AudioAnswer:
        key = self._key()
        chain = _chain()
        parts: list[dict[str, Any]] = []
        for clip in clips:
            parts.append({"text": clip.label})
            parts.append({"inlineData": {
                "mimeType": clip.mime, "data": base64.b64encode(clip.data).decode("ascii"),
            }})
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature, "maxOutputTokens": MAX_OUTPUT_TOKENS,
                "responseMimeType": "application/json",
            },
        }
        # the dearest model in the chain bounds the whole call, whichever one answers
        self.ledger.ensure_capacity(max(self._projected(m, system, clips) for m in chain))
        errors: list[str] = []
        async with httpx.AsyncClient(timeout=TIMEOUT_S, transport=self._transport) as client:
            for model in chain:
                answer = await self._one_model(client, key, model, payload, run_id, errors)
                if answer is not None:
                    return answer
        raise ModelUnavailableError("; ".join(errors) or "no model answered")

    async def _one_model(
        self, client: httpx.AsyncClient, key: str, model: str, payload: dict[str, Any],
        run_id: str, errors: list[str],
    ) -> AudioAnswer | None:
        for attempt in range(ATTEMPTS_PER_MODEL):
            try:
                resp = await client.post(
                    f"{API_ROOT}/{model}:generateContent", json=payload,
                    headers={"x-goog-api-key": key},
                )
            except httpx.HTTPError as exc:
                errors.append(f"{model}: {failure_reason(exc, 120)}")
            else:
                if resp.status_code == 200:
                    body = resp.json()
                    self._record(run_id, model, body)
                    text = _text_of(body)
                    if text.strip():
                        return AudioAnswer(text=text, model=model)
                    errors.append(f"{model}: empty reply")
                    return None
                errors.append(f"{model}: HTTP {resp.status_code}")
                if resp.status_code not in RETRY_STATUS:
                    return None
            if attempt < ATTEMPTS_PER_MODEL - 1:
                await self._sleep(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])
        return None

    def _record(self, run_id: str, model: str, body: dict[str, Any]) -> None:
        usage = body.get("usageMetadata") or {}
        prompt = int(usage.get("promptTokenCount") or 0)
        out = int(usage.get("candidatesTokenCount") or 0) + int(
            usage.get("thoughtsTokenCount") or 0
        )
        self.ledger.record(run_id, model, estimate_cost(model, prompt, out))
