"""The gateway's settings: only what the gateway process (chat, guardrails, IELTS mock) reads."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from shared.src.shared.config import (
    AnchoredPath,
    FeedbackSettings,
    LangfuseSettings,
    LlmSettings,
    SpendSettings,
)


class GatewaySettings(LlmSettings, FeedbackSettings, SpendSettings, LangfuseSettings):
    env: str = "dev"
    agent_service_url: str = "http://localhost:8002"
    # What the gateway does when the content-safety (moderation) call itself fails: "closed"
    # answers "safety check unavailable, try again" and stops the turn, "open" lets the turn
    # go on and records `moderation_failed_open`. Empty = closed everywhere except ENV=dev.
    moderation_on_error: Literal["", "closed", "open"] = ""
    # Dev-only: a message containing this text is treated as a self-harm moderation hit, so the
    # supportive reply can be tried without a paid moderation key. Ignored unless ENV=dev.
    safety_test_trigger: str = ""
    # IELTS mock: attempts database and other local-only files (gitignored), and the
    # number of grader samples whose per-criterion median is reported.
    ielts_data_dir: AnchoredPath = "ielts-local"
    ielts_grader_samples: int = 3


@lru_cache
def get_settings() -> GatewaySettings:
    return GatewaySettings()
