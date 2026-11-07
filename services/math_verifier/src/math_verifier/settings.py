"""The math verifier's settings: where it listens. It reads nothing else."""
from __future__ import annotations

from functools import lru_cache

from shared.src.shared.config import ServiceSettings


class MathVerifierSettings(ServiceSettings):
    math_verifier_host: str = "127.0.0.1"
    math_verifier_port: int = 8090


@lru_cache
def get_settings() -> MathVerifierSettings:
    return MathVerifierSettings()
