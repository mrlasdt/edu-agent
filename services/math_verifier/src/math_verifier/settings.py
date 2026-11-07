"""The math verifier's settings: where it listens. It reads nothing else.

Stdlib only: the math-verifier image ships neither shared/config.py nor pydantic-settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class MathVerifierSettings:
    math_verifier_host: str = "127.0.0.1"
    math_verifier_port: int = 8090


@lru_cache
def get_settings() -> MathVerifierSettings:
    defaults = MathVerifierSettings()
    return MathVerifierSettings(
        math_verifier_host=os.environ.get("MATH_VERIFIER_HOST", defaults.math_verifier_host),
        math_verifier_port=int(os.environ.get("MATH_VERIFIER_PORT", defaults.math_verifier_port)),
    )
