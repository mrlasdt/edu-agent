"""
Spend accounting for the metered real-model calls: eval runs and the IELTS mock.

The ledger is a JSON file (default `.eval-ledger/ledger.json`, gitignored) that
accumulates the cost of every call. Every call is checked against the cumulative
ceiling *before* it is made, using a worst-case projection, so a run aborts
instead of overshooting. Free models (`:free`, `openrouter/free`) cost zero.

The ceiling is the owner's AI spend cap for the IELTS mock and the evals (5 USD), which share
one ledger, so there is one number to look at. Other model calls (the GRE tutor's /chat) are
not metered here. `spend_ledger()` is that shared ledger as configured (`AI_LEDGER_PATH`,
`AI_SPEND_CAP_USD`).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.src.shared.config import get_settings

DEFAULT_CEILING_USD = 5.0
DEFAULT_LEDGER_PATH = Path(".eval-ledger/ledger.json")

# USD per 1M tokens (input, output). Unknown paid models fall back to a deliberately
# pessimistic price so the guard errs on the side of aborting.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o1": (15.00, 60.00),
    "typesafe/jev-1.14": (0.042, 0.0),
}
_UNKNOWN_PRICE = (10.0, 30.0)


class BudgetExceededError(RuntimeError):
    """Raised before a call that could push cumulative spend over the ceiling."""


def _bare(model: str) -> str:
    for prefix in ("openrouter/", "openai/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def is_free(model: str) -> bool:
    bare = _bare(model)
    return bare.endswith(":free") or bare == "openrouter/free"


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if is_free(model):
        return 0.0
    price_in, price_out = _PRICES.get(_bare(model), _UNKNOWN_PRICE)
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Ledger:
    path: Path = DEFAULT_LEDGER_PATH
    ceiling_usd: float = DEFAULT_CEILING_USD

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"entries": []}
        data: dict[str, Any] = json.loads(self.path.read_text())
        return data

    def spent(self) -> float:
        return float(sum(e["cost_usd"] for e in self._load()["entries"]))

    def remaining(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent())

    def ensure_capacity(self, projected_usd: float) -> None:
        """Abort when spent + projected would exceed the ceiling (free calls always pass)."""
        if projected_usd <= 0:
            return
        spent = self.spent()
        if spent + projected_usd > self.ceiling_usd:
            raise BudgetExceededError(
                f"budget guard: spent ${spent:.4f} + projected ${projected_usd:.4f} "
                f"exceeds ceiling ${self.ceiling_usd:.2f}"
            )

    def record(self, run_id: str, model: str, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        data = self._load()
        data["entries"].append(
            {"ts": time.time(), "run_id": run_id, "model": model, "cost_usd": cost_usd}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.path)


def spend_ledger(path: str | Path | None = None, ceiling_usd: float | None = None) -> Ledger:
    """The shared spend ledger at the configured path and cap, unless overridden."""
    settings = get_settings()
    return Ledger(
        Path(path or settings.ai_ledger_path),
        settings.ai_spend_cap_usd if ceiling_usd is None else ceiling_usd,
    )
