"""
mark_revision - the tool that lets the IELTS Writing coach re-mark a revised essay in chat.

The grader, the attempt store and the spend ledger live in the gateway, so the tool asks the
gateway (`POST /ielts/remark`) and never marks anything itself. What comes back is a practice
re-mark: untimed, not stored as a mock attempt, compared with the attempt the learner is being
coached from, and held to the same spend cap and rate limit as a mock.

Two things are kept out of the model's hands. The learner reads the re-mark as a card the agent
writes from the gateway's numbers (`card`), not as the model's retelling of them; and a refusal
(cap used up, too many requests, model down) is the gateway's own sentence, shown as it is.
At most one re-mark runs per turn.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from services.agent.src.agent.clients import pool
from services.agent.src.agent.settings import get_settings

_log = logging.getLogger(__name__)

TOOL_NAME = "mark_revision"
MARK_TIMEOUT_S = 180.0  # three marking passes and the feedback call, one after another

# A message with fewer words than this carries no essay to re-mark, so the turn offers no tool
# and streams as before. A real revision is 250 words; this leaves room for a partial one.
MIN_ESSAY_WORDS = 100
# Said to the model with a refusal: the learner has read the reason already, and it is not a fault.
REFUSAL_NOTE = (
    "The learner has already been shown this message. Do not call it a technical error and do "
    "not give a new band; coach from the earlier report only."
)

IELTS_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Re-mark a revised IELTS Writing Task 2 essay the learner pasted in their "
                "message, as an untimed practice mark that is not saved. Call it only when the "
                "learner asks you to re-mark or score the essay they pasted, at most once per "
                "message. It returns the new estimated `overall` band, the `overall_delta` "
                "against their earlier attempt and each criterion's `band` and `delta`; the app "
                "shows the learner the same numbers in a card, so do not repeat them. On failure "
                "it returns `error`; the app has already told the learner."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "essay": {
                        "type": "string",
                        "description": (
                            "The learner's whole revised essay, copied exactly as they wrote "
                            "it, without their request to you."
                        ),
                    }
                },
                "required": ["essay"],
                "additionalProperties": False,
            },
        },
    }
]


def offers_remark(message: str, attempt_id: str) -> bool:
    """Whether this turn has an attempt to compare with and an essay-sized message to mark."""
    return bool(attempt_id) and len(message.split()) >= MIN_ESSAY_WORDS


def _signed(delta: float | None) -> str:
    if delta is None:
        return "no earlier band"
    return "no change" if delta == 0 else f"{delta:+.1f}"


def card(remark: dict[str, Any]) -> str:
    """The re-mark as the learner reads it: label, band, change, criteria, disclaimer."""
    lines = [str(remark["label"])]
    if remark["status"] != "graded":
        lines.append(
            "This essay could not be marked: "
            f"{remark.get('ungraded_reason') or 'it is not a markable response'}"
        )
    else:
        low, high = remark["range"]
        lines.append(
            f"Estimated band {remark['overall']:.1f} (range {low:.1f} to {high:.1f}), "
            f"{_signed(remark.get('overall_delta'))} since the attempt you are coached from."
        )
        for c in remark["criteria"]:
            lines.append(f"- {c['name']}: {c['band']:.1f} ({_signed(c.get('delta'))})")
    lines.append(str(remark["disclaimer"]))
    return "\n".join(lines)


class RevisionTools:
    """One turn's re-mark: runs the one call the model may make and keeps what came back."""

    def __init__(self, attempt_id: str, *, gateway_url: str | None = None) -> None:
        self._attempt_id = attempt_id
        base = (gateway_url or get_settings().gateway_service_url).rstrip("/")
        self._url = f"{base}/ielts/remark"
        self._called = False
        self.remark: dict[str, Any] | None = None  # the gateway's re-mark, when it marked
        self.refusal: str | None = None  # the gateway's sentence, when it would not mark

    @property
    def notice(self) -> str:
        """What the learner reads first in the reply: the card, or why there is none."""
        if self.remark is not None:
            return card(self.remark)
        return self.refusal or ""

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Never raises: a refusal or an outage becomes an error the model can read."""
        essay = arguments.get("essay")
        if not isinstance(essay, str) or not essay.strip():
            return {"error": "essay is required: the learner's whole revised essay, verbatim."}
        if self._called:
            return {"error": "a re-mark already ran for this message; do not run another."}
        self._called = True
        try:
            client = pool.current()
            body = {"attempt_id": self._attempt_id, "essay": essay}
            if client is not None:
                response = await client.post(self._url, json=body, timeout=MARK_TIMEOUT_S)
            else:
                async with httpx.AsyncClient(timeout=MARK_TIMEOUT_S) as one_shot:
                    response = await one_shot.post(self._url, json=body)
        except httpx.HTTPError as exc:
            _log.warning("mark_revision: the gateway could not be reached: %s", type(exc).__name__)
            self.refusal = "The re-mark could not run just now because the marker is unreachable."
            return {"error": self.refusal, "note": REFUSAL_NOTE}
        if response.status_code != 200:
            self.refusal = _detail(response)
            return {"error": self.refusal, "note": REFUSAL_NOTE}
        self.remark = response.json()
        return self._result(self.remark)

    @staticmethod
    def _result(remark: dict[str, Any]) -> dict[str, Any]:
        if remark["status"] != "graded":
            return {"error": f"the essay could not be marked: {remark.get('ungraded_reason')}"}
        return {
            "overall": remark["overall"],
            "overall_delta": remark.get("overall_delta"),
            "criteria": [
                {"name": c["name"], "band": c["band"], "delta": c.get("delta")}
                for c in remark["criteria"]
            ],
        }


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    return f"The re-mark could not run just now (the marker answered {response.status_code})."
