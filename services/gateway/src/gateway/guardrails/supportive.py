"""The supportive reply for a message about self-harm: a fixed text, never a model call."""

from __future__ import annotations

# The moderation categories that mean "this person may be in distress". OpenAI's names; a
# provider's spelling with underscores is accepted too.
SELF_HARM_CATEGORIES = frozenset({"self-harm", "self-harm/intent", "self-harm/instructions"})

_FLAGGED_PREFIX = "content flagged: "

SUPPORTIVE_MESSAGE = (
    "I'm really sorry you're going through this, and I'm glad you said something. "
    "This tutor isn't the right help for something this serious, and you deserve support "
    "from a real person. If you might be in danger, please contact your local emergency "
    "services now. If you're in the US, you can call or text 988 to reach the Suicide and "
    "Crisis Lifeline at any time. It can also help to reach out to someone you trust. "
    "Your studying will be here whenever you want to come back to it."
)


def flagged_reason(categories: list[str]) -> str:
    """The reason string `check_content_safety` returns for a moderation hit."""
    return f"{_FLAGGED_PREFIX}{', '.join(categories)}"


def is_self_harm_reason(reason: str) -> bool:
    """True when a moderation reason names a self-harm category."""
    if not reason.startswith(_FLAGGED_PREFIX):
        return False
    named = reason[len(_FLAGGED_PREFIX):].split(",")
    categories = {c.strip().lower().replace("_", "-") for c in named}
    return not categories.isdisjoint(SELF_HARM_CATEGORIES)
