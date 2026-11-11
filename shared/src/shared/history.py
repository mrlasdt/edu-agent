"""The chat history as the chat-completion messages a model call takes."""
from __future__ import annotations

from collections.abc import Iterable

from shared.src.shared.models import Message, MessageRole


def history_to_messages(history: Iterable[Message]) -> list[dict[str, str]]:
    """Each message as a role and content; the candidate's are "user", the agent's "assistant"."""
    return [
        {
            "role": "user" if message.role == MessageRole.candidate else "assistant",
            "content": message.content,
        }
        for message in history
    ]
