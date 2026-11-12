"""
Helpers the chat agents share: the message list for a model call and draining its stream.
Pure functions, so an eval or a seam can call them without building an agent.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from shared.src.shared.history import history_to_messages
from shared.src.shared.models import Session


def build_messages(
    session: Session, user_message: str, system_prompt: str, *context_blocks: str
) -> list[dict[str, Any]]:
    """
    The model's input: the system prompt, the session history, then the candidate's message.

    The context blocks (notes, style examples; empty ones are skipped) are appended, in order and
    as given, to the last user turn: `<message><block 1><block 2>`. The caller owns each block's
    leading separator, so the layout the Quant eval baselines were measured on stays byte-exact.
    """
    last = user_message + "".join(block for block in context_blocks if block)
    return [
        {"role": "system", "content": system_prompt},
        *history_to_messages(session.history),
        {"role": "user", "content": last},
    ]


async def token_stream(stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    """The non-empty content of each chunk of a litellm stream, as it arrives."""
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


async def collect_stream(stream: AsyncIterator[Any]) -> tuple[list[str], str]:
    """Drain a litellm stream: the non-empty tokens, and the full text."""
    tokens = [token async for token in token_stream(stream)]
    return tokens, "".join(tokens)
