from __future__ import annotations

import re
from pathlib import Path
from typing import AsyncGenerator

import litellm

from services.agent.src.agent.base import BaseAgent
from services.agent.src.agent.critics.citation_checker import check_citations, strip_uncited_claims
from services.agent.src.agent.skills.retrieve_corpus import RetrieveCorpusSkill
from shared.src.shared.config import load_model_config
from shared.src.shared.models import MessageRole, Mode, Session

_PROMPT_ROOT = Path("prompts/aw_agent")

_ARGUMENT_KEYWORDS = re.compile(
    r"\b(the following|the argument|critique|examine the|stated and/or unstated|"
    r"write a response in which you examine|evaluate the argument)\b",
    re.IGNORECASE,
)


def _load_prompt(mode: Mode) -> str:
    path = _PROMPT_ROOT / mode.value / "v1.md"
    raw = path.read_text()
    return re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()


def _format_chunks_for_context(chunks: list) -> str:
    """Format retrieved chunks as numbered reference list for injection into prompt."""
    if not chunks:
        return ""
    lines = ["\n\n## Reference Material\n"]
    for i, chunk in enumerate(chunks, start=1):
        tier_label = {
            "global": "Official ETS material",
            "school": "Course notes",
            "candidate": "Your notes",
        }.get(getattr(chunk, "tier", "global"), "Reference")
        lines.append(
            f"[{i}] ({tier_label} — {getattr(chunk, 'page_or_section', '')})\n"
            f"{getattr(chunk, 'text', chunk.get('text', '') if isinstance(chunk, dict) else '')}"
        )
    return "\n\n".join(lines)


def _build_messages(
    session: Session, user_message: str, system_prompt: str, chunk_context: str
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in session.history:
        role = "user" if msg.role == MessageRole.candidate else "assistant"
        messages.append({"role": role, "content": msg.content})
    content = user_message
    if chunk_context:
        content = f"{user_message}\n{chunk_context}"
    messages.append({"role": "user", "content": content})
    return messages


async def _collect_stream(stream) -> tuple[list[str], str]:
    tokens: list[str] = []
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            tokens.append(content)
    return tokens, "".join(tokens)


class AWAgent(BaseAgent):
    """
    GRE Analytical Writing agent.

    Tutor mode: returns an essay plan (outline + thesis + evidence pointers).
    Solve mode: returns a complete sample essay.

    Both modes: retrieve-corpus Skill injects reference material; citation checker
    validates [N] markers post-generation.  On citation failure → regenerate once →
    if still failing, strip bad sentences and emit citation_stripped=True metadata.
    """

    def __init__(
        self,
        model_config_path: str | None = None,
        retrieve_corpus_skill: RetrieveCorpusSkill | None = None,
    ) -> None:
        self._model_config = load_model_config(model_config_path)
        self._retrieve_skill = retrieve_corpus_skill or RetrieveCorpusSkill()

    async def run(
        self, session: Session, message: str
    ) -> AsyncGenerator[str | dict, None]:
        chunks = await self._retrieve(session, message)
        chunk_context = _format_chunks_for_context(chunks)
        system_prompt = _load_prompt(session.mode)
        messages = _build_messages(session, message, system_prompt, chunk_context)
        model = self._model_config.litellm_model("aw_agent")
        temperature = self._model_config.role("aw_agent").temperature

        # Attempt 1
        response = await litellm.acompletion(
            model=model, messages=messages, temperature=temperature, stream=True
        )
        tokens, full_text = await _collect_stream(response)
        valid, invalid = self._check_citations(full_text, len(chunks))

        if valid:
            for token in tokens:
                yield token
            yield {"citation_stripped": False}
            return

        # Attempt 2: regenerate with citation feedback
        retry_messages = messages + [
            {
                "role": "assistant",
                "content": (
                    f"[My previous response contained invalid citation markers: "
                    f"{', '.join(invalid)}. I must only cite [1]–[{len(chunks)}].]"
                ),
            }
        ]
        response = await litellm.acompletion(
            model=model, messages=retry_messages, temperature=temperature, stream=True
        )
        tokens, full_text = await _collect_stream(response)
        valid, invalid = self._check_citations(full_text, len(chunks))

        if valid:
            for token in tokens:
                yield token
            yield {"citation_stripped": False}
            return

        # Strip invalid citations and emit with flag
        clean_text = strip_uncited_claims(full_text, invalid)
        yield clean_text
        yield {"citation_stripped": True}

    # ── seams ─────────────────────────────────────────────────────────────────

    async def _retrieve(self, session: Session, message: str) -> list:
        return await self._retrieve_skill.retrieve(
            query=message,
            candidate_id=session.candidate_id,
            school_id="",  # v1: no school for Candidates
            test_type=session.test_type,
            top_k=5,
        )

    def _check_citations(self, text: str, num_chunks: int) -> tuple[bool, list[str]]:
        return check_citations(text, num_chunks)

    async def _detect_task_type(self, message: str) -> str:
        """Return 'argument' if message matches Argument task keywords, else 'issue'."""
        if _ARGUMENT_KEYWORDS.search(message):
            return "argument"
        return "issue"
