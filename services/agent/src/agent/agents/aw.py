from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from typing import Any

from services.agent.src.agent.base import BaseAgent, Turn, with_note
from services.agent.src.agent.capabilities.match_student_style import MatchStudentStyle
from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk, RetrieveCorpus
from services.agent.src.agent.prompting import build_messages
from shared.src.shared.critics.citation_checker import check_citations, drop_uncited
from shared.src.shared.critics.style_critic import run_style_critic
from shared.src.shared.events import CitationsStripped, Event, SampleEssay, Token
from shared.src.shared.models import Mode, Session
from shared.src.shared.prompts import read_prompt
from shared.src.shared.turn_meter import TurnMeter

_MIN_PERSONAL_ESSAYS = 2

_TIER_LABELS = {
    "global": "Official ETS material",
    "school": "Course notes",
    "candidate": "Your notes",
}


def _format_chunks_for_context(chunks: list[CorpusChunk]) -> str:
    if not chunks:
        return ""
    lines = ["\n\n## Reference Material\n"]
    for i, chunk in enumerate(chunks, start=1):
        tier_label = _TIER_LABELS.get(chunk.tier, "Reference")
        lines.append(f"[{i}] ({tier_label} — {chunk.page_or_section})\n{chunk.text}")
    return "\n\n".join(lines)


def _format_exemplars(exemplars: list[str]) -> str:
    if not exemplars:
        return ""
    lines = ["\n\n## Your Previous Writing (style reference)\n"]
    for i, essay in enumerate(exemplars, start=1):
        lines.append(f"Example {i}:\n{essay}")
    return "\n\n".join(lines)


def _build_messages(
    session: Session,
    user_message: str,
    system_prompt: str,
    chunk_context: str,
    exemplar_context: str = "",
) -> list[dict[str, Any]]:
    """The AW turn's messages: style examples, then notes, each on a new line after the message."""
    blocks = [f"\n{block}" for block in (exemplar_context, chunk_context) if block]
    return build_messages(session, user_message, system_prompt, *blocks)


class AWAgent(BaseAgent[Turn]):
    """
    GRE Analytical Writing agent.

    Tutor mode: plan only (outline + thesis + evidence pointers).
    Solve mode: complete essay with:
      - retrieve-corpus → inject as numbered [N] references
      - personal style exemplars (if opted-in and >= 2 essays available)
      - style critic (Haiku-tier) → one rewrite pass on structural violation
      - citation checker → strip invalid citations on persistent failure
      - flags the essay as a sample (`SampleEssay`); the web app shows the notice
    Both modes check the reply's [N] citations, retry once, then strip what still fails.
    """

    ROLE = "aw_agent"
    TOP_K = 5

    def __init__(
        self,
        model_config_path: str | None = None,
        retrieve_corpus: RetrieveCorpus | None = None,
        match_student_style: MatchStudentStyle | None = None,
    ) -> None:
        super().__init__(model_config_path, retrieve_corpus or RetrieveCorpus())
        self._match_student_style = match_student_style or MatchStudentStyle()

    format_chunks = staticmethod(_format_chunks_for_context)

    async def tutor_turn(self, turn: Turn) -> AsyncGenerator[Event, None]:
        system_prompt = read_prompt("aw_agent", Mode.tutor.value, turn.version)
        messages = _build_messages(turn.session, turn.message, system_prompt, turn.chunk_context)
        tokens, full_text = await self.collect(turn, messages)
        tokens, full_text, valid, invalid = await self._cited(turn, messages, tokens, full_text)
        for event in _reply(tokens, full_text, valid, invalid, sample=False):
            yield event

    async def solve_turn(self, turn: Turn) -> AsyncGenerator[Event, None]:
        system_prompt = read_prompt("aw_agent", Mode.solve.value, turn.version)

        # Personal style exemplars (if opted-in + >= 2 essays)
        exemplars = await self._get_personal_style_exemplars(turn.session)
        exemplar_context = (
            _format_exemplars(exemplars) if len(exemplars) >= _MIN_PERSONAL_ESSAYS else ""
        )
        messages = _build_messages(
            turn.session, turn.message, system_prompt, turn.chunk_context, exemplar_context
        )
        tokens, full_text = await self.collect(turn, messages)

        # Style critic: one rewrite pass on structural violation
        style_ok, style_feedback = await self._run_style_critic(
            full_text, turn.trace_id, turn.meter
        )
        if not style_ok:
            rewrite = messages + [
                {"role": "assistant", "content": full_text},
                {
                    "role": "user",
                    "content": (
                        f"Please rewrite to fix: {style_feedback}. "
                        "Keep the same thesis and evidence."
                    ),
                },
            ]
            tokens, full_text = await self.collect(turn, rewrite)

        tokens, full_text, valid, invalid = await self._cited(turn, messages, tokens, full_text)
        for event in _reply(tokens, full_text, valid, invalid, sample=True):
            yield event

    async def _cited(
        self, turn: Turn, messages: list[dict[str, Any]], tokens: list[str], full_text: str
    ) -> tuple[list[str], str, bool, list[str]]:
        """
        The reply, checked for [N] citations to notes that were not shown. A failing reply is
        generated once more with a note naming the bad markers. Returns the reply, whether it is
        clean, and the markers it still cites wrongly.
        """
        count = len(turn.chunks)
        valid, invalid = self._check_citations(full_text, count)
        if valid:
            return tokens, full_text, True, []
        note = f"[Invalid citations: {', '.join(invalid)}. Must only cite [1]–[{count}].]"
        tokens, full_text = await self.collect(turn, with_note(messages, note))
        valid, invalid = self._check_citations(full_text, count)
        return tokens, full_text, valid, invalid

    # ── seams ─────────────────────────────────────────────────────────────────

    def _check_citations(self, text: str, num_chunks: int) -> tuple[bool, list[str]]:
        return check_citations(text, num_chunks)

    async def _run_style_critic(
        self, essay_text: str, trace_id: str = "", meter: TurnMeter | None = None
    ) -> tuple[bool, str]:
        return await run_style_critic(essay_text, self._model_config_path, trace_id, meter)

    async def _get_personal_style_exemplars(self, session: Session) -> list[str]:
        if not session.personal_style_enabled:
            return []
        return await self._match_student_style.get_exemplars(
            candidate_id=session.candidate_id,
            test_type=session.test_type,
        )


def _reply(
    tokens: list[str], full_text: str, valid: bool, invalid: list[str], *, sample: bool
) -> Iterator[Event]:
    """
    The events of a checked reply. A clean reply goes out token by token; one that still cites
    unknown notes goes out as one text with those sentences removed, and says so. A solved essay
    is also flagged as a sample, for the web app's notice.
    """
    if valid:
        for token in tokens:
            yield Token(text=token)
    else:
        yield Token(text=drop_uncited(full_text, invalid)[0])
    if sample:
        yield SampleEssay()
    yield CitationsStripped(stripped=not valid)
