from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from services.agent.src.agent.base import BaseAgent
from services.agent.src.agent.capabilities.match_student_style import MatchStudentStyle
from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk, RetrieveCorpus
from shared.src.shared.config import PROJECT_ROOT, load_model_config
from shared.src.shared.critics.citation_checker import check_citations, strip_uncited_claims
from shared.src.shared.critics.style_critic import run_style_critic
from shared.src.shared.events import CitationsStripped, Event, SampleEssay, Sources, Token
from shared.src.shared.models import MessageRole, Mode, Session
from shared.src.shared.observability import build_llm_metadata
from shared.src.shared.turn_meter import TurnMeter

_log = logging.getLogger(__name__)

_PROMPT_ROOT = PROJECT_ROOT / "prompts" / "aw_agent"

_ARGUMENT_KEYWORDS = re.compile(
    r"\b(the following|the argument|critique|examine the|stated and/or unstated|"
    r"write a response in which you examine|evaluate the argument)\b",
    re.IGNORECASE,
)

_MIN_PERSONAL_ESSAYS = 2


def _load_prompt(mode: Mode, version: str) -> str:
    path = _PROMPT_ROOT / mode.value / f"{version}.md"
    raw = path.read_text()
    return re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()


def _format_chunks_for_context(chunks: list[CorpusChunk]) -> str:
    if not chunks:
        return ""
    lines = ["\n\n## Reference Material\n"]
    for i, chunk in enumerate(chunks, start=1):
        tier_label = {
            "global": "Official ETS material",
            "school": "Course notes",
            "candidate": "Your notes",
        }.get(chunk.tier, "Reference")
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
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in session.history:
        role = "user" if msg.role == MessageRole.candidate else "assistant"
        messages.append({"role": role, "content": msg.content})
    content = user_message
    if exemplar_context:
        content = f"{content}\n{exemplar_context}"
    if chunk_context:
        content = f"{content}\n{chunk_context}"
    messages.append({"role": "user", "content": content})
    return messages


async def _collect_stream(stream: AsyncIterator[Any]) -> tuple[list[str], str]:
    tokens: list[str] = []
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            tokens.append(content)
    return tokens, "".join(tokens)


class AWAgent(BaseAgent):
    """
    GRE Analytical Writing agent.

    Tutor mode: plan only (outline + thesis + evidence pointers).
    Solve mode: complete essay with:
      - retrieve-corpus → inject as numbered [N] references
      - personal style exemplars (if opted-in and >= 2 essays available)
      - style critic (Haiku-tier) → one rewrite pass on structural violation
      - citation checker → strip invalid citations on persistent failure
      - flags the essay as a sample (`SampleEssay`); the web app shows the notice
    """

    def __init__(
        self,
        model_config_path: str | None = None,
        retrieve_corpus: RetrieveCorpus | None = None,
        match_student_style: MatchStudentStyle | None = None,
    ) -> None:
        self._model_config = load_model_config(model_config_path)
        self._retrieve_corpus = retrieve_corpus or RetrieveCorpus()
        self._match_student_style = match_student_style or MatchStudentStyle()
        self._model_config_path = model_config_path

    async def run(
        self, session: Session, message: str, trace_id: str = ""
    ) -> AsyncGenerator[Event, None]:
        version = self._model_config.prompt_version("aw_agent", session.mode.value)
        meter = TurnMeter(trace_id, "agent", "aw_agent", session.mode.value, version)
        try:
            with meter.phase("retrieval"):
                chunks = await self._retrieve(session, message)
            meter.annotate(notes_retrieved=len(chunks))
            if chunks:
                yield Sources(count=len(chunks))
            chunk_context = _format_chunks_for_context(chunks)
            obs_meta = build_llm_metadata(trace_id=trace_id, prompt_version=version)

            if session.mode == Mode.tutor:
                async for item in self._tutor_turn(
                    session, message, chunk_context, chunks, meter, version, obs_meta
                ):
                    yield item
            else:
                async for item in self._solve_turn(
                    session, message, chunk_context, chunks, meter, version, obs_meta, trace_id
                ):
                    yield item
        except Exception as exc:
            meter.finish(error=type(exc).__name__)
            raise
        finally:
            meter.finish()

    # ── tutor mode ────────────────────────────────────────────────────────────

    async def _tutor_turn(
        self,
        session: Session,
        message: str,
        chunk_context: str,
        chunks: list[CorpusChunk],
        meter: TurnMeter,
        version: str = "v1",
        obs_meta: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event, None]:
        system_prompt = _load_prompt(Mode.tutor, version)
        messages = _build_messages(session, message, system_prompt, chunk_context)
        model = self._model_config.litellm_model("aw_agent")
        temperature = self._model_config.role("aw_agent").temperature

        # Attempt 1
        response = meter.stream(
            model=model,
            messages=messages,
            temperature=temperature,
            **(obs_meta or {}),
        )
        tokens, full_text = await _collect_stream(response)
        valid, invalid = self._check_citations(full_text, len(chunks))

        if valid:
            for token in tokens:
                yield Token(text=token)
            yield CitationsStripped(stripped=False)
            return

        # Retry on citation failure
        retry_messages = messages + [
            {
                "role": "assistant",
                "content": (
                    f"[Invalid citations: {', '.join(invalid)}. "
                    f"Must only cite [1]–[{len(chunks)}].]"
                ),
            }
        ]
        response = meter.stream(
            model=model,
            messages=retry_messages,
            temperature=temperature,
            **(obs_meta or {}),
        )
        tokens, full_text = await _collect_stream(response)
        valid, invalid = self._check_citations(full_text, len(chunks))

        if valid:
            for token in tokens:
                yield Token(text=token)
            yield CitationsStripped(stripped=False)
            return

        yield Token(text=strip_uncited_claims(full_text, invalid))
        yield CitationsStripped(stripped=True)

    # ── solve mode ────────────────────────────────────────────────────────────

    async def _solve_turn(
        self,
        session: Session,
        message: str,
        chunk_context: str,
        chunks: list[CorpusChunk],
        meter: TurnMeter,
        version: str = "v1",
        obs_meta: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> AsyncGenerator[Event, None]:
        system_prompt = _load_prompt(Mode.solve, version)
        model = self._model_config.litellm_model("aw_agent")
        temperature = self._model_config.role("aw_agent").temperature

        # Personal style exemplars (if opted-in + >= 2 essays)
        exemplars = await self._get_personal_style_exemplars(session)
        exemplar_context = (
            _format_exemplars(exemplars) if len(exemplars) >= _MIN_PERSONAL_ESSAYS else ""
        )

        messages = _build_messages(session, message, system_prompt, chunk_context, exemplar_context)

        # Generate essay
        response = meter.stream(
            model=model,
            messages=messages,
            temperature=temperature,
            **(obs_meta or {}),
        )
        tokens, full_text = await _collect_stream(response)

        # Style critic — one rewrite pass on structural violation
        style_ok, style_feedback = await self._run_style_critic(full_text, trace_id, meter)
        if not style_ok:
            rewrite_messages = messages + [
                {"role": "assistant", "content": full_text},
                {
                    "role": "user",
                    "content": (
                        f"Please rewrite to fix: {style_feedback}. "
                        "Keep the same thesis and evidence."
                    ),
                },
            ]
            response = meter.stream(
                model=model,
                messages=rewrite_messages,
                temperature=temperature,
                **(obs_meta or {}),
            )
            tokens, full_text = await _collect_stream(response)

        # Citation check
        valid, invalid = self._check_citations(full_text, len(chunks))
        if not valid:
            retry_messages = messages + [
                {
                    "role": "assistant",
                    "content": (
                        f"[Invalid citations: {', '.join(invalid)}. "
                        f"Must only cite [1]–[{len(chunks)}].]"
                    ),
                }
            ]
            response = meter.stream(
                model=model,
                messages=retry_messages,
                temperature=temperature,
                **(obs_meta or {}),
            )
            tokens, full_text = await _collect_stream(response)
            valid, invalid = self._check_citations(full_text, len(chunks))
            if not valid:
                yield Token(text=strip_uncited_claims(full_text, invalid))
                yield SampleEssay()
                yield CitationsStripped(stripped=True)
                return

        # Emit essay, then flag it as a sample for the web app's notice
        for token in tokens:
            yield Token(text=token)
        yield SampleEssay()
        yield CitationsStripped(stripped=False)

    # ── seams ─────────────────────────────────────────────────────────────────

    async def _retrieve(self, session: Session, message: str) -> list[CorpusChunk]:
        """
        Fetch grounding notes from the Corpus Service. Fails open, like the Quant agent: an
        unreachable corpus costs the turn its grounding, not the answer.
        """
        try:
            return await self._retrieve_corpus.retrieve(
                query=message,
                candidate_id=session.candidate_id,
                school_id="",
                test_type=session.test_type,
                top_k=5,
            )
        except Exception:
            _log.warning("corpus retrieval failed; answering without notes", exc_info=True)
            return []

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

    async def _detect_task_type(self, message: str) -> str:
        if _ARGUMENT_KEYWORDS.search(message):
            return "argument"
        return "issue"
