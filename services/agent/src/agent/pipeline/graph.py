"""
LangGraph question-generation pipeline.

    plan -> (route by expert) -> generate -> verify -> critic -> accept
                  ^                  |          |
                  +------ revise <---+----------+      (bounded by max_attempts)
                                     |
                                  escalate  -> needs_review (human review queue)

- plan:      an LLM planner picks the question type and learning constraints; the router
             maps the type to one of three generator experts (numeric / choice / comparison).
- generate:  the expert writes one structured item. On a retry it also sees the failed
             checks and its previous attempt.
- verify:    deterministic tools (structure, leak, sympy solver, distractors) via the
             verify-math skill. The generator never grades its own key.
- critic:    an LLM pass for ambiguity and answer leaks, run only on items that already
             pass the deterministic tools.
- revise:    turns failed checks into feedback. After `max_attempts` the item is escalated
             to human review instead of being emitted.

Every step appends a TraceEvent, so each result carries its own audit trail.
"""
from __future__ import annotations

import json
import operator
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from services.agent.src.agent.pipeline.llm import Completer, extract_json, load_prompt
from services.agent.src.agent.pipeline.schema import (
    EXPERT_FOR_TYPE,
    CheckResult,
    Expert,
    GenerationRequest,
    PipelineResult,
    Plan,
    QuestionItem,
    QuestionType,
    TraceEvent,
)
from services.agent.src.agent.pipeline.tools import run_checks
from services.agent.src.agent.skills.verify_math import VerifyMathSkill

DEFAULT_MAX_ATTEMPTS = 3
_PREVIOUS_ATTEMPT_CHARS = 1500

ROLE_PLANNER = "question_planner"
ROLE_GENERATOR = "question_generator"
ROLE_CRITIC = "question_critic"


class PipelineState(TypedDict, total=False):
    request: GenerationRequest
    trace_id: str
    plan: Plan
    item: QuestionItem | None
    raw: str
    attempt: int
    feedback: list[str]
    checks: list[CheckResult]
    llm_calls: int
    status: str
    failures: list[str]
    trace: Annotated[list[TraceEvent], operator.add]


def rule_based_type(topic: str) -> QuestionType:
    """Keyword router used when the planner's reply is unusable and by the baseline."""
    t = topic.lower()
    if "compar" in t:
        return QuestionType.quantitative_comparison
    if any(k in t for k in ("equation", "solve", "entry", "numeric")):
        return QuestionType.numeric_entry
    if any(k in t for k in ("roots", "solutions", "all of", "select all")):
        return QuestionType.multiple_choice_multiple
    return QuestionType.multiple_choice_single


def rule_based_plan(request: GenerationRequest) -> Plan:
    qtype = request.question_type or rule_based_type(request.topic)
    return Plan(
        question_type=qtype,
        expert=EXPERT_FOR_TYPE[qtype],
        topic=request.topic,
        difficulty=request.difficulty,
    )


def parse_item(text: str) -> tuple[QuestionItem | None, str]:
    """(item, error). Tolerates fences and prose around the JSON object."""
    data = extract_json(text)
    if data is None:
        return None, "reply contained no JSON object"
    try:
        return QuestionItem.model_validate(data), ""
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return None, f"invalid item JSON at {where}: {first['msg']}"


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes"}
    return value is True


def _user(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": json.dumps(payload)}]


class QuestionPipeline:
    def __init__(
        self,
        completer: Completer,
        verify_skill: VerifyMathSkill,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        prompt_versions: dict[str, str] | None = None,
    ) -> None:
        self._llm = completer
        self._skill = verify_skill
        self._max_attempts = max_attempts
        self._versions = prompt_versions or {}
        self._graph = self._build()

    # ── public ───────────────────────────────────────────────────────────────

    async def run(self, request: GenerationRequest) -> PipelineResult:
        initial: PipelineState = {
            "request": request,
            "trace_id": uuid4().hex,
            "attempt": 0,
            "feedback": [],
            "llm_calls": 0,
            "failures": [],
            "trace": [],
        }
        final: PipelineState = await self._graph.ainvoke(
            initial, config={"recursion_limit": 8 * self._max_attempts + 10}
        )
        accepted = final.get("status") == "accepted"
        item = final.get("item")
        return PipelineResult(
            request=request,
            trace_id=final["trace_id"],
            status="accepted" if accepted else "needs_review",
            item=item if accepted else None,
            candidate=item,
            plan=final.get("plan"),
            attempts=final.get("attempt", 0),
            llm_calls=final.get("llm_calls", 0),
            failures=final.get("failures", []),
            trace=final.get("trace", []),
        )

    # ── graph ────────────────────────────────────────────────────────────────

    def _build(self) -> Any:
        g: Any = StateGraph(PipelineState)  # langgraph's node typing is too narrow for mypy strict
        g.add_node("plan", self._plan)
        for expert in Expert:
            g.add_node(expert.value, self._generate_for(expert))
        g.add_node("verify", self._verify)
        g.add_node("critic", self._critic)
        g.add_node("revise", self._revise)
        g.add_node("accept", self._accept)
        g.add_node("escalate", self._escalate)

        g.set_entry_point("plan")
        expert_targets = {e.value: e.value for e in Expert}
        g.add_conditional_edges("plan", self._route, expert_targets)
        for expert in Expert:
            g.add_edge(expert.value, "verify")
        g.add_conditional_edges(
            "verify", self._after_verify,
            {"critic": "critic", "revise": "revise", "escalate": "escalate"},
        )
        g.add_conditional_edges(
            "critic", self._after_critic,
            {"accept": "accept", "revise": "revise", "escalate": "escalate"},
        )
        g.add_conditional_edges("revise", self._route, expert_targets)
        g.add_edge("accept", END)
        g.add_edge("escalate", END)
        return g.compile()

    def _system(self, name: str) -> dict[str, Any]:
        return {"role": "system", "content": load_prompt(name, self._versions.get(name, "v1"))}

    async def _ask(
        self, role: str, prompt: str, payload: dict[str, Any], state: PipelineState
    ) -> str:
        return await self._llm.complete(
            role, [self._system(prompt), *_user(payload)], state["trace_id"]
        )

    # ── routing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _route(state: PipelineState) -> str:
        return state["plan"].expert.value

    def _after_verify(self, state: PipelineState) -> str:
        if all(c.passed for c in state["checks"]):
            return "critic"
        return "revise" if state["attempt"] < self._max_attempts else "escalate"

    def _after_critic(self, state: PipelineState) -> str:
        if all(c.passed for c in state["checks"]):
            return "accept"
        return "revise" if state["attempt"] < self._max_attempts else "escalate"

    # ── nodes ────────────────────────────────────────────────────────────────

    async def _plan(self, state: PipelineState) -> dict[str, Any]:
        request = state["request"]
        start = time.perf_counter()
        raw = await self._ask(ROLE_PLANNER, "planner", request.model_dump(mode="json"), state)
        data = extract_json(raw) or {}
        fallback = False
        try:
            qtype = request.question_type or QuestionType(data["question_type"])
            plan = Plan(
                question_type=qtype,
                expert=EXPERT_FOR_TYPE[qtype],
                topic=request.topic,
                difficulty=request.difficulty,
                learning_objective=str(data.get("learning_objective", ""))[:300],
                constraints=[str(c)[:200] for c in data.get("constraints", [])][:6],
            )
        except (KeyError, ValueError, TypeError):
            plan, fallback = rule_based_plan(request), True
        return {
            "plan": plan,
            "llm_calls": state["llm_calls"] + 1,
            "trace": [
                TraceEvent(
                    step="plan", model=self._llm.model_for(ROLE_PLANNER),
                    latency_s=time.perf_counter() - start,
                    data={
                        "plan": plan.model_dump(mode="json"),
                        "routed_to": plan.expert.value,
                        "rule_based_fallback": fallback,
                    },
                )
            ],
        }

    def _generate_for(self, expert: Expert) -> Callable[[PipelineState], Awaitable[dict[str, Any]]]:
        async def generate(state: PipelineState) -> dict[str, Any]:
            attempt = state["attempt"] + 1
            payload: dict[str, Any] = {
                "request_id": state["request"].request_id,
                "plan": state["plan"].model_dump(mode="json"),
            }
            if state["feedback"]:
                payload["feedback"] = state["feedback"]
                payload["previous_attempt"] = state.get("raw", "")[:_PREVIOUS_ATTEMPT_CHARS]
            start = time.perf_counter()
            raw = await self._ask(ROLE_GENERATOR, expert.value, payload, state)
            item, error = parse_item(raw)
            checks: list[CheckResult] = []
            if item is not None and item.question_type != state["plan"].question_type:
                error = (
                    f"question_type must be {state['plan'].question_type.value}, "
                    f"got {item.question_type.value}"
                )
            if error:
                checks.append(CheckResult(name="schema", passed=False, detail=error))
            return {
                "attempt": attempt,
                "item": item,
                "raw": raw,
                "checks": checks,
                "llm_calls": state["llm_calls"] + 1,
                "trace": [
                    TraceEvent(
                        step="generate", attempt=attempt, ok=not error,
                        model=self._llm.model_for(ROLE_GENERATOR),
                        latency_s=time.perf_counter() - start,
                        data={
                            "expert": expert.value,
                            "revision": bool(state["feedback"]),
                            "item": item.model_dump(mode="json") if item else None,
                            "error": error,
                        },
                    )
                ],
            }

        return generate

    async def _verify(self, state: PipelineState) -> dict[str, Any]:
        item = state["item"]
        start = time.perf_counter()
        checks = state["checks"] if item is None else await run_checks(item, self._skill)
        return {
            "checks": checks,
            "trace": [
                TraceEvent(
                    step="verify", attempt=state["attempt"],
                    ok=all(c.passed for c in checks),
                    latency_s=time.perf_counter() - start,
                    data={"checks": [c.model_dump() for c in checks]},
                )
            ],
        }

    async def _critic(self, state: PipelineState) -> dict[str, Any]:
        item = state["item"]
        assert item is not None  # only reached when the deterministic tools passed
        start = time.perf_counter()
        raw = await self._ask(ROLE_CRITIC, "critic", item.model_dump(mode="json"), state)
        verdict = extract_json(raw)
        if verdict is None:
            check = CheckResult(
                name="critic", passed=False, detail="critic reply was not parseable JSON"
            )
        else:
            issues = [str(i) for i in verdict.get("issues", [])][:4]
            flagged = _flag(verdict.get("ambiguous")) or _flag(verdict.get("leaks_answer"))
            check = CheckResult(
                name="critic", passed=not flagged,
                detail="; ".join(issues) or ("flagged without detail" if flagged else ""),
            )
        return {
            "checks": [*state["checks"], check],
            "llm_calls": state["llm_calls"] + 1,
            "trace": [
                TraceEvent(
                    step="critic", attempt=state["attempt"], ok=check.passed,
                    model=self._llm.model_for(ROLE_CRITIC),
                    latency_s=time.perf_counter() - start,
                    data={"check": check.model_dump(), "parsed": verdict is not None},
                )
            ],
        }

    @staticmethod
    async def _revise(state: PipelineState) -> dict[str, Any]:
        feedback = [f"{c.name}: {c.detail}" for c in state["checks"] if not c.passed]
        return {
            "feedback": feedback,
            "trace": [
                TraceEvent(step="revise", attempt=state["attempt"], data={"feedback": feedback})
            ],
        }

    @staticmethod
    async def _accept(state: PipelineState) -> dict[str, Any]:
        return {
            "status": "accepted",
            "trace": [TraceEvent(step="accept", attempt=state["attempt"])],
        }

    @staticmethod
    async def _escalate(state: PipelineState) -> dict[str, Any]:
        failures = [f"{c.name}: {c.detail}" for c in state["checks"] if not c.passed]
        return {
            "status": "needs_review",
            "failures": failures,
            "trace": [
                TraceEvent(
                    step="escalate", attempt=state["attempt"], ok=False,
                    data={"reason": "retries exhausted", "failures": failures},
                )
            ],
        }
