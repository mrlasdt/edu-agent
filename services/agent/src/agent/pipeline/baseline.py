"""The single-call baseline: one generation, no planner, no tools, no revise loop."""
from __future__ import annotations

import json
from uuid import uuid4

from services.agent.src.agent.pipeline.graph import ROLE_GENERATOR, parse_item, rule_based_plan
from services.agent.src.agent.pipeline.llm import Completer, load_prompt
from services.agent.src.agent.pipeline.schema import (
    GenerationRequest,
    PipelineResult,
    TraceEvent,
)


class SingleCallGenerator:
    """
    Same model role and the same item schema as the pipeline, but a single prompt that
    covers every question type and whatever comes back is emitted unchecked. This is the
    "before" the pipeline is measured against.
    """

    def __init__(self, completer: Completer, prompt_version: str = "v1") -> None:
        self._llm = completer
        self._version = prompt_version

    async def run(self, request: GenerationRequest) -> PipelineResult:
        trace_id = uuid4().hex
        plan = rule_based_plan(request)  # the type is fixed by the request or a keyword rule
        payload = {"request_id": request.request_id, "plan": plan.model_dump(mode="json")}
        messages = [
            {"role": "system", "content": load_prompt("baseline", self._version)},
            {"role": "user", "content": json.dumps(payload)},
        ]
        raw = await self._llm.complete(ROLE_GENERATOR, messages, trace_id)
        item, error = parse_item(raw)
        return PipelineResult(
            request=request,
            trace_id=trace_id,
            status="unchecked" if item else "unparseable",
            item=item,
            candidate=item,
            plan=plan,
            attempts=1,
            llm_calls=1,
            failures=[error] if error else [],
            trace=[
                TraceEvent(
                    step="generate", ok=item is not None,
                    model=self._llm.model_for(ROLE_GENERATOR),
                    data={"item": item.model_dump(mode="json") if item else None, "error": error},
                )
            ],
        )
