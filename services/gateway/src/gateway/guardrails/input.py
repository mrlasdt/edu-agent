"""Input guardrails: content safety check and completeness enrichment."""
from __future__ import annotations

import litellm
from shared.src.shared.config import get_settings, load_model_config
from shared.src.shared.models import Session
from services.gateway.src.gateway.guardrails.events import emit_guardrail_event

import httpx


async def check_content_safety(text: str, trace_id: str) -> tuple[bool, str]:
    """
    Call the OpenAI Moderation API.
    Returns (safe, reason) — reason is empty when safe=True.
    """
    result = await _call_moderation_api(text)
    if result.get("flagged"):
        flagged_cats = [k for k, v in result.get("categories", {}).items() if v]
        reason = f"content flagged: {', '.join(flagged_cats)}"
        emit_guardrail_event("input", "content_safety", "fail", trace_id, reason=reason)
        return False, reason
    emit_guardrail_event("input", "content_safety", "pass", trace_id)
    return True, ""


async def check_completeness(
    text: str, session: Session, trace_id: str
) -> tuple[bool, str]:
    """
    Haiku-tier LLM check: is the question complete enough to answer?
    Returns (complete, clarification_prompt).
    """
    result = await _call_enrichment_llm(text, session)
    complete = result.get("complete", True)
    clarification = result.get("clarification", "")
    label = "pass" if complete else "fail"
    emit_guardrail_event("input", "completeness", label, trace_id)
    return complete, clarification


async def _call_moderation_api(text: str) -> dict:
    """Call OpenAI Moderation API. Seam for unit test patching."""
    settings = get_settings()
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/moderations",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={"input": text},
        )
        response.raise_for_status()
        data = response.json()
        result = data["results"][0]
        return {"flagged": result["flagged"], "categories": result["categories"]}


async def _call_enrichment_llm(text: str, session: Session) -> dict:
    """
    Haiku-tier LLM call to detect incomplete questions.
    Returns {"complete": bool, "clarification": str}.
    Seam for unit test patching.
    """
    cfg = load_model_config()
    model = cfg.litellm_model("orchestrator")
    system = (
        f"You are checking if a GRE {session.section.upper()} question is complete and answerable. "
        "Reply with JSON: {\"complete\": true/false, \"clarification\": \"<question to ask if incomplete>\"}"
    )
    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        stream=False,
    )
    import json
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {"complete": True, "clarification": ""}
