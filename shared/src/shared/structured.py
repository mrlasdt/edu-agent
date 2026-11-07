"""
Schema-checked model output (ADR-0007, "Amendment: structured output is validated").

A caller that needs JSON from a model describes it as a pydantic model and does two things:

  1. asks for it with `response_format_for(model, Schema)`: the schema itself where litellm says
     the model supports `response_format=<schema>`, else plain `json_object` mode;
  2. checks the reply with `parse_structured(content, Schema)`, which raises
     `StructuredOutputError` for anything that is not valid JSON of that shape.

The check is the same whichever mode asked, so a model that ignores the schema (or a Router
fallback that answers from a model without schema support) is still validated. A reply that fails
the check is an error the caller must handle; it must never be read as an answer, since
defaulting a missing field or a stringly typed boolean is exactly what hides a broken model.
"""
from __future__ import annotations

import re
from typing import Any

import litellm
from pydantic import BaseModel, ValidationError

# Models in json_object mode often wrap the object in a ```json fence.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


class StructuredOutputError(ValueError):
    """The model's reply was not valid JSON of the requested shape."""


def response_format_for(model: str, schema: type[BaseModel]) -> type[BaseModel] | dict[str, Any]:
    """`schema` for a model litellm knows to support it, else `{"type": "json_object"}`."""
    try:
        supported = litellm.supports_response_schema(model=model)
    except Exception:  # an unknown model string: ask for plain JSON
        supported = False
    return schema if supported else {"type": "json_object"}


def parse_structured[T: BaseModel](content: object, schema: type[T]) -> T:
    """Validate a model reply as `schema`; raise StructuredOutputError if it does not fit."""
    if not isinstance(content, str) or not content.strip():
        raise StructuredOutputError(f"empty reply, expected a {schema.__name__} object")
    fenced = _FENCE.match(content)
    try:
        return schema.model_validate_json(fenced.group(1) if fenced else content)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"reply is not a valid {schema.__name__}: {exc.errors(include_input=False)}"
        ) from exc
