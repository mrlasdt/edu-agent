"""
Tests for schema-checked model output (shared.structured).

Behaviors under test:
  1. A model litellm lists as schema-capable is asked for the schema; any other (or an unknown)
     model is asked for plain json_object
  2. A reply is accepted only as valid JSON of the schema's shape, optionally in a ```json fence
  3. Anything else raises StructuredOutputError
"""
from unittest.mock import patch

import pytest
from pydantic import BaseModel, StrictBool

from shared.src.shared.structured import (
    StructuredOutputError,
    parse_structured,
    response_format_for,
)


class Flag(BaseModel):
    on: StrictBool


def test_a_schema_capable_model_is_asked_for_the_schema():
    assert response_format_for("gpt-4o-mini", Flag) is Flag


def test_a_model_without_schema_support_is_asked_for_json_object():
    assert response_format_for("openrouter/recursive/flint-v1.0:free", Flag) == {
        "type": "json_object"
    }


def test_an_unknown_model_string_is_asked_for_json_object():
    with patch("litellm.supports_response_schema", side_effect=Exception("unmapped model")):
        assert response_format_for("who/knows", Flag) == {"type": "json_object"}


@pytest.mark.parametrize(
    "content",
    ['{"on": true}', ' {"on": true} ', '```json\n{"on": true}\n```', '```\n{"on": true}\n```'],
)
def test_valid_replies_are_parsed(content):
    assert parse_structured(content, Flag) == Flag(on=True)


@pytest.mark.parametrize(
    "content",
    ['{"on": "true"}', '{"on": 1}', "{}", "[]", "true", "not json", '{"on": tr', "", "  ", None, 7],
)
def test_anything_else_raises(content):
    with pytest.raises(StructuredOutputError):
        parse_structured(content, Flag)
