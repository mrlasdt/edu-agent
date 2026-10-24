"""Wire the pipeline and its baseline from the shared model config."""
from __future__ import annotations

from services.agent.src.agent.pipeline.baseline import SingleCallGenerator
from services.agent.src.agent.pipeline.graph import DEFAULT_MAX_ATTEMPTS, QuestionPipeline
from services.agent.src.agent.pipeline.llm import LiteLLMCompleter
from services.agent.src.agent.skills.verify_math import VerifyMathSkill

_PROMPTS = ("planner", "numeric_expert", "choice_expert", "comparison_expert", "critic")


def build_pipeline(
    verify_skill: VerifyMathSkill | None = None,
    model_config_path: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> QuestionPipeline:
    completer = LiteLLMCompleter(model_config_path)
    return QuestionPipeline(
        completer,
        verify_skill or VerifyMathSkill(),
        max_attempts=max_attempts,
        prompt_versions={name: completer.prompt_version(name) for name in _PROMPTS},
    )


def build_baseline(model_config_path: str | None = None) -> SingleCallGenerator:
    completer = LiteLLMCompleter(model_config_path)
    return SingleCallGenerator(completer, completer.prompt_version("baseline"))
