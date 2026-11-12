"""
Every grader the evals use, in one place.

  quant_judge  - deterministic: checks a Quant answer with the math verifier, no model
  aw_judge     - an LLM scores an Analytical Writing essay against the rubric
  typed        - the typed-question judge (`yes_no`, `score`) the agent-in-the-loop eval asks:
                 LLMJudge on a real model, HeuristicJudge offline
"""
