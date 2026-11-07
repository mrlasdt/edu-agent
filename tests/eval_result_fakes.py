"""Builders for eval results in tests: a valid run provenance and a report around some suites."""
from __future__ import annotations

from evals.agent_eval.results import AwResult, EvalReport, RunMeta, SuiteResult


def run_meta(**overrides: object) -> RunMeta:
    fields: dict[str, object] = {
        "model": "offline/fake", "model_chain": ["offline/fake"], "samples": 1,
        "product_default_model": "openrouter/default", "is_product_default_model": False,
        "git_commit": None, "date_utc": "2026-01-01",
    }
    return RunMeta.model_validate({**fields, **overrides})


def eval_report(**suites: SuiteResult) -> EvalReport:
    return EvalReport(
        run_id="r", mode="offline", model_chain=["offline/fake"], judge="j", seed=1,
        run_meta=run_meta(), suites=dict(suites),
    )


def aw_result(total: int, dropped: int) -> AwResult:
    """The smallest valid suite result, with chosen unit counts."""
    return AwResult(
        n_judgements=total - dropped, within_tolerance=None, tolerance=1.0, mae=None, errors=0,
        units_total=total, units_dropped=dropped,
    )
