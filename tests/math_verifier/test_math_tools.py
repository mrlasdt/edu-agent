"""
The sympy side of the model-called math tools: what each returns, and what it says when the
model's input is wrong.

Two things are pinned. Results are short and carry the number the model needs (an agent that
reads a wall of text uses more context for nothing). Errors name the problem and the fix,
because the model's next move is a corrected call: "could not parse" alone gets the same bad
input back.
"""

import pytest
from starlette.testclient import TestClient

from services.math_verifier.src.math_verifier.sandbox import run_math_tool, run_math_tool_sync
from services.math_verifier.src.math_verifier.server import mcp
from services.math_verifier.src.math_verifier.tools import run_tool

# ── math_evaluate ────────────────────────────────────────────────────────────


def test_a_percent_chain_is_computed_exactly() -> None:
    result = run_tool("math_evaluate", {"expression": "(1 + 15/100) * (1 - 20/100)"})
    assert result == {"value": "0.92", "exact": "23/25"}


def test_an_integer_result_is_just_the_value() -> None:
    assert run_tool("math_evaluate", {"expression": "17*23*19"}) == {"value": "7429"}


def test_an_irrational_result_gives_the_decimal_and_the_exact_form() -> None:
    assert run_tool("math_evaluate", {"expression": "sqrt(50)"}) == {
        "value": "7.071067812",
        "exact": "5*sqrt(2)",
    }


def test_a_huge_integer_is_summarised_not_printed() -> None:
    result = run_tool("math_evaluate", {"expression": "2^200"})
    assert result["exact"] == "an integer with 61 digits"
    assert len(result["value"]) < 20


@pytest.mark.parametrize(
    ("expression", "digits"),
    [
        ("factorial(2000)", 5736),
        ("2^20000", 6021),
        ("10^1000 - 1", 1000),
        ("10^1000 + 1", 1001),
    ],
)
def test_an_integer_past_the_python_digit_limit_is_still_summarised(
    expression: str, digits: int
) -> None:
    result = run_tool("math_evaluate", {"expression": expression})
    assert result["exact"] == f"an integer with {digits} digits"
    assert "error" not in result
    assert len(result["value"]) < 40


def test_an_expression_with_a_variable_comes_back_simplified_and_says_so() -> None:
    result = run_tool("math_evaluate", {"expression": "(x+1)^2 - x^2"})
    assert result["value"] == "2*x + 1"
    assert "contains x" in result["note"]


def test_division_by_zero_is_an_error_not_infinity() -> None:
    assert "undefined" in run_tool("math_evaluate", {"expression": "1/0"})["error"]


@pytest.mark.parametrize(
    ("expression", "must_mention"),
    [
        ("15% of 80", "15/100"),
        ("x^", "write powers as `x^2`"),
        ("3 × 4", "write `*`"),
        ("((2+3)", "closing parentheses"),
        ("percent change", "not a known function"),
        ("[1+2]*3", "( ) only"),
        ("x^2 = 4", "math_solve"),
        ("", "empty"),
        ("9" * 201, "split the problem"),
    ],
)
def test_a_bad_expression_gets_an_error_that_says_what_to_write(
    expression: str, must_mention: str
) -> None:
    error = run_tool("math_evaluate", {"expression": expression})["error"]
    assert must_mention in error


# ── math_solve ───────────────────────────────────────────────────────────────


def test_a_quadratic_gives_both_roots_and_the_working() -> None:
    result = run_tool("math_solve", {"equation": "x^2 - 5x + 6 = 0"})
    assert result["roots"] == ["2", "3"]
    assert "decimals" not in result  # whole numbers need no decimal form
    assert any("discriminant" in step for step in result["steps"])


def test_non_integer_roots_come_with_decimals() -> None:
    result = run_tool("math_solve", {"equation": "3x^2 - 7x + 2 = 0"})
    assert result["roots"] == ["1/3", "2"]
    assert result["decimals"] == ["0.3333333333", "2"]


def test_irrational_roots_stay_exact() -> None:
    result = run_tool("math_solve", {"equation": "x^2 = 50"})
    assert result["roots"] == ["-5*sqrt(2)", "5*sqrt(2)"]


def test_no_real_solution_is_an_answer_not_an_error() -> None:
    result = run_tool("math_solve", {"equation": "x^2 + 1 = 0"})
    assert result["roots"] == []
    assert "no real solutions" in result["note"]


def test_an_expression_without_equals_is_solved_for_zero() -> None:
    assert run_tool("math_solve", {"equation": "2x - 8"})["roots"] == ["4"]


def test_several_letters_need_a_variable_and_the_error_says_which() -> None:
    error = run_tool("math_solve", {"equation": "x + y = 3"})["error"]
    assert "x, y" in error and "`variable`" in error
    assert run_tool("math_solve", {"equation": "x + y = 3", "variable": "y"})["roots"] == ["3 - x"]


def test_a_variable_that_is_not_in_the_equation_is_named() -> None:
    error = run_tool("math_solve", {"equation": "2x = 8", "variable": "y"})["error"]
    assert "`y` does not appear" in error and "x" in error


@pytest.mark.parametrize("equation", ["x > 3", "x = 1 = 2", "x == 3"])
def test_inequalities_and_systems_are_refused_with_the_way_out(equation: str) -> None:
    assert "error" in run_tool("math_solve", {"equation": equation})


def test_an_equation_with_an_empty_side_says_which_side_is_missing() -> None:
    error = run_tool("math_solve", {"equation": "x^2 = "})["error"]
    assert "one side of the `=` is empty" in error


# ── math_compare ─────────────────────────────────────────────────────────────


def test_a_numeric_comparison_gives_a_verdict_and_the_values() -> None:
    result = run_tool("math_compare", {"quantity_a": "3^2", "quantity_b": "2^3"})
    assert (result["verdict"], result["a"], result["b"]) == ("A", "9", "8")
    assert result["why"] == "A - B = 1"


def test_comparing_an_integer_past_the_python_digit_limit_works() -> None:
    result = run_tool("math_compare", {"quantity_a": "factorial(2000)", "quantity_b": "1"})
    assert "error" not in result
    assert result["verdict"] == "A"
    assert len(result["a"]) < 40
    assert len(result["why"]) < 80


def test_equal_quantities_are_c() -> None:
    result = run_tool("math_compare", {"quantity_a": "2/4", "quantity_b": "0.5"})
    assert result["verdict"] == "C"


def test_close_irrationals_are_ordered_correctly() -> None:
    result = run_tool("math_compare", {"quantity_a": "sqrt(2)+sqrt(3)", "quantity_b": "sqrt(5)+1"})
    assert result["verdict"] == "B"


def test_a_comparison_that_flips_with_the_variable_is_d_with_the_values_that_show_it() -> None:
    result = run_tool("math_compare", {"quantity_a": "x^2", "quantity_b": "x"})
    assert result["verdict"] == "D"
    assert "x=-3" in result["why"] and "restriction" in result["note"]


def test_a_comparison_that_never_flips_is_evidence_not_a_verdict() -> None:
    result = run_tool("math_compare", {"quantity_a": "x^2 + 1", "quantity_b": "0"})
    assert result["verdict"] == "undetermined"
    assert "not proof" in result["why"]


def test_identical_expressions_are_equal_for_every_value() -> None:
    result = run_tool("math_compare", {"quantity_a": "(x+1)^2", "quantity_b": "x^2 + 2x + 1"})
    assert result["verdict"] == "C"


# ── the call itself ──────────────────────────────────────────────────────────


def test_an_unknown_tool_lists_the_real_ones() -> None:
    error = run_tool("math_integrate", {})["error"]
    assert "math_evaluate, math_solve, math_compare" in error


def test_a_missing_or_extra_argument_is_named() -> None:
    assert "missing required argument `expression`" in run_tool("math_evaluate", {})["error"]
    assert "no argument `expr`" in run_tool("math_evaluate", {"expression": "1", "expr": "2"})[
        "error"
    ]


def test_arguments_must_be_an_object() -> None:
    assert "JSON object" in run_tool("math_evaluate", ["1+1"])["error"]  # type: ignore[arg-type]


def test_a_wrong_type_is_an_error_not_a_crash() -> None:
    assert "must be a string" in run_tool("math_evaluate", {"expression": 5})["error"]


# ── in the sandbox: the same door as the verifier ────────────────────────────


async def test_a_tool_call_runs_in_a_child_process_and_returns_the_answer() -> None:
    result = await run_math_tool("math_evaluate", {"expression": "2 + 3 * 4"})
    assert result == {"value": "14"}


@pytest.mark.parametrize(
    "hostile",
    ['__import__("os").getpid()', "open('/etc/passwd').read()", "().__class__.__bases__"],
)
async def test_hostile_text_in_a_tool_call_is_rejected_not_run(hostile: str) -> None:
    result = await run_math_tool("math_evaluate", {"expression": hostile})
    assert "error" in result and "value" not in result


def test_a_runaway_computation_is_stopped_and_the_error_tells_the_model_to_simplify() -> None:
    result = run_math_tool_sync("math_evaluate", {"expression": "9^9^9^9"}, timeout=1.0)
    assert "took longer than 1 s" in result["error"]
    assert "smaller steps" in result["error"]


def test_a_malformed_call_is_answered_without_starting_a_child_process() -> None:
    assert "unknown tool" in run_math_tool_sync("nope", {})["error"]


# ── over HTTP: the route the agent calls ─────────────────────────────────────

client = TestClient(mcp.sse_app())


def test_the_route_returns_the_tool_result() -> None:
    response = client.post("/tools/math_solve", json={"equation": "2x + 3 = 11"})
    assert response.status_code == 200
    assert response.json()["roots"] == ["4"]


def test_a_bad_call_over_http_is_still_a_200_with_an_error_for_the_model() -> None:
    response = client.post("/tools/math_evaluate", json={"expression": "15% of 80"})
    assert response.status_code == 200
    assert "15/100" in response.json()["error"]


def test_an_unknown_tool_over_http_is_a_404_that_lists_the_real_ones() -> None:
    response = client.post("/tools/math_integrate", json={})
    assert response.status_code == 404
    assert "math_compare" in response.json()["error"]


def test_a_body_that_is_not_json_is_an_error_not_a_server_failure() -> None:
    response = client.post("/tools/math_evaluate", content=b"not json")
    assert response.status_code == 200 and "JSON" in response.json()["error"]


def test_the_verify_route_is_unchanged() -> None:
    body = client.post("/tools/verify_math", json={"expression": "2+3", "expected": "5"}).json()
    assert body["verified"] is True
