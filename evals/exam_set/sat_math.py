"""
Original, parameterised SAT Math templates with machine-verified answers.

Nothing here is College Board text: each template is our own wording over the public SAT Math
content domains (Algebra, Advanced Math, Problem-Solving and Data Analysis, Geometry and
Trigonometry), with parameters drawn from a seeded RNG under constraints that keep answers
clean.

Per item:
  - the key is computed with sympy (`primary`);
  - a second, independent computation (`secondary`: exact `Fraction`/integer arithmetic, a
    different formula or a substitution check) must agree, otherwise generation raises;
  - multiple-choice distractors come from coded error models (a sign slip, a forgotten step...),
    each computed exactly and de-duplicated against the key;
  - `expression` + `ground_truth` follow the GRE quant files, so the product's sympy verifier
    (`services/math_verifier`) checks every item too (see tests);
  - the template id and seed are stored, and `regenerate` rebuilds an item bit for bit.

Regenerate the committed file with `python -m evals.exam_set generate-sat`.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import sympy as sp

from evals.exam_set.schema import (
    Design,
    ExamItem,
    LabelProvenance,
    LabelTier,
    Source,
)

FILE_NAME = "sat_math.jsonl"
SEEDS: tuple[int, ...] = (101, 202, 303)
GENERATED_DATE = "2026-10-24"
MAX_ATTEMPTS = 200


class VerificationDisagreementError(AssertionError):
    """The primary and secondary computations of an item disagree: never ship such an item."""


@dataclass
class Draft:
    """What a template returns for one parameter draw."""

    question: str
    expression: str  # evaluates (or solves) to the key under the product's sympy verifier
    primary: sp.Basic  # computed with sympy
    secondary: Fraction | sp.Basic  # computed another way; must equal `primary`
    distractors: dict[str, sp.Basic]  # error-model name -> wrong value
    secondary_name: str
    domain: str
    pi_multiple: bool = False  # answer of the form k*pi (secondary is then k)


class RedrawError(Exception):
    """Control flow: this parameter draw is unusable, draw again."""


Template = Callable[[random.Random], Draft]


def _r(n: int | Fraction) -> sp.Rational:
    f = Fraction(n)
    return sp.Rational(f.numerator, f.denominator)


def _fmt(value: sp.Basic) -> str:
    """Plain-text form of an answer: integers and fractions, or k*pi."""
    value = sp.nsimplify(value)
    if value.has(sp.pi):
        k = sp.simplify(value / sp.pi)
        return "pi" if k == 1 else f"{_fmt(k)}*pi"
    if isinstance(value, sp.Rational) and value.q != 1:
        return f"{value.p}/{value.q}"
    return str(value)


def _signed(n: int) -> str:
    return f"+ {n}" if n >= 0 else f"- {-n}"


def _show(value: sp.Basic) -> str:
    """Answer-choice text (what a student sees)."""
    return _fmt(value).replace("*pi", "π").replace("pi", "π")


# ── templates ────────────────────────────────────────────────────────────────


def t_linear_equation(rng: random.Random) -> Draft:
    a, c = rng.randint(2, 9), rng.randint(2, 9)
    x = rng.randint(-8, 12)
    if a == c:
        raise RedrawError
    b = rng.randint(-15, 15)
    d = a * x + b - c * x
    if b == 0 or d == 0:
        raise RedrawError
    xs = sp.symbols("x")
    primary = sp.solve(sp.Eq(a * xs + b, c * xs + d), xs)[0]
    secondary = Fraction(d - b, a - c)
    return Draft(
        question=f"If {a}x {_signed(b)} = {c}x {_signed(d)}, what is the value of x?",
        expression=f"{a}*x + ({b}) = {c}*x + ({d})",
        primary=primary,
        secondary=secondary,
        secondary_name="fraction_formula",
        distractors={
            "sign_slip_on_constants": _r(Fraction(b - d, a - c)),
            "added_coefficients": _r(Fraction(d - b, a + c)),
            "wrong_side_subtraction": _r(Fraction(d + b, a - c)),
            "off_by_one": _r(x + 1),
        },
        domain="Algebra",
    )


def t_line_intercept(rng: random.Random) -> Draft:
    m, b = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5]), rng.randint(-9, 9)
    x1, x2 = rng.sample(range(-4, 6), 2)
    y1, y2 = m * x1 + b, m * x2 + b
    mm, bb = sp.symbols("m b")
    sol = sp.solve([mm * x1 + bb - y1, mm * x2 + bb - y2], [mm, bb])
    secondary = Fraction(y1 * x2 - y2 * x1, x2 - x1)
    return Draft(
        question=(
            f"A line in the xy-plane passes through the points ({x1}, {y1}) and ({x2}, {y2}). "
            "What is the y-coordinate of the point where the line crosses the y-axis?"
        ),
        expression=f"(({y1})*({x2}) - ({y2})*({x1}))/(({x2}) - ({x1}))",
        primary=sol[bb],
        secondary=secondary,
        secondary_name="two_point_intercept_formula",
        distractors={
            "gave_slope": _r(m),
            "gave_x_intercept": _r(Fraction(-b, m)),
            "negated_intercept": _r(-b),
            "used_first_y": _r(y1),
        },
        domain="Algebra",
    )


def t_linear_system(rng: random.Random) -> Draft:
    x, y = rng.randint(1, 9), rng.randint(-6, 9)
    a, b = rng.randint(2, 5), rng.randint(2, 5)
    c, d = rng.randint(2, 5), rng.randint(2, 5)
    det = a * d - b * c
    if det == 0 or x == y:
        raise RedrawError
    e, f = a * x + b * y, c * x + d * y
    xs, ys = sp.symbols("x y")
    (sx, sy), *_ = sp.linsolve([a * xs + b * ys - e, c * xs + d * ys - f], [xs, ys])
    # Cramer's rule in exact arithmetic
    cx, cy = Fraction(e * d - b * f, det), Fraction(a * f - e * c, det)
    return Draft(
        question=(
            f"The system {a}x + {b}y = {e} and {c}x + {d}y = {f} has solution (x, y). "
            "What is the value of x - y?"
        ),
        expression=f"({e}*{d} - {b}*{f})/({det}) - ({a}*{f} - {e}*{c})/({det})",
        primary=sx - sy,
        secondary=cx - cy,
        secondary_name="cramers_rule",
        distractors={
            "x_plus_y": _r(x + y),
            "gave_x_only": _r(x),
            "reversed_sign": _r(y - x),
            "gave_y_only": _r(y),
        },
        domain="Algebra",
    )


def t_quadratic_roots(rng: random.Random) -> Draft:
    p, q = rng.sample([r for r in range(-9, 10) if r != 0], 2)
    if p + q == 0:
        raise RedrawError
    xs = sp.symbols("x")
    poly = sp.expand((xs - p) * (xs - q))
    b, c = int(-(p + q)), int(p * q)
    roots = sp.solve(poly, xs)
    # the secondary computation: substitute the roots back, then isqrt of the discriminant
    if any(r * r + b * r + c != 0 for r in (p, q)):
        raise VerificationDisagreementError("roots do not satisfy the polynomial")
    return Draft(
        question=(
            f"What is the positive difference between the two solutions of "
            f"x^2 {_signed(b)}x {_signed(c)} = 0?"
        ),
        expression=f"sqrt(({b})**2 - 4*({c}))",
        primary=sp.Abs(roots[0] - roots[1]),
        secondary=Fraction(math.isqrt(b * b - 4 * c)),
        secondary_name="discriminant_isqrt",
        distractors={
            "sum_of_solutions": _r(p + q),
            "product_of_solutions": _r(p * q),
            "difference_of_squares_of_discriminant": _r(b * b - 4 * c),
            "half_the_difference": _r(Fraction(abs(p - q), 2)),
        },
        domain="Advanced Math",
    )


def t_parabola_minimum(rng: random.Random) -> Draft:
    h, k = rng.randint(1, 9), rng.randint(-12, 12)
    xs = sp.symbols("x")
    f = xs**2 - 2 * h * xs + (h * h + k)
    vertex_x = sp.solve(sp.diff(f, xs), xs)[0]
    primary = f.subs(xs, vertex_x)
    secondary = Fraction(k)  # completing the square: (x - h)^2 + k
    return Draft(
        question=(
            f"For all real x, what is the minimum value of the function "
            f"f(x) = x^2 - {2 * h}x + {h * h + k}?"
        ),
        expression=f"{h * h + k} - {h}**2",
        primary=primary,
        secondary=secondary,
        secondary_name="completing_the_square",
        distractors={
            "gave_vertex_x": _r(h),
            "gave_constant_term": _r(h * h + k),
            "sign_slip": _r(-k),
            "used_h_not_h_squared": _r(h * h + k - h),
        },
        domain="Advanced Math",
    )


def t_exponential_growth(rng: random.Random) -> Draft:
    p0, period = rng.choice([40, 50, 75, 120, 150, 200]), rng.choice([2, 3, 4, 5])
    doublings = rng.randint(2, 5)
    n = period * doublings
    primary = sp.Integer(p0) * sp.Integer(2) ** sp.Integer(n // period)
    secondary = Fraction(p0)
    for _ in range(doublings):  # repeated doubling, no exponent formula
        secondary *= 2
    return Draft(
        question=(
            f"A bacteria culture has {p0} cells at time 0 and doubles every {period} hours. "
            f"How many cells are in the culture after {n} hours?"
        ),
        expression=f"{p0}*2**({n}/{period})",
        primary=primary,
        secondary=secondary,
        secondary_name="repeated_doubling",
        distractors={
            "multiplied_by_hours": _r(p0 * n),
            "doublings_added_not_multiplied": _r(p0 + doublings),
            "one_doubling_too_few": _r(p0 * 2 ** (doublings - 1)),
            "one_doubling_too_many": _r(p0 * 2 ** (doublings + 1)),
        },
        domain="Advanced Math",
    )


def t_percent_change(rng: random.Random) -> Draft:
    price = rng.choice([40, 60, 80, 120, 200, 250])
    up, down = rng.choice([10, 20, 25, 50]), rng.choice([10, 20, 25, 50])
    final = Fraction(price) * (100 + up) * (100 - down) / 10_000
    if final.denominator != 1:
        raise RedrawError
    primary = sp.Integer(price) * (1 + _r(Fraction(up, 100))) * (1 - _r(Fraction(down, 100)))
    # the secondary computation: two explicit steps in exact Fractions
    step1 = Fraction(price) + Fraction(price) * up / 100
    secondary = step1 - step1 * down / 100
    return Draft(
        question=(
            f"A jacket's price of ${price} is increased by {up}%. The new price is then "
            f"decreased by {down}%. What is the final price, in dollars?"
        ),
        expression=f"{price}*(1 + {up}/100)*(1 - {down}/100)",
        primary=primary,
        secondary=secondary,
        secondary_name="two_step_fractions",
        distractors={
            "net_percent_added": _r(Fraction(price) * (100 + up - down) / 100),
            "decrease_applied_first_to_original": _r(
                Fraction(price) + Fraction(price) * up / 100 - Fraction(price) * down / 100
            ),
            "only_increase": _r(Fraction(price) * (100 + up) / 100),
            "only_decrease": _r(Fraction(price) * (100 - down) / 100),
        },
        domain="Problem-Solving and Data Analysis",
    )


def t_missing_mean(rng: random.Random) -> Draft:
    values = [rng.randint(4, 40) for _ in range(4)]
    missing = rng.randint(4, 40)
    mean = Fraction(sum(values) + missing, 5)
    if mean.denominator != 1:
        raise RedrawError
    xs = sp.symbols("x")
    primary = sp.solve(sp.Eq((sum(values) + xs) / sp.Integer(5), sp.Integer(int(mean))), xs)[0]
    secondary = 5 * mean - sum(values)
    listed = ", ".join(str(v) for v in values)
    return Draft(
        question=(
            f"The mean of five numbers is {int(mean)}. Four of the numbers are {listed}. "
            "What is the fifth number?"
        ),
        expression=f"5*{int(mean)} - {sum(values)}",
        primary=primary,
        secondary=secondary,
        secondary_name="total_minus_known",
        distractors={
            "used_four_numbers_total": _r(int(mean) * 4 - sum(values)),
            "used_mean_as_missing": _r(int(mean)),
            "total_over_five": _r(Fraction(sum(values), 5)),
            "total_not_subtracted": _r(5 * int(mean)),
        },
        domain="Problem-Solving and Data Analysis",
    )


def t_rate_proportion(rng: random.Random) -> Draft:
    rate, minutes = rng.choice([12, 15, 18, 24, 36]), rng.choice([3, 4, 6, 9])
    hours = rng.choice([2, 3, 5])
    items_per_minute = Fraction(rate, minutes)
    total = items_per_minute * 60 * hours
    if total.denominator != 1:
        raise RedrawError
    xs = sp.symbols("x")
    primary = sp.solve(sp.Eq(sp.Integer(rate) / minutes, xs / (60 * hours)), xs)[0]
    secondary = Fraction(rate * 60 * hours, minutes)
    return Draft(
        question=(
            f"A machine fills {rate} boxes every {minutes} minutes at a constant rate. "
            f"How many boxes does it fill in {hours} hours?"
        ),
        expression=f"{rate}/{minutes}*60*{hours}",
        primary=primary,
        secondary=secondary,
        secondary_name="cross_multiplication",
        distractors={
            "hours_treated_as_minutes": _r(items_per_minute * hours),
            "inverted_rate": _r(Fraction(minutes, rate) * 60 * hours),
            "forgot_hours_conversion": _r(items_per_minute * 60),
            "added_instead_of_scaled": _r(rate + 60 * hours),
        },
        domain="Problem-Solving and Data Analysis",
    )


def t_hypotenuse(rng: random.Random) -> Draft:
    triple = rng.choice([(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)])
    k = rng.randint(2, 6)
    a, b, c = (k * t for t in triple)
    primary = sp.sqrt(sp.Integer(a) ** 2 + sp.Integer(b) ** 2)
    if math.hypot(a, b) != c or c * c != a * a + b * b:
        raise VerificationDisagreementError("hypotenuse mismatch")
    return Draft(
        question=(
            f"A right triangle has legs of length {a} and {b}. What is the length of its "
            "hypotenuse?"
        ),
        expression=f"sqrt({a}**2 + {b}**2)",
        primary=primary,
        secondary=Fraction(c),
        secondary_name="math_hypot_and_integer_square_check",
        distractors={
            "sum_of_legs": _r(a + b),
            "difference_of_squares": _r(abs(b * b - a * a)),
            "sum_of_squares": _r(a * a + b * b),
            "longer_leg_plus_one": _r(max(a, b) + 1),
        },
        domain="Geometry and Trigonometry",
    )


def t_circle_area(rng: random.Random) -> Draft:
    r = rng.randint(2, 12)
    circumference = 2 * r
    rs = sp.symbols("r", positive=True)
    radius = sp.solve(sp.Eq(2 * sp.pi * rs, circumference * sp.pi), rs)[0]
    primary = sp.pi * radius**2
    # the secondary computation: the multiple of pi from the radius, checked numerically
    if abs(float(primary) - math.pi * r * r) > 1e-9:
        raise VerificationDisagreementError("area mismatch")
    return Draft(
        question=(
            f"A circle has a circumference of {circumference}π. What is the area of the circle?"
        ),
        expression=f"{r * r}*pi",
        primary=primary,
        secondary=_r(r * r),
        secondary_name="radius_squared_and_numeric_check",
        distractors={
            "used_diameter_as_radius": _r(circumference**2) * sp.pi,
            "gave_circumference_multiple": _r(circumference) * sp.pi,
            "halved_instead_of_squared": _r(r) * sp.pi,
            "area_formula_with_two": _r(2 * r * r) * sp.pi,
        },
        domain="Geometry and Trigonometry",
        pi_multiple=True,
    )


@dataclass
class TemplateSpec:
    id: str
    build: Template
    kind: Literal["multiple_choice_single", "numeric_entry"]


TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec("sat_linear_equation", t_linear_equation, "multiple_choice_single"),
    TemplateSpec("sat_line_intercept", t_line_intercept, "numeric_entry"),
    TemplateSpec("sat_linear_system", t_linear_system, "multiple_choice_single"),
    TemplateSpec("sat_quadratic_roots", t_quadratic_roots, "multiple_choice_single"),
    TemplateSpec("sat_parabola_minimum", t_parabola_minimum, "numeric_entry"),
    TemplateSpec("sat_exponential_growth", t_exponential_growth, "multiple_choice_single"),
    TemplateSpec("sat_percent_change", t_percent_change, "numeric_entry"),
    TemplateSpec("sat_missing_mean", t_missing_mean, "multiple_choice_single"),
    TemplateSpec("sat_rate_proportion", t_rate_proportion, "multiple_choice_single"),
    TemplateSpec("sat_hypotenuse", t_hypotenuse, "multiple_choice_single"),
    TemplateSpec("sat_circle_area", t_circle_area, "numeric_entry"),
)
_BY_ID = {t.id: t for t in TEMPLATES}


# ── generation ───────────────────────────────────────────────────────────────


def _agree(draft: Draft) -> bool:
    primary = sp.nsimplify(draft.primary)
    if isinstance(draft.secondary, Fraction):
        second = _r(draft.secondary)
    else:
        second = sp.nsimplify(draft.secondary)
    if draft.pi_multiple:
        return bool(sp.simplify(primary - second * sp.pi) == 0)
    return bool(sp.simplify(primary - second) == 0)


def _choices(
    spec: TemplateSpec, draft: Draft, key: sp.Basic, rng: random.Random
) -> tuple[dict[str, str], str]:
    seen = {_fmt(key)}
    wrong: list[sp.Basic] = []
    for value in draft.distractors.values():
        text = _fmt(value)
        if text not in seen:
            seen.add(text)
            wrong.append(sp.nsimplify(value))
    # Distractors of the same kind as the key (whole numbers for a whole-number key) read as
    # plausible; odd-looking fractions are only used to make up the numbers.
    def whole(v: sp.Basic) -> bool:
        return bool(v.is_integer or v.has(sp.pi))

    wrong.sort(key=lambda v: 0 if whole(v) == whole(key) else 1)
    if len(wrong) < 3:
        raise RedrawError  # not enough distinct distractors for this draw
    options = [key, *wrong[:3]]
    rng.shuffle(options)
    letters = "ABCD"
    choices = {letters[i]: _show(v) for i, v in enumerate(options)}
    answer = letters[next(i for i, v in enumerate(options) if v == key)]
    return choices, answer


def generate(template_id: str, seed: int) -> ExamItem:
    """Build one verified item. Deterministic in (template_id, seed)."""
    spec = _BY_ID[template_id]
    rng = random.Random(f"{template_id}:{seed}")
    for _ in range(MAX_ATTEMPTS):
        try:
            draft = spec.build(rng)
            if not _agree(draft):
                raise VerificationDisagreementError(
                    f"{template_id} seed {seed}: sympy {draft.primary} != "
                    f"{draft.secondary_name} {draft.secondary}"
                )
            key = sp.nsimplify(draft.primary)
            choices, answer = (
                _choices(spec, draft, key, rng)
                if spec.kind == "multiple_choice_single"
                else (None, None)
            )
        except RedrawError:
            continue
        return ExamItem(
            id=f"{template_id}_{seed}",
            exam="sat",
            skill="sat_math",
            task_type=spec.kind,
            prompt=draft.question,
            question_type=spec.kind,
            expression=draft.expression,
            ground_truth=_fmt(key),
            choices=choices,
            answer_choice=answer,
            label_provenance=LabelProvenance(
                tier=LabelTier.solver,
                date=GENERATED_DATE,
                note=f"sympy and an independent {draft.secondary_name} computation agree",
            ),
            source=Source(kind="original", license="original to this repository (MIT)"),
            design=Design(
                generator=f"template:{template_id}",
                seed=seed,
                checks=["sympy", draft.secondary_name],
            ),
            extras={"domain": draft.domain},
        )
    raise RuntimeError(f"{template_id} seed {seed}: no valid parameter draw in {MAX_ATTEMPTS}")


def generate_all() -> list[ExamItem]:
    return [generate(t.id, seed) for t in TEMPLATES for seed in SEEDS]


def regenerate(item: ExamItem) -> ExamItem:
    """Rebuild an item from the template id and seed it stores."""
    assert item.design.seed is not None and item.design.generator.startswith("template:")
    return generate(item.design.generator.removeprefix("template:"), item.design.seed)
