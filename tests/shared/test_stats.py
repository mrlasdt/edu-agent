"""The shared statistics helpers: one answer for the gate, the eval reports and the importers."""
import pytest

from shared.src.shared.stats import (
    is_half_band,
    percentile,
    round_half_band,
    to_half_band,
    wilson,
)


def test_wilson_matches_the_published_example():
    lo, hi = wilson(4, 6) or (0, 0)
    assert (round(lo, 2), round(hi, 2)) == (0.30, 0.90)
    assert wilson(0, 0) is None


def test_wilson_stays_wide_at_a_perfect_small_sample():
    low, high = wilson(8, 8) or (0, 0)
    assert high == 1.0
    assert 0.6 < low < 0.7


def test_percentile_is_nearest_rank():
    assert percentile([], 0.5) is None
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 100.0], 0.5) == 3.0
    ten = list(range(1, 11))
    assert percentile(ten, 0.5) == 5
    assert percentile(ten, 0.95) == 10
    assert percentile([1, 2], 0.5) == 1
    assert percentile([3, 1, 2], 0.5) == 2  # unsorted input


@pytest.mark.parametrize(
    ("value", "expected"),
    [(6.25, 6.5), (6.24, 6.0), (6.625, 6.5), (6.75, 7.0), (5.0, 5.0)],
)
def test_round_half_band_rounds_a_quarter_up(value, expected):
    assert round_half_band(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("6.5", 6.5), (7, 7.0), ("0", 0.0), ("9", 9.0), ("6.3", None), ("9.5", None),
     ("-0.5", None), ("high", None), (None, None)],
)
def test_to_half_band_accepts_only_bands_a_marker_can_award(value, expected):
    assert to_half_band(value) == expected
    assert is_half_band(6.5) and not is_half_band(6.25)
