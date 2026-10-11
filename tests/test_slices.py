import numpy as np
import pytest

from evalsignal.slices import benjamini_hochberg, compare_slices, holm


def test_holm_matches_a_worked_example() -> None:
    # p = 0.01, 0.04, 0.03 -> sorted 0.01*3=0.03, 0.03*2=0.06, max(0.06, 0.04*1)=0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bh_matches_a_worked_example() -> None:
    # p = 0.01, 0.04, 0.03, m=3 -> 0.03, 0.04, 0.04 (monotone from the top)
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_adjustments_never_shrink_p_values_and_cap_at_one() -> None:
    p = [0.5, 0.9, 0.001, 0.2]
    for adjusted in (holm(p), benjamini_hochberg(p)):
        assert all(a >= raw for a, raw in zip(adjusted, p, strict=True))
        assert max(adjusted) <= 1.0
    assert all(h >= b for h, b in zip(holm(p), benjamini_hochberg(p), strict=True))


def test_slices_find_the_one_that_really_changed() -> None:
    rng = np.random.default_rng(0)
    names = np.repeat(np.array(["math", "law", "history"]), 400)
    baseline = rng.random(1200) < 0.7
    candidate = baseline.copy()
    math = names == "math"
    flip = math & (rng.random(1200) < 0.3) & baseline  # math loses many correct answers
    candidate[flip] = False
    results = {r.name: r for r in compare_slices(names, baseline, candidate, binary=True)}
    assert results["math"].p_holm < 0.001
    assert results["law"].comparison.difference == 0.0 and results["law"].p_holm == 1.0
