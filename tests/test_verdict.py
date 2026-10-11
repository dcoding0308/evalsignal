import numpy as np

from evalsignal.paired import compare_binary, compare_scores
from evalsignal.verdict import Outcome, decide


def pairs(both: int, only_base: int, only_cand: int, neither: int):  # type: ignore[no-untyped-def]
    baseline = np.repeat([True, True, False, False], [both, only_base, only_cand, neither])
    candidate = np.repeat([True, False, True, False], [both, only_base, only_cand, neither])
    return baseline, candidate


def test_small_noisy_drop_is_not_enough_evidence_and_says_what_would_be() -> None:
    verdict = decide(compare_binary(*pairs(74, 8, 5, 13)), binary=True)
    assert verdict.outcome is Outcome.NOT_ENOUGH_EVIDENCE
    assert verdict.headline.startswith("Not enough evidence")
    assert "Accuracy went from 82.0% to 79.0% (-3.0 points" in verdict.details[0]
    assert verdict.minimum_detectable is not None and verdict.minimum_detectable > 0.03
    assert verdict.items_needed is not None and verdict.items_needed > 100


def test_large_clear_drop_is_a_regression() -> None:
    verdict = decide(compare_binary(*pairs(600, 80, 20, 300)), binary=True)
    assert verdict.outcome is Outcome.REGRESSION
    assert verdict.headline.startswith("Real regression: -6.0 points")


def test_clear_gain_is_an_improvement() -> None:
    verdict = decide(compare_binary(*pairs(600, 20, 80, 300)), binary=True)
    assert verdict.outcome is Outcome.IMPROVEMENT


def test_tight_interval_inside_the_margin_is_no_meaningful_change() -> None:
    verdict = decide(compare_binary(*pairs(7000, 150, 150, 2700)), binary=True, margin=0.01)
    assert verdict.outcome is Outcome.NO_MEANINGFUL_CHANGE
    # Without a margin the same data only says "not enough evidence".
    assert decide(compare_binary(*pairs(7000, 150, 150, 2700)), binary=True).outcome is (
        Outcome.NOT_ENOUGH_EVIDENCE
    )


def test_scores_use_plain_numbers() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(6, 1, 40)
    verdict = decide(
        compare_scores(base, base + rng.normal(0.05, 1, 40)), binary=False, score_sd=1.0
    )
    assert "Mean score went from" in verdict.details[0]
    assert "points" not in verdict.details[0]


def test_significant_but_entirely_within_margin_is_no_meaningful_change() -> None:
    # 100,000 items, -0.2 points: clearly detectable, but the whole range is inside ±0.5 points.
    comparison = compare_binary(*pairs(80000, 600, 400, 19000))
    assert comparison.p_value < 0.001
    verdict = decide(comparison, binary=True, margin=0.005)
    assert verdict.outcome is Outcome.NO_MEANINGFUL_CHANGE
    assert any("statistically detectable" in d for d in verdict.details)
    # Without a margin, the same data is a real regression.
    assert decide(comparison, binary=True).outcome is Outcome.REGRESSION


def test_significant_change_whose_range_crosses_the_margin_is_still_a_regression() -> None:
    # -6 points, margin 7 points: the measured change is inside, but the range reaches past -7.
    comparison = compare_binary(*pairs(600, 80, 20, 300))
    assert comparison.ci_low < -0.07
    verdict = decide(comparison, binary=True, margin=0.07)
    assert verdict.outcome is Outcome.REGRESSION
    assert any("range reaches beyond it" in d for d in verdict.details)
