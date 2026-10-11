"""Exact, hand-checkable cases for the paired statistics."""

import numpy as np
import pytest
from scipy import stats

from evalsignal.paired import IntervalMethod, compare_binary, compare_scores, mcnemar_exact
from evalsignal.power import (
    approximate_items,
    exact_power,
    minimum_detectable_effect,
    required_items,
)


def pairs(both: int, only_base: int, only_cand: int, neither: int):  # type: ignore[no-untyped-def]
    baseline = np.repeat([True, True, False, False], [both, only_base, only_cand, neither])
    candidate = np.repeat([True, False, True, False], [both, only_base, only_cand, neither])
    return baseline, candidate


def test_mcnemar_exact_matches_the_binomial_by_hand() -> None:
    # 5 lost, 1 gained: p = 2 * P(X <= 1), X ~ Bin(6, 1/2) = 2 * 7/64.
    assert mcnemar_exact(5, 1) == pytest.approx(2 * 7 / 64)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5**10)


def test_binary_comparison_counts_and_difference() -> None:
    baseline, candidate = pairs(70, 10, 4, 16)
    result = compare_binary(baseline, candidate)
    assert (result.n, result.only_baseline, result.only_candidate) == (100, 10, 4)
    assert result.baseline_mean == pytest.approx(0.80)
    assert result.candidate_mean == pytest.approx(0.74)
    assert result.difference == pytest.approx(-0.06)
    assert result.discordance == pytest.approx(0.14)
    assert result.ci_low < -0.06 < result.ci_high


@pytest.mark.parametrize(
    "method", [IntervalMethod.WALD, IntervalMethod.AGRESTI_MIN, IntervalMethod.BOOTSTRAP]
)
def test_every_interval_contains_the_estimate_and_narrows_with_more_items(
    method: IntervalMethod,
) -> None:
    small = compare_binary(*pairs(70, 10, 4, 16), interval=method)
    large = compare_binary(*pairs(700, 100, 40, 160), interval=method)
    assert small.ci_low <= small.difference <= small.ci_high
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_wald_interval_matches_the_formula() -> None:
    result = compare_binary(*pairs(70, 10, 4, 16), interval=IntervalMethod.WALD)
    variance = ((10 + 4) - (4 - 10) ** 2 / 100) / 100**2
    half = stats.norm.ppf(0.975) * variance**0.5
    assert (result.ci_low, result.ci_high) == pytest.approx((-0.06 - half, -0.06 + half))


def test_no_change_gives_no_evidence() -> None:
    result = compare_binary(*pairs(80, 0, 0, 20))
    assert (result.difference, result.p_value) == (0.0, 1.0)


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        compare_binary(np.array([True, False]), np.array([True]))
    with pytest.raises(ValueError):
        compare_binary(np.array([], dtype=bool), np.array([], dtype=bool))
    with pytest.raises(ValueError):
        compare_binary(*pairs(1, 1, 1, 1), interval=IntervalMethod.T)


def test_scores_t_interval_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    baseline = rng.normal(6.0, 1.5, 80)
    candidate = baseline + rng.normal(0.4, 1.0, 80)
    result = compare_scores(baseline, candidate, interval=IntervalMethod.T)
    expected = stats.ttest_rel(candidate, baseline).confidence_interval(0.95)
    assert (result.ci_low, result.ci_high) == pytest.approx((expected.low, expected.high))
    assert result.p_value < 0.01  # a clear shift is detected by the permutation test


def test_scores_with_no_change_are_not_significant() -> None:
    scores = np.linspace(1, 10, 50)
    result = compare_scores(scores, scores.copy())
    assert (result.difference, result.p_value) == (0.0, 1.0)


def test_clustered_comparison_reports_clusters_and_wider_uncertainty() -> None:
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(20), 10)
    baseline = rng.random(200) < 0.7
    candidate = baseline.copy()
    for cluster in range(0, 20, 2):  # whole clusters flip together
        candidate[labels == cluster] = ~baseline[labels == cluster]
    plain = compare_binary(baseline, candidate)
    clustered = compare_binary(baseline, candidate, clusters=labels)
    assert clustered.clusters == 20
    assert (clustered.ci_high - clustered.ci_low) > (plain.ci_high - plain.ci_low)


def test_approximate_items_matches_the_closed_form() -> None:
    z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
    expected = (z_a * 0.12**0.5 + z_b * (0.12 - 0.03**2) ** 0.5) ** 2 / 0.03**2
    assert approximate_items(0.03, 0.12) == int(np.ceil(expected))


def test_required_items_reaches_the_promised_exact_power() -> None:
    n = required_items(0.03, 0.12)
    assert n >= approximate_items(0.03, 0.12)  # the approximation is optimistic
    assert exact_power(n, 0.03, 0.12) >= 0.8 > exact_power(n - 1, 0.03, 0.12)


def test_exact_power_by_hand_for_a_tiny_case() -> None:
    # n=1: one discordant item can never reach p < 0.05, so power is 0.
    assert exact_power(1, 0.1, 0.2) == 0.0
    # With no true change, power equals the (conservative) false-alarm rate.
    assert exact_power(200, 1e-9, 0.2) <= 0.05


def test_minimum_detectable_effect_is_the_inverse_of_required_items() -> None:
    effect = minimum_detectable_effect(500, 0.15)
    assert required_items(effect, 0.15) <= 500 < required_items(effect * 0.97, 0.15)


@pytest.mark.parametrize(("effect", "discordance"), [(0.0, 0.1), (0.2, 0.1), (0.05, 0.0)])
def test_impossible_power_questions_are_rejected(effect: float, discordance: float) -> None:
    with pytest.raises(ValueError):
        required_items(effect, discordance)
