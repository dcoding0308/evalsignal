import pytest

from evalsignal.power import (
    EXACT_LIMIT,
    approximate_items,
    exact_power,
    is_approximate,
    minimum_detectable_effect,
    required_items,
)


def test_required_items_reaches_the_target_with_exact_power() -> None:
    n = required_items(0.05, 0.15)
    assert not is_approximate(n)
    assert exact_power(n, 0.05, 0.15) >= 0.8
    assert exact_power(n - 1, 0.05, 0.15) < 0.8
    assert n >= approximate_items(0.05, 0.15)  # the approximation under-counts


def test_above_the_exact_limit_the_approximation_is_returned_unverified() -> None:
    # A 0.1-point change with 5% of items flipping needs far more than EXACT_LIMIT items.
    n = required_items(0.001, 0.05)
    assert n > EXACT_LIMIT
    assert is_approximate(n)
    assert n == approximate_items(0.001, 0.05)


def test_minimum_detectable_effect_above_the_limit_uses_the_formula() -> None:
    mde = minimum_detectable_effect(EXACT_LIMIT * 2, 0.05)
    assert is_approximate(EXACT_LIMIT * 2)
    assert mde == pytest.approx(2.8016 * (0.05 / (EXACT_LIMIT * 2)) ** 0.5, rel=1e-3)
