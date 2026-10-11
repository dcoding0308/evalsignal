"""How many items does an evaluation need?

For pass/fail outcomes compared with McNemar's test, power depends on two
things: the change you want to detect (in accuracy points) and the share of
items whose outcome flips between runs (the *discordance*). The more items
flip, the noisier the comparison, so the more items you need.

Two calculations:

* ``approximate_items``: the normal approximation of Connor (1987),

      n = (z_a * sqrt(p_d) + z_b * sqrt(p_d - delta^2))^2 / delta^2

  where ``p_d`` is the discordance and ``delta`` the true change.
* ``exact_power``: the power of the *exact* McNemar test actually used, by
  summing over every possible number of discordant items. Simulation showed
  the approximation over-promises (about 76-79% real power when 80% was
  promised), because the exact test is slightly conservative; so
  ``required_items`` starts from the approximation and increases n until the
  exact power reaches the target.

Above ``EXACT_LIMIT`` items the exact check is skipped and the normal
approximation is returned as is (it becomes accurate at that size, and the
exact sum gets slow). Callers say so in their output; ``is_approximate``
tells them when.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

# Above this many items the exact sum gets slow and the approximation is accurate.
EXACT_LIMIT = 100_000


def is_approximate(n: int) -> bool:
    """True when results for ``n`` items come from the normal approximation, not exact power."""
    return n > EXACT_LIMIT


def required_items(
    effect: float, discordance: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Items needed so the exact McNemar test detects ``effect`` with the given power.

    Verified with exact power up to ``EXACT_LIMIT`` items; above that, the
    normal approximation is returned unverified (see ``is_approximate``).
    """
    n = approximate_items(effect, discordance, alpha, power)
    if is_approximate(n):
        return n
    step = max(1, n // 100)
    while exact_power(n, effect, discordance, alpha) < power:
        n += step
    # Binary search back to the smallest n (within the last step) that reaches the target.
    low, high = max(1, n - step), n
    while low < high:
        middle = (low + high) // 2
        if exact_power(middle, effect, discordance, alpha) >= power:
            high = middle
        else:
            low = middle + 1
    return high


def exact_power(n: int, effect: float, discordance: float, alpha: float = 0.05) -> float:
    """Probability that the exact two-sided McNemar test rejects, for a true change ``effect``.

    The number of discordant items K ~ Binomial(n, discordance); given K = k,
    the number that improved C ~ Binomial(k, q) with q = (d + effect) / (2d).
    The test rejects when C falls in the exact binomial rejection region.
    """
    effect = abs(effect)
    _check(effect, discordance, alpha, 0.5)
    q = (discordance + effect) / (2 * discordance)
    k = np.arange(1, n + 1)
    weight = stats.binom.pmf(k, n, discordance)
    # Largest c_low with 2 * P(C <= c_low | 1/2) <= alpha, for every k at once.
    c_low = stats.binom.ppf(alpha / 2, k, 0.5).astype(np.int64)
    c_low -= (2 * stats.binom.cdf(c_low, k, 0.5) > alpha).astype(np.int64)
    possible = c_low >= 0  # too few discordant items: no outcome is significant
    reject = np.where(
        possible,
        stats.binom.cdf(c_low, k, q) + stats.binom.sf(k - c_low - 1, k, q),
        0.0,
    )
    return float((weight * reject).sum())


def approximate_items(
    effect: float, discordance: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Connor's normal approximation (slightly optimistic for the exact test)."""
    effect = abs(effect)
    _check(effect, discordance, alpha, power)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = (z_a * math.sqrt(discordance) + z_b * math.sqrt(discordance - effect**2)) ** 2 / effect**2
    return math.ceil(float(n))


def minimum_detectable_effect(
    n: int, discordance: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest change (as a fraction) that ``n`` items detect with the given power.

    Bisection on the exact power itself (which grows with the effect), so
    each step costs one vectorised power evaluation.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    _check(discordance, discordance, alpha, power)
    if is_approximate(n):  # normal approximation, unverified
        z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
        return float(min(discordance, z * math.sqrt(discordance / n)))
    if exact_power(n, discordance, discordance, alpha) < power:
        return float("nan")  # not even the largest possible change is reliably detectable
    low, high = 1e-9, discordance
    for _ in range(40):
        middle = (low + high) / 2
        if exact_power(n, middle, discordance, alpha) < power:
            low = middle
        else:
            high = middle
    return high


def _check(effect: float, discordance: float, alpha: float, power: float) -> None:
    if not 0 < discordance <= 1:
        raise ValueError("discordance must be in (0, 1]")
    if not 0 < effect <= discordance:
        raise ValueError("the change cannot exceed the share of items that flip")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
