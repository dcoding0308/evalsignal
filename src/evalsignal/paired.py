"""Paired comparisons of two evaluation runs on the same items.

Both runs answer the same questions, so the right comparison is *paired*:
item i in the baseline is compared with item i in the candidate. Pairing
removes the variation between easy and hard questions, which is why a paired
test detects real changes with far fewer items than comparing two
independent percentages.

Pass/fail outcomes
    The only items that carry information about a change are the
    *discordant* ones: passed before but not after (``b``), or after but not
    before (``c``). McNemar's exact test asks whether ``b`` and ``c`` could
    plausibly come from a fair coin. The accuracy difference is
    ``(c - b) / n``.

Scores (any number per item)
    The per-item differences are analysed directly: the mean difference,
    with a paired t interval or a bootstrap interval, and a sign-flip
    permutation test, which assumes only that, with no real change, each
    difference is as likely to be positive as negative.

Clustered items
    Items that are not independent (several questions about one document,
    several turns of one conversation) should be resampled together. Passing
    ``clusters`` makes intervals and tests resample or flip whole clusters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy import stats

Bools = npt.NDArray[np.bool_]
Floats = npt.NDArray[np.float64]


class IntervalMethod(StrEnum):
    WALD = "wald"  # textbook normal approximation for a paired difference
    AGRESTI_MIN = "agresti_min"  # Wald after adding 0.5 to each cell (Agresti & Min, 2005)
    BOOTSTRAP = "bootstrap"  # percentile bootstrap over items (or clusters)
    T = "t"  # paired t interval (scores)


@dataclass(frozen=True, slots=True)
class Comparison:
    n: int
    baseline_mean: float
    candidate_mean: float
    difference: float  # candidate minus baseline
    ci_low: float
    ci_high: float
    confidence: float
    p_value: float
    test: str
    interval: str
    # Pass/fail only: how the items moved between the two runs.
    only_baseline: int | None = None  # passed before, failed after
    only_candidate: int | None = None  # failed before, passed after
    clusters: int | None = None

    @property
    def discordance(self) -> float | None:
        """Share of items whose outcome changed; drives statistical power."""
        if self.only_baseline is None or self.only_candidate is None:
            return None
        return (self.only_baseline + self.only_candidate) / self.n


def compare_binary(
    baseline: Bools,
    candidate: Bools,
    confidence: float = 0.95,
    interval: IntervalMethod = IntervalMethod.AGRESTI_MIN,
    clusters: npt.NDArray[np.int64] | None = None,
    bootstrap_rounds: int = 4000,
    seed: int = 0,
) -> Comparison:
    """Compare pass/fail outcomes of two runs on the same items."""
    baseline, candidate = _validate(
        np.asarray(baseline, dtype=bool), np.asarray(candidate, dtype=bool)
    )
    n = len(baseline)
    only_baseline = int((baseline & ~candidate).sum())
    only_candidate = int((~baseline & candidate).sum())
    difference = (only_candidate - only_baseline) / n
    diffs = candidate.astype(np.float64) - baseline.astype(np.float64)

    if clusters is not None:
        low, high = _cluster_bootstrap(diffs, clusters, confidence, bootstrap_rounds, seed)
        p_value = _sign_flip_p(diffs, clusters, bootstrap_rounds, seed)
        test, method = "cluster sign-flip permutation", IntervalMethod.BOOTSTRAP
    else:
        p_value = mcnemar_exact(only_baseline, only_candidate)
        test, method = "McNemar exact", interval
        if interval == IntervalMethod.BOOTSTRAP:
            low, high = _bootstrap_counts(
                only_baseline, only_candidate, n, confidence, bootstrap_rounds, seed
            )
        elif interval == IntervalMethod.WALD:
            low, high = _paired_wald(only_baseline, only_candidate, n, confidence, add=0.0)
        elif interval == IntervalMethod.AGRESTI_MIN:
            low, high = _paired_wald(only_baseline, only_candidate, n, confidence, add=0.5)
        else:
            raise ValueError(f"interval {interval} does not apply to pass/fail outcomes")
    return Comparison(
        n=n,
        baseline_mean=float(baseline.mean()),
        candidate_mean=float(candidate.mean()),
        difference=difference,
        ci_low=max(-1.0, low),
        ci_high=min(1.0, high),
        confidence=confidence,
        p_value=p_value,
        test=test,
        interval=method.value,
        only_baseline=only_baseline,
        only_candidate=only_candidate,
        clusters=len(np.unique(clusters)) if clusters is not None else None,
    )


def compare_scores(
    baseline: Floats,
    candidate: Floats,
    confidence: float = 0.95,
    interval: IntervalMethod = IntervalMethod.BOOTSTRAP,
    clusters: npt.NDArray[np.int64] | None = None,
    rounds: int = 4000,
    seed: int = 0,
) -> Comparison:
    """Compare numeric scores (e.g. 1-10 ratings, BLEU) of two runs on the same items."""
    baseline, candidate = _validate(
        np.asarray(baseline, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    )
    diffs = candidate - baseline
    if clusters is not None:
        low, high = _cluster_bootstrap(diffs, clusters, confidence, rounds, seed)
        method = IntervalMethod.BOOTSTRAP
    elif interval == IntervalMethod.T:
        low, high = _t_interval(diffs, confidence)
        method = interval
    elif interval == IntervalMethod.BOOTSTRAP:
        low, high = _bootstrap(diffs, confidence, rounds, seed)
        method = interval
    else:
        raise ValueError(f"interval {interval} does not apply to scores")
    return Comparison(
        n=len(diffs),
        baseline_mean=float(baseline.mean()),
        candidate_mean=float(candidate.mean()),
        difference=float(diffs.mean()),
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        p_value=_sign_flip_p(diffs, clusters, rounds, seed),
        test="sign-flip permutation" + (" (clusters)" if clusters is not None else ""),
        interval=method.value,
        clusters=len(np.unique(clusters)) if clusters is not None else None,
    )


def mcnemar_exact(only_baseline: int, only_candidate: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test on the discordant items."""
    discordant = only_baseline + only_candidate
    if discordant == 0:
        return 1.0
    smaller = min(only_baseline, only_candidate)
    return float(min(1.0, 2 * stats.binom.cdf(smaller, discordant, 0.5)))


def _validate(baseline: npt.NDArray[np.generic], candidate: npt.NDArray[np.generic]):  # type: ignore[no-untyped-def]
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("baseline and candidate must be 1-D arrays of the same length")
    if len(baseline) == 0:
        raise ValueError("no items to compare")
    return baseline, candidate


def _paired_wald(b: int, c: int, n: int, confidence: float, add: float) -> tuple[float, float]:
    # With ``add`` = 0.5 every one of the four cells gets +0.5 (n grows by 2).
    b_, c_, n_ = b + add, c + add, n + 4 * add
    difference = (c_ - b_) / n_
    variance = ((b_ + c_) - (c_ - b_) ** 2 / n_) / n_**2
    half = _z(confidence) * math.sqrt(max(variance, 0.0))
    return difference - half, difference + half


def _bootstrap_counts(
    b: int, c: int, n: int, confidence: float, rounds: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap for pass/fail pairs, resampling the cell counts.

    Resampling n items with replacement is the same as drawing the counts of
    (only baseline, only candidate, unchanged) from a multinomial with the
    observed proportions, so this is exact and needs no per-item arrays.
    """
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, [b / n, c / n, (n - b - c) / n], size=rounds)
    means = (draws[:, 1] - draws[:, 0]) / n
    alpha = 1 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def _t_interval(diffs: Floats, confidence: float) -> tuple[float, float]:
    n = len(diffs)
    if n < 2:
        return -math.inf, math.inf
    half = stats.t.ppf(0.5 + confidence / 2, n - 1) * diffs.std(ddof=1) / math.sqrt(n)
    mean = float(diffs.mean())
    return mean - half, mean + half


def _bootstrap(diffs: Floats, confidence: float, rounds: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(diffs), size=(rounds, len(diffs)))
    means = diffs[samples].mean(axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def _cluster_sums(diffs: Floats, clusters: npt.NDArray[np.int64]) -> tuple[Floats, Floats]:
    _, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=diffs)
    sizes = np.bincount(inverse).astype(np.float64)
    return sums, sizes


def _cluster_bootstrap(
    diffs: Floats, clusters: npt.NDArray[np.int64], confidence: float, rounds: int, seed: int
) -> tuple[float, float]:
    """Resample whole clusters; the statistic is the item-weighted mean difference."""
    sums, sizes = _cluster_sums(diffs, np.asarray(clusters))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(rounds, len(sums)))
    means = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def _sign_flip_p(
    diffs: Floats, clusters: npt.NDArray[np.int64] | None, rounds: int, seed: int
) -> float:
    """Two-sided permutation p-value: under no change, each difference's sign is a coin flip.

    With clusters, whole clusters are flipped together.
    """
    if clusters is not None:
        units, sizes = _cluster_sums(diffs, np.asarray(clusters))
    else:
        units, sizes = diffs, np.ones_like(diffs)
    total = sizes.sum()
    observed = abs(units.sum() / total)
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(rounds, len(units)))
    flipped = np.abs((signs * units).sum(axis=1) / total)
    # Count the observed arrangement itself, so p is never exactly 0.
    return float((1 + (flipped >= observed - 1e-12).sum()) / (rounds + 1))


def _z(confidence: float) -> float:
    return float(stats.norm.ppf(0.5 + confidence / 2))
