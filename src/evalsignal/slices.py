"""Per-slice comparisons ("math got worse, coding got better") with multiple-testing control.

Testing many slices at once guarantees false alarms: with 57 slices and a 5%
false-alarm rate, about 3 slices look "significant" even if nothing changed.
Every slice's p-value is therefore adjusted:

* Holm (1979) controls the chance of *any* false alarm across all slices.
* Benjamini-Hochberg (1995) controls the expected *share* of flagged slices
  that are false alarms; it flags more, at the cost of a weaker guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from evalsignal.paired import Comparison, compare_binary, compare_scores


@dataclass(frozen=True, slots=True)
class SliceResult:
    name: str
    comparison: Comparison
    p_holm: float
    p_bh: float


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), monotone and capped at 1."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        running = min(running, p_values[index] * m / (rank + 1))
        adjusted[index] = running
    return adjusted.tolist()


def compare_slices(
    slices: npt.NDArray[np.str_],
    baseline: npt.NDArray[np.generic],
    candidate: npt.NDArray[np.generic],
    binary: bool,
    confidence: float = 0.95,
) -> list[SliceResult]:
    """Compare each slice separately, then adjust p-values across slices."""
    names = sorted(set(slices.tolist()))
    comparisons = []
    for name in names:
        mask = slices == name
        if binary:
            comparisons.append(
                compare_binary(
                    baseline[mask].astype(bool), candidate[mask].astype(bool), confidence
                )
            )
        else:
            comparisons.append(compare_scores(baseline[mask], candidate[mask], confidence))
    p = [c.p_value for c in comparisons]
    return [
        SliceResult(name, comparison, p_h, p_b)
        for name, comparison, p_h, p_b in zip(
            names, comparisons, holm(p), benjamini_hochberg(p), strict=True
        )
    ]
