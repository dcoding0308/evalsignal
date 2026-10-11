"""Turn a comparison into a decision and a plain-English explanation.

Four outcomes, checked in this order:

* ``no_meaningful_change``: a ``margin`` was given and the whole confidence
  interval lies inside ``±margin``, the size of change you have said does
  not matter. This wins even when the change is statistically detectable
  (with enough items, tiny changes are), and the explanation says so.
* ``regression`` / ``improvement``: otherwise, the change is statistically
  clear (p < alpha).
* ``not_enough_evidence``: otherwise the data cannot tell. The verdict then
  says how large a change the evaluation *could* detect, and how many items
  it would need to detect a change of the observed size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from scipy import stats

from evalsignal.paired import Comparison
from evalsignal.power import EXACT_LIMIT, minimum_detectable_effect, required_items


class Outcome(StrEnum):
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    NO_MEANINGFUL_CHANGE = "no_meaningful_change"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"


@dataclass(frozen=True, slots=True)
class Verdict:
    outcome: Outcome
    headline: str
    details: tuple[str, ...]
    # Set only for "not enough evidence": the smallest reliably detectable change
    # (as a fraction or in score units) and the items needed for the observed change.
    minimum_detectable: float | None
    items_needed: int | None


def decide(
    comparison: Comparison,
    binary: bool,
    alpha: float = 0.05,
    margin: float | None = None,
    power: float = 0.8,
    score_sd: float | None = None,
) -> Verdict:
    unit = _Unit(binary)
    c = comparison
    level = f"{c.confidence:.0%}"
    summary = (
        f"{unit.name} went from {unit.value(c.baseline_mean)} to {unit.value(c.candidate_mean)} "
        f"({unit.delta(c.difference)}; {level} range {unit.delta(c.ci_low)} to "
        f"{unit.delta(c.ci_high)}) on {c.n:,} items."
    )
    mde = needed = None  # only worked out when the answer is "not enough evidence"

    if margin is not None and within_margin(c, margin):
        outcome = Outcome.NO_MEANINGFUL_CHANGE
        headline = (
            f"No meaningful change: the whole {level} range lies within "
            f"±{unit.delta(margin, signed=False)}."
        )
        details = [summary]
        if c.p_value < alpha:
            details.append(
                f"The change is statistically detectable (p {_p(c.p_value)}) but smaller "
                f"than your margin of {unit.delta(margin, signed=False)}."
            )
    elif c.p_value < alpha:
        outcome = Outcome.REGRESSION if c.difference < 0 else Outcome.IMPROVEMENT
        word = "regression" if outcome is Outcome.REGRESSION else "improvement"
        headline = f"Real {word}: {unit.delta(c.difference)} (p {_p(c.p_value)})."
        details = [summary, f"A change this large is unlikely to be chance ({c.test})."]
        if margin is not None and abs(c.difference) < margin:
            details.append(
                f"The measured change is smaller than your margin of "
                f"{unit.delta(margin, signed=False)}, but the {level} range reaches beyond it."
            )
    else:
        outcome = Outcome.NOT_ENOUGH_EVIDENCE
        mde, needed = _sensitivity(c, binary, alpha, power, score_sd)
        headline = f"Not enough evidence of a real change (p {_p(c.p_value)})."
        details = [summary]
        if mde is not None:
            details.append(
                f"With {c.n:,} items, changes smaller than about "
                f"{unit.delta(mde, signed=False)} cannot be reliably detected."
            )
        elif binary and c.discordance:
            details.append(
                f"{c.n:,} items are too few: with only {c.discordance:.0%} of items changing "
                f"result, even the largest possible change could not be reliably detected."
            )
        if needed is not None:
            details.append(
                f"Detecting a change of {unit.delta(abs(c.difference), signed=False)} "
                f"would need about {needed:,} items"
                + (" (normal approximation)." if binary and needed > EXACT_LIMIT else ".")
            )
    return Verdict(outcome, headline, tuple(details), mde, needed)


def within_margin(comparison: Comparison, margin: float | None) -> bool:
    """True when a margin is given and the whole interval lies inside ``±margin``."""
    return margin is not None and -margin <= comparison.ci_low and comparison.ci_high <= margin


def _sensitivity(
    c: Comparison, binary: bool, alpha: float, power: float, score_sd: float | None
) -> tuple[float | None, int | None]:
    """(minimum detectable change at this n, items needed for the observed change)."""
    if binary:
        discordance = c.discordance
        if not discordance:
            return None, None
        mde = minimum_detectable_effect(c.n, discordance, alpha, power)
        observed = abs(c.difference)
        needed = (
            required_items(observed, discordance, alpha, power)
            if 0 < observed < discordance
            else None
        )
        return (None if math.isnan(mde) else mde), needed
    if not score_sd:
        return None, None
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    mde = float(z * score_sd / math.sqrt(c.n))
    observed = abs(c.difference)
    needed = math.ceil(float((z * score_sd / observed) ** 2)) if observed > 0 else None
    return mde, needed


class _Unit:
    """Formats values as percentages and points (pass/fail) or plain numbers (scores)."""

    def __init__(self, binary: bool) -> None:
        self.binary = binary
        self.name = "Accuracy" if binary else "Mean score"

    def value(self, x: float) -> str:
        return f"{x * 100:.1f}%" if self.binary else f"{x:.3g}"

    def delta(self, x: float, signed: bool = True) -> str:
        if self.binary:
            number = f"{x * 100:+.1f}" if signed else f"{abs(x) * 100:.1f}"
            return f"{number} points"
        return f"{x:+.3g}" if signed else f"{abs(x):.3g}"


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"
