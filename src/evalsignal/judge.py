"""How far can an automatic judge be trusted? Agreement with humans, against human-human agreement.

An LLM judge compares two answers and says which is better (or a tie). Its
verdicts are useful only if they agree with people about as often as people
agree with each other. This module measures exactly that, on items where
both the judge and humans gave a verdict.

Rules that avoid common mistakes:

* Verdicts are compared by **model identity**, never by slot ("A"/"B"):
  data files often put a particular model in a fixed slot, so slot-level
  counts can look like position bias when they are just file ordering.
* Position sensitivity is measured where it is observable: verdicts the
  judge itself flagged as inconsistent when the two answers were swapped.
* Agreement is reported with ties (3 classes) and without ties (only items
  both sides decided), plus Cohen's kappa, with bootstrap intervals over items.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

TIE = "tie"


@dataclass(frozen=True, slots=True)
class Verdict:
    item: str  # the question (and turn) being judged
    models: tuple[str, str]  # the two compared models, sorted
    winner: str  # a model name or "tie"
    judge: str
    inconsistent: bool = False  # the judge changed its verdict when answers were swapped


@dataclass(frozen=True, slots=True)
class Agreement:
    pairs: int  # comparisons made
    with_ties: float
    with_ties_ci: tuple[float, float]
    without_ties: float | None
    decisive_pairs: int
    kappa: float | None


@dataclass(frozen=True, slots=True)
class JudgeReport:
    judge: str
    judge_vs_humans: Agreement
    human_vs_human: Agreement
    inconsistent_share: float  # share of the judge's verdicts flagged inconsistent
    judge_verdicts: int


def verdict_from_slots(item: str, model_a: str, model_b: str, winner: str, judge: str) -> Verdict:
    """Build an identity-based verdict from a slot-based record ("model_a"/"model_b"/"tie")."""
    label = winner.strip().lower()
    inconsistent = "inconsistent" in label
    if label.startswith(TIE):
        chosen = TIE
    elif label in ("model_a", "a"):
        chosen = model_a
    elif label in ("model_b", "b"):
        chosen = model_b
    elif winner in (model_a, model_b):
        chosen = winner
    else:
        raise ValueError(f"unknown winner {winner!r} for {model_a} vs {model_b}")
    models = (min(model_a, model_b), max(model_a, model_b))
    return Verdict(item, models, chosen, judge, inconsistent)


def evaluate_judge(
    verdicts: list[Verdict], judge: str, is_human: object, seed: int = 0
) -> JudgeReport:
    """``is_human(judge_name) -> bool`` says which verdicts come from people."""
    human_by_key: dict[tuple[str, tuple[str, str]], list[str]] = defaultdict(list)
    judge_by_key: dict[tuple[str, tuple[str, str]], str] = {}
    judge_total = judge_inconsistent = 0
    for v in verdicts:
        key = (v.item, v.models)
        if v.judge == judge:
            judge_by_key[key] = v.winner
            judge_total += 1
            judge_inconsistent += v.inconsistent
        elif callable(is_human) and is_human(v.judge):
            human_by_key[key].append(v.winner)

    judge_pairs = [
        (judge_by_key[key], human, key)
        for key in judge_by_key.keys() & human_by_key.keys()
        for human in human_by_key[key]
    ]
    human_pairs = [
        (a, b, key)
        for key, labels in human_by_key.items()
        for a, b in itertools.combinations(labels, 2)
    ]
    return JudgeReport(
        judge=judge,
        judge_vs_humans=_agreement(judge_pairs, seed),
        human_vs_human=_agreement(human_pairs, seed),
        inconsistent_share=judge_inconsistent / max(judge_total, 1),
        judge_verdicts=judge_total,
    )


def _agreement(
    pairs: list[tuple[str, str, tuple[str, tuple[str, str]]]], seed: int, rounds: int = 2000
) -> Agreement:
    if not pairs:
        return Agreement(0, float("nan"), (float("nan"), float("nan")), None, 0, None)
    same = np.array([a == b for a, b, _ in pairs], dtype=np.float64)
    decisive = [(a, b) for a, b, _ in pairs if a != TIE and b != TIE]
    # Bootstrap over judged items (not over comparisons), since several
    # comparisons of the same item are not independent.
    keys = sorted({key for _, _, key in pairs})
    index = {key: i for i, key in enumerate(keys)}
    item_of = np.array([index[key] for _, _, key in pairs])
    sums = np.bincount(item_of, weights=same, minlength=len(keys))
    counts = np.bincount(item_of, minlength=len(keys)).astype(np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), size=(rounds, len(keys)))
    boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return Agreement(
        pairs=len(pairs),
        with_ties=float(same.mean()),
        with_ties_ci=(float(low), float(high)),
        without_ties=(sum(a == b for a, b in decisive) / len(decisive)) if decisive else None,
        decisive_pairs=len(decisive),
        kappa=_kappa(pairs),
    )


def _kappa(pairs: list[tuple[str, str, tuple[str, tuple[str, str]]]]) -> float | None:
    """Cohen's kappa over 3 identity-based classes: first model, second model, tie."""

    def label(winner: str, models: tuple[str, str]) -> str:
        return TIE if winner == TIE else ("first" if winner == models[0] else "second")

    a = [label(x, key[1]) for x, _, key in pairs]
    b = [label(y, key[1]) for _, y, key in pairs]
    classes = ("first", "second", TIE)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    expected = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in classes)
    return None if expected == 1 else (observed - expected) / (1 - expected)
