"""Check EvalSignal's statistics against simulations where the truth is known.

    uv run python audit/simulate.py

Pass/fail pairs are drawn from four cells: unchanged, passed only before
(probability (d - delta)/2), passed only after ((d + delta)/2), where d is
the discordance and delta the true accuracy change. Then:

1. False-alarm rate: with delta = 0, how often does McNemar's test say p < 0.05?
2. Coverage: how often does each 95% interval contain the true delta?
3. Power: at the sample size ``required_items`` recommends, how often is a
   real change detected? It should be close to the promised 80%.
4. Clusters: items in related groups whose flips lean the same way. With no
   overall change, how often do the item-level and the cluster-aware tests
   raise a false alarm?

Writes docs/results/simulation.json. A fixed seed makes it reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evalsignal.paired import IntervalMethod, compare_binary
from evalsignal.power import approximate_items, required_items

OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "simulation.json"
SIMULATIONS = 2000
ALPHA = 0.05
SIZES = [50, 100, 200, 500, 1000]
DISCORDANCES = [0.05, 0.15, 0.30]
METHODS = [IntervalMethod.WALD, IntervalMethod.AGRESTI_MIN, IntervalMethod.BOOTSTRAP]
BASE_ACCURACY = 0.7


def draw(rng: np.random.Generator, n: int, discordance: float, delta: float):  # type: ignore[no-untyped-def]
    """Baseline and candidate pass/fail arrays with a known true change ``delta``."""
    b, c = (discordance - delta) / 2, (discordance + delta) / 2
    both = BASE_ACCURACY - b  # baseline accuracy = both + b
    cells = rng.multinomial(n, [both, b, c, 1 - both - b - c])
    baseline = np.repeat([True, True, False, False], cells)
    candidate = np.repeat([True, False, True, False], cells)
    return baseline, candidate


def error_and_coverage(rng: np.random.Generator) -> list[dict[str, object]]:
    rows = []
    for discordance in DISCORDANCES:
        for n in SIZES:
            for delta in (0.0, round(discordance / 3, 4)):
                rejections = 0
                covered = dict.fromkeys(METHODS, 0)
                for i in range(SIMULATIONS):
                    baseline, candidate = draw(rng, n, discordance, delta)
                    for method in METHODS:
                        r = compare_binary(
                            baseline, candidate, interval=method, bootstrap_rounds=1000, seed=i
                        )
                        covered[method] += r.ci_low <= delta <= r.ci_high
                        if method == METHODS[0]:
                            rejections += r.p_value < ALPHA
                rows.append(
                    {
                        "discordance": discordance,
                        "n": n,
                        "true_delta": delta,
                        "rejection_rate": round(rejections / SIMULATIONS, 4),
                        **{
                            f"coverage_{m.value}": round(covered[m] / SIMULATIONS, 4)
                            for m in METHODS
                        },
                    }
                )
                print(rows[-1], flush=True)
    return rows


def power_check(rng: np.random.Generator) -> list[dict[str, object]]:
    rows = []
    for discordance, delta in [(0.10, 0.03), (0.15, 0.05), (0.20, 0.04), (0.30, 0.06)]:
        row: dict[str, object] = {"discordance": discordance, "true_delta": delta}
        for label, n in [
            ("normal_approximation", approximate_items(delta, discordance)),
            ("exact", required_items(delta, discordance)),
        ]:
            detected = 0
            for i in range(SIMULATIONS):
                baseline, candidate = draw(rng, n, discordance, delta)
                detected += compare_binary(baseline, candidate, seed=i).p_value < ALPHA
            row[f"{label}_n"] = n
            row[f"{label}_simulated_power"] = round(detected / SIMULATIONS, 4)
        row["promised_power"] = 0.8
        rows.append(row)
        print(rows[-1], flush=True)
    return rows


def cluster_check(rng: np.random.Generator) -> dict[str, object]:
    """40 clusters of 10 related items; each cluster's flips lean one way at random."""
    clusters, size, discordance, lean = 40, 10, 0.2, 0.15
    item_level = cluster_level = 0
    labels = np.repeat(np.arange(clusters), size)
    for i in range(SIMULATIONS // 4):  # cluster tests are slower; 500 runs suffice here
        baseline_parts, candidate_parts = [], []
        for _ in range(clusters):
            delta = rng.choice([-lean, lean])  # leans cancel out on average: no true change
            b, c = draw(rng, size, discordance, delta)
            baseline_parts.append(b)
            candidate_parts.append(c)
        baseline, candidate = np.concatenate(baseline_parts), np.concatenate(candidate_parts)
        item_level += compare_binary(baseline, candidate).p_value < ALPHA
        cluster_level += (
            compare_binary(
                baseline, candidate, clusters=labels, bootstrap_rounds=1000, seed=i
            ).p_value
            < ALPHA
        )
    runs = SIMULATIONS // 4
    result = {
        "clusters": clusters,
        "items_per_cluster": size,
        "true_delta": 0.0,
        "false_alarm_item_level_mcnemar": round(item_level / runs, 4),
        "false_alarm_cluster_permutation": round(cluster_level / runs, 4),
        "runs": runs,
    }
    print(result, flush=True)
    return result


def main() -> None:
    rng = np.random.default_rng(20261010)
    report = {
        "simulations_per_cell": SIMULATIONS,
        "alpha": ALPHA,
        "error_and_coverage": error_and_coverage(rng),
        "power": power_check(rng),
        "clusters": cluster_check(rng),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
