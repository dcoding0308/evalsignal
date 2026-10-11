"""Case study: Mistral-7B base vs Mistral-7B-Instruct, paired per question.

    uv run python audit/case_mistral.py

1. Overall comparison on MMLU (14,015 questions) and ARC-Challenge (1,170).
2. Per-subject comparison on MMLU's 57 subjects: how many look "significant"
   before and after correcting for testing 57 subjects at once.
3. Small evaluation sets: draw many random subsets of n questions, as a team
   with a small eval set would, and compare two decision rules against the
   full-set answer:
   * naive: "whichever model scored higher on the subset is better";
   * EvalSignal: improvement / regression only when the paired test says so,
     otherwise "not enough evidence".

Only derived statistics are written (docs/results/case_mistral.json); no
questions or model outputs are stored.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from evalsignal.paired import compare_binary
from evalsignal.slices import compare_slices
from evalsignal.verdict import decide

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "leaderboard"
OUT = ROOT / "docs" / "results" / "case_mistral.json"
SUBSET_SIZES = [50, 100, 200, 500, 1000]
DRAWS = 2000
SEED = 7
ALPHA = 0.05


def load(task_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(subject, base correct, instruct correct), paired by question hash."""
    subjects, base_all, instruct_all = [], [], []
    for path in sorted((RAW / "mistral-7b-base").glob(f"{task_prefix}*.parquet")):
        base = _by_hash(path)
        instruct = _by_hash(RAW / "mistral-7b-instruct" / path.name)
        for key in sorted(base.keys() & instruct.keys()):
            subjects.append(path.stem.removeprefix("hendrycksTest-"))
            base_all.append(base[key])
            instruct_all.append(instruct[key])
    return np.array(subjects), np.array(base_all), np.array(instruct_all)


def _by_hash(path: Path) -> dict[str, bool]:
    """Correct or not per question, using raw ``acc`` (correct when it is 1.0).

    The files also hold ``acc_norm`` (length-normalised), which the
    leaderboard used for some headline scores, so those can differ from ours.
    Runs are pinned in audit/download_data.py.
    """
    result = {}
    for row in pq.read_table(path, columns=["acc", "hashes"]).to_pylist():
        hashes = (
            row["hashes"] if isinstance(row["hashes"], dict) else ast.literal_eval(row["hashes"])
        )
        result[hashes["example"]] = float(row["acc"]) == 1.0
    return result


def summarise(name: str, base: np.ndarray, instruct: np.ndarray) -> dict[str, object]:
    comparison = compare_binary(base, instruct)
    verdict = decide(comparison, binary=True)
    return {
        "benchmark": name,
        "questions": comparison.n,
        "base_accuracy": round(comparison.baseline_mean, 4),
        "instruct_accuracy": round(comparison.candidate_mean, 4),
        "difference": round(comparison.difference, 4),
        "ci95": [round(comparison.ci_low, 4), round(comparison.ci_high, 4)],
        "p_value": comparison.p_value,
        "only_base_correct": comparison.only_baseline,
        "only_instruct_correct": comparison.only_candidate,
        "verdict": verdict.outcome.value,
        "headline": verdict.headline,
    }


def per_subject(subjects: np.ndarray, base: np.ndarray, instruct: np.ndarray) -> dict[str, object]:
    results = compare_slices(subjects, base, instruct, binary=True)
    raw = [r for r in results if r.comparison.p_value < 0.05]
    holm = [r for r in results if r.p_holm < 0.05]
    bh = [r for r in results if r.p_bh < 0.05]
    return {
        "subjects": len(results),
        "significant_unadjusted": len(raw),
        "significant_holm": len(holm),
        "significant_bh": len(bh),
        "flagged_by_holm": sorted(
            (r.name, round(r.comparison.difference, 4), round(r.p_holm, 5)) for r in holm
        ),
        "only_without_correction": sorted(r.name for r in raw if r.p_bh >= 0.05),
    }


def small_sets(base: np.ndarray, instruct: np.ndarray) -> list[dict[str, object]]:
    truth = float(instruct.mean() - base.mean())  # full-set answer: the sign that matters
    rng = np.random.default_rng(SEED)
    rows = []
    for n in SUBSET_SIZES:
        naive_wrong = signal_wrong = signal_right = signal_unsure = 0
        for _ in range(DRAWS):
            pick = rng.choice(len(base), size=n, replace=False)
            b, i = base[pick], instruct[pick]
            observed = float(i.mean() - b.mean())
            naive_wrong += observed != 0 and np.sign(observed) != np.sign(truth)
            # Same rule as decide() without a margin; skips its sample-size advice,
            # which is not needed here and is the slow part.
            comparison = compare_binary(b, i)
            if comparison.p_value >= ALPHA:
                signal_unsure += 1
            elif (comparison.difference > 0) == (truth > 0):
                signal_right += 1
            else:
                signal_wrong += 1
        rows.append(
            {
                "subset_size": n,
                "naive_points_the_wrong_way": round(float(naive_wrong) / DRAWS, 4),
                "evalsignal_correct_call": round(signal_right / DRAWS, 4),
                "evalsignal_not_enough_evidence": round(signal_unsure / DRAWS, 4),
                "evalsignal_wrong_call": round(signal_wrong / DRAWS, 4),
            }
        )
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    subjects, base, instruct = load("hendrycksTest-")
    _, arc_base, arc_instruct = load("arc")
    report = {
        "models": ["Mistral-7B-v0.1 (base)", "Mistral-7B-Instruct-v0.1"],
        "overall": [
            summarise("MMLU", base, instruct),
            summarise("ARC-Challenge", arc_base, arc_instruct),
        ],
        "mmlu_subjects": per_subject(subjects, base, instruct),
        "small_eval_sets_mmlu": small_sets(base, instruct),
        "draws_per_size": DRAWS,
    }
    for row in report["overall"]:
        print(row)
    print({k: v for k, v in report["mmlu_subjects"].items() if k != "flagged_by_holm"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
