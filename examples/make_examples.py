"""Generate the synthetic example files in this folder (all data is made up).

    uv run python examples/make_examples.py

support_bot/  a prompt change for a support assistant, 400 pass/fail test cases
              in four topics; the new prompt quietly breaks refund questions.
small_eval/   the same kind of change measured on only 40 cases: the score
              goes up, but 40 cases cannot tell.
ratings/      1-5 quality ratings before and after switching to a cheaper
              model; checks whether quality stayed within half a point.
upgrade/      a model upgrade measured on 600 cases: a clear improvement.
judge.csv     an automatic judge and three people comparing answers from two
              models on the same 120 questions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(2026)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def support_bot() -> None:
    topics = {"billing": 0.0, "shipping": 0.02, "account": 0.01, "refunds": -0.20}
    before, after = [], []
    for topic, shift in topics.items():
        for k in range(100):
            item = f"{topic}-{k:03d}"
            p_before = 0.80
            passed_before = rng.random() < p_before
            # Most cases keep their result; a few flip, more often in the shifted direction.
            if shift >= 0:
                flips_up = (not passed_before) and rng.random() < 0.25 + shift * 5
                flips_down = passed_before and rng.random() < 0.05
            else:
                flips_up = (not passed_before) and rng.random() < 0.10
                flips_down = passed_before and rng.random() < -shift * 1.4
            passed_after = (passed_before or flips_up) and not flips_down
            before.append({"id": item, "passed": passed_before, "slice": topic})
            after.append({"id": item, "passed": passed_after, "slice": topic})
    write_csv(HERE / "support_bot" / "before.csv", before)
    write_csv(HERE / "support_bot" / "after.csv", after)


def small_eval() -> None:
    before = [{"id": f"case-{k:02d}", "passed": rng.random() < 0.70} for k in range(40)]
    after = []
    for row in before:
        passed = bool(row["passed"])
        if passed and rng.random() < 0.08:
            passed = False
        elif not passed and rng.random() < 0.35:
            passed = True
        after.append({"id": row["id"], "passed": passed})
    write_csv(HERE / "small_eval" / "before.csv", before)
    write_csv(HERE / "small_eval" / "after.csv", after)


def ratings() -> None:
    path = HERE / "ratings"
    path.mkdir(parents=True, exist_ok=True)
    quality = rng.normal(3.8, 0.6, size=500)
    with (path / "before.jsonl").open("w") as b, (path / "after.jsonl").open("w") as a:
        for k, q in enumerate(quality):
            score_before = float(np.clip(np.round(q + rng.normal(0, 0.5)), 1, 5))
            score_after = float(np.clip(np.round(q - 0.03 + rng.normal(0, 0.5)), 1, 5))
            b.write(json.dumps({"id": f"chat-{k:03d}", "score": score_before}) + "\n")
            a.write(json.dumps({"id": f"chat-{k:03d}", "score": score_after}) + "\n")


def judge() -> None:
    rows: list[dict[str, object]] = []
    people = ["reviewer-1", "reviewer-2", "reviewer-3"]
    for k in range(120):
        item = f"q{k:03d}"
        truth = rng.choice(["model_a", "model_b", "tie"], p=[0.35, 0.45, 0.20])
        for name in [*people, "auto-judge"]:
            keep = 0.80 if name == "auto-judge" else 0.75
            if name != "auto-judge" and rng.random() < 0.3:
                continue  # not every person rated every question
            label = truth if rng.random() < keep else rng.choice(["model_a", "model_b", "tie"])
            if name == "auto-judge" and rng.random() < 0.12:
                label = "tie (inconsistent)"
            rows.append(
                {
                    "item": item,
                    "model_a": "assistant-small",
                    "model_b": "assistant-large",
                    "winner": label,
                    "judge": name,
                }
            )
    write_csv(HERE / "judge.csv", rows)


def upgrade() -> None:
    before, after = [], []
    for k in range(600):
        passed = rng.random() < 0.70
        # 4% of passes break; 28% of failures get fixed.
        passed_after = rng.random() >= 0.04 if passed else rng.random() < 0.28
        before.append({"id": f"task-{k:03d}", "passed": passed})
        after.append({"id": f"task-{k:03d}", "passed": passed_after})
    write_csv(HERE / "upgrade" / "before.csv", before)
    write_csv(HERE / "upgrade" / "after.csv", after)


if __name__ == "__main__":
    support_bot()
    small_eval()
    ratings()
    judge()
    upgrade()  # last, so the other files stay identical
