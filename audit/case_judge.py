"""Case study: how often does GPT-4 as a judge agree with expert humans (MT-Bench)?

    uv run python audit/case_judge.py

Each judged item is one MT-Bench question and turn, comparing two models.
GPT-4's verdicts are compared with every human verdict on the same item, by
model identity; the reference point is how often two humans agree on the
same item. Writes docs/results/judge_mt_bench.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq

from evalsignal.judge import evaluate_judge, verdict_from_slots

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mt_bench"
OUT = ROOT / "docs" / "results" / "judge_mt_bench.json"
COLUMNS = ["question_id", "model_a", "model_b", "winner", "judge", "turn"]


def main() -> None:
    rows = []
    for name in ("human-*.parquet", "gpt4_pair-*.parquet"):
        rows += pq.read_table(next(RAW.glob(name)), columns=COLUMNS).to_pylist()
    verdicts = [
        verdict_from_slots(
            f"{r['question_id']}/turn{r['turn']}",
            str(r["model_a"]),
            str(r["model_b"]),
            str(r["winner"]),
            "gpt4" if r["judge"] == "gpt4_pair" else f"human:{r['judge']}",
        )
        for r in rows
    ]
    report = evaluate_judge(verdicts, "gpt4", lambda j: str(j).startswith("human:"))
    result = {
        "dataset": "lmsys/mt_bench_human_judgments (CC-BY-4.0)",
        "items": "MT-Bench question and turn, two models compared",
        "gpt4_verdicts": report.judge_verdicts,
        "gpt4_inconsistent_share": round(report.inconsistent_share, 4),
        "gpt4_vs_humans": asdict(report.judge_vs_humans),
        "human_vs_human": asdict(report.human_vs_human),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
