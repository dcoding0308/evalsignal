"""Week 1 audit: can the downloaded data support the planned analyses?

    uv run python audit/data_audit.py

Checks that per-question results of the two Mistral models can be paired
question by question (by the harness's hash of the question), and how the
human and GPT-4 judgments in MT-Bench overlap. Writes docs/audit/data_audit.json.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "docs" / "audit" / "data_audit.json"


def leaderboard() -> dict[str, object]:
    base_dir, instruct_dir = (
        RAW / "leaderboard/mistral-7b-base",
        RAW / "leaderboard/mistral-7b-instruct",
    )
    tasks = sorted(p.stem for p in base_dir.glob("*.parquet"))
    per_task = {}
    totals = Counter()
    for task in tasks:
        base = _by_hash(base_dir / f"{task}.parquet")
        instruct = _by_hash(instruct_dir / f"{task}.parquet")
        shared = base.keys() & instruct.keys()
        both = sum(base[h] and instruct[h] for h in shared)
        only_base = sum(base[h] and not instruct[h] for h in shared)
        only_instruct = sum(instruct[h] and not base[h] for h in shared)
        per_task[task] = {
            "questions": len(shared),
            "unmatched": len(base.keys() ^ instruct.keys()),
            "base_accuracy": round(sum(base[h] for h in shared) / len(shared), 4),
            "instruct_accuracy": round(sum(instruct[h] for h in shared) / len(shared), 4),
            "both_correct": both,
            "only_base_correct": only_base,
            "only_instruct_correct": only_instruct,
        }
        group = "mmlu" if task.startswith("hendrycksTest-") else task
        totals[f"{group}_questions"] += len(shared)
        totals[f"{group}_unmatched"] += per_task[task]["unmatched"]
    sizes = sorted(v["questions"] for k, v in per_task.items() if k.startswith("hendrycksTest-"))
    return {
        "tasks": len(tasks),
        "totals": dict(totals),
        "mmlu_subject_sizes": {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]},
        "per_task": per_task,
    }


def _by_hash(path: Path) -> dict[str, bool]:
    table = pq.read_table(path, columns=["acc", "hashes"]).to_pylist()
    result = {}
    for row in table:
        hashes = row["hashes"]
        if isinstance(hashes, str):
            hashes = ast.literal_eval(hashes)
        result[hashes["example"]] = float(row["acc"]) == 1.0
    return result


def mt_bench() -> dict[str, object]:
    folder = RAW / "mt_bench"
    columns = ["question_id", "model_a", "model_b", "winner", "judge", "turn"]
    human = pq.read_table(next(folder.glob("human-*.parquet")), columns=columns).to_pylist()
    gpt4 = pq.read_table(next(folder.glob("gpt4_pair-*.parquet")), columns=columns).to_pylist()

    def key(row: dict[str, object]) -> tuple[object, ...]:
        return (row["question_id"], row["model_a"], row["model_b"], row["turn"])

    gpt4_keys = {key(r) for r in gpt4}
    human_keys = Counter(key(r) for r in human)
    return {
        "human_judgments": len(human),
        "human_judges": len({r["judge"] for r in human}),
        "gpt4_judgments": len(gpt4),
        "models_compared": len(
            {str(r["model_a"]) for r in human} | {str(r["model_b"]) for r in human}
        ),
        "human_winner_share": dict(Counter(str(r["winner"]) for r in human)),
        "gpt4_winner_share": dict(Counter(str(r["winner"]) for r in gpt4)),
        "pairs_judged_by_humans": len(human_keys),
        "pairs_judged_by_both": len(human_keys.keys() & gpt4_keys),
        "pairs_with_two_or_more_humans": sum(1 for n in human_keys.values() if n >= 2),
    }


def main() -> None:
    report = {"leaderboard": leaderboard(), "mt_bench": mt_bench()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: v for k, v in report["leaderboard"].items() if k != "per_task"}
    print(json.dumps({"leaderboard": summary, "mt_bench": report["mt_bench"]}, indent=2))


if __name__ == "__main__":
    main()
