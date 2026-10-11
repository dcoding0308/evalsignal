"""Build the static demo page from the synthetic examples.

    uv run python site/build.py

Runs EvalSignal on the files in examples/ and writes:

    site/index.html   the demo page, with every result precomputed
    site/report.html  the real HTML report for the support-bot example

The page has no backend and makes no calculations of its own: every number
on it comes from this script, and each one is shown with the command that
reproduces it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from evalsignal.power import is_approximate, minimum_detectable_effect, required_items
from evalsignal.report import Report, build_report, to_html, verdict_label

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
REPO = "https://github.com/dcoding0308/evalsignal"


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    story: str
    before: str
    after: str
    margin: float | None = None


SCENARIOS = [
    Scenario(
        key="upgrade",
        title="Model upgrade",
        story="A new model version, tested on 600 tasks.",
        before="examples/upgrade/before.csv",
        after="examples/upgrade/after.csv",
    ),
    Scenario(
        key="small_eval",
        title="Small test set",
        story="A prompt change, tested on only 40 cases. The score jumps 10 points.",
        before="examples/small_eval/before.csv",
        after="examples/small_eval/after.csv",
    ),
    Scenario(
        key="ratings",
        title="Cheaper model",
        story="Switching to a cheaper model, rated 1-5 on 500 chats. "
        "Changes under 0.25 do not matter to the team.",
        before="examples/ratings/before.jsonl",
        after="examples/ratings/after.jsonl",
        margin=0.25,
    ),
    Scenario(
        key="support_bot",
        title="Support assistant",
        story="A new prompt for a support assistant, tested on 400 cases in four topics.",
        before="examples/support_bot/before.csv",
        after="examples/support_bot/after.csv",
    ),
]

PLAN_DISAGREE = [5, 10, 15, 20, 25, 30]  # percent of items whose result differs
PLAN_CHANGE = [1, 2, 3, 4, 5, 6, 8, 10]  # points
PLAN_ITEMS = [50, 100, 200, 500, 1000, 2000, 5000]


def scenario(spec: Scenario) -> dict[str, object]:
    report = build_report(Path(spec.before), Path(spec.after), margin=spec.margin)
    command = f"evalsignal compare {spec.before} {spec.after}"
    if spec.margin is not None:
        command += f" --margin {spec.margin:g}"
    return {
        "key": spec.key,
        "title": spec.title,
        "story": spec.story,
        "command": command,
        **summary(report),
    }


def summary(report: Report) -> dict[str, object]:
    c = report.comparison
    regressed = {s.name for s in report.slice_regressions}
    return {
        "outcome": report.verdict.outcome.value,
        "label": verdict_label(report),
        "regression": report.regression,
        "headline": report.verdict.headline,
        "details": list(report.verdict.details),
        "binary": report.binary,
        "margin": report.margin,
        "n": c.n,
        "baseline": c.baseline_mean,
        "candidate": c.candidate_mean,
        "difference": c.difference,
        "ci": [c.ci_low, c.ci_high],
        "p": c.p_value,
        "flipped_down": c.only_baseline,
        "flipped_up": c.only_candidate,
        "slices": [
            {
                "name": s.name,
                "n": s.comparison.n,
                "baseline": s.comparison.baseline_mean,
                "candidate": s.comparison.candidate_mean,
                "difference": s.comparison.difference,
                "ci": [s.comparison.ci_low, s.comparison.ci_high],
                "p_holm": s.p_holm,
                "regression": s.name in regressed,
            }
            for s in report.slices
        ],
    }


def plan_tables() -> dict[str, object]:
    needed: dict[str, int | None] = {}
    for d in PLAN_DISAGREE:
        for change in PLAN_CHANGE:
            n = required_items(change / 100, d / 100) if change < d else None
            needed[f"{d}:{change}"] = n
            assert n is None or not is_approximate(n)  # the page shows exact results only
    detectable: dict[str, float | None] = {}
    for d in PLAN_DISAGREE:
        for items in PLAN_ITEMS:
            mde = minimum_detectable_effect(items, d / 100)
            detectable[f"{d}:{items}"] = None if mde != mde else mde  # NaN: nothing detectable
    return {
        "disagree": PLAN_DISAGREE,
        "change": PLAN_CHANGE,
        "items": PLAN_ITEMS,
        "needed": needed,
        "detectable": detectable,
    }


def main() -> None:
    os.chdir(ROOT)  # file names in the output stay relative, e.g. examples/...
    data = {
        "version": version("evalsignal"),
        "built": datetime.now(UTC).strftime("%Y-%m-%d"),
        "repo": REPO,
        "scenarios": [scenario(s) for s in SCENARIOS],
        "plan": plan_tables(),
    }
    support = SCENARIOS[3]
    report = build_report(Path(support.before), Path(support.after))
    (SITE / "report.html").write_text(to_html(report), encoding="utf-8")

    # "</" is escaped so no value can close the script element early.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    template = (SITE / "template.html").read_text(encoding="utf-8")
    (SITE / "index.html").write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print(f"wrote {SITE / 'index.html'} and {SITE / 'report.html'}")


if __name__ == "__main__":
    main()
