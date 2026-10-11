"""Command line: ``evalsignal compare | plan | judge``.

Exit codes: 0 done, 1 regression found with ``--fail-on-regression``, 2 bad input.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from evalsignal.judge import Agreement, evaluate_judge, verdict_from_slots
from evalsignal.power import (
    EXACT_LIMIT,
    is_approximate,
    minimum_detectable_effect,
    required_items,
)
from evalsignal.report import build_report, to_html, to_json, to_markdown, to_text

APPROXIMATE_NOTE = (
    f"(Above {EXACT_LIMIT:,} items this uses the normal approximation; "
    "it is not checked with exact power.)"
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.run(args))
    except (ValueError, OSError) as exc:
        print(f"evalsignal: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalsignal",
        description="Is an AI change a real improvement or just noise?",
    )
    commands = parser.add_subparsers(required=True, metavar="command")

    compare = commands.add_parser(
        "compare",
        help="compare two evaluation runs item by item",
        description="Compare a baseline run with a candidate run on the same items.",
    )
    compare.add_argument("baseline", type=Path, help="results before the change (CSV or JSONL)")
    compare.add_argument("candidate", type=Path, help="results after the change")
    compare.add_argument(
        "--margin",
        type=float,
        help="smallest change that matters: points for pass/fail (2 = 2 points), "
        "score units otherwise",
    )
    compare.add_argument("--alpha", type=float, default=0.05, help="significance level (0.05)")
    compare.add_argument(
        "--format", choices=["text", "markdown", "json"], default="text", help="printed output"
    )
    compare.add_argument("--html", type=Path, help="also write an HTML report to this file")
    compare.add_argument(
        "--fail-on-regression", action="store_true", help="exit with code 1 on a clear regression"
    )
    compare.set_defaults(run=_compare)

    plan = commands.add_parser(
        "plan",
        help="how many items an evaluation needs (pass/fail outcomes)",
        description="Give --change to get the items needed, or --items to get the smallest "
        "change those items can detect.",
    )
    plan.add_argument(
        "--disagree",
        type=float,
        required=True,
        help="percent of items where the two versions give different results (e.g. 15)",
    )
    target = plan.add_mutually_exclusive_group(required=True)
    target.add_argument("--change", type=float, help="change to detect, in points (e.g. 3)")
    target.add_argument("--items", type=int, help="number of items available")
    plan.add_argument("--alpha", type=float, default=0.05)
    plan.add_argument("--power", type=float, default=0.8, help="chance of detecting it (0.8)")
    plan.set_defaults(run=_plan)

    judge = commands.add_parser(
        "judge",
        help="how often an automatic judge agrees with people",
        description="CSV columns: item, model_a, model_b, winner (model_a/model_b/tie or a "
        "model name), judge. Every judge other than --judge is treated as a person.",
    )
    judge.add_argument("verdicts", type=Path)
    judge.add_argument("--judge", required=True, help="name of the automatic judge in the file")
    judge.set_defaults(run=_judge)
    return parser


def _compare(args: argparse.Namespace) -> int:
    if not 0 < args.alpha < 0.5:
        raise ValueError("--alpha must be between 0 and 0.5")
    report = build_report(args.baseline, args.candidate, alpha=args.alpha, margin=args.margin)
    render = {"text": to_text, "markdown": to_markdown, "json": to_json}[args.format]
    sys.stdout.write(render(report))
    if args.html:
        args.html.write_text(to_html(report), encoding="utf-8")
        print(f"\nHTML report: {args.html}", file=sys.stderr)
    return 1 if args.fail_on_regression and report.regression else 0


def _plan(args: argparse.Namespace) -> int:
    disagree = args.disagree / 100
    if args.change is not None:
        change = args.change / 100
        if not 0 < change < disagree:
            raise ValueError("--change must be above 0 and below --disagree")
        n = required_items(change, disagree, args.alpha, args.power)
        print(
            f"To detect a change of {args.change:g} points with {args.power:.0%} probability, "
            f"when about {args.disagree:g}% of items differ between versions: {n:,} items."
        )
        if is_approximate(n):
            print(APPROXIMATE_NOTE)
    else:
        mde = minimum_detectable_effect(args.items, disagree, args.alpha, args.power)
        if math.isnan(mde):
            print(f"{args.items:,} items are too few to reliably detect any change.")
        else:
            print(
                f"With {args.items:,} items, when about {args.disagree:g}% of items differ "
                f"between versions, changes smaller than about {mde * 100:.1f} points cannot "
                f"be reliably detected."
            )
            if is_approximate(args.items):
                print(APPROXIMATE_NOTE)
    return 0


def _judge(args: argparse.Namespace) -> int:
    with args.verdicts.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    needed = {"item", "model_a", "model_b", "winner", "judge"}
    if not rows or not needed <= rows[0].keys():
        raise ValueError(f"{args.verdicts}: needs columns {', '.join(sorted(needed))}")
    verdicts = [
        verdict_from_slots(r["item"], r["model_a"], r["model_b"], r["winner"], r["judge"])
        for r in rows
    ]
    report = evaluate_judge(verdicts, args.judge, lambda name: name != args.judge)
    if not report.judge_verdicts:
        raise ValueError(f"no verdicts from judge {args.judge!r}")
    print(f"Judge {args.judge!r}: {report.judge_verdicts:,} verdicts")
    _print_agreement(f"{args.judge} vs people", report.judge_vs_humans)
    _print_agreement("people vs people", report.human_vs_human)
    print(f"  marked inconsistent when answers were swapped: {report.inconsistent_share:.1%}")
    a, h = report.judge_vs_humans.with_ties, report.human_vs_human.with_ties
    if report.human_vs_human.pairs and report.judge_vs_humans.pairs:
        low, high = report.human_vs_human.with_ties_ci
        if a >= low:
            print("The judge agrees with people about as often as people agree with each other.")
        else:
            print(
                f"The judge agrees with people less often ({a:.1%}) than people agree with "
                f"each other ({h:.1%}, range {low:.1%} to {high:.1%})."
            )
    return 0


def _print_agreement(label: str, agreement: Agreement) -> None:
    if not agreement.pairs:
        print(f"  {label}: no shared items")
        return
    low, high = agreement.with_ties_ci
    line = (
        f"  {label}: {agreement.with_ties:.1%} agree (range {low:.1%} to {high:.1%}, "
        f"{agreement.pairs:,} comparisons)"
    )
    if agreement.without_ties is not None:
        line += f"; {agreement.without_ties:.1%} when neither says tie"
    if agreement.kappa is not None:
        line += f"; kappa {agreement.kappa:.2f}"
    print(line)


if __name__ == "__main__":
    sys.exit(main())
