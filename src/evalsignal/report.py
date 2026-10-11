"""Compare two result files and present the answer as text, Markdown, JSON or HTML.

The HTML report is a single self-contained file: no JavaScript, no external
resources, and every value from the input files is escaped.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from evalsignal.paired import Comparison, compare_binary, compare_scores
from evalsignal.results import Paired, pair, read_results
from evalsignal.slices import SliceResult, compare_slices
from evalsignal.verdict import Outcome, Verdict, decide, within_margin

MAX_LISTED_IDS = 20


@dataclass(frozen=True, slots=True)
class Report:
    baseline_file: str
    candidate_file: str
    binary: bool
    comparison: Comparison
    verdict: Verdict
    slices: list[SliceResult]
    only_in_baseline: list[str]
    only_in_candidate: list[str]
    alpha: float
    margin: float | None  # as a fraction for pass/fail, in score units otherwise

    @property
    def slice_regressions(self) -> list[SliceResult]:
        """Slices that clearly got worse: significant after Holm adjustment, and (when a
        margin is set) not entirely within it, the same rule as the overall verdict."""
        return [
            s
            for s in self.slices
            if s.p_holm < self.alpha
            and s.comparison.difference < 0
            and not within_margin(s.comparison, self.margin)
        ]

    @property
    def regression(self) -> bool:
        """A clear regression overall or in any slice."""
        return self.verdict.outcome is Outcome.REGRESSION or bool(self.slice_regressions)


def build_report(
    baseline_path: Path,
    candidate_path: Path,
    alpha: float = 0.05,
    margin: float | None = None,
    seed: int = 0,
) -> Report:
    """``margin`` is in points for pass/fail results (2 = 2 points), score units otherwise."""
    paired = pair(read_results(baseline_path), read_results(candidate_path))
    if margin is not None:
        if margin <= 0:
            raise ValueError("margin must be positive")
        margin = margin / 100 if paired.binary else margin
    confidence = 1 - alpha
    comparison = _compare(paired, confidence, seed)
    score_sd = (
        None
        if paired.binary or paired.baseline.size < 2
        else float(np.std(paired.candidate - paired.baseline, ddof=1))
    )
    verdict = decide(comparison, paired.binary, alpha=alpha, margin=margin, score_sd=score_sd)
    slices: list[SliceResult] = []
    if paired.slices is not None and len(set(paired.slices.tolist())) > 1:
        slices = compare_slices(
            paired.slices, paired.baseline, paired.candidate, paired.binary, confidence
        )
    return Report(
        baseline_file=str(baseline_path),
        candidate_file=str(candidate_path),
        binary=paired.binary,
        comparison=comparison,
        verdict=verdict,
        slices=slices,
        only_in_baseline=paired.only_in_baseline,
        only_in_candidate=paired.only_in_candidate,
        alpha=alpha,
        margin=margin,
    )


def _compare(paired: Paired, confidence: float, seed: int) -> Comparison:
    if paired.binary:
        return compare_binary(
            paired.baseline.astype(bool),
            paired.candidate.astype(bool),
            confidence=confidence,
            clusters=paired.clusters,
            seed=seed,
        )
    return compare_scores(
        paired.baseline,
        paired.candidate,
        confidence=confidence,
        clusters=paired.clusters,
        seed=seed,
    )


# ----------------------------------------------------------------------------- text


def to_text(report: Report) -> str:
    v = report.verdict
    lines = [verdict_label(report), "", v.headline, *v.details]
    lines += _notes(report)
    if report.slices:
        lines += ["", "By slice (p adjusted for the number of slices, Holm):"]
        width = max(len(s.name) for s in report.slices)
        for s in report.slices:
            flag = "  <- clear change" if s.p_holm < report.alpha else ""
            lines.append(
                f"  {s.name:<{width}}  {s.comparison.n:>6,} items  "
                f"{_delta(report, s.comparison.difference):>14}  "
                f"[{_delta(report, s.comparison.ci_low)} to "
                f"{_delta(report, s.comparison.ci_high)}]  "
                f"p {_pv(s.p_holm)}{flag}"
            )
    return "\n".join(lines) + "\n"


def to_markdown(report: Report) -> str:
    """Short summary for a pull-request comment."""
    v = report.verdict
    lines = [f"### EvalSignal: {verdict_label(report)}", "", f"**{v.headline}**", ""]
    lines += [f"- {d}" for d in v.details]
    lines += [f"- {n}" for n in _notes(report) if n]
    if report.slices:
        lines += ["", "| Slice | Items | Change | Range | p (Holm) |", "|---|---:|---:|---|---:|"]
        for s in report.slices:
            c = s.comparison
            name = s.name.replace("|", "\\|")
            bold = "**" if s.p_holm < report.alpha else ""
            lines.append(
                f"| {name} | {c.n:,} | {bold}{_delta(report, c.difference)}{bold} | "
                f"{_delta(report, c.ci_low)} to {_delta(report, c.ci_high)} | {_pv(s.p_holm)} |"
            )
    return "\n".join(lines) + "\n"


def to_json(report: Report) -> str:
    c = report.comparison
    data = {
        "outcome": report.verdict.outcome.value,
        "headline": report.verdict.headline,
        "details": list(report.verdict.details),
        "binary": report.binary,
        "comparison": asdict(c),
        "minimum_detectable_change": report.verdict.minimum_detectable,
        "items_needed_for_observed_change": report.verdict.items_needed,
        "alpha": report.alpha,
        "margin": report.margin,
        "only_in_baseline": report.only_in_baseline,
        "only_in_candidate": report.only_in_candidate,
        "slices": [
            {"name": s.name, "p_holm": s.p_holm, "p_bh": s.p_bh, **asdict(s.comparison)}
            for s in report.slices
        ],
    }
    return json.dumps(data, indent=2, default=_json_default) + "\n"


def _json_default(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialise {type(value).__name__}")


_LABELS = {
    Outcome.REGRESSION: "REGRESSION",
    Outcome.IMPROVEMENT: "IMPROVEMENT",
    Outcome.NO_MEANINGFUL_CHANGE: "NO MEANINGFUL CHANGE",
    Outcome.NOT_ENOUGH_EVIDENCE: "NOT ENOUGH EVIDENCE",
}


def verdict_label(report: Report) -> str:
    label = _LABELS[report.verdict.outcome]
    count = len(report.slice_regressions)
    if count and report.verdict.outcome is not Outcome.REGRESSION:
        label += f" OVERALL, REGRESSION IN {count} SLICE{'S' if count > 1 else ''}"
    return label


def _notes(report: Report) -> list[str]:
    notes = []
    c = report.comparison
    if report.slice_regressions:
        worse = ", ".join(
            f"{s.name} ({_delta(report, s.comparison.difference)})"
            for s in report.slice_regressions
        )
        notes.append(
            f"Warning: clear regression in {worse}, even though the overall result may hide it."
        )
    if c.clusters:
        notes.append(f"Items were grouped into {c.clusters} clusters and resampled together.")
    for label, ids in (
        ("baseline", report.only_in_baseline),
        ("candidate", report.only_in_candidate),
    ):
        if ids:
            shown = ", ".join(ids[:MAX_LISTED_IDS]) + (" ..." if len(ids) > MAX_LISTED_IDS else "")
            notes.append(
                f"{len(ids):,} items appear only in the {label} and were left out: {shown}"
            )
    return ["", *notes] if notes else []


def _delta(report: Report, x: float) -> str:
    return f"{x * 100:+.1f} pts" if report.binary else f"{x:+.3g}"


def _pv(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------- HTML

_COLOURS = {
    Outcome.REGRESSION: "#b42318",
    Outcome.IMPROVEMENT: "#067647",
    Outcome.NO_MEANINGFUL_CHANGE: "#175cd3",
    Outcome.NOT_ENOUGH_EVIDENCE: "#93370d",
}

_STYLE = """
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;
  background:#f8f9fb;color:#1d2433}
main{max-width:920px;margin:0 auto;padding:24px 16px}
h1{font-size:20px;margin:0 0 4px}
.files{color:#5d6b82;font-size:13px;word-break:break-all}
.badge{display:inline-block;color:#fff;font-weight:600;border-radius:6px;
  padding:4px 10px;margin:16px 0 8px}
.card{background:#fff;border:1px solid #e3e7ee;border-radius:8px;padding:16px;margin:12px 0}
.headline{font-size:17px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:6px 8px;border-bottom:1px solid #eef1f5;text-align:right}
th:first-child,td:first-child{text-align:left}
.clear{font-weight:600}
.bar{position:relative;height:28px;margin:18px 0 6px;background:#f1f3f7;border-radius:4px}
.zero{position:absolute;top:0;bottom:0;width:1px;background:#5d6b82}
.ci{position:absolute;top:9px;height:10px;border-radius:5px;opacity:.85}
.dot{position:absolute;top:6px;width:16px;height:16px;margin-left:-8px;
  border-radius:50%;border:2px solid #fff}
.scale{display:flex;justify-content:space-between;color:#5d6b82;font-size:12px}
.wrap{overflow-x:auto}
"""


def to_html(report: Report) -> str:
    v, c = report.verdict, report.comparison
    colour = _COLOURS[v.outcome]
    badge = _COLOURS[Outcome.REGRESSION] if report.regression else colour
    e = html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>EvalSignal report</title><style>{_STYLE}</style></head><body><main>",
        "<h1>EvalSignal report</h1>",
        f"<div class='files'>baseline: {e(report.baseline_file)}<br>"
        f"candidate: {e(report.candidate_file)}</div>",
        f"<div class='badge' style='background:{badge}'>{e(verdict_label(report))}</div>",
        f"<div class='card'><div class='headline'>{e(v.headline)}</div>",
        _interval_bar(report, colour),
        *(f"<p>{e(d)}</p>" for d in v.details),
        *(f"<p>{e(n)}</p>" for n in _notes(report) if n),
        "</div>",
        "<div class='card'><table>",
        f"<tr><th>Items compared</th><td>{c.n:,}</td></tr>",
        f"<tr><th>Baseline</th><td>{e(_value(report, c.baseline_mean))}</td></tr>",
        f"<tr><th>Candidate</th><td>{e(_value(report, c.candidate_mean))}</td></tr>",
        f"<tr><th>Change</th><td>{e(_delta(report, c.difference))}</td></tr>",
        f"<tr><th>{c.confidence:.0%} range</th><td>{e(_delta(report, c.ci_low))} to "
        f"{e(_delta(report, c.ci_high))}</td></tr>",
        f"<tr><th>p-value</th><td>{e(_pv(c.p_value))}</td></tr>",
        f"<tr><th>Method</th><td>{e(c.test)}; {e(c.interval)} interval</td></tr>",
    ]
    if c.only_baseline is not None:
        parts += [
            f"<tr><th>Passed before, failed after</th><td>{c.only_baseline:,}</td></tr>",
            f"<tr><th>Failed before, passed after</th><td>{c.only_candidate:,}</td></tr>",
        ]
    parts.append("</table></div>")
    if report.slices:
        parts += [
            "<div class='card'><b>By slice</b> (p adjusted for the number of slices)",
            "<div class='wrap'><table><tr><th>Slice</th><th>Items</th><th>Change</th>"
            "<th>Range</th><th>p (Holm)</th><th>p (BH)</th></tr>",
        ]
        for s in report.slices:
            sc = s.comparison
            cls = " class='clear'" if s.p_holm < report.alpha else ""
            if s in report.slice_regressions:
                cls = f" class='clear' style='color:{_COLOURS[Outcome.REGRESSION]}'"
            parts.append(
                f"<tr{cls}><td>{e(s.name)}</td><td>{sc.n:,}</td>"
                f"<td>{e(_delta(report, sc.difference))}</td>"
                f"<td>{e(_delta(report, sc.ci_low))} to {e(_delta(report, sc.ci_high))}</td>"
                f"<td>{e(_pv(s.p_holm))}</td><td>{e(_pv(s.p_bh))}</td></tr>"
            )
        parts.append("</table></div></div>")
    parts.append("</main></body></html>\n")
    return "".join(parts)


def _value(report: Report, x: float) -> str:
    return f"{x * 100:.1f}%" if report.binary else f"{x:.4g}"


def _interval_bar(report: Report, colour: str) -> str:
    """Change and its interval drawn on a scale centred on zero (plus the margin, if set)."""
    c = report.comparison
    values = [c.ci_low, c.ci_high, c.difference]
    if report.margin is not None:
        values += [-report.margin, report.margin]
    reach = max(abs(x) for x in values if math.isfinite(x)) * 1.15 or 1.0

    def pos(x: float) -> float:
        return 50 + 50 * max(-1.0, min(1.0, x / reach))

    margin_band = ""
    if report.margin is not None:
        left, right = pos(-report.margin), pos(report.margin)
        margin_band = (
            f"<div style='position:absolute;top:0;bottom:0;left:{left:.2f}%;"
            f"width:{right - left:.2f}%;background:#dbe7fb'></div>"
        )
    low, high = pos(c.ci_low), pos(c.ci_high)
    return (
        f"<div class='bar'>{margin_band}<div class='zero' style='left:50%'></div>"
        f"<div class='ci' style='left:{low:.2f}%;width:{max(high - low, 0.5):.2f}%;"
        f"background:{colour}'></div>"
        f"<div class='dot' style='left:{pos(c.difference):.2f}%;background:{colour}'></div></div>"
        f"<div class='scale'><span>{html.escape(_delta(report, -reach))}</span><span>0</span>"
        f"<span>{html.escape(_delta(report, reach))}</span></div>"
    )
