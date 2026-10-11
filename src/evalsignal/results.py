"""Read evaluation results and pair two runs item by item.

A results file is CSV or JSON lines with one row per evaluated item:

    id      required; identifies the item (question, test case)
    passed  true/false, yes/no, 1/0        } one of the two
    score   any number (rating, BLEU, ...) }
    slice   optional group name ("math", "coding") for per-slice results
    cluster optional group of related items, resampled together
    run     optional repeat number when each item was evaluated several times

Pairing is by ``id``. Items present in only one run are counted and
reported, never silently dropped. With repeats, each item's outcome is the
mean over its runs.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

_TRUE = {"true", "1", "yes", "y", "pass", "passed", "t"}
_FALSE = {"false", "0", "no", "n", "fail", "failed", "f"}


@dataclass(slots=True)
class Results:
    """One run: every row, keyed by item id (several rows per id if repeated)."""

    values: dict[str, list[float]]
    slices: dict[str, str]
    clusters: dict[str, str]
    binary: bool
    source: str = ""

    @property
    def repeated(self) -> bool:
        return any(len(v) > 1 for v in self.values.values())


@dataclass(slots=True)
class Paired:
    ids: list[str]
    baseline: npt.NDArray[np.float64]
    candidate: npt.NDArray[np.float64]
    binary: bool
    slices: npt.NDArray[np.str_] | None
    clusters: npt.NDArray[np.int64] | None
    only_in_baseline: list[str] = field(default_factory=list)
    only_in_candidate: list[str] = field(default_factory=list)


def read_results(path: Path) -> Results:
    rows = list(_rows(path))
    if not rows:
        raise ValueError(f"{path}: no rows")
    has_passed = "passed" in rows[0]
    if not has_passed and "score" not in rows[0]:
        raise ValueError(f"{path}: each row needs 'id' and either 'passed' or 'score'")
    values: dict[str, list[float]] = defaultdict(list)
    slices: dict[str, str] = {}
    clusters: dict[str, str] = {}
    for number, row in enumerate(rows, 1):
        item = str(row.get("id", "")).strip()
        if not item:
            raise ValueError(f"{path}: row {number} has no id")
        raw = row.get("passed") if has_passed else row.get("score")
        values[item].append(_parse_value(raw, has_passed, path, number))
        if row.get("slice") not in (None, ""):
            slices[item] = str(row["slice"])
        if row.get("cluster") not in (None, ""):
            clusters[item] = str(row["cluster"])
    binary = has_passed or all(v in (0.0, 1.0) for vs in values.values() for v in vs)
    return Results(dict(values), slices, clusters, binary, str(path))


def pair(baseline: Results, candidate: Results) -> Paired:
    shared = sorted(baseline.values.keys() & candidate.values.keys())
    if not shared:
        raise ValueError("the two runs have no item ids in common")
    if baseline.binary != candidate.binary:
        raise ValueError("one run has pass/fail outcomes and the other has scores")
    slices = {**baseline.slices, **candidate.slices}
    clusters = {**baseline.clusters, **candidate.clusters}
    cluster_ids = {
        name: i for i, name in enumerate(sorted({clusters[i] for i in shared if i in clusters}))
    }
    return Paired(
        ids=shared,
        baseline=np.array([np.mean(baseline.values[i]) for i in shared]),
        candidate=np.array([np.mean(candidate.values[i]) for i in shared]),
        # Averaging repeats makes outcomes fractional: compare them as scores.
        binary=baseline.binary and not (baseline.repeated or candidate.repeated),
        slices=np.array([slices.get(i, "(none)") for i in shared]) if slices else None,
        clusters=(
            np.array([cluster_ids[clusters[i]] for i in shared], dtype=np.int64)
            if clusters and all(i in clusters for i in shared)
            else None
        ),
        only_in_baseline=sorted(baseline.values.keys() - candidate.values.keys()),
        only_in_candidate=sorted(candidate.values.keys() - baseline.values.keys()),
    )


def _rows(path: Path):  # type: ignore[no-untyped-def]
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".jsonl", ".ndjson") or text.lstrip().startswith("{"):
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}: line {number} is not valid JSON") from exc
    else:
        yield from csv.DictReader(text.splitlines())


def _parse_value(raw: object, as_bool: bool, path: Path, number: int) -> float:
    if as_bool:
        if isinstance(raw, bool):
            return float(raw)
        text = str(raw).strip().lower()
        if text in _TRUE:
            return 1.0
        if text in _FALSE:
            return 0.0
        raise ValueError(f"{path}: row {number}: 'passed' must be true/false, got {raw!r}")
    try:
        value = float(str(raw))
    except ValueError as exc:
        raise ValueError(f"{path}: row {number}: 'score' must be a number, got {raw!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{path}: row {number}: 'score' must be finite")
    return value
