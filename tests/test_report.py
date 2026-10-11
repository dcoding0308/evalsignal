from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalsignal.cli import main
from evalsignal.report import build_report, to_html, to_json, to_markdown, to_text


def write(path: Path, rows: list[str], header: str = "id,passed,slice") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


@pytest.fixture
def runs(tmp_path: Path) -> tuple[Path, Path]:
    """Overall flat, but slice "b" loses 30 of 100 passes and slice "a" gains 30."""
    before, after = [], []
    for k in range(100):
        before.append(f"a{k},{k < 50},a")
        after.append(f"a{k},{k < 80},a")
        before.append(f"b{k},{k < 80},b")
        after.append(f"b{k},{k < 50},b")
    return write(tmp_path / "before.csv", before), write(tmp_path / "after.csv", after)


def test_flat_overall_but_slice_regression(runs: tuple[Path, Path]) -> None:
    report = build_report(*runs)
    assert report.comparison.difference == 0
    assert [s.name for s in report.slice_regressions] == ["b"]
    assert report.regression
    text = to_text(report)
    assert text.startswith("NOT ENOUGH EVIDENCE OVERALL, REGRESSION IN 1 SLICE")
    assert "clear regression in b (-30.0 pts)" in text
    assert "| b | 200" not in to_markdown(report)  # slice rows count items per slice
    assert "| b | 100 | **-30.0 pts**" in to_markdown(report)


def test_slice_regression_respects_the_margin(runs: tuple[Path, Path]) -> None:
    # Slice "b" drops 30 points; its whole range lies within a 50-point margin.
    assert not build_report(*runs, margin=50).slice_regressions
    assert not build_report(*runs, margin=50).regression
    assert [s.name for s in build_report(*runs, margin=5).slice_regressions] == ["b"]


def test_margin_is_in_points_for_pass_fail(tmp_path: Path) -> None:
    rows = [f"x{k},{k % 2 == 0}" for k in range(2000)]
    before = write(tmp_path / "b.csv", rows, "id,passed")
    after = write(tmp_path / "a.csv", rows, "id,passed")
    report = build_report(before, after, margin=2)
    assert report.margin == pytest.approx(0.02)
    assert report.verdict.outcome.value == "no_meaningful_change"
    with pytest.raises(ValueError, match="positive"):
        build_report(before, after, margin=0)


def test_json_is_valid_and_lists_missing_items(tmp_path: Path) -> None:
    before = write(tmp_path / "b.csv", ["x,true", "y,false", "only-before,true"], "id,passed")
    after = write(tmp_path / "a.csv", ["x,true", "y,true"], "id,passed")
    data = json.loads(to_json(build_report(before, after)))
    assert data["only_in_baseline"] == ["only-before"]
    assert data["comparison"]["n"] == 2
    assert data["outcome"] == "not_enough_evidence"


def test_html_escapes_everything_from_the_input(tmp_path: Path) -> None:
    evil = "<script>alert(1)</script>"
    slices = [evil if k % 2 else "ok" for k in range(40)]
    before = write(tmp_path / "b.csv", [f'x{k},true,"{s}"' for k, s in enumerate(slices)])
    after = write(tmp_path / "a.csv", [f'x{k},{k > 9},"{s}"' for k, s in enumerate(slices)])
    page = to_html(build_report(before, after))
    assert "<script" not in page
    assert "&lt;script&gt;" in page


def test_cli_exit_codes(runs: tuple[Path, Path], tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    before, after = (str(p) for p in runs)
    assert main(["compare", before, after]) == 0
    assert main(["compare", before, after, "--fail-on-regression"]) == 1
    page = tmp_path / "report.html"
    assert main(["compare", before, after, "--html", str(page)]) == 0
    assert page.read_text().startswith("<!doctype html>")
    assert main(["compare", before, str(tmp_path / "missing.csv")]) == 2
    assert "evalsignal:" in capsys.readouterr().err


def test_cli_plan(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["plan", "--disagree", "15", "--change", "3"]) == 0
    assert "1,362 items" in capsys.readouterr().out
    assert main(["plan", "--disagree", "15", "--items", "10"]) == 0
    assert "too few" in capsys.readouterr().out
    assert main(["plan", "--disagree", "5", "--change", "8"]) == 2
    assert main(["plan", "--disagree", "5", "--change", "0.1"]) == 0
    assert "normal approximation" in capsys.readouterr().out


def test_cli_judge(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    rows = ["item,model_a,model_b,winner,judge"]
    for k in range(20):
        rows += [f"q{k},m1,m2,model_a,bot", f"q{k},m1,m2,model_a,ann", f"q{k},m2,m1,model_b,bob"]
    path = tmp_path / "judge.csv"
    path.write_text("\n".join(rows) + "\n")
    assert main(["judge", str(path), "--judge", "bot"]) == 0
    out = capsys.readouterr().out
    assert "bot vs people: 100.0% agree" in out
    assert "people vs people: 100.0% agree" in out
    assert main(["judge", str(path), "--judge", "nobody"]) == 2
