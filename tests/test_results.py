from pathlib import Path

import pytest

from evalsignal.results import pair, read_results


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_csv_pass_fail_with_slices(tmp_path: Path) -> None:
    run = read_results(
        write(tmp_path / "a.csv", "id,passed,slice\nq1,true,math\nq2,no,law\nq3,1,math\n")
    )
    assert run.binary and run.values == {"q1": [1.0], "q2": [0.0], "q3": [1.0]}
    assert run.slices == {"q1": "math", "q2": "law", "q3": "math"}


def test_jsonl_scores(tmp_path: Path) -> None:
    run = read_results(
        write(tmp_path / "a.jsonl", '{"id": "q1", "score": 7.5}\n{"id": "q2", "score": 3}\n')
    )
    assert not run.binary and run.values == {"q1": [7.5], "q2": [3.0]}


def test_pairing_reports_unmatched_items(tmp_path: Path) -> None:
    a = read_results(write(tmp_path / "a.csv", "id,passed\nq1,1\nq2,0\nq3,1\n"))
    b = read_results(write(tmp_path / "b.csv", "id,passed\nq2,1\nq3,1\nq4,0\n"))
    paired = pair(a, b)
    assert paired.ids == ["q2", "q3"]
    assert paired.baseline.tolist() == [0.0, 1.0] and paired.candidate.tolist() == [1.0, 1.0]
    assert (paired.only_in_baseline, paired.only_in_candidate) == (["q1"], ["q4"])


def test_repeated_runs_are_averaged_and_compared_as_scores(tmp_path: Path) -> None:
    a = read_results(write(tmp_path / "a.csv", "id,passed,run\nq1,1,1\nq1,0,2\nq2,1,1\nq2,1,2\n"))
    b = read_results(write(tmp_path / "b.csv", "id,passed\nq1,1\nq2,1\n"))
    paired = pair(a, b)
    assert paired.baseline.tolist() == [0.5, 1.0]
    assert not paired.binary


def test_clusters_are_numbered(tmp_path: Path) -> None:
    a = read_results(
        write(tmp_path / "a.csv", "id,passed,cluster\nq1,1,doc7\nq2,0,doc7\nq3,1,doc2\n")
    )
    paired = pair(a, a)
    assert paired.clusters is not None and paired.clusters.tolist() == [1, 1, 0]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("id,passed\nq1,maybe\n", "must be true/false"),
        ("id,score\nq1,abc\n", "must be a number"),
        ("id,other\nq1,1\n", "either 'passed' or 'score'"),
        ("id,passed\n,1\n", "has no id"),
    ],
)
def test_bad_files_fail_with_a_clear_message(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_results(write(tmp_path / "bad.csv", text))


def test_runs_with_nothing_in_common_are_rejected(tmp_path: Path) -> None:
    a = read_results(write(tmp_path / "a.csv", "id,passed\nq1,1\n"))
    b = read_results(write(tmp_path / "b.csv", "id,passed\nq9,1\n"))
    with pytest.raises(ValueError, match="no item ids in common"):
        pair(a, b)
