import pytest

from evalsignal.judge import TIE, evaluate_judge, verdict_from_slots


def is_human(name: object) -> bool:
    return str(name).startswith("human")


def test_slots_are_turned_into_model_identities() -> None:
    v = verdict_from_slots("q1", "llama", "gpt", "model_b", "judge")
    assert (v.models, v.winner) == (("gpt", "llama"), "gpt")
    assert verdict_from_slots("q1", "llama", "gpt", "tie (inconsistent)", "j").inconsistent
    with pytest.raises(ValueError):
        verdict_from_slots("q1", "a", "b", "model_c", "j")


def test_agreement_is_by_identity_even_when_slots_are_swapped() -> None:
    verdicts = [
        verdict_from_slots("q1", "x", "y", "model_a", "gpt4"),  # x wins
        verdict_from_slots("q1", "y", "x", "model_b", "human_1"),  # x wins, other slot order
        verdict_from_slots("q2", "x", "y", "model_b", "gpt4"),  # y wins
        verdict_from_slots("q2", "x", "y", "tie", "human_1"),
    ]
    report = evaluate_judge(verdicts, "gpt4", is_human)
    agreement = report.judge_vs_humans
    assert (agreement.pairs, agreement.with_ties) == (2, 0.5)
    assert (agreement.decisive_pairs, agreement.without_ties) == (1, 1.0)


def test_human_baseline_and_inconsistency_share() -> None:
    verdicts = [
        verdict_from_slots("q1", "x", "y", "model_a", "human_1"),
        verdict_from_slots("q1", "x", "y", "model_a", "human_2"),
        verdict_from_slots("q1", "x", "y", "model_b", "human_3"),
        verdict_from_slots("q1", "x", "y", "tie (inconsistent)", "gpt4"),
        verdict_from_slots("q2", "x", "y", "model_a", "gpt4"),
    ]
    report = evaluate_judge(verdicts, "gpt4", is_human)
    assert report.human_vs_human.pairs == 3  # (1,2) agree, (1,3) and (2,3) disagree
    assert report.human_vs_human.with_ties == pytest.approx(1 / 3)
    assert report.inconsistent_share == 0.5
    assert TIE == "tie"


def test_perfect_agreement_has_kappa_one() -> None:
    verdicts = []
    for i, winner in enumerate(["model_a", "model_b", "tie", "model_a"]):
        verdicts.append(verdict_from_slots(f"q{i}", "x", "y", winner, "gpt4"))
        verdicts.append(verdict_from_slots(f"q{i}", "x", "y", winner, "human_1"))
    agreement = evaluate_judge(verdicts, "gpt4", is_human).judge_vs_humans
    assert agreement.with_ties == 1.0 and agreement.kappa == pytest.approx(1.0)
