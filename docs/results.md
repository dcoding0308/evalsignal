# Results

Three questions, each answered with a script you can rerun. Every number
below comes from a JSON file in `docs/results/`.

```bash
uv run python audit/download_data.py   # ~66 MB, checksummed; stays in data/ (not in git)
uv run python audit/simulate.py        # docs/results/simulation.json   (a few minutes)
uv run python audit/case_mistral.py    # docs/results/case_mistral.json (~5 s)
uv run python audit/case_judge.py      # docs/results/judge_mt_bench.json (~5 s)
```

Only derived statistics are published. No questions, answers or model
outputs from any dataset are stored in this repository.

## 1. Do the statistics keep their promises? (simulation)

**Protocol.** Pass/fail results for two versions were simulated where the
true answer is known: 30 settings (items n = 50 to 1,000; share of items
that change result between versions 5%, 15%, 30%; true change zero or one
third of that share), 2,000 runs each, fixed seed.

| Check | Promise | Measured |
|---|---|---|
| False alarms (true change = 0), McNemar exact test | at most 5% | **0.05% to 4.6%** in all 15 no-change settings |
| 95% range, Agresti–Min (default) | covers the true change 95% of the time | **94.1% to 99.7%** |
| 95% range, simple Wald | same | 87.0% to 97.9% (too narrow at n = 50) |
| 95% range, bootstrap | same | 77.2% to 97.9% (too narrow at n = 50) |

Agresti–Min was chosen as the default because of this table; the simpler
methods are only acceptable from about 200 items.

**Sample-size advice.** The usual textbook formula for "how many items do I
need" promised 80% power but delivered **77.2% to 79.6%**. EvalSignal now
computes power exactly and adds items until the exact power reaches the
target; the simulated power at the sizes it recommends is **79.4% to
80.5%**.

| Disagreement | True change | Textbook n | Its real power | EvalSignal n | Its real power |
|---|---|---:|---:|---:|---:|
| 10% | 3 pts | 870 | 78.8% | 920 | 79.4% |
| 15% | 5 pts | 469 | 77.2% | 498 | 79.5% |
| 20% | 4 pts | 979 | 79.6% | 1,021 | 79.4% |
| 30% | 6 pts | 652 | 77.4% | 681 | 80.5% |

(Each simulated power has a standard error of about 0.9 points.)

**Related items.** When items come in groups (for example, several
questions about the same document) and a group tends to move together,
item-by-item tests are overconfident. With 40 groups of 10 and no true
change, the item-level McNemar test raised a false alarm **12.2%** of the
time; resampling whole groups (the `cluster` column) brought it back to
**3.2%** (500 runs).

## 2. A real model change: Mistral-7B base → Mistral-7B-Instruct

**Protocol.** Per-question results from the archived Open LLM Leaderboard
(`open-llm-leaderboard-old`) evaluation runs of each model:

| Model | Run |
|---|---|
| Mistral-7B-v0.1 | `2023-09-27T15-30-59.039834` |
| Mistral-7B-Instruct-v0.1 | `2023-10-10T06-38-48.353025` |

- **Metric: raw per-question `acc`.** A question counts as correct when its
  `acc` is 1.0. The files also contain `acc_norm` (answer likelihood
  normalised by length), which the leaderboard used for some headline
  scores such as ARC, so published leaderboard numbers can differ from the
  ones here. Both models are measured the same way, which is what a paired
  comparison needs.
- Same settings for both runs: 5 example questions in the prompt for MMLU,
  25 for ARC-Challenge.
- Questions are paired by the harness's hash of the question text (0
  unmatched). 14,015 MMLU and 1,170 ARC-Challenge questions are the paired
  counts in these result files, not a claim about every version of those
  benchmarks.

| Benchmark | Questions | Base | Instruct | Change | 95% range | Verdict |
|---|---:|---:|---:|---:|---|---|
| MMLU | 14,015 | 62.6% | 53.7% | −8.9 pts | −9.7 to −8.1 | regression, p < 0.001 |
| ARC-Challenge | 1,170 | 56.8% | 52.3% | −4.5 pts | −6.7 to −2.3 | regression, p < 0.001 |

On MMLU, 2,287 questions went from right to wrong and 1,038 from wrong to
right: the paired view shows how much churn hides behind one average.

**Per subject (57 MMLU subjects).**

| Rule | Subjects flagged as changed |
|---|---:|
| p < 0.05 for each subject, no correction | 37 |
| Benjamini–Hochberg (controls the share of false discoveries) | 33 |
| Holm (controls the chance of any false alarm) | 18 |

All 18 subjects that survive Holm got worse (for example anatomy −20.7
pts, prehistory −15.7 pts, professional psychology −16.4 pts). Testing 57
subjects without correction would report about three of them as "changed"
by chance even if nothing changed; EvalSignal adjusts by default.

**How often do small test sets mislead?** Random subsets of n questions
were drawn from MMLU (2,000 per size) and judged two ways, against the
full-set answer (Instruct is worse):

| Questions | "Higher score wins" points the wrong way | EvalSignal: correct call | EvalSignal: not enough evidence | EvalSignal: wrong call |
|---:|---:|---:|---:|---:|
| 50 | 7.2% | 15.6% | 84.4% | 0.05% |
| 100 | 2.3% | 37.4% | 62.7% | 0% |
| 200 | 0.35% | 72.0% | 28.0% | 0% |
| 500 | 0% | 98.6% | 1.4% | 0% |
| 1,000 | 0% | 100% | 0% | 0% |

This is a large change (−8.9 points); for smaller changes the naive rule
is wrong far more often. EvalSignal's answer to a small test set is
usually "not enough evidence, here is how many items you would need",
rather than a confident wrong answer.

## 3. Can an LLM judge be trusted? (MT-Bench)

**Protocol.** MT-Bench human judgments (3,355 from 65 experts) and GPT-4
pairwise judgments (2,400) on the same question, turn and pair of models.
Verdicts are compared by model identity (not by "A"/"B" slot), with 95%
ranges from resampling questions.

| | Agreement (with ties) | Agreement when neither says tie | Cohen's kappa | Comparisons |
|---|---|---:|---:|---:|
| GPT-4 vs a human | **66.2%** (64.2–68.2) | 85.6% | 0.48 | 3,355 |
| Human vs another human | **65.5%** (62.6–68.4) | 82.7% | 0.47 | 2,368 |

GPT-4 agrees with experts about as often as experts agree with each other.
But in **15.8%** of its verdicts, GPT-4 changed its answer when the two
responses swapped places (recorded as "tie (inconsistent)"). The judge is
about as reliable as a person on average and sensitive to answer order.

## Limits

- The case study is one model pair on public benchmarks, chosen because
  per-question results are published. It shows the method, not a claim
  about Mistral in general.
- The leaderboard data has no stated licence, so it is used locally only.
- Power calculations assume items are independent; with groups, use the
  `cluster` column (the sample-size advice does not yet account for groups).
- Scores (not pass/fail) use a bootstrap interval and a sign-flip
  permutation test; they were not part of the simulation study above.
