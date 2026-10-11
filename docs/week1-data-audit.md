# Week 1 data audit: go / no-go

**Decision: GO.**

```bash
uv run python audit/download_data.py   # ~66 MB, with a checksum manifest
uv run python audit/data_audit.py      # writes docs/audit/data_audit.json
```

## Sources

| Dataset | Use | Size | Licence |
|---|---|---|---|
| `lmsys/mt_bench_human_judgments` (Hugging Face) | Judge reliability: expert humans and GPT-4 judging the same answer pairs | 1.4 MB | CC-BY-4.0 (dataset card) |
| `open-llm-leaderboard-old/details_mistralai__Mistral-7B-v0.1` and `..._Mistral-7B-Instruct-v0.1` | Paired before/after case studies: per-question MMLU and ARC-Challenge results | ~64 MB (only MMLU and ARC files) | **None stated**: used locally only; only derived statistics are published |

The MMLU and ARC question texts inside the leaderboard files have their own
licences; this project never republishes questions, answers or model
outputs from any dataset. Everything downloaded stays in `data/` (not in
git); `data/raw/MANIFEST.json` records each file's URL, size and SHA-256.

## Findings

### Paired results: complete ✅
- **14,015 MMLU questions** (57 subjects, 91–1,534 questions each) and
  **1,170 ARC-Challenge questions** are present for both models.
- Questions are paired by the evaluation harness's hash of the question
  text: **0 unmatched** in either direction.
- Each row has `acc` (1 = correct) and `acc_norm` (length-normalised). The
  case study uses raw `acc`: a question is correct when `acc` is 1.0. This
  gives a paired binary outcome per question, exactly what paired tests
  need. Runs used: base `2023-09-27T15-30-59.039834`, Instruct
  `2023-10-10T06-38-48.353025` (see `audit/download_data.py`).

### Judge reliability: usable ✅
- 3,355 human judgments from 65 judges and 2,400 GPT-4 pairwise judgments
  over 6 models, as "which answer is better, or a tie".
- **1,232** answer pairs were judged by both humans and GPT-4; **761** by two
  or more humans, which gives a human-vs-human agreement baseline.
- **Checked:** GPT-4 picks "model_b" 1,610 times and "model_a" 190
  times, while humans are balanced (1,282 vs 1,293). This turned out to be
  how pairs are ordered in the GPT-4 file (the weaker model usually sits in
  slot A), not position bias. Agreement is therefore computed by model
  identity, never by slot. GPT-4 also has 380 verdicts labelled "tie
  (inconsistent)": its answer changed when the two responses
  swapped places, a direct measure of position sensitivity.

## Plan confirmed
- Simulations validate the statistics (false-alarm rate, interval coverage, power).
- Mistral base vs Instruct: overall, per subject (with multiple-testing
  correction), and "how often small test sets mislead".
- MT-Bench: judge vs human agreement compared with human vs human agreement.
