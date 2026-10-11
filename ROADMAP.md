# Roadmap

## Done

- Paired comparison of two runs: McNemar exact test and Agresti–Min
  interval (pass/fail), bootstrap and sign-flip permutation (scores).
- Grouped test cases (`cluster`) resampled together.
- Per-slice results with Holm and Benjamini–Hochberg correction, and a
  warning when a slice clearly regresses while the overall result does not.
- Four-way verdict with "not enough evidence" sample-size advice.
- Exact sample-size and minimum-detectable-change calculations.
- Judge-vs-people agreement, benchmarked against people-vs-people.
- CLI, HTML report, Markdown for PR comments, JSON, GitHub Action example.
- Simulation study and two case studies on public data.
- Static demo page built from the tool's own output (`site/`).

## Next

- Sample-size advice that accounts for grouped test cases.
- Sequential testing: stop an evaluation early once the answer is clear,
  without inflating false alarms.
- Run-to-run noise: when a model is non-deterministic, separate "this
  change" from "the same version run twice".
- Comparing more than two versions at once.
- Simulation study for score outcomes (so far only pass/fail is simulated).

## Corrections log

Mistakes found by checking the work, and what changed.

| Found | Problem | Fix |
|---|---|---|
| Simulation | The textbook sample-size formula promised 80% power but delivered 77–80%. | Exact McNemar power; recommended sizes now deliver 79–81%. |
| Simulation | The simple (Wald) and bootstrap 95% ranges covered the truth only 87% and 77% of the time at 50 items. | Agresti–Min is the default; the others remain as options. |
| Simulation | Item-level tests raised 12% false alarms when related items move together. | `cluster` column with group-level resampling (3% false alarms). |
| Data audit | GPT-4 seemed to prefer the second answer 1,610 to 190, which looked like position bias. | It was the file's ordering of models; agreement is computed by model identity, and position sensitivity is measured from the judge's own swap-inconsistency flags. |
| Examples | A small test set produced "not enough evidence" with no explanation when no change size was detectable. | The verdict now says the test set is too small to detect any change. |
| Examples | A flat overall result hid a clear regression in one slice. | Slice regressions are shown in the headline and count for `--fail-on-regression`. |
| Review | The README summarised the 50-question study as "1 wrong call, otherwise not enough evidence", hiding the 15.6% correct calls. | All three outcomes reported. |
| Review | The Mistral case study did not say it uses raw `acc` (not `acc_norm`) or which runs. | Metric, runs, prompt settings and paired counts documented. |
| Review | A statistically significant change whose whole range lay inside the margin was reported as a regression, contradicting the documented meaning of `no_meaningful_change`. | The margin is checked first, for the overall result and for each slice. |
| Review | Above 100,000 items, sample-size advice came from the approximation without saying so. | `is_approximate`; the CLI and verdict say when the approximation is used; tested. |
| Demo page | Sample-size advice stepped back one item at a time after overshooting, taking seconds for large sizes. | Binary search: same answers on all 41 checked settings, about 4x faster. |
