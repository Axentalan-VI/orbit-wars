# Experiments

Every scored submission to [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars),
newest first, taken from this account's submission history. The metric is final
ship count, higher is better.

**The headline: v7_prediction scored 636.1 on 2026-04-21 and nothing beat it in
the 13 submissions that followed.** Thirteen later versions, three weeks of
work, and the peak was on day two of iterating.

| ID | Date | Version / change | Public LB | Notes |
|----|------|------------------|-----------|-------|
| E020 | 2026-05-14 | (notebook v20) | 512.9 | |
| E019 | 2026-05-13 | (notebook v19) | 553.4 | |
| E018 | 2026-05-13 | (notebook v18) | 569.9 | |
| E017 | 2026-04-29 | (notebook v17) | 507.2 | |
| E016 | 2026-04-29 | (notebook v16) | 538.2 | |
| E015 | 2026-04-28 | `v14_payback` | 596.4 | retake what was just lost |
| E014 | 2026-04-28 | v7.1 | 596.9 | revision of the peak; did not beat it |
| E013 | 2026-04-27 | `v13_evac_tight` | 568.7 | stricter give-up threshold |
| E012 | 2026-04-23 | `v12_evacuation` | 577.0 | abandon unholdable planets |
| E011 | 2026-04-22 | `v11_patience` | 608.6 | **second best**; wait for growth |
| E010 | 2026-04-21 | (notebook v10) | 567.7 | |
| E009 | 2026-04-21 | `v9_control` | 595.7 | hold territory |
| E008 | 2026-04-21 | `v8_targeting` | 575.1 | choose the target, not the nearest |
| E007 | 2026-04-21 | `v7_prediction` | **636.1** | **best ever recorded** |
| E006 | 2026-04-21 | `v6_surgical` | 510.6 | v3 core + tactical upgrades |
| E005 | 2026-04-20 | (notebook v5) | 590.8 | |
| E004 | 2026-04-20 | `v4_lookahead` | 522.3 | |
| E003 | 2026-04-20 | `v3_ledger` | 599.4 | strong early baseline |
| E002 | 2026-04-20 | `v2_adaptive` | 531.9 | |
| E001 | 2026-04-20 | (first notebook) | 409.2 | starting point |

## What this table says

- **Opponent prediction (v7) was the single biggest win**, +36 over the best
  heuristic before it, and nothing since has matched it.
- **The spread among the good bots is ~40 points (595-636)** on a single
  leaderboard sample. Without a local win-rate measurement over many seeded
  matches, most of the differences between v8 and v16 are indistinguishable
  from noise — they were read as progress at the time.
- **No cross-validation equivalent was ever recorded.** `eval/run_match.py`
  exists and can play seeded matches locally; the results of those runs were
  never written down, so every decision above rests on one noisy LB number.

That is the lesson worth carrying: for a game competition, local win rate over
N seeded matches is the CV, and it has to be logged before the submission, not
after.
