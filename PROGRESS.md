# Progress

## Current status

Finished. The competition closed on 2026-07-07 with a final standing of
**3232 of 4729 teams**; the best submission was `v7_prediction` at **636.1**
final ship count. 17 agent versions are in `agents/`, 18 local tests pass.

## Last session (2026-10-06)

- Recovered the full submission history into `EXPERIMENTS.md` — it had never
  been written down anywhere.
- Found that **v7 was the peak and 13 later submissions never beat it**, which
  was not visible while the work was happening.
- Documented versions v7-v18 in the README; the write-ups had stopped at v6.
- Fixed `pytest`: `scripts/test_*.py` are match runners needing
  `kaggle_environments`, so a bare run aborted on 13 collection errors instead
  of running the real suite in `tests/`. Added `pytest.ini`.

## Open issues

- **No local evaluation was ever recorded.** `eval/run_match.py` can play
  seeded matches, but no win rates were logged, so the version-to-version
  decisions rest on single noisy leaderboard numbers.
- Versions v7-v18 are listed but not written up: what each one changed and
  why it did or did not work.

## Next steps (prioritized)

1. Run `eval/run_match.py` for v7 against v3, v9, v11 and v14 over ~200 seeded
   matches each and record the win rates. This is the measurement that was
   missing all along, and it answers whether v7 was genuinely best or lucky.
2. Write up the version history properly — it is the most interesting thing
   about this repo and the portfolio already points at it.

## Decisions & rationale

- Numbering has gaps (no v15, v17) where a version was tried and dropped
  without a submission.
- `v7.1` was an attempt to improve the peak and scored 596.9 against 636.1, so
  the original v7 stayed the submission of record (2026-04-28).
