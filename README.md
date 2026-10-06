# Orbit Wars

Kaggle competition: https://www.kaggle.com/competitions/orbit-wars

This repo contains a sequence of bot versions for Orbit Wars.  
The goal is to improve score step by step while keeping logic stable and fast.

## Result

**636.1 final ship count**, the best of 20 scored submissions across the seven
bot versions below. The competition closed on 2026-07-07, with a final standing
of 3232 of 4729 teams. Each version's own
behaviour and what changed between them is documented further down — the
version history is the point of this repo.

## Game in 20 seconds

- Map: continuous 100x100 board.
- Planets rotate around a central sun.
- Fleets are launched with angle + ship count.
- 500 turns, 1 second per turn.
- Winner is highest total ships at the end.

## Project layout

- `agents/` - all bot versions.
- `eval/run_match.py` - local 2p and 4p matches.
- `scripts/` - utilities for testing and notebook generation.
- `rules.py` - helper logic for mechanics.
- `tests/` - unit tests.

## Bot versions (detailed explanation)

### v0_random

Folder: `agents/v0_random/`  
How it works:
- Sends legal random actions (random source, random direction, random ship count).
- Does not estimate travel outcomes, defense, or enemy reactions.

Why it exists:
- Validates that action formatting and submission plumbing are correct.
- Serves as the minimum baseline to confirm later bots are truly better.

### v1_heuristic

Folder: `agents/v1_heuristic/`  
How it works:
- Computes candidate attacks from owned planets to neutral/enemy planets.
- Scores each candidate with a simple ROI idea:
	production gain, distance/ETA, and ships required.
- Greedily picks top candidates while keeping a local defense reserve.

Strength:
- Much better than random because it has a clear economy-first objective.

Limitation:
- Uses mostly snapshot logic (current board state), so it can misread situations
	where multiple fleets are already in flight.

### v2_adaptive

Folder: `agents/v2_adaptive/`  
How it works:
- Keeps v1-style candidate scoring but changes weights by game context.
- Typical context switches:
	early expansion, mid-game balancing, late-game consolidation.
- Adjusts aggression/defense bias when behind or under pressure.

Strength:
- More flexible than v1; avoids one fixed playstyle.

Limitation:
- Still limited by snapshot-based targeting and imperfect fleet-impact estimates.

### v3_ledger

Folder: `agents/v3_ledger/`  
How it works:
- Builds an arrival ledger per planet:
	`arrivals[planet_id] = [(eta, owner, ships), ...]`.
- Predicts where each in-flight fleet will impact and when.
- Replays arrivals over time to answer:
	"How many ships are really required at ETA t to capture/hold this planet?"
- Uses timed defense planning (`H_DEF`) to reserve enough ships for likely threats.
- Supports wave stacking (multiple sources can reinforce the same target).
- Adds 4-player bias to avoid kingmaker behavior by preferring weaker opponents.

Why it performed well:
- Biggest jump in tactical accuracy came from time-aware accounting,
	not from complex search.
- Decisions are still fast and robust under leaderboard diversity.

### v4_lookahead

Folder: `agents/v4_lookahead/`  
How it works:
- Starts from v3 structure, then generates multiple policy profiles
	(balanced/aggressive/defensive/comet-heavy).
- For each profile, builds an action set and evaluates it with short-horizon
	forward simulation (`H_EVAL`) across planets.
- Chooses the profile/action set with best simulated score.

Strength:
- Better local planning than pure greedy in some self-play setups.

Why it underperformed on Kaggle:
- Added complexity increased variance and made decisions less stable.
- Local self-play wins did not transfer well against diverse external bots.

### v5_opponent

Folder: `agents/v5_opponent/`  
How it works:
- Inherits v4 lookahead.
- Adds an explicit opponent launch model during evaluation:
	predicts likely enemy attacks from their surplus planets onto weak owned targets.
- Merges these hypothetical enemy launches into forward simulation before scoring.

Strength:
- Corrected some v4 blind spots and improved local robustness.

Why still below v3 on Kaggle:
- Opponent model assumptions were not universally accurate.
- Strategic overfitting to local matchups remained a problem.

### v6_surgical

Folder: `agents/v6_surgical/`  
How it works:
- Reverts to v3 as the stable backbone.
- Adds only three low-risk tactical rules:
	1. Skip contested neutrals where two enemies likely collide (let them trade).
	2. Preserve reserve shortly before comet spawns.
	3. Penalize/avoid captures likely to be retaken immediately.

Design philosophy:
- Prefer small, explainable improvements over heavy simulation.
- Keep the strongest part of v3 (ledger accuracy) and avoid v4/v5 over-complexity.

Current goal:
- Keep or beat v3 leaderboard score with incremental, testable changes.

## Later versions (v7 onward)

The write-ups above stop at v6, but the repo carries eleven more agents. Their
names say what each one tried; the per-version reasoning is not written up yet.

| Version | Idea in the name |
| --- | --- |
| `v7_prediction` | predict the opponent's next move, not just its policy |
| `v8_targeting` | choose which opponent planet to hit, rather than the nearest |
| `v9_control` | hold territory instead of trading ships |
| `v10_rank` | rank candidate moves by expected value |
| `v11_patience` | wait for growth before committing a fleet |
| `v12_evacuation` | abandon a planet that cannot be held |
| `v13_evac_tight` | the same, with a stricter give-up threshold |
| `v14_payback` | prioritise retaking what was just lost |
| `v16_hoard` | accumulate ships rather than spend them continuously |
| `v18_phantom_scorched` | feint plus denial: leave nothing worth capturing |

Numbering has gaps (no v15, v17) where a version was tried and dropped.

## What changed most from version to version

- v0 -> v1: from random actions to value-based targeting.
- v1 -> v2: context-aware weights (adaptive behavior).
- v2 -> v3: time-based fleet ledger (major quality jump).
- v3 -> v4: profile lookahead simulation (more complex, less stable).
- v4 -> v5: explicit opponent prediction in lookahead.
- v5 -> v6: back to v3 core + minimal tactical upgrades.

## Quick local run

```powershell
python -m venv .venv
pip install -r requirements.txt
python eval/run_match.py --a agents/v6_surgical --b agents/v3_ledger --n 5 --mode 2p
```

## Notebook submissions

Each agent folder can contain a Kaggle notebook-style submission file:

- `agents/v3_ledger/submission.ipynb`
- `agents/v5_opponent/submission.ipynb`
- `agents/v6_surgical/submission.ipynb`

These notebooks write `submission.py` in one cell using `%%writefile`.
