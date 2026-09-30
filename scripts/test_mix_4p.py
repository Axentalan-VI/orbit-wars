"""Kaggle-proxy battery: candidate vs RANDOM MIX of historical agents.

Rationale: 4p self-play vs 3x v7 over-rewards defensive/conservative changes
because all 3 opponents play identically and strong. Kaggle's leaderboard
matches you against a diverse pool with a wide skill range. We mimic that
by sampling 3 opponents per game from a weighted pool of past bots.

Usage:
    .venv\Scripts\python.exe scripts\test_mix_4p.py CANDIDATE_DIR LABEL [n_games]

Example:
    python scripts\test_mix_4p.py agents\v14_payback v14 40
"""
import importlib.util, sys, time, random, argparse
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(line_buffering=True)
from kaggle_environments import make

ROOT = Path(r"e:\Kaggle\Orbit Wars")

# Opponent pool: (dir_name, label, weight). Weights tuned to roughly mimic
# the spread of skill levels we have submitted to Kaggle. Heavier weight on
# "average" bots (v3-v6) since most leaderboard opponents are mid-skill.
POOL = [
    ("v0_random",      "v0",  1),
    ("v1_heuristic",   "v1",  2),
    ("v2_adaptive",    "v2",  2),
    ("v3_ledger",      "v3",  3),
    ("v4_lookahead",   "v4",  3),
    ("v5_opponent",    "v5",  3),
    ("v6_surgical",    "v6",  3),
    ("v7_prediction",  "v7",  3),
]


def load(p, label):
    spec = importlib.util.spec_from_file_location(label, Path(p) / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[label] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def placements(rewards):
    order = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)
    pts = [0] * len(rewards)
    for rank, seat in enumerate(order):
        pts[seat] = 3 - rank
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand_dir")
    ap.add_argument("label")
    ap.add_argument("n_games", nargs="?", type=int, default=40)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cand_path = Path(args.cand_dir)
    if not cand_path.is_absolute():
        cand_path = ROOT / cand_path
    cand = load(cand_path, args.label)

    # Pre-load every pool agent once.
    pool_agents = {}
    expanded = []
    for dir_name, lab, wt in POOL:
        pool_agents[lab] = load(ROOT / "agents" / dir_name, lab)
        expanded.extend([lab] * wt)

    cand_pts = []
    seat_pts = {0: [], 1: [], 2: [], 3: []}
    opp_pts = Counter()      # total placement points by opponent label
    opp_games = Counter()    # how many times each opponent appeared
    head2head_wins = Counter()   # cand pts > opp pts
    head2head_games = Counter()  # cand and opp both in same game

    for g in range(args.n_games):
        cand_seat = rng.randrange(4)
        opp_labels = [rng.choice(expanded) for _ in range(3)]
        agents = [None] * 4
        agents[cand_seat] = cand
        j = 0
        seat_to_opp = {}
        for s in range(4):
            if s == cand_seat:
                continue
            agents[s] = pool_agents[opp_labels[j]]
            seat_to_opp[s] = opp_labels[j]
            j += 1
        env = make("orbit_wars", configuration={"num_players": 4}, debug=False)
        t0 = time.time()
        env.run(agents)
        elapsed = time.time() - t0
        rewards = [env.steps[-1][i]["reward"] for i in range(4)]
        pts = placements(rewards)
        cand_pts.append(pts[cand_seat])
        seat_pts[cand_seat].append(pts[cand_seat])
        for s, lab in seat_to_opp.items():
            opp_pts[lab] += pts[s]
            opp_games[lab] += 1
            head2head_games[lab] += 1
            if pts[cand_seat] > pts[s]:
                head2head_wins[lab] += 1
        print(f"  game={g+1}/{args.n_games} cand_seat={cand_seat} "
              f"opps={opp_labels} pts={pts} cand_pts={pts[cand_seat]} ({elapsed:.0f}s)",
              flush=True)

    n = len(cand_pts)
    avg = sum(cand_pts) / n
    # Reference: in a fully balanced 4p game, expected pts = 1.5 (avg of 3+2+1+0).
    # If candidate beats the random mix on average, avg > 1.5.
    print()
    print(f"===== MIX BATTERY RESULTS ({n} games) =====", flush=True)
    print(f"{args.label} avg placement pts: {avg:.3f}  (1.5 = neutral)", flush=True)
    print(f"  delta vs neutral: {avg - 1.5:+.3f}", flush=True)
    print()
    print("Per-seat:", flush=True)
    for s in range(4):
        v = seat_pts[s]
        a = sum(v) / len(v) if v else 0
        print(f"  seat {s}: n={len(v):2d} avg={a:.2f}", flush=True)
    print()
    print("Per-opponent (cand head-to-head):", flush=True)
    for lab in sorted(opp_games):
        n_o = opp_games[lab]
        avg_o = opp_pts[lab] / n_o
        h2h_w = head2head_wins[lab]
        h2h_n = head2head_games[lab]
        h2h_rate = h2h_w / h2h_n if h2h_n else 0
        print(f"  vs {lab}: appeared in {n_o:3d} games  opp_avg_pts={avg_o:.2f}  "
              f"cand_beats_opp_rate={h2h_rate:.2%}", flush=True)


if __name__ == "__main__":
    main()
