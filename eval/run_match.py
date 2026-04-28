"""Run a few matches between two agents using kaggle_environments.

Usage:
    python eval/run_match.py --a agents/v1_heuristic --b agents/v0_random --n 5
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from statistics import mean


def load_agent(path: str):
    p = pathlib.Path(path)
    main = p / "main.py" if p.is_dir() else p
    spec = importlib.util.spec_from_file_location(f"agent_{p.stem}", main)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.agent


def run(a_path: str, b_path: str, n: int, mode: str, seed: int) -> dict:
    from kaggle_environments import make

    a = load_agent(a_path)
    b = load_agent(b_path)
    players = 4 if mode == "4p" else 2
    agents = [a, b] + ([b, a] if players == 4 else [])

    wins = [0] * players
    scores: list[list[float]] = [[] for _ in range(players)]
    for i in range(n):
        env = make("orbit_wars", debug=False)
        env.reset(players)
        env.run(agents)
        rewards = [s["reward"] or 0 for s in env.state]
        scores_i = list(rewards)
        for idx, s in enumerate(scores_i):
            scores[idx].append(float(s))
        winner = max(range(players), key=lambda i: scores_i[i])
        # If tie, no winner credit.
        if scores_i.count(scores_i[winner]) == 1:
            wins[winner] += 1
        print(f"  game {i + 1}/{n}: scores={scores_i} winner={winner}")

    return {
        "mode": mode,
        "n": n,
        "wins": wins,
        "mean_scores": [mean(s) for s in scores],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="path to agent A dir or main.py")
    ap.add_argument("--b", required=True, help="path to agent B dir or main.py")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--mode", choices=("2p", "4p"), default="2p")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"A = {args.a}")
    print(f"B = {args.b}")
    res = run(args.a, args.b, args.n, args.mode, args.seed)
    print()
    print("Result:", res)
    total = sum(res["wins"]) or 1
    a_wr = res["wins"][0] / total
    print(f"A win-rate (decisive games): {a_wr:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
