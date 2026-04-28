"""CEM tuner for v4 scoring profiles.

Optimises a flat weight vector by sampling around a Gaussian mean, running
matches vs v3_ledger, scoring, keeping the elite, refitting mean/std.

Usage:
    python scripts/tune.py --pop 12 --elite 3 --gens 3 --games-per 2 --workers 4

Weight vector (7 dims, per profile; we tune ONE profile used as all 4 slots):
    comet_mul        : [1.0, 2.5]
    neutral_mul      : [0.7, 1.3]
    aggression_mul   : [1.0, 1.7]
    safety_margin    : [1, 3]  (rounded to int)
    weakest_opp_mul  : [1.0, 1.4]
    other_opp_mul    : [0.7, 1.0]
    compact_pen      : [0.0, 0.08]

Benchmark opponent: agents/v3_ledger.
Score per candidate = win_rate - 0.5 (so >0 means beats baseline).
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PARAM_NAMES = [
    "comet_mul", "neutral_mul", "aggression_mul", "safety_margin",
    "weakest_opp_mul", "other_opp_mul", "compact_pen",
]
PARAM_BOUNDS = {
    "comet_mul": (1.0, 2.5),
    "neutral_mul": (0.7, 1.3),
    "aggression_mul": (1.0, 1.7),
    "safety_margin": (1.0, 3.0),
    "weakest_opp_mul": (1.0, 1.4),
    "other_opp_mul": (0.7, 1.0),
    "compact_pen": (0.0, 0.08),
}
INITIAL_MEAN = {
    "comet_mul": 1.6, "neutral_mul": 1.0, "aggression_mul": 1.3,
    "safety_margin": 1.0, "weakest_opp_mul": 1.15, "other_opp_mul": 0.9,
    "compact_pen": 0.025,
}
INITIAL_STD = {
    "comet_mul": 0.4, "neutral_mul": 0.15, "aggression_mul": 0.2,
    "safety_margin": 0.6, "weakest_opp_mul": 0.1, "other_opp_mul": 0.08,
    "compact_pen": 0.02,
}


def clip(name, v):
    lo, hi = PARAM_BOUNDS[name]
    return max(lo, min(hi, v))


def sample(mean, std, rng):
    return {n: clip(n, rng.gauss(mean[n], std[n])) for n in PARAM_NAMES}


def to_profile(vec):
    p = dict(vec)
    p["safety_margin"] = int(round(p["safety_margin"]))
    return p


def _evaluate_trial(args):
    """Worker: run `games` matches of tuned v4 vs v4-default, return win_rate."""
    vec, games, seed_base = args
    import importlib.util

    v4_path = ROOT / "agents" / "v4_lookahead" / "main.py"

    # Candidate: tuned v4 (single profile)
    spec_a = importlib.util.spec_from_file_location("v4_tuned", v4_path)
    v4a = importlib.util.module_from_spec(spec_a)
    spec_a.loader.exec_module(v4a)
    prof = to_profile(vec)
    v4a.PROFILES_OVERRIDE = [prof]

    # Baseline: v4 default profiles (fresh module so state is isolated)
    spec_b = importlib.util.spec_from_file_location("v4_default", v4_path)
    v4b = importlib.util.module_from_spec(spec_b)
    spec_b.loader.exec_module(v4b)
    # PROFILES_OVERRIDE stays None -> uses DEFAULT_PROFILES

    from kaggle_environments import make
    wins = 0
    scores_a = []
    for i in range(games):
        env = make("orbit_wars", configuration={"teamsEnabled": False}, debug=False)
        if i % 2 == 0:
            env.run([v4a.agent, v4b.agent])
            a_reward = env.steps[-1][0]["reward"]
        else:
            env.run([v4b.agent, v4a.agent])
            a_reward = env.steps[-1][1]["reward"]
        scores_a.append(a_reward)
        if a_reward > 0:
            wins += 1
    return wins / games, scores_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--games-per", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "scripts" / "tune_result.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mean = dict(INITIAL_MEAN)
    std = dict(INITIAL_STD)

    history = []
    overall_best = (-1.0, None)
    t0 = time.time()

    for g in range(args.gens):
        pop = [sample(mean, std, rng) for _ in range(args.pop)]
        # also include the current mean as a reference candidate
        pop[0] = dict(mean)

        print(f"\n=== Generation {g+1}/{args.gens} (pop={len(pop)}, "
              f"games/cand={args.games_per}, workers={args.workers}) ===")
        for i, v in enumerate(pop):
            print(f"  cand{i}: " + " ".join(f"{n}={v[n]:.3f}" for n in PARAM_NAMES))

        tasks = [(pop[i], args.games_per, args.seed + g * 1000 + i) for i in range(len(pop))]
        t1 = time.time()
        with mp.Pool(args.workers) as pool:
            results = pool.map(_evaluate_trial, tasks)
        print(f"  gen took {time.time()-t1:.1f}s")

        scored = []
        for i, (wr, sc) in enumerate(results):
            scored.append((wr, i, pop[i], sc))
            print(f"  cand{i} win_rate={wr:.2f} scores={sc}")

        scored.sort(reverse=True, key=lambda x: x[0])
        elite = scored[:args.elite]
        print(f"  elite (top {args.elite}):")
        for wr, i, v, sc in elite:
            print(f"    wr={wr:.2f} cand{i}")

        if elite[0][0] > overall_best[0]:
            overall_best = (elite[0][0], elite[0][2])

        # refit mean/std from elite
        for n in PARAM_NAMES:
            vals = [e[2][n] for e in elite]
            m = sum(vals) / len(vals)
            var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
            # shrink std each gen but keep a floor
            new_std = max(var ** 0.5, INITIAL_STD[n] * 0.3)
            mean[n] = m
            std[n] = new_std

        history.append({
            "gen": g + 1,
            "mean": dict(mean),
            "std": dict(std),
            "best_wr": elite[0][0],
        })

    out = {
        "best_win_rate": overall_best[0],
        "best_vec": overall_best[1],
        "best_profile": to_profile(overall_best[1]) if overall_best[1] else None,
        "final_mean": mean,
        "final_std": std,
        "history": history,
        "total_seconds": time.time() - t0,
        "config": vars(args),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nBest win_rate={overall_best[0]:.3f}")
    print(f"Best profile: {out['best_profile']}")
    print(f"Wrote {args.out}  elapsed={out['total_seconds']:.1f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
