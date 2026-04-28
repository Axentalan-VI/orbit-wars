"""Large head-to-head battery. Writes results to stdout + log file."""
import importlib.util
import sys
import time
from pathlib import Path

from kaggle_environments import make

ROOT = Path(r"e:\Kaggle\Orbit Wars")


def load(p, label):
    spec = importlib.util.spec_from_file_location(label, Path(p) / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[label] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def battle(name_a, path_a, name_b, path_b, n):
    fn_a = load(path_a, f"mod_{name_a}")
    fn_b = load(path_b, f"mod_{name_b}")
    wins_a = wins_b = draws = 0
    scores = []
    t0 = time.time()
    for i in range(n):
        env = make("orbit_wars", configuration={"teamsEnabled": False}, debug=False)
        if i % 2 == 0:
            env.run([fn_a, fn_b])
            ra, rb = env.steps[-1][0]["reward"], env.steps[-1][1]["reward"]
        else:
            env.run([fn_b, fn_a])
            rb, ra = env.steps[-1][0]["reward"], env.steps[-1][1]["reward"]
        scores.append((ra, rb))
        if ra > rb:
            wins_a += 1
        elif rb > ra:
            wins_b += 1
        else:
            draws += 1
        dt = time.time() - t0
        print(f"  [{name_a} vs {name_b}] game {i+1}/{n}: a={ra} b={rb}"
              f"  ({wins_a}-{wins_b}-{draws}) t={dt:.0f}s", flush=True)
    print(f"FINAL {name_a} vs {name_b}: {wins_a}-{wins_b}-{draws}  scores={scores}",
          flush=True)
    return wins_a, wins_b, draws


N = 8
battle("v5", ROOT / "agents" / "v5_opponent", "v3", ROOT / "agents" / "v3_ledger", N)
battle("v4", ROOT / "agents" / "v4_lookahead", "v3", ROOT / "agents" / "v3_ledger", N)
battle("v5", ROOT / "agents" / "v5_opponent", "v4", ROOT / "agents" / "v4_lookahead", N)
