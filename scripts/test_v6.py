"""v6 vs v3 head-to-head, 10 games, alternating sides."""
import importlib.util, sys, time
from pathlib import Path
from kaggle_environments import make

ROOT = Path(r"e:\Kaggle\Orbit Wars")

def load(p, label):
    spec = importlib.util.spec_from_file_location(label, Path(p) / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[label] = mod
    spec.loader.exec_module(mod)
    return mod.agent

a = load(ROOT / "agents" / "v6_surgical", "v6")
b = load(ROOT / "agents" / "v3_ledger", "v3")

N = 10
wa = wb = dr = 0
for i in range(N):
    env = make("orbit_wars", configuration={"teamsEnabled": False}, debug=False)
    if i % 2 == 0:
        env.run([a, b])
        ra, rb = env.steps[-1][0]["reward"], env.steps[-1][1]["reward"]
    else:
        env.run([b, a])
        rb, ra = env.steps[-1][0]["reward"], env.steps[-1][1]["reward"]
    if ra > rb: wa += 1
    elif rb > ra: wb += 1
    else: dr += 1
    print(f"game {i+1}/{N}: v6={ra} v3={rb}  ({wa}-{wb}-{dr}) t={time.time():.0f}", flush=True)
print(f"\nFINAL v6 vs v3: {wa}-{wb}-{dr}", flush=True)
