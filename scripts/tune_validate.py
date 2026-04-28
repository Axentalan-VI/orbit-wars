"""Validate best tuned profile vs default v4 with more games."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

res = json.loads((ROOT / "scripts" / "tune_result.json").read_text())
best_prof = res["best_profile"]
print("Best profile:", json.dumps(best_prof, indent=2))

v4_path = ROOT / "agents" / "v4_lookahead" / "main.py"

def load(label):
    spec = importlib.util.spec_from_file_location(label, v4_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

tuned = load("v4_tuned_val")
tuned.PROFILES_OVERRIDE = [best_prof]
default = load("v4_default_val")

from kaggle_environments import make

N = 10
wins = 0
detail = []
for i in range(N):
    env = make("orbit_wars", configuration={"teamsEnabled": False}, debug=False)
    if i % 2 == 0:
        env.run([tuned.agent, default.agent])
        r = env.steps[-1][0]["reward"]
    else:
        env.run([default.agent, tuned.agent])
        r = env.steps[-1][1]["reward"]
    detail.append(r)
    if r > 0:
        wins += 1
    print(f"  game {i+1}: {'tuned' if i%2==0 else 'default'} first  reward_tuned={r}")

print(f"\nTuned win rate: {wins}/{N} = {wins/N:.2%}")
print(f"Detail: {detail}")
