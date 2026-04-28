"""Render v12-vs-v7 games as HTML replays for visual inspection."""
import importlib.util, sys
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)
from kaggle_environments import make

ROOT = Path(r"e:\Kaggle\Orbit Wars")
OUT = ROOT / "replays"
OUT.mkdir(exist_ok=True)


def load(p, label):
    spec = importlib.util.spec_from_file_location(label, Path(p) / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[label] = mod
    spec.loader.exec_module(mod)
    return mod.agent


v12 = load(ROOT / "agents" / "v12_evacuation", "v12")
v7 = load(ROOT / "agents" / "v7_prediction", "v7")

# Render 4 games: v12 in each seat vs 3x v7
for v12_seat in range(4):
    agents = [v7, v7, v7, v7]
    agents[v12_seat] = v12
    env = make("orbit_wars", configuration={"num_players": 4}, debug=False)
    env.run(agents)
    rewards = [env.steps[-1][i]["reward"] for i in range(4)]
    html = env.render(mode="html")
    out = OUT / f"v12_seat{v12_seat}_vs_v7.html"
    out.write_text(html, encoding="utf-8")
    print(f"seat={v12_seat} rewards={rewards} -> {out}", flush=True)

# Also: 4x v12 and 4x v7 mirror matches
for label, agents in [("all_v12", [v12]*4), ("all_v7", [v7]*4)]:
    env = make("orbit_wars", configuration={"num_players": 4}, debug=False)
    env.run(agents)
    rewards = [env.steps[-1][i]["reward"] for i in range(4)]
    html = env.render(mode="html")
    out = OUT / f"{label}.html"
    out.write_text(html, encoding="utf-8")
    print(f"{label} rewards={rewards} -> {out}", flush=True)

print(f"\nOpen any HTML in: {OUT}")
