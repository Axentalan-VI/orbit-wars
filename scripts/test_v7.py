"""v7 head-to-head tests."""
import importlib.util, sys
from pathlib import Path
from kaggle_environments import make

ROOT = Path(r"e:\Kaggle\Orbit Wars")

def load(p, label):
    spec = importlib.util.spec_from_file_location(label, Path(p) / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[label] = mod
    spec.loader.exec_module(mod)
    return mod.agent

def battle(a_name, a_path, b_name, b_path, n):
    fa = load(a_path, f"a_{a_name}")
    fb = load(b_path, f"b_{b_name}")
    wa = wb = dr = 0
    for i in range(n):
        env = make("orbit_wars", configuration={"teamsEnabled": False}, debug=False)
        if i % 2 == 0:
            env.run([fa, fb])
            ra, rb = env.steps[-1][0]["reward"], env.steps[-1][1]["reward"]
        else:
            env.run([fb, fa])
            rb, ra = env.steps[-1][0]["reward"], env.steps[-1][1]["reward"]
        if ra > rb: wa += 1
        elif rb > ra: wb += 1
        else: dr += 1
        print(f"  [{a_name} vs {b_name}] {i+1}/{n}: a={ra} b={rb}  ({wa}-{wb}-{dr})", flush=True)
    print(f"FINAL {a_name} vs {b_name}: {wa}-{wb}-{dr}", flush=True)

battle("v7", ROOT/"agents"/"v7_prediction", "v3", ROOT/"agents"/"v3_ledger", 10)
battle("v7", ROOT/"agents"/"v7_prediction", "v6", ROOT/"agents"/"v6_surgical", 10)
