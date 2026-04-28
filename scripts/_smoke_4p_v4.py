import importlib.util
import sys
import time
from pathlib import Path

from kaggle_environments import make

ROOT = Path(r"e:\Kaggle\Orbit Wars")


def load(p):
    name = f"agent_{Path(p).name}"
    spec = importlib.util.spec_from_file_location(name, Path(p) / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


v4 = load(ROOT / "agents" / "v4_lookahead")
v0 = load(ROOT / "agents" / "v0_random")

env = make("orbit_wars", configuration={"teamsEnabled": False}, debug=True)
t0 = time.time()
env.run([v4, v0, v0, v0])
dur = time.time() - t0
scores = [s["reward"] for s in env.steps[-1]]
print("done", round(dur, 1), "s scores=", scores)
