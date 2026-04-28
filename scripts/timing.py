"""Measure max per-turn compute time for each agent in a full game."""
import importlib.util
import sys
import time
from pathlib import Path

from kaggle_environments import make

ROOT = Path(r"e:\Kaggle\Orbit Wars")


def wrap(mod_path, label):
    spec = importlib.util.spec_from_file_location(label, Path(mod_path) / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[label] = mod
    spec.loader.exec_module(mod)
    real = mod.agent
    times = []

    def timed(obs, cfg=None):
        t0 = time.time()
        r = real(obs, cfg)
        times.append(time.time() - t0)
        return r

    timed._times = times
    timed._label = label
    return timed


for label, path in [
    ("v3", ROOT / "agents" / "v3_ledger"),
    ("v4", ROOT / "agents" / "v4_lookahead"),
    ("v5", ROOT / "agents" / "v5_opponent"),
]:
    a = wrap(path, f"a_{label}")
    b = wrap(path, f"b_{label}")
    env = make("orbit_wars", debug=False)
    env.run([a, b])
    import statistics as S
    print(f"{label}: turns={len(a._times)}  "
          f"mean={S.mean(a._times)*1000:.0f}ms  "
          f"p95={sorted(a._times)[int(0.95*len(a._times))]*1000:.0f}ms  "
          f"max={max(a._times)*1000:.0f}ms")
