import importlib.util, time, sys
sys.stdout.reconfigure(line_buffering=True)
from kaggle_environments import make


def load(p):
    s = importlib.util.spec_from_file_location(p, p + "/main.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.agent


v3 = load("agents/v3_ledger")
v0 = load("agents/v0_random")

env = make("orbit_wars", configuration={"num_players": 4}, debug=True)
print("4p FFA smoke test v3 vs 3x v0")
t0 = time.time()
r = env.run([v3, v0, v0, v0])
elapsed = time.time() - t0
scores = [r[-1][i]["reward"] for i in range(4)]
print("done", round(elapsed, 1), "s  scores=", scores)
