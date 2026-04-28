"""v12 4-player FFA self-play (Kaggle proxy)."""
import importlib.util, sys, time
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)
from kaggle_environments import make

ROOT = Path(r"e:\Kaggle\Orbit Wars")


def load(p, label):
    spec = importlib.util.spec_from_file_location(label, Path(p) / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[label] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def placements(rewards):
    order = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)
    pts = [0] * len(rewards)
    for rank, seat in enumerate(order):
        pts[seat] = 3 - rank
    return pts


def run_battery(target_path, target_name, baseline_path, baseline_name, n_per_seat=5):
    target = load(target_path, target_name)
    base = load(baseline_path, baseline_name)
    seat_points_target = {0: [], 1: [], 2: [], 3: []}
    seat_points_base = {0: [], 1: [], 2: [], 3: []}
    total = 0
    for target_seat in range(4):
        for g in range(n_per_seat):
            agents = [base, base, base, base]
            agents[target_seat] = target
            env = make("orbit_wars", configuration={"num_players": 4}, debug=False)
            t0 = time.time()
            env.run(agents)
            elapsed = time.time() - t0
            rewards = [env.steps[-1][i]["reward"] for i in range(4)]
            pts = placements(rewards)
            seat_points_target[target_seat].append(pts[target_seat])
            for s in range(4):
                if s != target_seat:
                    seat_points_base[s].append(pts[s])
            total += 1
            print(f"  seat={target_seat} game={g+1}/{n_per_seat} rewards={rewards} "
                  f"pts={pts} target_pts={pts[target_seat]} ({elapsed:.0f}s)",
                  flush=True)
    tgt_flat = [p for seat in seat_points_target.values() for p in seat]
    base_flat = [p for seat in seat_points_base.values() for p in seat]
    tgt_avg = sum(tgt_flat) / len(tgt_flat) if tgt_flat else 0
    base_avg = sum(base_flat) / len(base_flat) if base_flat else 0
    print()
    print(f"===== RESULTS ({total} games) =====", flush=True)
    print(f"{target_name} avg placement pts: {tgt_avg:.3f}  (samples={len(tgt_flat)})", flush=True)
    print(f"{baseline_name} avg placement pts: {base_avg:.3f}  (samples={len(base_flat)})", flush=True)
    print(f"delta: {tgt_avg - base_avg:+.3f}", flush=True)
    for s in range(4):
        pts = seat_points_target[s]
        avg = sum(pts) / len(pts) if pts else 0
        print(f"  {target_name} at seat {s}: {pts} avg={avg:.2f}", flush=True)


run_battery(
    ROOT / "agents" / "v12_evacuation", "v12",
    ROOT / "agents" / "v7_prediction", "v7",
    n_per_seat=5,
)
