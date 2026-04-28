"""Package an agent directory into a submittable archive.

Usage:
    python scripts/package_agent.py agents/v1_heuristic [--no-validate]

Produces ``submissions/<name>-<timestamp>.tar.gz`` containing ``main.py`` and
any additional files from the agent directory. A local validation episode is
played (self-play, 2p) unless ``--no-validate`` is passed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import shutil
import sys
import tarfile
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SUBDIR = ROOT / "submissions"


def validate(agent_dir: pathlib.Path) -> None:
    sys.path.insert(0, str(agent_dir))
    try:
        from kaggle_environments import make  # type: ignore
    except ImportError:
        print("kaggle_environments not installed - skipping validation.")
        return
    import importlib.util

    main = agent_dir / "main.py"
    spec = importlib.util.spec_from_file_location("agent_validate", main)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    for n in (2, 4):
        env = make("orbit_wars", debug=True)
        env.reset(n)
        env.run([mod.agent] * n)
        statuses = [s["status"] for s in env.state]
        if any(st == "ERROR" for st in statuses):
            raise RuntimeError(
                f"Validation ({n}p) failed: statuses={statuses}"
            )
        print(f"  {n}p validation OK, final scores: "
              f"{[s['reward'] for s in env.state]}")


def package(agent_dir: pathlib.Path, do_validate: bool) -> pathlib.Path:
    if not (agent_dir / "main.py").exists():
        raise FileNotFoundError(f"{agent_dir}/main.py not found")

    if do_validate:
        print(f"Validating {agent_dir}...")
        validate(agent_dir)

    SUBDIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = agent_dir.name
    out = SUBDIR / f"{name}-{stamp}.tar.gz"

    with tempfile.TemporaryDirectory() as td:
        staging = pathlib.Path(td) / "agent"
        shutil.copytree(agent_dir, staging)
        with tarfile.open(out, "w:gz") as tf:
            for f in staging.rglob("*"):
                if f.is_file():
                    tf.add(f, arcname=f.relative_to(staging))
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_dir", type=pathlib.Path)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    agent_dir = args.agent_dir.resolve()
    package(agent_dir, do_validate=not args.no_validate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
