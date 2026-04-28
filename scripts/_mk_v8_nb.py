"""Build submission.ipynb for v8_targeting."""
import json
from pathlib import Path

agent_dir = Path(r"e:\Kaggle\Orbit Wars\agents\v8_targeting")
src = (agent_dir / "main.py").read_text(encoding="utf-8")

cell_source = ["%%writefile submission.py\n"] + src.splitlines(keepends=True)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell_source,
        }
    ],
}

out = agent_dir / "submission.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written {out}  ({len(src)} chars)")
