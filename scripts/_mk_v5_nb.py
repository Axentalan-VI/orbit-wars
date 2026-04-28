import json
from pathlib import Path

src = Path(r"e:\Kaggle\Orbit Wars\agents\v5_opponent\main.py").read_text(encoding="utf-8")
nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%%writefile submission.py\n"] + src.splitlines(keepends=True),
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(r"e:\Kaggle\Orbit Wars\agents\v5_opponent\submission.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
