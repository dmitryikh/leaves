# /// script
# dependencies = []
# ///
"""
Coordinator: runs all per-case training scripts.
Run with: uv run train.py
"""

import subprocess
import pathlib

here = pathlib.Path(__file__).parent

for s in [
    "test_binary_logistic.py",
    "test_regression.py",
    "test_multiclass.py",
    "test_poisson.py",
]:
    print(f"\n=== Running {s} ===")
    subprocess.run(["uv", "run", str(here / s)], check=True)
