"""
Master runner — executes all offline training steps in order.

Usage:
  python offline_training/run_all.py              # run all steps
  python offline_training/run_all.py --from 2     # resume from step N
  python offline_training/run_all.py --only 3     # run just step N

Steps:
  1  compute_embeddings   (slow ~50min CPU, run once)
  2  train_irt_bridge     (fast ~2min)
  3  train_ncf            (medium ~15min)
  4  fit_ensemble         (fast ~5min)
  5  fit_acquisition      (fast ~2min)
  6  package_submission   (copies artifacts to my_submission/, makes ZIP)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "offline_training"
PYTHON = sys.executable

STEPS = [
    (1, "01_compute_embeddings.py",  "Compute sentence-transformer embeddings"),
    (2, "02_train_irt_bridge.py",    "Train IRT content bridge"),
    (3, "03_train_ncf.py",           "Train NCF binary classifier"),
    (4, "04_fit_ensemble.py",        "Fit ensemble weights + Platt calibration"),
    (5, "05_fit_acquisition.py",     "Fit k-means acquisition centroids"),
    (6, "06_package_submission.py",  "Package submission ZIP"),
]


def run_step(n: int, script: str, label: str) -> bool:
    print(f"\n{'='*60}", flush=True)
    print(f"STEP {n}: {label}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    result = subprocess.run([PYTHON, str(SCRIPTS / script)], cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n✓ Step {n} completed in {elapsed/60:.1f} min", flush=True)
        return True
    else:
        print(f"\n✗ Step {n} FAILED (exit code {result.returncode})", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step number (default: 1)")
    parser.add_argument("--only", dest="only_step", type=int, default=None,
                        help="Run only this step number")
    args = parser.parse_args()

    print(f"\nPredictive Evaluation Challenge — Offline Training", flush=True)
    print(f"Python: {PYTHON}", flush=True)

    for n, script, label in STEPS:
        if args.only_step and n != args.only_step:
            continue
        if n < args.from_step:
            print(f"\nSkipping step {n} ({label})")
            continue
        ok = run_step(n, script, label)
        if not ok:
            print(f"\nAborting at step {n}. Fix the error and re-run with --from {n}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("All steps complete. Submission is at: my_submission/submission.zip")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
