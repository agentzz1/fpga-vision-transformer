"""run_analysis.py — turnkey BR41N.IO Data-Analysis runner.

    python run_analysis.py path/to/dataset.mat [--paradigm mi|p300|auto]

Loads the .mat, runs the right SOTA pipeline(s), prints cross-validated results +
an honest method comparison (the thing judges reward). Auto-detects MI (balanced
2-class) vs P300 (imbalanced) when --paradigm auto.
"""
from __future__ import annotations
import argparse, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load_mat
import mi_pipeline, p300_pipeline, riemann_pipeline


def auto_paradigm(y):
    vals, counts = np.unique(y, return_counts=True)
    imbalance = counts.max() / counts.min()
    return "p300" if (len(vals) == 2 and imbalance >= 2.5) else "mi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat"); ap.add_argument("--paradigm", default="auto",
                    choices=["mi", "p300", "auto"])
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=0.8)
    a = ap.parse_args()
    X, y, fs = load_mat(a.mat, fs=a.fs, tmin=a.tmin, tmax=a.tmax)
    print(f"Loaded {X.shape} trials, classes={sorted(set(y.tolist()))}, fs={fs}")
    par = auto_paradigm(y) if a.paradigm == "auto" else a.paradigm
    print(f"Paradigm: {par}\n--- results (5-fold CV) ---")
    if par == "p300":
        print(" ", p300_pipeline.evaluate(X, y, fs=fs))
    else:
        print(" ", mi_pipeline.evaluate(X, y, fs=fs))
        print(" ", riemann_pipeline.evaluate(X, y, fs=fs))
    print("\nReport these CV numbers + an ablation vs a baseline (e.g. raw-LDA) to judges.")


if __name__ == "__main__":
    main()
