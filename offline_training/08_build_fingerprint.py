"""
Build per-(subject, benchmark) pass-rate matrix from responses.parquet.

This produces `subject_benchmark_fingerprint.json` — a lookup of empirical
pass rates that the runtime model uses as a Bayesian mixture: at test time,
the K=5 labeled examples identify which training benchmarks the test
benchmark resembles (by likelihood), and predictions for each subject are
a posterior-weighted average over those training benchmarks.

This is NOT a trained model — just a groupby aggregation. Runs in seconds
on local CPU. No Modal needed.

Usage:
    python offline_training/08_build_fingerprint.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Beta-Binomial shrinkage strength (pseudo-obs from benchmark prior). Stops
# single-observation (subject, benchmark) cells from being overconfident.
K_SHRINK = 5


def main() -> None:
    src = DATA / "responses.parquet"
    print(f"Loading {src} ...", flush=True)
    df = pd.read_parquet(src)
    print(f"Responses: {len(df):,}", flush=True)

    # Map subject_id -> display name (first line of subject_content after "Name:")
    subj_meta = df.drop_duplicates("subject_id")[["subject_id", "subject_content"]].copy()
    subj_meta["name"] = subj_meta["subject_content"].fillna("").apply(
        lambda c: c.split("\n")[0].replace("Name:", "").strip() if c else ""
    )
    sid_to_name = dict(zip(subj_meta["subject_id"], subj_meta["name"]))

    # Global benchmark pass rate (the shrinkage prior)
    bm_stats = df.groupby("benchmark_id")["label"].agg(["mean", "count"])
    bm_prior_mean = bm_stats["mean"].to_dict()

    # Per-(subject, benchmark) pass rate
    print("Computing per-(subject, benchmark) pass rates ...", flush=True)
    stats = df.groupby(["subject_id", "benchmark_id"])["label"].agg(["mean", "count"]).reset_index()
    print(f"Non-empty cells: {len(stats):,}", flush=True)

    # Beta-Binomial shrinkage toward benchmark prior:
    # smoothed = (n * empirical_mean + K * prior_mean) / (n + K)
    def smooth(row):
        n = row["count"]
        p = row["mean"]
        prior = bm_prior_mean[row["benchmark_id"]]
        return (n * p + K_SHRINK * prior) / (n + K_SHRINK)

    stats["smoothed"] = stats.apply(smooth, axis=1)

    # Build nested dict: name -> {benchmark_id: smoothed_pass_rate}
    fp_by_name: dict[str, dict[str, float]] = {}
    fp_by_content: dict[str, dict[str, float]] = {}

    for _, row in stats.iterrows():
        sid = row["subject_id"]
        name = sid_to_name.get(sid) or sid
        bm = row["benchmark_id"]
        if name not in fp_by_name:
            fp_by_name[name] = {}
        fp_by_name[name][bm] = round(float(row["smoothed"]), 4)

    # Also store by full subject_content for exact-match lookup
    for _, row in subj_meta.iterrows():
        sid = row["subject_id"]
        content = (row["subject_content"] or "").strip()
        name = sid_to_name.get(sid) or sid
        if content and name in fp_by_name:
            fp_by_content[content] = fp_by_name[name]

    benchmarks = sorted(df["benchmark_id"].unique().tolist())

    out = {
        "benchmarks": benchmarks,
        "by_name": fp_by_name,
        "by_content": fp_by_content,
        "bm_prior_mean": {k: round(float(v), 4) for k, v in bm_prior_mean.items()},
        "global_mean": round(float(df["label"].mean()), 4),
        "k_shrink": K_SHRINK,
    }

    out_path = DATA / "subject_benchmark_fingerprint.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path.name} ({size_kb:.0f} KB)", flush=True)
    print(f"  subjects: {len(fp_by_name)}", flush=True)
    print(f"  benchmarks: {len(benchmarks)}", flush=True)
    print(f"  avg cells per subject: {sum(len(v) for v in fp_by_name.values()) / len(fp_by_name):.1f}",
          flush=True)


if __name__ == "__main__":
    main()
