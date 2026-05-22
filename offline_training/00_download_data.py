"""
Download the HuggingFace training dataset and build the subject theta lookup.

Outputs (written to ../data/):
  responses.parquet   - cleaned response table (subject_id, item_id, benchmark_id,
                        test_condition, label, item_content, subject_content)
  subject_theta.json  - {display_name: theta, ...} from subject_abilities.csv
  item_lookup.parquet - item_id -> item_content + IRT params (where available)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}


# ---------------------------------------------------------------------------
# 1. List response files (exclude registry + trace files)
# ---------------------------------------------------------------------------
print("Listing repo files...", flush=True)
api = HfApi()
repo_files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
response_files = sorted(
    name
    for name in repo_files
    if name.endswith(".parquet")
    and name not in REGISTRY_FILES
    and not name.endswith("_traces.parquet")
)
print(f"Found {len(response_files)} response files:", response_files[:5], "...")

# ---------------------------------------------------------------------------
# 2. Download registry tables
# ---------------------------------------------------------------------------
print("Downloading registry tables...", flush=True)
items_ds = load_dataset(REPO_ID, data_files="items.parquet", split="train")
subjects_ds = load_dataset(REPO_ID, data_files="subjects.parquet", split="train")
benchmarks_ds = load_dataset(REPO_ID, data_files="benchmarks.parquet", split="train")

items_df = items_ds.to_pandas()
subjects_df = subjects_ds.to_pandas()
benchmarks_df = benchmarks_ds.to_pandas()

print(f"Items: {len(items_df)} rows, cols: {list(items_df.columns)}")
print(f"Subjects: {len(subjects_df)} rows, cols: {list(subjects_df.columns)}")
print(f"Benchmarks: {len(benchmarks_df)} rows, cols: {list(benchmarks_df.columns)}")
print("\nSample subjects:\n", subjects_df.head(3).to_string())
print("\nSample items:\n", items_df.head(2).to_string())

# ---------------------------------------------------------------------------
# 3. Download response tables
# ---------------------------------------------------------------------------
print("\nDownloading response tables...", flush=True)
response_features = Features(
    {
        "subject_id": Value("string"),
        "item_id": Value("string"),
        "benchmark_id": Value("string"),
        "trial": Value("int64"),
        "test_condition": Value("string"),
        "response": Value("float64"),
        "correct_answer": Value("string"),
        "trace": Value("string"),
    }
)

responses_ds = load_dataset(
    REPO_ID,
    data_files=response_files,
    features=response_features,
    split="train",
)
responses_df = responses_ds.to_pandas()
print(f"Raw responses: {len(responses_df):,} rows")
print("Response value distribution (sample):")
print(responses_df["response"].value_counts().head(10))

# ---------------------------------------------------------------------------
# 4. Clean: keep binary labels, deduplicate, keep first trial
# ---------------------------------------------------------------------------
# Filter to binary labels
binary_mask = responses_df["response"].isin([0.0, 1.0])
print(f"\nBinary responses: {binary_mask.sum():,} / {len(responses_df):,}")
resp = responses_df[binary_mask].copy()
resp["label"] = resp["response"].astype(int)
resp["test_condition"] = resp["test_condition"].fillna("none")

# Keep first trial per (subject_id, item_id, test_condition)
resp = resp.sort_values("trial").drop_duplicates(
    subset=["subject_id", "item_id", "test_condition"], keep="first"
)
print(f"After dedup: {len(resp):,} rows")

# ---------------------------------------------------------------------------
# 5. Build subject_content strings (matching runtime format)
# ---------------------------------------------------------------------------
def render_subject_content(row) -> str:
    name = row.get("display_name") or row.get("subject_id", "unknown")
    lines = [f"Name: {name}"]
    for key, label in [("provider", "Organization"), ("params", "Parameters"),
                       ("release_date", "Released"), ("family", "Family")]:
        val = row.get(key)
        if val and str(val).strip():
            lines.append(f"{label}: {val}")
    return "\n".join(lines)

subjects_df["subject_content"] = subjects_df.apply(render_subject_content, axis=1)
subjects_map = subjects_df.set_index("subject_id")[["subject_content"]].to_dict()["subject_content"]

# ---------------------------------------------------------------------------
# 6. Join item content + subject content into responses
# ---------------------------------------------------------------------------
# Determine item content column name
item_content_col = "content" if "content" in items_df.columns else items_df.columns[1]
print(f"\nUsing item content column: '{item_content_col}'")
items_map = items_df.set_index("item_id")[item_content_col].to_dict()

resp["item_content"] = resp["item_id"].map(items_map)
resp["subject_content"] = resp["subject_id"].map(subjects_map)

# Drop rows where we couldn't join content
before = len(resp)
resp = resp.dropna(subset=["item_content", "subject_content"])
print(f"After joining content: {len(resp):,} rows (dropped {before - len(resp):,})")

# Keep only needed columns
resp = resp[["subject_id", "item_id", "benchmark_id", "test_condition",
             "label", "item_content", "subject_content"]].reset_index(drop=True)

resp.to_parquet(DATA_DIR / "responses.parquet", index=False)
print(f"\nSaved responses.parquet: {len(resp):,} rows")
print("Benchmark distribution:\n", resp["benchmark_id"].value_counts().head(15).to_string())

# ---------------------------------------------------------------------------
# 7. Build subject theta lookup
# ---------------------------------------------------------------------------
subject_abilities = pd.read_csv(DATA_DIR / "subject_abilities.csv")
print(f"\nSubject abilities: {len(subject_abilities)} rows, cols: {list(subject_abilities.columns)}")

# Join display_name from subjects_df
theta_df = subject_abilities.merge(
    subjects_df[["subject_id", "subject_content"]],
    on="subject_id", how="left"
)

# Parse name from subject_content (first line: "Name: ...")
def extract_name(content: str) -> str:
    if not isinstance(content, str):
        return ""
    first_line = content.split("\n")[0]
    return first_line.replace("Name:", "").strip()

theta_df["display_name"] = theta_df["subject_content"].apply(extract_name)

# Build lookup: display_name -> theta, also subject_id -> theta
theta_lookup = {}
for _, row in theta_df.iterrows():
    theta_lookup[row["subject_id"]] = float(row["theta"])
    if row["display_name"]:
        theta_lookup[row["display_name"]] = float(row["theta"])

# Also store full subject_content -> theta for fuzzy matching
theta_by_content = {}
for _, row in theta_df.iterrows():
    if isinstance(row["subject_content"], str):
        theta_by_content[row["subject_content"]] = float(row["theta"])

combined = {"by_name": theta_lookup, "by_content": theta_by_content,
            "mean_theta": float(theta_df["theta"].mean()),
            "std_theta": float(theta_df["theta"].std())}

with open(DATA_DIR / "subject_theta.json", "w") as f:
    json.dump(combined, f, indent=2)
print(f"Saved subject_theta.json with {len(theta_lookup)} name entries")
print(f"Theta stats: mean={theta_df['theta'].mean():.3f}, std={theta_df['theta'].std():.3f}")
print(f"  min={theta_df['theta'].min():.3f}, max={theta_df['theta'].max():.3f}")

# ---------------------------------------------------------------------------
# 8. Build item lookup with IRT params
# ---------------------------------------------------------------------------
item_params = pd.read_csv(DATA_DIR / "item_params.csv")
print(f"\nItem params: {len(item_params)} rows, cols: {list(item_params.columns)}")

item_lookup = items_df.copy()
item_lookup = item_lookup.rename(columns={item_content_col: "item_content"})
item_lookup = item_lookup.merge(item_params, on="item_id", how="left")
item_lookup.to_parquet(DATA_DIR / "item_lookup.parquet", index=False)
print(f"Saved item_lookup.parquet: {len(item_lookup)} rows")
print(f"Items with IRT params: {item_lookup['difficulty'].notna().sum()}")

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
print(f"  Total training responses: {len(resp):,}")
print(f"  Unique subjects: {resp['subject_id'].nunique()}")
print(f"  Unique items: {resp['item_id'].nunique()}")
print(f"  Unique benchmarks: {resp['benchmark_id'].nunique()}")
print(f"  Overall pass rate: {resp['label'].mean():.3f}")
print(f"  Subjects with theta: {len(subject_abilities)}")
print(f"  Items with IRT params: {item_lookup['difficulty'].notna().sum()}")
