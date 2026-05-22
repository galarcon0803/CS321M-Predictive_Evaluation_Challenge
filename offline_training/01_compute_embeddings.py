"""
Pre-compute sentence-transformer embeddings for all unique items and subjects.
Run once; outputs are cached to disk for subsequent training scripts.

Outputs (written to ../data/embeddings/):
  item_embs.npy      - (N_items, 768) float32, row-aligned with item_ids.json
  item_ids.json      - list of item_id strings (index -> item_id)
  subject_embs.npy   - (N_subjects, 768) float32, row-aligned with subject_ids.json
  subject_ids.json   - list of subject_id strings
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 512
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading encoder: {MODEL_NAME}", flush=True)
encoder = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------------------------
# 1. Item embeddings
# ---------------------------------------------------------------------------
print("Loading item lookup...", flush=True)
items_df = pd.read_parquet(DATA_DIR / "item_lookup.parquet")

# Deduplicate by item_id (keep first)
items_df = items_df.drop_duplicates(subset="item_id").reset_index(drop=True)
print(f"Unique items: {len(items_df)}")

item_ids = items_df["item_id"].tolist()
content_col = "item_content" if "item_content" in items_df.columns else "content"
item_texts = items_df[content_col].fillna("").tolist()

print(f"Encoding {len(item_texts)} item texts in batches of {BATCH_SIZE}...", flush=True)
item_embs = encoder.encode(
    item_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
).astype(np.float32)

np.save(EMB_DIR / "item_embs.npy", item_embs)
with open(EMB_DIR / "item_ids.json", "w") as f:
    json.dump(item_ids, f)
print(f"Saved item_embs.npy: shape={item_embs.shape}")

# ---------------------------------------------------------------------------
# 2. Subject embeddings
# ---------------------------------------------------------------------------
print("\nLoading response data for subjects...", flush=True)
resp_df = pd.read_parquet(DATA_DIR / "responses.parquet",
                          columns=["subject_id", "subject_content"])
subjects = resp_df.drop_duplicates("subject_id").reset_index(drop=True)
print(f"Unique subjects: {len(subjects)}")

subject_ids = subjects["subject_id"].tolist()
subject_texts = subjects["subject_content"].fillna("").tolist()

print(f"Encoding {len(subject_texts)} subject descriptions...", flush=True)
subject_embs = encoder.encode(
    subject_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
).astype(np.float32)

np.save(EMB_DIR / "subject_embs.npy", subject_embs)
with open(EMB_DIR / "subject_ids.json", "w") as f:
    json.dump(subject_ids, f)
print(f"Saved subject_embs.npy: shape={subject_embs.shape}")

print("\nDone. Embedding files written to:", EMB_DIR)
