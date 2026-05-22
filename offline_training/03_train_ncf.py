"""
Train Neural Collaborative Filtering (NCF) model.

Architecture: subject_emb (768) || item_emb (768) -> MLP -> sigmoid

Uses pre-computed embeddings from 01_compute_embeddings.py.

Outputs (to ../data/):
  ncf_head.pt       - trained MLP state dict
  ncf_meta.json     - input dim info
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings"

# ---------------------------------------------------------------------------
# 1. Load embeddings + index maps
# ---------------------------------------------------------------------------
print("Loading embeddings...", flush=True)
with open(EMB_DIR / "item_ids.json") as f:
    item_ids_list = json.load(f)
with open(EMB_DIR / "subject_ids.json") as f:
    subject_ids_list = json.load(f)

item_id_to_idx = {iid: i for i, iid in enumerate(item_ids_list)}
subject_id_to_idx = {sid: i for i, sid in enumerate(subject_ids_list)}

item_embs = np.load(EMB_DIR / "item_embs.npy")      # (N_items, 768)
subject_embs = np.load(EMB_DIR / "subject_embs.npy") # (N_subjects, 768)
print(f"Item embs: {item_embs.shape}, Subject embs: {subject_embs.shape}")

# ---------------------------------------------------------------------------
# 2. Load responses and build training arrays
# ---------------------------------------------------------------------------
print("Loading responses...", flush=True)
resp_df = pd.read_parquet(DATA_DIR / "responses.parquet",
                          columns=["subject_id", "item_id", "label"])
print(f"Responses: {len(resp_df):,}")

# Map to embedding indices
resp_df["item_idx"] = resp_df["item_id"].map(item_id_to_idx)
resp_df["subject_idx"] = resp_df["subject_id"].map(subject_id_to_idx)
resp_df = resp_df.dropna(subset=["item_idx", "subject_idx"])
resp_df["item_idx"] = resp_df["item_idx"].astype(int)
resp_df["subject_idx"] = resp_df["subject_idx"].astype(int)
print(f"After index join: {len(resp_df):,}")

# ---------------------------------------------------------------------------
# 3. Cold-start validation split (hold out 20% of ITEMS, not rows)
# ---------------------------------------------------------------------------
unique_items = resp_df["item_id"].unique()
np.random.seed(42)
val_items = set(np.random.choice(unique_items, size=int(0.2 * len(unique_items)), replace=False))
train_mask = ~resp_df["item_id"].isin(val_items)

train_df = resp_df[train_mask].reset_index(drop=True)
val_df = resp_df[~train_mask].reset_index(drop=True)
print(f"Train: {len(train_df):,} rows ({train_df['item_id'].nunique()} items)")
print(f"Val:   {len(val_df):,} rows ({val_df['item_id'].nunique()} items) [item cold-start]")

# ---------------------------------------------------------------------------
# 4. Build feature matrices
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

item_embs_t = torch.from_numpy(item_embs).to(device)     # pre-loaded to GPU
subject_embs_t = torch.from_numpy(subject_embs).to(device)

EMB_DIM = item_embs.shape[1]  # 384 for MiniLM

def make_XY(df, max_rows=None):
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    s_idx = torch.tensor(df["subject_idx"].values, dtype=torch.long)
    i_idx = torch.tensor(df["item_idx"].values, dtype=torch.long)
    labels = torch.tensor(df["label"].values, dtype=torch.float32)
    # Lookup embeddings (on GPU)
    s_emb = subject_embs_t[s_idx.to(device)]
    i_emb = item_embs_t[i_idx.to(device)]
    X = torch.cat([s_emb, i_emb], dim=1).cpu()
    return X, labels

print("Building train features...", flush=True)
X_train, y_train = make_XY(train_df, max_rows=2_000_000)
print(f"  X_train: {X_train.shape}")

print("Building val features...", flush=True)
X_val, y_val = make_XY(val_df, max_rows=300_000)
print(f"  X_val: {X_val.shape}")

# ---------------------------------------------------------------------------
# 5. Define NCF model
# ---------------------------------------------------------------------------
class NCFHead(nn.Module):
    def __init__(self, in_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = NCFHead(in_dim=EMB_DIM * 2).to(device)
print(f"\n=== [03] NCF Training ===")
print(f"Device: {device}  |  Emb dim: {EMB_DIM}  |  NCF input: {EMB_DIM*2}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Epochs: {EPOCHS}  |  Batch: 4096  |  Progress printed every epoch", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

# DataLoader
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=0)

# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------
EPOCHS = 15
best_val_loss = float("inf")
best_state = None

X_val_d = X_val.to(device)
y_val_d = y_val.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits = model(X_b)
        loss = criterion(logits, y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_d)
        val_loss = criterion(val_logits, y_val_d).item()
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

    val_logloss = log_loss(y_val.numpy(), val_probs)
    val_auc = roc_auc_score(y_val.numpy(), val_probs)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch+1:2d}: train_bce={total_loss/n_batches:.4f}  "
          f"val_bce={val_loss:.4f}  val_logloss={val_logloss:.4f}  val_auc={val_auc:.4f}")

# Load best model
if best_state:
    model.load_state_dict(best_state)

# Final metrics
model.eval()
with torch.no_grad():
    val_probs_best = torch.sigmoid(model(X_val_d)).cpu().numpy()

final_logloss = log_loss(y_val.numpy(), val_probs_best)
final_auc = roc_auc_score(y_val.numpy(), val_probs_best)
print(f"\nBest model: val_logloss={final_logloss:.4f}  val_auc={final_auc:.4f}")
print(f"Baseline (constant 0.664): {log_loss(y_val.numpy(), np.full_like(val_probs_best, 0.664)):.4f}")

# Save
torch.save(model.state_dict(), DATA_DIR / "ncf_head.pt")
meta = {"in_dim": EMB_DIM * 2, "emb_dim": EMB_DIM, "val_logloss": final_logloss, "val_auc": final_auc}
with open(DATA_DIR / "ncf_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved ncf_head.pt and ncf_meta.json")
