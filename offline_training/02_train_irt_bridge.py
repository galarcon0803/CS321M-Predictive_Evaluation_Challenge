"""
Train IRT content bridge: item embedding -> (difficulty, discrimination).

Uses the 3,088 items with known IRT params from item_params.csv.
The bridge lets us predict IRT params for unseen items at test time.

Outputs (to ../data/):
  irt_bridge.pt        - trained MLP state dict
  irt_bridge_meta.json - normalization stats for a and b
  subject_theta_full.json - theta for all subjects (lookup + pass-rate proxy)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings"

# ---------------------------------------------------------------------------
# 1. Load embeddings index
# ---------------------------------------------------------------------------
with open(EMB_DIR / "item_ids.json") as f:
    item_ids_list = json.load(f)
item_id_to_idx = {iid: i for i, iid in enumerate(item_ids_list)}

item_embs = np.load(EMB_DIR / "item_embs.npy")  # (N, 768)
print(f"Item embeddings: {item_embs.shape}")

# ---------------------------------------------------------------------------
# 2. Load IRT params
# ---------------------------------------------------------------------------
item_params = pd.read_csv(DATA_DIR / "item_params.csv")
print(f"IRT params: {len(item_params)} items")

# Join embeddings to IRT params
item_params["emb_idx"] = item_params["item_id"].map(item_id_to_idx)
item_params = item_params.dropna(subset=["emb_idx"])
item_params["emb_idx"] = item_params["emb_idx"].astype(int)
print(f"IRT items with embeddings: {len(item_params)}")

X = item_embs[item_params["emb_idx"].values]  # (N_irt, 768)
y_a = item_params["discrimination"].values.astype(np.float32)
y_b = item_params["difficulty"].values.astype(np.float32)

# Clip extreme values
y_a = np.clip(y_a, 0.1, 6.0)
y_b = np.clip(y_b, -5.0, 5.0)

# Log-transform discrimination (it's always positive, log makes it more symmetric)
y_log_a = np.log(y_a).astype(np.float32)

# Normalize targets
b_mean, b_std = y_b.mean(), y_b.std()
log_a_mean, log_a_std = y_log_a.mean(), y_log_a.std()
y_b_norm = (y_b - b_mean) / (b_std + 1e-8)
y_log_a_norm = (y_log_a - log_a_mean) / (log_a_std + 1e-8)

print(f"Difficulty stats: mean={y_b.mean():.3f} std={y_b.std():.3f}")
print(f"Discrimination stats: mean={y_a.mean():.3f} std={y_a.std():.3f}")

meta = {
    "b_mean": float(b_mean), "b_std": float(b_std),
    "log_a_mean": float(log_a_mean), "log_a_std": float(log_a_std),
}

# ---------------------------------------------------------------------------
# 3. Quick ridge regression baseline (cross-val to check signal)
# ---------------------------------------------------------------------------
print("\nRidge regression cross-val (difficulty)...")
ridge_b = Ridge(alpha=1.0)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse_b = []
for tr, val in kf.split(X):
    ridge_b.fit(X[tr], y_b[tr])
    pred = ridge_b.predict(X[val])
    cv_mse_b.append(mean_squared_error(y_b[val], pred))
print(f"  Difficulty MSE: {np.mean(cv_mse_b):.4f} ± {np.std(cv_mse_b):.4f}  (baseline: {y_b.var():.4f})")

print("Ridge regression cross-val (log-discrimination)...")
ridge_a = Ridge(alpha=1.0)
cv_mse_a = []
for tr, val in kf.split(X):
    ridge_a.fit(X[tr], y_log_a[tr])
    pred = ridge_a.predict(X[val])
    cv_mse_a.append(mean_squared_error(y_log_a[val], pred))
print(f"  Log-discrimination MSE: {np.mean(cv_mse_a):.4f} ± {np.std(cv_mse_a):.4f}  (baseline: {y_log_a.var():.4f})")

# ---------------------------------------------------------------------------
# 4. MLP regressor
# ---------------------------------------------------------------------------
class IRTBridge(nn.Module):
    def __init__(self, in_dim=384, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),  # [log_a_norm, b_norm]
        )

    def forward(self, x):
        return self.net(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n=== [02] IRT Bridge Training ===")
print(f"Device: {device}  |  Training items: {int(0.8*len(X))}  |  Val items: {int(0.2*len(X))}")
print(f"Epochs: 200  |  Batch: 128  |  Progress printed every 10 epochs", flush=True)

X_t = torch.from_numpy(X).to(device)
y_t = torch.from_numpy(
    np.stack([y_log_a_norm, y_b_norm], axis=1).astype(np.float32)
).to(device)

# Train/val split (80/20 by item)
n = len(X_t)
idx = torch.randperm(n)
n_train = int(0.8 * n)
tr_idx, val_idx = idx[:n_train], idx[n_train:]

model = IRTBridge().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

EPOCHS = 200
BATCH = 128
best_val_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):
    model.train()
    perm = tr_idx[torch.randperm(len(tr_idx))]
    train_loss = 0.0
    n_batches = 0
    for i in range(0, len(perm), BATCH):
        batch_idx = perm[i:i+BATCH]
        pred = model(X_t[batch_idx])
        loss = nn.functional.mse_loss(pred, y_t[batch_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_t[val_idx])
            val_loss = nn.functional.mse_loss(val_pred, y_t[val_idx]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch+1:3d}: train_loss={train_loss/n_batches:.4f}  val_loss={val_loss:.4f}")

# Load best model
if best_state:
    model.load_state_dict(best_state)

# Final validation MSE in original scale
model.eval()
with torch.no_grad():
    val_out = model(X_t[val_idx]).cpu().numpy()

pred_b = val_out[:, 1] * b_std + b_mean
pred_log_a = val_out[:, 0] * log_a_std + log_a_mean
pred_a = np.exp(pred_log_a)

true_b = y_b[val_idx.cpu().numpy()]
true_a = y_a[val_idx.cpu().numpy()]

mse_b = mean_squared_error(true_b, pred_b)
mse_a = mean_squared_error(true_a, pred_a)
print(f"\nVal MSE (difficulty): {mse_b:.4f}  (std={true_b.std():.4f})")
print(f"Val MSE (discrimination): {mse_a:.4f}  (std={true_a.std():.4f})")

# Save model
torch.save(model.state_dict(), DATA_DIR / "irt_bridge.pt")
with open(DATA_DIR / "irt_bridge_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nSaved irt_bridge.pt and irt_bridge_meta.json")

# ---------------------------------------------------------------------------
# 5. Build full subject theta lookup (including pass-rate proxy for unknowns)
# ---------------------------------------------------------------------------
print("\nBuilding full subject theta lookup...", flush=True)
theta_data = json.load(open(DATA_DIR / "subject_theta.json"))
known_by_name = theta_data["by_name"]  # subject_id or display_name -> theta
known_by_content = theta_data["by_content"]  # full subject_content -> theta
mean_theta = theta_data["mean_theta"]

resp_df = pd.read_parquet(DATA_DIR / "responses.parquet",
                          columns=["subject_id", "subject_content", "label"])

# Compute per-subject pass rate as theta proxy
subject_stats = resp_df.groupby("subject_id").agg(
    pass_rate=("label", "mean"),
    n_responses=("label", "count"),
    subject_content=("subject_content", "first"),
).reset_index()

# Convert pass rate to logit scale (IRT theta proxy)
def pass_rate_to_theta(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p)) * 0.6  # scale roughly to IRT theta range

subject_stats["proxy_theta"] = pass_rate_to_theta(subject_stats["pass_rate"])

# Build comprehensive lookup: subject_id -> theta
full_theta = {}
for _, row in subject_stats.iterrows():
    sid = row["subject_id"]
    # Check if we have a real IRT theta
    if sid in known_by_name:
        theta = known_by_name[sid]
    elif row["subject_content"] in known_by_content:
        theta = known_by_content[row["subject_content"]]
    else:
        # Use pass-rate proxy
        theta = row["proxy_theta"]
    full_theta[sid] = float(theta)

# Also add display-name lookups
subjects_with_content = resp_df.drop_duplicates("subject_id")[["subject_id", "subject_content"]]
for _, row in subjects_with_content.iterrows():
    content = row["subject_content"]
    first_line = content.split("\n")[0] if isinstance(content, str) else ""
    name = first_line.replace("Name:", "").strip()
    if name and name not in full_theta:
        full_theta[name] = full_theta.get(row["subject_id"], mean_theta)

coverage_pct = sum(1 for sid in subject_stats["subject_id"] if sid in [k for k in theta_data["by_name"]]) / len(subject_stats) * 100
print(f"IRT theta coverage: {coverage_pct:.1f}% of subjects")
print(f"Total entries in full theta lookup: {len(full_theta)}")

full_theta_data = {
    "by_id_or_name": full_theta,
    "by_content": known_by_content,
    "mean_theta": mean_theta,
}
with open(DATA_DIR / "subject_theta_full.json", "w") as f:
    json.dump(full_theta_data, f, indent=2)
print("Saved subject_theta_full.json")
