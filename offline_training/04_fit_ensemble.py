"""
Fit ensemble weights + Platt calibration over NCF + IRT bridge predictions.

Evaluates on item cold-start validation split.
Outputs:
  ../data/ensemble_meta.json  - weights w_irt, w_ncf + Platt params
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit as sigmoid
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings"

# ---------------------------------------------------------------------------
# 1. Load everything
# ---------------------------------------------------------------------------
print("Loading models and data...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# NCF
class NCFHead(nn.Module):
    def __init__(self, in_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

with open(DATA_DIR / "ncf_meta.json") as _f:
    _ncf_meta = json.load(_f)
ncf = NCFHead(in_dim=_ncf_meta["in_dim"]).to(device)
ncf.load_state_dict(torch.load(DATA_DIR / "ncf_head.pt", map_location=device))
ncf.eval()

# IRT bridge
class IRTBridge(nn.Module):
    def __init__(self, in_dim=768, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.net(x)

irt_bridge = IRTBridge().to(device)
irt_bridge.load_state_dict(torch.load(DATA_DIR / "irt_bridge.pt", map_location=device))
irt_bridge.eval()

with open(DATA_DIR / "irt_bridge_meta.json") as f:
    irt_meta = json.load(f)

with open(DATA_DIR / "subject_theta_full.json") as f:
    theta_data = json.load(f)

# Embeddings
with open(EMB_DIR / "item_ids.json") as f:
    item_ids_list = json.load(f)
with open(EMB_DIR / "subject_ids.json") as f:
    subject_ids_list = json.load(f)
item_id_to_idx = {iid: i for i, iid in enumerate(item_ids_list)}
subject_id_to_idx = {sid: i for i, sid in enumerate(subject_ids_list)}
item_embs = torch.from_numpy(np.load(EMB_DIR / "item_embs.npy")).to(device)
subject_embs = torch.from_numpy(np.load(EMB_DIR / "subject_embs.npy")).to(device)

# ---------------------------------------------------------------------------
# 2. Build val set (same cold-start split as train_ncf.py)
# ---------------------------------------------------------------------------
print("Loading validation data...", flush=True)
resp_df = pd.read_parquet(DATA_DIR / "responses.parquet",
                          columns=["subject_id", "item_id", "label"])
resp_df["item_idx"] = resp_df["item_id"].map(item_id_to_idx)
resp_df["subject_idx"] = resp_df["subject_id"].map(subject_id_to_idx)
resp_df = resp_df.dropna(subset=["item_idx", "subject_idx"])
resp_df = resp_df.astype({"item_idx": int, "subject_idx": int})

unique_items = resp_df["item_id"].unique()
np.random.seed(42)
val_items = set(np.random.choice(unique_items, size=int(0.2 * len(unique_items)), replace=False))
val_df = resp_df[resp_df["item_id"].isin(val_items)].reset_index(drop=True)
val_df = val_df.sample(min(100_000, len(val_df)), random_state=42)
print(f"Val rows: {len(val_df):,}")

# ---------------------------------------------------------------------------
# 3. NCF predictions on val set
# ---------------------------------------------------------------------------
print("Computing NCF predictions...", flush=True)
s_idx = torch.tensor(val_df["subject_idx"].values, dtype=torch.long, device=device)
i_idx = torch.tensor(val_df["item_idx"].values, dtype=torch.long, device=device)

CHUNK = 10_000
ncf_probs = []
with torch.no_grad():
    for start in range(0, len(val_df), CHUNK):
        end = min(start + CHUNK, len(val_df))
        s_emb = subject_embs[s_idx[start:end]]
        i_emb = item_embs[i_idx[start:end]]
        X = torch.cat([s_emb, i_emb], dim=1)
        logits = ncf(X)
        ncf_probs.append(torch.sigmoid(logits).cpu().numpy())
ncf_probs = np.concatenate(ncf_probs)

# ---------------------------------------------------------------------------
# 4. IRT bridge predictions on val set
# ---------------------------------------------------------------------------
print("Computing IRT bridge predictions...", flush=True)

def get_theta(subject_id: str, content: str) -> float:
    tid = theta_data["by_id_or_name"]
    if subject_id in tid:
        return tid[subject_id]
    if content in theta_data["by_content"]:
        return theta_data["by_content"][content]
    # parse name from content
    name = content.split("\n")[0].replace("Name:", "").strip() if isinstance(content, str) else ""
    if name in tid:
        return tid[name]
    return theta_data["mean_theta"]

# Get subject thetas (from responses.parquet which has subject_content)
resp_full = pd.read_parquet(DATA_DIR / "responses.parquet",
                            columns=["subject_id", "subject_content"]).drop_duplicates("subject_id")
subject_theta_map = {}
for _, row in resp_full.iterrows():
    subject_theta_map[row["subject_id"]] = get_theta(row["subject_id"], row["subject_content"])

val_df["theta"] = val_df["subject_id"].map(subject_theta_map).fillna(theta_data["mean_theta"])

b_mean = irt_meta["b_mean"]; b_std = irt_meta["b_std"]
log_a_mean = irt_meta["log_a_mean"]; log_a_std = irt_meta["log_a_std"]

irt_probs = []
with torch.no_grad():
    for start in range(0, len(val_df), CHUNK):
        end = min(start + CHUNK, len(val_df))
        batch_i_idx = i_idx[start:end]
        i_emb = item_embs[batch_i_idx]
        out = irt_bridge(i_emb).cpu().numpy()  # (batch, 2) = [log_a_norm, b_norm]
        log_a = out[:, 0] * log_a_std + log_a_mean
        b = out[:, 1] * b_std + b_mean
        a = np.exp(log_a)
        thetas = val_df["theta"].values[start:end]
        p = sigmoid(a * (thetas - b))
        irt_probs.append(p)
irt_probs = np.concatenate(irt_probs)

y_val = val_df["label"].values

print(f"NCF logloss: {log_loss(y_val, np.clip(ncf_probs, 1e-7, 1-1e-7)):.4f}  AUC: {roc_auc_score(y_val, ncf_probs):.4f}")
print(f"IRT logloss: {log_loss(y_val, np.clip(irt_probs, 1e-7, 1-1e-7)):.4f}  AUC: {roc_auc_score(y_val, irt_probs):.4f}")

# ---------------------------------------------------------------------------
# 5. Fit ensemble weights via optimization
# ---------------------------------------------------------------------------
def ensemble_logloss(w):
    w_ncf, w_irt = w[0], 1.0 - w[0]
    p = np.clip(w_ncf * ncf_probs + w_irt * irt_probs, 1e-7, 1-1e-7)
    return log_loss(y_val, p)

result = minimize(ensemble_logloss, x0=[0.7], bounds=[(0.0, 1.0)], method="L-BFGS-B")
w_ncf = float(result.x[0])
w_irt = 1.0 - w_ncf
print(f"\nOptimal weights: NCF={w_ncf:.3f}, IRT={w_irt:.3f}")
ensemble_probs = np.clip(w_ncf * ncf_probs + w_irt * irt_probs, 1e-7, 1-1e-7)
print(f"Ensemble logloss: {log_loss(y_val, ensemble_probs):.4f}  AUC: {roc_auc_score(y_val, ensemble_probs):.4f}")

# ---------------------------------------------------------------------------
# 6. Platt scaling on ensemble
# ---------------------------------------------------------------------------
print("\nFitting Platt scaling...")
platt = LogisticRegression(C=1e4, solver="lbfgs")
platt.fit(ensemble_probs.reshape(-1, 1), y_val)
calibrated = np.clip(platt.predict_proba(ensemble_probs.reshape(-1, 1))[:, 1], 1e-7, 1-1e-7)
print(f"Calibrated logloss: {log_loss(y_val, calibrated):.4f}  AUC: {roc_auc_score(y_val, calibrated):.4f}")

platt_coef = float(platt.coef_[0][0])
platt_intercept = float(platt.intercept_[0])

# ---------------------------------------------------------------------------
# 7. Save ensemble meta
# ---------------------------------------------------------------------------
ensemble_meta = {
    "w_ncf": w_ncf,
    "w_irt": w_irt,
    "platt_coef": platt_coef,
    "platt_intercept": platt_intercept,
    "val_metrics": {
        "ncf_logloss": float(log_loss(y_val, np.clip(ncf_probs, 1e-7, 1-1e-7))),
        "irt_logloss": float(log_loss(y_val, np.clip(irt_probs, 1e-7, 1-1e-7))),
        "ensemble_logloss": float(log_loss(y_val, ensemble_probs)),
        "calibrated_logloss": float(log_loss(y_val, calibrated)),
    }
}
with open(DATA_DIR / "ensemble_meta.json", "w") as f:
    json.dump(ensemble_meta, f, indent=2)
print("\nSaved ensemble_meta.json")
print(json.dumps(ensemble_meta, indent=2))
