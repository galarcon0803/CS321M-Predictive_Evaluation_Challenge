"""
Modal GPU training script — runs all offline training steps on a B200 GPU.

Usage:
  modal run offline_training/modal_train.py

This will:
  1. Upload local data/ files to the Modal volume
  2. Run steps 01-05 on a B200 GPU
  3. Download trained artifacts back to data/ on your machine

Prerequisites:
  pip install modal
  modal token new   (authenticate once)
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image: install all Python deps
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pyarrow",
        "datasets",
        "huggingface_hub",
    )
)

app = modal.App("predictive-eval-training", image=image)

# Persistent volume to cache HF model weights across runs
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Helper: read local file as bytes
# ---------------------------------------------------------------------------
def _read(path: Path) -> bytes:
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Remote training function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100",
    timeout=60 * 60 * 2,  # 2 hours max
    volumes={"/root/.cache/huggingface": hf_cache},
    memory=32768,
)
def train_all(
    responses_parquet: bytes,
    item_lookup_parquet: bytes,
    item_params_csv: bytes,
    subject_abilities_csv: bytes,
    subject_theta_json: bytes,
    external_items_parquet: bytes | None = None,
) -> dict[str, bytes]:
    """
    Runs steps 01-05 entirely on the remote GPU.
    Returns a dict of filename -> bytes for all trained artifacts.
    """
    import json
    import math
    import tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from scipy.optimize import minimize
    from scipy.special import expit as sigmoid
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[modal] device={device}", flush=True)

    # Write input files to temp dir
    tmp = Path(tempfile.mkdtemp())
    (tmp / "responses.parquet").write_bytes(responses_parquet)
    (tmp / "item_lookup.parquet").write_bytes(item_lookup_parquet)
    (tmp / "item_params.csv").write_bytes(item_params_csv)
    (tmp / "subject_abilities.csv").write_bytes(subject_abilities_csv)
    (tmp / "subject_theta.json").write_bytes(subject_theta_json)
    if external_items_parquet:
        (tmp / "external_items.parquet").write_bytes(external_items_parquet)
        print(f"[modal] external_items.parquet uploaded ({len(external_items_parquet)//1024} KB)",
              flush=True)

    EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_SIZE = 2048

    # -----------------------------------------------------------------------
    # STEP 1: Compute embeddings
    # Key improvement over v1: embed benchmark+condition+item_content together
    # so training embeddings match what predict() uses at test time.
    # We embed unique (item_id, condition) pairs since same item under different
    # conditions gets a different embedding.
    # -----------------------------------------------------------------------
    print("\n=== STEP 1: Computing embeddings ===", flush=True)
    encoder = SentenceTransformer(EMB_MODEL, device=device)

    resp_df = pd.read_parquet(tmp / "responses.parquet")

    # Build unique (item_id, condition) combinations with their text
    item_cond = resp_df[["item_id", "benchmark_id", "test_condition", "item_content"]].drop_duplicates(
        subset=["item_id", "test_condition"]
    ).reset_index(drop=True)
    item_cond["test_condition"] = item_cond["test_condition"].fillna("none")
    item_cond["item_content"] = item_cond["item_content"].fillna("")

    # Text = "Benchmark: X\nCondition: Y\n<item_content>" — matches predict()
    item_cond["emb_text"] = (
        "Benchmark: " + item_cond["benchmark_id"] + "\n"
        + "Condition: " + item_cond["test_condition"] + "\n"
        + item_cond["item_content"]
    )

    item_cond_keys = list(zip(item_cond["item_id"], item_cond["test_condition"]))
    item_cond_to_idx = {k: i for i, k in enumerate(item_cond_keys)}
    # Also keep item_id-only index (uses first condition seen) for IRT bridge
    item_id_to_idx = {}
    for i, (iid, cond) in enumerate(item_cond_keys):
        if iid not in item_id_to_idx:
            item_id_to_idx[iid] = i

    print(f"Encoding {len(item_cond)} item×condition pairs ...", flush=True)
    item_embs = encoder.encode(
        item_cond["emb_text"].tolist(), batch_size=BATCH_SIZE,
        show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)
    print(f"Item embs: {item_embs.shape}", flush=True)

    subjects = resp_df.drop_duplicates("subject_id").reset_index(drop=True)
    subject_ids_list = subjects["subject_id"].tolist()
    subject_texts = subjects["subject_content"].fillna("").tolist()
    subject_id_to_idx = {sid: i for i, sid in enumerate(subject_ids_list)}

    print(f"Encoding {len(subject_texts)} subjects ...", flush=True)
    subject_embs = encoder.encode(subject_texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                                  convert_to_numpy=True).astype(np.float32)
    print(f"Subject embs: {subject_embs.shape}", flush=True)

    EMB_DIM = item_embs.shape[1]

    # -----------------------------------------------------------------------
    # STEP 1b: Joint 2PL IRT fit on the full response matrix
    # Fits theta_s (per subject), a_i, b_i (per item×condition) jointly via SGD
    # on the BCE loss. Extends IRT params from the 3,088 items in
    # item_params.csv to ALL ~70K items in responses.parquet — and gives proper
    # IRT theta for all 909 subjects (vs the pass-rate proxy for 655 of them).
    # -----------------------------------------------------------------------
    print("\n=== STEP 1b: Joint 2PL IRT fit ===", flush=True)
    import torch.nn.functional as F

    n_subj = len(subject_ids_list)
    n_item = len(item_cond_keys)

    irt_resp = resp_df.copy()
    irt_resp["test_condition"] = irt_resp["test_condition"].fillna("none")
    irt_resp["s_idx_irt"] = irt_resp["subject_id"].map(subject_id_to_idx)
    irt_resp["i_idx_irt"] = [
        item_cond_to_idx.get((iid, cond), -1)
        for iid, cond in zip(irt_resp["item_id"], irt_resp["test_condition"])
    ]
    irt_resp = irt_resp[irt_resp["i_idx_irt"] >= 0]
    irt_resp = irt_resp.dropna(subset=["s_idx_irt"])
    irt_resp["s_idx_irt"] = irt_resp["s_idx_irt"].astype(int)
    irt_resp["i_idx_irt"] = irt_resp["i_idx_irt"].astype(int)
    print(f"IRT: {len(irt_resp):,} responses  {n_subj} subjects  {n_item} items",
          flush=True)

    s_irt_t = torch.tensor(irt_resp["s_idx_irt"].values, device=device, dtype=torch.long)
    i_irt_t = torch.tensor(irt_resp["i_idx_irt"].values, device=device, dtype=torch.long)
    y_irt_t = torch.tensor(irt_resp["label"].values, device=device, dtype=torch.float32)

    theta_p = torch.zeros(n_subj, device=device, requires_grad=True)
    log_a_p = torch.zeros(n_item, device=device, requires_grad=True)
    b_p     = torch.zeros(n_item, device=device, requires_grad=True)
    opt_irt = torch.optim.Adam([theta_p, log_a_p, b_p], lr=0.05)

    BS_IRT = 200_000
    EPOCHS_IRT = 25
    n_resp_irt = len(y_irt_t)

    for epoch in range(EPOCHS_IRT):
        perm = torch.randperm(n_resp_irt, device=device)
        tot, nb = 0.0, 0
        for start in range(0, n_resp_irt, BS_IRT):
            bb = perm[start:start + BS_IRT]
            s_b = s_irt_t[bb]; i_b = i_irt_t[bb]; y_b = y_irt_t[bb]
            a_b = torch.exp(log_a_p[i_b].clamp(-2.0, 1.8))   # a in [0.135, 6.05]
            b_b = b_p[i_b].clamp(-5.0, 5.0)
            logit = a_b * (theta_p[s_b] - b_b)
            loss = F.binary_cross_entropy_with_logits(logit, y_b)
            loss = loss + 1e-4 * ((log_a_p[i_b] ** 2).mean() +
                                  (b_p[i_b] ** 2).mean())
            opt_irt.zero_grad(); loss.backward(); opt_irt.step()
            tot += loss.item(); nb += 1
        # identifiability: center theta at 0, shift b by same amount
        with torch.no_grad():
            m = theta_p.mean()
            theta_p.data.sub_(m)
            b_p.data.sub_(m)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                a_med = torch.exp(log_a_p).median().item()
                a_lo, a_hi = (torch.exp(log_a_p).quantile(0.05).item(),
                              torch.exp(log_a_p).quantile(0.95).item())
            print(f"  IRT epoch {epoch+1:2d}/{EPOCHS_IRT}: bce={tot/nb:.4f}  "
                  f"theta_std={theta_p.std().item():.3f}  "
                  f"a_med={a_med:.2f}  a_p05-p95=[{a_lo:.2f},{a_hi:.2f}]  "
                  f"b_std={b_p.std().item():.3f}", flush=True)

    theta_fit = theta_p.detach().cpu().numpy()
    a_fit = np.exp(log_a_p.detach().cpu().numpy()).clip(0.1, 6.0).astype(np.float32)
    b_fit = b_p.detach().cpu().numpy().clip(-5.0, 5.0).astype(np.float32)
    print(f"Joint IRT done. theta range=[{theta_fit.min():.2f},{theta_fit.max():.2f}]  "
          f"a range=[{a_fit.min():.2f},{a_fit.max():.2f}]  "
          f"b range=[{b_fit.min():.2f},{b_fit.max():.2f}]", flush=True)

    # -----------------------------------------------------------------------
    # STEP 2: Train IRT bridge on ALL (item, condition) pairs
    # Targets come from joint IRT fit (vs old: only 3,088 items in item_params.csv)
    # -----------------------------------------------------------------------
    print("\n=== STEP 2: Training IRT bridge (all items) ===", flush=True)

    # Filter items with enough observations for reliable IRT params
    obs_count = np.zeros(n_item, dtype=np.int64)
    np.add.at(obs_count, irt_resp["i_idx_irt"].values, 1)
    keep_mask = obs_count >= 5
    print(f"Items kept (>=5 obs): {keep_mask.sum():,} / {n_item}", flush=True)

    X_irt = item_embs[keep_mask]
    y_a = a_fit[keep_mask]
    y_b = b_fit[keep_mask]
    y_log_a = np.log(y_a)

    b_mean, b_std = float(y_b.mean()), float(y_b.std())
    log_a_mean, log_a_std = float(y_log_a.mean()), float(y_log_a.std())
    y_b_norm = (y_b - b_mean) / (b_std + 1e-8)
    y_log_a_norm = (y_log_a - log_a_mean) / (log_a_std + 1e-8)

    irt_meta = {"b_mean": b_mean, "b_std": b_std,
                "log_a_mean": log_a_mean, "log_a_std": log_a_std,
                "in_dim": EMB_DIM, "emb_model": EMB_MODEL,
                "n_items_trained": int(keep_mask.sum())}

    class IRTBridge(nn.Module):
        def __init__(self, in_dim=384, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(hidden, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 2),
            )
        def forward(self, x): return self.net(x)

    # Cold-start train/val split BY item_id (not random row split): a test item
    # is always a new item, so the bridge has to generalize across items.
    kept_item_keys = [item_cond_keys[i] for i in np.where(keep_mask)[0]]
    kept_unique_item_ids = list({k[0] for k in kept_item_keys})
    np.random.seed(321)
    val_iids_irt = set(np.random.choice(
        kept_unique_item_ids,
        size=int(0.2 * len(kept_unique_item_ids)),
        replace=False,
    ))
    val_split_mask = np.array([k[0] in val_iids_irt for k in kept_item_keys])
    tr_split_mask = ~val_split_mask

    X_t = torch.from_numpy(X_irt).to(device)
    y_t = torch.from_numpy(np.stack([y_log_a_norm, y_b_norm], axis=1).astype(np.float32)).to(device)
    tr_idx = torch.from_numpy(np.where(tr_split_mask)[0]).to(device)
    val_idx = torch.from_numpy(np.where(val_split_mask)[0]).to(device)
    print(f"IRT bridge: train={len(tr_idx):,}  val={len(val_idx):,} (cold-start by item_id)",
          flush=True)

    irt_model = IRTBridge(in_dim=EMB_DIM).to(device)
    opt = torch.optim.AdamW(irt_model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    best_val, best_state = float("inf"), None

    BS_BRIDGE = 512
    for epoch in range(200):
        irt_model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        tl = 0.0; nb = 0
        for i in range(0, len(perm), BS_BRIDGE):
            b = perm[i:i+BS_BRIDGE]
            loss = nn.functional.mse_loss(irt_model(X_t[b]), y_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        sched.step()
        if (epoch + 1) % 10 == 0:
            irt_model.eval()
            with torch.no_grad():
                # chunk val to avoid OOM if val set is big
                vl_parts = []
                for vs in range(0, len(val_idx), 4096):
                    vb = val_idx[vs:vs+4096]
                    vl_parts.append(
                        nn.functional.mse_loss(irt_model(X_t[vb]), y_t[vb], reduction="sum").item()
                    )
                vl = sum(vl_parts) / (len(val_idx) * 2)  # 2 outputs
            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in irt_model.state_dict().items()}
            print(f"  IRT epoch {epoch+1:3d}: train={tl/nb:.4f}  val={vl:.4f}", flush=True)

    if best_state: irt_model.load_state_dict(best_state)
    print(f"IRT bridge best val loss: {best_val:.4f}", flush=True)
    irt_meta["val_mse"] = float(best_val)

    # -----------------------------------------------------------------------
    # STEP 2b: Per-item difficulty bridge
    # Trains MLP: item_emb → residual logit difficulty (within-benchmark)
    # Residual = logit(item_pass_rate) - logit(benchmark_mean_pass_rate)
    # Captures item-level difficulty variation independent of benchmark level;
    # alpha handles the benchmark-level offset at test time.
    # -----------------------------------------------------------------------
    print("\n=== STEP 2b: Per-item difficulty bridge ===", flush=True)

    def _safe_logit(p_arr, eps=0.01):
        p_arr = np.clip(p_arr, eps, 1.0 - eps)
        return np.log(p_arr / (1.0 - p_arr))

    # Per-(item, condition, benchmark) pass rate
    item_bench_stats = resp_df.groupby(
        ["item_id", "test_condition", "benchmark_id"]
    ).agg(pass_rate=("label", "mean"), n_obs=("label", "count")).reset_index()

    # Benchmark mean pass rate for residualization
    bench_mean_pr = resp_df.groupby("benchmark_id")["label"].mean().rename("bench_mean_pr")
    item_bench_stats = item_bench_stats.join(bench_mean_pr, on="benchmark_id")

    item_bench_stats["residual_logit"] = (
        _safe_logit(item_bench_stats["pass_rate"].values.astype(float))
        - _safe_logit(item_bench_stats["bench_mean_pr"].values.astype(float))
    )

    # Map items to their (benchmark+condition-prefixed) embeddings
    item_bench_stats["test_condition"] = item_bench_stats["test_condition"].fillna("none")
    item_bench_stats["emb_idx"] = [
        item_cond_to_idx.get((iid, cond), item_id_to_idx.get(iid))
        for iid, cond in zip(item_bench_stats["item_id"], item_bench_stats["test_condition"])
    ]
    item_bench_stats = item_bench_stats.dropna(subset=["emb_idx"]).reset_index(drop=True)
    item_bench_stats["emb_idx"] = item_bench_stats["emb_idx"].astype(int)

    # Only use items with >=5 observations for reliable pass rate estimates
    item_bench_stats_filt = item_bench_stats[
        item_bench_stats["n_obs"] >= 5
    ].reset_index(drop=True)
    print(f"Items >=5 obs: {len(item_bench_stats_filt):,} / {len(item_bench_stats):,}", flush=True)

    X_diff_int = item_embs[item_bench_stats_filt["emb_idx"].values]
    y_diff_int = item_bench_stats_filt["residual_logit"].values.astype(np.float32)
    print(f"y_diff (internal): mean={y_diff_int.mean():.3f}  std={y_diff_int.std():.3f}",
          flush=True)

    # Cold-start train/val split on internal items only (val stays internal)
    unique_diff_items = item_bench_stats_filt["item_id"].unique()
    np.random.seed(123)
    val_diff_set = set(np.random.choice(
        unique_diff_items, size=int(0.2 * len(unique_diff_items)), replace=False
    ))
    diff_train_mask = ~item_bench_stats_filt["item_id"].isin(val_diff_set)
    diff_val_mask = item_bench_stats_filt["item_id"].isin(val_diff_set)

    X_diff_tr = X_diff_int[diff_train_mask.values]
    y_diff_tr = y_diff_int[diff_train_mask.values]
    X_diff_val = X_diff_int[diff_val_mask.values]
    y_diff_val = y_diff_int[diff_val_mask.values]

    # Extend training set with external items (MMLU, ARC, BigBench, etc.)
    # Val stays internal-only so we get a clean cold-start generalization signal.
    ext_path = tmp / "external_items.parquet"
    if ext_path.exists():
        print("  Loading external items ...", flush=True)
        ext_df = pd.read_parquet(ext_path)
        # Use same prefix format as internal items so embeddings are in the same space
        ext_texts = (
            "Benchmark: " + ext_df["benchmark_id"].fillna("unknown") + "\n"
            + "Condition: none\n"
            + ext_df["item_content"].fillna("")
        ).tolist()
        print(f"  Encoding {len(ext_texts):,} external items (with benchmark prefix) ...",
              flush=True)
        ext_embs = encoder.encode(
            ext_texts, batch_size=BATCH_SIZE, show_progress_bar=True,
            convert_to_numpy=True
        ).astype(np.float32)

        ext_pass = ext_df["pass_rate"].values.astype(float)
        ext_logits = np.log(np.clip(ext_pass, 0.02, 0.98) /
                            np.clip(1 - ext_pass, 0.02, 0.98)).astype(np.float32)
        ext_logits -= ext_logits.mean()  # center so mean=0 like internal residuals

        X_diff_tr = np.concatenate([X_diff_tr, ext_embs], axis=0)
        y_diff_tr = np.concatenate([y_diff_tr, ext_logits], axis=0)
        print(f"  Train after external: {len(X_diff_tr):,}  Val (internal only): {len(X_diff_val):,}",
              flush=True)
        print(f"  y_diff_tr combined: mean={y_diff_tr.mean():.3f}  std={y_diff_tr.std():.3f}",
              flush=True)

    X_diff_tr_t = torch.from_numpy(X_diff_tr).to(device)
    y_diff_tr_t = torch.from_numpy(y_diff_tr).to(device)
    X_diff_val_t = torch.from_numpy(X_diff_val).to(device)
    y_diff_val_t = torch.from_numpy(y_diff_val).to(device)
    print(f"DiffBridge train: {len(X_diff_tr_t):,}  val: {len(X_diff_val_t):,}", flush=True)

    class ItemDiffBridge(nn.Module):
        def __init__(self, in_dim=384):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    diff_model = ItemDiffBridge(in_dim=EMB_DIM).to(device)
    diff_opt = torch.optim.AdamW(diff_model.parameters(), lr=3e-4, weight_decay=1e-3)
    diff_sched = torch.optim.lr_scheduler.CosineAnnealingLR(diff_opt, T_max=100)
    diff_loader = DataLoader(
        TensorDataset(X_diff_tr_t.cpu(), y_diff_tr_t.cpu()),
        batch_size=512, shuffle=True, num_workers=0
    )

    best_diff_val, best_diff_state = float("inf"), None
    for epoch in range(100):
        diff_model.train()
        tl = 0.0; nb = 0
        for xb, yb in diff_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(diff_model(xb), yb)
            diff_opt.zero_grad(); loss.backward(); diff_opt.step()
            tl += loss.item(); nb += 1
        diff_sched.step()
        diff_model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(diff_model(X_diff_val_t), y_diff_val_t).item()
        if vl < best_diff_val:
            best_diff_val = vl
            best_diff_state = {k: v.cpu().clone() for k, v in diff_model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print(f"  DiffBridge epoch {epoch+1:3d}: train={tl/nb:.4f}  val={vl:.4f}", flush=True)

    if best_diff_state:
        diff_model.load_state_dict(best_diff_state)
    print(f"DiffBridge best val MSE: {best_diff_val:.4f}", flush=True)
    diff_meta = {"in_dim": EMB_DIM, "val_mse": float(best_diff_val), "emb_model": EMB_MODEL}

    # Build full subject theta lookup from the joint-IRT fit (Step 1b).
    # Every subject with response data gets a proper IRT theta (vs the old
    # pass-rate proxy for 655 / 909 subjects).
    theta_raw = json.loads((tmp / "subject_theta.json").read_text())
    known_by_content = theta_raw["by_content"]
    mean_theta = float(theta_fit.mean())

    full_theta = {}
    by_content_irt = {}
    for sid, sidx in subject_id_to_idx.items():
        th = float(theta_fit[sidx])
        full_theta[sid] = th
    # Add name-based and content-based aliases
    subj_meta = resp_df.drop_duplicates("subject_id")[["subject_id", "subject_content"]]
    for _, row in subj_meta.iterrows():
        sid = row["subject_id"]
        content = (row["subject_content"] or "")
        name = content.split("\n")[0].replace("Name:", "").strip() if content else ""
        if name and sid in full_theta:
            full_theta[name] = full_theta[sid]
        if content and sid in full_theta:
            by_content_irt[content] = full_theta[sid]

    subject_theta_full = {"by_id_or_name": full_theta,
                          "by_content": by_content_irt,
                          "mean_theta": mean_theta}
    print(f"Subject theta (IRT): {len(full_theta)} entries, "
          f"mean={mean_theta:.3f}, std={float(theta_fit.std()):.3f}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 3: Train NCF
    # -----------------------------------------------------------------------
    print("\n=== STEP 3: Training NCF ===", flush=True)
    # Look up by (item_id, condition) to use condition-aware embeddings
    resp_df["test_condition"] = resp_df["test_condition"].fillna("none")
    resp_df["item_cond_idx"] = [
        item_cond_to_idx.get((iid, cond), item_id_to_idx.get(iid))
        for iid, cond in zip(resp_df["item_id"], resp_df["test_condition"])
    ]
    resp_df["subject_idx"] = resp_df["subject_id"].map(subject_id_to_idx)
    resp_df = resp_df.dropna(subset=["item_cond_idx", "subject_idx"])
    resp_df = resp_df.astype({"item_cond_idx": int, "subject_idx": int})

    unique_items = resp_df["item_id"].unique()
    np.random.seed(42)
    val_items = set(np.random.choice(unique_items, size=int(0.2 * len(unique_items)), replace=False))
    train_df = resp_df[~resp_df["item_id"].isin(val_items)].reset_index(drop=True)
    val_df_ncf = resp_df[resp_df["item_id"].isin(val_items)].reset_index(drop=True)

    print(f"Train: {len(train_df):,}  Val: {len(val_df_ncf):,}", flush=True)

    item_embs_t = torch.from_numpy(item_embs).to(device)
    subject_embs_t = torch.from_numpy(subject_embs).to(device)

    def make_XY(df, max_rows=None):
        if max_rows and len(df) > max_rows: df = df.sample(max_rows, random_state=42)
        s = torch.tensor(df["subject_idx"].values, dtype=torch.long, device=device)
        i = torch.tensor(df["item_cond_idx"].values, dtype=torch.long, device=device)
        X = torch.cat([subject_embs_t[s], item_embs_t[i]], dim=1).cpu()
        y = torch.tensor(df["label"].values, dtype=torch.float32)
        return X, y

    X_tr, y_tr = make_XY(train_df, max_rows=2_000_000)
    X_val, y_val = make_XY(val_df_ncf, max_rows=300_000)
    print(f"X_train: {X_tr.shape}  X_val: {X_val.shape}", flush=True)

    class NCFHead(nn.Module):
        def __init__(self, in_dim=768):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    ncf = NCFHead(in_dim=EMB_DIM * 2).to(device)
    print(f"NCF parameters: {sum(p.numel() for p in ncf.parameters()):,}", flush=True)
    ncf_opt = torch.optim.AdamW(ncf.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=4096, shuffle=True, num_workers=0)

    X_val_d = X_val.to(device); y_val_d = y_val.to(device)
    best_ncf_val, best_ncf_state = float("inf"), None
    EPOCHS = 15

    for epoch in range(EPOCHS):
        ncf.train()
        tl = 0.0; nb = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(ncf(xb), yb)
            ncf_opt.zero_grad(); loss.backward(); ncf_opt.step()
            tl += loss.item(); nb += 1
        ncf.eval()
        with torch.no_grad():
            vlogits = ncf(X_val_d)
            vl = crit(vlogits, y_val_d).item()
            vprobs = torch.sigmoid(vlogits).cpu().numpy()
        vll = log_loss(y_val.numpy(), vprobs)
        vauc = roc_auc_score(y_val.numpy(), vprobs)
        if vl < best_ncf_val:
            best_ncf_val = vl
            best_ncf_state = {k: v.cpu().clone() for k, v in ncf.state_dict().items()}
        print(f"  NCF epoch {epoch+1:2d}/{EPOCHS}: train_bce={tl/nb:.4f}  "
              f"val_bce={vl:.4f}  val_logloss={vll:.4f}  val_auc={vauc:.4f}", flush=True)

    if best_ncf_state: ncf.load_state_dict(best_ncf_state)
    ncf_meta = {"in_dim": EMB_DIM * 2, "emb_dim": EMB_DIM,
                "val_logloss": float(vll), "val_auc": float(vauc)}

    # -----------------------------------------------------------------------
    # STEP 4: Fit ensemble + Platt calibration
    # -----------------------------------------------------------------------
    print("\n=== STEP 4: Fitting ensemble ===", flush=True)
    val_sample = val_df_ncf.sample(min(100_000, len(val_df_ncf)), random_state=42)
    s_idx = torch.tensor(val_sample["subject_idx"].values, dtype=torch.long, device=device)
    i_idx = torch.tensor(val_sample["item_cond_idx"].values, dtype=torch.long, device=device)
    y_ens = val_sample["label"].values

    CHUNK = 10_000
    ncf.eval(); irt_model.eval()
    ncf_probs, irt_probs = [], []

    for start in range(0, len(val_sample), CHUNK):
        end = min(start + CHUNK, len(val_sample))
        s_e = subject_embs_t[s_idx[start:end]]
        i_e = item_embs_t[i_idx[start:end]]
        with torch.no_grad():
            ncf_probs.append(torch.sigmoid(ncf(torch.cat([s_e, i_e], dim=1))).cpu().numpy())
            out = irt_model(i_e).cpu().numpy()
        log_a = out[:, 0] * log_a_std + log_a_mean
        b_pred = out[:, 1] * b_std + b_mean
        a_pred = np.exp(log_a)
        sids = val_sample["subject_id"].values[start:end]
        thetas = np.array([full_theta.get(sid, mean_theta) for sid in sids])
        irt_probs.append(sigmoid(a_pred * (thetas - b_pred)))

    ncf_probs = np.concatenate(ncf_probs)
    irt_probs = np.concatenate(irt_probs)

    print(f"NCF logloss: {log_loss(y_ens, np.clip(ncf_probs, 1e-7, 1-1e-7)):.4f}")
    print(f"IRT logloss: {log_loss(y_ens, np.clip(irt_probs, 1e-7, 1-1e-7)):.4f}", flush=True)

    def ens_ll(w):
        p = np.clip(w[0]*ncf_probs + (1-w[0])*irt_probs, 1e-7, 1-1e-7)
        return log_loss(y_ens, p)

    res = minimize(ens_ll, x0=[0.7], bounds=[(0.0, 1.0)], method="L-BFGS-B")
    w_ncf = float(res.x[0]); w_irt = 1.0 - w_ncf
    ens_probs = np.clip(w_ncf*ncf_probs + w_irt*irt_probs, 1e-7, 1-1e-7)
    print(f"Ensemble weights: NCF={w_ncf:.3f}  IRT={w_irt:.3f}", flush=True)
    print(f"Ensemble logloss: {log_loss(y_ens, ens_probs):.4f}", flush=True)

    platt = LogisticRegression(C=1e4, solver="lbfgs")
    platt.fit(ens_probs.reshape(-1, 1), y_ens)
    cal = np.clip(platt.predict_proba(ens_probs.reshape(-1, 1))[:, 1], 1e-7, 1-1e-7)
    print(f"Calibrated logloss: {log_loss(y_ens, cal):.4f}", flush=True)

    ensemble_meta = {
        "w_ncf": w_ncf, "w_irt": w_irt,
        "platt_coef": float(platt.coef_[0][0]),
        "platt_intercept": float(platt.intercept_[0]),
        "val_metrics": {
            "ncf_logloss": float(log_loss(y_ens, np.clip(ncf_probs, 1e-7, 1-1e-7))),
            "irt_logloss": float(log_loss(y_ens, np.clip(irt_probs, 1e-7, 1-1e-7))),
            "ensemble_logloss": float(log_loss(y_ens, ens_probs)),
            "calibrated_logloss": float(log_loss(y_ens, cal)),
        }
    }

    # -----------------------------------------------------------------------
    # STEP 5: Fit k-means acquisition centroids
    # -----------------------------------------------------------------------
    print("\n=== STEP 5: Fitting k-means centroids ===", flush=True)
    km = MiniBatchKMeans(n_clusters=64, n_init=5, random_state=42, verbose=0)
    km.fit(item_embs)
    centroids = km.cluster_centers_.astype(np.float32)
    print(f"Centroids: {centroids.shape}  inertia={km.inertia_:.2f}", flush=True)

    # -----------------------------------------------------------------------
    # Serialize all outputs
    # -----------------------------------------------------------------------
    print("\n=== Serializing artifacts ===", flush=True)
    artifacts = {}

    buf = io.BytesIO()
    torch.save(irt_model.state_dict(), buf)
    artifacts["irt_bridge.pt"] = buf.getvalue()

    buf = io.BytesIO()
    torch.save(ncf.state_dict(), buf)
    artifacts["ncf_head.pt"] = buf.getvalue()

    buf = io.BytesIO()
    torch.save(diff_model.state_dict(), buf)
    artifacts["item_difficulty_bridge.pt"] = buf.getvalue()

    artifacts["irt_bridge_meta.json"] = json.dumps(irt_meta, indent=2).encode()
    artifacts["ncf_meta.json"] = json.dumps(ncf_meta, indent=2).encode()
    artifacts["ensemble_meta.json"] = json.dumps(ensemble_meta, indent=2).encode()
    artifacts["item_difficulty_meta.json"] = json.dumps(diff_meta, indent=2).encode()
    artifacts["subject_theta_full.json"] = json.dumps(subject_theta_full, indent=2).encode()

    buf = io.BytesIO()
    np.save(buf, centroids)
    artifacts["centroids.npy"] = buf.getvalue()

    sizes = {k: f"{len(v)/1024:.0f} KB" for k, v in artifacts.items()}
    print("Artifact sizes:", sizes, flush=True)
    return artifacts


# ---------------------------------------------------------------------------
# Local entrypoint: upload data, call remote fn, download results
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data"

    print("Uploading local data files to Modal ...", flush=True)
    required = [
        DATA / "responses.parquet",
        DATA / "item_lookup.parquet",
        DATA / "item_params.csv",
        DATA / "subject_abilities.csv",
        DATA / "subject_theta.json",
    ]
    for f in required:
        if not f.exists():
            raise FileNotFoundError(
                f"Missing: {f}\nRun offline_training/00_download_data.py first."
            )

    ext_path = DATA / "external_items.parquet"
    ext_bytes = _read(ext_path) if ext_path.exists() else None
    if ext_bytes:
        print(f"  Including external_items.parquet ({len(ext_bytes)//1024} KB)", flush=True)
    else:
        print("  No external_items.parquet found; run 07_fetch_external_data.py to add it.",
              flush=True)

    artifacts = train_all.remote(
        responses_parquet=_read(DATA / "responses.parquet"),
        item_lookup_parquet=_read(DATA / "item_lookup.parquet"),
        item_params_csv=_read(DATA / "item_params.csv"),
        subject_abilities_csv=_read(DATA / "subject_abilities.csv"),
        subject_theta_json=_read(DATA / "subject_theta.json"),
        external_items_parquet=ext_bytes,
    )

    print("\nDownloading artifacts to data/ ...", flush=True)
    for name, blob in artifacts.items():
        dest = DATA / name
        dest.write_bytes(blob)
        print(f"  Saved {dest.name}  ({len(blob)/1024:.0f} KB)")

    print("\nDone! Now run:")
    print("  python offline_training/06_package_submission.py")
