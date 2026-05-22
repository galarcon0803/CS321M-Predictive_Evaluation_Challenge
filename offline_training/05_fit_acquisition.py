"""
Fit k-means centroids for diversity-based acquisition function.

Output:
  ../data/centroids.npy  - (64, 768) float32 cluster centroids on item embeddings
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings"

N_CLUSTERS = 64

print(f"Loading item embeddings...", flush=True)
item_embs = np.load(EMB_DIR / "item_embs.npy")
print(f"Shape: {item_embs.shape}")

print(f"Fitting k-means with {N_CLUSTERS} clusters...", flush=True)
km = MiniBatchKMeans(n_clusters=N_CLUSTERS, n_init=5, random_state=42, verbose=1)
km.fit(item_embs)

centroids = km.cluster_centers_.astype(np.float32)
np.save(DATA_DIR / "centroids.npy", centroids)
print(f"Saved centroids.npy: shape={centroids.shape}")
print(f"Inertia: {km.inertia_:.2f}")
