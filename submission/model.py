"""
Predictive Evaluation Challenge — submission model.

Strategy: 2PL IRT with text bridge for per-item (a, b).

  P(correct) = sigmoid(a_item * (theta_subject - b_item) + alpha_benchmark)

  theta_subject = subject ability (joint 1PL/2PL IRT fit over training data)
  a_item, b_item = item discrimination + difficulty, predicted from item text
                   via a small MLP bridge trained on joint-IRT params from
                   ~70K training items (all conditions). Falls back to a 1PL
                   diff bridge if the IRT bridge is unavailable.
  alpha_benchmark = per-benchmark calibration shift, fitted online from the
                    K=5 labeled examples per benchmark each round, with
                    sample-size shrinkage toward the pooled global alpha.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Local smoke test flag — skip slow model loads during local validation
# ---------------------------------------------------------------------------
_LOCAL_SMOKE = os.environ.get("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST", "").lower() in {
    "1", "true", "yes", "on"
}
print(f"[model] smoke_test={_LOCAL_SMOKE}", flush=True)

# ---------------------------------------------------------------------------
# Subject theta lookup (joint-IRT fit over all 909 subjects)
# ---------------------------------------------------------------------------
with open(_HERE / "subject_theta.json") as _f:
    _THETA_DATA = json.load(_f)

_THETA_BY_ID: dict[str, float] = _THETA_DATA["by_id_or_name"]
_THETA_BY_CONTENT: dict[str, float] = _THETA_DATA["by_content"]
_MEAN_THETA: float = _THETA_DATA["mean_theta"]

_TRAINING_PASS_RATE = 0.664
_DEFAULT_ALPHA: float = math.log(_TRAINING_PASS_RATE / (1.0 - _TRAINING_PASS_RATE)) - _MEAN_THETA

print(f"[model] mean_theta={_MEAN_THETA:.3f}  default_alpha={_DEFAULT_ALPHA:.3f}", flush=True)
print(f"[model] subjects with known theta: {len(_THETA_BY_ID)}", flush=True)


# ---------------------------------------------------------------------------
# Text bridges (optional): IRT bridge (preferred) -> diff bridge (fallback)
# ---------------------------------------------------------------------------
def _declared_model() -> str:
    p = _HERE / "models.txt"
    if not p.exists():
        return ""
    lines = [l.strip() for l in p.read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    return lines[0] if lines else ""


def _resolve_cache_dir() -> str | None:
    candidates = [
        os.environ.get("HF_HOME", "").strip(),
        "/app/hf_cache",
        str(_HERE / ".hf_cache"),
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(p, os.W_OK):
            return str(p)
    return None


_ENCODER = None
_HAS_IRT_BRIDGE = False
_IRT_BRIDGE_MODEL = None
_IRT_NORM = None  # (log_a_mean, log_a_std, b_mean, b_std)
_ITEM_IRT_CACHE: dict[str, tuple[float, float]] = {}

_HAS_DIFF_MODEL = False
_DIFF_MODEL = None
_ITEM_DIFF_CACHE: dict[str, float] = {}

_repo_id = _declared_model()
_irt_pt = _HERE / "irt_bridge.pt"
_irt_meta_f = _HERE / "irt_bridge_meta.json"
_diff_pt = _HERE / "item_difficulty_bridge.pt"
_diff_meta_f = _HERE / "item_difficulty_meta.json"


def _ensure_encoder():
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    from sentence_transformers import SentenceTransformer
    cache_dir = _resolve_cache_dir()
    _ENCODER = SentenceTransformer(_repo_id, cache_folder=cache_dir)
    return _ENCODER


if not _LOCAL_SMOKE and _irt_pt.exists() and _irt_meta_f.exists() and _repo_id:
    try:
        import torch
        import torch.nn as nn

        _irt_meta = json.load(open(_irt_meta_f))
        _IRT_NORM = (
            float(_irt_meta["log_a_mean"]),
            float(_irt_meta["log_a_std"]),
            float(_irt_meta["b_mean"]),
            float(_irt_meta["b_std"]),
        )
        _in_dim_irt = int(_irt_meta["in_dim"])
        _ensure_encoder()

        class _IRTBridge(nn.Module):
            def __init__(self, in_dim: int, hidden: int = 256) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(hidden, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(128, 2),
                )

            def forward(self, x):
                return self.net(x)

        _IRT_BRIDGE_MODEL = _IRTBridge(in_dim=_in_dim_irt)
        _IRT_BRIDGE_MODEL.load_state_dict(
            torch.load(_irt_pt, map_location="cpu", weights_only=True)
        )
        _IRT_BRIDGE_MODEL.eval()
        _HAS_IRT_BRIDGE = True
        print(f"[model] IRT bridge loaded (in_dim={_in_dim_irt})", flush=True)
    except Exception as _e:
        print(f"[model] IRT bridge unavailable: {_e}", flush=True)
        _HAS_IRT_BRIDGE = False

if not _LOCAL_SMOKE and _diff_pt.exists() and _diff_meta_f.exists() and _repo_id:
    try:
        import torch
        import torch.nn as nn

        _diff_meta = json.load(open(_diff_meta_f))
        _in_dim_diff = int(_diff_meta["in_dim"])
        _ensure_encoder()

        class _DiffBridge(nn.Module):
            def __init__(self, in_dim: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(128, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        _DIFF_MODEL = _DiffBridge(_in_dim_diff)
        _DIFF_MODEL.load_state_dict(
            torch.load(_diff_pt, map_location="cpu", weights_only=True)
        )
        _DIFF_MODEL.eval()
        _HAS_DIFF_MODEL = True
        print(f"[model] diff bridge fallback loaded (in_dim={_in_dim_diff})", flush=True)
    except Exception as _e:
        print(f"[model] diff bridge fallback unavailable: {_e}", flush=True)

if _LOCAL_SMOKE:
    print("[model] skipping HF load for local smoke test.", flush=True)
elif not (_HAS_IRT_BRIDGE or _HAS_DIFF_MODEL):
    print("[model] no text bridge present; using pure IRT theta.", flush=True)


# ---------------------------------------------------------------------------
# Per-round adaptive alpha state (per-benchmark + global)
# Per-benchmark alpha shrinks toward global alpha by sample size:
#   alpha_b = (n_b * alpha_b_raw + K * alpha_global) / (n_b + K)
# ---------------------------------------------------------------------------
_ROUND_ALPHAS: dict[str, float] = {}
_ROUND_GLOBAL_ALPHA: float | None = None
_ROUND_LABELED_LEN: int = 0
_BM_ALPHA_PRIOR_K: float = 8.0


# ---------------------------------------------------------------------------
# Subject behavioral fingerprint (Bayesian mixture over training benchmarks)
#
# Idea: at test time the K=5 labeled examples reveal which training benchmark
# the test benchmark most resembles. For each candidate training benchmark b,
# compute P(K labels | test_bm = b) using the per-subject pass-rate matrix.
# Softmax → posterior weights w_b. Then predict for any (subject, item):
#     P(correct) = Σ_b w_b · pass_rate(subject, b)
#
# Captures subject-benchmark specialization that single-theta IRT cannot
# (e.g., GPT-4 is unusually strong at math, Llama at instructions).
# ---------------------------------------------------------------------------
_FP_DATA: dict | None = None
_FP_BY_NAME: dict[str, dict[str, float]] = {}
_FP_BY_CONTENT: dict[str, dict[str, float]] = {}
_FP_BENCHMARKS: list[str] = []
_FP_BM_PRIOR: dict[str, float] = {}
_FP_GLOBAL_MEAN: float = 0.664

_fp_path = _HERE / "subject_benchmark_fingerprint.json"
if _fp_path.exists():
    try:
        with open(_fp_path) as _ff:
            _FP_DATA = json.load(_ff)
        _FP_BENCHMARKS = _FP_DATA.get("benchmarks", [])
        _FP_BY_NAME = _FP_DATA.get("by_name", {})
        _FP_BY_CONTENT = _FP_DATA.get("by_content", {})
        _FP_BM_PRIOR = _FP_DATA.get("bm_prior_mean", {})
        _FP_GLOBAL_MEAN = float(_FP_DATA.get("global_mean", 0.664))
        print(
            f"[model] fingerprint loaded: {len(_FP_BY_NAME)} subjects × "
            f"{len(_FP_BENCHMARKS)} benchmarks",
            flush=True,
        )
    except Exception as _e:
        print(f"[model] fingerprint unavailable: {_e}", flush=True)
        _FP_DATA = None

_MIXTURE_WEIGHTS: list[tuple[str, float]] | None = None  # [(benchmark_id, weight)]

# Per-benchmark logit-shift calibration of mixture predictions.
# The per-round label budget is K * m where K=5 and m = number of data
# categories in the round (typically 10-20), so the labeled list has 50-100
# entries, NOT 5. That's enough to fit a simple per-benchmark calibration on
# top of the mixture's posterior-weighted output.
_CALIB_SHIFTS: dict[str, float] = {}
_CALIB_PRIOR_K: float = 4.0  # shrinkage strength toward zero shift


def _fp_lookup(subject_content: str) -> dict[str, float] | None:
    """Return the {benchmark_id: pass_rate} dict for a subject, or None."""
    if not _FP_DATA:
        return None
    if subject_content in _FP_BY_CONTENT:
        return _FP_BY_CONTENT[subject_content]
    first_line = subject_content.split("\n")[0] if subject_content else ""
    name = first_line.replace("Name:", "").strip()
    if name in _FP_BY_NAME:
        return _FP_BY_NAME[name]
    return None


def _fp_prob(subject_fp: dict[str, float] | None, benchmark: str) -> float:
    """Subject's pass rate on `benchmark`, falling back to the benchmark prior."""
    if subject_fp is not None and benchmark in subject_fp:
        return subject_fp[benchmark]
    return _FP_BM_PRIOR.get(benchmark, _FP_GLOBAL_MEAN)


def _update_mixture_weights(labeled: list[dict]) -> None:
    """Compute posterior P(test_bm = b | labels) over training benchmarks `b`."""
    global _MIXTURE_WEIGHTS

    if not labeled or not _FP_DATA or not _FP_BENCHMARKS:
        _MIXTURE_WEIGHTS = None
        return

    # Pre-resolve labeled examples to (fp, label) pairs
    resolved = []
    for ex in labeled:
        try:
            fp = _fp_lookup(ex.get("subject_content", ""))
            l = int(ex["label"])
            resolved.append((fp, l))
        except Exception:
            continue
    if not resolved:
        _MIXTURE_WEIGHTS = None
        return

    log_weights = []
    for bm in _FP_BENCHMARKS:
        ll = 0.0
        for fp, l in resolved:
            p = _fp_prob(fp, bm)
            p = max(0.05, min(0.95, p))
            ll += math.log(p) if l == 1 else math.log(1.0 - p)
        log_weights.append(ll)

    max_lw = max(log_weights)
    raw_weights = [math.exp(lw - max_lw) for lw in log_weights]
    total = sum(raw_weights)
    if total <= 0:
        _MIXTURE_WEIGHTS = None
        return
    weights = [w / total for w in raw_weights]
    _MIXTURE_WEIGHTS = list(zip(_FP_BENCHMARKS, weights))


def _predict_via_mixture(input_dict: dict) -> float | None:
    """Posterior-weighted prediction. Returns None if fingerprint is unavailable."""
    if _MIXTURE_WEIGHTS is None or not _FP_DATA:
        return None
    fp = _fp_lookup(input_dict.get("subject_content", ""))
    # Even if the test subject has no fingerprint entries, we can still use
    # benchmark priors weighted by the mixture. That's strictly better than
    # the global mean baseline. So don't return None for missing fp.
    pred = 0.0
    for bm, w in _MIXTURE_WEIGHTS:
        pred += w * _fp_prob(fp, bm)
    return pred


def _update_calib_shifts(labeled: list[dict]) -> None:
    """Fit a per-benchmark logit shift to correct systematic mixture bias.

    For each benchmark with >=2 labeled examples (mixed labels), find the
    shift that minimizes BCE on those labels:
        P_calibrated = sigmoid(logit(P_mix) + shift_b)
    Shrunk toward 0 by sample size: shift_b ← (n / (n + K)) * shift_raw.
    """
    global _CALIB_SHIFTS
    _CALIB_SHIFTS = {}
    if not labeled or _MIXTURE_WEIGHTS is None:
        return

    by_benchmark: dict[str, list[dict]] = {}
    for ex in labeled:
        bm = ex.get("benchmark", "")
        by_benchmark.setdefault(bm, []).append(ex)

    for bm, exs in by_benchmark.items():
        # Get mixture predictions for these labeled examples
        logits, labels = [], []
        for ex in exs:
            try:
                p = _predict_via_mixture(ex)
                if p is None:
                    continue
                p = max(0.001, min(0.999, p))
                logits.append(math.log(p / (1.0 - p)))
                labels.append(int(ex["label"]))
            except Exception:
                continue

        n = len(logits)
        if n < 2 or len(set(labels)) < 2:
            continue

        # Find shift minimizing BCE via gradient descent
        shift = 0.0
        lr = 0.5
        for _ in range(50):
            grad = sum(
                1.0 / (1.0 + math.exp(-(lg + shift))) - l
                for lg, l in zip(logits, labels)
            )
            shift -= lr * grad / n

        # Shrinkage toward 0 (no shift)
        w = n / (n + _CALIB_PRIOR_K)
        _CALIB_SHIFTS[bm] = w * shift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_theta(subject_content: str) -> float:
    if subject_content in _THETA_BY_CONTENT:
        return _THETA_BY_CONTENT[subject_content]
    first_line = subject_content.split("\n")[0] if subject_content else ""
    name = first_line.replace("Name:", "").strip()
    if name in _THETA_BY_ID:
        return _THETA_BY_ID[name]
    return _MEAN_THETA


def _get_item_ab(item_content: str, benchmark: str, condition: str) -> tuple[float, float]:
    """Predict (a, b) for an item from the IRT bridge. Returns (1.0, 0.0) if unavailable."""
    if not _HAS_IRT_BRIDGE or not item_content:
        return 1.0, 0.0
    key = f"{benchmark}\t{condition}\t{item_content}"
    if key in _ITEM_IRT_CACHE:
        return _ITEM_IRT_CACHE[key]
    import torch
    import numpy as np
    emb_text = f"Benchmark: {benchmark}\nCondition: {condition}\n{item_content}"
    emb = _ENCODER.encode([emb_text], convert_to_numpy=True).astype(np.float32)
    with torch.no_grad():
        out = _IRT_BRIDGE_MODEL(torch.from_numpy(emb)).numpy().squeeze()
    log_a_mean, log_a_std, b_mean, b_std = _IRT_NORM
    log_a = float(out[0]) * log_a_std + log_a_mean
    b = float(out[1]) * b_std + b_mean
    # Clip to training range
    log_a = max(min(log_a, math.log(6.0)), math.log(0.1))
    a = math.exp(log_a)
    b = max(min(b, 5.0), -5.0)
    _ITEM_IRT_CACHE[key] = (a, b)
    return a, b


def _get_item_diff(item_content: str, benchmark: str, condition: str) -> float:
    """Fallback: predict residual logit difficulty from the diff bridge."""
    if not _HAS_DIFF_MODEL or not item_content:
        return 0.0
    key = f"{benchmark}\t{condition}\t{item_content}"
    if key in _ITEM_DIFF_CACHE:
        return _ITEM_DIFF_CACHE[key]
    import torch
    import numpy as np
    emb_text = f"Benchmark: {benchmark}\nCondition: {condition}\n{item_content}"
    emb = _ENCODER.encode([emb_text], convert_to_numpy=True).astype(np.float32)
    with torch.no_grad():
        offset = float(_DIFF_MODEL(torch.from_numpy(emb)).item())
    _ITEM_DIFF_CACHE[key] = offset
    return offset


def _base_logit(input_dict: dict) -> float:
    """The IRT-side logit (without alpha) for a given (subject, item).

    Uses the DIFF bridge (not the IRT bridge). Empirically the diff bridge
    has val_mse=3.3 (predictions near zero) — those small, conservative
    outputs add real ranking signal (sub #11 AUC 0.65) without misranking.
    The IRT bridge's val_mse=0.79 looks better but its more-confident
    cold-start predictions actively misorder items (sub #14 AUC 0.57).
    """
    theta = _get_theta(input_dict.get("subject_content", ""))
    if _HAS_DIFF_MODEL:
        benchmark = input_dict.get("benchmark", "")
        condition = input_dict.get("condition", "none")
        item_content = input_dict.get("item_content", "")
        return theta + _get_item_diff(item_content, benchmark, condition)
    return theta


def _estimate_alpha(labeled: list[dict]) -> float:
    """Fit alpha (additive logit shift) on a set of labeled examples by MLE."""
    if not labeled:
        return _DEFAULT_ALPHA

    logits, labels = [], []
    for ex in labeled:
        try:
            base = _base_logit(ex)
            l = int(ex["label"])
            logits.append(base)
            labels.append(l)
        except Exception:
            continue

    n = len(logits)
    if n == 0:
        return _DEFAULT_ALPHA
    if n < 2 or len(set(labels)) < 2:
        mean_label = max(0.05, min(0.95, sum(labels) / n))
        mean_logit = sum(logits) / n
        return float(math.log(mean_label / (1.0 - mean_label)) - mean_logit)

    mean_logit = sum(logits) / n
    mean_label = sum(labels) / n
    alpha = math.log(max(mean_label, 0.05) / max(1.0 - mean_label, 0.05)) - mean_logit
    lr = 0.5
    for _ in range(50):
        grad = sum(
            1.0 / (1.0 + math.exp(-(lg + alpha))) - l
            for lg, l in zip(logits, labels)
        )
        alpha -= lr * grad / n
    return float(alpha)


def _update_alphas(labeled: list[dict]) -> None:
    global _ROUND_ALPHAS, _ROUND_GLOBAL_ALPHA

    by_benchmark: dict[str, list[dict]] = {}
    for ex in labeled:
        bm = ex.get("benchmark", "")
        by_benchmark.setdefault(bm, []).append(ex)

    _ROUND_GLOBAL_ALPHA = _estimate_alpha(labeled)
    global_alpha = _ROUND_GLOBAL_ALPHA if _ROUND_GLOBAL_ALPHA is not None else _DEFAULT_ALPHA
    _ROUND_ALPHAS = {}
    for bm, exs in by_benchmark.items():
        n = len(exs)
        bm_raw = _estimate_alpha(exs)
        w = n / (n + _BM_ALPHA_PRIOR_K)
        _ROUND_ALPHAS[bm] = w * bm_raw + (1.0 - w) * global_alpha


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(subject answers item correctly) in [0, 1].

    Primary: Bayesian mixture over training benchmarks using the per-subject
    fingerprint. Falls back to IRT theta + global alpha if fingerprint can't
    be evaluated.
    """
    global _ROUND_LABELED_LEN

    labeled = labeled or []
    if len(labeled) != _ROUND_LABELED_LEN:
        _update_alphas(labeled)
        _update_mixture_weights(labeled)
        _update_calib_shifts(labeled)
        _ROUND_LABELED_LEN = len(labeled)

    # Try mixture model first
    p = _predict_via_mixture(input)

    if p is not None:
        # Apply per-benchmark logit-shift calibration (if any) on the mixture
        benchmark = input.get("benchmark", "")
        if benchmark in _CALIB_SHIFTS:
            shift = _CALIB_SHIFTS[benchmark]
            p = max(0.001, min(0.999, p))
            logit = math.log(p / (1.0 - p)) + shift
            p = 1.0 / (1.0 + math.exp(-logit))
    else:
        # Fallback: IRT theta + global alpha
        base = _base_logit(input)
        alpha = _ROUND_GLOBAL_ALPHA if _ROUND_GLOBAL_ALPHA is not None else _DEFAULT_ALPHA
        p = 1.0 / (1.0 + math.exp(-(base + alpha)))

    # 3PL-style guessing/ceiling floor. Most test items are MCQ — clipping
    # prevents catastrophic log-loss on overconfident wrong predictions.
    return float(max(0.10, min(0.90, p)))
