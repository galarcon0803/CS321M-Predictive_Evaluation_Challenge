# Predictive Evaluation Challenge

Stanford CS 321M | AI Measurement Science

Predicts the probability that a language model correctly answers a benchmark question, given text descriptions of the model and question plus a small labeled set revealed per round.

---

## Method

**1. Joint 2PL IRT for subject ability**
Fits θ_s for 909 subjects jointly with per-item (a, b) on 3.58M training responses via SGD.

**2. Behavioral fingerprint matrix M[s, b]**
Per-(subject, benchmark) empirical pass-rate matrix from training responses with Beta-Binomial shrinkage (κ = 5).

**3. Bayesian mixture over training benchmarks**
K · m labeled examples per round (K = 5 per data category, m ≈ 10–20) determine a posterior w_b over training benchmarks. Prediction:

```
P_mix(s) = Σ_b  w_b · M[s, b]
```

**4. Per-benchmark logit-shift calibration + 3PL clip**
For each test benchmark with ≥ 2 mixed-label examples, fit a scalar shift s_b\* by MLE with shrinkage (κ_s = 4):

```
P_final = clip( σ(logit(P_mix) + s_b*),  0.10, 0.90 )
```

---

## Repository structure

```
submission/                          
  model.py                              # predict() — mixture + calibration + clip
  labeling.py                           # acquisition_function() — uncertainty sampling
  subject_theta.json                    # theta_s for 909 subjects
  subject_benchmark_fingerprint.json    # M[s, b] pass-rate matrix
  item_difficulty_bridge.pt             # Diff bridge weights (fallback only)
  item_difficulty_meta.json
  irt_bridge.pt                         # IRT bridge weights (bundled, unused in primary path)
  irt_bridge_meta.json
  models.txt                            # sentence-transformers/all-MiniLM-L6-v2
  requirements.txt

offline_training/
  00_download_data.py                   # Download training data from HuggingFace
  07_fetch_external_data.py             # Fetch 130K external items
  08_build_fingerprint.py               # Build M[s, b] from responses.parquet
  modal_train.py                        # Joint 2PL IRT + bridge training on Modal A100
  06_package_submission.py              # Package my_submission.zip
```

---

## Reproducing

```bash
# 1. Fetch external data (CPU)
python offline_training/07_fetch_external_data.py

# 2. Build fingerprint matrix (CPU, ~10 seconds)
python offline_training/08_build_fingerprint.py

# 3. Train IRT + bridges (Modal A100)
modal run offline_training/modal_train.py

# 4. Package and smoke-test
python offline_training/06_package_submission.py
python starting_kit/tools/run_smoke_test.py my_submission/
```

---

## Data

- **Internal:** `aims-foundations/measurement-db` — 3.58M responses, 909 subjects, 70,873 items, 16 benchmarks
- **External (130K items):** HellaSwag, WinoGrande, BBH, GSM8K, MMLU, ARC — diff bridge training only

---

## Dependencies

```
sentence-transformers
torch
numpy
scikit-learn
scipy
modal          # offline training only
pandas         # offline training only
pyarrow        # offline training only
```
