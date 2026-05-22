"""
Fetch external per-item model response data to extend the item difficulty bridge.

Sources tried (in order):
  1. open-llm-leaderboard/details_* — per-sample correctness for thousands of models
     on MMLU, ARC-Challenge, HellaSwag, WinoGrande, TruthfulQA
  2. allenai/ai2_arc — ARC-Easy vs ARC-Challenge split gives a binary difficulty label
  3. cais/mmlu — 57-category MMLU; uses published per-category avg accuracy as difficulty
  4. google/bigbench — diverse task items; "hard" tasks get a difficulty label

Output: data/external_items.parquet
  columns: item_content (str), pass_rate (float), benchmark_id (str), n_obs (int)

This file is read by modal_train.py Step 2b to extend item difficulty bridge training.

Run:
  python offline_training/07_fetch_external_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

OUT = DATA / "external_items.parquet"

# ---------------------------------------------------------------------------
# Published per-category MMLU accuracy (averaged across ~50 models from
# the Open LLM Leaderboard as of mid-2024). Used as a proxy when per-sample
# data is unavailable.
# ---------------------------------------------------------------------------
MMLU_CATEGORY_ACC: dict[str, float] = {
    "abstract_algebra": 0.38, "anatomy": 0.62, "astronomy": 0.67,
    "business_ethics": 0.64, "clinical_knowledge": 0.69, "college_biology": 0.72,
    "college_chemistry": 0.47, "college_computer_science": 0.55,
    "college_mathematics": 0.40, "college_medicine": 0.63,
    "college_physics": 0.43, "computer_security": 0.74,
    "conceptual_physics": 0.59, "econometrics": 0.47, "electrical_engineering": 0.61,
    "elementary_mathematics": 0.54, "formal_logic": 0.44, "global_facts": 0.48,
    "high_school_biology": 0.77, "high_school_chemistry": 0.53,
    "high_school_computer_science": 0.66, "high_school_european_history": 0.75,
    "high_school_geography": 0.79, "high_school_government_and_politics": 0.86,
    "high_school_macroeconomics": 0.68, "high_school_mathematics": 0.40,
    "high_school_microeconomics": 0.72, "high_school_physics": 0.42,
    "high_school_psychology": 0.83, "high_school_statistics": 0.55,
    "high_school_us_history": 0.82, "high_school_world_history": 0.80,
    "human_aging": 0.74, "human_sexuality": 0.74, "international_law": 0.81,
    "jurisprudence": 0.75, "logical_fallacies": 0.77, "machine_learning": 0.48,
    "management": 0.79, "marketing": 0.86, "medical_genetics": 0.73,
    "miscellaneous": 0.82, "moral_disputes": 0.72, "moral_scenarios": 0.49,
    "nutrition": 0.74, "philosophy": 0.74, "prehistory": 0.77,
    "professional_accounting": 0.53, "professional_law": 0.51,
    "professional_medicine": 0.72, "professional_psychology": 0.70,
    "public_relations": 0.73, "security_studies": 0.73, "sociology": 0.83,
    "us_foreign_policy": 0.85, "virology": 0.56, "world_religions": 0.83,
}


def _render_mcq(question: str, choices: list[str]) -> str:
    """Format MCQ as plain text the same way we format item_content."""
    opts = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(choices))
    return f"{question}\n{opts}"


# ---------------------------------------------------------------------------
# Source 1: Open LLM Leaderboard per-sample details
# ---------------------------------------------------------------------------
def fetch_leaderboard_details(max_models: int = 200) -> pd.DataFrame:
    """
    Try to load per-sample correctness from open-llm-leaderboard/details_*.
    Returns empty DataFrame if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset, get_dataset_config_names
        print("  Checking open-llm-leaderboard/details configs ...", flush=True)
        configs = get_dataset_config_names("open-llm-leaderboard/details")
        print(f"  Found {len(configs)} model configs", flush=True)
    except Exception as e:
        print(f"  open-llm-leaderboard/details unavailable: {e}", flush=True)
        return pd.DataFrame()

    rows = []
    TASKS = {"arc_challenge", "hellaswag", "mmlu", "winogrande", "truthfulqa_mc"}
    for cfg in configs[:max_models]:
        try:
            ds = load_dataset("open-llm-leaderboard/details", cfg,
                              split="latest", trust_remote_code=False)
            df = ds.to_pandas()
            if "task" not in df.columns or "doc" not in df.columns:
                continue
            for task in TASKS:
                sub = df[df["task"].str.contains(task, na=False)]
                if sub.empty:
                    continue
                for _, row in sub.iterrows():
                    doc = row.get("doc", {})
                    correct = row.get("acc", row.get("acc_norm", None))
                    if correct is None or doc is None:
                        continue
                    # Extract item text from doc dict
                    q = doc.get("question", doc.get("query", doc.get("goal", "")))
                    choices = doc.get("choices", doc.get("endings", []))
                    if isinstance(choices, dict):
                        choices = choices.get("text", [])
                    content = _render_mcq(q, choices) if choices else str(q)
                    rows.append({
                        "item_content": content,
                        "model": cfg,
                        "correct": int(float(correct) > 0.5),
                        "benchmark_id": f"external_{task}",
                    })
        except Exception:
            continue
        if len(rows) % 10000 == 0 and rows:
            print(f"    {len(rows):,} rows so far ...", flush=True)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"  Leaderboard details: {len(df):,} rows from {df['model'].nunique()} models",
          flush=True)
    # Aggregate to per-item pass rates
    agg = (
        df.groupby(["item_content", "benchmark_id"])
        .agg(pass_rate=("correct", "mean"), n_obs=("correct", "count"))
        .reset_index()
    )
    print(f"  → {len(agg):,} unique items", flush=True)
    return agg


# ---------------------------------------------------------------------------
# Source 2: ARC (difficulty from easy/challenge split)
# ---------------------------------------------------------------------------
def fetch_arc() -> pd.DataFrame:
    try:
        from datasets import load_dataset
        rows = []
        for split_name, difficulty in [("ARC-Challenge", 0.55), ("ARC-Easy", 0.78)]:
            ds = load_dataset("allenai/ai2_arc", split_name, split="train+validation+test",
                              trust_remote_code=False)
            for ex in ds:
                choices = ex["choices"]["text"]
                content = _render_mcq(ex["question"], choices)
                rows.append({
                    "item_content": content,
                    "pass_rate": difficulty,
                    "n_obs": 0,  # proxy, not real obs count
                    "benchmark_id": f"external_{split_name.lower().replace('-','_')}",
                })
        df = pd.DataFrame(rows).drop_duplicates("item_content")
        print(f"  ARC: {len(df):,} items", flush=True)
        return df
    except Exception as e:
        print(f"  ARC unavailable: {e}", flush=True)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Source 3: MMLU (difficulty from published per-category accuracy)
# ---------------------------------------------------------------------------
def fetch_mmlu() -> pd.DataFrame:
    try:
        from datasets import load_dataset
        rows = []
        for category, acc in MMLU_CATEGORY_ACC.items():
            try:
                ds = load_dataset("cais/mmlu", category,
                                  split="test+validation+dev",
                                  trust_remote_code=False)
                for ex in ds:
                    choices = [ex[k] for k in ["A", "B", "C", "D"] if k in ex]
                    if not choices:
                        choices = ex.get("choices", [])
                    content = _render_mcq(ex["question"], choices)
                    rows.append({
                        "item_content": content,
                        "pass_rate": acc,
                        "n_obs": 0,
                        "benchmark_id": f"external_mmlu_{category}",
                    })
            except Exception:
                continue
        df = pd.DataFrame(rows).drop_duplicates("item_content")
        print(f"  MMLU: {len(df):,} items across {df['benchmark_id'].nunique()} categories",
              flush=True)
        return df
    except Exception as e:
        print(f"  MMLU unavailable: {e}", flush=True)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Source 4: BIG-Bench Hard (repackaged, works with current datasets library)
# 23 hard reasoning tasks, avg model accuracy ~0.35
# ---------------------------------------------------------------------------
BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]

def fetch_bbh() -> pd.DataFrame:
    try:
        from datasets import load_dataset
        rows = []
        for task in BBH_TASKS:
            try:
                ds = load_dataset("lukaemon/bbh", task, split="test",
                                  trust_remote_code=False)
                for ex in ds:
                    inp = str(ex.get("input", ""))
                    if len(inp) < 10:
                        continue
                    rows.append({
                        "item_content": inp,
                        "pass_rate": 0.35,  # BBH is specifically the hard subset
                        "n_obs": 0,
                        "benchmark_id": f"external_bbh_{task}",
                    })
            except Exception:
                continue
        df = pd.DataFrame(rows).drop_duplicates("item_content") if rows else pd.DataFrame()
        if not df.empty:
            print(f"  BIG-Bench Hard: {len(df):,} items across {df['benchmark_id'].nunique()} tasks",
                  flush=True)
        else:
            print("  BIG-Bench Hard: no items fetched", flush=True)
        return df
    except Exception as e:
        print(f"  BIG-Bench Hard unavailable: {e}", flush=True)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Source 5: GSM8K — grade school math reasoning (~0.50 avg accuracy)
# ---------------------------------------------------------------------------
def fetch_gsm8k() -> pd.DataFrame:
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train+test", trust_remote_code=False)
        rows = []
        for ex in ds:
            q = str(ex.get("question", ""))
            if len(q) < 10:
                continue
            rows.append({
                "item_content": q,
                "pass_rate": 0.50,
                "n_obs": 0,
                "benchmark_id": "external_gsm8k",
            })
        df = pd.DataFrame(rows).drop_duplicates("item_content")
        print(f"  GSM8K: {len(df):,} items", flush=True)
        return df
    except Exception as e:
        print(f"  GSM8K unavailable: {e}", flush=True)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Source 6: HellaSwag — commonsense completion (~0.85 avg accuracy)
# ---------------------------------------------------------------------------
def fetch_hellaswag() -> pd.DataFrame:
    try:
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="train+validation",
                          trust_remote_code=False)
        rows = []
        for ex in ds:
            ctx = str(ex.get("ctx", ""))
            endings = ex.get("endings", [])
            if not ctx or not endings:
                continue
            content = _render_mcq(ctx, endings)
            rows.append({
                "item_content": content,
                "pass_rate": 0.85,
                "n_obs": 0,
                "benchmark_id": "external_hellaswag",
            })
        df = pd.DataFrame(rows).drop_duplicates("item_content")
        print(f"  HellaSwag: {len(df):,} items", flush=True)
        return df
    except Exception as e:
        print(f"  HellaSwag unavailable: {e}", flush=True)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Source 7: WinoGrande — commonsense pronoun resolution (~0.75 avg accuracy)
# ---------------------------------------------------------------------------
def fetch_winogrande() -> pd.DataFrame:
    try:
        from datasets import load_dataset
        ds = load_dataset("winogrande", "winogrande_xl", split="train+validation",
                          trust_remote_code=False)
        rows = []
        for ex in ds:
            sentence = str(ex.get("sentence", ""))
            opt1 = str(ex.get("option1", ""))
            opt2 = str(ex.get("option2", ""))
            if not sentence:
                continue
            content = _render_mcq(sentence, [opt1, opt2])
            rows.append({
                "item_content": content,
                "pass_rate": 0.75,
                "n_obs": 0,
                "benchmark_id": "external_winogrande",
            })
        df = pd.DataFrame(rows).drop_duplicates("item_content")
        print(f"  WinoGrande: {len(df):,} items", flush=True)
        return df
    except Exception as e:
        print(f"  WinoGrande unavailable: {e}", flush=True)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_frames: list[pd.DataFrame] = []

    print("\n=== Source 1: Open LLM Leaderboard per-sample details ===", flush=True)
    df1 = fetch_leaderboard_details(max_models=200)
    if not df1.empty:
        all_frames.append(df1)

    print("\n=== Source 2: ARC ===", flush=True)
    df2 = fetch_arc()
    if not df2.empty:
        all_frames.append(df2)

    print("\n=== Source 3: MMLU ===", flush=True)
    df3 = fetch_mmlu()
    if not df3.empty:
        all_frames.append(df3)

    print("\n=== Source 4: BIG-Bench Hard ===", flush=True)
    df4 = fetch_bbh()
    if not df4.empty:
        all_frames.append(df4)

    print("\n=== Source 5: GSM8K ===", flush=True)
    df5 = fetch_gsm8k()
    if not df5.empty:
        all_frames.append(df5)

    print("\n=== Source 6: HellaSwag ===", flush=True)
    df6 = fetch_hellaswag()
    if not df6.empty:
        all_frames.append(df6)

    print("\n=== Source 7: WinoGrande ===", flush=True)
    df7 = fetch_winogrande()
    if not df7.empty:
        all_frames.append(df7)

    if not all_frames:
        print("\nNo external data fetched. Check internet access and try again.")
        raise SystemExit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates("item_content").reset_index(drop=True)

    # Sanity check
    combined["pass_rate"] = combined["pass_rate"].clip(0.02, 0.98)
    combined["n_obs"] = combined["n_obs"].fillna(0).astype(int)

    combined.to_parquet(OUT, index=False)
    print(f"\nSaved {len(combined):,} external items → {OUT}")
    print(f"Benchmarks: {combined['benchmark_id'].nunique()}")
    print(combined.groupby("benchmark_id").size().sort_values(ascending=False).head(20).to_string())
    print("\nNext: update modal_train.py to include external_items.parquet in Step 2b training")
