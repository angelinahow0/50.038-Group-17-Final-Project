#!/usr/bin/env python3
"""
run_training.py  —  Person 2 · Model training pipeline
========================================================
Match-and-Go! · Group 17 · 50.038

Orchestrates all Person 2 steps in order:
  Step 1  Generate synthetic users      →  preprocessing/data/synthetic/users_5k.csv
  Step 2  Generate training labels      →  preprocessing/data/synthetic/training_labels.csv
           (runs V1 over all 5k users)     preprocessing/data/synthetic/features.csv
  Step 3  Train/val/test split          →  preprocessing/data/synthetic/train.csv / val.csv / test.csv
  Step 4  Train XGBoost + LightGBM
  Step 5  Select best model + export   →  models/v2_ranker.pkl
  Step 6  Final evaluation (V1 vs V2)  →  evaluation/results/final_metrics.csv
  Step 7  Ablation study               →  evaluation/results/ablation.csv
                                           evaluation/results/ablation_chart.png

Prerequisites (must exist before running this script):
  preprocessing/data/merged/city_master.csv  ← produced by run_pipeline.py  (Person 1)

Usage:
  # Full run — first time (~5–10 min):
  python run_training.py

  # Labels already exist, skip to training:
  python run_training.py --skip-labels

  # Labels + split exist, jump straight to training:
  python run_training.py --skip-labels --skip-split

  # Skip the 7-model ablation study (saves ~5 min):
  python run_training.py --skip-ablation
"""

from __future__ import annotations
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Canonical paths ────────────────────────────────────────────────────────────
DEST_CSV       = os.path.join(ROOT, "preprocessing", "data", "merged",    "city_master.csv")
DATA_SYN       = os.path.join(ROOT, "preprocessing", "data", "synthetic")
USERS_CSV      = os.path.join(DATA_SYN, "users_5k.csv")
LABELS_CSV     = os.path.join(DATA_SYN, "training_labels.csv")
FEATURES_CSV   = os.path.join(DATA_SYN, "features.csv")
TRAIN_CSV      = os.path.join(DATA_SYN, "train.csv")
VAL_CSV        = os.path.join(DATA_SYN, "val.csv")
TEST_CSV       = os.path.join(DATA_SYN, "test.csv")
MODEL_PKL      = os.path.join(ROOT, "models",      "v2_ranker.pkl")
EVAL_DIR       = os.path.join(ROOT, "evaluation",  "results")
METRICS_CSV    = os.path.join(EVAL_DIR, "final_metrics.csv")
ABLATION_CSV   = os.path.join(EVAL_DIR, "ablation.csv")
ABLATION_CHART = os.path.join(EVAL_DIR, "ablation_chart.png")

FEATURES = [
    "budget_score", "climate_score", "activity_score",
    "attraction_score", "visual_score",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def header(msg: str) -> None:
    print(f"\n{'═'*60}\n  {msg}\n{'═'*60}")


def check_prerequisites() -> None:
    if not os.path.exists(DEST_CSV):
        print(f"\n[ERROR] destinations file not found: {DEST_CSV}")
        print("  → Run   python run_pipeline.py   first (Person 1 preprocessing).")
        sys.exit(1)
    n = len(pd.read_csv(DEST_CSV))
    print(f"  ✓ Found city_master.csv  ({n} cities)")


# ── Step 1: Synthetic users ────────────────────────────────────────────────────

def step_generate_users() -> None:
    header("Step 1 — Generate synthetic users")
    if os.path.exists(USERS_CSV):
        n = len(pd.read_csv(USERS_CSV))
        print(f"  ✓ users_5k.csv already exists ({n:,} rows) — skipping")
        return

    import json, random
    rng = np.random.default_rng(42)
    random.seed(42)

    BUDGETS    = ["<50", "50-100", "100-200", "200+"]
    ACTIVITIES = ["beach", "culture", "food", "nature",
                  "adventure", "wellness", "urban", "wildlife"]
    CLIMATES   = ["warm", "mild", "cool", "snowy", "any"]
    DURATIONS  = [3, 7, 14, 30]
    VISUALS    = ["tropical", "urban", "historical", "any"]

    users = []
    for i in range(5000):
        users.append({
            "user_id":       f"u{i:05d}",
            "budget_tier":   str(rng.choice(BUDGETS,  p=[0.18, 0.35, 0.28, 0.19])),
            "activities":    json.dumps(random.sample(ACTIVITIES, int(rng.integers(1, 4)))),
            "travel_month":  int(rng.integers(1, 13)),
            "climate_pref":  str(rng.choice(CLIMATES, p=[0.38, 0.29, 0.17, 0.08, 0.08])),
            "trip_duration": int(rng.choice(DURATIONS, p=[0.12, 0.42, 0.28, 0.18])),
            "visual_pref":   str(rng.choice(VISUALS,  p=[0.30, 0.25, 0.30, 0.15])),
        })

    os.makedirs(DATA_SYN, exist_ok=True)
    df = pd.DataFrame(users)
    df.to_csv(USERS_CSV, index=False)
    print(f"  ✓ Generated {len(df):,} users → {USERS_CSV}")


# ── Step 2: Training labels ────────────────────────────────────────────────────

def step_generate_labels() -> None:
    header("Step 2 — Generate training labels via V1 baseline")
    if not os.path.exists(DEST_CSV):
        print(f"  ✗ city_master.csv not found. Run run_pipeline.py first.")
        sys.exit(1)

    import json
    from tqdm import tqdm
    from models.baseline import TravelRecommender

    dest  = pd.read_csv(DEST_CSV)
    users = pd.read_csv(USERS_CSV)
    n_dest = len(dest)
    print(f"  {len(users):,} users × {n_dest} destinations = {len(users)*n_dest:,} rows")

    model   = TravelRecommender(dest)
    records = []

    for _, u in tqdm(users.iterrows(), total=len(users), desc="  Scoring", unit="user"):
        try:
            acts = json.loads(u["activities"])
        except Exception:
            acts = []

        results = model.recommend(
            budget_range       = u["budget_tier"],
            activities         = acts,
            travel_month       = int(u["travel_month"]),
            climate_preference = u["climate_pref"],
            trip_duration      = int(u["trip_duration"]),
            visual_pref        = u.get("visual_pref", "any"),
            top_n              = n_dest,
        )

        for rank_pos, (_, row) in enumerate(results.iterrows(), start=1):
            records.append({
                "user_id":          u["user_id"],
                "city":             row.get("city", ""),
                "rank_pos":         rank_pos,
                "v1_score":         row.get("composite_score",  0.0),
                "budget_score":     row.get("budget_score",     0.0),
                "climate_score":    row.get("climate_score",    0.0),
                "activity_score":   row.get("activity_score",   0.0),
                "attraction_score": row.get("attraction_score", 0.0),
                "visual_score":     row.get("visual_score",     0.0),
            })

    labels = pd.DataFrame(records)

    # Sanity check
    bad = 0
    for uid, grp in labels.groupby("user_id"):
        s = grp.sort_values("rank_pos")
        if len(s) >= 2 and s.iloc[0]["v1_score"] < s.iloc[1]["v1_score"] - 1e-6:
            bad += 1
    status = f"⚠ {bad} ordering issues" if bad else "✓ rank ordering verified"
    print(f"  {status}")

    labels.to_csv(LABELS_CSV, index=False)
    print(f"  ✓ Saved {len(labels):,} rows → {LABELS_CSV}")

    feat_cols = ["user_id", "city", "rank_pos", "v1_score"] + FEATURES
    labels[feat_cols].to_csv(FEATURES_CSV, index=False)
    print(f"  ✓ Saved features           → {FEATURES_CSV}")


# ── Step 3: Split ─────────────────────────────────────────────────────────────

def step_split() -> None:
    header("Step 3 — Train / val / test split  (80 / 10 / 10 by user_id)")
    from sklearn.model_selection import train_test_split

    labels  = pd.read_csv(LABELS_CSV)
    users   = labels["user_id"].unique().tolist()
    tr_u, tmp = train_test_split(users, test_size=0.20, random_state=42)
    va_u, te_u = train_test_split(tmp,  test_size=0.50, random_state=42)

    train = labels[labels["user_id"].isin(tr_u)]
    val   = labels[labels["user_id"].isin(va_u)]
    test  = labels[labels["user_id"].isin(te_u)]

    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(  VAL_CSV,   index=False)
    test.to_csv( TEST_CSV,  index=False)

    print(f"  Train : {len(tr_u):,} users  ({len(train):,} rows)")
    print(f"  Val   : {len(va_u):,} users  ({len(val):,} rows)")
    print(f"  Test  : {len(te_u):,} users  ({len(test):,} rows)")
    overlap = len(set(tr_u) & set(va_u) & set(te_u))
    print(f"  ✓ Zero user overlap: {overlap} shared users")


# ── Step 4: Train ─────────────────────────────────────────────────────────────

def _add_relevance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw = df.groupby("user_id")["rank_pos"].transform("max") - df["rank_pos"] + 1
    df["relevance"] = raw.clip(upper=30).astype(int)
    return df


def _build_Xy(df: pd.DataFrame):
    df = _add_relevance(df)
    X  = df[FEATURES].fillna(0.0)
    y  = df["relevance"]
    g  = df.groupby("user_id", sort=False).size().values
    return X, y, g, df


def _val_ndcg(model, X_va: pd.DataFrame, val_eval: pd.DataFrame) -> float:
    from evaluation.evaluate import ndcg_at_k
    scores = model.predict(X_va)
    val_eval = val_eval.copy()
    val_eval["pred"] = scores
    ndcgs = []
    for _, grp in val_eval.groupby("user_id"):
        pred_order = grp.sort_values("pred", ascending=False)["relevance"].tolist()
        ideal      = sorted(pred_order, reverse=True)
        ndcgs.append(ndcg_at_k(pred_order, ideal, k=5))
    return float(np.mean(ndcgs))


def step_train(val_df: pd.DataFrame) -> tuple:
    header("Step 4 — V2 Model training")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"  Train rows : {len(train_df):,}")
    print(f"  Val   rows : {len(val_df):,}")

    X_tr, y_tr, g_tr, _      = _build_Xy(train_df)
    X_va, y_va, g_va, va_eval = _build_Xy(val_df)

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("\n  ── XGBoost  (rank:ndcg) ──")
    import xgboost as xgb
    t0 = time.time()
    xgb_model = xgb.XGBRanker(
        objective="rank:ndcg", n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="ndcg@5",
        early_stopping_rounds=20, verbosity=0,
    )
    xgb_model.fit(
        X_tr, y_tr, group=g_tr,
        eval_set=[(X_va, y_va)], eval_group=[g_va],
        verbose=False,
    )
    xgb_ndcg = _val_ndcg(xgb_model, X_va, va_eval)
    print(f"  Val NDCG@5 = {xgb_ndcg:.4f}  ({time.time()-t0:.0f}s)")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    print("\n  ── LightGBM  (lambdarank) ──")
    import lightgbm as lgb
    t0 = time.time()
    tr_ds = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
    va_ds = lgb.Dataset(X_va, label=y_va, group=g_va, reference=tr_ds)
    lgb_model = lgb.train(
        {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
         "num_leaves": 31, "learning_rate": 0.05, "verbose": -1},
        tr_ds, num_boost_round=300, valid_sets=[va_ds],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_ndcg = _val_ndcg(lgb_model, X_va, va_eval)
    print(f"  Val NDCG@5 = {lgb_ndcg:.4f}  ({time.time()-t0:.0f}s)")

    return xgb_model, lgb_model, xgb_ndcg, lgb_ndcg


# ── Step 5: Export ────────────────────────────────────────────────────────────

def step_export(xgb_model, lgb_model, xgb_ndcg: float, lgb_ndcg: float):
    header("Step 5 — Select best model and export")
    if xgb_ndcg >= lgb_ndcg:
        best, name = xgb_model, f"XGBoost  (NDCG@5={xgb_ndcg:.4f})"
    else:
        best, name = lgb_model, f"LightGBM (NDCG@5={lgb_ndcg:.4f})"
    print(f"  Winner: {name}")
    os.makedirs(os.path.dirname(MODEL_PKL), exist_ok=True)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(best, f)
    print(f"  ✓ Saved → {MODEL_PKL}")
    return best


# ── Step 6: Final evaluation ──────────────────────────────────────────────────

def step_final_eval(best_model, test_df: pd.DataFrame) -> None:
    header("Step 6 — Final evaluation on held-out test set")
    from evaluation.evaluate import evaluate_model, evaluate_v1

    _, _, _, te_eval = _build_Xy(test_df)

    v1 = evaluate_v1(te_eval)
    v2 = evaluate_model(best_model, te_eval)

    print(f"\n  {'Metric':<12} {'V1 Baseline':>13} {'V2 Ranker':>11} {'Δ':>8}")
    print(f"  {'─'*48}")
    rows = []
    for key in ["ndcg@5", "mrr", "p@3"]:
        s1, s2 = v1.get(key, 0.0), v2.get(key, 0.0)
        pct = (s2 - s1) / (s1 + 1e-10) * 100
        sign = "+" if pct >= 0 else ""
        print(f"  {key:<12} {s1:>13.4f} {s2:>11.4f} {sign}{pct:>6.1f}%")
        rows.append({"metric": key, "v1": s1, "v2": s2, "change_pct": pct})

    os.makedirs(EVAL_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
    print(f"\n  ✓ Saved → {METRICS_CSV}")


# ── Step 7: Ablation ──────────────────────────────────────────────────────────

def step_ablation(train_df: pd.DataFrame, val_df: pd.DataFrame,
                   test_df: pd.DataFrame) -> None:
    header("Step 7 — Ablation study  (6 model trains)")
    import xgboost as xgb
    from evaluation.evaluate import evaluate_model

    conditions = [None] + FEATURES
    results = []

    for zeroed in conditions:
        label = "Full model" if zeroed is None else f"No {zeroed.replace('_score','')}"
        print(f"\n  {label} ...", end=" ", flush=True)
        t0 = time.time()

        tr = train_df.copy(); va = val_df.copy(); te = test_df.copy()
        if zeroed:
            for d in (tr, va, te):
                d[zeroed] = 0.0

        X_tr, y_tr, g_tr, _       = _build_Xy(tr)
        X_va, y_va, g_va, _       = _build_Xy(va)
        _, _, _, te_eval           = _build_Xy(te)

        m = xgb.XGBRanker(
            objective="rank:ndcg", n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42,
            early_stopping_rounds=20, verbosity=0,
        )
        m.fit(X_tr, y_tr, group=g_tr,
              eval_set=[(X_va, y_va)], eval_group=[g_va],
              verbose=False)

        metrics = evaluate_model(m, te_eval)
        results.append({"condition": label, **metrics})
        print(f"NDCG@5={metrics['ndcg@5']:.4f}  ({time.time()-t0:.0f}s)")

    df_ab = pd.DataFrame(results)
    df_ab.to_csv(ABLATION_CSV, index=False)
    print(f"\n  ✓ Saved → {ABLATION_CSV}")

    # Bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#27ae60"] + ["#e74c3c"] * (len(df_ab) - 1)
        bars = ax.bar(df_ab["condition"], df_ab["ndcg@5"],
                      color=colors, width=0.6, edgecolor="white")
        for bar, v in zip(bars, df_ab["ndcg@5"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title("Ablation Study — NDCG@5 by Feature Removed",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("NDCG@5")
        ax.set_ylim(0, df_ab["ndcg@5"].max() * 1.18)
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(ABLATION_CHART, dpi=150)
        plt.close()
        print(f"  ✓ Chart  → {ABLATION_CHART}")
    except Exception as e:
        print(f"  ⚠ Chart skipped: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(skip_labels=False, skip_split=False, skip_ablation=False) -> None:
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Match-and-Go! · V2 Training Pipeline     ║")
    print("║   Person 2 · Group 17 · 50.038             ║")
    print("╚══════════════════════════════════════════════╝")
    t_start = time.time()

    check_prerequisites()

    step_generate_users()

    if not skip_labels:
        step_generate_labels()
    else:
        print("\n[skip] Label generation  (--skip-labels)")

    if not skip_split:
        step_split()
    else:
        print("[skip] Train/val/test split  (--skip-split)")

    val_df  = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    xgb_m, lgb_m, xgb_n, lgb_n = step_train(val_df)
    best = step_export(xgb_m, lgb_m, xgb_n, lgb_n)

    step_final_eval(best, test_df)

    if not skip_ablation:
        train_df = pd.read_csv(TRAIN_CSV)
        step_ablation(train_df, val_df, test_df)
    else:
        print("\n[skip] Ablation study  (--skip-ablation)")

    print(f"\n{'═'*60}")
    print(f"  All done ✓  ({time.time()-t_start:.0f}s)")
    print(f"  Model   → {MODEL_PKL}")
    print(f"  Metrics → {METRICS_CSV}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match-and-Go V2 training pipeline")
    parser.add_argument("--skip-labels",   action="store_true")
    parser.add_argument("--skip-split",    action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()
    main(args.skip_labels, args.skip_split, args.skip_ablation)