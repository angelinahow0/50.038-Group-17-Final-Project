"""
models/ranker.py
================
V2 Learned Ranker — LambdaRank / XGBoost
Person 2 ownership: Match-and-Go! · Group 17 · 50.038

Trained on 5,000 synthetic user profiles labelled by V1 baseline.
Uses five sub-scores as features and learns non-linear interactions.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Feature columns (must match training) ────────────────────────────────────
FEATURES = [
    "budget_score",
    "climate_score",
    "activity_score",
    "attraction_score",
    "visual_score",
]

MODEL_PATH = Path(__file__).parent / "v2_ranker.pkl"


# ── Predict function (public API) ────────────────────────────────────────────

def predict(
    user_profile: dict,
    destinations_df: pd.DataFrame,
    model=None,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Rank destinations for a given user profile using the V2 learned ranker.

    Parameters
    ----------
    user_profile : dict with keys:
        budget_tier     : '<50' | '50-100' | '100-200' | '200+'
        activities      : list[str]  e.g. ['beach', 'food']
        travel_month    : int 1-12
        climate_pref    : 'warm' | 'mild' | 'cool' | 'snowy' | 'any'
        trip_duration   : int (days)
        visual_pref     : 'tropical' | 'urban' | 'historical' | 'any'  (optional)
    destinations_df : DataFrame produced by Person 1
    model           : trained ranker object; loads from disk if None
    top_n           : number of results to return

    Returns
    -------
    DataFrame sorted by v2_score descending, with all sub-score columns.
    """
    from models.baseline import TravelRecommender  # local import to avoid circulars

    if model is None:
        model = load_model()

    # Step 1 — score all destinations with V1 to get sub-scores
    rec = TravelRecommender(destinations_df)
    scored = rec.recommend(
        budget_range       = user_profile.get("budget_tier", "50-100"),
        activities         = user_profile.get("activities", []),
        travel_month       = int(user_profile.get("travel_month", 6)),
        climate_preference = user_profile.get("climate_pref", "any"),
        trip_duration      = int(user_profile.get("trip_duration", 7)),
        visual_pref        = user_profile.get("visual_pref", "any"),
        top_n              = len(destinations_df),
    )

    # Step 2 — build feature matrix
    X = scored[FEATURES].fillna(0.0)

    # Step 3 — predict with V2 model
    raw_scores = model.predict(X)

    # Step 4 — attach scores and sort (higher predicted score = better rank)
    scored = scored.copy()
    scored["v2_score"] = raw_scores
    scored["composite_score"] = scored["v2_score"]   # keep UI column name consistent

    result = scored.sort_values("v2_score", ascending=False).reset_index(drop=True)
    return result.head(top_n)


# ── Model persistence ─────────────────────────────────────────────────────────

def load_model(path: Optional[Path] = None):
    """Load the pickled V2 ranker from disk."""
    p = Path(path) if path else MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"V2 model not found at {p}. "
            "Run run_pipeline.py to train the model first."
        )
    with open(p, "rb") as f:
        return pickle.load(f)


def save_model(model, path: Optional[Path] = None) -> None:
    """Pickle the trained model to disk."""
    p = Path(path) if path else MODEL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(model, f)
    print(f"[ranker] Model saved → {p}")


# ── Training helpers (called from run_pipeline.py) ───────────────────────────

def build_feature_matrix(labels_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Convert training_labels.csv into (X, y, groups) for ranking models.

    y is the *inverse* rank position so that higher relevance = higher score,
    which is what XGBoost rank:ndcg and LightGBM lambdarank expect.
    """
    df = labels_df.copy()

    # Inverse rank: rank 1 (best) gets highest label
    df["relevance"] = (
        df.groupby("user_id")["rank_pos"]
          .transform("max") - df["rank_pos"] + 1
    ).astype(int)

    X      = df[FEATURES].fillna(0.0)
    y      = df["relevance"]
    groups = df.groupby("user_id", sort=False).size().values

    return X, y, groups


def train_xgboost(X_tr, y_tr, g_tr, X_va, y_va, g_va) -> object:
    """Train an XGBoost LambdaRank model."""
    import xgboost as xgb

    model = xgb.XGBRanker(
        objective        = "rank:ndcg",
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
        eval_metric      = "ndcg@5",
        early_stopping_rounds = 20,
        verbosity        = 0,
    )

    model.fit(
        X_tr, y_tr,
        group     = g_tr,
        eval_set  = [(X_va, y_va)],
        eval_group= [g_va],
        verbose   = False,
    )

    best_score = model.best_score
    print(f"[XGBoost] Best val NDCG@5 = {best_score:.4f}  "
          f"(round {model.best_iteration})")
    return model


def train_lightgbm(X_tr, y_tr, g_tr, X_va, y_va, g_va) -> object:
    """Train a LightGBM LambdaRank model."""
    import lightgbm as lgb

    train_ds = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
    val_ds   = lgb.Dataset(X_va, label=y_va, group=g_va, reference=train_ds)

    params = {
        "objective"    : "lambdarank",
        "metric"       : "ndcg",
        "ndcg_eval_at" : [5],
        "num_leaves"   : 31,
        "learning_rate": 0.05,
        "n_estimators" : 300,
        "verbose"      : -1,
    }

    model = lgb.train(
        params,
        train_ds,
        num_boost_round  = 300,
        valid_sets       = [val_ds],
        callbacks        = [
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    ndcg5 = model.best_score["valid_0"]["ndcg@5"]
    print(f"[LightGBM] Best val NDCG@5 = {ndcg5:.4f}  "
          f"(round {model.best_iteration})")
    return model


def evaluate_val_ndcg(model, X_va: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """
    Compute mean NDCG@5 on the validation set.

    Works for both XGBoost and LightGBM models.
    """
    from evaluation.evaluate import ndcg_at_k

    scores = model.predict(X_va)
    val_df = val_df.copy()
    val_df["pred_score"] = scores

    ndcg_scores = []
    for uid, grp in val_df.groupby("user_id"):
        grp_sorted   = grp.sort_values("pred_score", ascending=False)
        pred_order   = list(grp_sorted.index)
        true_rel     = dict(zip(grp.index, grp["relevance"]))
        rel_in_order = [true_rel[i] for i in pred_order]
        ideal        = sorted(rel_in_order, reverse=True)
        ndcg_scores.append(ndcg_at_k(rel_in_order, ideal, k=5))

    return float(np.mean(ndcg_scores))


# ── Model card (printed after training) ──────────────────────────────────────

MODEL_CARD = """
╔══════════════════════════════════════════════════════════════╗
║              Match-and-Go! V2 Ranker — Model Card           ║
╠══════════════════════════════════════════════════════════════╣
║  Task          : Learning-to-rank travel destinations        ║
║  Algorithm     : LambdaRank (XGBoost / LightGBM)           ║
║  Features      : budget_score, climate_score,               ║
║                  activity_score, attraction_score,          ║
║                  visual_score  (5 features)                 ║
║  Training data : 5,000 synthetic user profiles              ║
║                  × ~50 destinations = ~250k rows            ║
║  Label source  : V1 weighted baseline (rank positions)      ║
║  Objective     : rank:ndcg / lambdarank                     ║
║  Split         : 80/10/10 train/val/test (by user_id)       ║
║  Output        : models/v2_ranker.pkl                       ║
╚══════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(MODEL_CARD)
    print("[ranker.py] Import this module or run run_pipeline.py to train.")