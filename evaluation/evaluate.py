"""
evaluation/evaluate.py
=======================
Evaluation framework — NDCG@K, MRR, Precision@K
Person 3 ownership (Person 2 depends on this for training metrics)
Match-and-Go! · Group 17 · 50.038
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FEATURES = [
    "budget_score",
    "climate_score",
    "activity_score",
    "attraction_score",
    "visual_score",
]


# ── Core metric functions ────────────────────────────────────────────────────

def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at position k."""
    r = np.array(relevances[:k], dtype=float)
    if r.size == 0:
        return 0.0
    return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))


def ndcg_at_k(actual_relevances: list[float],
               ideal_relevances: list[float],
               k: int = 5) -> float:
    """
    Normalised DCG@k.

    Parameters
    ----------
    actual_relevances : relevance scores in predicted rank order
    ideal_relevances  : relevance scores in ideal (best) order
    k                 : cutoff position
    """
    idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if idcg < 1e-10:
        return 0.0
    return dcg_at_k(actual_relevances, k) / idcg


def mrr(predicted_list: list[Any], relevant_items: set | list) -> float:
    """
    Mean Reciprocal Rank.

    Returns 1/(rank of first relevant item), or 0 if none found in top-K.
    """
    relevant_set = set(relevant_items)
    for i, item in enumerate(predicted_list):
        if item in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(predicted_list: list[Any],
                    relevant_items: list | set,
                    k: int = 3) -> float:
    """Fraction of top-k predicted items that are relevant."""
    top_k = set(predicted_list[:k])
    return len(top_k & set(relevant_items)) / k


# ── Model evaluation wrapper ─────────────────────────────────────────────────

def evaluate_model(model, test_df: pd.DataFrame, k_ndcg: int = 5,
                   k_prec: int = 3) -> dict:
    """
    Evaluate a trained ranker on the test set.

    Parameters
    ----------
    model    : trained XGBoost / LightGBM ranker (has .predict())
    test_df  : DataFrame with FEATURES + user_id + rank_pos columns
    k_ndcg   : NDCG cutoff
    k_prec   : Precision cutoff

    Returns
    -------
    dict with keys: ndcg@5, mrr, p@3, n_users
    """
    ndcg_scores, mrr_scores, prec_scores = [], [], []

    X_all = test_df[FEATURES].fillna(0.0)
    pred_scores = model.predict(X_all)
    test_df = test_df.copy()
    test_df["pred_score"] = pred_scores

    # Add relevance if not already present
    if "relevance" not in test_df.columns:
        test_df["relevance"] = (
            test_df.groupby("user_id")["rank_pos"]
                   .transform("max") - test_df["rank_pos"] + 1
        ).astype(int)

    for uid, grp in test_df.groupby("user_id"):
        grp_pred  = grp.sort_values("pred_score", ascending=False)
        grp_ideal = grp.sort_values("relevance",  ascending=False)

        # NDCG@k
        actual_rel = list(grp_pred["relevance"])
        ideal_rel  = list(grp_ideal["relevance"])
        ndcg_scores.append(ndcg_at_k(actual_rel, ideal_rel, k=k_ndcg))

        # MRR — "relevant" = top-3 destinations by V1 rank
        top3_cities = set(grp_ideal.head(3)["city"])
        pred_cities = list(grp_pred["city"])
        mrr_scores.append(mrr(pred_cities, top3_cities))

        # Precision@k
        prec_scores.append(precision_at_k(pred_cities, top3_cities, k=k_prec))

    return {
        f"ndcg@{k_ndcg}": float(np.mean(ndcg_scores)),
        "mrr":             float(np.mean(mrr_scores)),
        f"p@{k_prec}":    float(np.mean(prec_scores)),
        "n_users":         len(ndcg_scores),
    }


# ── V1 baseline evaluation (rule-based, no model object) ─────────────────────

def evaluate_v1(test_df: pd.DataFrame, k_ndcg: int = 5, k_prec: int = 3) -> dict:
    """
    Evaluate V1 by treating rank_pos directly as the model's ranking.
    rank_pos 1 = best; lower is better, so we invert for scoring.
    """
    ndcg_scores, mrr_scores, prec_scores = [], [], []

    if "relevance" not in test_df.columns:
        test_df = test_df.copy()
        test_df["relevance"] = (
            test_df.groupby("user_id")["rank_pos"]
                   .transform("max") - test_df["rank_pos"] + 1
        ).astype(int)

    for uid, grp in test_df.groupby("user_id"):
        grp_v1    = grp.sort_values("rank_pos", ascending=True)   # rank_pos 1 = best
        grp_ideal = grp.sort_values("relevance", ascending=False)

        actual_rel = list(grp_v1["relevance"])
        ideal_rel  = list(grp_ideal["relevance"])
        ndcg_scores.append(ndcg_at_k(actual_rel, ideal_rel, k=k_ndcg))

        top3_cities = set(grp_ideal.head(3)["city"])
        pred_cities = list(grp_v1["city"])
        mrr_scores.append(mrr(pred_cities, top3_cities))
        prec_scores.append(precision_at_k(pred_cities, top3_cities, k=k_prec))

    return {
        f"ndcg@{k_ndcg}": float(np.mean(ndcg_scores)),
        "mrr":             float(np.mean(mrr_scores)),
        f"p@{k_prec}":    float(np.mean(prec_scores)),
        "n_users":         len(ndcg_scores),
    }


# ── Unit test ─────────────────────────────────────────────────────────────────

def _unit_tests() -> None:
    print("Running evaluation unit tests ...")

    # Perfect ranking: NDCG@5 should be 1.0
    perfect = [5, 4, 3, 2, 1, 0]
    ideal   = [5, 4, 3, 2, 1, 0]
    score   = ndcg_at_k(perfect, ideal, k=5)
    assert abs(score - 1.0) < 1e-6, f"Perfect ranking NDCG@5 = {score}, expected 1.0"
    print(f"  ✓ Perfect ranking NDCG@5 = {score:.4f}")

    # Reversed ranking: NDCG@5 should be << 1
    reversed_r = [0, 1, 2, 3, 4, 5]
    score_rev  = ndcg_at_k(reversed_r, ideal, k=5)
    assert score_rev < 0.5, f"Reversed NDCG@5 = {score_rev}, expected < 0.5"
    print(f"  ✓ Reversed ranking NDCG@5 = {score_rev:.4f}")

    # MRR: first hit at position 1 → MRR = 1.0
    assert mrr(["A", "B", "C"], {"A"}) == 1.0
    assert abs(mrr(["A", "B", "C"], {"B"}) - 0.5) < 1e-9
    assert mrr(["A", "B", "C"], {"D"}) == 0.0
    print("  ✓ MRR edge cases pass")

    # Precision@3: 2 hits in top 3
    assert abs(precision_at_k(["A", "B", "C"], {"A", "B", "D"}, k=3) - 2/3) < 1e-9
    print("  ✓ Precision@3 edge case passes")

    print("All evaluation unit tests passed ✓\n")


if __name__ == "__main__":
    _unit_tests()