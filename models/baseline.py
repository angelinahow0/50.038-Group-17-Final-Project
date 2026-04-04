"""
models/baseline.py
==================
V1 Weighted Baseline — TravelRecommender
Person 2 ownership: Match-and-Go! · Group 17 · 50.038

Score = 0.25 × Budget + 0.20 × Climate + 0.35 × Activity + 0.20 × Visual

Fixed weights, no training data required.
Works with data/processed/destinations.csv produced by Person 1.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Weight constants (V1) ──────────────────────────────────────────────────────
W_BUDGET   = 0.25
W_CLIMATE  = 0.20
W_ACTIVITY = 0.35
W_VISUAL   = 0.20

# Budget tier ordering (cheaper = lower index)
BUDGET_ORDER = {"<50": 0, "50-100": 1, "100-200": 2, "200+": 3}
BUDGET_LABELS = ["<50", "50-100", "100-200", "200+"]

# Climate comfort ranges (°C)
CLIMATE_RANGES = {
    "warm":  (25, 35),
    "mild":  (18, 25),
    "cool":  (5, 18),
    "snowy": (-20, 5),
    "any":   (-20, 45),
}

# Month index → column suffix mapping
MONTH_COLS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]

# Visual preference keyword mapping
VISUAL_KEYWORDS = {
    "tropical":   ["tropical", "beach", "island", "palm", "bali", "phuket",
                   "caribbean", "maldives", "hawaii"],
    "urban":      ["urban", "city", "metropolis", "skyline", "new york",
                   "tokyo", "singapore", "london", "dubai", "hong kong"],
    "historical": ["historical", "heritage", "ancient", "ruins", "medieval",
                   "culture", "museum", "rome", "athens", "istanbul"],
    "any":        [],   # matches everything
}


class TravelRecommender:
    """V1 rule-based travel recommender using fixed weighted sub-scores."""

    def __init__(self, destinations_df: pd.DataFrame) -> None:
        self.dest = destinations_df.copy()
        self._normalise_columns()

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(
        self,
        budget_range: str,
        activities: list[str],
        travel_month: int,
        climate_preference: str,
        trip_duration: int,
        top_n: int = 5,
        visual_pref: str = "any",
    ) -> pd.DataFrame:
        """
        Score every destination and return the top-N as a DataFrame.

        Parameters
        ----------
        budget_range        : one of '<50', '50-100', '100-200', '200+'
        activities          : list of activity strings (e.g. ['beach', 'food'])
        travel_month        : 1-12 (January = 1)
        climate_preference  : 'warm', 'mild', 'cool', 'snowy', or 'any'
        trip_duration       : trip length in days (used for budget scaling)
        top_n               : number of results to return (use 50 for all)
        visual_pref         : 'tropical', 'urban', 'historical', or 'any'

        Returns
        -------
        DataFrame sorted by composite_score (desc), with sub-score columns.
        """
        df = self.dest.copy()

        df["budget_score"]    = df.apply(
            lambda r: self._budget_score(r, budget_range), axis=1)
        df["climate_score"]   = df.apply(
            lambda r: self._climate_score(r, travel_month, climate_preference), axis=1)
        df["activity_score"]  = df.apply(
            lambda r: self._activity_score(r, activities), axis=1)
        df["attraction_score"] = df.apply(
            lambda r: self._attraction_score(r), axis=1)
        df["visual_score"]    = df.apply(
            lambda r: self._visual_score(r, visual_pref), axis=1)

        df["composite_score"] = (
            W_BUDGET   * df["budget_score"]
            + W_CLIMATE  * df["climate_score"]
            + W_ACTIVITY * df["activity_score"]
            + W_VISUAL   * df["visual_score"]
        )

        result = (
            df.sort_values("composite_score", ascending=False)
              .reset_index(drop=True)
        )
        return result.head(top_n)

    # ── Sub-score helpers ─────────────────────────────────────────────────────

    def _budget_score(self, row: pd.Series, budget_range: str) -> float:
        """1.0 if destination fits budget tier, partial credit for close tiers."""
        dest_tier = self._infer_budget_tier(row)
        user_idx  = BUDGET_ORDER.get(budget_range.lower().replace("$", "").replace(" ", ""), 1)
        dest_idx  = BUDGET_ORDER.get(dest_tier, 1)
        gap = abs(user_idx - dest_idx)
        return max(0.0, 1.0 - 0.35 * gap)

    def _climate_score(self, row: pd.Series, month: int, climate_pref: str) -> float:
        """Score based on how well destination temperature matches preference."""
        col = f"avg_temp_{MONTH_COLS[month - 1]}"
        temp = self._safe_float(row.get(col, np.nan))
        if np.isnan(temp):
            return 0.5   # neutral when data missing

        low, high = CLIMATE_RANGES.get(climate_pref.lower(), (-20, 45))
        if low <= temp <= high:
            return 1.0
        margin = min(abs(temp - low), abs(temp - high))
        return max(0.0, 1.0 - margin / 15.0)

    def _activity_score(self, row: pd.Series, activities: list[str]) -> float:
        """Jaccard-style overlap between user activities and destination tags."""
        if not activities:
            return 0.5

        dest_tags = self._parse_tags(row.get("activity_tags", ""))
        if not dest_tags:
            return 0.2

        user_set = {a.lower().strip() for a in activities}
        dest_set = {t.lower().strip() for t in dest_tags}
        matches  = user_set & dest_set
        return min(1.0, len(matches) / max(len(user_set), 1))

    def _attraction_score(self, row: pd.Series) -> float:
        """Normalised tourism / satisfaction rating (0–1)."""
        rating = self._safe_float(row.get("tourism_rating", np.nan))
        if not np.isnan(rating):
            return min(1.0, rating / 10.0)

        satisfaction = self._safe_float(row.get("tourist_satisfaction", np.nan))
        if not np.isnan(satisfaction):
            return min(1.0, satisfaction / 10.0)

        return 0.5

    def _visual_score(self, row: pd.Series, visual_pref: str) -> float:
        """Keyword match of destination against user visual preference."""
        if visual_pref.lower() == "any":
            return 0.7   # neutral boost for 'any' preference

        keywords = VISUAL_KEYWORDS.get(visual_pref.lower(), [])
        if not keywords:
            return 0.5

        haystack = " ".join([
            str(row.get("city",    "")),
            str(row.get("country", "")),
            str(row.get("activity_tags", "")),
            str(row.get("description", "")),
        ]).lower()

        hits = sum(1 for kw in keywords if kw in haystack)
        return min(1.0, 0.3 + 0.7 * hits / max(len(keywords), 1))

    # ── Utility helpers ───────────────────────────────────────────────────────

    def _normalise_columns(self) -> None:
        """Standardise column names so downstream code is consistent."""
        rename = {}
        col_map = {
            "destination": "city",
            "tags":        "activity_tags",
            "activities":  "activity_tags",
        }
        for old, new in col_map.items():
            if old in self.dest.columns and new not in self.dest.columns:
                rename[old] = new
        if rename:
            self.dest.rename(columns=rename, inplace=True)

        # Ensure social_score column exists
        if "social_score" not in self.dest.columns:
            self.dest["social_score"] = 0.5

        # Ensure city column is string
        if "city" in self.dest.columns:
            self.dest["city"] = self.dest["city"].astype(str).str.strip()

    def _infer_budget_tier(self, row: pd.Series) -> str:
        """Map cost-of-living index to a budget tier label."""
        # Try cost_of_living_index first, then daily_cost, rent_index
        cost = self._safe_float(row.get("cost_of_living_index", np.nan))
        if np.isnan(cost):
            cost = self._safe_float(row.get("daily_cost", np.nan))
        if np.isnan(cost):
            cost = self._safe_float(row.get("rent_index", np.nan))

        if np.isnan(cost):
            return "50-100"   # sensible default

        # Heuristic: CoL index roughly maps to USD/day
        if cost < 40:
            return "<50"
        elif cost < 70:
            return "50-100"
        elif cost < 100:
            return "100-200"
        else:
            return "200+"

    @staticmethod
    def _parse_tags(raw: Any) -> list[str]:
        """Parse activity_tags whether stored as list, JSON string, or CSV."""
        if isinstance(raw, list):
            return raw
        if not isinstance(raw, str) or not raw.strip():
            return []
        raw = raw.strip()
        # Try JSON first
        if raw.startswith("["):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        # Fall back to comma-separated
        return [t.strip() for t in raw.split(",") if t.strip()]

    @staticmethod
    def _safe_float(val: Any) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "destinations.csv"
    )

    if not os.path.exists(DATA_PATH):
        print(f"[baseline.py] destinations.csv not found at {DATA_PATH}")
        print("  → Run preprocessing pipeline first (Person 1 task).")
    else:
        df   = pd.read_csv(DATA_PATH)
        rec  = TravelRecommender(df)
        top5 = rec.recommend(
            budget_range       = "50-100",
            activities         = ["beach", "food"],
            travel_month       = 7,
            climate_preference = "warm",
            trip_duration      = 7,
        )
        print("── V1 Baseline smoke-test ──")
        print(top5[["city", "composite_score",
                     "budget_score", "climate_score",
                     "activity_score", "visual_score"]].to_string())