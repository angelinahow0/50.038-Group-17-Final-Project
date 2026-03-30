"""
03_merge_datasets.py
=====================
Merges all 4 cleaned datasets into a city-level master table.

Join strategy
-------------
Backbone: worldwide_travel_cities_clean.csv  (city-level, has lat/lon)

Step 1  WTC city  +  Meteostat monthly
        Join key: city name (normalised)
        Result: city x month table with WTC scores + Meteostat climate

Step 2  city-month  ->  city annual aggregates
        Collapse the monthly climate into annual normals per city

Step 3  city  +  Kaggle image city summary
        Join key: city name (normalised)
        Left join: unmatched cities keep NaN image fields

Step 4  city  +  Cost of Living
        Join key: WTC country -> CoL Country (normalised)
        STRICT: cities whose country has no CoL row are DROPPED.
        (Per spec: drop cities with no country match)

Outputs (data/merged/)
----------------------
  city_master.csv      - one row per city, all features merged
  city_master_slim.csv - reduced column set for recommender
  merge_audit.csv      - per-step match / drop report
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows consoles that default to CP1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE   = os.path.join(os.path.dirname(__file__), "..")
CLEAN  = os.path.join(BASE, "data", "cleaned")
MERGED = os.path.join(BASE, "data", "merged")
os.makedirs(MERGED, exist_ok=True)

audit_rows = []

def norm(s):
    """Normalise city/country name for joining."""
    if not isinstance(s, str):
        return ""
    return (s.strip().lower()
             .replace("é","e").replace("è","e").replace("ê","e")
             .replace("í","i").replace("ó","o").replace("ú","u")
             .replace("ü","u").replace("ö","o").replace("ä","a")
             .replace("ñ","n").replace("ç","c")
             .replace("-"," ").replace("_"," "))

def record_audit(step, left_n, right_n, matched, dropped):
    audit_rows.append({
        "step": step,
        "left_rows": left_n,
        "right_rows": right_n,
        "matched": matched,
        "dropped": dropped,
        "match_rate_pct": round(matched / left_n * 100, 1) if left_n else 0,
    })
    print(f"    {step}: {left_n} -> matched {matched}, dropped {dropped}")


# ===================================================
# LOAD CLEANED DATA
# ===================================================
print("Loading cleaned datasets ...")

def load(fname):
    path = os.path.join(CLEAN, fname)
    if not os.path.exists(path):
        print(f"  [WARN] {fname} not found - skipping")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {fname}: {df.shape}")
    return df

df_wtc     = load("worldwide_travel_cities_clean.csv")
df_meteo   = load("meteostat_city_monthly_clean.csv")
df_image   = load("image_city_summary_clean.csv")
df_col     = load("cost_of_living_clean.csv")
df_wtc_mon = load("wtc_monthly_temps_clean.csv")   # WTC monthly temps (fallback for Meteostat)

if df_wtc is None:
    raise RuntimeError("worldwide_travel_cities_clean.csv is required as the backbone.")


# ===================================================
# STEP 1  -  WTC + Meteostat monthly climate
# ===================================================
print("\n== Step 1: WTC + Meteostat monthly ==")

# WTC is the backbone; add normalised join keys
df_wtc["_city_key"] = df_wtc["city"].map(norm)

if df_meteo is not None:
    df_meteo["_city_key"] = df_meteo["city"].map(norm)

    # Aggregate Meteostat to annual city-level first
    meteo_annual = df_meteo.groupby(["_city_key", "city", "city_country",
                                    "station_id", "latitude", "longitude"]).agg(
        meteo_avg_temp_c        = ("temp",            "mean"),
        meteo_min_temp_c        = ("tmin",            "min"),
        meteo_max_temp_c        = ("tmax",            "max"),
        meteo_extreme_min_c     = ("txmn",            "min"),
        meteo_extreme_max_c     = ("txmx",            "max"),
        meteo_annual_precip_mm  = ("prcp",            "sum"),
        meteo_avg_pres_hpa      = ("pres",            "mean"),
        meteo_total_sun_h       = ("tsun",            lambda x: (x / 60).sum()),
        meteo_avg_daily_sun_h   = ("avg_daily_sun_h", "mean"),
        meteo_avg_comfort       = ("comfort_index",   "mean"),
        meteo_best_comfort      = ("comfort_index",   "max"),
        meteo_wet_months        = ("is_wet_month",    "sum"),
        meteo_cold_months       = ("is_cold_month",   "sum"),
        meteo_hot_months        = ("is_hot_month",    "sum"),
        meteo_years_of_data     = ("year",            "nunique"),
        meteo_data_coverage_pct = ("temp",            lambda x: round(x.notna().mean() * 100, 1)),
    ).reset_index()

    for col in ["meteo_avg_temp_c", "meteo_annual_precip_mm", "meteo_avg_pres_hpa",
                "meteo_total_sun_h", "meteo_avg_daily_sun_h",
                "meteo_avg_comfort", "meteo_best_comfort"]:
        meteo_annual[col] = meteo_annual[col].round(2)

    # Also compute best comfort month
    best_month_idx = df_meteo.groupby("_city_key")["comfort_index"].idxmax()
    best_months = df_meteo.loc[best_month_idx, ["_city_key", "month"]].rename(
        columns={"month": "meteo_best_comfort_month"})
    meteo_annual = meteo_annual.merge(best_months, on="_city_key", how="left")

    # Join to WTC backbone
    master = df_wtc.merge(
        meteo_annual.drop(columns=["city","country"], errors="ignore"),
        on="_city_key", how="left"
    )
    matched = master["meteo_avg_temp_c"].notna().sum()
    record_audit("WTC + Meteostat", len(df_wtc), len(meteo_annual), matched,
                 len(df_wtc) - matched)
else:
    master = df_wtc.copy()
    print("  [SKIP] No Meteostat data; WTC monthly temps will be used as climate fallback")

# If Meteostat data unavailable, derive annual climate from WTC monthly temps
if df_wtc_mon is not None and "meteo_avg_temp_c" not in master.columns:
    df_wtc_mon["_city_key"] = df_wtc_mon["city"].map(norm)
    wtc_annual_clim = df_wtc_mon.groupby("_city_key").agg(
        wtc_annual_avg_temp = ("wtc_avg_temp", "mean"),
        wtc_annual_min_temp = ("wtc_min_temp", "min"),
        wtc_annual_max_temp = ("wtc_max_temp", "max"),
    ).reset_index().round(2)
    master = master.merge(wtc_annual_clim, on="_city_key", how="left")

# If Meteostat available, also annotate best/worst month from WTC temps
if df_wtc_mon is not None:
    df_wtc_mon["_city_key"] = df_wtc_mon["city"].map(norm)
    wtc_best = (df_wtc_mon.groupby("_city_key")["wtc_avg_temp"]
                .agg(lambda x: x.idxmax() if not x.isna().all() else np.nan))
    # Map index back to month number
    idx_to_month = df_wtc_mon["month"].to_dict()
    wtc_best_month = wtc_best.map(idx_to_month).reset_index()
    wtc_best_month.columns = ["_city_key", "wtc_warmest_month"]
    master = master.merge(wtc_best_month, on="_city_key", how="left")


# ===================================================
# STEP 2  -  master + Kaggle image city summary
# ===================================================
print("\n== Step 2: master + Kaggle Images ==")

if df_image is not None:
    df_image["_city_key"] = df_image["city"].map(norm)
    before = len(master)
    master = master.merge(
        df_image.drop(columns=["city", "country"], errors="ignore"),
        on="_city_key", how="left"
    )
    matched = master["image_score"].notna().sum()
    record_audit("master + Images", before, len(df_image), matched,
                 before - matched)
else:
    print("  [SKIP] No image data")


# ===================================================
# STEP 3  -  master + Cost of Living (STRICT country join)
# ===================================================
print("\n== Step 3: master + Cost of Living (strict country join) ==")

if df_col is not None:
    # Normalise country names for joining
    df_col["_country_key"] = df_col["Country"].map(norm)
    master["_country_key"] = master["country"].map(norm)

    # Country name reconciliation map (WTC country -> CoL Country style)
    COUNTRY_ALIASES = {
        "uk":                "united kingdom",
        "usa":               "united states",
        "united states of america": "united states",
        "czechia":           "czech republic",
        "south korea":       "south korea",
        "uae":               "united arab emirates",
        "russia":            "russia",
        "taiwan":            "taiwan",
        "hong kong":         "hong kong",
        "macau":             "macau",
    }
    master["_country_key"] = master["_country_key"].replace(COUNTRY_ALIASES)

    before = len(master)

    # Strict inner-style: keep only cities whose country has a CoL match
    col_countries = set(df_col["_country_key"].unique())
    has_col_match = master["_country_key"].isin(col_countries)
    dropped = (~has_col_match).sum()

    if dropped > 0:
        print(f"    Dropping {dropped} cities with no CoL country match:")
        dropped_cities = master.loc[~has_col_match, ["city","country"]].values.tolist()
        for c, co in dropped_cities:
            print(f"      - {c} ({co})")

    master = master[has_col_match].copy()

    # Now left join CoL columns
    col_cols_to_join = [
        "_country_key", "Rank",
        "Cost of Living Index", "Rent Index",
        "Cost of Living Plus Rent Index", "Groceries Index",
        "Restaurant Price Index", "Local Purchasing Power Index",
        "affordability_tier", "value_score", "has_complete_indices",
    ]
    col_cols_to_join = [c for c in col_cols_to_join if c in df_col.columns]
    # Rename CoL columns to avoid spaces in downstream processing
    col_subset = df_col[col_cols_to_join].copy()
    col_subset = col_subset.rename(columns={
        "Rank":                           "col_rank",
        "Cost of Living Index":           "col_cost_of_living_index",
        "Rent Index":                     "col_rent_index",
        "Cost of Living Plus Rent Index": "col_cost_plus_rent_index",
        "Groceries Index":                "col_groceries_index",
        "Restaurant Price Index":         "col_restaurant_index",
        "Local Purchasing Power Index":   "col_purchasing_power_index",
        "affordability_tier":             "col_affordability_tier",
        "value_score":                    "col_value_score",
        "has_complete_indices":           "col_has_complete_indices",
    })

    master = master.merge(col_subset, on="_country_key", how="left")
    matched = master["col_cost_of_living_index"].notna().sum()
    record_audit("master + CoL (strict)", before, len(df_col), matched, before - matched)
else:
    print("  [SKIP] No CoL data")


# ===================================================
# STEP 4  -  post-merge feature engineering
# ===================================================
print("\n== Step 4: feature engineering ==")

df = master.copy()

# Clean up join key columns
df = df.drop(columns=["_city_key", "_country_key"], errors="ignore")

# WTC score columns (1-5 scale) - already in master from backbone
SCORE_COLS = ["culture", "adventure", "nature", "beaches",
              "nightlife", "cuisine", "wellness", "urban", "seclusion"]

# Climate source: prefer Meteostat, fallback to WTC monthly
if "meteo_avg_temp_c" in df.columns:
    df["climate_avg_temp_c"] = df["meteo_avg_temp_c"]
    df["climate_source"]     = "meteostat"
elif "wtc_annual_avg_temp" in df.columns:
    df["climate_avg_temp_c"] = df["wtc_annual_avg_temp"]
    df["climate_source"]     = "wtc_monthly"
else:
    df["climate_avg_temp_c"] = np.nan
    df["climate_source"]     = "none"

# Master travel score: weighted blend of available signals
def safe_norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn or pd.isna(mx):
        return pd.Series(5.0, index=s.index)
    return (s - mn) / (mx - mn) * 10

components = {}

# WTC composite score (activity dimensions, 1-5 mean -> normalise to 0-10)
wtc_present = [c for c in SCORE_COLS if c in df.columns]
if wtc_present:
    df["wtc_composite"] = df[wtc_present].mean(axis=1)
    components["wtc"]    = safe_norm(df["wtc_composite"]) * 0.40

# Climate comfort from Meteostat (already 0-10)
if "meteo_avg_comfort" in df.columns:
    components["climate"] = df["meteo_avg_comfort"].fillna(5) * 0.25

# Kaggle image attractiveness (0–100 → normalise)
if "image_score" in df.columns:
    components["visual"] = safe_norm(df["image_score"].fillna(0)) * 0.15

# CoL: higher purchasing power + lower cost = better value (invert cost index)
if "col_cost_of_living_index" in df.columns:
    components["value"] = safe_norm(
        100 - df["col_cost_of_living_index"].fillna(50)) * 0.20

if components:
    df["master_travel_score"] = sum(components.values()).clip(0, 10).round(2)
else:
    df["master_travel_score"] = 5.0

# Climate zone from WTC region + climate data
def climate_zone(row):
    temp = row.get("climate_avg_temp_c", np.nan)
    if pd.isna(temp):
        return "Unknown"
    if temp > 24: return "Tropical/Hot"
    if temp > 17: return "Warm/Mediterranean"
    if temp > 10: return "Temperate"
    if temp > 0:  return "Cool"
    return "Cold"

df["climate_zone"] = df.apply(climate_zone, axis=1)

# Data source coverage flag
def coverage(row):
    srcs = []
    if pd.notna(row.get("meteo_avg_temp_c")) or pd.notna(row.get("wtc_annual_avg_temp")):
        srcs.append("climate")
    if pd.notna(row.get("image_score")):
        srcs.append("visual")
    if pd.notna(row.get("col_cost_of_living_index")):
        srcs.append("cost")
    # WTC is always present (backbone)
    srcs.append("wtc_ratings")
    return "|".join(srcs)

df["data_sources"] = df.apply(coverage, axis=1)
df["source_count"] = df["data_sources"].str.count(r"\|") + 1

# Dedup & sort
df = df.drop_duplicates(subset=["id"])
df = df.sort_values("master_travel_score", ascending=False).reset_index(drop=True)
df["rank_overall"] = df.index + 1

print(f"\n  Master table: {len(df)} cities x {len(df.columns)} features")
print(f"  Score range: {df['master_travel_score'].min():.2f} – {df['master_travel_score'].max():.2f}")
print(f"  Cities with all 3 external sources: {(df['source_count'] >= 4).sum()}")


# ===================================================
# SAVE
# ===================================================
df.to_csv(os.path.join(MERGED, "city_master.csv"), index=False)
print(f"\n  Saved city_master.csv  ({len(df)} rows x {len(df.columns)} cols)")

# Slim version: key fields only for recommender
SLIM_COLS = [
    # Identity
    "id", "city", "country", "region", "latitude", "longitude",
    "short_description", "budget_level",
    # Master score
    "master_travel_score", "rank_overall", "data_sources", "source_count",
    "climate_zone", "climate_avg_temp_c", "climate_source",
    # WTC activity scores (1-5)
    "culture", "adventure", "nature", "beaches", "nightlife",
    "cuisine", "wellness", "urban", "seclusion", "composite_score",
    # WTC trip duration flags
    "ideal_weekend", "ideal_short_trip", "ideal_one_week", "ideal_long_trip",
    # Climate (Meteostat)
    "meteo_avg_temp_c", "meteo_annual_precip_mm", "meteo_avg_rhum_pct",
    "meteo_avg_daily_sun_h", "meteo_avg_comfort", "meteo_best_comfort_month",
    "meteo_wet_months", "meteo_cold_months", "meteo_hot_months",
    # Kaggle Images
    "image_score", "image_count", "avg_text_quality",
    "avg_colorfulness", "avg_brightness", "avg_contrast", "avg_attractiveness",
    # Cost of Living (country-level proxy)
    "col_cost_of_living_index", "col_rent_index", "col_groceries_index",
    "col_restaurant_index", "col_purchasing_power_index",
    "col_affordability_tier", "col_value_score",
]
slim = df[[c for c in SLIM_COLS if c in df.columns]]
slim.to_csv(os.path.join(MERGED, "city_master_slim.csv"), index=False)
print(f"  Saved city_master_slim.csv  ({len(slim)} rows x {len(slim.columns)} cols)")

# Audit
audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(os.path.join(MERGED, "merge_audit.csv"), index=False)
print(f"  Saved merge_audit.csv")

print("\n[Done] Merge complete.")
print(audit_df.to_string(index=False))
