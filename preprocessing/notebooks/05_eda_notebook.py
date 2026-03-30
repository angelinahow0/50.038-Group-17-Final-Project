"""
05_eda_notebook.py
===================
Full Exploratory Data Analysis for the Travel Recommender pipeline.

Covers:
  Section 1 – Dataset overviews & data quality
  Section 2 – Meteostat: Climate patterns & global distribution
  Section 3 – Kaggle Images: Visual appeal & attractiveness scores
  Section 4 – Cost of Living: Price tiers & affordability
  Section 5 – Worldwide Travel Cities: Ratings & correlations
  Section 6 – Merged master: Score distributions & rankings
  Section 7 – Synthetic users: Persona analysis & preference space
  Section 8 – Interaction matrix: Rating distributions

All charts saved to notebooks/eda_figures/ as high-res PNGs.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────
BASE      = os.path.join(os.path.dirname(__file__), '..')
CLEAN     = os.path.join(BASE, 'data', 'cleaned')
MERGED    = os.path.join(BASE, 'data', 'merged')
SYNTHETIC = os.path.join(BASE, 'data', 'synthetic')
FIGS      = os.path.join(BASE, 'notebooks', 'eda_figures')
os.makedirs(FIGS, exist_ok=True)

# ── Style ──────────────────────────────────────────────
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
           "#3B1F2B", "#44BBA4", "#E94F37", "#393E41"]
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150,
    "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

def savefig(name):
    path = os.path.join(FIGS, name + ".png")
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {name}.png")

def section(title, n):
    print(f"\n{'═'*60}")
    print(f"  Section {n}: {title}")
    print(f"{'═'*60}")

# ══════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════
print("Loading datasets …")
df_meteo  = pd.read_csv(os.path.join(CLEAN, "meteostat_city_monthly_clean.csv"))
_img_path   = os.path.join(CLEAN, "image_city_summary_clean.csv")
_posts_path = os.path.join(CLEAN, "image_posts_clean.csv")
df_image    = pd.read_csv(_img_path)   if os.path.exists(_img_path)   else None
df_img_posts= pd.read_csv(_posts_path) if os.path.exists(_posts_path) else None
df_col    = pd.read_csv(os.path.join(CLEAN, "cost_of_living_clean.csv"))
df_wtc    = pd.read_csv(os.path.join(CLEAN, "worldwide_travel_cities_clean.csv"))
df_master = pd.read_csv(os.path.join(MERGED, "city_master_slim.csv"))
df_users  = pd.read_csv(os.path.join(SYNTHETIC, "synthetic_users.csv"))
df_inter  = pd.read_csv(os.path.join(SYNTHETIC, "synthetic_user_interactions.csv"))
print("All datasets loaded.")


# ══════════════════════════════════════════════════════
# SECTION 1 – Data Quality Dashboard
# ══════════════════════════════════════════════════════
section("Dataset Quality Overview", 1)

datasets = {
    "Meteostat\n(Climate)": df_meteo,
    "Cost of\nLiving": df_col,
    "Worldwide\nTravel Cities": df_wtc,
}
if df_image is not None:
    datasets["Kaggle\nImages"] = df_image

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Dataset Quality Dashboard", fontsize=15, fontweight="bold", y=1.02)

# 1a. Dataset sizes
names = list(datasets.keys())
rows  = [len(d) for d in datasets.values()]
cols  = [len(d.columns) for d in datasets.values()]

ax = axes[0]
x = np.arange(len(names))
bars = ax.bar(x, rows, color=PALETTE[:4], alpha=0.85, width=0.5)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Row count")
ax.set_title("Dataset Sizes")
for bar, v in zip(bars, rows):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# 1b. Missing value rates
ax = axes[1]
null_rates = [(d.isnull().sum().sum() / (d.shape[0]*d.shape[1])*100) for d in datasets.values()]
bars = ax.bar(x, null_rates, color=PALETTE[4:8], alpha=0.85, width=0.5)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Missing value %")
ax.set_title("Overall Missing Rate (%)")
for bar, v in zip(bars, null_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# 1c. Column count
ax = axes[2]
bars = ax.barh(names, cols, color=PALETTE[:4], alpha=0.85)
ax.set_xlabel("Number of columns")
ax.set_title("Feature Dimensionality")
for bar, v in zip(bars, cols):
    ax.text(v + 0.3, bar.get_y() + bar.get_height()/2, str(v),
            va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
savefig("01a_data_quality_overview")

# 1d. Missing heatmap for WTC (most columns)
fig, ax = plt.subplots(figsize=(14, 4))
null_map = df_wtc.isnull().astype(int)
# Sample 50 cols max
sample_cols = null_map.columns[:50]
sns.heatmap(null_map[sample_cols].T, cmap=["#EAECEE", "#C73E1D"],
            cbar=False, yticklabels=True, xticklabels=False, ax=ax, linewidths=0)
ax.set_title("Missing Value Map – Worldwide Travel Cities (rows × columns)", fontweight="bold")
ax.set_xlabel("Cities (rows)")
ax.set_ylabel("Columns (first 50)")
plt.tight_layout()
savefig("01b_missing_heatmap_wtc")


# ══════════════════════════════════════════════════════
# SECTION 2 – Meteostat: Climate Analysis
# ══════════════════════════════════════════════════════
section("Meteostat Climate Analysis", 2)

# 2a. Monthly temperature profiles by city (top 12 by data coverage)
top_cities = df_meteo.groupby("city")["temp"].count().nlargest(12).index.tolist()
df_top = df_meteo[df_meteo["city"].isin(top_cities)]

fig, axes = plt.subplots(3, 4, figsize=(18, 10))
axes = axes.flatten()
for i, city in enumerate(top_cities):
    ax = axes[i]
    sub = df_top[df_top["city"] == city]
    monthly = sub.groupby("month")["temp"].agg(["mean", "std"]).reset_index()
    ax.fill_between(monthly["month"],
                    monthly["mean"] - monthly["std"],
                    monthly["mean"] + monthly["std"],
                    alpha=0.3, color=PALETTE[i % len(PALETTE)])
    ax.plot(monthly["month"], monthly["mean"],
            color=PALETTE[i % len(PALETTE)], lw=2.5, marker="o", ms=4)
    ax.set_title(city, fontweight="bold", fontsize=9)
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Temp °C")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.axhline(0, color="gray", lw=0.5, ls="--")

fig.suptitle("Monthly Temperature Profiles – Top 12 Cities\n(mean ± 1 std across years)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("02a_temp_profiles_cities")

# 2b. Precipitation vs Temperature scatter (annual city means)
annual_city = df_meteo.groupby("city").agg(
    avg_temp      = ("temp",            "mean"),
    total_precip  = ("prcp",            "sum"),
    avg_sunshine  = ("avg_daily_sun_h", "mean"),
    comfort       = ("comfort_index",   "mean"),
    hot_months    = ("is_hot_month",    "sum"),
    cold_months   = ("is_cold_month",   "sum"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
sc = ax.scatter(annual_city["avg_temp"], annual_city["total_precip"],
                c=annual_city["comfort"], cmap="RdYlGn",
                alpha=0.7, s=60, vmin=0, vmax=10)
plt.colorbar(sc, ax=ax, label="Comfort Index")
ax.set_xlabel("Annual Average Temperature (°C)")
ax.set_ylabel("Annual Precipitation (mm)")
ax.set_title("Temperature vs Precipitation\n(colour = comfort index)", fontweight="bold")

# Label top 10 most comfortable
for _, row in annual_city.nlargest(10, "comfort").iterrows():
    ax.annotate(row["city"], (row["avg_temp"], row["total_precip"]),
                fontsize=7, xytext=(3, 3), textcoords="offset points")

ax = axes[1]
sc = ax.scatter(annual_city["avg_sunshine"], annual_city["comfort"],
                c=annual_city["avg_temp"], cmap="RdYlGn", alpha=0.7, s=60)
plt.colorbar(sc, ax=ax, label="Avg Temp °C")
ax.set_xlabel("Average Daily Sunshine (hours)")
ax.set_ylabel("Climate Comfort Index (0–10)")
ax.set_title("Sunshine vs Comfort Index\n(colour = temperature)", fontweight="bold")

plt.tight_layout()
savefig("02b_climate_scatter")

# 2c. Top 20 cities by comfort score
top_comfort = annual_city.nlargest(20, "comfort")
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(range(len(top_comfort)), top_comfort["comfort"].values,
               color=PALETTE[0], alpha=0.85)
ax.set_yticks(range(len(top_comfort)))
ax.set_yticklabels(top_comfort["city"].values, fontsize=9)
ax.set_xlabel("Climate Comfort Index (0–10)")
ax.set_title("Top 20 Cities by Climate Comfort Score", fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
savefig("02c_top_comfort_cities")

# 2d. Heatmap of monthly avg temperature (top 30 cities by comfort)
top30 = annual_city.nlargest(30, "comfort")["city"].tolist()
comfort_matrix = df_meteo[df_meteo["city"].isin(top30)].pivot_table(
    index="city", columns="month", values="temp", aggfunc="mean")
comfort_matrix.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(comfort_matrix, cmap="RdYlGn", annot=True, fmt=".1f",
            linewidths=0.3, ax=ax, cbar_kws={"label": "Avg Temp °C"})
ax.set_title("Monthly Average Temperature – Top 30 Cities by Comfort", fontweight="bold")
plt.tight_layout()
savefig("02d_monthly_temp_heatmap")

# 2e. Hot/cold month distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(annual_city["hot_months"], bins=13, range=(0,12),
        color=PALETTE[2], alpha=0.8, edgecolor="white")
ax.set_xlabel("Number of Hot Months (avg temp > 28°C)")
ax.set_ylabel("Cities")
ax.set_title("Distribution of Hot Months per City", fontweight="bold")

ax = axes[1]
ax.hist(annual_city["cold_months"], bins=13, range=(0,12),
        color=PALETTE[0], alpha=0.8, edgecolor="white")
ax.set_xlabel("Number of Cold Months (avg temp < 10°C)")
ax.set_ylabel("Cities")
ax.set_title("Distribution of Cold Months per City", fontweight="bold")
plt.tight_layout()
savefig("02e_hot_cold_months")


# ══════════════════════════════════════════════════════
# SECTION 3 – Kaggle Images: Visual Appeal Analysis
# ══════════════════════════════════════════════════════
section("Kaggle Images: Visual Appeal & Attractiveness", 3)

if df_image is None or df_img_posts is None:
    print("  [SKIP] image_city_summary_clean.csv / image_posts_clean.csv not found — "
          "run 01 and 02 pipeline scripts first")
else:
    df_posts = df_img_posts.copy()

    # 3a. Distribution of per-image scores
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Kaggle Tourism Image Score Distributions", fontsize=13, fontweight="bold")

    # 3a. Distribution of per-image scores
    score_cols  = ["text_quality", "colorfulness_norm", "image_attractiveness"]
    score_labels = ["Text Quality (0–10)", "Colorfulness (normalised)",
                    "Image Attractiveness (0–10)"]
    for ax, col, label, color in zip(axes, score_cols, score_labels, PALETTE[:3]):
        data = df_posts[col].dropna() if col in df_posts.columns else pd.Series(dtype=float)
        if data.empty:
            ax.set_title(label + " (no data)")
            continue
        ax.hist(data, bins=30, color=color, alpha=0.80, edgecolor="white")
        ax.axvline(data.median(), color="red", ls="--", lw=1.5,
                   label=f"Median: {data.median():.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=9)

    plt.tight_layout()
    savefig("03a_image_score_distributions")

    # 3b. Top cities by image_score (city summary)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Kaggle Images: City-Level Attractiveness", fontsize=13,
                 fontweight="bold")

    top_img = df_image.nlargest(min(20, len(df_image)), "image_score")
    ax = axes[0]
    bars = ax.barh(range(len(top_img)), top_img["image_score"],
                   color=PALETTE[1], alpha=0.85)
    ax.set_yticks(range(len(top_img)))
    ax.set_yticklabels(top_img["city"], fontsize=9)
    ax.set_xlabel("Image Attractiveness Score (0–100)")
    ax.set_title("Top Cities by Image Score", fontweight="bold")
    ax.invert_yaxis()

    # 3c. Scatter: text quality vs colorfulness
    ax = axes[1]
    if "avg_text_quality" in df_image.columns and "avg_colorfulness" in df_image.columns:
        sc = ax.scatter(df_image["avg_colorfulness"], df_image["avg_text_quality"],
                        c=df_image["image_score"], cmap="viridis", alpha=0.75, s=60)
        plt.colorbar(sc, ax=ax, label="Image Score")
        ax.set_xlabel("Avg Colorfulness")
        ax.set_ylabel("Avg Text Quality (0–10)")
        ax.set_title("Colorfulness vs Text Quality\n(colour = image score)",
                     fontweight="bold")

    plt.tight_layout()
    savefig("03b_image_city_scores")


# ══════════════════════════════════════════════════════
# SECTION 4 – Cost of Living Analysis
# ══════════════════════════════════════════════════════
section("Cost of Living Analysis", 4)

INDEX_COLS = [
    "Cost of Living Index",
    "Rent Index",
    "Cost of Living Plus Rent Index",
    "Groceries Index",
    "Restaurant Price Index",
    "Local Purchasing Power Index",
]

# 4a. Distribution of each index
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(INDEX_COLS):
    ax = axes[i]
    data = df_col[col].dropna()
    ax.hist(data, bins=30, color=PALETTE[i % len(PALETTE)], alpha=0.8, edgecolor="white")
    ax.axvline(data.median(), color="red", ls="--", lw=1.5,
               label=f"Median: {data.median():.1f}")
    ax.set_xlabel(col)
    ax.set_ylabel("Countries")
    ax.set_title(col, fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
fig.suptitle("Distribution of Cost of Living Indices (139 Countries)", 
             fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("04a_col_distributions")

# 4b. Correlation heatmap of all indices
corr = df_col[INDEX_COLS].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_title("Cost Index Correlation Matrix", fontweight="bold")
plt.tight_layout()
savefig("04b_cost_correlation_heatmap")

# 4c. Affordability tier breakdown
fig, ax = plt.subplots(figsize=(8, 5))
tier_counts = df_col["affordability_tier"].value_counts()
tier_colors = {"Budget": "#44BBA4", "Mid-range": "#2E86AB",
               "Expensive": "#F18F01", "Premium": "#C73E1D"}
bars = ax.bar(tier_counts.index, tier_counts.values,
              color=[tier_colors.get(t, PALETTE[0]) for t in tier_counts.index],
              alpha=0.85)
ax.set_ylabel("Number of Countries")
ax.set_title("Countries by Affordability Tier", fontweight="bold")
for bar, v in zip(bars, tier_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(v), ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
savefig("04c_affordability_tiers")

# 4d. Top 20 most affordable vs most expensive countries
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

top_cheap = df_col.nsmallest(20, "Cost of Living Index")
ax = axes[0]
ax.barh(range(len(top_cheap)), top_cheap["Cost of Living Index"],
        color="#44BBA4", alpha=0.85)
ax.set_yticks(range(len(top_cheap)))
ax.set_yticklabels(top_cheap["Country"], fontsize=9)
ax.set_xlabel("Cost of Living Index")
ax.set_title("Top 20 Most Affordable Countries", fontweight="bold")
ax.invert_yaxis()

top_exp = df_col.nlargest(20, "Cost of Living Index")
ax = axes[1]
ax.barh(range(len(top_exp)), top_exp["Cost of Living Index"],
        color="#C73E1D", alpha=0.85)
ax.set_yticks(range(len(top_exp)))
ax.set_yticklabels(top_exp["Country"], fontsize=9)
ax.set_xlabel("Cost of Living Index")
ax.set_title("Top 20 Most Expensive Countries", fontweight="bold")
ax.invert_yaxis()

plt.tight_layout()
savefig("04d_cheapest_vs_expensive")

# 4e. Value score: purchasing power vs cost of living
fig, ax = plt.subplots(figsize=(12, 6))
valid = df_col.dropna(subset=["Cost of Living Index", "Local Purchasing Power Index"])
sc = ax.scatter(valid["Cost of Living Index"], valid["Local Purchasing Power Index"],
                c=valid["value_score"], cmap="RdYlGn", alpha=0.7, s=60)
plt.colorbar(sc, ax=ax, label="Value Score (PPP / CoL)")
ax.set_xlabel("Cost of Living Index")
ax.set_ylabel("Local Purchasing Power Index")
ax.set_title("Cost of Living vs Purchasing Power\n(colour = value score)", fontweight="bold")
# Label top 10 value countries
for _, row in valid.nlargest(10, "value_score").iterrows():
    ax.annotate(row["Country"],
                (row["Cost of Living Index"], row["Local Purchasing Power Index"]),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.tight_layout()
savefig("04e_value_score_scatter")


# ══════════════════════════════════════════════════════
# SECTION 5 – Worldwide Travel Cities: Ratings
# ══════════════════════════════════════════════════════
section("Worldwide Travel Cities Ratings", 5)

# Real WTC score columns: culture, adventure, nature, beaches, nightlife,
# cuisine, wellness, urban, seclusion  (integer 1-5)
rating_cols = ["culture", "adventure", "nature", "beaches",
               "nightlife", "cuisine", "wellness", "urban", "seclusion"]
rating_cols = [c for c in rating_cols if c in df_wtc.columns]

# 5a. Rating distributions
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(rating_cols[:8]):
    ax = axes[i]
    data = df_wtc[col].dropna()
    ax.hist(data, bins=25, color=PALETTE[i % len(PALETTE)], alpha=0.8, edgecolor="white")
    ax.axvline(data.mean(), color="red", ls="--", lw=1.5, label=f"μ={data.mean():.1f}")
    ax.set_title(col.replace("_rating", "").replace("_", " ").title(), fontweight="bold")
    ax.set_xlabel("Rating (1–10)")
    ax.set_ylabel("Cities")
    ax.legend(fontsize=8)
fig.suptitle("Distribution of City Ratings Across All Dimensions", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("05a_rating_distributions")

# 5b. Rating correlation heatmap
corr_rat = df_wtc[rating_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_rat, annot=True, fmt=".2f", cmap="YlOrRd",
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_title("Rating Dimensions Correlation", fontweight="bold")
plt.tight_layout()
savefig("05b_rating_correlation")

# 5c. Top 20 cities by composite score
top20_wtc = df_wtc.nlargest(20, "composite_score")
fig, ax = plt.subplots(figsize=(12, 7))
region_colors = {
    "europe":        PALETTE[0],
    "asia":          PALETTE[1],
    "north_america": PALETTE[2],
    "south_america": PALETTE[3],
    "africa":        PALETTE[4],
    "oceania":       PALETTE[5],
    "middle_east":   PALETTE[6],
}
bar_colors = [region_colors.get(r, PALETTE[0]) for r in top20_wtc["region"]]
bars = ax.barh(range(len(top20_wtc)), top20_wtc["composite_score"],
               color=bar_colors, alpha=0.85)
ax.set_yticks(range(len(top20_wtc)))
ax.set_yticklabels(top20_wtc["city"] + " (" + top20_wtc["country"] + ")", fontsize=9)
ax.set_xlabel("Composite Score (mean of 9 activity dimensions, 1–5)")
ax.set_title("Top 20 Cities by Composite Vibe Score", fontweight="bold")
ax.invert_yaxis()
handles = [mpatches.Patch(color=c, label=k) for k, c in region_colors.items()
           if k in top20_wtc["region"].values]
ax.legend(handles=handles, loc="lower right", fontsize=9, title="Region")
plt.tight_layout()
savefig("05c_top20_composite")

# 5d. Budget level and ideal duration breakdown
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
budget_counts = df_wtc["budget_level"].value_counts()
budget_colors = {"Budget": "#44BBA4", "Mid-range": "#2E86AB", "Luxury": "#C73E1D"}
ax.bar(budget_counts.index, budget_counts.values,
       color=[budget_colors.get(b, PALETTE[0]) for b in budget_counts.index],
       alpha=0.85)
ax.set_ylabel("Cities")
ax.set_title("Cities by Budget Level", fontweight="bold")
for i, v in enumerate(budget_counts.values):
    ax.text(i, v + 1, str(v), ha="center", fontweight="bold")

ax = axes[1]
duration_cols = ["ideal_weekend", "ideal_short_trip", "ideal_one_week", "ideal_long_trip"]
duration_labels = ["Weekend", "Short Trip", "One Week", "Long Trip"]
duration_counts = [df_wtc[c].sum() for c in duration_cols]
ax.bar(duration_labels, duration_counts, color=PALETTE[:4], alpha=0.85)
ax.set_ylabel("Cities")
ax.set_title("Cities by Ideal Trip Duration", fontweight="bold")
for i, v in enumerate(duration_counts):
    ax.text(i, v + 1, str(v), ha="center", fontweight="bold")

plt.tight_layout()
savefig("05d_budget_duration")

# 5e. Region distribution
fig, ax = plt.subplots(figsize=(10, 5))
region_counts = df_wtc["region"].value_counts()
ax.bar(region_counts.index, region_counts.values,
       color=[region_colors.get(r, PALETTE[0]) for r in region_counts.index],
       alpha=0.85)
ax.set_ylabel("Cities")
ax.set_title("Cities by Region", fontweight="bold")
ax.set_xticklabels(region_counts.index, rotation=15, ha="right")
for i, v in enumerate(region_counts.values):
    ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
plt.tight_layout()
savefig("05e_region_distribution")


# ══════════════════════════════════════════════════════
# SECTION 6 – Master Table: Integrated Analysis
# ══════════════════════════════════════════════════════
section("Merged Master: Integrated Analysis", 6)

# 6a. Master travel score distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.hist(df_master["master_travel_score"].dropna(), bins=30,
        color=PALETTE[0], alpha=0.8, edgecolor="white")
ax.set_xlabel("Master Travel Score (0–10)")
ax.set_ylabel("Cities")
ax.set_title("Distribution of Master Travel Scores", fontweight="bold")
ax.axvline(df_master["master_travel_score"].median(), color="red", ls="--",
           label=f"Median: {df_master['master_travel_score'].median():.2f}")
ax.legend()

ax = axes[1]
source_counts = df_master["source_count"].value_counts().sort_index()
ax.bar(source_counts.index.astype(str), source_counts.values,
       color=[PALETTE[i] for i in range(len(source_counts))], alpha=0.85)
ax.set_xlabel("Number of Data Sources")
ax.set_ylabel("Cities")
ax.set_title("Data Source Coverage per City", fontweight="bold")
for i, v in enumerate(source_counts.values):
    ax.text(i, v + 0.3, str(v), ha="center", fontweight="bold")
plt.tight_layout()
savefig("06a_master_score_distribution")

# 6b. Score vs Cost of Living scatter
fig, ax = plt.subplots(figsize=(12, 7))
region_colors = {
    "europe":        PALETTE[0],
    "asia":          PALETTE[1],
    "north_america": PALETTE[2],
    "south_america": PALETTE[3],
    "africa":        PALETTE[4],
    "oceania":       PALETTE[5],
    "middle_east":   PALETTE[6],
}
valid = df_master[["col_cost_of_living_index", "master_travel_score",
                   "region", "city"]].dropna()
for region, grp in valid.groupby("region"):
    ax.scatter(grp["col_cost_of_living_index"], grp["master_travel_score"],
               alpha=0.7, s=60, color=region_colors.get(region, PALETTE[0]),
               label=region)
# Label top 10
for _, row in valid.nlargest(10, "master_travel_score").iterrows():
    ax.annotate(row["city"],
                (row["col_cost_of_living_index"], row["master_travel_score"]),
                fontsize=7.5, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Cost of Living Index (country-level proxy)")
ax.set_ylabel("Master Travel Score (0–10)")
ax.set_title("Travel Score vs Cost of Living: Value Frontier", fontweight="bold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, title="Region")
plt.tight_layout()
savefig("06b_score_vs_cost")

# 6c. Top 25 overall ranking
top25 = df_master.nlargest(25, "master_travel_score")
fig, ax = plt.subplots(figsize=(13, 9))
bar_colors = [region_colors.get(r, PALETTE[0]) for r in top25["region"]]
ax.barh(range(len(top25)), top25["master_travel_score"],
        color=bar_colors, alpha=0.85)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(
    [f"#{i+1}  {r['city']}, {r['country']}"
     for i, (_, r) in enumerate(top25.iterrows())], fontsize=9)
ax.set_xlabel("Master Travel Score")
ax.set_title("Top 25 Cities – Master Travel Recommender Ranking", fontweight="bold")
ax.invert_yaxis()
handles = [mpatches.Patch(color=c, label=k) for k, c in region_colors.items()
           if k in top25["region"].values]
ax.legend(handles=handles, loc="lower right", fontsize=9, title="Region")
plt.tight_layout()
savefig("06c_top25_ranking")

# 6d. Climate zone breakdown
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
zone_counts = df_master["climate_zone"].value_counts()
ax.bar(zone_counts.index, zone_counts.values,
       color=PALETTE[:len(zone_counts)], alpha=0.85)
ax.set_ylabel("Cities")
ax.set_title("Cities by Climate Zone", fontweight="bold")
ax.set_xticklabels(zone_counts.index, rotation=15, ha="right")
for i, v in enumerate(zone_counts.values):
    ax.text(i, v + 0.3, str(v), ha="center", fontweight="bold")

ax = axes[1]
afford_counts = df_master["col_affordability_tier"].value_counts()
afford_colors = {"Budget": "#44BBA4", "Mid-range": "#2E86AB",
                 "Expensive": "#F18F01", "Premium": "#C73E1D"}
ax.bar(afford_counts.index, afford_counts.values,
       color=[afford_colors.get(t, PALETTE[0]) for t in afford_counts.index],
       alpha=0.85)
ax.set_ylabel("Cities")
ax.set_title("Cities by Affordability Tier", fontweight="bold")
for i, v in enumerate(afford_counts.values):
    ax.text(i, v + 0.3, str(v), ha="center", fontweight="bold")
plt.tight_layout()
savefig("06d_climate_zone_affordability")

# 6e. Meteostat comfort vs WTC composite score
fig, ax = plt.subplots(figsize=(10, 6))
valid = df_master[["meteo_avg_comfort", "composite_score", "region", "city"]].dropna()
for region, grp in valid.groupby("region"):
    ax.scatter(grp["meteo_avg_comfort"], grp["composite_score"],
               alpha=0.7, s=60, color=region_colors.get(region, PALETTE[0]),
               label=region)
ax.set_xlabel("Meteostat Climate Comfort Index (0–10)")
ax.set_ylabel("WTC Composite Vibe Score (1–5)")
ax.set_title("Climate Comfort vs Activity Vibe Score", fontweight="bold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, title="Region")
plt.tight_layout()
savefig("06e_comfort_vs_vibe")


# ══════════════════════════════════════════════════════
# SECTION 7 – Synthetic Users: Persona Analysis
# ══════════════════════════════════════════════════════
section("Synthetic User Profiles – Persona Analysis", 7)

# 7a. Persona distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
pc = df_users["persona"].value_counts()
wedges, texts, autotexts = ax.pie(pc.values, labels=pc.index, autopct="%1.1f%%",
                                   colors=PALETTE, startangle=140, pctdistance=0.82)
for at in autotexts: at.set_fontsize(9)
ax.set_title("Traveler Persona Distribution (n=5,000)", fontweight="bold")

ax = axes[1]
age_by_persona = df_users.groupby("persona")["age"].agg(["mean", "std"]).reset_index()
age_by_persona = age_by_persona.sort_values("mean")
y = range(len(age_by_persona))
ax.barh(y, age_by_persona["mean"], xerr=age_by_persona["std"],
        color=PALETTE[:len(age_by_persona)], alpha=0.8, capsize=5)
ax.set_yticks(y)
ax.set_yticklabels(age_by_persona["persona"], fontsize=9)
ax.set_xlabel("Age (mean ± 1 std)")
ax.set_title("Age Distribution by Persona", fontweight="bold")
plt.tight_layout()
savefig("07a_persona_distribution")

# 7b. Budget distribution by persona
fig, ax = plt.subplots(figsize=(13, 7))
persona_order = df_users.groupby("persona")["typical_daily_budget_usd"].median().sort_values().index
budget_data = [df_users[df_users["persona"] == p]["typical_daily_budget_usd"].values
               for p in persona_order]
bp = ax.boxplot(budget_data, patch_artist=True, vert=False, notch=True)
for patch, color in zip(bp["boxes"], PALETTE[:len(persona_order)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_yticks(range(1, len(persona_order)+1))
ax.set_yticklabels(persona_order, fontsize=9)
ax.set_xlabel("Daily Budget (USD/day)")
ax.set_title("Daily Budget Distribution by Traveler Persona", fontweight="bold")
ax.set_xlim(0, 600)
plt.tight_layout()
savefig("07b_budget_by_persona")

# 7c. Preference radar (average per persona)
pref_cols = [c for c in df_users.columns if c.startswith("pref_")]
persona_avg = df_users.groupby("persona")[pref_cols].mean()

# Radar chart
labels = [c.replace("pref_", "").replace("_", " ").title() for c in pref_cols]
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
for i, (persona, row) in enumerate(persona_avg.iterrows()):
    values = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, values, linewidth=2, linestyle="solid",
            color=PALETTE[i % len(PALETTE)], label=persona)
    ax.fill(angles, values, alpha=0.07, color=PALETTE[i % len(PALETTE)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1)
ax.set_title("Preference Profiles by Traveler Persona", fontweight="bold",
             pad=25, fontsize=13)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
savefig("07c_preference_radar")

# 7d. Geographic origin distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
nc = df_users["nationality"].value_counts().head(10)
ax.bar(nc.index, nc.values, color=PALETTE * 2, alpha=0.8)
ax.set_ylabel("Users")
ax.set_title("Top 10 User Nationalities", fontweight="bold")
ax.set_xticklabels(nc.index, rotation=25, ha="right")

ax = axes[1]
income_counts = df_users["income_tier"].value_counts()
income_colors = {"<30k": "#44BBA4", "30-70k": "#2E86AB",
                 "70-150k": "#F18F01", "150k+": "#C73E1D"}
ax.bar(income_counts.index, income_counts.values,
       color=[income_colors.get(t, PALETTE[0]) for t in income_counts.index], alpha=0.85)
ax.set_ylabel("Users")
ax.set_title("Income Tier Distribution", fontweight="bold")
for i, v in enumerate(income_counts.values):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
plt.tight_layout()
savefig("07d_demographics")

# 7e. Travel frequency & trip duration heat
fig, ax = plt.subplots(figsize=(9, 6))
hdata = df_users.groupby(["trips_per_year", "avg_trip_days"]).size().unstack(fill_value=0)
hdata = hdata.iloc[:12, :20]   # clip for clarity
sns.heatmap(hdata, cmap="Blues", ax=ax, linewidths=0.2,
            cbar_kws={"label": "User count"})
ax.set_xlabel("Average Trip Duration (days)")
ax.set_ylabel("Trips per Year")
ax.set_title("Travel Frequency × Trip Duration Heatmap", fontweight="bold")
plt.tight_layout()
savefig("07e_travel_frequency_heatmap")

# 7f. Embedding space (first 2 dims scatter by persona)
fig, ax = plt.subplots(figsize=(10, 8))
for i, persona in enumerate(df_users["persona"].unique()):
    sub = df_users[df_users["persona"] == persona]
    ax.scatter(sub["emb_0"], sub["emb_1"], alpha=0.4, s=15,
               color=PALETTE[i % len(PALETTE)], label=persona)
ax.set_xlabel("Adventure Preference")
ax.set_ylabel("Culture Preference")
ax.set_title("User Preference Space (Adventure vs Culture)\ncoloured by Persona",
             fontweight="bold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.tight_layout()
savefig("07f_preference_embedding")


# ══════════════════════════════════════════════════════
# SECTION 8 – Interaction Matrix Analysis
# ══════════════════════════════════════════════════════
section("User × City Interaction Analysis", 8)

rated = df_inter[df_inter["interaction_type"] == "visited_and_rated"]
wishlisted = df_inter[df_inter["interaction_type"] == "wishlisted"]

# 8a. Rating distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
ax.hist(rated["rating"].dropna(), bins=20, color=PALETTE[0], alpha=0.8, edgecolor="white")
ax.set_xlabel("Rating (1–5 stars)")
ax.set_ylabel("Count")
ax.set_title("Rating Distribution", fontweight="bold")
ax.axvline(rated["rating"].mean(), color="red", ls="--",
           label=f"Mean: {rated['rating'].mean():.2f}")
ax.legend()

ax = axes[1]
interactions_per_user = df_inter.groupby("user_id").size()
ax.hist(interactions_per_user, bins=40, color=PALETTE[1], alpha=0.8, edgecolor="white")
ax.set_xlabel("Interactions per user (visits + wishlist)")
ax.set_ylabel("Users")
ax.set_title("User Activity Distribution", fontweight="bold")
ax.axvline(interactions_per_user.median(), color="red", ls="--",
           label=f"Median: {interactions_per_user.median():.0f}")
ax.legend()

ax = axes[2]
city_popularity = rated.groupby("city").size().sort_values(ascending=False).head(20)
ax.barh(range(len(city_popularity)), city_popularity.values, color=PALETTE[2], alpha=0.8)
ax.set_yticks(range(len(city_popularity)))
ax.set_yticklabels(city_popularity.index, fontsize=8)
ax.set_xlabel("Number of ratings")
ax.set_title("Top 20 Most Visited Cities\n(by user interactions)", fontweight="bold")
ax.invert_yaxis()

plt.tight_layout()
savefig("08a_interaction_analysis")

# 8b. Rating by persona
fig, ax = plt.subplots(figsize=(13, 6))
merged_int = rated.merge(df_users[["user_id", "persona", "income_tier"]],
                          on="user_id", how="left")
persona_ratings = merged_int.groupby("persona")["rating"].agg(["mean", "std", "count"])
persona_ratings = persona_ratings.sort_values("mean")
y = range(len(persona_ratings))
ax.barh(y, persona_ratings["mean"], xerr=persona_ratings["std"] / np.sqrt(persona_ratings["count"]),
        color=PALETTE[:len(persona_ratings)], alpha=0.8, capsize=5)
ax.set_yticks(y)
ax.set_yticklabels(persona_ratings.index, fontsize=9)
ax.set_xlabel("Average City Rating (1–5)")
ax.set_title("Average City Ratings by Traveler Persona\n(95% CI approximated from std/√n)",
             fontweight="bold")
ax.set_xlim(3, 4.5)
ax.axvline(rated["rating"].mean(), color="red", ls="--", alpha=0.5, label="Overall mean")
ax.legend(fontsize=9)
plt.tight_layout()
savefig("08b_ratings_by_persona")

# ── Summary stats printout ─────────────────────
print("\n" + "═"*60)
print("  EDA SUMMARY STATISTICS")
print("═"*60)
print(f"\nMeteostat:    {df_meteo['city'].nunique()} cities × 12 months = {len(df_meteo):,} rows")
if df_img_posts is not None and df_image is not None:
    n_matched  = df_img_posts["city"].notna().sum()
    n_total    = len(df_img_posts)
    avg_score  = df_img_posts["image_attractiveness"].mean()
    avg_tq     = df_img_posts["text_quality"].mean()
    avg_color  = df_img_posts["colorfulness"].mean()
    top_city   = df_image.nlargest(1, "image_score").iloc[0]
    print(f"Kaggle Images:{n_total} total images | {n_matched} city-matched "
          f"({n_matched/n_total*100:.0f}%)")
    print(f"              {df_image['city'].nunique()} city summaries | "
          f"image_score range {df_image['image_score'].min():.0f}–"
          f"{df_image['image_score'].max():.0f}")
    print(f"              avg attractiveness={avg_score:.2f}/10 | "
          f"avg text quality={avg_tq:.2f}/10 | avg colorfulness={avg_color:.1f}")
    print(f"              top city: {top_city['city']} ({top_city['country']}) "
          f"score={top_city['image_score']:.1f}")
else:
    print("Kaggle Images: not yet generated — run 01 and 02 pipeline scripts")
print(f"Cost of Living:{len(df_col)} countries × {len(df_col.columns)} features")
print(f"WTC Ratings:   {len(df_wtc)} cities × {len(df_wtc.columns)} features")
print(f"\nMerged Master: {len(df_master)} cities × {len(df_master.columns)} features")
print(f"Synthetic Users: {len(df_users):,} profiles × {len(df_users.columns)} features")
print(f"Interactions: {len(df_inter):,} records "
      f"({len(rated):,} rated, {len(wishlisted):,} wishlisted)")
print(f"\nAvg rating: {rated['rating'].mean():.2f} | "
      f"Std: {rated['rating'].std():.2f}")
print(f"Avg interactions/user: {df_inter.groupby('user_id').size().mean():.1f}")
print(f"\nFigures saved to: {FIGS}")
print("\n✅ EDA Complete! All figures saved.")
