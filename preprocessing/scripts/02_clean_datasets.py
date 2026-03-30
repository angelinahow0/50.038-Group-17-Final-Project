"""
02_clean_datasets.py
=====================
Cleans each raw dataset using its exact real schema.

Meteostat parquet schema (from bulk pipeline)
---------------------------------------------
  date          – datetime built from year+month (day=1)
  tavg          – monthly average temperature (°C)
  tmin          – monthly minimum temperature (°C)
  tmax          – monthly maximum temperature (°C)
  prcp          – total monthly precipitation (mm)
  snow          – average snow depth (mm)
  wdir          – average wind direction (°)
  wspd          – average wind speed (km/h)
  wpgt          – average wind peak gust (km/h)
  pres          – average sea-level pressure (hPa)
  tsun          – total sunshine duration (minutes/month)
  station_id    – Meteostat station ID string
  station_name  – English station name
  country       – full country name (via pycountry alpha_2 lookup)
  latitude      – station latitude
  longitude     – station longitude
  elevation     – station elevation (m), can be NaN

City assignment: joined from meteostat_city_station_map.csv
  (nearest station per WTC city by Haversine distance, built in 01_fetch)

Kaggle tourism image schema
---------------------------
  filename   – JPEG filename (e.g. istockphoto-XXXXXXX-612x612.jpg)
  filepath   – absolute path to image file
  description – caption extracted from XMP dc:description (Windows Subject field)

Cost of Living (country-level)
-------------------------------
  Rank, Country,
  Cost of Living Index, Rent Index, Cost of Living Plus Rent Index,
  Groceries Index, Restaurant Price Index, Local Purchasing Power Index

Worldwide Travel Cities (city-level)
--------------------------------------
  id, city, country, region, short_description, latitude, longitude,
  avg_temp_monthly (JSON), ideal_durations (JSON array),
  budget_level, culture, adventure, nature, beaches, nightlife,
  cuisine, wellness, urban, seclusion

Outputs (data/cleaned/)
-----------------------
  meteostat_city_monthly_clean.csv   – monthly normals per city (via station map)
  meteostat_city_annual_clean.csv    – annual aggregates per city
  image_posts_clean.csv              – per-image scores (text quality + visual metrics)
  image_city_summary_clean.csv       – city-level image attractiveness aggregates
  cost_of_living_clean.csv           – cleaned country-level CoL table
  worldwide_travel_cities_clean.csv  – cleaned city table, JSON fields expanded
  wtc_monthly_temps_clean.csv        – avg_temp_monthly JSON expanded to long rows
"""

import os
import re
import json
import math
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] Pillow not installed — visual metrics will default to 0.  "
          "Run: pip install Pillow")

BASE  = os.path.join(os.path.dirname(__file__), "..")
RAW   = os.path.join(BASE, "data", "raw")
CLEAN = os.path.join(BASE, "data", "cleaned")
os.makedirs(CLEAN, exist_ok=True)

def iqr_cap(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))

def log_shape(label, df):
    print(f"    {label:50s} ({df.shape[0]:,} x {df.shape[1]})  "
          f"nulls={df.isnull().sum().sum():,}")


# ═══════════════════════════════════════════════════════
# 1. METEOSTAT – parquet → city-level monthly + annual
# ═══════════════════════════════════════════════════════
print("\n== [1] Meteostat: parquet -> city monthly/annual ==")

parquet_path = os.path.join(RAW, "meteostat_monthly_raw.parquet")
map_path     = os.path.join(RAW, "meteostat_city_station_map.csv")

if not os.path.exists(parquet_path):
    print("  [SKIP] meteostat_monthly_raw.parquet not found")
elif not os.path.exists(map_path):
    print("  [SKIP] meteostat_city_station_map.csv not found — "
          "run 01_fetch_raw_data.py with WTC CSV present")
else:
    # ── Load ──────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_path)
    print(f"  Columns: {df.columns.tolist()}")
    print(df.head(2))
    log_shape("Loaded parquet", df)

    # Real column names from this parquet:
    # temp  – monthly average temperature (°C)
    # tmin  – monthly min temperature (°C)
    # tmax  – monthly max temperature (°C)
    # txmn  – extreme minimum temperature (°C)
    # txmx  – extreme maximum temperature (°C)
    # prcp  – total monthly precipitation (mm)
    # pres  – average sea-level pressure (hPa)
    # tsun  – total sunshine duration (minutes/month)
    METEO_NUMERIC = ["temp", "tmin", "tmax", "txmn", "txmx", "prcp", "pres", "tsun"]

    for col in METEO_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Physical domain validation ─────────────────────────────────
    for col in ["temp", "tmin", "tmax", "txmn", "txmx"]:
        df.loc[df[col].lt(-89) | df[col].gt(58), col] = np.nan

    # tmin <= temp <= tmax ordering check
    bad_order = (
        (df["tmin"].notna() & df["temp"].notna() & (df["tmin"] > df["temp"])) |
        (df["temp"].notna() & df["tmax"].notna() & (df["temp"] > df["tmax"]))
    )
    df.loc[bad_order, ["tmin", "tmax"]] = np.nan
    print(f"    Temperature ordering violations nullified: {bad_order.sum()}")

    df.loc[df["prcp"].lt(0), "prcp"] = np.nan
    df.loc[df["pres"].lt(870) | df["pres"].gt(1084), "pres"] = np.nan
    df.loc[df["tsun"].lt(0)   | df["tsun"].gt(44640), "tsun"] = np.nan

    # ── tsun: convert from total minutes/month to avg daily sunshine hours ──
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year

    days_map = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    df["days_in_month"]   = df["month"].map(days_map)
    df["avg_daily_sun_h"] = (df["tsun"] / df["days_in_month"] / 60).round(2)
    df.loc[df["avg_daily_sun_h"] > 24, "avg_daily_sun_h"] = np.nan

    # ── Attach city name via spatial map ───────────────────────────
    city_map = pd.read_csv(map_path)
    reliable_map = city_map[city_map["station_reliable"] == True][
        ["city", "city_country", "station_id", "distance_km"]
    ].copy()

    unmatched_cities = city_map[~city_map["station_reliable"]]["city"].tolist()
    if unmatched_cities:
        print(f"    Cities excluded (station > 150 km): {unmatched_cities}")

    df = df.merge(reliable_map, on="station_id", how="inner")
    print(f"    After city join: {df['city'].nunique()} cities, "
          f"{len(df):,} city-month rows")

    # ── Climate comfort index (0–10) ───────────────────────────────
    df["comfort_index"] = (
        10
        - 0.30 * (df["temp"] - 22).abs()
        - 0.005 * df["prcp"].fillna(0)
        + 0.30  * df["avg_daily_sun_h"].fillna(0)
    ).clip(0, 10).round(2)

    # ── Seasonal flags ─────────────────────────────────────────────
    df["is_wet_month"]  = df["prcp"] > df.groupby("city")["prcp"].transform("median")
    df["is_cold_month"] = df["temp"] < 10
    df["is_hot_month"]  = df["temp"] > 28

    log_shape("City-month cleaned", df)
    df.to_csv(os.path.join(CLEAN, "meteostat_city_monthly_clean.csv"), index=False)
    print("  Saved meteostat_city_monthly_clean.csv")

    # ── Annual aggregation per city ────────────────────────────────
    annual = df.groupby(["city", "city_country", "station_id",
                         "latitude", "longitude", "elevation"]).agg(
        annual_avg_temp_c      = ("temp",            "mean"),
        annual_min_temp_c      = ("tmin",            "min"),
        annual_max_temp_c      = ("tmax",            "max"),
        annual_extreme_min_c   = ("txmn",            "min"),
        annual_extreme_max_c   = ("txmx",            "max"),
        annual_precip_mm       = ("prcp",            "sum"),
        annual_avg_pres_hpa    = ("pres",            "mean"),
        annual_total_sun_h     = ("tsun",            lambda x: (x / 60).sum()),
        annual_avg_daily_sun_h = ("avg_daily_sun_h", "mean"),
        avg_comfort_index      = ("comfort_index",   "mean"),
        best_comfort_index     = ("comfort_index",   "max"),
        best_comfort_month     = ("comfort_index",   lambda x: x.idxmax() if not x.isna().all() else np.nan),
        wet_month_count        = ("is_wet_month",    "sum"),
        cold_month_count       = ("is_cold_month",   "sum"),
        hot_month_count        = ("is_hot_month",    "sum"),
        years_of_data          = ("year",            "nunique"),
        data_coverage_pct      = ("temp",            lambda x: round(x.notna().mean() * 100, 1)),
    ).reset_index()

    # Map best_comfort_month index back to month number
    idx_to_month = df["month"].to_dict()
    annual["best_comfort_month"] = annual["best_comfort_month"].map(idx_to_month)

    for col in ["annual_avg_temp_c", "annual_precip_mm", "annual_avg_pres_hpa",
                "annual_total_sun_h", "annual_avg_daily_sun_h",
                "avg_comfort_index", "best_comfort_index"]:
        annual[col] = annual[col].round(2)

    log_shape("Annual aggregated", annual)
    annual.to_csv(os.path.join(CLEAN, "meteostat_city_annual_clean.csv"), index=False)
    print("  Saved meteostat_city_annual_clean.csv")


# ═══════════════════════════════════════════════════════
# 2. KAGGLE TOURISM IMAGES – score per image + city aggregation
# ═══════════════════════════════════════════════════════
print("\n== [2] Kaggle Tourism Images: score + city aggregation ==")

images_path = os.path.join(RAW, "kaggle_images_raw.csv")
if not os.path.exists(images_path):
    print("  [SKIP] kaggle_images_raw.csv not found — run 01_fetch_raw_data.py first")
else:
    df = pd.read_csv(images_path)
    log_shape("Loaded raw image metadata", df)

    # ── City matching: map description text → (city, country) ──────
    # Priority-ordered regex patterns (first match wins).
    # City names are aligned with the WTC dataset where possible.
    CITY_PATTERNS = [
        # East Asia
        (r"hong[\s\-]?kong|kowloon|victoria.harbour",                    "Hong Kong",         "China"),
        (r"macau|macao",                                                  "Macau",             "China"),
        (r"shanghai|bund|nanjing.road|huangpu|pudong",                    "Shanghai",          "China"),
        (r"beijing|forbidden.city|tiananmen|great.wall|temple.of.heaven", "Beijing",           "China"),
        (r"chengdu|giant.panda",                                          "Chengdu",           "China"),
        (r"hangzhou|west.lake",                                           "Hangzhou",          "China"),
        (r"fenghuang",                                                    "Fenghuang",         "China"),
        (r"kaohsiung",                                                    "Kaohsiung",         "Taiwan"),
        (r"taipei|taiwan",                                                "Taipei",            "Taiwan"),
        (r"nagasaki",                                                     "Nagasaki",          "Japan"),
        (r"shirakawa",                                                    "Shirakawa-go",      "Japan"),
        (r"kyoto",                                                        "Kyoto",             "Japan"),
        (r"tokyo",                                                        "Tokyo",             "Japan"),
        (r"osaka",                                                        "Osaka",             "Japan"),
        (r"japan",                                                        "Tokyo",             "Japan"),
        # Southeast Asia
        (r"singapore|sentosa",                                            "Singapore",         "Singapore"),
        (r"bangkok|thailand",                                             "Bangkok",           "Thailand"),
        (r"bali|ubud",                                                    "Bali",              "Indonesia"),
        (r"indonesia",                                                    "Bali",              "Indonesia"),
        (r"kuala.lumpur|batu.caves|malaysia",                             "Kuala Lumpur",      "Malaysia"),
        (r"danang|da.nang|banahil|hoi.an",                               "Da Nang",           "Vietnam"),
        (r"ho.chi.minh",                                                  "Ho Chi Minh City",  "Vietnam"),
        (r"vietnam|hanoi",                                                "Hanoi",             "Vietnam"),
        (r"manila|philippines",                                           "Manila",            "Philippines"),
        # South Asia
        (r"mumbai|bombay",                                                "Mumbai",            "India"),
        (r"agra|taj.mahal",                                               "Agra",              "India"),
        (r"rajasthan|jaipur|jodhpur",                                     "Jaipur",            "India"),
        (r"rani.ki.vav|patan|gujarat",                                    "Ahmedabad",         "India"),
        (r"india",                                                        "New Delhi",         "India"),
        (r"sri.lanka|polonnaruwa|colombo|kandy",                          "Colombo",           "Sri Lanka"),
        # Middle East
        (r"dubai",                                                        "Dubai",             "United Arab Emirates"),
        (r"manama|bahrain",                                               "Manama",            "Bahrain"),
        (r"petra|aqaba|jordan",                                           "Amman",             "Jordan"),
        # Turkey & Caucasus
        (r"istanbul|cappadocia|hattusa|bogazkoy|iztuzu",                  "Istanbul",          "Turkey"),
        (r"yerevan|armenia|garni|geghard",                                "Yerevan",           "Armenia"),
        # Europe – Western
        (r"paris|eiffel|versailles|chartres",                             "Paris",             "France"),
        (r"nice|cote.d.azur",                                             "Nice",              "France"),
        (r"france",                                                       "Paris",             "France"),
        (r"rome|colosseum|vatican",                                       "Rome",              "Italy"),
        (r"catania|sicily|etna",                                          "Catania",           "Italy"),
        (r"genoa|genova",                                                 "Genoa",             "Italy"),
        (r"venice|venezia",                                               "Venice",            "Italy"),
        (r"italy",                                                        "Rome",              "Italy"),
        (r"barcelona",                                                    "Barcelona",         "Spain"),
        (r"caceres|extremadura",                                          "Madrid",            "Spain"),
        (r"madrid|spain",                                                 "Madrid",            "Spain"),
        (r"lisbon|belem|portugal",                                        "Lisbon",            "Portugal"),
        (r"amsterdam|netherlands",                                        "Amsterdam",         "Netherlands"),
        # Europe – Central & Eastern
        (r"budapest|hungary",                                             "Budapest",          "Hungary"),
        (r"prague|czechia|czech",                                         "Prague",            "Czechia"),
        (r"vienna|wachau|austria",                                        "Vienna",            "Austria"),
        (r"mostar|bosnia",                                                "Mostar",            "Bosnia and Herzegovina"),
        (r"berat|albania",                                                "Berat",             "Albania"),
        (r"moscow|kremlin|novodevichy|russia",                            "Moscow",            "Russia"),
        # Europe – Northern & British Isles
        (r"helsinki|finland|suomenlinna",                                 "Helsinki",          "Finland"),
        (r"stockholm|sweden",                                             "Stockholm",         "Sweden"),
        (r"london|stonehenge|england",                                    "London",            "United Kingdom"),
        (r"edinburgh|scotland",                                           "Edinburgh",         "United Kingdom"),
        (r"reykjavik|iceland|skogafoss|eystrahorn|valahnukur",            "Reykjavik",         "Iceland"),
        # Africa
        (r"marrakech|morocco|atlas.mountain",                             "Marrakech",         "Morocco"),
        (r"cape.town|south.africa",                                       "Cape Town",         "South Africa"),
        (r"cairo|egypt",                                                  "Cairo",             "Egypt"),
        # North America
        (r"new.york|manhattan|world.trade.center|brooklyn",               "New York",          "United States"),
        (r"seattle|amazon.sphere",                                        "Seattle",           "United States"),
        (r"orlando|universal.studios|florida",                            "Orlando",           "United States"),
        (r"honolulu|hawaii|uss.arizona",                                  "Honolulu",          "United States"),
        (r"monument.valley|utah|grand.canyon|arizona",                    "Las Vegas",         "United States"),
        (r"chicago",                                                      "Chicago",           "United States"),
        (r"los.angeles|california",                                       "Los Angeles",       "United States"),
        (r"san.francisco",                                                "San Francisco",     "United States"),
        (r"usa|united.states",                                            "New York",          "United States"),
        # Mexico & Caribbean
        (r"mexico.city|teotihuacan|tenochtitlan",                         "Mexico City",       "Mexico"),
        (r"chichen.itza|yucatan|cancun|mayan|palenque|tikal",             "Cancún",            "Mexico"),
        (r"mexico",                                                       "Mexico City",       "Mexico"),
        (r"havana|cuba|camaguey",                                         "Havana",            "Cuba"),
        # South America
        (r"rio.de.janeiro|maracana",                                      "Rio de Janeiro",    "Brazil"),
        (r"sao.paulo|tiradentes|brazil",                                  "São Paulo",         "Brazil"),
        (r"buenos.aires|argentina",                                       "Buenos Aires",      "Argentina"),
        (r"sucre|la.paz|bolivia",                                         "La Paz",            "Bolivia"),
        (r"quito|ecuador|mitad.del.mundo",                                "Quito",             "Ecuador"),
    ]

    def match_city(text):
        if not text or not isinstance(text, str):
            return None, None
        t = text.lower()
        for pattern, city, country in CITY_PATTERNS:
            if re.search(pattern, t):
                return city, country
        return None, None

    df[["city", "country"]] = df["description"].apply(
        lambda x: pd.Series(match_city(x))
    )
    matched_n = df["city"].notna().sum()
    print(f"    City matched: {matched_n}/{len(df)} images")

    # ── Text quality score (0–10) ───────────────────────────────────
    # Counts positive aesthetic/sentiment keywords present in the description.
    POSITIVE_KEYWORDS = {
        "beautiful", "stunning", "amazing", "breathtaking", "magnificent",
        "spectacular", "gorgeous", "splendid", "scenic", "picturesque",
        "majestic", "dramatic", "vibrant", "wonderful", "grand", "enchanting",
        "colorful", "serene", "tranquil", "pristine", "lovely", "charming",
        "aerial", "panoramic", "skyline", "sunset", "sunrise", "golden",
        "ancient", "historic", "heritage", "unesco", "famous", "iconic",
        "landmark", "monument", "traditional", "cultural", "mystical",
        "waterfall", "mountain", "valley", "beach", "ocean", "lake",
        "castle", "palace", "cathedral", "temple", "archaeological",
    }

    def text_quality(desc):
        if not desc or not isinstance(desc, str):
            return 3.0
        words = set(re.findall(r"[a-z]+", desc.lower()))
        return min(10.0, 3.0 + len(words & POSITIVE_KEYWORDS) * 1.0)

    df["text_quality"] = df["description"].apply(text_quality)

    # ── Visual metrics from image pixels (requires Pillow) ──────────
    def visual_metrics(filepath):
        if not PIL_AVAILABLE or not isinstance(filepath, str):
            return 0.0, 0.0, 0.0, 0.0
        try:
            img = PILImage.open(filepath).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

            # Hasler & Süsstrunk (2003) colorfulness metric
            rg = r - g
            yb = 0.5 * (r + g) - b
            colorfulness = float(
                np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
            )

            brightness = float(arr.mean())          # 0–255

            gray = 0.299 * r + 0.587 * g + 0.114 * b
            contrast = float(gray.std())

            # Sharpness proxy: mean squared gradient magnitude
            dy = np.diff(gray, axis=0).astype(np.float32)
            dx = np.diff(gray, axis=1).astype(np.float32)
            sharpness = float(0.5 * (dy.var() + dx.var()))

            return colorfulness, brightness, contrast, sharpness
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    df[["colorfulness", "brightness", "contrast", "sharpness"]] = df["filepath"].apply(
        lambda p: pd.Series(visual_metrics(p))
    )

    # ── Normalise visual features to 0–10 across this dataset ───────
    def safe_norm_0_10(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(5.0, index=s.index)
        return (s - mn) / (mx - mn) * 10

    df["colorfulness_norm"] = safe_norm_0_10(df["colorfulness"])
    df["contrast_norm"]     = safe_norm_0_10(df["contrast"])
    df["sharpness_norm"]    = safe_norm_0_10(df["sharpness"])
    # Prefer moderate brightness (~128); penalise very dark or very bright images
    df["brightness_score"]  = (
        1 - ((df["brightness"] - 128).abs() / 128)
    ).clip(0, 1) * 10

    # Composite attractiveness per image (0–10)
    df["image_attractiveness"] = (
        df["text_quality"]      * 0.35
        + df["colorfulness_norm"] * 0.25
        + df["contrast_norm"]     * 0.20
        + df["sharpness_norm"]    * 0.10
        + df["brightness_score"]  * 0.10
    ).clip(0, 10).round(3)

    log_shape("Image-level scored", df)
    # Drop filepath before saving (absolute path not useful in CSV)
    df.drop(columns=["filepath"], errors="ignore").to_csv(
        os.path.join(CLEAN, "image_posts_clean.csv"), index=False
    )
    print("  Saved image_posts_clean.csv")

    # ── City-level aggregation ───────────────────────────────────────
    df_matched = df[df["city"].notna()].copy()
    if df_matched.empty:
        print("  [WARN] No images matched to cities — city summary skipped")
    else:
        city_agg = df_matched.groupby(["city", "country"]).agg(
            image_count        = ("filename",             "count"),
            avg_text_quality   = ("text_quality",         "mean"),
            avg_colorfulness   = ("colorfulness",         "mean"),
            avg_brightness     = ("brightness",           "mean"),
            avg_contrast       = ("contrast",             "mean"),
            avg_sharpness      = ("sharpness",            "mean"),
            avg_attractiveness = ("image_attractiveness", "mean"),
        ).reset_index()

        for col in ["avg_text_quality", "avg_colorfulness", "avg_brightness",
                    "avg_contrast", "avg_sharpness", "avg_attractiveness"]:
            city_agg[col] = city_agg[col].round(3)

        # Image Score (0–100): normalise avg_attractiveness across cities
        mn = city_agg["avg_attractiveness"].min()
        mx = city_agg["avg_attractiveness"].max()
        if mx > mn:
            city_agg["image_score"] = (
                (city_agg["avg_attractiveness"] - mn) / (mx - mn) * 100
            ).clip(0, 100).round(1)
        else:
            city_agg["image_score"] = 50.0

        city_agg.to_csv(os.path.join(CLEAN, "image_city_summary_clean.csv"), index=False)
        print(f"  Saved image_city_summary_clean.csv  ({len(city_agg)} cities)")
        print("  Top cities by image_score:")
        for _, row in city_agg.nlargest(5, "image_score").iterrows():
            print(f"    {row['city']:20s}  score={row['image_score']:.1f}  "
                  f"n={int(row['image_count'])}")


# ═══════════════════════════════════════════════════════
# 3. COST OF LIVING – country-level (exact real columns)
# ═══════════════════════════════════════════════════════
print("\n== [3] Cost of Living: clean ==")

col_path = os.path.join(RAW, "cost_of_living_raw.csv")
if not os.path.exists(col_path):
    print("  [SKIP] cost_of_living_raw.csv not found")
else:
    # Find first non-empty row to use as header
    raw = pd.read_csv(col_path, header=None)
    header_row = next(
        i for i, row in raw.iterrows()
        if row.notna().any() and row.astype(str).str.strip().ne("").any()
    )
    df = pd.read_csv(col_path, header=header_row)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").reset_index(drop=True)
    log_shape("Loaded raw CoL", df)

    # Exact real column names
    INDEX_COLS = [
        "Cost of Living Index",
        "Rent Index",
        "Cost of Living Plus Rent Index",
        "Groceries Index",
        "Restaurant Price Index",
        "Local Purchasing Power Index",
    ]

    for col in INDEX_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce").astype("Int64")

    # Normalise country name: strip whitespace, title case
    df["Country"] = df["Country"].str.strip().str.title()

    # Dedup: keep lowest Rank per country (most reliable/recent)
    before = len(df)
    df = df.sort_values("Rank").drop_duplicates(subset=["Country"])
    print(f"    Deduped: removed {before - len(df)} duplicate countries")

    # Domain: indices must be > 0
    for col in INDEX_COLS:
        df.loc[df[col] <= 0, col] = np.nan

    df["has_complete_indices"] = df[INDEX_COLS].notna().all(axis=1)
    n_incomplete = (~df["has_complete_indices"]).sum()
    if n_incomplete:
        print(f"    [WARN] {n_incomplete} countries missing at least one index")

    # Affordability tier based on Cost of Living Index (NYC ≈ 100)
    df["affordability_tier"] = pd.cut(
        df["Cost of Living Index"],
        bins=[0, 35, 60, 85, float("inf")],
        labels=["Budget", "Mid-range", "Expensive", "Premium"],
        right=False,
    )

    # Value score: how much purchasing power per unit cost
    df["value_score"] = (
        df["Local Purchasing Power Index"] /
        df["Cost of Living Index"].replace(0, np.nan)
    ).round(3)

    log_shape("Cleaned CoL", df)
    df.to_csv(os.path.join(CLEAN, "cost_of_living_clean.csv"), index=False)
    print(f"  Saved cost_of_living_clean.csv  ({len(df)} countries)")


# ═══════════════════════════════════════════════════════
# 4. WORLDWIDE TRAVEL CITIES – expand JSON, validate scores
# ═══════════════════════════════════════════════════════
print("\n== [4] Worldwide Travel Cities: clean + expand JSON ==")

wtc_path = os.path.join(RAW, "worldwide_travel_cities_raw.csv")
if not os.path.exists(wtc_path):
    print("  [SKIP] worldwide_travel_cities_raw.csv not found")
else:
    df = pd.read_csv(wtc_path)
    log_shape("Loaded raw WTC", df)

    # ── Activity score columns: integer 1–5 ────────────────────────
    SCORE_COLS = ["culture", "adventure", "nature", "beaches",
                  "nightlife", "cuisine", "wellness", "urban", "seclusion"]

    for col in SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(1, 5)
        df[col] = df[col].fillna(df[col].median())

    # ── Coordinates ─────────────────────────────────────────────────
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df.loc[df["latitude"].abs()  > 90,  "latitude"]  = np.nan
    df.loc[df["longitude"].abs() > 180, "longitude"] = np.nan

    # ── budget_level ─────────────────────────────────────────────────
    valid_budgets = {"Budget", "Mid-range", "Luxury"}
    df.loc[~df["budget_level"].isin(valid_budgets), "budget_level"] = np.nan

    # ── Expand avg_temp_monthly JSON → long table ──────────────────
    # Real schema: {"1":{"avg":X,"max":X,"min":X}, ..., "12":{...}}
    monthly_rows = []
    for _, row in df.iterrows():
        try:
            temp_data = json.loads(row["avg_temp_monthly"]) if pd.notna(row["avg_temp_monthly"]) else {}
        except Exception:
            temp_data = {}

        for month_str, vals in temp_data.items():
            try:
                monthly_rows.append({
                    "id":           row["id"],
                    "city":         row["city"],
                    "country":      row["country"],
                    "region":       row["region"],
                    "month":        int(month_str),
                    "wtc_avg_temp": float(vals.get("avg", np.nan)),
                    "wtc_max_temp": float(vals.get("max", np.nan)),
                    "wtc_min_temp": float(vals.get("min", np.nan)),
                })
            except Exception:
                continue

    df_monthly = pd.DataFrame(monthly_rows)

    # Validate temperature ranges
    for col in ["wtc_avg_temp", "wtc_max_temp", "wtc_min_temp"]:
        df_monthly.loc[df_monthly[col].abs() > 60, col] = np.nan

    # Enforce ordering: min <= avg <= max
    bad = (
        (df_monthly["wtc_min_temp"] > df_monthly["wtc_avg_temp"]) |
        (df_monthly["wtc_avg_temp"] > df_monthly["wtc_max_temp"])
    )
    df_monthly.loc[bad, ["wtc_min_temp", "wtc_max_temp"]] = np.nan
    print(f"    Monthly temp rows: {len(df_monthly):,}  |  "
          f"ordering violations: {bad.sum()}")

    # ── Expand ideal_durations JSON array → boolean flags ──────────
    # Real values: "Weekend", "Short trip", "One week", "Long trip"
    ALL_DURATIONS = ["Weekend", "Short trip", "One week", "Long trip"]
    for dur in ALL_DURATIONS:
        col_name = "ideal_" + dur.lower().replace(" ", "_")
        def make_flag(x, d=dur):
            if pd.isna(x):
                return False
            try:
                return d in json.loads(x)
            except Exception:
                return False
        df[col_name] = df["ideal_durations"].apply(make_flag)

    # ── Composite vibe score: mean of all 9 activity dimensions ────
    df["composite_score"] = df[SCORE_COLS].mean(axis=1).round(2)

    # ── Dedup by id ─────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["id"])
    print(f"    Deduped: removed {before - len(df)} duplicate city IDs")

    # Drop raw JSON blobs (preserved in separate monthly file)
    df_city_out = df.drop(columns=["avg_temp_monthly", "ideal_durations"], errors="ignore")
    log_shape("Cleaned WTC city-level", df_city_out)
    df_city_out.to_csv(os.path.join(CLEAN, "worldwide_travel_cities_clean.csv"), index=False)
    print("  Saved worldwide_travel_cities_clean.csv")

    df_monthly.to_csv(os.path.join(CLEAN, "wtc_monthly_temps_clean.csv"), index=False)
    print(f"  Saved wtc_monthly_temps_clean.csv  ({len(df_monthly):,} rows)")

print("\n[Done] All cleaning complete. Files in data/cleaned/")
